/**
 * Agent Observability System — TypeScript port.
 *
 * Implements the three pillars of agent observability:
 *   - Tracing:  captures the full decision tree of a single request
 *   - Logging:  structured JSON events at each step (via winston/pino)
 *   - Metrics:  rolling aggregates for system health dashboards
 *
 * Additional components:
 *   - TokenAccountant  – per-user/session/model cost tracking
 *   - DecisionTracer   – captures reasoning to explain "why did it do that?"
 *
 * Requirements: no external runtime dependencies beyond Node built-ins.
 * Install winston for production logging: npm install winston
 *
 * See: docs/05-the-tool-ecosystem/03-agent-observability.md
 */

"use strict";

import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";

// ---------------------------------------------------------------------------
// Pricing defaults (USD per 1 000 tokens, input / output)
// ---------------------------------------------------------------------------

export const DEFAULT_PRICING: Record<string, { input: number; output: number }> = {
  "gpt-4o":            { input: 0.0025,   output: 0.01 },
  "gpt-4o-mini":       { input: 0.00015,  output: 0.0006 },
  "claude-3-5-sonnet": { input: 0.003,    output: 0.015 },
  "claude-3-haiku":    { input: 0.00025,  output: 0.00125 },
  "gemini-1.5-pro":    { input: 0.00125,  output: 0.005 },
  "gemini-1.5-flash":  { input: 0.000075, output: 0.0003 },
  "unknown":           { input: 0.001,    output: 0.003 },
};

function computeCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
  pricing: typeof DEFAULT_PRICING
): number {
  const rates = pricing[model] ?? pricing["unknown"]!;
  return (inputTokens * rates.input + outputTokens * rates.output) / 1000;
}

function nowMs(): number {
  return Date.now();
}

function nowSec(): number {
  return Date.now() / 1000;
}

function uuid(): string {
  return crypto.randomUUID();
}

// ===========================================================================
// Core data model: Span and Trace
// ===========================================================================

export interface SpanData {
  spanId: string;
  parentSpanId: string | null;
  type: string;   // "llm_call" | "tool_call" | "planning" | "execution"
  name: string;
  startTime: number;     // ms epoch
  endTime: number | null;
  inputData: Record<string, unknown> | null;
  outputData: Record<string, unknown> | null;
  inputTokens: number;
  outputTokens: number;
  tokensUsed: number;
  cost: number;
  model: string | null;
  status: "running" | "success" | "error";
  errorMessage: string | null;
  metadata: Record<string, unknown>;
}

export class Span implements SpanData {
  spanId   = uuid();
  parentSpanId: string | null = null;
  type     = "";
  name     = "";
  startTime = nowMs();
  endTime: number | null = null;
  inputData: Record<string, unknown> | null = null;
  outputData: Record<string, unknown> | null = null;
  inputTokens  = 0;
  outputTokens = 0;
  tokensUsed   = 0;
  cost         = 0;
  model: string | null = null;
  status: "running" | "success" | "error" = "running";
  errorMessage: string | null = null;
  metadata: Record<string, unknown> = {};

  finish(
    outputData?: Record<string, unknown>,
    status: "success" | "error" = "success",
    errorMessage?: string
  ): this {
    this.endTime      = nowMs();
    this.outputData   = outputData ?? null;
    this.status       = status;
    this.errorMessage = errorMessage ?? null;
    return this;
  }

  get durationMs(): number {
    return (this.endTime ?? nowMs()) - this.startTime;
  }

  toJSON(): SpanData & { durationMs: number } {
    return {
      spanId:        this.spanId,
      parentSpanId:  this.parentSpanId,
      type:          this.type,
      name:          this.name,
      startTime:     this.startTime,
      endTime:       this.endTime,
      durationMs:    Math.round(this.durationMs),
      inputData:     this.inputData,
      outputData:    this.outputData,
      inputTokens:   this.inputTokens,
      outputTokens:  this.outputTokens,
      tokensUsed:    this.tokensUsed,
      cost:          +this.cost.toFixed(6),
      model:         this.model,
      status:        this.status,
      errorMessage:  this.errorMessage,
      metadata:      this.metadata,
    };
  }
}

export interface TraceData {
  traceId:     string;
  userQuery:   string;
  userId:      string | null;
  sessionId:   string | null;
  status:      "success" | "error";
  durationMs:  number;
  llmCalls:    number;
  toolCalls:   number;
  totalTokens: number;
  totalCost:   number;
  spans:       ReturnType<Span["toJSON"]>[];
  metadata:    Record<string, unknown>;
}

export class Trace {
  traceId   = uuid();
  userQuery = "";
  userId:   string | null = null;
  sessionId: string | null = null;
  spans: Span[] = [];
  startTime = nowMs();
  endTime:  number | null = null;
  metadata: Record<string, unknown> = {};

  addSpan(span: Span): Span {
    this.spans.push(span);
    return span;
  }

  newSpan(type: string, name: string, parent?: Span): Span {
    const span = new Span();
    span.type = type;
    span.name = name;
    span.parentSpanId = parent?.spanId ?? null;
    this.spans.push(span);
    return span;
  }

  finish(): this {
    this.endTime = nowMs();
    return this;
  }

  get durationMs(): number {
    return (this.endTime ?? nowMs()) - this.startTime;
  }

  get totalTokens(): number {
    return this.spans.reduce((s, sp) => s + sp.tokensUsed, 0);
  }

  get totalCost(): number {
    return this.spans.reduce((s, sp) => s + sp.cost, 0);
  }

  get llmCallCount(): number {
    return this.spans.filter(s => s.type === "llm_call").length;
  }

  get toolCallCount(): number {
    return this.spans.filter(s => s.type === "tool_call").length;
  }

  get hasError(): boolean {
    return this.spans.some(s => s.status === "error");
  }

  get status(): "success" | "error" {
    return this.hasError ? "error" : "success";
  }

  toJSON(): TraceData {
    return {
      traceId:     this.traceId,
      userQuery:   this.userQuery,
      userId:      this.userId,
      sessionId:   this.sessionId,
      status:      this.status,
      durationMs:  Math.round(this.durationMs),
      llmCalls:    this.llmCallCount,
      toolCalls:   this.toolCallCount,
      totalTokens: this.totalTokens,
      totalCost:   +this.totalCost.toFixed(6),
      spans:       this.spans.map(s => s.toJSON()),
      metadata:    this.metadata,
    };
  }
}

// ===========================================================================
// TraceExporter protocol + ConsoleExporter
// ===========================================================================

export interface TraceExporter {
  export(trace: Trace): void;
}

export class ConsoleExporter implements TraceExporter {
  export(trace: Trace): void {
    const icon   = trace.hasError ? "✗" : "✓";
    const border = "═".repeat(56);
    console.log(`\n${border}`);
    console.log(
      `${icon} Trace ${trace.traceId.slice(0, 8)}  ` +
      `query='${trace.userQuery.slice(0, 55)}'`
    );
    console.log(
      `  duration=${trace.durationMs}ms  ` +
      `tokens=${trace.totalTokens}  ` +
      `cost=$${trace.totalCost.toFixed(4)}`
    );
    console.log(
      `  llm_calls=${trace.llmCallCount}  ` +
      `tool_calls=${trace.toolCallCount}  ` +
      `status=${trace.status}`
    );
    console.log();
    for (const span of trace.spans) {
      const indent = span.parentSpanId ? "    " : "  ";
      const sIcon  = span.status === "error" ? "✗" : "·";
      const err    = span.errorMessage ? `  ERROR: ${span.errorMessage}` : "";
      console.log(
        `${indent}${sIcon} [${span.type.padEnd(10)}] ${span.name.padEnd(22)} ` +
        `${span.durationMs.toString().padStart(6)}ms  ` +
        `tokens=${span.tokensUsed}${err}`
      );
    }
    console.log(`${border}\n`);
  }
}

export class JSONFileExporter implements TraceExporter {
  private outputDir: string;

  constructor(outputDir = "./traces/") {
    this.outputDir = outputDir;
    fs.mkdirSync(outputDir, { recursive: true });
  }

  export(trace: Trace): void {
    const filepath = path.join(this.outputDir, `${trace.traceId}.json`);
    fs.writeFileSync(filepath, JSON.stringify(trace.toJSON(), null, 2), "utf8");
  }
}

// ===========================================================================
// TraceCollector
// ===========================================================================

export class TraceCollector {
  private exporter: TraceExporter;
  private traces = new Map<string, Trace>();

  constructor(exporter?: TraceExporter) {
    this.exporter = exporter ?? new ConsoleExporter();
  }

  startTrace(
    userQuery: string,
    userId?: string,
    sessionId?: string,
    metadata?: Record<string, unknown>
  ): Trace {
    const trace = new Trace();
    trace.userQuery  = userQuery;
    trace.userId     = userId ?? null;
    trace.sessionId  = sessionId ?? null;
    trace.metadata   = metadata ?? {};
    this.traces.set(trace.traceId, trace);
    return trace;
  }

  endTrace(trace: Trace): void {
    if (!trace.endTime) trace.finish();
    this.exporter.export(trace);
  }

  getTrace(traceId: string): Trace | undefined {
    return this.traces.get(traceId);
  }

  queryTraces(opts: {
    userId?:    string;
    status?:    "success" | "error";
    since?:     Date;
    limit?:     number;
  } = {}): Trace[] {
    const sinceMs = opts.since?.getTime() ?? 0;
    const results = Array.from(this.traces.values()).filter(t =>
      (!opts.userId || t.userId === opts.userId) &&
      (!opts.status || t.status === opts.status) &&
      t.startTime >= sinceMs
    );
    results.sort((a, b) => b.startTime - a.startTime);
    return results.slice(0, opts.limit ?? 100);
  }
}

// ===========================================================================
// AgentMetrics
// ===========================================================================

export interface MetricsSummary {
  requests:      number;
  avgLatencyMs:  number;
  p95LatencyMs:  number;
  p99LatencyMs:  number;
  avgTokens:     number;
  avgCost:       number;
  totalCost:     number;
  errorRatePct:  number;
  avgLlmCalls:   number;
  avgToolCalls:  number;
}

export interface Alert {
  kind:      string;
  message:   string;
  value:     number;
  threshold: number;
  severity:  "warning" | "critical";
}

function percentile(values: number[], pct: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx    = Math.floor(sorted.length * pct / 100);
  return sorted[Math.min(idx, sorted.length - 1)]!;
}

function average(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

/** Fixed-capacity ring buffer. */
class RingBuffer<T> {
  private _buf: T[] = [];
  constructor(private readonly maxLen: number) {}
  push(v: T): void {
    this._buf.push(v);
    if (this._buf.length > this.maxLen) this._buf.shift();
  }
  toArray(): T[] { return [...this._buf]; }
  get length(): number { return this._buf.length; }
}

export class AgentMetrics {
  private windowSize: number;
  private timestamps   = new RingBuffer<number>(1000);
  private llmLatency   = new RingBuffer<number>(1000);
  private toolLatency  = new RingBuffer<number>(1000);
  private totalLatency = new RingBuffer<number>(1000);
  private tokens       = new RingBuffer<number>(1000);
  private costs        = new RingBuffer<number>(1000);
  private toolCalls    = new RingBuffer<number>(1000);
  private llmCalls     = new RingBuffer<number>(1000);
  private errors       = new RingBuffer<number>(1000);

  constructor(windowSize = 1000) {
    this.windowSize = windowSize;
  }

  record(trace: Trace): void {
    const now = nowSec();
    this.timestamps.push(now);
    this.totalLatency.push(trace.durationMs);
    this.tokens.push(trace.totalTokens);
    this.costs.push(trace.totalCost);
    this.toolCalls.push(trace.toolCallCount);
    this.llmCalls.push(trace.llmCallCount);
    this.errors.push(trace.hasError ? 1 : 0);

    for (const span of trace.spans) {
      if (span.type === "llm_call"  && span.endTime) this.llmLatency.push(span.durationMs);
      if (span.type === "tool_call" && span.endTime) this.toolLatency.push(span.durationMs);
    }
  }

  getSummary(_windowMinutes = 60): MetricsSummary {
    const lat   = this.totalLatency.toArray();
    const tok   = this.tokens.toArray();
    const cost  = this.costs.toArray();
    const err   = this.errors.toArray();
    const llm   = this.llmCalls.toArray();
    const tool  = this.toolCalls.toArray();

    return {
      requests:     lat.length,
      avgLatencyMs: +average(lat).toFixed(1),
      p95LatencyMs: +percentile(lat, 95).toFixed(1),
      p99LatencyMs: +percentile(lat, 99).toFixed(1),
      avgTokens:    +average(tok).toFixed(1),
      avgCost:      +average(cost).toFixed(4),
      totalCost:    +cost.reduce((s, v) => s + v, 0).toFixed(4),
      errorRatePct: +(average(err) * 100).toFixed(2),
      avgLlmCalls:  +average(llm).toFixed(2),
      avgToolCalls: +average(tool).toFixed(2),
    };
  }

  detectAnomalies(): Alert[] {
    const summary = this.getSummary();
    const alerts: Alert[] = [];

    if (summary.errorRatePct > 5) {
      alerts.push({
        kind:      "error_rate",
        message:   `High error rate: ${summary.errorRatePct.toFixed(1)}%`,
        value:     summary.errorRatePct,
        threshold: 5,
        severity:  summary.errorRatePct > 20 ? "critical" : "warning",
      });
    }

    if (summary.requests >= 10 && summary.p95LatencyMs > 2 * summary.avgLatencyMs) {
      alerts.push({
        kind:      "latency",
        message:   `P95 ${summary.p95LatencyMs}ms > 2× avg ${summary.avgLatencyMs}ms`,
        value:     summary.p95LatencyMs,
        threshold: 2 * summary.avgLatencyMs,
        severity:  "warning",
      });
    }

    if (summary.avgCost > 0.50) {
      alerts.push({
        kind:      "cost",
        message:   `High cost per request: $${summary.avgCost.toFixed(2)}`,
        value:     summary.avgCost,
        threshold: 0.50,
        severity:  summary.avgCost > 2.0 ? "critical" : "warning",
      });
    }

    return alerts;
  }

  exportPrometheus(): string {
    const s = this.getSummary();
    return [
      `agent_requests_total ${s.requests}`,
      `agent_latency_avg_ms ${s.avgLatencyMs}`,
      `agent_latency_p95_ms ${s.p95LatencyMs}`,
      `agent_error_rate_pct ${s.errorRatePct}`,
      `agent_avg_tokens_per_request ${s.avgTokens}`,
      `agent_avg_cost_per_request ${s.avgCost}`,
      `agent_total_cost_usd ${s.totalCost}`,
    ].join("\n");
  }
}

// ===========================================================================
// AgentLogger — structured JSON logging
// ===========================================================================

const REDACTED_KEYS = new Set([
  "api_key", "apikey", "secret", "password", "token",
  "authorization", "credential", "private_key",
]);

function redact(obj: unknown): unknown {
  if (obj === null || typeof obj !== "object") return obj;
  if (Array.isArray(obj)) return obj.map(redact);
  return Object.fromEntries(
    Object.entries(obj as Record<string, unknown>).map(([k, v]) => [
      k,
      REDACTED_KEYS.has(k.toLowerCase()) ? "[REDACTED]" : redact(v),
    ])
  );
}

type LogLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR";

const LOG_LEVEL_RANK: Record<LogLevel, number> = {
  DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3,
};

export class AgentLogger {
  private minLevel: number;

  constructor(logLevel: LogLevel = "INFO") {
    this.minLevel = LOG_LEVEL_RANK[logLevel] ?? 1;
  }

  private emit(level: LogLevel, payload: Record<string, unknown>): void {
    if (LOG_LEVEL_RANK[level] < this.minLevel) return;
    const line = JSON.stringify({
      level,
      timestamp: new Date().toISOString(),
      ...redact(payload) as Record<string, unknown>,
    });
    if (level === "ERROR") {
      process.stderr.write(line + "\n");
    } else {
      process.stdout.write(line + "\n");
    }
  }

  logLlmCall(traceId: string, spanId: string, model: string,
             messagesCount: number, estimatedTokens: number): void {
    this.emit("INFO", {
      event: "llm_call_start", traceId, spanId,
      model, messagesCount, estimatedTokens,
    });
  }

  logLlmResponse(traceId: string, spanId: string, model: string,
                 inputTokens: number, outputTokens: number,
                 latencyMs: number, hasToolCalls: boolean): void {
    this.emit("INFO", {
      event: "llm_call_complete", traceId, spanId, model,
      inputTokens, outputTokens,
      totalTokens: inputTokens + outputTokens,
      latencyMs: Math.round(latencyMs),
      hasToolCalls,
    });
  }

  logToolExecution(traceId: string, spanId: string,
                   toolName: string, paramsSummary: string): void {
    this.emit("INFO", {
      event: "tool_call_start", traceId, spanId, toolName, paramsSummary,
    });
  }

  logToolResult(traceId: string, spanId: string, toolName: string,
                success: boolean, resultSummary: string, latencyMs: number): void {
    this.emit(success ? "INFO" : "WARNING", {
      event: "tool_call_complete", traceId, spanId, toolName,
      success, resultSummary: resultSummary.slice(0, 200),
      latencyMs: Math.round(latencyMs),
    });
  }

  logContextManagement(traceId: string, action: string,
                       originalTokens: number, resultTokens: number): void {
    this.emit("WARNING", {
      event: "context_management", traceId, action,
      originalTokens, resultTokens,
      tokensRemoved: originalTokens - resultTokens,
    });
  }

  logError(traceId: string, error: Error, context: Record<string, unknown>): void {
    this.emit("ERROR", {
      event: "agent_error", traceId,
      errorType:    error.constructor.name,
      errorMessage: error.message,
      context,
    });
  }
}

// ===========================================================================
// TokenAccountant
// ===========================================================================

interface TokenRecord {
  traceId:      string;
  userId:       string;
  sessionId:    string;
  model:        string;
  inputTokens:  number;
  outputTokens: number;
  cost:         number;
  timestamp:    number;  // seconds epoch
}

export class TokenAccountant {
  private pricing: typeof DEFAULT_PRICING;
  private records: TokenRecord[] = [];
  private budgetAlerts = new Map<string, number>();

  constructor(pricing?: typeof DEFAULT_PRICING) {
    this.pricing = pricing ?? DEFAULT_PRICING;
  }

  record(trace: Trace, userId: string, sessionId: string): void {
    for (const span of trace.spans) {
      if (span.type !== "llm_call") continue;
      const r: TokenRecord = {
        traceId:      trace.traceId,
        userId,
        sessionId,
        model:        span.model ?? "unknown",
        inputTokens:  span.inputTokens,
        outputTokens: span.outputTokens,
        cost:         span.cost,
        timestamp:    nowSec(),
      };
      this.records.push(r);

      const budget = this.budgetAlerts.get(userId);
      if (budget !== undefined) {
        const total = this.getUserCost(userId);
        if (total > budget) {
          console.warn(
            `[BUDGET ALERT] User '${userId}' exceeded $${budget.toFixed(2)}: ` +
            `current $${total.toFixed(4)}`
          );
        }
      }
    }
  }

  getUserCost(userId: string, days = 30): number {
    const cutoff = nowSec() - days * 86400;
    return this.records
      .filter(r => r.userId === userId && r.timestamp >= cutoff)
      .reduce((s, r) => s + r.cost, 0);
  }

  getSessionCost(sessionId: string): number {
    return this.records
      .filter(r => r.sessionId === sessionId)
      .reduce((s, r) => s + r.cost, 0);
  }

  getModelUsage(model: string, days = 30): Record<string, unknown> {
    const cutoff = nowSec() - days * 86400;
    const recs   = this.records.filter(
      r => r.model === model && r.timestamp >= cutoff
    );
    return {
      model,
      calls:        recs.length,
      inputTokens:  recs.reduce((s, r) => s + r.inputTokens, 0),
      outputTokens: recs.reduce((s, r) => s + r.outputTokens, 0),
      totalCost:    +recs.reduce((s, r) => s + r.cost, 0).toFixed(6),
    };
  }

  getDailyCostReport(): Record<string, unknown> {
    const cutoff = nowSec() - 86400;
    const recent = this.records.filter(r => r.timestamp >= cutoff);

    const byModel: Record<string, { calls: number; inputTokens: number;
                                    outputTokens: number; cost: number }> = {};
    for (const r of recent) {
      if (!byModel[r.model]) {
        byModel[r.model] = { calls: 0, inputTokens: 0, outputTokens: 0, cost: 0 };
      }
      byModel[r.model]!.calls++;
      byModel[r.model]!.inputTokens  += r.inputTokens;
      byModel[r.model]!.outputTokens += r.outputTokens;
      byModel[r.model]!.cost         += r.cost;
    }

    const uniqueUsers = new Set(recent.map(r => r.userId)).size;
    return {
      date:               new Date().toISOString().slice(0, 10),
      totalRequests:      recent.length,
      totalInputTokens:   recent.reduce((s, r) => s + r.inputTokens, 0),
      totalOutputTokens:  recent.reduce((s, r) => s + r.outputTokens, 0),
      totalCost:          +recent.reduce((s, r) => s + r.cost, 0).toFixed(6),
      uniqueUsers,
      byModel,
    };
  }

  setBudgetAlert(userId: string, maxCost: number): void {
    this.budgetAlerts.set(userId, maxCost);
  }
}

// ===========================================================================
// DecisionTracer
// ===========================================================================

export interface Decision {
  decisionId: string;
  timestamp:  number;
  step:       string;
  context:    Record<string, unknown>;
  options:    string[];
  chosen:     string;
  reasoning:  string;
}

export class DecisionTracer {
  decisions: Decision[] = [];

  capture(
    step: string,
    context: Record<string, unknown>,
    options: string[],
    chosen: string,
    reasoning: string
  ): Decision {
    const d: Decision = {
      decisionId: uuid(),
      timestamp:  nowSec(),
      step,
      context,
      options,
      chosen,
      reasoning,
    };
    this.decisions.push(d);
    return d;
  }

  replay(traceId?: string): string {
    if (!this.decisions.length) return "(no decisions recorded)";
    const header = `# Agent Decision Trail${traceId ? ` — trace ${traceId}` : ""}`;
    const body = this.decisions.map((d, i) => [
      `## Step ${i + 1}: ${d.step}`,
      `  Context:   ${JSON.stringify(d.context)}`,
      `  Options:   ${d.options.join(", ") || "(none)"}`,
      `  Chosen:    ${d.chosen}`,
      `  Reasoning: ${d.reasoning}`,
    ].join("\n")).join("\n\n");
    return `${header}\n\n${body}`;
  }

  findDivergence(expectedChoice: string): string {
    for (let i = 0; i < this.decisions.length; i++) {
      const d = this.decisions[i]!;
      if (d.options.includes(expectedChoice) && d.chosen !== expectedChoice) {
        return (
          `Divergence at step ${i + 1} (${d.step}):\n` +
          `  Expected:  '${expectedChoice}'\n` +
          `  Chosen:    '${d.chosen}'\n` +
          `  Reasoning: ${d.reasoning}\n` +
          `  Context:   ${JSON.stringify(d.context)}`
        );
      }
    }
    return (
      `No divergence found — '${expectedChoice}' was either chosen or not in options.`
    );
  }
}

// ===========================================================================
// Simulated LLM + ObservableAgent (demo)
// ===========================================================================

class SimulatedLLM {
  static readonly MODEL        = "gpt-4o-mini";
  static readonly INPUT_TOKENS = 512;
  static readonly OUTPUT_TOKENS = 128;

  constructor(private readonly failOn?: string) {}

  async chat(query: string): Promise<{
    model: string; inputTokens: number; outputTokens: number; content: string;
  }> {
    if (this.failOn && query.includes(this.failOn)) {
      throw new Error(`Simulated LLM failure for query: ${query}`);
    }
    await new Promise(r => setTimeout(r, 50));
    return {
      model:        SimulatedLLM.MODEL,
      inputTokens:  SimulatedLLM.INPUT_TOKENS,
      outputTokens: SimulatedLLM.OUTPUT_TOKENS,
      content:      `Simulated answer for: ${query.slice(0, 40)}`,
    };
  }
}

export class ObservableAgent {
  private llm: SimulatedLLM;

  constructor(
    private readonly collector:  TraceCollector,
    private readonly metrics:    AgentMetrics,
    private readonly logger:     AgentLogger,
    private readonly accountant: TokenAccountant,
    private readonly dt:         DecisionTracer,
    private readonly pricing     = DEFAULT_PRICING,
    failOn?: string,
  ) {
    this.llm = new SimulatedLLM(failOn);
  }

  private async llmSpan(
    trace: Trace, parent: Span, callName: string, query: string
  ): Promise<Span> {
    const span = trace.newSpan("llm_call", callName, parent);
    this.logger.logLlmCall(
      trace.traceId, span.spanId,
      SimulatedLLM.MODEL, 1, SimulatedLLM.INPUT_TOKENS
    );
    const resp = await this.llm.chat(query);
    span.model        = resp.model;
    span.inputTokens  = resp.inputTokens;
    span.outputTokens = resp.outputTokens;
    span.tokensUsed   = resp.inputTokens + resp.outputTokens;
    span.cost = computeCost(resp.model, resp.inputTokens, resp.outputTokens, this.pricing);
    span.finish({ content: resp.content });
    this.logger.logLlmResponse(
      trace.traceId, span.spanId, resp.model,
      resp.inputTokens, resp.outputTokens, span.durationMs, false
    );
    return span;
  }

  async run(
    query: string,
    userId    = "demo_user",
    sessionId = "demo_session"
  ): Promise<{ answer: string; traceId: string }> {
    const trace = this.collector.startTrace(query, userId, sessionId);
    try {
      // Planning
      const planSpan = trace.newSpan("planning", "generate_plan");
      this.dt.capture(
        "plan", { query },
        ["direct_answer", "tool_use"],
        "tool_use",
        "Query requires external data lookup"
      );
      planSpan.finish({ steps: 2 });

      // Execution
      const execSpan = trace.newSpan("execution", "execute_plan", planSpan);

      // Tool call
      const toolSpan = trace.newSpan("tool_call", "get_data", execSpan);
      this.logger.logToolExecution(
        trace.traceId, toolSpan.spanId, "get_data", `query=${query.slice(0, 30)}`
      );
      await new Promise(r => setTimeout(r, 20));
      toolSpan.finish({ data: "mock_result" });
      this.logger.logToolResult(
        trace.traceId, toolSpan.spanId, "get_data",
        true, '{"data":"mock_result"}', toolSpan.durationMs
      );

      // Synthesis LLM call
      const synthSpan = await this.llmSpan(trace, execSpan, "synthesise", query);
      execSpan.finish({ toolCalls: 1, llmCalls: 1 });

      // Answer generation
      await this.llmSpan(trace, execSpan, "generate_answer", query);

      const answer = `Answer to '${query.slice(0, 40)}': ${synthSpan.outputData?.["content"]}`;
      trace.finish();
      this.collector.endTrace(trace);
      this.metrics.record(trace);
      this.accountant.record(trace, userId, sessionId);
      return { answer, traceId: trace.traceId };

    } catch (err) {
      if (trace.spans.length > 0) {
        trace.spans[trace.spans.length - 1]!.finish(
          undefined, "error", String(err)
        );
      }
      this.logger.logError(
        trace.traceId, err as Error, { query, userId }
      );
      trace.finish();
      this.collector.endTrace(trace);
      this.metrics.record(trace);
      throw err;
    }
  }
}

// ===========================================================================
// Demo
// ===========================================================================

async function runDemo(): Promise<void> {
  console.log("=".repeat(60));
  console.log("AGENT OBSERVABILITY DEMO (TypeScript)");
  console.log("=".repeat(60));

  const collector  = new TraceCollector(new ConsoleExporter());
  const metrics    = new AgentMetrics();
  const logger     = new AgentLogger("WARNING");
  const accountant = new TokenAccountant();
  const dt         = new DecisionTracer();

  accountant.setBudgetAlert("alice", 0.01);

  const agent = new ObservableAgent(
    collector, metrics, logger, accountant, dt,
    DEFAULT_PRICING, "CRASH"
  );

  const queries: Array<[string, string, string]> = [
    ["Compare AAPL vs MSFT stock performance", "alice", "s1"],
    ["Summarise the latest earnings report",    "bob",   "s2"],
    ["CRASH: trigger a deliberate failure",     "alice", "s1"],
  ];

  let failingTraceId: string | undefined;

  for (const [query, uid, sid] of queries) {
    console.log(`\n>>> Running: ${JSON.stringify(query)}`);
    try {
      const result = await agent.run(query, uid, sid);
      console.log(`    ✓ ${result.answer.slice(0, 80)}`);
    } catch (err) {
      const failed = collector.queryTraces({ userId: uid, status: "error" });
      if (failed.length) failingTraceId = failed[0]!.traceId;
      console.log(`    ✗ Error: ${err}`);
    }
  }

  if (failingTraceId) {
    console.log("\n" + "─".repeat(60));
    console.log("FULL TRACE FOR FAILING QUERY");
    console.log("─".repeat(60));
    const t = collector.getTrace(failingTraceId);
    if (t) console.log(JSON.stringify(t.toJSON(), null, 2));
  }

  console.log("\n" + "─".repeat(60));
  console.log("DECISION TRAIL");
  console.log("─".repeat(60));
  console.log(dt.replay());

  console.log("\n" + "─".repeat(60));
  console.log("METRICS SUMMARY");
  console.log("─".repeat(60));
  console.log(JSON.stringify(metrics.getSummary(), null, 2));

  console.log("\n" + "─".repeat(60));
  console.log("DAILY COST REPORT");
  console.log("─".repeat(60));
  console.log(JSON.stringify(accountant.getDailyCostReport(), null, 2));

  console.log("\n" + "─".repeat(60));
  console.log("PROMETHEUS METRICS");
  console.log("─".repeat(60));
  console.log(metrics.exportPrometheus());
}

// Run demo when executed directly
// (TypeScript compiled to CommonJS: use require.main === module)
if (process.argv[1] && process.argv[1].endsWith("agent_observability.js")) {
  runDemo().catch(console.error);
}
