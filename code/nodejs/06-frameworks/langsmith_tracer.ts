/**
 * LangSmith-style tracing for any agent — not just LangChain.
 *
 * Key insight: observability tools don't require the framework.
 * This module shows how to add structured tracing to any agent.
 * See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
 */

export interface TraceSpan {
  id: string;
  name: string;
  type: "llm" | "tool" | "chain" | "agent";
  startTime: number;
  endTime?: number;
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  error?: string;
  metadata: Record<string, unknown>;
  children: TraceSpan[];
}

export interface Trace {
  traceId: string;
  sessionId: string;
  rootSpan: TraceSpan;
  startTime: number;
  endTime?: number;
  totalDurationMs?: number;
  totalTokens?: number;
  metadata: Record<string, unknown>;
}

let spanCounter = 0;

/**
 * LangSmith-style tracer that works with any agent code.
 */
export class LangSmithTracer {
  private traces: Trace[] = [];
  private activeTrace?: Trace;
  private spanStack: TraceSpan[] = [];

  constructor(private sessionId = "default-session") {}

  /** Start a new trace (equivalent to LangSmith run). */
  startTrace(name: string, metadata: Record<string, unknown> = {}): string {
    const traceId = `trace-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const rootSpan: TraceSpan = {
      id: `span-${++spanCounter}`,
      name,
      type: "agent",
      startTime: Date.now(),
      inputs: {},
      metadata,
      children: [],
    };
    this.activeTrace = {
      traceId,
      sessionId: this.sessionId,
      rootSpan,
      startTime: Date.now(),
      metadata,
    };
    this.spanStack = [rootSpan];
    return traceId;
  }

  /** Start a child span within the current trace. */
  startSpan(
    name: string,
    type: TraceSpan["type"],
    inputs: Record<string, unknown> = {}
  ): string {
    const span: TraceSpan = {
      id: `span-${++spanCounter}`,
      name,
      type,
      startTime: Date.now(),
      inputs,
      metadata: {},
      children: [],
    };
    const parent = this.spanStack[this.spanStack.length - 1];
    if (parent) parent.children.push(span);
    this.spanStack.push(span);
    return span.id;
  }

  /** End the current span. */
  endSpan(outputs: Record<string, unknown> = {}, error?: string): void {
    const span = this.spanStack.pop();
    if (span) {
      span.endTime = Date.now();
      span.outputs = outputs;
      span.error = error;
    }
  }

  /** End the active trace. */
  endTrace(totalTokens?: number): Trace | undefined {
    if (!this.activeTrace) return undefined;
    this.activeTrace.endTime = Date.now();
    this.activeTrace.totalDurationMs = this.activeTrace.endTime - this.activeTrace.startTime;
    this.activeTrace.totalTokens = totalTokens;
    if (this.spanStack.length > 0) {
      const root = this.spanStack[0];
      root.endTime = this.activeTrace.endTime;
    }
    this.traces.push(this.activeTrace);
    const trace = this.activeTrace;
    this.activeTrace = undefined;
    this.spanStack = [];
    return trace;
  }

  /** Wrap an async function call with automatic tracing. */
  async traced<T>(
    name: string,
    type: TraceSpan["type"],
    inputs: Record<string, unknown>,
    fn: () => Promise<T>
  ): Promise<T> {
    this.startSpan(name, type, inputs);
    try {
      const result = await fn();
      this.endSpan({ result: String(result).slice(0, 100) });
      return result;
    } catch (err) {
      this.endSpan({}, String(err));
      throw err;
    }
  }

  /** Get all recorded traces. */
  getTraces(): Trace[] {
    return this.traces;
  }

  /** Print a summary of all traces. */
  printSummary(): void {
    console.log(`\nTracing Summary (session: ${this.sessionId}):`);
    for (const trace of this.traces) {
      console.log(`  [${trace.traceId}] ${trace.rootSpan.name}: ${trace.totalDurationMs}ms, ${trace.totalTokens ?? "?"} tokens`);
      this.printSpan(trace.rootSpan, "    ");
    }
  }

  private printSpan(span: TraceSpan, indent: string): void {
    const dur = span.endTime ? `${span.endTime - span.startTime}ms` : "running";
    const status = span.error ? " ❌" : "";
    console.log(`${indent}[${span.type}] ${span.name}: ${dur}${status}`);
    span.children.forEach((child) => this.printSpan(child, indent + "  "));
  }
}

// Demo
async function main(): Promise<void> {
  const tracer = new LangSmithTracer("demo-session");
  tracer.startTrace("rag-agent-demo", { query: "What is RAG?" });

  await tracer.traced("retrieve-docs", "tool", { query: "What is RAG?" }, async () => {
    await new Promise((r) => setTimeout(r, 10));
    return ["doc1", "doc2"];
  });

  await tracer.traced("generate-answer", "llm", { model: "gpt-4o-mini" }, async () => {
    await new Promise((r) => setTimeout(r, 20));
    return "RAG grounds LLM answers in retrieved context.";
  });

  tracer.endTrace(512);
  tracer.printSummary();
}

main().catch(console.error);
