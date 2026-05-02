/**
 * Token-aware conversation memory manager (TypeScript port).
 *
 * Strategies:
 *   - none          Return full history unchanged.
 *   - truncate      Drop oldest complete turns, keep system prompt + recent.
 *   - summarize     LLM-compress old messages, keep recent verbatim.
 *   - sliding_window Rolling summary of old + verbatim recent (default).
 *
 * Token counting: character-based approximation (~4 chars/token for English).
 * For production use, install `js-tiktoken` and replace `countTokensApprox`
 * with `tiktoken.encoding_for_model(model).encode(text).length`.
 *
 * See: docs/03-memory-and-retrieval/01-short-term-memory.md
 */

import OpenAI from "openai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
  tool_calls?: unknown[];
  tool_call_id?: string;
  [key: string]: unknown;
}

export type Strategy = "none" | "truncate" | "summarize" | "sliding_window";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Tokens reserved for the model's output response */
const OUTPUT_RESERVE = 4096;

/** Per-message formatting overhead (role, separators) */
const MSG_OVERHEAD = 4;

/** Priming overhead for assistant reply */
const PRIMING_OVERHEAD = 2;

/**
 * Approximate token count for a string.
 * English prose: ~4 chars per token.  Use js-tiktoken for exactness.
 */
function approxTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

/**
 * Count tokens in a message list including per-message overhead.
 * Mirrors the Python `count_tokens` implementation.
 */
export function countTokens(messages: Message[]): number {
  let total = 0;
  for (const msg of messages) {
    total += MSG_OVERHEAD;
    for (const [, value] of Object.entries(msg)) {
      if (value == null) continue;
      if (typeof value === "string") {
        total += approxTokens(value);
      } else {
        total += approxTokens(JSON.stringify(value));
      }
    }
  }
  total += PRIMING_OVERHEAD;
  return total;
}

// ---------------------------------------------------------------------------
// Grouping helper
// ---------------------------------------------------------------------------

function groupIntoTurns(messages: Message[]): Message[][] {
  const turns: Message[][] = [];
  let current: Message[] = [];

  for (const msg of messages) {
    current.push(msg);
    if (
      msg.role === "assistant" &&
      msg.content &&
      !msg.tool_calls?.length
    ) {
      turns.push(current);
      current = [];
    }
  }
  if (current.length > 0) turns.push(current);
  return turns;
}

// ---------------------------------------------------------------------------
// Memory Manager
// ---------------------------------------------------------------------------

export class MemoryManager {
  readonly model: string;
  readonly maxTokens: number;
  messages: Message[];

  private readonly _client: OpenAI;
  private _summaryCache: string | null = null;
  private _summaryInputLen = 0;

  constructor(options: {
    model?: string;
    maxTokens?: number;
    systemPrompt?: string;
    client?: OpenAI;
  } = {}) {
    this.model = options.model ?? "gpt-4o";
    this.maxTokens = options.maxTokens ?? 100_000;
    this.messages = [
      { role: "system", content: options.systemPrompt ?? "" },
    ];
    this._client =
      options.client ??
      new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "" });
  }

  // ------------------------------------------------------------------
  // Appenders
  // ------------------------------------------------------------------

  addMessage(message: Message): void {
    this.messages.push(message);
    this._summaryCache = null;
  }

  addUserMessage(content: string): void {
    this.addMessage({ role: "user", content });
  }

  addAssistantMessage(
    content?: string | null,
    toolCalls?: unknown[]
  ): void {
    const msg: Message = { role: "assistant", content: content ?? null };
    if (toolCalls?.length) msg.tool_calls = toolCalls;
    this.addMessage(msg);
  }

  addToolResult(toolCallId: string, content: string): void {
    this.addMessage({ role: "tool", tool_call_id: toolCallId, content });
  }

  // ------------------------------------------------------------------
  // Token count / cost
  // ------------------------------------------------------------------

  tokenCount(): number {
    return countTokens(this.messages);
  }

  estimatedCost(options: {
    inputPricePer1k?: number;
    outputPricePer1k?: number;
  } = {}): {
    inputTokens: number;
    estimatedOutputTokens: number;
    inputCostUsd: number;
    outputCostUsd: number;
    totalCostUsd: number;
  } {
    const inputPricePer1k = options.inputPricePer1k ?? 0.0025;
    const outputPricePer1k = options.outputPricePer1k ?? 0.01;
    const inputTok = this.tokenCount();
    const estimatedOutput = OUTPUT_RESERVE;
    return {
      inputTokens: inputTok,
      estimatedOutputTokens: estimatedOutput,
      inputCostUsd: +(inputTok / 1000 * inputPricePer1k).toFixed(6),
      outputCostUsd: +(estimatedOutput / 1000 * outputPricePer1k).toFixed(6),
      totalCostUsd: +(
        inputTok / 1000 * inputPricePer1k +
        estimatedOutput / 1000 * outputPricePer1k
      ).toFixed(6),
    };
  }

  // ------------------------------------------------------------------
  // Rollback
  // ------------------------------------------------------------------

  rollback(toMessageIndex: number): void {
    if (toMessageIndex < 1 || toMessageIndex > this.messages.length) {
      throw new RangeError(
        `toMessageIndex must be in [1, ${this.messages.length}]; got ${toMessageIndex}`
      );
    }
    this.messages = this.messages.slice(0, toMessageIndex);
    this._summaryCache = null;
  }

  // ------------------------------------------------------------------
  // Strategy dispatch
  // ------------------------------------------------------------------

  async getMessages(
    strategy: Strategy = "sliding_window",
    recentCount = 10
  ): Promise<Message[]> {
    const currentTokens = this.tokenCount();

    if (strategy === "none" || currentTokens <= this.maxTokens) {
      return this.messages;
    }

    if (strategy === "truncate") return this._applyTruncation();
    if (strategy === "summarize") return this._applySummarization(recentCount);
    if (strategy === "sliding_window")
      return this._applySlidingWindow(recentCount);

    throw new Error(
      `Unknown strategy '${strategy}'. Choose: none, truncate, summarize, sliding_window.`
    );
  }

  // ------------------------------------------------------------------
  // Truncation
  // ------------------------------------------------------------------

  private _applyTruncation(): Message[] {
    const systemMsg = this.messages[0];
    const turns = groupIntoTurns(this.messages.slice(1));

    const systemTokens = countTokens([systemMsg]);
    let budget = this.maxTokens - systemTokens;
    const kept: Message[][] = [];

    for (const turn of [...turns].reverse()) {
      const turnTokens = countTokens(turn);
      if (turnTokens <= budget) {
        kept.unshift(turn);
        budget -= turnTokens;
      } else {
        break;
      }
    }

    return [systemMsg, ...kept.flat()];
  }

  // ------------------------------------------------------------------
  // Summarization
  // ------------------------------------------------------------------

  private async _applySummarization(recentCount: number): Promise<Message[]> {
    const systemMsg = this.messages[0];
    const conversation = this.messages.slice(1);

    if (conversation.length <= recentCount) return this.messages;

    const toSummarize = conversation.slice(0, -recentCount);
    const recent = conversation.slice(-recentCount);
    const summary = await this._getOrBuildSummary(toSummarize);

    return [
      systemMsg,
      { role: "user", content: `[Conversation summary: ${summary}]` },
      ...recent,
    ];
  }

  // ------------------------------------------------------------------
  // Sliding window
  // ------------------------------------------------------------------

  private async _applySlidingWindow(recentCount: number): Promise<Message[]> {
    const systemMsg = this.messages[0];
    const conversation = this.messages.slice(1);

    if (conversation.length <= recentCount) return this.messages;

    const toSummarize = conversation.slice(0, -recentCount);
    const recent = conversation.slice(-recentCount);
    const summary = await this._getOrBuildSummary(toSummarize);

    const result: Message[] = [systemMsg];
    if (summary) {
      result.push({ role: "user", content: `[Conversation so far: ${summary}]` });
    }
    result.push(...recent);
    return result;
  }

  // ------------------------------------------------------------------
  // Summary helpers
  // ------------------------------------------------------------------

  private async _getOrBuildSummary(messages: Message[]): Promise<string> {
    if (
      this._summaryCache !== null &&
      this._summaryInputLen === messages.length
    ) {
      return this._summaryCache;
    }

    const summary = await this._callSummarizer(messages);
    this._summaryCache = summary;
    this._summaryInputLen = messages.length;
    return summary;
  }

  private async _callSummarizer(messages: Message[]): Promise<string> {
    const formatted = formatMessagesForSummary(messages);
    const response = await this._client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: SUMMARIZER_PROMPT,
        },
        { role: "user", content: formatted },
      ],
      max_tokens: 512,
    });
    return response.choices[0].message.content ?? "";
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SUMMARIZER_PROMPT = `Summarize the following conversation between a user and an AI assistant.
Focus on information the assistant needs to continue helping the user.

INCLUDE:
- The user's original request and any changes to it
- Information gathered from tools (with specific values: numbers, dates, names)
- Decisions the assistant made and why
- Actions taken and their outcomes
- Pending tasks or unanswered questions

FORMAT: Dense paragraph, third person past tense. Be concise but complete.`;

function formatMessagesForSummary(messages: Message[]): string {
  return messages
    .map((msg) => {
      const role = msg.role.toUpperCase();
      if (msg.tool_calls) {
        return `${role} [tool_call]: ${JSON.stringify(msg.tool_calls)}`;
      }
      if (msg.role === "tool") {
        return `TOOL RESULT [${msg.tool_call_id ?? ""}]: ${msg.content ?? ""}`;
      }
      if (msg.content) return `${role}: ${msg.content}`;
      return null;
    })
    .filter(Boolean)
    .join("\n");
}

// ---------------------------------------------------------------------------
// Token Tracker
// ---------------------------------------------------------------------------

export const PRICING: Record<string, { input: number; output: number }> = {
  "gpt-4o": { input: 0.0025, output: 0.01 },
  "gpt-4o-mini": { input: 0.00015, output: 0.0006 },
  "gpt-3.5-turbo": { input: 0.0005, output: 0.0015 },
};

export interface TokenUsageRecord {
  model: string;
  inputTokens: number;
  outputTokens: number;
  timestamp: string;
  purpose: string;
}

export class TokenTracker {
  private _records: TokenUsageRecord[] = [];
  readonly budgetCap: number | null;
  private _budgetWarned = false;

  constructor(options: { budgetCap?: number } = {}) {
    this.budgetCap = options.budgetCap ?? null;
  }

  recordCall(
    model: string,
    inputTokens: number,
    outputTokens: number,
    purpose = ""
  ): TokenUsageRecord {
    const record: TokenUsageRecord = {
      model,
      inputTokens,
      outputTokens,
      timestamp: new Date().toISOString(),
      purpose,
    };
    this._records.push(record);

    if (this.budgetCap !== null) {
      const frac = this.totalCost() / this.budgetCap;
      if (frac >= 0.8 && !this._budgetWarned) {
        this._budgetWarned = true;
        console.warn(
          `[TokenTracker] Budget warning: $${this.totalCost().toFixed(4)} / $${this.budgetCap} (${(frac * 100).toFixed(0)}% used)`
        );
      }
    }
    return record;
  }

  totalInputTokens(): number {
    return this._records.reduce((s, r) => s + r.inputTokens, 0);
  }

  totalOutputTokens(): number {
    return this._records.reduce((s, r) => s + r.outputTokens, 0);
  }

  totalCost(): number {
    return this._records.reduce((s, r) => {
      const p = PRICING[r.model];
      if (!p) return s;
      return s + r.inputTokens / 1000 * p.input + r.outputTokens / 1000 * p.output;
    }, 0);
  }

  isBudgetExceeded(): boolean {
    return this.budgetCap !== null && this.totalCost() >= this.budgetCap;
  }

  generateReport(): string {
    const lines: string[] = [
      "=".repeat(60),
      "TOKEN USAGE REPORT",
      "=".repeat(60),
      `  Total calls:         ${this._records.length}`,
      `  Total input tokens:  ${this.totalInputTokens().toLocaleString()}`,
      `  Total output tokens: ${this.totalOutputTokens().toLocaleString()}`,
      `  Total cost:          $${this.totalCost().toFixed(6)}`,
    ];
    if (this.budgetCap !== null) {
      const pct = (this.totalCost() / this.budgetCap) * 100;
      lines.push(`  Budget cap:          $${this.budgetCap.toFixed(2)}`);
      lines.push(`  Budget used:         ${pct.toFixed(1)}%`);
    }
    return lines.join("\n");
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const SYSTEM = "You are a helpful research assistant.";
  const mem = new MemoryManager({
    model: "gpt-4o",
    maxTokens: 2000,
    systemPrompt: SYSTEM,
  });

  for (let i = 0; i < 20; i++) {
    mem.addUserMessage(`Turn ${i}: What is topic ${i % 5}?`);
    mem.addAssistantMessage(`Answer ${i}: Here is information about topic ${i % 5}.`);
  }

  console.log(
    `Full history: ${mem.tokenCount()} tokens, ${mem.messages.length} messages`
  );

  // Override summarizer for demo
  (mem as unknown as { _callSummarizer: (m: Message[]) => Promise<string> })
    ._callSummarizer = async (msgs) =>
    `[Demo summary: ${msgs.length} messages compressed]`;

  for (const strategy of ["truncate", "summarize", "sliding_window"] as Strategy[]) {
    const msgs = await mem.getMessages(strategy, 6);
    console.log(
      `  ${strategy.padEnd(15)}: ${countTokens(msgs).toString().padStart(5)} tokens, ${msgs.length.toString().padStart(3)} messages`
    );
  }

  const tracker = new TokenTracker({ budgetCap: 0.01 });
  tracker.recordCall("gpt-4o", 1500, 350, "plan");
  tracker.recordCall("gpt-4o-mini", 800, 200, "summarize");
  console.log(tracker.generateReport());
}

main().catch(console.error);
