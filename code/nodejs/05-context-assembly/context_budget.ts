/**
 * Context Budget Manager — zone-based token allocation and enforcement.
 *
 * Every LLM call passes through the budget enforcer which ensures each zone
 * stays within its allocated token quota.  When a zone overflows, the enforcer
 * applies the appropriate compression strategy and records what it did.
 *
 * See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
 */

import { get_encoding, encoding_for_model, type TiktokenModel } from "tiktoken";

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

function getEncoding(model: string = "gpt-4o") {
  try {
    return encoding_for_model(model as TiktokenModel);
  } catch {
    return get_encoding("cl100k_base");
  }
}

export type ContentInput = string | Record<string, unknown> | Array<unknown> | null | undefined;

/**
 * Count tokens for strings, message arrays, tool definition arrays, or dicts.
 *
 * Follows OpenAI chat-completion token-counting rules (3 overhead tokens per
 * message, +1 for the `name` field, 3 primer tokens appended).
 */
export function countTokens(content: ContentInput, model: string = "gpt-4o"): number {
  if (content == null) return 0;

  const enc = getEncoding(model);

  if (typeof content === "string") {
    const count = enc.encode(content).length;
    enc.free();
    return count;
  }

  if (Array.isArray(content)) {
    if (content.length === 0) {
      enc.free();
      return 0;
    }

    // Message list: every element has a "role" key
    if (
      content.every(
        (m): m is Record<string, unknown> =>
          typeof m === "object" && m !== null && "role" in m
      )
    ) {
      const TOKENS_PER_MESSAGE = 3;
      const TOKENS_PER_NAME = 1;
      let total = 0;
      for (const msg of content as Record<string, unknown>[]) {
        total += TOKENS_PER_MESSAGE;
        for (const [key, val] of Object.entries(msg)) {
          if (typeof val === "string") {
            total += enc.encode(val).length;
          } else if (val != null) {
            total += enc.encode(JSON.stringify(val)).length;
          }
          if (key === "name") total += TOKENS_PER_NAME;
        }
      }
      total += 3; // reply primer
      enc.free();
      return total;
    }

    // List of tool definitions or arbitrary objects
    const total = content.reduce(
      (sum, item) => sum + countTokens(item as ContentInput, model),
      0
    );
    enc.free();
    return total;
  }

  if (typeof content === "object") {
    const count = enc.encode(JSON.stringify(content)).length;
    enc.free();
    return count;
  }

  enc.free();
  return 0;
}

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

export interface ZoneAudit {
  zone: string;
  originalTokens: number;
  budgetTokens: number;
  finalTokens: number;
  actionTaken:
    | "within_budget"
    | "truncated"
    | "sliding_window"
    | "filtered"
    | "reserved";
  readonly tokensSaved: number;
}

function makeZoneAudit(
  zone: string,
  originalTokens: number,
  budgetTokens: number,
  finalTokens: number,
  actionTaken: ZoneAudit["actionTaken"]
): ZoneAudit {
  return {
    zone,
    originalTokens,
    budgetTokens,
    finalTokens,
    actionTaken,
    get tokensSaved() {
      return Math.max(0, this.originalTokens - this.finalTokens);
    },
  };
}

export interface EnforceResult {
  systemPrompt: string;
  messages: Record<string, unknown>[];
  dynamicContext: string;
  toolDefinitions: Record<string, unknown>[];
  audit: Record<string, ZoneAudit>;
  warnings: string[];
  readonly totalTokensSaved: number;
  readonly totalTokensUsed: number;
}

function makeEnforceResult(partial: {
  systemPrompt: string;
  messages: Record<string, unknown>[];
  dynamicContext: string;
  toolDefinitions: Record<string, unknown>[];
}): EnforceResult {
  const audit: Record<string, ZoneAudit> = {};
  const warnings: string[] = [];
  return {
    ...partial,
    audit,
    warnings,
    get totalTokensSaved() {
      return Object.values(this.audit).reduce(
        (sum, a) => sum + a.tokensSaved,
        0
      );
    },
    get totalTokensUsed() {
      return Object.values(this.audit).reduce(
        (sum, a) => sum + a.finalTokens,
        0
      );
    },
  };
}

export class BudgetExceededError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BudgetExceededError";
  }
}

// ---------------------------------------------------------------------------
// Pricing (USD per 1 M tokens)
// ---------------------------------------------------------------------------

const PRICING: Record<string, { input: number; output?: number }> = {
  "gpt-4o":            { input: 2.50,   output: 10.00 },
  "gpt-4o-mini":       { input: 0.15,   output: 0.60 },
  "claude-3.5-sonnet": { input: 3.00,   output: 15.00 },
  "claude-3-haiku":    { input: 0.25,   output: 1.25 },
  "gemini-1.5-pro":    { input: 3.50,   output: 10.50 },
  "gemini-1.5-flash":  { input: 0.075,  output: 0.30 },
};

// ---------------------------------------------------------------------------
// ContextBudget
// ---------------------------------------------------------------------------

export type ZoneName =
  | "system_prompt"
  | "tool_definitions"
  | "dynamic_context"
  | "conversation_history"
  | "response_buffer";

/**
 * Define and enforce token allocation across context zones.
 *
 * @example
 * ```ts
 * const budget = new ContextBudget(128_000);
 * const result = budget.enforce(systemPrompt, messages, dynamicContext, tools);
 * console.log(result.totalTokensSaved, "tokens saved");
 * ```
 */
export class ContextBudget {
  readonly totalTokens: number;
  readonly model: string;
  allocations: Record<ZoneName, number>;

  constructor(totalTokens: number = 128_000, model: string = "gpt-4o") {
    this.totalTokens = totalTokens;
    this.model = model;
    this.allocations = {
      system_prompt:        0.02,
      tool_definitions:     0.05,
      dynamic_context:      0.45,
      conversation_history: 0.33,
      response_buffer:      0.15,
    };
  }

  // ----------------------------------------------------------------
  // Allocation management
  // ----------------------------------------------------------------

  /**
   * Set the allocation fraction for a zone.
   * The sum of all allocations must remain <= 1.0.
   */
  setAllocation(zone: ZoneName, percentage: number): void {
    if (!(zone in this.allocations)) {
      throw new Error(`Unknown zone '${zone}'`);
    }
    if (percentage < 0 || percentage > 1) {
      throw new Error(`Percentage must be in [0, 1]; got ${percentage}`);
    }
    const newTotal =
      Object.entries(this.allocations)
        .filter(([k]) => k !== zone)
        .reduce((sum, [, v]) => sum + v, 0) + percentage;
    if (newTotal > 1.0 + 1e-9) {
      throw new Error(
        `Allocation total would be ${newTotal.toFixed(3)} > 1.0 after setting '${zone}' to ${percentage}`
      );
    }
    this.allocations[zone] = percentage;
  }

  /** Get the token budget for a zone. */
  getTokenBudget(zone: ZoneName): number {
    if (!(zone in this.allocations)) throw new Error(`Unknown zone '${zone}'`);
    return Math.floor(this.totalTokens * this.allocations[zone]);
  }

  /** Get all zone token budgets. */
  getAllBudgets(): Record<ZoneName, number> {
    return Object.fromEntries(
      (Object.keys(this.allocations) as ZoneName[]).map((z) => [
        z,
        this.getTokenBudget(z),
      ])
    ) as Record<ZoneName, number>;
  }

  // ----------------------------------------------------------------
  // Measurement
  // ----------------------------------------------------------------

  /** Measure token usage for content in a zone. */
  measureZone(_zone: ZoneName, content: ContentInput): number {
    return countTokens(content, this.model);
  }

  // ----------------------------------------------------------------
  // Enforcement
  // ----------------------------------------------------------------

  /**
   * Enforce the budget on all zones.
   *
   * Compression strategies:
   * - **systemPrompt**: Truncate from end, preserve first instructions.
   * - **toolDefinitions**: Keep tools that fit; trim descriptions.
   * - **dynamicContext**: Truncate to budget.
   * - **conversationHistory**: Sliding window — drop oldest messages.
   * - **responseBuffer**: Reservation only; nothing to compress.
   */
  enforce(
    systemPrompt: string,
    messages: Record<string, unknown>[],
    dynamicContext: string = "",
    toolDefinitions: Record<string, unknown>[] = []
  ): EnforceResult {
    const result = makeEnforceResult({
      systemPrompt,
      messages: [...messages],
      dynamicContext,
      toolDefinitions: [...toolDefinitions],
    });
    const budgets = this.getAllBudgets();

    // 1. System prompt
    const spTokens = this.measureZone("system_prompt", systemPrompt);
    const spBudget = budgets.system_prompt;
    if (spTokens <= spBudget) {
      result.audit.system_prompt = makeZoneAudit(
        "system_prompt", spTokens, spBudget, spTokens, "within_budget"
      );
    } else {
      const compressed = this.compressSystemPrompt(systemPrompt, spBudget);
      const finalTok = countTokens(compressed, this.model);
      result.systemPrompt = compressed;
      const msg = `system_prompt: ${spTokens.toLocaleString()} tokens exceeded budget ${spBudget.toLocaleString()}; truncated to ${finalTok.toLocaleString()} tokens.`;
      result.warnings.push(msg);
      result.audit.system_prompt = makeZoneAudit(
        "system_prompt", spTokens, spBudget, finalTok, "truncated"
      );
    }

    // 2. Tool definitions
    const tdTokens = this.measureZone("tool_definitions", toolDefinitions);
    const tdBudget = budgets.tool_definitions;
    if (tdTokens <= tdBudget) {
      result.audit.tool_definitions = makeZoneAudit(
        "tool_definitions", tdTokens, tdBudget, tdTokens, "within_budget"
      );
    } else {
      const compressed = this.compressToolDefinitions(toolDefinitions, tdBudget);
      const finalTok = countTokens(compressed, this.model);
      result.toolDefinitions = compressed;
      const msg = `tool_definitions: ${tdTokens.toLocaleString()} tokens exceeded budget ${tdBudget.toLocaleString()}; trimmed to ${finalTok.toLocaleString()} tokens (${compressed.length}/${toolDefinitions.length} tools kept).`;
      result.warnings.push(msg);
      result.audit.tool_definitions = makeZoneAudit(
        "tool_definitions", tdTokens, tdBudget, finalTok, "filtered"
      );
    }

    // 3. Dynamic context
    const dcTokens = this.measureZone("dynamic_context", dynamicContext);
    const dcBudget = budgets.dynamic_context;
    if (dcTokens <= dcBudget) {
      result.audit.dynamic_context = makeZoneAudit(
        "dynamic_context", dcTokens, dcBudget, dcTokens, "within_budget"
      );
    } else {
      const compressed = this.compressDynamicContext(dynamicContext, dcBudget);
      const finalTok = countTokens(compressed, this.model);
      result.dynamicContext = compressed;
      const msg = `dynamic_context: ${dcTokens.toLocaleString()} tokens exceeded budget ${dcBudget.toLocaleString()}; truncated to ${finalTok.toLocaleString()} tokens.`;
      result.warnings.push(msg);
      result.audit.dynamic_context = makeZoneAudit(
        "dynamic_context", dcTokens, dcBudget, finalTok, "truncated"
      );
    }

    // 4. Conversation history
    const history = messages.filter((m) => m["role"] !== "system");
    const histTokens = countTokens(history, this.model);
    const histBudget = budgets.conversation_history;
    if (histTokens <= histBudget) {
      result.audit.conversation_history = makeZoneAudit(
        "conversation_history", histTokens, histBudget, histTokens, "within_budget"
      );
    } else {
      const compressed = this.compressHistory(history, histBudget);
      const finalTok = countTokens(compressed, this.model);
      const systemMsgs = result.messages.filter((m) => m["role"] === "system");
      result.messages = [...systemMsgs, ...compressed];
      const msg = `conversation_history: ${histTokens.toLocaleString()} tokens exceeded budget ${histBudget.toLocaleString()}; sliding window applied, ${finalTok.toLocaleString()} tokens kept.`;
      result.warnings.push(msg);
      result.audit.conversation_history = makeZoneAudit(
        "conversation_history", histTokens, histBudget, finalTok, "sliding_window"
      );
    }

    // 5. Response buffer
    const rbBudget = budgets.response_buffer;
    result.audit.response_buffer = makeZoneAudit(
      "response_buffer", 0, rbBudget, 0, "reserved"
    );

    return result;
  }

  // ----------------------------------------------------------------
  // Compression strategies
  // ----------------------------------------------------------------

  private compressSystemPrompt(prompt: string, maxTokens: number): string {
    const enc = getEncoding(this.model);
    const tokens = enc.encode(prompt);
    if (tokens.length <= maxTokens) {
      enc.free();
      return prompt;
    }
    const result = new TextDecoder().decode(enc.decode(tokens.slice(0, maxTokens)));
    enc.free();
    return result;
  }

  private compressToolDefinitions(
    tools: Record<string, unknown>[],
    maxTokens: number
  ): Record<string, unknown>[] {
    const kept: Record<string, unknown>[] = [];
    let used = 0;
    for (const tool of tools) {
      const t = countTokens(tool, this.model);
      if (used + t <= maxTokens) {
        kept.push(tool);
        used += t;
      }
      if (maxTokens - used < 50) break;
    }
    return kept;
  }

  private compressDynamicContext(context: string, maxTokens: number): string {
    const enc = getEncoding(this.model);
    const tokens = enc.encode(context);
    if (tokens.length <= maxTokens) {
      enc.free();
      return context;
    }
    const result = new TextDecoder().decode(enc.decode(tokens.slice(0, maxTokens)));
    enc.free();
    return result;
  }

  private compressHistory(
    messages: Record<string, unknown>[],
    maxTokens: number
  ): Record<string, unknown>[] {
    if (messages.length === 0) return messages;

    const kept: Record<string, unknown>[] = [];
    let used = 0;
    for (const msg of [...messages].reverse()) {
      const t = countTokens([msg], this.model);
      if (used + t <= maxTokens) {
        kept.unshift(msg);
        used += t;
      } else {
        break;
      }
    }

    if (kept.length === 0 && messages.length > 0) {
      return [messages[messages.length - 1]];
    }
    return kept;
  }

  // ----------------------------------------------------------------
  // Cost estimation
  // ----------------------------------------------------------------

  /**
   * Return the estimated USD cost for `inputTokens` + `outputTokens`.
   * Returns 0 if the model is not in the pricing table.
   */
  estimateCost(
    inputTokens: number,
    outputTokens: number,
    model?: string
  ): number {
    const m = (model ?? this.model).toLowerCase();
    const pricing = PRICING[m];
    if (!pricing) return 0;
    return (
      (inputTokens  / 1_000 * pricing.input) +
      (outputTokens / 1_000 * (pricing.output ?? 0))
    );
  }
}
