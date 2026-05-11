/**
 * Token usage tracker — audit trail and cost estimator for LLM calls.
 *
 * Records every API call's token usage, computes running costs, enforces
 * budget caps, and generates human-readable reports.
 * See: docs/03-memory-and-retrieval/01-short-term-memory.md
 */

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface CallRecord {
  id: string;
  model: string;
  component: string;
  usage: TokenUsage;
  costUsd: number;
  timestamp: number;
}

export interface TrackerReport {
  totalCalls: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalTokens: number;
  totalCostUsd: number;
  byModel: Record<string, { calls: number; tokens: number; costUsd: number }>;
  byComponent: Record<string, { calls: number; tokens: number; costUsd: number }>;
}

// Pricing per 1k tokens (input, output) USD
const MODEL_PRICING: Record<string, [number, number]> = {
  "gpt-4o":      [0.005,   0.015],
  "gpt-4o-mini": [0.00015, 0.0006],
};

/**
 * Thread-safe (single-threaded JS) token usage tracker.
 */
export class TokenTracker {
  private records: CallRecord[] = [];
  private totalCostUsd = 0;
  private budgetUsd?: number;
  private callCounter = 0;

  constructor(options: { budgetUsd?: number } = {}) {
    this.budgetUsd = options.budgetUsd;
  }

  /** Record a completed API call. */
  record(model: string, component: string, usage: TokenUsage): CallRecord {
    const [inputCost, outputCost] = MODEL_PRICING[model] ?? [0, 0];
    const costUsd =
      (usage.promptTokens / 1000) * inputCost +
      (usage.completionTokens / 1000) * outputCost;

    this.totalCostUsd += costUsd;

    if (this.budgetUsd !== undefined && this.totalCostUsd > this.budgetUsd) {
      throw new Error(
        `Budget cap exceeded: $${this.totalCostUsd.toFixed(4)} > $${this.budgetUsd}`
      );
    }

    const rec: CallRecord = {
      id: `call-${++this.callCounter}`,
      model,
      component,
      usage,
      costUsd,
      timestamp: Date.now(),
    };
    this.records.push(rec);
    return rec;
  }

  /** Generate a summary report. */
  report(): TrackerReport {
    const byModel: TrackerReport["byModel"] = {};
    const byComponent: TrackerReport["byComponent"] = {};

    for (const r of this.records) {
      if (!byModel[r.model]) byModel[r.model] = { calls: 0, tokens: 0, costUsd: 0 };
      byModel[r.model].calls++;
      byModel[r.model].tokens += r.usage.totalTokens;
      byModel[r.model].costUsd += r.costUsd;

      if (!byComponent[r.component]) byComponent[r.component] = { calls: 0, tokens: 0, costUsd: 0 };
      byComponent[r.component].calls++;
      byComponent[r.component].tokens += r.usage.totalTokens;
      byComponent[r.component].costUsd += r.costUsd;
    }

    return {
      totalCalls: this.records.length,
      totalPromptTokens: this.records.reduce((s, r) => s + r.usage.promptTokens, 0),
      totalCompletionTokens: this.records.reduce((s, r) => s + r.usage.completionTokens, 0),
      totalTokens: this.records.reduce((s, r) => s + r.usage.totalTokens, 0),
      totalCostUsd: this.totalCostUsd,
      byModel,
      byComponent,
    };
  }

  /** Print a formatted report to stdout. */
  printReport(): void {
    const r = this.report();
    console.log("\nToken Usage Report:");
    console.log(`  Total calls  : ${r.totalCalls}`);
    console.log(`  Total tokens : ${r.totalTokens}`);
    console.log(`  Total cost   : $${r.totalCostUsd.toFixed(6)}`);
    console.log("\n  By model:");
    for (const [model, s] of Object.entries(r.byModel)) {
      console.log(`    ${model}: ${s.calls} calls, ${s.tokens} tokens, $${s.costUsd.toFixed(6)}`);
    }
    console.log("\n  By component:");
    for (const [comp, s] of Object.entries(r.byComponent)) {
      console.log(`    ${comp}: ${s.calls} calls, ${s.tokens} tokens`);
    }
  }

  /** Reset all records. */
  reset(): void {
    this.records = [];
    this.totalCostUsd = 0;
    this.callCounter = 0;
  }

  get totalCost(): number {
    return this.totalCostUsd;
  }
}

// Demo
function main(): void {
  const tracker = new TokenTracker({ budgetUsd: 1.0 });

  tracker.record("gpt-4o", "rag-retriever", { promptTokens: 512, completionTokens: 128, totalTokens: 640 });
  tracker.record("gpt-4o-mini", "embedding", { promptTokens: 256, completionTokens: 0, totalTokens: 256 });
  tracker.record("gpt-4o", "answer-gen", { promptTokens: 1024, completionTokens: 256, totalTokens: 1280 });

  tracker.printReport();
}

main();
