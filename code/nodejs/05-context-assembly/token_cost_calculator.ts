/**
 * Token Cost Calculator — cost modeling for LLM context optimization.
 *
 * Calculates token costs across models, identifies waste, and suggests
 * concrete optimizations with projected savings.
 * See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
 */

// USD per 1M tokens (input, output)
const PRICING: Record<string, [number, number]> = {
  "gpt-4o":             [5.00,  15.00],
  "gpt-4o-mini":        [0.15,   0.60],
  "gpt-4-turbo":        [10.00, 30.00],
  "claude-3-5-sonnet":  [3.00,  15.00],
  "claude-3-haiku":     [0.25,   1.25],
};

const CHARS_PER_TOKEN = 4; // approximation

export function approxTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export interface CostEstimate {
  model: string;
  inputTokens: number;
  outputTokens: number;
  inputCostUsd: number;
  outputCostUsd: number;
  totalCostUsd: number;
}

/** Estimate cost for a single API call. */
export function estimateCost(
  model: string,
  inputTokens: number,
  outputTokens: number
): CostEstimate {
  const [inputRate, outputRate] = PRICING[model] ?? [0, 0];
  const inputCostUsd = (inputTokens / 1_000_000) * inputRate;
  const outputCostUsd = (outputTokens / 1_000_000) * outputRate;
  return {
    model,
    inputTokens,
    outputTokens,
    inputCostUsd,
    outputCostUsd,
    totalCostUsd: inputCostUsd + outputCostUsd,
  };
}

export interface OptimizationSuggestion {
  area: string;
  currentTokens: number;
  optimizedTokens: number;
  savedTokens: number;
  projectedSavingUsd: number;
  suggestion: string;
}

export interface WasteReport {
  totalInputTokens: number;
  redundantSystemTokens: number;
  redundantHistoryTokens: number;
  suggestions: OptimizationSuggestion[];
}

/**
 * Analyze a prompt for token waste and return optimization suggestions.
 */
export function analyzeWaste(
  systemPrompt: string,
  messages: { role: string; content: string }[],
  model = "gpt-4o-mini",
  expectedOutputTokens = 256
): WasteReport {
  const sysTokens = approxTokens(systemPrompt);
  const historyTokens = messages.reduce((s, m) => s + approxTokens(m.content), 0);
  const totalInputTokens = sysTokens + historyTokens;

  const [inputRate] = PRICING[model] ?? [0, 0];
  const costPerToken = inputRate / 1_000_000;

  const suggestions: OptimizationSuggestion[] = [];

  // System prompt too large
  if (sysTokens > 500) {
    const optimized = Math.floor(sysTokens * 0.6);
    suggestions.push({
      area: "System prompt",
      currentTokens: sysTokens,
      optimizedTokens: optimized,
      savedTokens: sysTokens - optimized,
      projectedSavingUsd: (sysTokens - optimized) * costPerToken,
      suggestion: "Compress system prompt by removing redundant instructions and examples.",
    });
  }

  // Long history
  if (messages.length > 10 && historyTokens > 2000) {
    const optimized = Math.floor(historyTokens * 0.4);
    suggestions.push({
      area: "Conversation history",
      currentTokens: historyTokens,
      optimizedTokens: optimized,
      savedTokens: historyTokens - optimized,
      projectedSavingUsd: (historyTokens - optimized) * costPerToken,
      suggestion: "Summarize older conversation turns to reduce history tokens.",
    });
  }

  return {
    totalInputTokens,
    redundantSystemTokens: sysTokens > 500 ? sysTokens - 500 : 0,
    redundantHistoryTokens: historyTokens > 2000 ? historyTokens - 2000 : 0,
    suggestions,
  };
}

/** Compare cost across models for the same prompt. */
export function compareModelCosts(
  inputTokens: number,
  outputTokens: number
): CostEstimate[] {
  return Object.keys(PRICING).map((model) => estimateCost(model, inputTokens, outputTokens));
}

/** Print a cost comparison table. */
export function printCostComparison(estimates: CostEstimate[]): void {
  console.log("\nModel Cost Comparison:");
  console.log("  Model".padEnd(25) + "Input".padStart(10) + "Output".padStart(10) + "Total".padStart(12));
  console.log("  " + "-".repeat(55));
  for (const e of estimates.sort((a, b) => a.totalCostUsd - b.totalCostUsd)) {
    console.log(
      `  ${e.model.padEnd(23)} $${e.inputCostUsd.toFixed(6).padStart(9)} $${e.outputCostUsd.toFixed(6).padStart(9)} $${e.totalCostUsd.toFixed(6).padStart(10)}`
    );
  }
}

// Demo
function main(): void {
  const inputTokens = 2000;
  const outputTokens = 500;

  console.log("Token Cost Calculator Demo");
  printCostComparison(compareModelCosts(inputTokens, outputTokens));

  const sysPrompt = "You are a helpful assistant. ".repeat(30);
  const msgs = Array.from({ length: 15 }, (_, i) => ({
    role: i % 2 === 0 ? "user" : "assistant",
    content: "This is a conversation turn with some content. ".repeat(5),
  }));

  const waste = analyzeWaste(sysPrompt, msgs);
  console.log("\nWaste Analysis:");
  console.log(`  Total input tokens: ${waste.totalInputTokens}`);
  waste.suggestions.forEach((s) => {
    console.log(`  [${s.area}] Save ${s.savedTokens} tokens ($${s.projectedSavingUsd.toFixed(6)})`);
    console.log(`    → ${s.suggestion}`);
  });
}

main();
