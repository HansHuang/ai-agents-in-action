/**
 * Entry point for 05-context-assembly demos.
 *
 * Run: npx tsx main.ts
 */

import { evaluateCondition } from "./condition_engine.js";
import { analyzeDensity, rankByDensity } from "./density_analyzer.js";
import { extractiveSummarize } from "./extractive_summarizer.js";
import { buildStructuredContext, type ContextChunk } from "./context_optimizer.js";
import { compareModelCosts, printCostComparison, analyzeWaste } from "./token_cost_calculator.js";

function demoConditionEngine(): void {
  console.log("=== Condition Engine ===");
  const vars = { plan: "premium", country: "US", score: 0.85 };
  const conditions = [
    "plan == 'premium'",
    "score > 0.7",
    "plan == 'premium' AND country == 'US'",
    "plan == 'free' OR country == 'US'",
  ];
  conditions.forEach((c) => console.log(`  "${c}" → ${evaluateCondition(c, vars)}`));
}

function demoDensityAnalyzer(): void {
  console.log("\n=== Density Analyzer ===");
  const samples = [
    "Actually it is very important to note that this is quite relevant.",
    "GPT-4 achieves 86.4% on MMLU with 128K context window and 1.8T parameters.",
  ];
  rankByDensity(samples).forEach((r) => {
    console.log(`  score=${r.score.toFixed(3)}: "${r.text.slice(0, 60)}..."`);
  });
}

function demoExtractiveSummarizer(): void {
  console.log("\n=== Extractive Summarizer ===");
  const text =
    "RAG reduces hallucinations by grounding answers in retrieved documents. " +
    "The context window is a finite resource that must be managed carefully. " +
    "Vector databases store embedding vectors for fast nearest-neighbour search. " +
    "Fine-tuning adapts a base model to a specific domain or task.";
  const result = extractiveSummarize(text, "How does RAG help with accuracy?", { maxSentences: 2 });
  result.forEach((s, i) => console.log(`  [${i + 1}] ${s}`));
}

function demoContextOptimizer(): void {
  console.log("\n=== Context Optimizer ===");
  const chunks: ContextChunk[] = [
    { id: "c1", text: "RAG grounds answers in retrieved context.", priority: 0.9, source: "rag-overview" },
    { id: "c2", text: "TypeScript adds static typing to JavaScript.", priority: 0.4, source: "ts-intro" },
    { id: "c3", text: "TypeScript adds types to JavaScript.", priority: 0.3, source: "ts-dup" },
    { id: "c4", text: "Vector databases enable semantic search.", priority: 0.7, source: "vector-db" },
  ];
  const ctx = buildStructuredContext(chunks, { addTableOfContents: true, deduplicationThreshold: 0.7 });
  console.log(ctx);
}

function demoCostCalculator(): void {
  console.log("\n=== Token Cost Calculator ===");
  printCostComparison(compareModelCosts(1500, 400));
  const waste = analyzeWaste("You are a helpful assistant. ".repeat(25), [], "gpt-4o-mini");
  if (waste.suggestions.length > 0) {
    console.log(`  Suggestion: ${waste.suggestions[0].suggestion}`);
  }
}

async function main(): Promise<void> {
  demoConditionEngine();
  demoDensityAnalyzer();
  demoExtractiveSummarizer();
  demoContextOptimizer();
  demoCostCalculator();
  console.log("\nAll context-assembly demos complete.");
}

main().catch(console.error);
