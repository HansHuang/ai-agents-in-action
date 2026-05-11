/**
 * Provider benchmark — measures latency, token throughput, and cost across
 * multiple OpenAI-compatible models on a fixed prompt set.
 * See: docs/05-the-tool-ecosystem/01-model-providers.md
 */

import OpenAI from "openai";

interface BenchmarkResult {
  model: string;
  promptTokens: number;
  completionTokens: number;
  latencyMs: number;
  tokensPerSec: number;
  estimatedCostUsd: number;
}

// Cost per 1k tokens (input, output) in USD
const MODEL_PRICING: Record<string, [number, number]> = {
  "gpt-4o":        [0.005,   0.015],
  "gpt-4o-mini":   [0.00015, 0.0006],
};

const BENCHMARK_PROMPTS = [
  "What is the capital of France?",
  "Summarise the key principles of object-oriented programming in 3 bullet points.",
  "Write a haiku about software engineering.",
];

async function benchmarkModel(
  model: string,
  client: OpenAI
): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];
  const [inputCost, outputCost] = MODEL_PRICING[model] ?? [0, 0];

  for (const prompt of BENCHMARK_PROMPTS) {
    const t0 = Date.now();
    const response = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
    });
    const latencyMs = Date.now() - t0;
    const usage = response.usage!;
    const costUsd =
      (usage.prompt_tokens / 1000) * inputCost +
      (usage.completion_tokens / 1000) * outputCost;

    results.push({
      model,
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      latencyMs,
      tokensPerSec: Math.round((usage.completion_tokens / latencyMs) * 1000),
      estimatedCostUsd: costUsd,
    });
  }
  return results;
}

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const models = ["gpt-4o-mini", "gpt-4o"];
  const allResults: BenchmarkResult[] = [];

  for (const model of models) {
    console.log(`Benchmarking ${model}...`);
    const results = await benchmarkModel(model, client);
    allResults.push(...results);
  }

  // Aggregate by model
  const byModel = new Map<string, { latency: number[]; cost: number[]; tps: number[] }>();
  for (const r of allResults) {
    if (!byModel.has(r.model)) byModel.set(r.model, { latency: [], cost: [], tps: [] });
    const m = byModel.get(r.model)!;
    m.latency.push(r.latencyMs);
    m.cost.push(r.estimatedCostUsd);
    m.tps.push(r.tokensPerSec);
  }

  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

  console.log(`\n${"Model".padEnd(18)} ${"Avg Latency(ms)".padStart(16)} ${"Avg tok/s".padStart(10)} ${"Avg cost($)".padStart(12)}`);
  console.log("-".repeat(60));
  for (const [model, stats] of byModel) {
    console.log(
      `${model.padEnd(18)} ${String(Math.round(avg(stats.latency))).padStart(16)} ${String(Math.round(avg(stats.tps))).padStart(10)} ${avg(stats.cost).toFixed(6).padStart(12)}`
    );
  }
}

main().catch(console.error);
