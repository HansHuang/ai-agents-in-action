/**
 * provider_comparison.ts — Compare the same agent across multiple LLM providers.
 *
 * Uses the Vercel AI SDK's unified interface to run identical prompts through
 * 7 provider/model combinations and prints a performance + cost comparison table.
 *
 * Usage:
 *   npx tsx provider_comparison.ts
 *
 * Required environment variables (set the ones you have API keys for):
 *   OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, GROQ_API_KEY
 */

import { generateText, tool } from "ai";
import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { groq } from "@ai-sdk/groq";
import { z } from "zod";
import { LanguageModelV1 } from "@ai-sdk/provider";

// ──── Types ───────────────────────────────────────────────────────────────────

interface ProviderConfig {
  provider: string;
  modelName: string;
  model: LanguageModelV1;
  /** USD per 1M tokens */
  pricing: { input: number; output: number };
}

interface BenchmarkResult {
  provider: string;
  modelName: string;
  ttft: number | null;       // seconds to first token
  totalTime: number;         // seconds total
  inputTokens: number;
  outputTokens: number;
  estimatedCostUsd: number;
  calledWeatherTool: boolean;
  calledInvestTool: boolean;
  answerLength: number;
  qualityScore: number;      // 0–10 heuristic
  error: string | null;
}

// ──── Provider catalogue ──────────────────────────────────────────────────────

function buildProviders(): ProviderConfig[] {
  const configs: ProviderConfig[] = [];

  if (process.env.OPENAI_API_KEY) {
    configs.push(
      {
        provider: "OpenAI",
        modelName: "gpt-4o",
        model: openai("gpt-4o"),
        pricing: { input: 2.5, output: 10.0 },
      },
      {
        provider: "OpenAI",
        modelName: "gpt-4o-mini",
        model: openai("gpt-4o-mini"),
        pricing: { input: 0.15, output: 0.6 },
      }
    );
  }

  if (process.env.ANTHROPIC_API_KEY) {
    configs.push(
      {
        provider: "Anthropic",
        modelName: "claude-3-5-sonnet-20241022",
        model: anthropic("claude-3-5-sonnet-20241022"),
        pricing: { input: 3.0, output: 15.0 },
      },
      {
        provider: "Anthropic",
        modelName: "claude-3-haiku-20240307",
        model: anthropic("claude-3-haiku-20240307"),
        pricing: { input: 0.25, output: 1.25 },
      }
    );
  }

  if (process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
    configs.push(
      {
        provider: "Google",
        modelName: "gemini-1.5-pro",
        model: google("gemini-1.5-pro"),
        pricing: { input: 3.5, output: 10.5 },
      },
      {
        provider: "Google",
        modelName: "gemini-1.5-flash",
        model: google("gemini-1.5-flash"),
        pricing: { input: 0.075, output: 0.3 },
      }
    );
  }

  if (process.env.GROQ_API_KEY) {
    configs.push({
      provider: "Groq",
      modelName: "llama-3.1-70b-versatile",
      model: groq("llama-3.1-70b-versatile"),
      pricing: { input: 0.59, output: 0.79 },
    });
  }

  return configs;
}

// ──── Test tools ──────────────────────────────────────────────────────────────

/** Mock weather tool for benchmarking. */
const getWeather = tool({
  description: "Get the current weather for a city.",
  parameters: z.object({
    city: z.string().describe("City name, e.g. 'Tokyo, JP'"),
    units: z.enum(["celsius", "fahrenheit"]).optional(),
  }),
  execute: async ({ city, units = "celsius" }) => ({
    city,
    temperature: units === "celsius" ? 18 : 64,
    condition: "partly cloudy",
    humidity: 72,
    units,
  }),
});

/** Mock stock analysis tool for benchmarking. */
const analyzeStock = tool({
  description: "Get a simple risk assessment for a stock ticker.",
  parameters: z.object({
    ticker: z.string().describe("Stock ticker symbol, e.g. 'AAPL'"),
  }),
  execute: async ({ ticker }) => ({
    ticker,
    currentPrice: 189.5,
    peRatio: 29.4,
    analystConsensus: "hold",
    riskLevel: "moderate",
    note: "Not financial advice. Consult a licensed advisor.",
  }),
});

const TEST_TOOLS = { getWeather, analyzeStock };

// ──── Benchmark harness ───────────────────────────────────────────────────────

const TEST_PROMPT =
  "What's the weather in Tokyo right now? And separately, should I invest in AAPL stock?";

const SYSTEM_PROMPT =
  "You are a helpful assistant. Use the available tools to answer questions accurately.";

/**
 * Estimate a quality score (0–10) based on response characteristics.
 * This is a heuristic — it checks whether the response is substantive
 * and references the data that the tools returned.
 */
function estimateQuality(text: string, calledWeather: boolean, calledInvest: boolean): number {
  let score = 5;
  if (calledWeather) score += 1;
  if (calledInvest) score += 1;
  if (text.toLowerCase().includes("tokyo")) score += 0.5;
  if (text.toLowerCase().includes("aapl") || text.toLowerCase().includes("apple")) score += 0.5;
  if (text.length > 200) score += 0.5;
  if (text.length > 400) score += 0.5;
  return Math.min(10, Math.round(score));
}

/**
 * Run a single provider benchmark.
 */
async function runBenchmark(config: ProviderConfig): Promise<BenchmarkResult> {
  const start = performance.now();
  let ttft: number | null = null;
  let calledWeatherTool = false;
  let calledInvestTool = false;

  try {
    const result = await generateText({
      model: config.model,
      system: SYSTEM_PROMPT,
      prompt: TEST_PROMPT,
      tools: TEST_TOOLS,
      maxSteps: 3,
    });

    const totalTime = (performance.now() - start) / 1000;

    // Check which tools were called
    for (const step of result.steps) {
      for (const tc of step.toolCalls ?? []) {
        if (tc.toolName === "getWeather") calledWeatherTool = true;
        if (tc.toolName === "analyzeStock") calledInvestTool = true;
      }
    }

    const inputTokens = result.usage?.promptTokens ?? 0;
    const outputTokens = result.usage?.completionTokens ?? 0;
    const estimatedCostUsd =
      (inputTokens / 1_000_000) * config.pricing.input +
      (outputTokens / 1_000_000) * config.pricing.output;

    return {
      provider: config.provider,
      modelName: config.modelName,
      ttft, // generateText doesn't expose TTFT; would need streamText
      totalTime,
      inputTokens,
      outputTokens,
      estimatedCostUsd,
      calledWeatherTool,
      calledInvestTool,
      answerLength: result.text.length,
      qualityScore: estimateQuality(result.text, calledWeatherTool, calledInvestTool),
      error: null,
    };
  } catch (err) {
    const totalTime = (performance.now() - start) / 1000;
    return {
      provider: config.provider,
      modelName: config.modelName,
      ttft: null,
      totalTime,
      inputTokens: 0,
      outputTokens: 0,
      estimatedCostUsd: 0,
      calledWeatherTool: false,
      calledInvestTool: false,
      answerLength: 0,
      qualityScore: 0,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

// ──── Table rendering ─────────────────────────────────────────────────────────

function pad(value: string, width: number): string {
  return value.padEnd(width).substring(0, width);
}

function renderTable(results: BenchmarkResult[]): void {
  const header =
    `${"Provider".padEnd(12)} | ${"Model".padEnd(28)} | ${"TTFT".padEnd(6)} | ` +
    `${"Total".padEnd(6)} | ${"Tokens".padEnd(7)} | ${"Cost".padEnd(8)} | ` +
    `${"Weather".padEnd(7)} | ${"Invest".padEnd(6)} | Quality`;
  const separator = "─".repeat(header.length);

  console.log("\n" + separator);
  console.log(header);
  console.log(separator);

  for (const r of results) {
    if (r.error) {
      console.log(
        `${pad(r.provider, 12)} | ${pad(r.modelName, 28)} | ERROR: ${r.error.substring(0, 50)}`
      );
      continue;
    }
    const ttft = r.ttft !== null ? `${r.ttft.toFixed(1)}s` : "n/a";
    const total = `${r.totalTime.toFixed(1)}s`;
    const tokens = (r.inputTokens + r.outputTokens).toLocaleString();
    const cost = `$${r.estimatedCostUsd.toFixed(4)}`;
    const weather = r.calledWeatherTool ? "✓" : "✗";
    const invest = r.calledInvestTool ? "✓" : "✗";
    const quality = `${r.qualityScore}/10`;

    console.log(
      `${pad(r.provider, 12)} | ${pad(r.modelName, 28)} | ${pad(ttft, 6)} | ` +
        `${pad(total, 6)} | ${pad(tokens, 7)} | ${pad(cost, 8)} | ` +
        `${pad(weather, 7)} | ${pad(invest, 6)} | ${quality}`
    );
  }
  console.log(separator + "\n");
}

/**
 * Suggest the best provider based on the benchmark results.
 * "Best balance" = highest quality-to-cost ratio among fast responders.
 */
function recommend(results: BenchmarkResult[]): void {
  const valid = results.filter(
    (r) => r.error === null && r.calledWeatherTool && r.calledInvestTool
  );

  if (valid.length === 0) {
    console.log("No providers completed the test successfully.");
    return;
  }

  // Score = quality / (cost_per_call * 10000 + 0.1) — rewards quality, penalises cost
  const scored = valid
    .map((r) => ({
      ...r,
      balanceScore: r.qualityScore / (r.estimatedCostUsd * 10_000 + 0.1),
    }))
    .sort((a, b) => b.balanceScore - a.balanceScore);

  const best = scored[0];
  const cheapest = [...valid].sort(
    (a, b) => a.estimatedCostUsd - b.estimatedCostUsd
  )[0];
  const fastest = [...valid].sort((a, b) => a.totalTime - b.totalTime)[0];

  console.log("═══ RECOMMENDATION ═══════════════════════════════════════\n");
  console.log(
    `Best balance of speed, cost, and quality: ${best.provider} / ${best.modelName}`
  );
  console.log(
    `  Quality ${best.qualityScore}/10 · ${best.totalTime.toFixed(1)}s · $${best.estimatedCostUsd.toFixed(4)}\n`
  );
  console.log(
    `Cheapest:  ${cheapest.provider} / ${cheapest.modelName} @ $${cheapest.estimatedCostUsd.toFixed(4)}`
  );
  console.log(
    `Fastest:   ${fastest.provider} / ${fastest.modelName} @ ${fastest.totalTime.toFixed(1)}s\n`
  );
  console.log("Use gpt-4o or claude-3-5-sonnet when answer quality is critical.");
  console.log(
    "Use gpt-4o-mini or gemini-1.5-flash when cost and latency matter more than depth.\n"
  );
}

// ──── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const providers = buildProviders();

  if (providers.length === 0) {
    console.error(
      "No provider API keys found. Set at least one of:\n" +
        "  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, GROQ_API_KEY"
    );
    process.exit(1);
  }

  console.log(`Running benchmark across ${providers.length} provider(s)…`);
  console.log(`Test prompt: "${TEST_PROMPT}"\n`);

  // Warm-up call to avoid cold-start bias (uses first available provider)
  console.log("Warming up…");
  try {
    await generateText({
      model: providers[0].model,
      prompt: "Say 'ready' in one word.",
      maxTokens: 5,
    });
  } catch {
    // Ignore warm-up failures
  }
  console.log("Warm-up complete. Starting benchmark…\n");

  // Run all benchmarks (sequentially to avoid rate limit collisions)
  const results: BenchmarkResult[] = [];
  for (const config of providers) {
    process.stdout.write(`  Testing ${config.provider} / ${config.modelName}… `);
    const result = await runBenchmark(config);
    results.push(result);
    console.log(result.error ? `ERROR: ${result.error.substring(0, 60)}` : `done (${result.totalTime.toFixed(1)}s)`);
  }

  renderTable(results);
  recommend(results);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
