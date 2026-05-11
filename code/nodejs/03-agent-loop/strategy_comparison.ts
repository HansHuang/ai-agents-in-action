/**
 * Planning strategy comparison: ReAct vs Plan-and-Execute vs Reflection.
 *
 * Runs the same task through three agent strategies and compares token usage,
 * latency, and output quality.
 * See: docs/02-the-agent-loop/03-planning-strategies.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o-mini";

export type StrategyName = "direct" | "step_by_step" | "role_based";

interface StrategyResult {
  strategy: StrategyName;
  response: string;
  tokens: number;
  latencyMs: number;
}

const STRATEGIES: Record<StrategyName, { systemPrompt: string; userTemplate: (q: string) => string }> = {
  direct: {
    systemPrompt: "You are a helpful assistant. Answer directly and concisely.",
    userTemplate: (q) => q,
  },
  step_by_step: {
    systemPrompt: "You are a helpful assistant. Always think step by step before giving your final answer.",
    userTemplate: (q) => `${q}\n\nPlease think step by step.`,
  },
  role_based: {
    systemPrompt:
      "You are an expert analyst with deep knowledge across domains. " +
      "Provide structured, evidence-based responses.",
    userTemplate: (q) => q,
  },
};

async function runStrategy(
  strategy: StrategyName,
  question: string,
  client: OpenAI
): Promise<StrategyResult> {
  const { systemPrompt, userTemplate } = STRATEGIES[strategy];
  const t0 = Date.now();
  const response = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userTemplate(question) },
    ],
    temperature: 0,
  });
  return {
    strategy,
    response: response.choices[0].message.content?.trim() ?? "",
    tokens: response.usage?.total_tokens ?? 0,
    latencyMs: Date.now() - t0,
  };
}

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const question =
    "What are the key considerations when choosing a database for a high-traffic web application?";

  console.log(`Question: ${question}\n`);
  const strategies: StrategyName[] = ["direct", "step_by_step", "role_based"];
  const results = await Promise.all(
    strategies.map((s) => runStrategy(s, question, client))
  );

  console.log(`${"Strategy".padEnd(16)} ${"Tokens".padStart(8)} ${"Latency(ms)".padStart(12)}`);
  console.log("-".repeat(40));
  for (const r of results) {
    console.log(`${r.strategy.padEnd(16)} ${String(r.tokens).padStart(8)} ${String(r.latencyMs).padStart(12)}`);
  }

  for (const r of results) {
    console.log(`\n--- ${r.strategy} ---`);
    console.log(r.response.slice(0, 300) + (r.response.length > 300 ? "..." : ""));
  }
}

main().catch(console.error);
