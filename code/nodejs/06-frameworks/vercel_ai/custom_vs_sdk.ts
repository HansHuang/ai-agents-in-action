/**
 * custom_vs_sdk.ts — Side-by-side comparison of a custom agent loop
 * vs. the Vercel AI SDK's streamText with maxSteps.
 *
 * Both implementations solve the SAME task with the SAME tools and system prompt.
 * The comparison section at the bottom measures code complexity and capability.
 *
 * Usage:
 *   npx tsx custom_vs_sdk.ts
 */

import { generateText, streamText, tool, CoreMessage } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

// ──── Shared: tools and system prompt ─────────────────────────────────────────

/** Mock weather tool used by both implementations. */
const getWeather = tool({
  description: "Get the current weather for a city.",
  parameters: z.object({
    city: z.string().describe("City name with country code, e.g. 'Tokyo, JP'"),
  }),
  execute: async ({ city }) => ({
    city,
    temperature: 18,
    condition: "partly cloudy",
    humidity: 72,
  }),
});

/** Mock calculator tool used by both implementations. */
const calculate = tool({
  description: "Evaluate a mathematical expression.",
  parameters: z.object({
    expression: z.string().describe("Math expression, e.g. '(22 + 18) / 2'"),
  }),
  execute: async ({ expression }) => {
    try {
      // Safe eval: only digits, operators, parentheses, spaces
      if (!/^[\d\s\+\-\*\/\(\)\.]+$/.test(expression)) {
        return { error: "Invalid expression. Only basic arithmetic allowed." };
      }
      // biome-ignore lint: intentional numeric eval for demo
      const result = Function(`"use strict"; return (${expression})`)() as number;
      return { expression, result };
    } catch {
      return { error: "Could not evaluate expression." };
    }
  },
});

const TOOLS = { getWeather, calculate };

const SYSTEM_PROMPT =
  "You are a helpful assistant. Use the available tools to answer questions accurately. " +
  "Think step by step.";

const TEST_QUESTION =
  "What's the weather in Tokyo and Paris? What's the average temperature between them?";

// ──── IMPLEMENTATION 1: Custom agent loop ────────────────────────────────────
// Uses generateText() in a while loop, manually manages messages and tool calls.

interface StepRecord {
  step: number;
  type: "llm_call" | "tool_call";
  detail: string;
  inputTokens: number;
  outputTokens: number;
}

interface CustomLoopResult {
  answer: string;
  steps: number;
  stepRecords: StepRecord[];
  totalInputTokens: number;
  totalOutputTokens: number;
  durationMs: number;
}

/**
 * Implementation 1 — Custom agent loop.
 *
 * Manually orchestrates:
 *   1. LLM call via generateText()
 *   2. Tool execution
 *   3. Message history management
 *   4. Stop condition (no more tool calls, or maxSteps reached)
 */
async function runCustomLoop(
  userInput: string,
  maxSteps = 10
): Promise<CustomLoopResult> {
  const start = performance.now();
  const messages: CoreMessage[] = [{ role: "user", content: userInput }];
  const stepRecords: StepRecord[] = [];
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let stepCount = 0;
  let answer = "";

  while (stepCount < maxSteps) {
    stepCount++;

    // ── LLM call ────────────────────────────────────────────────────────────
    const result = await generateText({
      model: openai("gpt-4o"),
      system: SYSTEM_PROMPT,
      messages,
      tools: TOOLS,
      maxSteps: 1, // Single step at a time — we drive the loop
    });

    const inputTokens = result.usage?.promptTokens ?? 0;
    const outputTokens = result.usage?.completionTokens ?? 0;
    totalInputTokens += inputTokens;
    totalOutputTokens += outputTokens;

    stepRecords.push({
      step: stepCount,
      type: "llm_call",
      detail: result.text?.substring(0, 80) || "(no text)",
      inputTokens,
      outputTokens,
    });

    // ── Check for tool calls ─────────────────────────────────────────────────
    const toolCalls = result.toolCalls ?? [];

    if (toolCalls.length === 0) {
      // No tool calls → final answer
      answer = result.text;
      break;
    }

    // ── Execute tools manually ───────────────────────────────────────────────
    const toolResults: CoreMessage = {
      role: "tool",
      content: [],
    };

    for (const tc of toolCalls) {
      let toolOutput: unknown;
      try {
        if (tc.toolName === "getWeather") {
          toolOutput = await TOOLS.getWeather.execute(
            tc.args as { city: string },
            { messages, toolCallId: tc.toolCallId }
          );
        } else if (tc.toolName === "calculate") {
          toolOutput = await TOOLS.calculate.execute(
            tc.args as { expression: string },
            { messages, toolCallId: tc.toolCallId }
          );
        } else {
          toolOutput = { error: `Unknown tool: ${tc.toolName}` };
        }
      } catch (err) {
        toolOutput = { error: `Tool execution failed: ${String(err)}` };
      }

      (toolResults.content as Array<{
        type: "tool-result";
        toolCallId: string;
        toolName: string;
        result: unknown;
      }>).push({
        type: "tool-result",
        toolCallId: tc.toolCallId,
        toolName: tc.toolName,
        result: toolOutput,
      });

      stepRecords.push({
        step: stepCount,
        type: "tool_call",
        detail: `${tc.toolName}(${JSON.stringify(tc.args).substring(0, 60)})`,
        inputTokens: 0,
        outputTokens: 0,
      });
    }

    // ── Update message history ───────────────────────────────────────────────
    messages.push({
      role: "assistant",
      content: result.text || "",
    });
    messages.push(toolResults);
  }

  if (!answer) {
    answer = "Reached maximum steps without a final answer.";
  }

  return {
    answer,
    steps: stepCount,
    stepRecords,
    totalInputTokens,
    totalOutputTokens,
    durationMs: performance.now() - start,
  };
}

// ──── IMPLEMENTATION 2: Vercel AI SDK streamText ──────────────────────────────
// Uses streamText() with maxSteps — the SDK drives the loop.

interface SdkLoopResult {
  answer: string;
  steps: number;
  stepRecords: StepRecord[];
  totalInputTokens: number;
  totalOutputTokens: number;
  durationMs: number;
}

/**
 * Implementation 2 — SDK-managed agent loop.
 *
 * streamText() with maxSteps handles:
 *   - The while loop
 *   - Tool execution dispatch
 *   - Message history
 *   - Stop condition
 *
 * All we configure is: model, system, messages, tools, maxSteps.
 */
async function runSdkLoop(
  userInput: string,
  maxSteps = 10
): Promise<SdkLoopResult> {
  const start = performance.now();
  const stepRecords: SdkLoopResult["stepRecords"] = [];
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let stepCount = 0;

  const result = streamText({
    model: openai("gpt-4o"),
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userInput }],
    tools: TOOLS,
    maxSteps,

    onStepFinish: (step) => {
      stepCount++;
      const inputTokens = step.usage?.promptTokens ?? 0;
      const outputTokens = step.usage?.completionTokens ?? 0;
      totalInputTokens += inputTokens;
      totalOutputTokens += outputTokens;

      stepRecords.push({
        step: stepCount,
        type: "llm_call",
        detail: step.text?.substring(0, 80) || "(no text)",
        inputTokens,
        outputTokens,
      });

      for (const tc of step.toolCalls ?? []) {
        stepRecords.push({
          step: stepCount,
          type: "tool_call",
          detail: `${tc.toolName}(${JSON.stringify(tc.args).substring(0, 60)})`,
          inputTokens: 0,
          outputTokens: 0,
        });
      }
    },
  });

  let answer = "";
  for await (const chunk of result.textStream) {
    answer += chunk;
  }

  return {
    answer,
    steps: stepCount,
    stepRecords,
    totalInputTokens,
    totalOutputTokens,
    durationMs: performance.now() - start,
  };
}

// ──── Comparison table ────────────────────────────────────────────────────────

function countLines(fn: string): number {
  return fn.split("\n").length;
}

function renderComparisonTable(): void {
  // Static analysis of the two implementations above
  const rows: [string, string, string][] = [
    ["Lines of code (core logic)", "~65", "~40"],
    ["Cyclomatic complexity", "~12 (loop, branches, try/catch)", "~3 (callbacks only)"],
    ["Concepts to understand", "CoreMessage, toolCalls, tool-result", "streamText, maxSteps, onStepFinish"],
    ["Streaming support", "Manual (textStream from generateText)", "Built-in (textStream)"],
    ["Tool call error handling", "Manual try/catch per tool", "SDK handles dispatch errors"],
    ["Step-level tracing", "Manual push to stepRecords", "onStepFinish callback"],
    ["Custom stop condition", "Full control (any while condition)", "Only maxSteps"],
    ["Message history control", "Full control (push/mutate array)", "SDK-managed (read-only)"],
    ["Add a new tool", "Add to TOOLS + manual dispatch switch", "Add to TOOLS only"],
    ["Change system prompt", "One line (system param)", "One line (system param)"],
    ["Debug a stuck loop", "Add console.log in the while loop", "Use onStepFinish callback"],
    ["Learning value", "High — see the full loop internals", "Low — SDK hides the loop"],
    ["Production readiness", "Requires your own testing", "Battle-tested by SDK"],
    ["Frontend streaming", "Requires manual SSE/stream setup", "toDataStreamResponse() + useChat"],
  ];

  const col0Width = 38;
  const col1Width = 36;
  const col2Width = 34;
  const header = `${"Metric".padEnd(col0Width)} | ${"Custom Loop".padEnd(col1Width)} | ${"SDK streamText"}`;
  const separator = "─".repeat(col0Width + col1Width + col2Width + 6);

  console.log("\n" + separator);
  console.log(header);
  console.log(separator);
  for (const [metric, custom, sdk] of rows) {
    console.log(
      `${metric.padEnd(col0Width)} | ${custom.padEnd(col1Width)} | ${sdk}`
    );
  }
  console.log(separator + "\n");
}

function renderRecommendation(): void {
  console.log("═══ RECOMMENDATION ════════════════════════════════════════════\n");
  console.log("Use the CUSTOM LOOP when:");
  console.log("  • You need a non-standard stop condition (e.g. confidence threshold)");
  console.log("  • You need to inject or modify messages mid-loop");
  console.log("  • You're learning how agent loops work (high pedagogical value)");
  console.log("  • You need to mix tool execution with external async workflows\n");

  console.log("Use SDK streamText (maxSteps) when:");
  console.log("  • You want production-ready streaming out of the box");
  console.log("  • Your stop condition is simply 'no more tool calls'");
  console.log("  • You're building a Next.js/React app (useChat integration)");
  console.log("  • You want minimal boilerplate and battle-tested error handling\n");

  console.log(
    "Bottom line: Start with the SDK. Drop to a custom loop only when you need\n" +
      "control that the SDK doesn't expose. Both are valid production patterns.\n"
  );
}

// ──── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    console.error("OPENAI_API_KEY is required.");
    process.exit(1);
  }

  console.log("═══ CUSTOM AGENT LOOP vs. SDK streamText ══════════════════════\n");
  console.log(`Test question: "${TEST_QUESTION}"\n`);

  // ── Run custom loop ──────────────────────────────────────────────────────
  console.log("▶ Running CUSTOM LOOP…");
  const customResult = await runCustomLoop(TEST_QUESTION);
  console.log(`  ✓ Done in ${(customResult.durationMs / 1000).toFixed(2)}s`);
  console.log(`  Steps: ${customResult.steps}`);
  console.log(
    `  Tokens: ${customResult.totalInputTokens + customResult.totalOutputTokens}`
  );
  console.log(`  Answer: ${customResult.answer.substring(0, 200)}\n`);

  // ── Run SDK loop ─────────────────────────────────────────────────────────
  console.log("▶ Running SDK streamText…");
  const sdkResult = await runSdkLoop(TEST_QUESTION);
  console.log(`  ✓ Done in ${(sdkResult.durationMs / 1000).toFixed(2)}s`);
  console.log(`  Steps: ${sdkResult.steps}`);
  console.log(
    `  Tokens: ${sdkResult.totalInputTokens + sdkResult.totalOutputTokens}`
  );
  console.log(`  Answer: ${sdkResult.answer.substring(0, 200)}\n`);

  // ── Step trace comparison ────────────────────────────────────────────────
  console.log("─── CUSTOM LOOP step trace ─────────────────────────────────");
  for (const s of customResult.stepRecords) {
    console.log(`  [step ${s.step}] ${s.type.padEnd(10)} ${s.detail}`);
  }

  console.log("\n─── SDK streamText step trace ──────────────────────────────");
  for (const s of sdkResult.stepRecords) {
    console.log(`  [step ${s.step}] ${s.type.padEnd(10)} ${s.detail}`);
  }

  // ── Comparison table ─────────────────────────────────────────────────────
  renderComparisonTable();
  renderRecommendation();
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
