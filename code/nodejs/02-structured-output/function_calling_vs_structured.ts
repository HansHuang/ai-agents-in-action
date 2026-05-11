/**
 * Function Calling vs. Structured Output: side-by-side comparison.
 *
 * Runs the same sentiment-extraction task through both API paths on 5 test
 * texts and prints a summary comparing success rate, total tokens, and
 * average latency for each method.
 * See: docs/01-foundations/03-structured-output.md
 */

import OpenAI from "openai";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const MODEL = "gpt-4o";

// ---------------------------------------------------------------------------
// Shared schema
// ---------------------------------------------------------------------------

const SentimentSchema = z.object({
  sentiment: z.enum(["positive", "negative", "neutral"]),
  confidence: z.number().min(0).max(1),
  key_phrases: z.array(z.string()).default([]),
});
type Sentiment = z.infer<typeof SentimentSchema>;

// ---------------------------------------------------------------------------
// Function-calling definition (Path A)
// ---------------------------------------------------------------------------

const FUNCTION_TOOL: OpenAI.Chat.ChatCompletionTool = {
  type: "function",
  function: {
    name: "classify_sentiment",
    description: "Classify the sentiment of the provided text.",
    parameters: {
      type: "object",
      properties: {
        sentiment: {
          type: "string",
          enum: ["positive", "negative", "neutral"],
          description: "Overall sentiment.",
        },
        confidence: {
          type: "number",
          description: "Confidence score between 0 and 1.",
        },
        key_phrases: {
          type: "array",
          items: { type: "string" },
          description: "Key phrases that drove the classification.",
        },
      },
      required: ["sentiment", "confidence"],
    },
  },
};

// ---------------------------------------------------------------------------
// Path A: function calling
// ---------------------------------------------------------------------------

async function classifyWithFunctionCalling(
  text: string,
  client: OpenAI
): Promise<{ result: Sentiment | null; tokens: number; latencyMs: number }> {
  const t0 = Date.now();
  const response = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You classify sentiment." },
      { role: "user", content: text },
    ],
    tools: [FUNCTION_TOOL],
    tool_choice: { type: "function", function: { name: "classify_sentiment" } },
  });
  const latencyMs = Date.now() - t0;
  const tokens = response.usage?.total_tokens ?? 0;
  const toolCall = response.choices[0].message.tool_calls?.[0];
  if (!toolCall) return { result: null, tokens, latencyMs };
  try {
    const raw = JSON.parse(toolCall.function.arguments);
    const result = SentimentSchema.parse(raw);
    return { result, tokens, latencyMs };
  } catch {
    return { result: null, tokens, latencyMs };
  }
}

// ---------------------------------------------------------------------------
// Path B: structured output (json_schema)
// ---------------------------------------------------------------------------

async function classifyWithStructuredOutput(
  text: string,
  client: OpenAI
): Promise<{ result: Sentiment | null; tokens: number; latencyMs: number }> {
  const t0 = Date.now();
  const response = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You classify sentiment. Respond in JSON." },
      { role: "user", content: text },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "sentiment_response",
        schema: zodToJsonSchema(SentimentSchema) as Record<string, unknown>,
        strict: true,
      },
    },
  });
  const latencyMs = Date.now() - t0;
  const tokens = response.usage?.total_tokens ?? 0;
  const content = response.choices[0].message.content;
  if (!content) return { result: null, tokens, latencyMs };
  try {
    const raw = JSON.parse(content);
    const result = SentimentSchema.parse(raw);
    return { result, tokens, latencyMs };
  } catch {
    return { result: null, tokens, latencyMs };
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

const TEST_TEXTS = [
  "I absolutely love this product! It exceeded all my expectations.",
  "This is the worst purchase I have ever made.",
  "The package arrived on time. Nothing special.",
  "Not bad, but could be better. Some features are missing.",
  "Outstanding customer service and lightning-fast delivery!",
];

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  const fcStats = { success: 0, tokens: 0, latencyMs: 0 };
  const soStats = { success: 0, tokens: 0, latencyMs: 0 };

  console.log(`${"Text".padEnd(50)} ${"FC".padEnd(12)} ${"SO".padEnd(12)}`);
  console.log("-".repeat(76));

  for (const text of TEST_TEXTS) {
    const [fc, so] = await Promise.all([
      classifyWithFunctionCalling(text, client),
      classifyWithStructuredOutput(text, client),
    ]);
    if (fc.result) fcStats.success++;
    fcStats.tokens += fc.tokens;
    fcStats.latencyMs += fc.latencyMs;
    if (so.result) soStats.success++;
    soStats.tokens += so.tokens;
    soStats.latencyMs += so.latencyMs;

    const fcLabel = fc.result?.sentiment ?? "FAIL";
    const soLabel = so.result?.sentiment ?? "FAIL";
    console.log(
      `${text.slice(0, 48).padEnd(50)} ${fcLabel.padEnd(12)} ${soLabel}`
    );
  }

  const n = TEST_TEXTS.length;
  console.log("\nSummary:");
  console.log(`${"".padEnd(22)} ${"FunctionCalling".padEnd(18)} ${"StructuredOutput"}`);
  console.log(`${"Success rate".padEnd(22)} ${`${fcStats.success}/${n}`.padEnd(18)} ${soStats.success}/${n}`);
  console.log(
    `${"Total tokens".padEnd(22)} ${String(fcStats.tokens).padEnd(18)} ${soStats.tokens}`
  );
  console.log(
    `${"Avg latency (ms)".padEnd(22)} ${String(Math.round(fcStats.latencyMs / n)).padEnd(18)} ${Math.round(soStats.latencyMs / n)}`
  );
}

main().catch(console.error);
