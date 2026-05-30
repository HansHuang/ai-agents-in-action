/**
 * Zero-shot vs few-shot classification comparison.
 *
 * Demonstrates reliability gain from few-shot examples for sentiment
 * classification. Prints results side-by-side with token counts.
 * See: docs/01-foundations/02-prompt-engineering.md
 */

import OpenAI from "openai";
import { encoding_for_model } from "js-tiktoken";

const MODEL = "gpt-4o";
const ALLOWED_LABELS = new Set(["Positive", "Negative", "Neutral"]);

const ZERO_SHOT_SYSTEM =
  "Classify the sentiment of the following text as exactly one of: " +
  "Positive, Negative, or Neutral.";

const FEW_SHOT_SYSTEM = `Classify the sentiment of the following text as exactly one of: Positive, Negative, or Neutral.
Respond with exactly one word.

Examples:
Text: "I love this product!" → Positive
Text: "This is absolutely terrible." → Negative
Text: "It arrived on time." → Neutral
`;

const TEST_INPUT =
  "The new update is fine, I guess. Not bad, but nothing to get excited about.";

function countTokens(messages: OpenAI.Chat.ChatCompletionMessageParam[]): number {
  const enc = encoding_for_model("gpt-4o");
  let total = 3; // reply primer
  for (const msg of messages) {
    total += 3;
    for (const val of Object.values(msg)) {
      if (typeof val === "string") total += enc.encode(val).length;
    }
  }
  enc.free();
  return total;
}

async function classify(
  systemPrompt: string,
  text: string,
  client: OpenAI
): Promise<{ label: string; estimatedPromptTokens: number; actualPromptTokens: number }> {
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `Text: "${text}"` },
  ];
  const estimatedPromptTokens = countTokens(messages);
  const response = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0,
    max_tokens: 10,
  });
  return {
    label: response.choices[0].message.content?.trim() ?? "",
    estimatedPromptTokens,
    actualPromptTokens: response.usage?.prompt_tokens ?? 0,
  };
}

function formatStatus(label: string): string {
  return ALLOWED_LABELS.has(label) ? "ok" : "drift";
}

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  console.log(`Input: "${TEST_INPUT}"\n`);
  console.log(
    `${"Approach".padEnd(12)} ${"Result".padEnd(12)} ${"Format".padEnd(8)} ${"Est. prompt".padStart(12)} ${"API prompt".padStart(12)}`
  );
  console.log("-".repeat(64));

  const zero = await classify(ZERO_SHOT_SYSTEM, TEST_INPUT, client);
  console.log(
    `${"Zero-shot".padEnd(12)} ${zero.label.padEnd(12)} ${formatStatus(zero.label).padEnd(8)} ${String(zero.estimatedPromptTokens).padStart(12)} ${String(zero.actualPromptTokens).padStart(12)}`
  );

  const few = await classify(FEW_SHOT_SYSTEM, TEST_INPUT, client);
  console.log(
    `${"Few-shot".padEnd(12)} ${few.label.padEnd(12)} ${formatStatus(few.label).padEnd(8)} ${String(few.estimatedPromptTokens).padStart(12)} ${String(few.actualPromptTokens).padStart(12)}`
  );

  const overhead = few.estimatedPromptTokens - zero.estimatedPromptTokens;
  console.log(`\nFew-shot overhead: +${overhead} estimated prompt tokens per request`);
  console.log("A drift status means the model ignored the exact one-word label format.");
}

main().catch(console.error);
