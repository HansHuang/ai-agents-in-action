/**
 * Structured output extraction using Zod schemas and the OpenAI API.
 *
 * Demonstrates:
 * - Defining a Zod schema with .describe() on every field
 * - Converting Zod → JSON Schema via zod-to-json-schema for the OpenAI API
 * - extractSentiment(): strict json_schema response_format + safeParse retry loop
 *
 * Run:  npx tsx zod_extraction.ts
 *
 * See docs/01-foundations/03-structured-output.md — "Language-Specific Patterns"
 */

import OpenAI from "openai";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SentimentSchema = z.object({
  sentiment: z
    .enum(["positive", "negative", "neutral"])
    .describe(
      "Overall sentiment: positive (favorable/happy), " +
        "negative (unfavorable/unhappy), or neutral (neither)."
    ),
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe("Confidence in the classification, 0.0 (none) to 1.0 (certain)."),
  keyPhrases: z
    .array(z.string())
    .max(5)
    .optional()
    .describe("Up to 5 key phrases that most influenced the classification."),
});

type Sentiment = z.infer<typeof SentimentSchema>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL = "gpt-4o";
const MAX_RETRIES = 2;

type Message = { role: "system" | "user" | "assistant"; content: string };

// Convert the Zod schema once at module load.
// zod-to-json-schema wraps definitions under a "$defs" key; we need the root schema.
const _zodResult = zodToJsonSchema(SentimentSchema, {
  name: "SentimentSchema",
  $refStrategy: "none",
});
// When name is provided the actual schema is in .definitions[name].
const _jsonSchema =
  (
    _zodResult as {
      definitions?: Record<string, unknown>;
    }
  ).definitions?.["SentimentSchema"] ?? _zodResult;

// ---------------------------------------------------------------------------
// extractSentiment
// ---------------------------------------------------------------------------

/**
 * Extract sentiment from text, retrying up to MAX_RETRIES times on failure.
 *
 * @param text - The text to classify. Must be non-empty.
 * @returns A validated Sentiment object.
 * @throws Error if text is empty or max retries are exhausted.
 */
export async function extractSentiment(text: string): Promise<Sentiment> {
  if (!text.trim()) throw new Error("text must not be empty");

  const client = new OpenAI({ apiKey: process.env["OPENAI_API_KEY"] });
  const messages: Message[] = [
    {
      role: "system",
      content:
        "You are a precise sentiment analysis engine. " +
        "Classify the sentiment of the user's text accurately.",
    },
    { role: "user", content: text },
  ];

  for (let attempt = 1; attempt <= MAX_RETRIES + 1; attempt++) {
    const response = await client.chat.completions.create({
      model: MODEL,
      messages,
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "sentiment_response",
          schema: _jsonSchema as Record<string, unknown>,
          strict: true,
        },
      },
    });

    const raw = response.choices[0].message.content ?? "";

    // Step 1: parse JSON.
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      const msg = `Response was not valid JSON: ${raw.slice(0, 200)}`;
      console.warn(`  Attempt ${attempt} failed — ${msg}`);
      if (attempt > MAX_RETRIES) throw new Error(`Max retries exceeded. ${msg}`);
      messages.push({ role: "assistant", content: raw });
      messages.push({
        role: "user",
        content: "Your response was not valid JSON. Please respond with valid JSON.",
      });
      continue;
    }

    // Step 2: validate against the Zod schema.
    const result = SentimentSchema.safeParse(parsed);
    if (result.success) {
      console.log(`  Attempt ${attempt}: success`);
      return result.data;
    }

    const errorSummary = result.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join("; ");

    console.warn(`  Attempt ${attempt} failed — ${errorSummary}`);

    if (attempt > MAX_RETRIES) {
      throw new Error(`Max retries exceeded. Last errors: ${errorSummary}`);
    }

    messages.push({ role: "assistant", content: raw });
    messages.push({
      role: "user",
      content: `Your response did not match the required schema. Errors: ${errorSummary}. Please fix and retry.`,
    });
  }

  throw new Error("extractSentiment: unreachable");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const tests = [
    "I absolutely love this, it changed my life!",
    "It's fine I guess, nothing special.",
    "Terrible product, broke after one day.",
  ];

  for (const text of tests) {
    console.log(`Text: ${JSON.stringify(text)}`);
    const result = await extractSentiment(text);
    console.log(`  Sentiment  : ${result.sentiment}`);
    console.log(`  Confidence : ${result.confidence.toFixed(2)}`);
    console.log(`  Key Phrases: ${JSON.stringify(result.keyPhrases ?? null)}`);
    console.log();
  }
}

main().catch(console.error);
