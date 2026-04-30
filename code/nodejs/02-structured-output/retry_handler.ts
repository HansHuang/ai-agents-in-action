/**
 * Reusable parse-validate-retry handler for structured LLM extraction.
 *
 * extractWithRetry() is a generic function that:
 * - Calls the OpenAI API with a JSON schema derived from a Zod schema
 * - Parses the response with Zod's safeParse
 * - On validation failure, appends a human-readable error to messages and retries
 * - Logs each attempt number, success/failure, and error details
 * - Returns { success: true, data } or { success: false, error } — never throws
 *
 * Import and reuse across any extraction task in this repo.
 *
 * See docs/01-foundations/03-structured-output.md — "The Parse-Validate-Retry Pattern"
 */

import OpenAI from "openai";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const MODEL = "gpt-4o";

type Message = { role: "system" | "user" | "assistant"; content: string };

export type ExtractSuccess<T> = { success: true; data: T };
export type ExtractFailure = { success: false; error: string };
export type ExtractResult<T> = ExtractSuccess<T> | ExtractFailure;

/**
 * Call the LLM and validate into schema, retrying on failure.
 *
 * @param messages   - Message array (a working copy is made; original is not mutated).
 * @param schema     - Zod schema defining the expected output shape.
 * @param maxRetries - Max retry attempts after the first call (default 3).
 * @param model      - OpenAI model to use.
 * @returns A discriminated union: { success: true, data } | { success: false, error }.
 */
export async function extractWithRetry<T>(
  messages: Message[],
  schema: z.ZodSchema<T>,
  maxRetries = 3,
  model = MODEL
): Promise<ExtractResult<T>> {
  const client = new OpenAI({ apiKey: process.env["OPENAI_API_KEY"] });

  // Build the JSON schema once.
  const zodResult = zodToJsonSchema(schema, {
    name: "OutputSchema",
    $refStrategy: "none",
  });
  const jsonSchema =
    (zodResult as { definitions?: Record<string, unknown> }).definitions?.[
      "OutputSchema"
    ] ?? zodResult;

  const workingMessages = [...messages];

  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    console.log(`[extractWithRetry] Attempt ${attempt}/${maxRetries + 1}`);

    const response = await client.chat.completions.create({
      model,
      messages: workingMessages,
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "output_schema",
          schema: jsonSchema as Record<string, unknown>,
          strict: true,
        },
      },
    });

    const raw = response.choices[0].message.content ?? "";

    // Step 1: JSON parse.
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      const errorMsg = `Response was not valid JSON: ${raw.slice(0, 200)}`;
      console.warn(`[extractWithRetry] Attempt ${attempt} failed — ${errorMsg}`);
      if (attempt > maxRetries) return { success: false, error: errorMsg };
      workingMessages.push({ role: "assistant", content: raw });
      workingMessages.push({
        role: "user",
        content: "Your response was not valid JSON. Please respond with valid JSON only.",
      });
      continue;
    }

    // Step 2: Zod validation.
    const result = schema.safeParse(parsed);
    if (result.success) {
      console.log(`[extractWithRetry] Attempt ${attempt} succeeded`);
      return { success: true, data: result.data };
    }

    const errorSummary = result.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join("; ");

    console.warn(`[extractWithRetry] Attempt ${attempt} failed — ${errorSummary}`);

    if (attempt > maxRetries) {
      return {
        success: false,
        error: `Max retries exceeded. Last errors: ${errorSummary}`,
      };
    }

    workingMessages.push({ role: "assistant", content: raw });
    workingMessages.push({
      role: "user",
      content: `Your response did not match the required schema. Errors: ${errorSummary}. Please fix and retry.`,
    });
  }

  return { success: false, error: "extractWithRetry: unreachable" };
}
