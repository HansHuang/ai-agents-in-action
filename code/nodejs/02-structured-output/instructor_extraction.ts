/**
 * Instructor-style structured extraction using Zod schemas.
 *
 * Demonstrates extracting structured data from unstructured text using
 * OpenAI's json_schema response format with Zod for runtime validation.
 * See: docs/01-foundations/03-structured-output.md
 */

import OpenAI from "openai";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const MODEL = "gpt-4o";

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const PersonSchema = z.object({
  name: z.string(),
  age: z.number().int().optional(),
  occupation: z.string().optional(),
  location: z.string().optional(),
});

const EventSchema = z.object({
  title: z.string(),
  date: z.string().optional(),
  location: z.string().optional(),
  participants: z.array(PersonSchema).default([]),
  key_topics: z.array(z.string()).default([]),
});

type Person = z.infer<typeof PersonSchema>;
type Event = z.infer<typeof EventSchema>;

// ---------------------------------------------------------------------------
// Generic extractor
// ---------------------------------------------------------------------------

/**
 * Extract structured data from text using a Zod schema.
 */
async function extract<T>(
  schema: z.ZodType<T>,
  schemaName: string,
  text: string,
  client: OpenAI
): Promise<T> {
  const jsonSchema = zodToJsonSchema(schema) as Record<string, unknown>;
  const response = await client.chat.completions.create({
    model: MODEL,
    messages: [
      {
        role: "system",
        content:
          "You extract structured information from text. Return only JSON matching the schema.",
      },
      { role: "user", content: text },
    ],
    response_format: {
      type: "json_schema",
      json_schema: { name: schemaName, schema: jsonSchema, strict: true },
    },
    temperature: 0,
  });
  const content = response.choices[0].message.content;
  if (!content) throw new Error("Empty response from model");
  return schema.parse(JSON.parse(content));
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

const SAMPLE_TEXTS = [
  {
    label: "Tech conference article",
    text:
      "Last Tuesday, Sarah Chen (35, software engineer from San Francisco) " +
      "delivered a keynote at the AI Summit 2025 in New York. The talk covered " +
      "agent architectures and safety alignment. About 500 attendees were present, " +
      "including Dr. James Park from MIT.",
  },
  {
    label: "Meeting notes",
    text:
      "The product review on March 14th was attended by Alice Johnson (PM), " +
      "Bob Smith (Engineering), and Carol White (Design). Key topics: " +
      "roadmap Q3, customer feedback analysis, and launch timeline.",
  },
];

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  for (const { label, text } of SAMPLE_TEXTS) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`[${label}]`);
    console.log(`Text: ${text}\n`);

    const event: Event = await extract(EventSchema, "event", text, client);

    console.log("Extracted event:");
    console.log(`  Title       : ${event.title}`);
    console.log(`  Date        : ${event.date ?? "—"}`);
    console.log(`  Location    : ${event.location ?? "—"}`);
    console.log(`  Key topics  : ${event.key_topics.join(", ") || "—"}`);
    console.log(`  Participants (${event.participants.length}):`);
    for (const p of event.participants) {
      const parts = [p.name];
      if (p.age) parts.push(`age ${p.age}`);
      if (p.occupation) parts.push(p.occupation);
      if (p.location) parts.push(p.location);
      console.log(`    - ${parts.join(", ")}`);
    }
  }
}

main().catch(console.error);
