/**
 * Advanced retrieval techniques for RAG pipelines.
 *
 * Implements HyDE, multi-query, and contextual retrieval strategies.
 * See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
 */

import OpenAI from "openai";
import { VectorDocument } from "./vector_database.js";

const MODEL = "gpt-4o-mini";

export interface RetrievalConfig {
  topK?: number;
  model?: string;
}

export type SearchFn = (query: string, topK: number) => Promise<VectorDocument[]>;

// ---------------------------------------------------------------------------
// HyDE: Hypothetical Document Embeddings
// ---------------------------------------------------------------------------

/**
 * Generate a hypothetical answer, then search with it.
 * Answers are more similar to documents than short questions.
 */
export async function hydeRetrieve(
  query: string,
  searchFn: SearchFn,
  client: OpenAI,
  config: RetrievalConfig = {}
): Promise<VectorDocument[]> {
  const model = config.model ?? MODEL;
  const topK = config.topK ?? 5;

  const hypothetical = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "system",
        content:
          "Write a short passage (2-4 sentences) that would answer the question. " +
          "Write as if it's from a reference document.",
      },
      { role: "user", content: query },
    ],
    temperature: 0,
    max_tokens: 200,
  });

  const hypotheticalDoc = hypothetical.choices[0].message.content?.trim() ?? query;
  return searchFn(hypotheticalDoc, topK);
}

// ---------------------------------------------------------------------------
// Multi-query retrieval
// ---------------------------------------------------------------------------

/**
 * Rephrase the question several ways, retrieve for each, and merge results.
 */
export async function multiQueryRetrieve(
  query: string,
  searchFn: SearchFn,
  client: OpenAI,
  config: RetrievalConfig & { numQueries?: number } = {}
): Promise<VectorDocument[]> {
  const model = config.model ?? MODEL;
  const topK = config.topK ?? 5;
  const numQueries = config.numQueries ?? 3;

  const resp = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "system",
        content: `Generate ${numQueries} different phrasings of the question. Return as JSON array of strings.`,
      },
      { role: "user", content: query },
    ],
    response_format: { type: "json_object" },
    temperature: 0.5,
  });

  let queries: string[] = [query];
  try {
    const parsed = JSON.parse(resp.choices[0].message.content ?? "{}") as Record<string, unknown>;
    const arr = Object.values(parsed)[0];
    if (Array.isArray(arr)) queries = [query, ...(arr as string[]).slice(0, numQueries - 1)];
  } catch {
    // fall back to original query only
  }

  const seenIds = new Set<string>();
  const merged: VectorDocument[] = [];

  await Promise.all(
    queries.map(async (q) => {
      const results = await searchFn(q, topK);
      for (const doc of results) {
        if (!seenIds.has(doc.id)) {
          seenIds.add(doc.id);
          merged.push(doc);
        }
      }
    })
  );

  return merged.slice(0, topK * 2);
}

// ---------------------------------------------------------------------------
// Contextual retrieval
// ---------------------------------------------------------------------------

/**
 * Enrich the query with conversation context before searching.
 */
export async function contextualRetrieve(
  query: string,
  conversationHistory: OpenAI.Chat.ChatCompletionMessageParam[],
  searchFn: SearchFn,
  client: OpenAI,
  config: RetrievalConfig = {}
): Promise<VectorDocument[]> {
  const model = config.model ?? MODEL;
  const topK = config.topK ?? 5;

  if (conversationHistory.length === 0) {
    return searchFn(query, topK);
  }

  const historyStr = conversationHistory
    .slice(-4)
    .map((m) => `${m.role}: ${typeof m.content === "string" ? m.content : ""}`)
    .join("\n");

  const resp = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "system",
        content: "Rewrite the question to be self-contained, incorporating relevant context from the conversation history. Return only the rewritten question.",
      },
      { role: "user", content: `History:\n${historyStr}\n\nQuestion: ${query}` },
    ],
    temperature: 0,
    max_tokens: 100,
  });

  const enrichedQuery = resp.choices[0].message.content?.trim() ?? query;
  return searchFn(enrichedQuery, topK);
}
