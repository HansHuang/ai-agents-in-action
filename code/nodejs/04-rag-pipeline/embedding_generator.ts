/**
 * Embedding generation and comparison utilities (TypeScript port).
 *
 * Demonstrates how to generate text embeddings using OpenAI's API, compare
 * vectors with cosine similarity, and find semantically similar texts.
 *
 * See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
 */

import OpenAI from "openai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SimilarityResult {
  text: string;
  score: number;
}

// ---------------------------------------------------------------------------
// EmbeddingGenerator
// ---------------------------------------------------------------------------

/**
 * Generate embeddings from text using configurable OpenAI models.
 *
 * @param model     - The OpenAI embedding model to use.
 * @param dimensions - Optional dimension reduction (supported by
 *                     text-embedding-3-* models only). `undefined` uses
 *                     the model default.
 */
export class EmbeddingGenerator {
  readonly model: string;
  readonly dimensions: number | undefined;
  private readonly client: OpenAI;

  constructor(
    model: string = "text-embedding-3-small",
    dimensions?: number,
  ) {
    this.model = model;
    this.dimensions = dimensions;
    this.client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }

  // -------------------------------------------------------------------------
  // Core embedding methods
  // -------------------------------------------------------------------------

  /**
   * Generate an embedding for a single text string.
   *
   * @param text - The text to embed.
   * @returns Array of floats representing the embedding vector.
   */
  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text,
      ...this.extraParams(),
    });
    return response.data[0].embedding;
  }

  /**
   * Generate embeddings for multiple texts in a single API call.
   *
   * Batching is significantly more efficient than calling `embed` in a loop.
   * The API accepts up to 2,048 inputs per request.
   *
   * @param texts - Array of texts to embed.
   * @returns Array of embedding vectors, in the same order as `texts`.
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const response = await this.client.embeddings.create({
      model: this.model,
      input: texts,
      ...this.extraParams(),
    });
    // Sort by index to guarantee order matches the input.
    const sorted = [...response.data].sort((a, b) => a.index - b.index);
    return sorted.map((d) => d.embedding);
  }

  /**
   * Embed a text with automatic exponential-backoff retry on transient failures.
   *
   * @param text        - The text to embed.
   * @param maxRetries  - Maximum number of retry attempts.
   * @param backoffBase - Base sleep time in milliseconds. Doubles each attempt.
   * @returns Array of floats representing the embedding vector.
   */
  async embedWithRetry(
    text: string,
    maxRetries: number = 3,
    backoffBase: number = 1000,
  ): Promise<number[]> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.embed(text);
      } catch (err) {
        lastError = err;
        if (attempt < maxRetries) {
          const sleepMs = backoffBase * Math.pow(2, attempt);
          console.warn(
            `Embedding attempt ${attempt + 1}/${maxRetries} failed: ${err}. Retrying in ${sleepMs}ms…`,
          );
          await sleep(sleepMs);
        }
      }
    }
    throw lastError;
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private extraParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    if (this.dimensions !== undefined) {
      params.dimensions = this.dimensions;
    }
    return params;
  }
}

// ---------------------------------------------------------------------------
// EmbeddingComparator
// ---------------------------------------------------------------------------

/** Compare embeddings and find semantically similar texts. */
export class EmbeddingComparator {
  // -------------------------------------------------------------------------
  // Distance / similarity metrics
  // -------------------------------------------------------------------------

  /**
   * Calculate cosine similarity between two embedding vectors.
   *
   * Returns a value in [-1, 1]. Values closer to 1 indicate semantically
   * similar texts.
   *
   * @param a - First embedding vector.
   * @param b - Second embedding vector.
   * @returns Cosine similarity score.
   */
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
    }
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }

  /**
   * Calculate Euclidean distance between two embedding vectors.
   *
   * Smaller values indicate more similar texts. For text embeddings, prefer
   * `cosineSimilarity` unless you have a reason to care about magnitude.
   *
   * @param a - First embedding vector.
   * @param b - Second embedding vector.
   * @returns Euclidean distance (>= 0).
   */
  euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
    }
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  // -------------------------------------------------------------------------
  // Higher-level utilities
  // -------------------------------------------------------------------------

  /**
   * Rank candidate texts by semantic similarity to a query.
   *
   * Embeds the query and all candidates in a single batch call.
   *
   * @param query      - The question or search string.
   * @param candidates - Texts to compare against the query.
   * @param generator  - `EmbeddingGenerator` to use.
   * @returns Array of `{ text, score }` objects sorted highest-similarity first.
   */
  async findMostSimilar(
    query: string,
    candidates: string[],
    generator: EmbeddingGenerator,
  ): Promise<SimilarityResult[]> {
    const allTexts = [query, ...candidates];
    const allEmbeddings = await generator.embedBatch(allTexts);
    const queryEmbedding = allEmbeddings[0];
    const candidateEmbeddings = allEmbeddings.slice(1);

    const results: SimilarityResult[] = candidates.map((text, i) => ({
      text,
      score: this.cosineSimilarity(queryEmbedding, candidateEmbeddings[i]),
    }));
    results.sort((a, b) => b.score - a.score);
    return results;
  }

  /**
   * Generate a similarity matrix formatted as ASCII art.
   *
   * @param texts     - List of texts to compare pairwise.
   * @param generator - `EmbeddingGenerator` used to embed the texts.
   * @returns Multi-line string ready to log.
   */
  async visualizeSimilarity(
    texts: string[],
    generator: EmbeddingGenerator,
  ): Promise<string> {
    const embeddings = await generator.embedBatch(texts);
    const n = texts.length;
    const labels = texts.map((_, i) => `Text${i + 1}`);
    const colW = 7;
    const labelW = Math.max(...labels.map((l) => l.length)) + 2;

    const header =
      " ".repeat(labelW) + labels.map((l) => l.padStart(colW)).join("");
    const rows = [header];
    for (let i = 0; i < n; i++) {
      let row = labels[i].padEnd(labelW);
      for (let j = 0; j < n; j++) {
        const score = this.cosineSimilarity(embeddings[i], embeddings[j]);
        row += score.toFixed(2).padStart(colW);
      }
      rows.push(row);
    }
    return rows.join("\n");
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const generator = new EmbeddingGenerator("text-embedding-3-small");
  const comparator = new EmbeddingComparator();

  const weatherTexts = [
    "It's raining heavily outside right now.",
    "The forecast calls for sunny skies all week.",
    "A blizzard is expected to hit the northeast tonight.",
  ];
  const stockTexts = [
    "The S&P 500 rose 1.2% on strong earnings reports.",
    "Investors are worried about rising interest rates.",
    "Tech stocks led the market rally this afternoon.",
    "The Federal Reserve signalled a rate pause today.",
  ];
  const sportsTexts = [
    "The home team scored three goals in the second half.",
    "The marathon world record was broken yesterday.",
    "Championship play-offs begin next weekend.",
  ];
  const allTexts = [...weatherTexts, ...stockTexts, ...sportsTexts];
  const categories = [
    ...Array(weatherTexts.length).fill("weather"),
    ...Array(stockTexts.length).fill("stocks"),
    ...Array(sportsTexts.length).fill("sports"),
  ];

  console.log("Generating embeddings for 10 texts across 3 categories…");
  const embeddings = await generator.embedBatch(allTexts);
  console.log(`  Each embedding has ${embeddings[0].length} dimensions.\n`);

  console.log("=== Within-category similarity (expect > 0.7) ===");
  for (let i = 0; i < allTexts.length; i++) {
    for (let j = i + 1; j < allTexts.length; j++) {
      if (categories[i] === categories[j]) {
        const sim = comparator.cosineSimilarity(embeddings[i], embeddings[j]);
        console.log(`  [${categories[i]}]  ${sim.toFixed(3)}`);
      }
    }
  }

  console.log("\n=== Cross-category similarity (expect < 0.3) ===");
  for (let i = 0; i < allTexts.length; i++) {
    for (let j = i + 1; j < allTexts.length; j++) {
      if (categories[i] !== categories[j]) {
        const sim = comparator.cosineSimilarity(embeddings[i], embeddings[j]);
        console.log(`  [${categories[i]} vs ${categories[j]}]  ${sim.toFixed(3)}`);
      }
    }
  }

  console.log("\n=== Similarity matrix (first 4 texts) ===");
  const matrix = await comparator.visualizeSimilarity(allTexts.slice(0, 4), generator);
  console.log(matrix);

  console.log("\n=== Query: 'What's the market doing today?' ===");
  const results = await comparator.findMostSimilar(
    "What's the market doing today?",
    allTexts,
    generator,
  );
  results.slice(0, 5).forEach((r, i) => {
    console.log(`  #${i + 1}  score=${r.score.toFixed(3)}  ${r.text.slice(0, 60)}`);
  });
}

main().catch(console.error);
