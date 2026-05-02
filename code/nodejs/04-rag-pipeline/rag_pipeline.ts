/**
 * Complete Retrieval-Augmented Generation (RAG) pipeline (TypeScript port).
 *
 * Four-phase pipeline:
 *   1. INGEST  — Load → Chunk → Embed → Store
 *   2. RETRIEVE — Embed query → Search → Filter by threshold
 *   3. AUGMENT  — Build prompt with retrieved context
 *   4. GENERATE — Call LLM with augmented prompt
 *
 * See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
 */

import OpenAI from "openai";
import { readFileSync, readdirSync, statSync } from "fs";
import { join, extname, basename } from "path";
import { EmbeddingGenerator } from "./embedding_generator.js";
import { SimpleVectorStore, SearchResult } from "./simple_vector_store.js";

// ---------------------------------------------------------------------------
// Prompt template
// ---------------------------------------------------------------------------

const RAG_SYSTEM_PROMPT = `\
You are a helpful assistant that answers questions based on the provided documents.

Rules:
1. Answer ONLY using information from the documents below.
2. If the documents don't contain the answer, say exactly: \
"I don't have information about that in my knowledge base."
3. Cite sources using [Source: filename] format.
4. If multiple documents are relevant, synthesize information from all of them.
5. If documents contain conflicting information, note the conflict and cite both sources.
6. Do not use any knowledge outside the provided documents.

Documents:
{document_context}

When answering, structure your response as:
1. Direct answer to the question
2. Supporting details from the documents
3. Source citations
`;

const CITATION_SUFFIX =
  "\n\nIMPORTANT: You MUST cite every factual claim with [Source: <filename>]. " +
  "Do not make any statement without an explicit citation.";

const SUPPORTED_EXTENSIONS = new Set([".txt", ".md", ".rst", ".text"]);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RAGResponse {
  answer: string;
  sources: string[];
  retrievedChunks: Array<{ text: string; score: number; metadata: Record<string, unknown> }>;
  similarityScores: number[];
  tokensUsed: number;
  pipelineSteps: string[];
}

// ---------------------------------------------------------------------------
// Simple chunker (fixed-size with overlap)
// ---------------------------------------------------------------------------

function chunkText(
  text: string,
  chunkSize: number = 256,
  overlap: number = 50,
): string[] {
  // Split on word boundaries by approximate character count
  // (A token ≈ 4 chars; multiply chunkSize × 4 for character-based splitting)
  const charSize = chunkSize * 4;
  const charStep = Math.max(1, (chunkSize - overlap) * 4);
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + charSize));
    start += charStep;
  }
  return chunks.filter((c) => c.trim().length > 0);
}

// ---------------------------------------------------------------------------
// RAGPipeline
// ---------------------------------------------------------------------------

/**
 * Complete Retrieval-Augmented Generation pipeline.
 *
 * @param vectorStore  - Pre-created {@link SimpleVectorStore}.
 * @param embedder     - Pre-created {@link EmbeddingGenerator}.
 * @param model        - OpenAI chat model name.
 * @param chunkSize    - Target chunk size in tokens.
 * @param overlap      - Token overlap between adjacent chunks.
 * @param retrievalK   - Default number of results to return.
 * @param similarityThreshold - Default minimum cosine-similarity score.
 */
export class RAGPipeline {
  readonly vectorStore: SimpleVectorStore;
  readonly embedder: EmbeddingGenerator;
  readonly model: string;
  readonly chunkSize: number;
  readonly overlap: number;
  readonly retrievalK: number;
  readonly similarityThreshold: number;

  private readonly client: OpenAI;

  constructor(
    vectorStore: SimpleVectorStore,
    embedder: EmbeddingGenerator,
    model: string = "gpt-4o",
    chunkSize: number = 256,
    overlap: number = 50,
    retrievalK: number = 5,
    similarityThreshold: number = 0.7,
  ) {
    this.vectorStore = vectorStore;
    this.embedder = embedder;
    this.model = model;
    this.chunkSize = chunkSize;
    this.overlap = overlap;
    this.retrievalK = retrievalK;
    this.similarityThreshold = similarityThreshold;
    this.client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }

  // -------------------------------------------------------------------------
  // Phase 1 — Ingest
  // -------------------------------------------------------------------------

  /**
   * Load, chunk, embed, and store all documents in a directory.
   *
   * @param directory - Path to a directory containing text files.
   * @returns `{ documentsProcessed, chunksCreated, errors }`
   */
  async ingestDirectory(
    directory: string,
  ): Promise<{ documentsProcessed: number; chunksCreated: number; errors: string[] }> {
    let documentsProcessed = 0;
    let chunksCreated = 0;
    const errors: string[] = [];

    const entries = readdirSync(directory);
    for (const entry of entries.sort()) {
      const filePath = join(directory, entry);
      const stat = statSync(filePath);
      if (!stat.isFile()) continue;
      if (!SUPPORTED_EXTENSIONS.has(extname(entry).toLowerCase())) continue;
      try {
        const text = readFileSync(filePath, "utf-8");
        const n = await this.ingestText(text, { source: entry, path: filePath });
        chunksCreated += n;
        documentsProcessed += 1;
      } catch (err) {
        errors.push(`${entry}: ${err}`);
      }
    }
    return { documentsProcessed, chunksCreated, errors };
  }

  /**
   * Ingest a single text document.
   *
   * @param text     - Raw document text.
   * @param metadata - Optional metadata (e.g. `{ source: "faq.md" }`).
   * @returns Number of chunks created and stored.
   */
  async ingestText(
    text: string,
    metadata: Record<string, unknown> = {},
  ): Promise<number> {
    const chunks = chunkText(text, this.chunkSize, this.overlap);
    if (chunks.length === 0) return 0;

    const embeddings = await this.embedder.embedBatch(chunks);
    for (let i = 0; i < chunks.length; i++) {
      this.vectorStore.add(chunks[i], embeddings[i], {
        ...metadata,
        chunkIndex: i,
        totalChunks: chunks.length,
      });
    }
    return chunks.length;
  }

  // -------------------------------------------------------------------------
  // Phase 2 — Retrieve
  // -------------------------------------------------------------------------

  private async retrieve(
    question: string,
    k: number,
    threshold: number,
    steps: string[],
  ): Promise<SearchResult[]> {
    steps.push("RETRIEVE: embedding query");
    const queryEmbedding = await this.embedder.embed(question);
    steps.push(`RETRIEVE: searching vector store (k=${k}, threshold=${threshold})`);
    const results = this.vectorStore.searchWithThreshold(queryEmbedding, threshold, k);
    steps.push(`RETRIEVE: found ${results.length} chunk(s) above threshold`);
    return results;
  }

  // -------------------------------------------------------------------------
  // Phase 3 — Augment
  // -------------------------------------------------------------------------

  private buildMessages(
    question: string,
    retrieved: SearchResult[],
    extraSuffix: string = "",
  ): Array<{ role: string; content: string }> {
    const contextParts = retrieved.map((doc, i) => {
      const source = (doc.metadata.source as string | undefined) ?? "unknown";
      return `[Document ${i + 1} — Source: ${source}]\n${doc.text}`;
    });
    const documentContext = contextParts.join("\n\n---\n\n");
    const systemPrompt =
      RAG_SYSTEM_PROMPT.replace("{document_context}", documentContext) + extraSuffix;
    return [
      { role: "system", content: systemPrompt },
      { role: "user", content: question },
    ];
  }

  // -------------------------------------------------------------------------
  // Phase 4 — Generate
  // -------------------------------------------------------------------------

  private async generate(
    messages: Array<{ role: string; content: string }>,
    steps: string[],
  ): Promise<[string, number]> {
    steps.push(`GENERATE: calling ${this.model}`);
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: messages as OpenAI.ChatCompletionMessageParam[],
      temperature: 0.3,
    });
    const answer = response.choices[0].message.content ?? "";
    const tokens = response.usage?.total_tokens ?? 0;
    steps.push(`GENERATE: received ${tokens} total tokens`);
    return [answer, tokens];
  }

  // -------------------------------------------------------------------------
  // Public query interface
  // -------------------------------------------------------------------------

  /**
   * Answer a question using the RAG pipeline.
   *
   * @param question  - Natural-language question.
   * @param k         - Override the default number of retrieved chunks.
   * @param threshold - Override the default similarity threshold.
   * @returns {@link RAGResponse}
   */
  async query(
    question: string,
    k?: number,
    threshold?: number,
  ): Promise<RAGResponse> {
    const resolvedK = k ?? this.retrievalK;
    const resolvedThreshold = threshold ?? this.similarityThreshold;
    const steps: string[] = ["INGEST: (already complete)"];

    const results = await this.retrieve(question, resolvedK, resolvedThreshold, steps);

    if (results.length === 0) {
      steps.push("AUGMENT: no results above threshold — short-circuit");
      return {
        answer: "I don't have information about that in my knowledge base.",
        sources: [],
        retrievedChunks: [],
        similarityScores: [],
        tokensUsed: 0,
        pipelineSteps: steps,
      };
    }

    steps.push(`AUGMENT: building prompt with ${results.length} document(s)`);
    const messages = this.buildMessages(question, results);
    const [answer, tokens] = await this.generate(messages, steps);

    const sources = [...new Set(results.map((r) => (r.metadata.source as string) ?? "unknown"))];
    return {
      answer,
      sources,
      retrievedChunks: results.map((r) => ({ text: r.text, score: r.score, metadata: r.metadata })),
      similarityScores: results.map((r) => r.score),
      tokensUsed: tokens,
      pipelineSteps: steps,
    };
  }

  /**
   * Same as {@link query} but enforces citation format in the answer.
   */
  async queryWithCitations(
    question: string,
    k?: number,
    threshold?: number,
  ): Promise<RAGResponse> {
    const resolvedK = k ?? this.retrievalK;
    const resolvedThreshold = threshold ?? this.similarityThreshold;
    const steps: string[] = ["INGEST: (already complete)"];

    const results = await this.retrieve(question, resolvedK, resolvedThreshold, steps);

    if (results.length === 0) {
      steps.push("AUGMENT: no results above threshold — short-circuit");
      return {
        answer: "I don't have information about that in my knowledge base.",
        sources: [],
        retrievedChunks: [],
        similarityScores: [],
        tokensUsed: 0,
        pipelineSteps: steps,
      };
    }

    steps.push(`AUGMENT: building citation-enforced prompt with ${results.length} document(s)`);
    const messages = this.buildMessages(question, results, CITATION_SUFFIX);
    const [answer, tokens] = await this.generate(messages, steps);

    const sources = [...new Set(results.map((r) => (r.metadata.source as string) ?? "unknown"))];
    return {
      answer,
      sources,
      retrievedChunks: results.map((r) => ({ text: r.text, score: r.score, metadata: r.metadata })),
      similarityScores: results.map((r) => r.score),
      tokensUsed: tokens,
      pipelineSteps: steps,
    };
  }

  /**
   * Answer multiple questions sequentially.
   */
  async batchQuery(questions: string[]): Promise<RAGResponse[]> {
    const results: RAGResponse[] = [];
    for (const q of questions) {
      results.push(await this.query(q));
    }
    return results;
  }

  /**
   * Add a single document to the knowledge base (incremental ingest).
   */
  async addDocument(text: string, metadata: Record<string, unknown> = {}): Promise<void> {
    await this.ingestText(text, metadata);
  }

  /**
   * Remove all chunks whose `metadata.source` matches `sourceId`.
   *
   * @returns Number of chunks removed.
   */
  removeDocument(sourceId: string): number {
    const ids = (this.vectorStore as unknown as { documents: Array<{ id: string; metadata: Record<string, unknown> }> })
      .documents
      .filter((d) => d.metadata?.source === sourceId)
      .map((d) => d.id);
    for (const id of ids) {
      this.vectorStore.delete(id);
    }
    return ids.length;
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function runDemo(): Promise<void> {
  const SAMPLE_DOCS: Record<string, string> = {
    "return-policy.md":
      "Our return policy allows returns within 30 days of purchase. " +
      "Damaged items require photo evidence. Refunds processed within 5-7 business days.",
    "shipping-info.md":
      "Standard shipping: 3-5 days at $4.99 (free over $50). " +
      "Express: $14.99, 1-2 days. Overnight: $29.99 before 2 PM EST.",
    "faq.md":
      "Payment methods: Visa, Mastercard, PayPal, Apple Pay. " +
      "All transactions use 256-bit TLS encryption.",
    "products.md":
      "WidgetPro 3000: $199.99, 2-year warranty. " +
      "WidgetLite: $49.99, 1-year warranty.",
  };

  console.log("=".repeat(60));
  console.log("RAG PIPELINE DEMO (TypeScript)");
  console.log("=".repeat(60));

  const embedder = new EmbeddingGenerator("text-embedding-3-small");
  const vectorStore = new SimpleVectorStore();
  const pipeline = new RAGPipeline(vectorStore, embedder, "gpt-4o", 200, 40, 4, 0.6);

  console.log("\n--- Ingesting sample documents ---");
  for (const [source, text] of Object.entries(SAMPLE_DOCS)) {
    const n = await pipeline.ingestText(text, { source });
    console.log(`  ${source}: ${n} chunk(s)`);
  }
  console.log(`  Total stored: ${vectorStore.count()} chunks`);

  const queries = [
    ["1. Simple factual", "How long do I have to return a damaged item?"],
    ["2. Multi-doc synthesis", "If I order a $60 item and want it tomorrow, what are my shipping options?"],
    ["3. No relevant docs", "What is the capital of France?"],
    ["4. Citation-enforced", "What warranty does the WidgetPro 3000 include?"],
  ] as const;

  for (const [label, question] of queries) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`${label}`);
    console.log(`Q: ${question}`);
    console.log("-".repeat(60));

    const response =
      label.includes("Citation")
        ? await pipeline.queryWithCitations(question)
        : await pipeline.query(question);

    for (const step of response.pipelineSteps) {
      console.log(`  → ${step}`);
    }
    console.log(`\nAnswer:\n${response.answer}`);
    console.log(`Sources: ${JSON.stringify(response.sources)}`);
    console.log(`Scores:  ${JSON.stringify(response.similarityScores.map((s) => +s.toFixed(3)))}`);
    console.log(`Tokens:  ${response.tokensUsed}`);
  }
}

// Run demo when executed directly
runDemo().catch(console.error);
