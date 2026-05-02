/**
 * In-memory vector store with cosine similarity search (TypeScript port).
 *
 * Suitable for prototyping and datasets up to ~10,000 documents.
 * For production, switch to a dedicated vector database (Qdrant, Pinecone, etc.).
 *
 * See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
 */

import { readFileSync, writeFileSync } from "fs";
import { randomUUID } from "crypto";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StoredDocument {
  id: string;
  text: string;
  embedding: number[];
  metadata: Record<string, unknown>;
}

export interface SearchResult {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface BatchItem {
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// SimpleVectorStore
// ---------------------------------------------------------------------------

/**
 * In-memory vector store for prototyping and small datasets.
 *
 * Documents are stored as plain objects and searched with brute-force
 * cosine similarity — O(n) per query. Acceptable for up to ~10,000 documents.
 */
export class SimpleVectorStore {
  private documents: StoredDocument[] = [];

  // -------------------------------------------------------------------------
  // Mutation
  // -------------------------------------------------------------------------

  /**
   * Add a single document and return its generated ID.
   *
   * @param text      - The document's raw text content.
   * @param embedding - Pre-computed embedding vector for `text`.
   * @param metadata  - Optional key-value pairs for filtering.
   * @returns A UUID string that uniquely identifies the stored document.
   */
  add(
    text: string,
    embedding: number[],
    metadata?: Record<string, unknown>,
  ): string {
    const id = randomUUID();
    this.documents.push({ id, text, embedding, metadata: metadata ?? {} });
    return id;
  }

  /**
   * Add multiple documents at once.
   *
   * @param items - Array of `{ text, embedding, metadata? }` objects.
   * @returns Array of generated document IDs in the same order as `items`.
   */
  addBatch(items: BatchItem[]): string[] {
    return items.map((item) =>
      this.add(item.text, item.embedding, item.metadata),
    );
  }

  /**
   * Remove a document by its ID.
   *
   * @param docId - The ID returned by `add` or `addBatch`.
   * @returns `true` if a document was found and removed, `false` otherwise.
   */
  delete(docId: string): boolean {
    const before = this.documents.length;
    this.documents = this.documents.filter((d) => d.id !== docId);
    return this.documents.length < before;
  }

  /** Remove all documents from the store. */
  clear(): void {
    this.documents = [];
  }

  // -------------------------------------------------------------------------
  // Query
  // -------------------------------------------------------------------------

  /**
   * Return the `k` most similar documents to `queryEmbedding`.
   *
   * Metadata filtering is applied before scoring.
   *
   * @param queryEmbedding - Embedding vector for the user's query.
   * @param k              - Maximum number of results to return.
   * @param filterMetadata - If provided, only documents whose metadata
   *                         contains all the specified key-value pairs are
   *                         considered.
   * @returns Array of `SearchResult` objects sorted highest-similarity first.
   */
  search(
    queryEmbedding: number[],
    k: number = 5,
    filterMetadata?: Record<string, unknown>,
  ): SearchResult[] {
    const candidates = this.applyFilter(filterMetadata);
    if (candidates.length === 0) return [];

    const scored: SearchResult[] = candidates.map((doc) => ({
      id: doc.id,
      text: doc.text,
      score: cosineSimilarity(queryEmbedding, doc.embedding),
      metadata: doc.metadata,
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  /**
   * Search and return only results at or above `threshold` similarity.
   *
   * @param queryEmbedding - Embedding vector for the user's query.
   * @param threshold      - Minimum cosine similarity score (inclusive).
   * @param k              - Maximum number of results to return.
   * @returns Filtered `SearchResult` array.
   */
  searchWithThreshold(
    queryEmbedding: number[],
    threshold: number = 0.7,
    k: number = 5,
  ): SearchResult[] {
    return this.search(queryEmbedding, k).filter((r) => r.score >= threshold);
  }

  // -------------------------------------------------------------------------
  // Introspection
  // -------------------------------------------------------------------------

  /** Return the number of documents currently stored. */
  count(): number {
    return this.documents.length;
  }

  // -------------------------------------------------------------------------
  // Persistence
  // -------------------------------------------------------------------------

  /**
   * Persist the store to a JSON file.
   *
   * Intended for small datasets only.
   *
   * @param filepath - Destination file path.
   */
  save(filepath: string): void {
    writeFileSync(filepath, JSON.stringify(this.documents), "utf-8");
  }

  /**
   * Replace the current store contents by loading from `filepath`.
   *
   * @param filepath - Path to a JSON file written by `save`.
   */
  load(filepath: string): void {
    const data = readFileSync(filepath, "utf-8");
    this.documents = JSON.parse(data) as StoredDocument[];
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private applyFilter(
    filterMetadata?: Record<string, unknown>,
  ): StoredDocument[] {
    if (!filterMetadata || Object.keys(filterMetadata).length === 0) {
      return this.documents;
    }
    return this.documents.filter((doc) =>
      Object.entries(filterMetadata).every(
        ([k, v]) => doc.metadata[k] === v,
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Module-level helper
// ---------------------------------------------------------------------------

function cosineSimilarity(a: number[], b: number[]): number {
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

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function randomUnitVector(dim: number, seed: number): number[] {
  // Simple LCG for reproducibility without external deps.
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
  const v = Array.from({ length: dim }, () => rand() - 0.5);
  const norm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
  return v.map((x) => x / norm);
}

function addNoise(v: number[], scale: number, seed: number): number[] {
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0xffffffff - 0.5) * 2;
  };
  const noisy = v.map((x) => x + rand() * scale);
  const norm = Math.sqrt(noisy.reduce((acc, x) => acc + x * x, 0));
  return noisy.map((x) => x / norm);
}

function main(): void {
  import("os").then((os) => {
    import("path").then((path) => {
      const store = new SimpleVectorStore();
      const dim = 16;
      const supportCenter = randomUnitVector(dim, 42);
      const marketingCenter = supportCenter.map((x) => -x);

      const supportDocs = [
        "How do I return a damaged item?",
        "What is the refund timeline?",
        "Can I exchange a product for a different size?",
        "How long does shipping take after an order is placed?",
        "Where do I track my order status?",
        "Is there a restocking fee for returns?",
        "How do I report a missing package?",
        "Can I cancel my order after it ships?",
        "What payment methods are accepted?",
        "How do I apply a coupon code at checkout?",
      ];
      const marketingDocs = [
        "Introducing our new summer collection.",
        "Get 20% off your first order with code WELCOME20.",
        "Shop our top-rated products of the year.",
        "Free shipping on orders over $50.",
        "New arrivals every Monday.",
        "Subscribe to our newsletter for exclusive deals.",
        "Gift cards available in any denomination.",
        "Follow us on social media for style inspiration.",
        "Our loyalty program earns you points on every purchase.",
        "Refer a friend and both of you get $10 off.",
      ];

      const supportIds = supportDocs.map((text, i) =>
        store.add(text, addNoise(supportCenter, 0.15, 100 + i), {
          category: "support",
        }),
      );
      marketingDocs.forEach((text, i) =>
        store.add(text, addNoise(marketingCenter, 0.15, 200 + i), {
          category: "marketing",
        }),
      );

      console.log(`Store contains ${store.count()} documents.`);

      const queryEmb = addNoise(supportCenter, 0.05, 999);

      console.log("\n=== Search: query near support cluster ===");
      store.search(queryEmb, 5).forEach((r) => {
        console.log(
          `  score=${r.score.toFixed(3)}  [${r.metadata["category"]}]  ${r.text}`,
        );
      });

      console.log("\n=== Metadata-filtered search (category=support only) ===");
      store
        .search(queryEmb, 5, { category: "support" })
        .forEach((r) => console.log(`  score=${r.score.toFixed(3)}  ${r.text}`));

      console.log("\n=== Threshold search (>= 0.90) ===");
      const thresh = store.searchWithThreshold(queryEmb, 0.9, 10);
      console.log(`  ${thresh.length} results above threshold.`);

      store.delete(supportIds[0]);
      console.log(`\nDeleted first support doc. Store now has ${store.count()} docs.`);

      const tmpPath = path.join(os.tmpdir(), `vector_store_${Date.now()}.json`);
      store.save(tmpPath);
      console.log(`\nSaved store to ${tmpPath}.`);

      const store2 = new SimpleVectorStore();
      store2.load(tmpPath);
      console.log(`Loaded store has ${store2.count()} documents.`);
      store2.search(queryEmb, 3).forEach((r) => {
        console.log(`  score=${r.score.toFixed(3)}  ${r.text}`);
      });
    });
  });
}

main();
