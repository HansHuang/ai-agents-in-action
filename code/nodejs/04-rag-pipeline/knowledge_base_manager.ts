/**
 * Incremental knowledge base management for evolving document collections.
 *
 * Handles document additions, updates, and deletions without rebuilding
 * the entire index. Uses content hashing to avoid redundant re-embedding.
 * See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
 */

import { createHash } from "crypto";

export interface KBDocument {
  id: string;
  content: string;
  contentHash: string;
  addedAt: number;
  updatedAt: number;
  metadata?: Record<string, unknown>;
}

export interface EmbedAndStoreFn {
  (id: string, content: string, metadata?: Record<string, unknown>): Promise<void>;
}

export interface RemoveFn {
  (id: string): Promise<void>;
}

/**
 * KnowledgeBaseManager tracks document state and drives incremental updates.
 */
export class KnowledgeBaseManager {
  private docs = new Map<string, KBDocument>();

  constructor(
    private embedAndStore: EmbedAndStoreFn,
    private remove: RemoveFn
  ) {}

  /** Add or update a document. Returns true if the index was updated. */
  async addDocument(
    id: string,
    content: string,
    metadata?: Record<string, unknown>
  ): Promise<boolean> {
    const hash = hashContent(content);
    const existing = this.docs.get(id);

    if (existing && existing.contentHash === hash) {
      return false; // unchanged
    }

    await this.embedAndStore(id, content, metadata);
    const now = Date.now();
    this.docs.set(id, {
      id,
      content,
      contentHash: hash,
      addedAt: existing?.addedAt ?? now,
      updatedAt: now,
      metadata,
    });
    return true;
  }

  /** Delete a document from the KB and vector index. */
  async deleteDocument(id: string): Promise<boolean> {
    if (!this.docs.has(id)) return false;
    await this.remove(id);
    this.docs.delete(id);
    return true;
  }

  /** List all tracked documents. */
  list(): KBDocument[] {
    return Array.from(this.docs.values());
  }

  /** Check if a document exists. */
  has(id: string): boolean {
    return this.docs.has(id);
  }

  /** Get a document by ID. */
  get(id: string): KBDocument | undefined {
    return this.docs.get(id);
  }

  /** Bulk-load documents, only re-embedding changed ones. */
  async bulkLoad(
    documents: { id: string; content: string; metadata?: Record<string, unknown> }[]
  ): Promise<{ added: number; updated: number; unchanged: number }> {
    let added = 0, updated = 0, unchanged = 0;
    for (const doc of documents) {
      const existed = this.docs.has(doc.id);
      const changed = await this.addDocument(doc.id, doc.content, doc.metadata);
      if (!changed) unchanged++;
      else if (existed) updated++;
      else added++;
    }
    return { added, updated, unchanged };
  }
}

function hashContent(content: string): string {
  return createHash("sha256").update(content).digest("hex");
}

// Demo
async function main(): Promise<void> {
  const indexed = new Map<string, string>();

  const manager = new KnowledgeBaseManager(
    async (id, content) => {
      console.log(`  → embedding "${id}"`);
      indexed.set(id, content);
    },
    async (id) => {
      console.log(`  → removing "${id}"`);
      indexed.delete(id);
    }
  );

  console.log("Bulk loading 3 documents...");
  const result = await manager.bulkLoad([
    { id: "doc1", content: "TypeScript is a typed superset of JavaScript." },
    { id: "doc2", content: "Vector databases store embedding vectors for semantic search." },
    { id: "doc3", content: "RAG improves answer quality with retrieved context." },
  ]);
  console.log("Result:", result);

  console.log("\nRe-loading with one updated document...");
  const result2 = await manager.bulkLoad([
    { id: "doc1", content: "TypeScript is a typed superset of JavaScript." }, // unchanged
    { id: "doc2", content: "Vector databases enable efficient nearest-neighbour search." }, // changed
  ]);
  console.log("Result:", result2);

  console.log("\nDeleting doc3...");
  await manager.deleteDocument("doc3");
  console.log("Remaining:", manager.list().map((d) => d.id));
}

main().catch(console.error);
