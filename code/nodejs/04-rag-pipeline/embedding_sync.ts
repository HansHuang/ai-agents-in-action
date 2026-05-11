/**
 * Embedding sync manager: keep a vector index in sync with source documents.
 *
 * Detects document changes via content hashing and performs incremental
 * adds/updates/deletes without rebuilding the entire index.
 * See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
 */

import { createHash } from "crypto";

export interface StoredDocument {
  id: string;
  content: string;
  contentHash: string;
  updatedAt: number;
  metadata?: Record<string, unknown>;
}

export interface SyncReport {
  added: number;
  updated: number;
  deleted: number;
  unchanged: number;
  errors: string[];
  durationMs: number;
}

export type SyncHealth = "healthy" | "degraded" | "stale";

// ---------------------------------------------------------------------------
// InMemoryDocumentStore
// ---------------------------------------------------------------------------

export class InMemoryDocumentStore {
  private docs = new Map<string, StoredDocument>();

  upsert(id: string, content: string, metadata?: Record<string, unknown>): StoredDocument {
    const contentHash = hashContent(content);
    const doc: StoredDocument = { id, content, contentHash, updatedAt: Date.now(), metadata };
    this.docs.set(id, doc);
    return doc;
  }

  delete(id: string): boolean {
    return this.docs.delete(id);
  }

  get(id: string): StoredDocument | undefined {
    return this.docs.get(id);
  }

  all(): StoredDocument[] {
    return Array.from(this.docs.values());
  }

  has(id: string): boolean {
    return this.docs.has(id);
  }
}

// ---------------------------------------------------------------------------
// EmbeddingSyncManager
// ---------------------------------------------------------------------------

export interface VectorIndexAdapter {
  upsert(id: string, content: string, metadata?: Record<string, unknown>): Promise<void>;
  delete(id: string): Promise<void>;
  hasId(id: string): Promise<boolean>;
  allIds(): Promise<string[]>;
}

export class EmbeddingSyncManager {
  private indexedHashes = new Map<string, string>(); // id -> contentHash

  constructor(
    private store: InMemoryDocumentStore,
    private index: VectorIndexAdapter
  ) {}

  /** Sync the vector index with the current document store state. */
  async sync(): Promise<SyncReport> {
    const start = Date.now();
    const report: SyncReport = { added: 0, updated: 0, deleted: 0, unchanged: 0, errors: [], durationMs: 0 };

    const docs = this.store.all();

    for (const doc of docs) {
      try {
        const prevHash = this.indexedHashes.get(doc.id);
        if (!prevHash) {
          await this.index.upsert(doc.id, doc.content, doc.metadata);
          this.indexedHashes.set(doc.id, doc.contentHash);
          report.added++;
        } else if (prevHash !== doc.contentHash) {
          await this.index.upsert(doc.id, doc.content, doc.metadata);
          this.indexedHashes.set(doc.id, doc.contentHash);
          report.updated++;
        } else {
          report.unchanged++;
        }
      } catch (err) {
        report.errors.push(`${doc.id}: ${String(err)}`);
      }
    }

    // Delete docs from index that no longer exist in the store
    const indexedIds = await this.index.allIds();
    for (const id of indexedIds) {
      if (!this.store.has(id)) {
        try {
          await this.index.delete(id);
          this.indexedHashes.delete(id);
          report.deleted++;
        } catch (err) {
          report.errors.push(`delete ${id}: ${String(err)}`);
        }
      }
    }

    report.durationMs = Date.now() - start;
    return report;
  }

  /** Return overall health of the index vs the document store. */
  health(): SyncHealth {
    const storeCount = this.store.all().length;
    const indexedCount = this.indexedHashes.size;
    if (indexedCount === 0 && storeCount > 0) return "stale";
    if (Math.abs(indexedCount - storeCount) > storeCount * 0.1) return "degraded";
    return "healthy";
  }
}

function hashContent(content: string): string {
  return createHash("sha256").update(content).digest("hex");
}

// Demo
async function main(): Promise<void> {
  const store = new InMemoryDocumentStore();
  store.upsert("doc1", "The quick brown fox jumps.");
  store.upsert("doc2", "Vector databases store embedding vectors.");
  store.upsert("doc3", "RAG pipelines improve answer quality.");

  // Mock index adapter
  const indexedDocs = new Map<string, string>();
  const adapter: VectorIndexAdapter = {
    async upsert(id, content) { indexedDocs.set(id, content); },
    async delete(id) { indexedDocs.delete(id); },
    async hasId(id) { return indexedDocs.has(id); },
    async allIds() { return Array.from(indexedDocs.keys()); },
  };

  const manager = new EmbeddingSyncManager(store, adapter);
  const report = await manager.sync();
  console.log("Sync report:", report);
  console.log("Health:", manager.health());

  // Update a doc and re-sync
  store.upsert("doc1", "The quick brown fox leaped over the fence.");
  const report2 = await manager.sync();
  console.log("After update:", report2);
}

main().catch(console.error);
