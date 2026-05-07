/**
 * Vector database abstraction layer — TypeScript port.
 *
 * Provides a unified interface for multiple vector database backends.
 * Swap backends with a single config change; no changes to calling code.
 *
 * Backends: SimpleVectorStore (in-memory), ChromaDB, QdrantDB, PineconeDB.
 *
 * @example
 * ```ts
 * import { VectorDBFactory, VectorDocument } from "./vector_database.js";
 *
 * const db = VectorDBFactory.create("chroma", { collectionName: "my_docs" });
 * await db.insert([{ id: "1", text: "hello", embedding: [0.1, 0.2], metadata: {} }]);
 * const results = await db.search([0.1, 0.2], 5);
 * ```
 *
 * See: docs/05-the-tool-ecosystem/02-vector-databases.md
 */

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

export interface VectorDocument {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

export interface SearchResult {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Abstract interface
// ---------------------------------------------------------------------------

export interface VectorDatabase {
  /** Insert (upsert) documents.  Returns the count inserted. */
  insert(documents: VectorDocument[]): Promise<number>;

  /** Return the k nearest documents to queryEmbedding. */
  search(
    queryEmbedding: number[],
    k?: number,
    filterMetadata?: Record<string, unknown>,
  ): Promise<SearchResult[]>;

  /** Delete documents by ID.  Returns count deleted. */
  delete(ids: string[]): Promise<number>;

  /** Total number of documents currently stored. */
  count(): Promise<number>;

  /** Remove all documents. */
  clear(): Promise<void>;

  /**
   * Insert in batches.  Override if the backend has a more efficient native
   * batch API; the default implementation calls insert() in chunks.
   */
  batchInsert(documents: VectorDocument[], batchSize?: number): Promise<number>;
}

// ---------------------------------------------------------------------------
// Helpers
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

async function defaultBatchInsert(
  db: VectorDatabase,
  documents: VectorDocument[],
  batchSize: number,
): Promise<number> {
  let total = 0;
  for (let i = 0; i < documents.length; i += batchSize) {
    total += await db.insert(documents.slice(i, i + batchSize));
  }
  return total;
}

// ---------------------------------------------------------------------------
// Backend: SimpleVectorStore
// ---------------------------------------------------------------------------

interface StoredDoc {
  id: string;
  text: string;
  embedding: number[];
  metadata: Record<string, unknown>;
}

/**
 * In-memory vector store — no dependencies required.
 *
 * Brute-force O(n) cosine similarity.  Suitable for tests, prototyping,
 * and datasets up to ~10,000 documents.
 */
export class SimpleVectorStore implements VectorDatabase {
  private docs: StoredDoc[] = [];

  async insert(documents: VectorDocument[]): Promise<number> {
    for (const doc of documents) {
      this.docs = this.docs.filter((d) => d.id !== doc.id);
      this.docs.push({
        id: doc.id,
        text: doc.text,
        embedding: doc.embedding,
        metadata: doc.metadata ?? {},
      });
    }
    return documents.length;
  }

  async search(
    queryEmbedding: number[],
    k = 5,
    filterMetadata?: Record<string, unknown>,
  ): Promise<SearchResult[]> {
    let candidates = this.docs;
    if (filterMetadata) {
      candidates = candidates.filter((d) =>
        Object.entries(filterMetadata).every(
          ([key, val]) => d.metadata[key] === val,
        ),
      );
    }
    const scored: SearchResult[] = candidates.map((d) => ({
      id: d.id,
      text: d.text,
      score: cosineSimilarity(queryEmbedding, d.embedding),
      metadata: d.metadata,
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  async delete(ids: string[]): Promise<number> {
    const idSet = new Set(ids);
    const before = this.docs.length;
    this.docs = this.docs.filter((d) => !idSet.has(d.id));
    return before - this.docs.length;
  }

  async count(): Promise<number> {
    return this.docs.length;
  }

  async clear(): Promise<void> {
    this.docs = [];
  }

  async batchInsert(documents: VectorDocument[], batchSize = 100): Promise<number> {
    return defaultBatchInsert(this, documents, batchSize);
  }
}

// ---------------------------------------------------------------------------
// Backend: ChromaDB
// ---------------------------------------------------------------------------

export interface ChromaDBOptions {
  collectionName?: string;
  persistDirectory?: string;
}

/**
 * Chroma vector database backend.
 *
 * Uses the ``chromadb`` npm package.  Data is persisted to disk using a
 * persistent client.  Chroma returns cosine *distances* in [0, 2]; we
 * convert to similarity as ``1 - distance / 2``.
 */
export class ChromaDB implements VectorDatabase {
  private collectionName: string;
  private persistDirectory: string;
  private collection: unknown = null;

  constructor({
    collectionName = "documents",
    persistDirectory = "./chroma_db",
  }: ChromaDBOptions = {}) {
    this.collectionName = collectionName;
    this.persistDirectory = persistDirectory;
  }

  private async getCollection(): Promise<unknown> {
    if (this.collection) return this.collection;
    // Dynamic import so the package is optional at module load time
    const { ChromaClient } = await import("chromadb");
    const client = new ChromaClient({ path: this.persistDirectory });
    this.collection = await client.getOrCreateCollection({
      name: this.collectionName,
      metadata: { "hnsw:space": "cosine" },
    });
    return this.collection;
  }

  async insert(documents: VectorDocument[]): Promise<number> {
    if (documents.length === 0) return 0;
    const col = (await this.getCollection()) as {
      upsert: (args: {
        ids: string[];
        embeddings: number[][];
        documents: string[];
        metadatas: Record<string, unknown>[];
      }) => Promise<void>;
    };
    await col.upsert({
      ids: documents.map((d) => d.id),
      embeddings: documents.map((d) => d.embedding),
      documents: documents.map((d) => d.text),
      metadatas: documents.map((d) => d.metadata ?? {}),
    });
    return documents.length;
  }

  async search(
    queryEmbedding: number[],
    k = 5,
    filterMetadata?: Record<string, unknown>,
  ): Promise<SearchResult[]> {
    const col = (await this.getCollection()) as {
      query: (args: {
        queryEmbeddings: number[][];
        nResults: number;
        where?: Record<string, unknown>;
      }) => Promise<{
        ids: string[][];
        documents: (string | null)[][];
        distances: number[][];
        metadatas: Record<string, unknown>[][];
      }>;
      count: () => Promise<number>;
    };
    const n = Math.min(k, await col.count());
    if (n === 0) return [];
    const results = await col.query({
      queryEmbeddings: [queryEmbedding],
      nResults: n,
      where: filterMetadata,
    });
    return results.ids[0].map((id, i) => ({
      id,
      text: results.documents[0][i] ?? "",
      score: 1 - (results.distances[0][i] ?? 0) / 2,
      metadata: results.metadatas[0][i] ?? {},
    }));
  }

  async delete(ids: string[]): Promise<number> {
    const col = (await this.getCollection()) as {
      delete: (args: { ids: string[] }) => Promise<void>;
    };
    await col.delete({ ids });
    return ids.length;
  }

  async count(): Promise<number> {
    const col = (await this.getCollection()) as { count: () => Promise<number> };
    return col.count();
  }

  async clear(): Promise<void> {
    const { ChromaClient } = await import("chromadb");
    const client = new ChromaClient({ path: this.persistDirectory });
    await client.deleteCollection({ name: this.collectionName });
    this.collection = null;
  }

  async batchInsert(documents: VectorDocument[], batchSize = 100): Promise<number> {
    return defaultBatchInsert(this, documents, batchSize);
  }
}

// ---------------------------------------------------------------------------
// Backend: QdrantDB
// ---------------------------------------------------------------------------

export interface QdrantDBOptions {
  collectionName?: string;
  host?: string;
  port?: number;
  dimension?: number;
}

/**
 * Qdrant vector database backend.
 *
 * Uses ``@qdrant/js-client-rest``.  Auto-creates the collection on first use.
 */
export class QdrantDB implements VectorDatabase {
  private collectionName: string;
  private host: string;
  private port: number;
  private dimension: number;
  private client: unknown = null;

  constructor({
    collectionName = "documents",
    host = "localhost",
    port = 6333,
    dimension = 1536,
  }: QdrantDBOptions = {}) {
    this.collectionName = collectionName;
    this.host = host;
    this.port = port;
    this.dimension = dimension;
  }

  private async getClient(): Promise<unknown> {
    if (this.client) return this.client;
    const { QdrantClient } = await import("@qdrant/js-client-rest");
    const client = new QdrantClient({ host: this.host, port: this.port });

    const { collections } = await client.getCollections();
    const exists = collections.some(
      (c: { name: string }) => c.name === this.collectionName,
    );
    if (!exists) {
      await client.createCollection(this.collectionName, {
        vectors: { size: this.dimension, distance: "Cosine" },
      });
    }
    this.client = client;
    return this.client;
  }

  async insert(documents: VectorDocument[]): Promise<number> {
    if (documents.length === 0) return 0;
    const client = (await this.getClient()) as {
      upsert: (
        name: string,
        args: {
          wait: boolean;
          points: { id: string; vector: number[]; payload: Record<string, unknown> }[];
        },
      ) => Promise<void>;
    };
    await client.upsert(this.collectionName, {
      wait: true,
      points: documents.map((d) => ({
        id: d.id,
        vector: d.embedding,
        payload: { text: d.text, ...(d.metadata ?? {}) },
      })),
    });
    return documents.length;
  }

  async search(
    queryEmbedding: number[],
    k = 5,
    filterMetadata?: Record<string, unknown>,
  ): Promise<SearchResult[]> {
    const client = (await this.getClient()) as {
      search: (
        name: string,
        args: {
          vector: number[];
          limit: number;
          filter?: { must: { key: string; match: { value: unknown } }[] };
          withPayload: boolean;
        },
      ) => Promise<{ id: string; score: number; payload?: Record<string, unknown> }[]>;
    };

    const filter = filterMetadata
      ? {
          must: Object.entries(filterMetadata).map(([key, value]) => ({
            key,
            match: { value },
          })),
        }
      : undefined;

    const hits = await client.search(this.collectionName, {
      vector: queryEmbedding,
      limit: k,
      filter,
      withPayload: true,
    });

    return hits.map((h) => {
      const { text, ...rest } = (h.payload ?? {}) as { text?: string } & Record<string, unknown>;
      return {
        id: String(h.id),
        text: text ?? "",
        score: h.score,
        metadata: rest,
      };
    });
  }

  async delete(ids: string[]): Promise<number> {
    const client = (await this.getClient()) as {
      delete: (name: string, args: { wait: boolean; points: string[] }) => Promise<void>;
    };
    await client.delete(this.collectionName, { wait: true, points: ids });
    return ids.length;
  }

  async count(): Promise<number> {
    const client = (await this.getClient()) as {
      count: (name: string) => Promise<{ count: number }>;
    };
    const result = await client.count(this.collectionName);
    return result.count;
  }

  async clear(): Promise<void> {
    const client = (await this.getClient()) as {
      deleteCollection: (name: string) => Promise<void>;
    };
    await client.deleteCollection(this.collectionName);
    this.client = null;
  }

  async batchInsert(documents: VectorDocument[], batchSize = 100): Promise<number> {
    return defaultBatchInsert(this, documents, batchSize);
  }
}

// ---------------------------------------------------------------------------
// Backend: PineconeDB
// ---------------------------------------------------------------------------

export interface PineconeDBOptions {
  apiKey: string;
  indexName?: string;
  dimension?: number;
}

/**
 * Pinecone vector database backend.
 *
 * Uses ``@pinecone-database/pinecone``.  Auto-creates the index if absent.
 * Metadata is stored as Pinecone record metadata; ``text`` is a reserved key.
 */
export class PineconeDB implements VectorDatabase {
  private apiKey: string;
  private indexName: string;
  private dimension: number;
  private index: unknown = null;

  constructor({ apiKey, indexName = "documents", dimension = 1536 }: PineconeDBOptions) {
    this.apiKey = apiKey;
    this.indexName = indexName;
    this.dimension = dimension;
  }

  private async getIndex(): Promise<unknown> {
    if (this.index) return this.index;
    const { Pinecone } = await import("@pinecone-database/pinecone");
    const pc = new Pinecone({ apiKey: this.apiKey });

    const existing = (await pc.listIndexes()).indexes ?? [];
    const names = existing.map((i: { name: string }) => i.name);
    if (!names.includes(this.indexName)) {
      await pc.createIndex({
        name: this.indexName,
        dimension: this.dimension,
        metric: "cosine",
        spec: { serverless: { cloud: "aws", region: "us-east-1" } },
      });
    }
    this.index = pc.index(this.indexName);
    return this.index;
  }

  async insert(documents: VectorDocument[]): Promise<number> {
    if (documents.length === 0) return 0;
    const idx = (await this.getIndex()) as {
      upsert: (
        records: { id: string; values: number[]; metadata: Record<string, unknown> }[],
      ) => Promise<void>;
    };
    await idx.upsert(
      documents.map((d) => ({
        id: d.id,
        values: d.embedding,
        metadata: { text: d.text, ...(d.metadata ?? {}) },
      })),
    );
    return documents.length;
  }

  async search(
    queryEmbedding: number[],
    k = 5,
    filterMetadata?: Record<string, unknown>,
  ): Promise<SearchResult[]> {
    const idx = (await this.getIndex()) as {
      query: (args: {
        vector: number[];
        topK: number;
        filter?: Record<string, unknown>;
        includeMetadata: boolean;
      }) => Promise<{
        matches?: { id: string; score: number; metadata?: Record<string, unknown> }[];
      }>;
    };
    const response = await idx.query({
      vector: queryEmbedding,
      topK: k,
      filter: filterMetadata,
      includeMetadata: true,
    });
    return (response.matches ?? []).map((m) => {
      const { text, ...rest } = (m.metadata ?? {}) as { text?: string } & Record<string, unknown>;
      return {
        id: m.id,
        text: text ?? "",
        score: m.score ?? 0,
        metadata: rest,
      };
    });
  }

  async delete(ids: string[]): Promise<number> {
    const idx = (await this.getIndex()) as {
      deleteMany: (ids: string[]) => Promise<void>;
    };
    await idx.deleteMany(ids);
    return ids.length;
  }

  async count(): Promise<number> {
    const idx = (await this.getIndex()) as {
      describeIndexStats: () => Promise<{ totalRecordCount?: number }>;
    };
    const stats = await idx.describeIndexStats();
    return stats.totalRecordCount ?? 0;
  }

  async clear(): Promise<void> {
    const idx = (await this.getIndex()) as {
      deleteAll: () => Promise<void>;
    };
    await idx.deleteAll();
  }

  async batchInsert(documents: VectorDocument[], batchSize = 100): Promise<number> {
    return defaultBatchInsert(this, documents, batchSize);
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

type DBOptions =
  | ({ type: "simple" } & Record<string, never>)
  | ({ type: "chroma" } & ChromaDBOptions)
  | ({ type: "qdrant" } & QdrantDBOptions)
  | ({ type: "pinecone" } & PineconeDBOptions);

/**
 * Create vector database instances by name or config object.
 *
 * @example
 * ```ts
 * const db = VectorDBFactory.create("simple");
 * const db2 = VectorDBFactory.createFromConfig({ type: "chroma", collectionName: "docs" });
 * ```
 */
export class VectorDBFactory {
  static create(type: "simple"): SimpleVectorStore;
  static create(type: "chroma", options?: ChromaDBOptions): ChromaDB;
  static create(type: "qdrant", options?: QdrantDBOptions): QdrantDB;
  static create(type: "pinecone", options: PineconeDBOptions): PineconeDB;
  static create(type: string, options?: Record<string, unknown>): VectorDatabase {
    switch (type) {
      case "simple":
        return new SimpleVectorStore();
      case "chroma":
        return new ChromaDB(options as ChromaDBOptions);
      case "qdrant":
        return new QdrantDB(options as QdrantDBOptions);
      case "pinecone":
        return new PineconeDB(options as PineconeDBOptions);
      default:
        throw new Error(
          `Unknown database type: ${type}. Available: simple, chroma, qdrant, pinecone`,
        );
    }
  }

  /** Create from a config object where ``type`` selects the backend. */
  static createFromConfig(config: DBOptions): VectorDatabase {
    const { type, ...options } = config;
    return VectorDBFactory.create(type as never, options as Record<string, unknown>);
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function runDemo(): Promise<void> {
  const DIM = 64;
  const N_DOCS = 100;

  // Deterministic pseudo-random
  let seed = 42;
  const rand = (): number => {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  };
  const randEmbed = (): number[] => {
    const v = Array.from({ length: DIM }, () => rand() * 2 - 1);
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return v.map((x) => x / norm);
  };

  const embeddings = Array.from({ length: N_DOCS }, randEmbed);
  const docs: VectorDocument[] = embeddings.map((emb, i) => ({
    id: String(i),
    text: `Document ${i} about topic ${i % 5}`,
    embedding: emb,
    metadata: { category: `cat_${i % 3}` },
  }));
  const queryEmbedding = embeddings[0];

  const backends: [string, VectorDatabase][] = [["Simple", new SimpleVectorStore()]];

  console.log("=".repeat(60));
  console.log("Vector Database Abstraction Demo (TypeScript)");
  console.log("=".repeat(60));

  for (const [name, db] of backends) {
    const t0 = performance.now();
    const inserted = await db.batchInsert(docs);
    const insertMs = performance.now() - t0;

    const t1 = performance.now();
    const results = await db.search(queryEmbedding, 5);
    const searchMs = performance.now() - t1;

    console.log(`\n[${name}]`);
    console.log(`  Inserted ${inserted} docs in ${insertMs.toFixed(1)} ms`);
    console.log(`  Top-5 search in ${searchMs.toFixed(2)} ms`);
    console.log(`  Count: ${await db.count()}`);
    console.log(`  Top result: id=${results[0]?.id}, score=${results[0]?.score.toFixed(4)}`);

    const filtered = await db.search(queryEmbedding, 5, { category: "cat_0" });
    console.log(`  Filtered (category=cat_0): [${filtered.map((r) => r.id).join(", ")}]`);
  }
}

// Run when executed directly: ts-node vector_database.ts
const isMain = import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  runDemo().catch(console.error);
}
