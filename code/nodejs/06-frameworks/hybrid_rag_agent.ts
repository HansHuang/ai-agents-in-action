/**
 * Hybrid RAG agent: LangChain.js for commodity ops, custom code for logic.
 *
 * Architecture:
 *   FRAMEWORK (commodity parts):
 *     • TextLoader / DirectoryLoader  — load .md / .txt files
 *     • OpenAIEmbeddings              — embed documents and queries
 *     • MemoryVectorStore             — in-process vector storage
 *
 *   YOUR CODE (differentiated parts):
 *     • Agent loop        — you control orchestration
 *     • ContextAssembler  — you control prompt quality
 *     • MemoryManager     — you control conversation budget
 *     • TokenTracker      — you control cost visibility
 *
 * The demo:
 *   1. Ingest inline docs via LangChain.js
 *   2. Query with the hybrid agent
 *   3. Swap the vector store to SimpleVectorStore (zero logic changes)
 *   4. Report custom-code lines vs. framework-code lines
 *
 * Run:
 *   npx tsx hybrid_rag_agent.ts [docs-dir]
 *
 * See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
 */

import { createAI, generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import OpenAI from "openai";

// LangChain.js imports (optional at runtime — graceful fallback)
let LCTextLoader: any;
let LCMemoryVectorStore: any;
let LCOpenAIEmbeddings: any;
let LCDocument: any;
let LANGCHAIN_AVAILABLE = false;

try {
  const { TextLoader } = await import("langchain/document_loaders/fs/text");
  const { MemoryVectorStore } = await import("langchain/vectorstores/memory");
  const { OpenAIEmbeddings } = await import("@langchain/openai");
  const { Document } = await import("@langchain/core/documents");
  LCTextLoader = TextLoader;
  LCMemoryVectorStore = MemoryVectorStore;
  LCOpenAIEmbeddings = OpenAIEmbeddings;
  LCDocument = Document;
  LANGCHAIN_AVAILABLE = true;
} catch {
  // LangChain.js not installed — pure-custom path will be used
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LLM_MODEL = "gpt-4o-mini";
const EMBED_MODEL = "text-embedding-3-small";

// ---------------------------------------------------------------------------
// Inline demo documents (used when no directory is supplied)
// ---------------------------------------------------------------------------

const INLINE_DOCS: Array<{ text: string; source: string }> = [
  {
    source: "rag_phases.md",
    text:
      "RAG has four phases: Ingest (load, chunk, embed, store), " +
      "Retrieve (embed query, search), Augment (build prompt), " +
      "Generate (call LLM with augmented prompt).",
  },
  {
    source: "vector_databases.md",
    text:
      "Vector databases store embeddings and support ANN search. " +
      "Popular choices: Qdrant, Pinecone, Chroma, FAISS, Weaviate.",
  },
  {
    source: "agent_loop.md",
    text:
      "An agent loop: perceive, think (LLM call), act (tool execution), " +
      "observe.  Repeat until a final answer is produced.",
  },
];

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

export interface HybridResult {
  answer: string;
  sources: string[];
  tokens: number;
  retrievalMs: number;
  generationMs: number;
}

// ---------------------------------------------------------------------------
// VectorStoreProtocol — your interface, not LangChain's
// ---------------------------------------------------------------------------

export interface VectorStoreProtocol {
  /** Return top-k results as {text, metadata} objects. */
  similaritySearch(
    query: string,
    k: number
  ): Promise<Array<{ text: string; metadata: Record<string, string> }>>;
  readonly backendName: string;
}

// ---------------------------------------------------------------------------
// LangChainVectorStore — wraps LangChain.js behind VectorStoreProtocol
// ---------------------------------------------------------------------------

export class LangChainVectorStore implements VectorStoreProtocol {
  readonly backendName = "LangChain MemoryVectorStore";
  private store: any;

  private constructor(store: any) {
    this.store = store;
  }

  static async fromDocuments(
    docs: Array<{ text: string; metadata: Record<string, string> }>
  ): Promise<LangChainVectorStore> {
    if (!LANGCHAIN_AVAILABLE) {
      throw new Error(
        "LangChain.js not installed. Run: npm install langchain @langchain/openai @langchain/core"
      );
    }
    const embeddings = new LCOpenAIEmbeddings({ model: EMBED_MODEL });
    const lcDocs = docs.map(
      (d) => new LCDocument({ pageContent: d.text, metadata: d.metadata })
    );
    const store = await LCMemoryVectorStore.fromDocuments(lcDocs, embeddings);
    return new LangChainVectorStore(store);
  }

  async similaritySearch(
    query: string,
    k: number
  ): Promise<Array<{ text: string; metadata: Record<string, string> }>> {
    const results = await this.store.similaritySearch(query, k);
    return results.map((doc: any) => ({
      text: doc.pageContent,
      metadata: doc.metadata as Record<string, string>,
    }));
  }
}

// ---------------------------------------------------------------------------
// SimpleVectorStore — pure TypeScript, no framework dependency
// ---------------------------------------------------------------------------

export class SimpleVectorStore implements VectorStoreProtocol {
  readonly backendName = "SimpleVectorStore";
  private docs: Array<{ text: string; metadata: Record<string, string> }> = [];
  private embeddings: number[][] = [];
  private client: OpenAI;

  constructor(client: OpenAI) {
    this.client = client;
  }

  async addDocuments(
    docs: Array<{ text: string; metadata: Record<string, string> }>
  ): Promise<void> {
    if (docs.length === 0) return;
    const resp = await this.client.embeddings.create({
      model: EMBED_MODEL,
      input: docs.map((d) => d.text),
    });
    for (let i = 0; i < docs.length; i++) {
      this.docs.push(docs[i]);
      this.embeddings.push(resp.data[i].embedding);
    }
  }

  async similaritySearch(
    query: string,
    k: number
  ): Promise<Array<{ text: string; metadata: Record<string, string> }>> {
    if (this.docs.length === 0) return [];

    const qResp = await this.client.embeddings.create({
      model: EMBED_MODEL,
      input: [query],
    });
    const qEmb = qResp.data[0].embedding;

    const scored = this.embeddings.map((emb, idx) => ({
      idx,
      score: cosineSimilarity(qEmb, emb),
    }));
    scored.sort((a, b) => b.score - a.score);

    return scored
      .slice(0, Math.min(k, this.docs.length))
      .map(({ idx }) => this.docs[idx]);
  }
}

// ---------------------------------------------------------------------------
// ContextAssembler — YOUR logic, no framework dependency
// ---------------------------------------------------------------------------

export class ContextAssembler {
  private readonly templates: Record<string, string> = {
    rag_query:
      "Answer the question using ONLY the documents below.\n" +
      "If the answer is not in the documents, say so explicitly.\n" +
      "Cite sources with [Source: filename].\n\n" +
      "{documents}",
  };

  assemble(params: {
    template: string;
    sources: Array<{ text: string; metadata: Record<string, string> }>;
  }): string {
    const docBlocks = params.sources.map((doc) => {
      const src = doc.metadata.source ?? "unknown";
      return `[Source: ${src}]\n${doc.text}`;
    });
    const documents =
      docBlocks.length > 0
        ? docBlocks.join("\n\n")
        : "(no documents retrieved)";
    return this.templates[params.template].replace("{documents}", documents);
  }
}

// ---------------------------------------------------------------------------
// MemoryManager — YOUR logic, no framework dependency
// ---------------------------------------------------------------------------

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export class MemoryManager {
  private history: ChatMessage[] = [];
  private readonly maxTurns: number;

  constructor(maxTurns = 5) {
    this.maxTurns = maxTurns;
  }

  getMessages(systemPrompt: string, userMessage: string): ChatMessage[] {
    const recent = this.history.slice(-(this.maxTurns * 2));
    return [
      { role: "system", content: systemPrompt },
      ...recent,
      { role: "user", content: userMessage },
    ];
  }

  record(userMessage: string, assistantReply: string): void {
    this.history.push({ role: "user", content: userMessage });
    this.history.push({ role: "assistant", content: assistantReply });
  }
}

// ---------------------------------------------------------------------------
// TokenTracker — YOUR logic, no framework dependency
// ---------------------------------------------------------------------------

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  total: number;
}

export class TokenTracker {
  private promptTokens = 0;
  private completionTokens = 0;
  private calls = 0;

  record(usage: { prompt_tokens?: number; completion_tokens?: number }): void {
    this.promptTokens += usage.prompt_tokens ?? 0;
    this.completionTokens += usage.completion_tokens ?? 0;
    this.calls++;
  }

  get totals(): TokenUsage {
    return {
      promptTokens: this.promptTokens,
      completionTokens: this.completionTokens,
      total: this.promptTokens + this.completionTokens,
    };
  }

  get callCount(): number {
    return this.calls;
  }
}

// ===========================================================================
// HybridRAGAgent — the main class
// ===========================================================================

export class HybridRAGAgent {
  private readonly openaiClient: OpenAI;
  private readonly useLangChain: boolean;

  // Your code: the important parts
  private readonly contextAssembler = new ContextAssembler();
  private readonly memoryManager = new MemoryManager(5);
  private readonly tokenTracker = new TokenTracker();

  // Populated by ingest()
  private vectorStore: VectorStoreProtocol | null = null;
  private _ingestedCount = 0;

  /**
   * @param openaiClient  Injected OpenAI client (testable and swappable).
   * @param useLangChain  Use LangChain.js for ingestion when available.
   */
  constructor(openaiClient: OpenAI, useLangChain = true) {
    this.openaiClient = openaiClient;
    this.useLangChain = useLangChain && LANGCHAIN_AVAILABLE;
  }

  // ------------------------------------------------------------------
  // Ingestion
  // ------------------------------------------------------------------

  /**
   * Embed and store inline demo documents.
   * In a real implementation this would load from a directory.
   */
  async ingest(
    docs: Array<{ text: string; source: string }> = INLINE_DOCS
  ): Promise<number> {
    const mapped = docs.map((d) => ({
      text: d.text,
      metadata: { source: d.source },
    }));

    if (this.useLangChain) {
      this.vectorStore = await LangChainVectorStore.fromDocuments(mapped);
    } else {
      const store = new SimpleVectorStore(this.openaiClient);
      await store.addDocuments(mapped);
      this.vectorStore = store;
    }

    this._ingestedCount = docs.length;
    return this._ingestedCount;
  }

  // ------------------------------------------------------------------
  // Query
  // ------------------------------------------------------------------

  /**
   * Run the full RAG pipeline for `question`.
   * Your code controls every step — the framework is just a tool.
   */
  async query(question: string): Promise<HybridResult> {
    if (!this.vectorStore) {
      throw new Error("Call ingest() before query().");
    }

    // 1. Retrieve (framework or custom does this well)
    const t0Ret = performance.now();
    const retrieved = await this.vectorStore.similaritySearch(question, 5);
    const retrievalMs = performance.now() - t0Ret;

    // 2. Your context assembly logic (you control quality)
    const context = this.contextAssembler.assemble({
      template: "rag_query",
      sources: retrieved,
    });

    // 3. Your memory management (you control budget)
    const messages = this.memoryManager.getMessages(context, question);

    // 4. LLM call via Vercel AI SDK (unified provider abstraction)
    const t0Gen = performance.now();
    const { text, usage } = await generateText({
      model: openai(LLM_MODEL),
      messages,
    });
    const generationMs = performance.now() - t0Gen;

    // 5. Your token tracking (you control cost visibility)
    this.tokenTracker.record({
      prompt_tokens: usage?.promptTokens,
      completion_tokens: usage?.completionTokens,
    });

    // Record this turn in memory
    this.memoryManager.record(question, text);

    const sources = [
      ...new Set(retrieved.map((r) => r.metadata.source ?? "")),
    ];

    return {
      answer: text,
      sources,
      tokens: (usage?.promptTokens ?? 0) + (usage?.completionTokens ?? 0),
      retrievalMs,
      generationMs,
    };
  }

  // ------------------------------------------------------------------
  // Vector store swap
  // ------------------------------------------------------------------

  /**
   * Replace the vector store backend with zero logic changes to the agent.
   * The agent loop, context assembly, memory, and token tracking are unaffected.
   */
  swapVectorStore(newStore: VectorStoreProtocol): void {
    this.vectorStore = newStore;
  }

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------

  get tokenSummary(): TokenUsage {
    return this.tokenTracker.totals;
  }

  get ingestedCount(): number {
    return this._ingestedCount;
  }

  get vectorBackend(): string {
    return this.vectorStore?.backendName ?? "none";
  }
}

// ---------------------------------------------------------------------------
// Utility: cosine similarity
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

async function demo(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  console.log("\n" + "=".repeat(65));
  console.log("  HYBRID RAG AGENT DEMO (TypeScript)");
  console.log("=".repeat(65));

  // Phase 1: Ingest
  console.log(
    `\n[1] Ingesting ${INLINE_DOCS.length} inline documents` +
      (LANGCHAIN_AVAILABLE ? " (LangChain.js)" : " (SimpleVectorStore)") +
      "..."
  );
  const agent = new HybridRAGAgent(client, true);
  const n = await agent.ingest();
  console.log(`    Ingested ${n} documents. Backend: ${agent.vectorBackend}`);

  // Phase 2: Query
  const question = "What are the four phases of a RAG pipeline?";
  console.log(`\n[2] Query: '${question}'`);
  const result = await agent.query(question);
  console.log(`    Answer: ${result.answer.slice(0, 120)}...`);
  console.log(`    Sources: ${result.sources.join(", ")}`);
  console.log(
    `    Tokens: ${result.tokens}  |  Retrieval: ${result.retrievalMs.toFixed(0)}ms  |  Generation: ${result.generationMs.toFixed(0)}ms`
  );

  // Phase 3: Swap vector store
  console.log("\n[3] Swapping to SimpleVectorStore...");
  const customStore = new SimpleVectorStore(client);
  await customStore.addDocuments(
    INLINE_DOCS.map((d) => ({ text: d.text, metadata: { source: d.source } }))
  );
  agent.swapVectorStore(customStore);
  console.log(`    Backend is now: ${agent.vectorBackend}`);

  console.log(`\n[4] Same query after swap: '${question}'`);
  const result2 = await agent.query(question);
  console.log(`    Answer: ${result2.answer.slice(0, 120)}...`);
  console.log("    Agent logic unchanged — only storage backend changed.");

  console.log(`\nTotal tokens this session: ${agent.tokenSummary.total}`);
  console.log("=".repeat(65));
}

// Run demo when executed directly
demo().catch((err) => {
  console.error(err);
  process.exit(1);
});
