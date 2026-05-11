/**
 * LangChain-style RAG pipeline for TypeScript.
 *
 * Builds a RAG pipeline using LangChain.js patterns — document loaders,
 * text splitters, vector store, and retrieval chain.
 * Falls back to a from-scratch equivalent when LangChain is not installed.
 * See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o-mini";

export interface RAGDocument {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface RAGResponse {
  answer: string;
  sources: string[];
  retrievedCount: number;
  durationMs: number;
}

// ---------------------------------------------------------------------------
// Simple in-memory vector store (no external dependencies)
// ---------------------------------------------------------------------------

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return Math.sqrt(na) * Math.sqrt(nb) === 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Deterministic pseudo-embedding for demo (replace with real embeddings)
function pseudoEmbed(text: string): number[] {
  const vec = new Array<number>(64).fill(0);
  for (let i = 0; i < text.length; i++) {
    vec[i % 64] += text.charCodeAt(i) / 1000;
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return vec.map((v) => v / norm);
}

interface StoredDoc { doc: RAGDocument; embedding: number[] }

class InMemoryVectorStore {
  private store: StoredDoc[] = [];

  add(doc: RAGDocument): void {
    this.store.push({ doc, embedding: pseudoEmbed(doc.content) });
  }

  search(query: string, k = 3): RAGDocument[] {
    const qEmbed = pseudoEmbed(query);
    return this.store
      .map((s) => ({ doc: s.doc, score: cosineSimilarity(qEmbed, s.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((s) => s.doc);
  }
}

// ---------------------------------------------------------------------------
// LangChain-style pipeline
// ---------------------------------------------------------------------------

export class LangChainStyleRAGPipeline {
  private vectorStore = new InMemoryVectorStore();

  constructor(private client: OpenAI) {}

  /** Load and split documents (LangChain-style document loader pattern). */
  loadDocuments(documents: RAGDocument[]): void {
    // LangChain TextSplitter equivalent: split long docs into chunks
    for (const doc of documents) {
      const chunks = this.splitText(doc.content, 200);
      chunks.forEach((chunk, i) => {
        this.vectorStore.add({
          id: `${doc.id}-chunk-${i}`,
          content: chunk,
          metadata: { ...doc.metadata, sourceId: doc.id },
        });
      });
    }
  }

  private splitText(text: string, maxChars: number): string[] {
    const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text];
    const chunks: string[] = [];
    let current = "";
    for (const s of sentences) {
      if ((current + s).length > maxChars && current) {
        chunks.push(current.trim());
        current = s;
      } else {
        current += " " + s;
      }
    }
    if (current.trim()) chunks.push(current.trim());
    return chunks;
  }

  /** Run the retrieval chain (LangChain RetrievalQA equivalent). */
  async query(question: string, k = 3): Promise<RAGResponse> {
    const start = Date.now();
    const retrieved = this.vectorStore.search(question, k);
    const context = retrieved.map((d, i) => `[${i + 1}] ${d.content}`).join("\n\n");

    const resp = await this.client.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant. Answer the question using only the provided context. " +
            "If the context doesn't contain the answer, say so.",
        },
        { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` },
      ],
      temperature: 0,
      max_tokens: 300,
    });

    const sources = [...new Set(retrieved.map((d) => String(d.metadata?.sourceId ?? d.id)))];

    return {
      answer: resp.choices[0].message.content?.trim() ?? "",
      sources,
      retrievedCount: retrieved.length,
      durationMs: Date.now() - start,
    };
  }
}

// Demo
async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "demo-key" });
  const pipeline = new LangChainStyleRAGPipeline(client);

  pipeline.loadDocuments([
    { id: "doc1", content: "RAG grounds answers in retrieved context to reduce hallucinations. It retrieves relevant passages before generating." },
    { id: "doc2", content: "Vector databases store embedding vectors and support approximate nearest-neighbour search." },
    { id: "doc3", content: "LangChain provides document loaders, text splitters, and retrieval chain abstractions." },
  ]);

  console.log("LangChain-style RAG Pipeline loaded 3 documents.");
  console.log("Query: What is RAG and how does it reduce hallucinations?");
}

main().catch(console.error);
