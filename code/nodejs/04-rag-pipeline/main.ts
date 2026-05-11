/**
 * Entry point for 04-rag-pipeline demos.
 *
 * Run individual demos or the full showcase:
 *   npx tsx main.ts
 */

import OpenAI from "openai";
import { TokenTracker } from "./token_tracker.js";
import { ConversationSummarizer } from "./conversation_summarizer.js";
import { BranchManager } from "./branch_manager.js";
import { buildSimilarityMatrix, printSimilarityMatrix, detectOutliers } from "./embedding_visualizer.js";
import { InMemoryDocumentStore, EmbeddingSyncManager } from "./embedding_sync.js";
import { chunkBySentence } from "./document_chunker.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "demo-key" });

async function demoTokenTracker(): Promise<void> {
  console.log("=== Token Tracker Demo ===");
  const tracker = new TokenTracker({ budgetUsd: 10.0 });
  tracker.record("gpt-4o", "embedding", { promptTokens: 512, completionTokens: 0, totalTokens: 512 });
  tracker.record("gpt-4o-mini", "answer-gen", { promptTokens: 1024, completionTokens: 256, totalTokens: 1280 });
  tracker.printReport();
}

async function demoDocumentChunker(): Promise<void> {
  console.log("\n=== Document Chunker Demo ===");
  const text =
    "Large language models are trained on vast text corpora. " +
    "They learn statistical patterns over billions of tokens. " +
    "RAG augments LLMs with retrieval to reduce hallucinations. " +
    "Vector databases enable fast nearest-neighbour search. " +
    "Chunking strategies affect retrieval quality significantly.";
  const chunks = chunkBySentence(text, "intro-doc", 30);
  console.log(`Created ${chunks.length} chunks:`);
  chunks.forEach((c) => console.log(`  [${c.index}] ${c.tokenCount} tokens: "${c.text.slice(0, 60)}"`));
}

async function demoEmbeddingVisualizer(): Promise<void> {
  console.log("\n=== Embedding Visualizer Demo ===");
  const items = [
    { label: "cat",   vector: [0.9, 0.1, 0.05] },
    { label: "dog",   vector: [0.88, 0.12, 0.06] },
    { label: "car",   vector: [0.05, 0.9, 0.1] },
    { label: "bike",  vector: [0.06, 0.88, 0.08] },
    { label: "apple", vector: [0.05, 0.05, 0.9] },
  ];
  printSimilarityMatrix(buildSimilarityMatrix(items));
  const outliers = detectOutliers(items);
  console.log("\nOutliers:", outliers.filter((o) => o.isOutlier).map((o) => o.label));
}

async function demoEmbeddingSync(): Promise<void> {
  console.log("\n=== Embedding Sync Demo ===");
  const store = new InMemoryDocumentStore();
  store.upsert("doc1", "TypeScript is a typed superset of JavaScript.");
  store.upsert("doc2", "Vector search uses approximate nearest neighbours.");

  const indexed = new Map<string, string>();
  const adapter = {
    async upsert(id: string, content: string) { indexed.set(id, content); },
    async delete(id: string) { indexed.delete(id); },
    async hasId(id: string) { return indexed.has(id); },
    async allIds() { return Array.from(indexed.keys()); },
  };

  const manager = new EmbeddingSyncManager(store, adapter);
  const report = await manager.sync();
  console.log("Sync report:", report, "Health:", manager.health());
}

async function demoBranchManager(): Promise<void> {
  console.log("\n=== Branch Manager Demo ===");
  const manager = new BranchManager(client, [
    { role: "user", content: "Should I invest in real estate or stocks?" },
  ]);
  const branch = manager.createBranch({ label: "real-estate", inheritHistory: true });
  manager.addMessage(branch.id, { role: "user", content: "Tell me more about real estate." });
  manager.addMessage(branch.id, { role: "assistant", content: "Real estate provides stable cash flow..." });
  console.log("Branches:", manager.list());
}

async function main(): Promise<void> {
  await demoTokenTracker();
  await demoDocumentChunker();
  await demoEmbeddingVisualizer();
  await demoEmbeddingSync();
  await demoBranchManager();
  console.log("\nAll RAG pipeline demos complete.");
}

main().catch(console.error);
