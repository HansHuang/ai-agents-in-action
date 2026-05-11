/**
 * Tests for nodejs/04-rag-pipeline — SimpleVectorStore, chunkByTokens, MemoryManager
 *
 * No LLM or embedding API calls — uses mock embeddings (random vectors).
 * Run: node --import tsx/esm --test test_rag.ts
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { SimpleVectorStore } from "./simple_vector_store.js";
import { chunkByTokens } from "./document_chunker.js";
import { MemoryManager } from "./memory_manager.js";

// ---------------------------------------------------------------------------
// SimpleVectorStore
// ---------------------------------------------------------------------------

describe("SimpleVectorStore", () => {
  function randVec(dim = 8): number[] {
    return Array.from({ length: dim }, () => Math.random() * 2 - 1);
  }

  it("adds documents and reports correct count", () => {
    const store = new SimpleVectorStore();
    store.add("First document", randVec());
    store.add("Second document", randVec());
    assert.equal(store.count(), 2);
  });

  it("searches and returns results sorted by score desc", () => {
    const store = new SimpleVectorStore();
    const queryVec = [1, 0, 0, 0, 0, 0, 0, 0];
    store.add("Alpha", queryVec);   // perfect match (cosine = 1)
    store.add("Beta", [-1, 0, 0, 0, 0, 0, 0, 0]);  // opposite
    const results = store.search(queryVec, 2);
    assert.ok(results.length > 0);
    assert.equal(results[0].text, "Alpha", "Best match should be first");
    // Scores in descending order
    for (let i = 1; i < results.length; i++) {
      assert.ok(results[i - 1].score >= results[i].score, "Results should be sorted desc");
    }
  });

  it("search result includes text and score", () => {
    const store = new SimpleVectorStore();
    const vec = randVec();
    store.add("My text", vec, { tag: "test" });
    const results = store.search(vec, 1);
    assert.ok(results.length === 1);
    assert.equal(results[0].text, "My text");
    assert.ok(typeof results[0].score === "number");
  });

  it("delete removes document", () => {
    const store = new SimpleVectorStore();
    const id = store.add("To delete", randVec());
    assert.equal(store.count(), 1);
    store.delete(id);
    assert.equal(store.count(), 0);
  });
});

// ---------------------------------------------------------------------------
// Document Chunker (chunkByTokens function)
// ---------------------------------------------------------------------------

describe("DocumentChunker", () => {
  const SAMPLE = "Word ".repeat(200).trim(); // ~200 words

  it("chunks long text into multiple segments", () => {
    const chunks = chunkByTokens(SAMPLE, "src", { chunkSize: 50, chunkOverlap: 10 });
    assert.ok(chunks.length > 1, `Expected > 1 chunk, got ${chunks.length}`);
  });

  it("all chunks are non-empty strings", () => {
    const chunks = chunkByTokens(SAMPLE, "src", { chunkSize: 50, chunkOverlap: 10 });
    for (const c of chunks) {
      assert.ok(typeof c.text === "string" && c.text.trim().length > 0);
    }
  });

  it("chunking short text returns single chunk", () => {
    const chunks = chunkByTokens("Hello world", "src", { chunkSize: 500, chunkOverlap: 50 });
    assert.equal(chunks.length, 1);
  });
});

// ---------------------------------------------------------------------------
// MemoryManager
// ---------------------------------------------------------------------------

describe("MemoryManager", () => {
  it("can be instantiated and add messages", () => {
    // MemoryManager is a conversation context manager, not key-value store
    // It requires an OpenAI client but we can mock it
    process.env.OPENAI_API_KEY ??= "test-key-not-used";
    const mm = new MemoryManager({ maxTokens: 1000 });
    mm.addUserMessage("user likes TypeScript");
    mm.addAssistantMessage("That's great!");
    assert.ok(mm.messages.length >= 2);
  });

  it("tokenCount returns a number", () => {
    process.env.OPENAI_API_KEY ??= "test-key-not-used";
    const mm = new MemoryManager();
    mm.addUserMessage("Hello world");
    assert.ok(typeof mm.tokenCount() === "number");
    assert.ok(mm.tokenCount() > 0);
  });
});
