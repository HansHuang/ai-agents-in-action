/**
 * Tests for nodejs/08-evaluation — RetrievalEvaluator, EvaluationDashboard
 *
 * No LLM calls — uses mock retrievers.
 * Run: ./node_modules/.bin/tsx --test test_evaluation.ts
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { RetrievalEvaluator } from "./agent_evaluator.js";
import type { RetrievalTestCase } from "./agent_evaluator.js";
import { EvaluationDashboard } from "./evaluation_dashboard.js";

// ---------------------------------------------------------------------------
// RetrievalEvaluator
// ---------------------------------------------------------------------------

describe("RetrievalEvaluator", () => {
  // Mock retriever that always returns doc0 first
  const perfectRetriever = {
    search: async (_query: string, k: number) =>
      Array.from({ length: k }, (_, i) => ({ id: `doc${i}` })),
  };

  const testCases: RetrievalTestCase[] = [
    { query: "capital of France", relevant_doc_ids: ["doc0"], min_results_expected: 1 },
    { query: "machine learning", relevant_doc_ids: ["doc0"], min_results_expected: 1 },
  ];

  it("perfect retriever gets hit rate of 1.0", async () => {
    const evaluator = new RetrievalEvaluator(perfectRetriever, testCases);
    const report = await evaluator.evaluate(3);
    assert.ok(report.hit_rate >= 0.9, `Expected high hit rate, got ${report.hit_rate}`);
  });

  it("evaluate returns valid metrics structure", async () => {
    const evaluator = new RetrievalEvaluator(perfectRetriever, testCases);
    const report = await evaluator.evaluate(3);
    assert.ok(typeof report.hit_rate === "number");
    assert.ok(typeof report.mrr === "number");
    assert.ok(typeof report.total_queries === "number");
    assert.ok(Array.isArray(report.per_query));
  });
});

// ---------------------------------------------------------------------------
// EvaluationDashboard
// ---------------------------------------------------------------------------

describe("EvaluationDashboard", () => {
  // The EvaluationDashboard reads aggregate.hitRate, aggregate.successRate etc.
  // We provide a mock matching the shape it actually accesses.
  const mockReport = {
    retrieval: {
      aggregate: { hitRate: 0.85, precisionAtK: 0.70, recallAtK: 0.75, mrr: 0.80 },
    },
    generation: {
      aggregate: { avgFaithfulness: 0.82, avgRelevance: 0.78, avgCompleteness: 0.75, rulePassRate: 0.90 },
    },
    endToEnd: {
      aggregate: { successRate: 0.80, avgTurns: 3.2 },
    },
  };

  it("EvaluationDashboard can be instantiated", () => {
    const dashboard = new EvaluationDashboard();
    assert.ok(dashboard instanceof EvaluationDashboard);
  });

  it("printFullReport does not throw with valid mock data", () => {
    const dashboard = new EvaluationDashboard();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    assert.doesNotThrow(() => dashboard.printFullReport(mockReport as any));
  });
});

