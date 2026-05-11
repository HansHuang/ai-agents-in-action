/**
 * Step-by-step extraction from framework to custom code.
 *
 * Four stages, each more framework-independent:
 *   Step 1 — FRAMEWORK-OWNED: framework handles everything
 *   Step 2 — EXTRACT RETRIEVAL: custom retrieval, framework for loading
 *   Step 3 — EXTRACT LOADING: custom loading + ingestion
 *   Step 4 — PURE CUSTOM: zero framework imports
 *
 * See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o-mini";

export interface ExtractionStage {
  step: number;
  name: string;
  description: string;
  frameworkDependencies: number;  // simulated count
  customLines: number;
  answer: string;
  durationMs: number;
}

export interface ExtractionReport {
  query: string;
  stages: ExtractionStage[];
  recommendation: string;
}

// Simulated document corpus
const DOCUMENTS = [
  "RAG reduces hallucinations by grounding answers in retrieved documents.",
  "Vector databases use approximate nearest-neighbour algorithms for fast search.",
  "Embeddings are dense numerical representations of semantic meaning.",
  "LangChain provides loaders, splitters, and vector store integrations.",
  "Custom code gives full control but requires implementing patterns from scratch.",
];

// ---------------------------------------------------------------------------
// Stage implementations (all use the same underlying LLM call)
// ---------------------------------------------------------------------------

async function answerWithContext(query: string, docs: string[], client: OpenAI): Promise<string> {
  const context = docs.slice(0, 3).join("\n");
  const resp = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "Answer using only the provided context. Be concise." },
      { role: "user", content: `Context:\n${context}\n\nQuestion: ${query}` },
    ],
    temperature: 0,
    max_tokens: 150,
  });
  return resp.choices[0].message.content?.trim() ?? "";
}

function simpleSearch(query: string, docs: string[]): string[] {
  const queryWords = new Set(query.toLowerCase().split(/\s+/));
  return docs
    .map((doc) => ({
      doc,
      score: doc.toLowerCase().split(/\s+/).filter((w) => queryWords.has(w)).length,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((r) => r.doc);
}

/**
 * Run all four extraction stages and return a comparison report.
 */
export async function runExtractionComparison(
  query: string,
  client: OpenAI
): Promise<ExtractionReport> {
  const stages: ExtractionStage[] = [];

  const stageConfigs = [
    {
      step: 1,
      name: "Framework-Owned",
      description: "Framework handles loading, retrieval, and generation",
      frameworkDeps: 5,
      customLines: 5,
      docs: DOCUMENTS,
    },
    {
      step: 2,
      name: "Extract Retrieval",
      description: "Custom keyword retrieval, framework for document loading only",
      frameworkDeps: 3,
      customLines: 15,
      docs: simpleSearch(query, DOCUMENTS),
    },
    {
      step: 3,
      name: "Extract Loading",
      description: "Custom loading + retrieval, no framework chains",
      frameworkDeps: 1,
      customLines: 30,
      docs: simpleSearch(query, DOCUMENTS),
    },
    {
      step: 4,
      name: "Pure Custom",
      description: "Zero framework imports — complete control",
      frameworkDeps: 0,
      customLines: 50,
      docs: simpleSearch(query, DOCUMENTS),
    },
  ];

  for (const cfg of stageConfigs) {
    const start = Date.now();
    const answer = await answerWithContext(query, cfg.docs, client);
    stages.push({
      step: cfg.step,
      name: cfg.name,
      description: cfg.description,
      frameworkDependencies: cfg.frameworkDeps,
      customLines: cfg.customLines,
      answer,
      durationMs: Date.now() - start,
    });
  }

  return {
    query,
    stages,
    recommendation:
      "Start at Step 1 for rapid prototyping. Extract to Step 4 only when you need full control, observability, or to eliminate a heavy dependency.",
  };
}

/** Print the extraction report. */
export function printExtractionReport(report: ExtractionReport): void {
  console.log(`\n=== Extraction Report: "${report.query}" ===\n`);
  console.log(
    "Step".padEnd(4) + "Name".padEnd(25) + "FW Deps".padStart(8) + "Custom LOC".padStart(12) + "Time".padStart(8)
  );
  console.log("-".repeat(60));
  for (const s of report.stages) {
    console.log(
      `${s.step}`.padEnd(4) +
      s.name.padEnd(25) +
      `${s.frameworkDependencies}`.padStart(8) +
      `${s.customLines}`.padStart(12) +
      `${s.durationMs}ms`.padStart(8)
    );
  }
  console.log(`\nRecommendation: ${report.recommendation}`);
}
