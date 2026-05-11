/**
 * RAG pipeline evaluation framework.
 *
 * Measures retrieval quality (hit rate, MRR, precision@k, recall@k) and
 * generation quality (faithfulness, LLM-as-judge).
 * See: docs/08-evaluation-and-guardrails/01-evaluating-agents.md
 */

import OpenAI from "openai";

export interface RetrievalTestCase {
  query: string;
  relevantDocIds: string[];
}

export interface RetrievedDoc {
  id: string;
  text: string;
}

export interface RetrievalMetrics {
  hitRate: number;
  mrr: number;
  precisionAtK: number;
  recallAtK: number;
}

export interface GenerationTestCase {
  question: string;
  context: string;
  answer: string;
  groundTruth?: string;
}

export interface GenerationMetrics {
  faithfulnessScore: number; // 0–1 (LLM judge)
  relevanceScore: number;    // 0–1 (LLM judge)
  judgeRationale: string;
}

// ---------------------------------------------------------------------------
// Retrieval quality
// ---------------------------------------------------------------------------

/**
 * Evaluate retrieval quality given test cases and a retrieval function.
 */
export async function evaluateRetrieval(
  testCases: RetrievalTestCase[],
  retrieveFn: (query: string, topK: number) => Promise<RetrievedDoc[]>,
  topK = 5
): Promise<RetrievalMetrics> {
  let hitRate = 0, mrr = 0, precision = 0, recall = 0;

  for (const tc of testCases) {
    const results = await retrieveFn(tc.query, topK);
    const ids = results.map((r) => r.id);

    const hit = ids.some((id) => tc.relevantDocIds.includes(id));
    if (hit) hitRate++;

    const firstRelevantRank = ids.findIndex((id) => tc.relevantDocIds.includes(id));
    if (firstRelevantRank !== -1) mrr += 1 / (firstRelevantRank + 1);

    const relevantRetrieved = ids.filter((id) => tc.relevantDocIds.includes(id)).length;
    precision += relevantRetrieved / topK;
    recall += tc.relevantDocIds.length > 0 ? relevantRetrieved / tc.relevantDocIds.length : 0;
  }

  const n = testCases.length || 1;
  return {
    hitRate: hitRate / n,
    mrr: mrr / n,
    precisionAtK: precision / n,
    recallAtK: recall / n,
  };
}

// ---------------------------------------------------------------------------
// Generation quality
// ---------------------------------------------------------------------------

const JUDGE_PROMPT = `Evaluate the answer on two dimensions:
1. Faithfulness (0-10): Is the answer supported by the provided context?
2. Relevance (0-10): Does the answer address the question?

Respond as JSON: {"faithfulness": <0-10>, "relevance": <0-10>, "rationale": "<brief explanation>"}`;

/**
 * LLM-as-judge evaluation of answer quality.
 */
export async function evaluateGeneration(
  testCase: GenerationTestCase,
  client: OpenAI,
  model = "gpt-4o-mini"
): Promise<GenerationMetrics> {
  const userMsg = `Context: ${testCase.context}\n\nQuestion: ${testCase.question}\n\nAnswer: ${testCase.answer}`;

  const resp = await client.chat.completions.create({
    model,
    messages: [
      { role: "system", content: JUDGE_PROMPT },
      { role: "user", content: userMsg },
    ],
    response_format: { type: "json_object" },
    temperature: 0,
  });

  try {
    const parsed = JSON.parse(resp.choices[0].message.content ?? "{}") as {
      faithfulness: number;
      relevance: number;
      rationale: string;
    };
    return {
      faithfulnessScore: parsed.faithfulness / 10,
      relevanceScore: parsed.relevance / 10,
      judgeRationale: parsed.rationale ?? "",
    };
  } catch {
    return { faithfulnessScore: 0, relevanceScore: 0, judgeRationale: "Parse error" };
  }
}

/** Print retrieval metrics to stdout. */
export function printRetrievalMetrics(m: RetrievalMetrics): void {
  console.log("\nRetrieval Evaluation:");
  console.log(`  Hit Rate    : ${(m.hitRate * 100).toFixed(1)}%`);
  console.log(`  MRR         : ${m.mrr.toFixed(3)}`);
  console.log(`  Precision@k : ${(m.precisionAtK * 100).toFixed(1)}%`);
  console.log(`  Recall@k    : ${(m.recallAtK * 100).toFixed(1)}%`);
}
