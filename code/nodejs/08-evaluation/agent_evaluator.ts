/**
 * agent_evaluator.ts
 * ==================
 * Three-level agent evaluation framework — TypeScript port.
 *
 * Evaluates AI agents across three dimensions:
 *   1. Retrieval   — Hit Rate, Precision@K, Recall@K, MRR, NDCG@K
 *   2. Generation  — Rule-based checks + LLM-as-judge (faithfulness, relevance, completeness)
 *   3. End-to-End  — Task Success Rate across multi-turn scenarios
 *
 * Also provides a ContinuousEvaluationPipeline with regression detection.
 *
 * @see docs/08-evaluation-and-guardrails/01-evaluating-agents.md
 */

import OpenAI from "openai";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Runtime validators (Zod schemas)
// ---------------------------------------------------------------------------

const RetrievalTestCaseSchema = z.object({
  query: z.string().min(1),
  relevant_doc_ids: z.array(z.string()),
  partially_relevant_doc_ids: z.array(z.string()).optional(),
  irrelevant_doc_ids: z.array(z.string()).optional(),
  min_results_expected: z.number().int().nonnegative().default(1),
});

const GenerationTestCaseSchema = z.object({
  query: z.string().min(1),
  expected_answer_contains: z.array(z.string()).optional(),
  expected_answer_not_contains: z.array(z.string()).optional(),
  expected_sources: z.array(z.string()).optional(),
  min_answer_length: z.number().int().nonnegative().default(20),
  max_answer_length: z.number().int().positive().default(2000),
  reference_answer: z.string().optional(),
  evaluation_criteria: z.string().optional(),
});

const EndToEndTestCaseSchema = z.object({
  scenario: z.string().min(1),
  user_messages: z.array(z.string()).min(1),
  expected_outcome: z.enum(["resolved", "escalated", "information_provided"]),
  expected_tools_called: z.array(z.string()).optional(),
  max_turns_expected: z.number().int().positive().default(5),
  forbidden_behaviors: z.array(z.string()).optional(),
});

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RetrievalTestCase = z.infer<typeof RetrievalTestCaseSchema>;
export type GenerationTestCase = z.infer<typeof GenerationTestCaseSchema>;
export type EndToEndTestCase = z.infer<typeof EndToEndTestCaseSchema>;

export interface RetrievalMetrics {
  query: string;
  hit: number;
  precision_at_k: number;
  recall_at_k: number;
  reciprocal_rank: number;
  ndcg_at_k: number;
  relevant_found: number;
  relevant_total: number;
  retrieved_ids: string[];
}

export interface RetrievalReport {
  hit_rate: number;
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  total_queries: number;
  queries_with_zero_results: number;
  per_query: RetrievalMetrics[];
}

export interface JudgeResult {
  dimension: string;
  passed: boolean;
  score: number;
  issues: string[];
  explanation: string;
}

export interface RuleChecks {
  contains_required?: boolean;
  missing_required?: string[];
  avoids_forbidden?: boolean;
  found_forbidden?: string[];
  length_ok: boolean;
  sources_cited?: boolean;
  all_passed: boolean;
}

export interface GenerationQueryResult {
  query: string;
  response: string;
  rule_checks: RuleChecks;
  judge_checks: Record<string, JudgeResult>;
  overall_pass: boolean;
}

export interface GenerationReport {
  overall_pass_rate: number;
  contains_required_rate: number;
  avoids_forbidden_rate: number;
  faithfulness_pass_rate: number | null;
  relevance_pass_rate: number | null;
  completeness_pass_rate: number | null;
  total_queries: number;
  per_query: GenerationQueryResult[];
}

export interface ScenarioResult {
  scenario: string;
  outcome: string;
  expected_outcome: string;
  turns_taken: number;
  tools_called: string[];
  expected_tools: string[] | undefined;
  success: boolean;
}

export interface EndToEndReport {
  task_success_rate: number;
  avg_turns_to_resolution: number;
  total_scenarios: number;
  per_scenario: ScenarioResult[];
}

export interface FullEvaluationReport {
  retrieval: RetrievalReport;
  generation: GenerationReport;
  end_to_end: EndToEndReport;
}

export interface RegressionCheck {
  has_regressions: boolean;
  regressions: string[];
  baseline: FullEvaluationReport;
  current: FullEvaluationReport;
}

// ---------------------------------------------------------------------------
// Interfaces for injected dependencies
// ---------------------------------------------------------------------------

export interface Retriever {
  search(query: string, k: number): Promise<Array<{ id: string }>>;
}

export interface AgentResponse {
  content: string;
  metadata: {
    retrieved_documents?: string[];
  };
  tool_calls?: Array<{ name: string }>;
}

export interface Agent {
  run(
    query: string,
    options?: { conversation_history?: Array<{ role: string; content: string }> }
  ): Promise<AgentResponse>;
}

// ===========================================================================
// LEVEL 1 — RETRIEVAL EVALUATION
// ===========================================================================

/**
 * Evaluate retrieval quality using standard information-retrieval metrics.
 *
 * @example
 * const evaluator = new RetrievalEvaluator(retriever, testCases);
 * const report = await evaluator.evaluate(5);
 */
export class RetrievalEvaluator {
  constructor(
    private readonly retriever: Retriever,
    private readonly testCases: RetrievalTestCase[]
  ) {}

  /** Evaluate retrieval across all test cases at depth {@link k}. */
  async evaluate(k = 5): Promise<RetrievalReport> {
    const results: RetrievalMetrics[] = [];
    for (const test of this.testCases) {
      const retrieved = await this.retriever.search(test.query, k);
      const retrievedIds = retrieved.map((d) => d.id);
      results.push(this._calculateMetrics(test, retrievedIds, k));
    }
    return this._aggregate(results);
  }

  _calculateMetrics(
    test: RetrievalTestCase,
    retrievedIds: string[],
    k: number
  ): RetrievalMetrics {
    const relevant = new Set(test.relevant_doc_ids);
    const retrievedSet = new Set(retrievedIds.slice(0, k));

    // Hit: at least one relevant doc retrieved
    const hit = relevant.size > 0 && [...relevant].some((id) => retrievedSet.has(id)) ? 1 : 0;

    // Precision@K
    const truePositives = [...relevant].filter((id) => retrievedSet.has(id)).length;
    const precision = retrievedSet.size > 0 ? truePositives / retrievedSet.size : 0;

    // Recall@K
    const recall = relevant.size > 0 ? truePositives / relevant.size : 1.0;

    // MRR
    let reciprocal_rank = 0;
    for (let i = 0; i < retrievedIds.length; i++) {
      if (relevant.has(retrievedIds[i])) {
        reciprocal_rank = 1 / (i + 1);
        break;
      }
    }

    // NDCG@K
    const ndcg = this._calculateNdcg(test, retrievedIds, k);

    return {
      query: test.query,
      hit,
      precision_at_k: precision,
      recall_at_k: recall,
      reciprocal_rank,
      ndcg_at_k: ndcg,
      relevant_found: truePositives,
      relevant_total: relevant.size,
      retrieved_ids: retrievedIds,
    };
  }

  _calculateNdcg(test: RetrievalTestCase, retrievedIds: string[], k: number): number {
    const relevanceScores = new Map<string, number>();
    for (const id of test.relevant_doc_ids) relevanceScores.set(id, 2);
    for (const id of test.partially_relevant_doc_ids ?? []) {
      if (!relevanceScores.has(id)) relevanceScores.set(id, 1);
    }

    let dcg = 0;
    for (let i = 0; i < Math.min(retrievedIds.length, k); i++) {
      const rel = relevanceScores.get(retrievedIds[i]) ?? 0;
      dcg += rel / Math.log2(i + 2);
    }

    const idealRels = [...relevanceScores.values()].sort((a, b) => b - a).slice(0, k);
    let idcg = 0;
    for (let i = 0; i < idealRels.length; i++) {
      idcg += idealRels[i] / Math.log2(i + 2);
    }

    return idcg > 0 ? dcg / idcg : 0;
  }

  _aggregate(results: RetrievalMetrics[]): RetrievalReport {
    const n = results.length;
    const avg = (fn: (r: RetrievalMetrics) => number): number =>
      results.reduce((sum, r) => sum + fn(r), 0) / n;

    return {
      hit_rate: avg((r) => r.hit),
      precision_at_k: avg((r) => r.precision_at_k),
      recall_at_k: avg((r) => r.recall_at_k),
      mrr: avg((r) => r.reciprocal_rank),
      ndcg_at_k: avg((r) => r.ndcg_at_k),
      total_queries: n,
      queries_with_zero_results: results.filter(
        (r) => r.relevant_found === 0 && r.relevant_total > 0
      ).length,
      per_query: results,
    };
  }
}

// ===========================================================================
// LEVEL 2 — GENERATION EVALUATION
// ===========================================================================

/**
 * Use an LLM to evaluate agent response quality on faithfulness, relevance,
 * and completeness dimensions.
 *
 * Runs at temperature 0.1 for consistent, reproducible scoring.
 * Known biases: length preference; may be fooled by confident-sounding text.
 */
export class LLMJudge {
  static readonly FAITHFULNESS_PROMPT = `You are evaluating whether an AI response is faithful to the provided source documents.

Faithfulness means: Every factual claim in the response is directly supported by at least one of the source documents. The response does not add information not found in the sources.

SOURCE DOCUMENTS:
{source_documents}

RESPONSE TO EVALUATE:
{response}

Evaluate the response for faithfulness. Output JSON:
{
    "is_faithful": true/false,
    "score": 1-5,
    "unsupported_claims": ["claim1", "claim2"],
    "explanation": "Brief explanation of your evaluation"
}`;

  static readonly RELEVANCE_PROMPT = `You are evaluating whether an AI response is relevant to the user's question.

Relevance means: The response directly addresses what the user asked. It does not go off-topic or provide unnecessary information.

USER QUESTION:
{user_question}

RESPONSE TO EVALUATE:
{response}

Evaluate the response for relevance. Output JSON:
{
    "is_relevant": true/false,
    "score": 1-5,
    "off_topic_parts": ["part1", "part2"],
    "explanation": "Brief explanation"
}`;

  static readonly COMPLETENESS_PROMPT = `You are evaluating whether an AI response completely answers the user's question.

Completeness means: The response addresses ALL parts of the user's question. If the user asked multiple questions, all are answered. If the user asked for a comparison, both sides are covered.

USER QUESTION:
{user_question}

RESPONSE TO EVALUATE:
{response}

Evaluate the response for completeness. Output JSON:
{
    "is_complete": true/false,
    "score": 1-5,
    "missing_parts": ["unanswered question 1", "unanswered question 2"],
    "explanation": "Brief explanation"
}`;

  private readonly openai: OpenAI;

  constructor(private readonly model = "gpt-4o") {
    this.openai = new OpenAI();
  }

  /** Evaluate whether {@link response} is grounded in {@link sourceDocuments}. */
  async evaluateFaithfulness(
    response: string,
    sourceDocuments: string[]
  ): Promise<JudgeResult> {
    const sourcesText = sourceDocuments
      .map((doc, i) => `[Document ${i + 1}]\n${doc}`)
      .join("\n\n---\n\n")
      .slice(0, 10_000);

    const prompt = LLMJudge.FAITHFULNESS_PROMPT.replace(
      "{source_documents}",
      sourcesText
    ).replace("{response}", response.slice(0, 5_000));

    const result = await this._callJudge(prompt);
    return {
      dimension: "faithfulness",
      passed: result.is_faithful ?? false,
      score: result.score ?? 1,
      issues: result.unsupported_claims ?? [],
      explanation: result.explanation ?? "",
    };
  }

  /** Evaluate whether {@link response} is relevant to {@link userQuestion}. */
  async evaluateRelevance(response: string, userQuestion: string): Promise<JudgeResult> {
    const prompt = LLMJudge.RELEVANCE_PROMPT.replace(
      "{user_question}",
      userQuestion
    ).replace("{response}", response.slice(0, 5_000));

    const result = await this._callJudge(prompt);
    return {
      dimension: "relevance",
      passed: result.is_relevant ?? false,
      score: result.score ?? 1,
      issues: result.off_topic_parts ?? [],
      explanation: result.explanation ?? "",
    };
  }

  /** Evaluate whether {@link response} completely answers {@link userQuestion}. */
  async evaluateCompleteness(response: string, userQuestion: string): Promise<JudgeResult> {
    const prompt = LLMJudge.COMPLETENESS_PROMPT.replace(
      "{user_question}",
      userQuestion
    ).replace("{response}", response.slice(0, 5_000));

    const result = await this._callJudge(prompt);
    return {
      dimension: "completeness",
      passed: result.is_complete ?? false,
      score: result.score ?? 1,
      issues: result.missing_parts ?? [],
      explanation: result.explanation ?? "",
    };
  }

  /** @internal */
  async _callJudge(prompt: string): Promise<Record<string, unknown>> {
    const completion = await this.openai.chat.completions.create({
      model: this.model,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      temperature: 0.1,
    });
    const text = completion.choices[0]?.message?.content ?? "{}";
    return JSON.parse(text) as Record<string, unknown>;
  }
}

/**
 * Evaluate generation quality using rule-based checks and LLM-as-judge.
 */
export class GenerationEvaluator {
  private readonly judge: LLMJudge;

  constructor(
    private readonly agent: Agent,
    private readonly testCases: GenerationTestCase[],
    judge?: LLMJudge
  ) {
    this.judge = judge ?? new LLMJudge();
  }

  async evaluate(): Promise<GenerationReport> {
    const results: GenerationQueryResult[] = [];

    for (const test of this.testCases) {
      const response = await this.agent.run(test.query);
      const ruleChecks = this._ruleBasedChecks(test, response.content);
      const judgeChecks: Record<string, JudgeResult> = {};

      const sourceDocs = response.metadata.retrieved_documents ?? [];
      judgeChecks.faithfulness = await this.judge.evaluateFaithfulness(
        response.content,
        sourceDocs
      );
      judgeChecks.relevance = await this.judge.evaluateRelevance(
        response.content,
        test.query
      );
      judgeChecks.completeness = await this.judge.evaluateCompleteness(
        response.content,
        test.query
      );

      const overallPass =
        ruleChecks.all_passed &&
        Object.values(judgeChecks).every((j) => j.passed);

      results.push({ query: test.query, response: response.content, rule_checks: ruleChecks, judge_checks: judgeChecks, overall_pass: overallPass });
    }
    return this._aggregate(results);
  }

  _ruleBasedChecks(test: GenerationTestCase, response: string): RuleChecks {
    const lower = response.toLowerCase();
    const checks: Partial<RuleChecks> = {};

    if (test.expected_answer_contains) {
      const missing = test.expected_answer_contains.filter(
        (p) => !lower.includes(p.toLowerCase())
      );
      checks.contains_required = missing.length === 0;
      checks.missing_required = missing;
    }

    if (test.expected_answer_not_contains) {
      const found = test.expected_answer_not_contains.filter((p) =>
        lower.includes(p.toLowerCase())
      );
      checks.avoids_forbidden = found.length === 0;
      checks.found_forbidden = found;
    }

    checks.length_ok =
      response.length >= (test.min_answer_length ?? 20) &&
      response.length <= (test.max_answer_length ?? 2000);

    if (test.expected_sources) {
      checks.sources_cited = test.expected_sources.every((src) =>
        lower.includes(src.toLowerCase())
      );
    }

    const boolValues = [
      checks.contains_required,
      checks.avoids_forbidden,
      checks.length_ok,
      checks.sources_cited,
    ].filter((v): v is boolean => v !== undefined);

    checks.all_passed = boolValues.every(Boolean);
    return checks as RuleChecks;
  }

  _aggregate(results: GenerationQueryResult[]): GenerationReport {
    const n = results.length;
    const passRate = (fn: (r: GenerationQueryResult) => boolean): number =>
      results.filter(fn).length / n;

    const hasJudge = results.length > 0 && Object.keys(results[0].judge_checks).length > 0;

    return {
      overall_pass_rate: passRate((r) => r.overall_pass),
      contains_required_rate: passRate(
        (r) => r.rule_checks.contains_required !== false
      ),
      avoids_forbidden_rate: passRate(
        (r) => r.rule_checks.avoids_forbidden !== false
      ),
      faithfulness_pass_rate: hasJudge
        ? passRate((r) => r.judge_checks.faithfulness?.passed ?? true)
        : null,
      relevance_pass_rate: hasJudge
        ? passRate((r) => r.judge_checks.relevance?.passed ?? true)
        : null,
      completeness_pass_rate: hasJudge
        ? passRate((r) => r.judge_checks.completeness?.passed ?? true)
        : null,
      total_queries: n,
      per_query: results,
    };
  }
}

// ===========================================================================
// LEVEL 3 — END-TO-END EVALUATION
// ===========================================================================

const RESOLUTION_MARKERS = [
  "is there anything else",
  "i hope that helps",
  "your request has been",
  "i've completed",
  "would you like me to",
];

/**
 * Evaluate the agent on realistic multi-turn scenarios.
 */
export class EndToEndEvaluator {
  constructor(
    private readonly agent: Agent,
    private readonly testCases: EndToEndTestCase[]
  ) {}

  async evaluate(): Promise<EndToEndReport> {
    const results: ScenarioResult[] = [];

    for (const test of this.testCases) {
      const conversation: Array<{ role: string; content: string }> = [];
      const toolsCalled: string[] = [];
      let outcome = "unknown";
      let turnsTaken = 0;

      for (let i = 0; i < test.user_messages.length; i++) {
        const message = test.user_messages[i];
        const response = await this.agent.run(message, {
          conversation_history: conversation,
        });
        turnsTaken = i + 1;

        conversation.push({ role: "user", content: message });
        conversation.push({ role: "assistant", content: response.content });

        for (const tc of response.tool_calls ?? []) {
          toolsCalled.push(tc.name);
        }

        if (this._isResolved(response, test)) {
          outcome = "resolved";
          break;
        }
      }

      if (outcome === "unknown") {
        outcome =
          turnsTaken >= test.max_turns_expected ? "unresolved" : "incomplete";
      }

      results.push({
        scenario: test.scenario,
        outcome,
        expected_outcome: test.expected_outcome,
        turns_taken: turnsTaken,
        tools_called: toolsCalled,
        expected_tools: test.expected_tools_called,
        success: outcome === test.expected_outcome,
      });
    }

    const n = results.length;
    return {
      task_success_rate: results.filter((r) => r.success).length / n,
      avg_turns_to_resolution:
        results.reduce((s, r) => s + r.turns_taken, 0) / n,
      total_scenarios: n,
      per_scenario: results,
    };
  }

  _isResolved(response: AgentResponse, _test: EndToEndTestCase): boolean {
    const lower = response.content.toLowerCase();
    return RESOLUTION_MARKERS.some((m) => lower.includes(m));
  }
}

// ===========================================================================
// CONTINUOUS EVALUATION PIPELINE
// ===========================================================================

const REGRESSION_THRESHOLD = 0.05;

/**
 * Run evaluation on every change and alert on metric regressions (> 5% drop).
 */
export class ContinuousEvaluationPipeline {
  private baseline: FullEvaluationReport | null = null;

  constructor(
    private readonly harness: unknown,
    private retrievalEvaluator: RetrievalEvaluator,
    private generationEvaluator: GenerationEvaluator,
    private endToEndEvaluator: EndToEndEvaluator
  ) {}

  /** Run all evaluators and store the result as the performance baseline. */
  async setBaseline(): Promise<void> {
    this.baseline = await this.runAll();
    console.log("Baseline set:", this._summary(this.baseline));
  }

  /** Run all three evaluation levels. */
  async runAll(): Promise<FullEvaluationReport> {
    const retrieval = await this.retrievalEvaluator.evaluate();
    const generation = await this.generationEvaluator.evaluate();
    const endToEnd = await this.endToEndEvaluator.evaluate();
    return { retrieval, generation, end_to_end: endToEnd };
  }

  /**
   * Compare current metrics against the baseline.
   * Flags any metric that drops by more than {@link REGRESSION_THRESHOLD}.
   */
  async checkRegression(): Promise<RegressionCheck> {
    if (!this.baseline) {
      throw new Error("No baseline set. Call setBaseline() first.");
    }
    const current = await this.runAll();
    const regressions: string[] = [];

    const check = (name: string, base: number, curr: number): void => {
      if (curr < base - REGRESSION_THRESHOLD) {
        regressions.push(
          `${name} dropped from ${(base * 100).toFixed(1)}% to ${(curr * 100).toFixed(1)}%`
        );
      }
    };

    check("Hit Rate", this.baseline.retrieval.hit_rate, current.retrieval.hit_rate);
    check("MRR", this.baseline.retrieval.mrr, current.retrieval.mrr);
    check("NDCG@5", this.baseline.retrieval.ndcg_at_k, current.retrieval.ndcg_at_k);
    check(
      "Generation Pass Rate",
      this.baseline.generation.overall_pass_rate,
      current.generation.overall_pass_rate
    );
    check(
      "Task Success Rate",
      this.baseline.end_to_end.task_success_rate,
      current.end_to_end.task_success_rate
    );

    return {
      has_regressions: regressions.length > 0,
      regressions,
      baseline: this.baseline,
      current,
    };
  }

  private _summary(r: FullEvaluationReport): string {
    return (
      `hit_rate=${(r.retrieval.hit_rate * 100).toFixed(1)}%  ` +
      `pass_rate=${(r.generation.overall_pass_rate * 100).toFixed(1)}%  ` +
      `task_success=${(r.end_to_end.task_success_rate * 100).toFixed(1)}%`
    );
  }
}

// ===========================================================================
// DEMO STUBS
// ===========================================================================

class DemoRetriever implements Retriever {
  constructor(
    private corpus: Record<string, string[]>,
    private degraded = false
  ) {}

  async search(query: string, k: number): Promise<Array<{ id: string }>> {
    let docs = this.corpus[query] ?? [];
    if (this.degraded) {
      docs = [...docs].reverse();
    }
    return docs.slice(0, k).map((id) => ({ id }));
  }
}

class DemoAgent implements Agent {
  private static responses: Record<string, string> = {
    "What's your return policy?":
      "You may return items within 30 days in original packaging with a receipt. Source: return-policy.md Is there anything else I can help you with?",
    "How much does shipping cost?":
      "Shipping is free on orders over $50. Source: shipping-info.md Is there anything else I can help you with?",
    "Compare the Pro and Enterprise plans":
      "The Pro plan starts at $29/month. The Enterprise plan starts at $299/month. Is there anything else I can help you with?",
    "Do you ship to Germany?":
      "Yes, we ship internationally including Germany. Source: international-shipping.md Is there anything else I can help you with?",
  };

  async run(query: string, _options?: unknown): Promise<AgentResponse> {
    const content =
      DemoAgent.responses[query] ??
      `I can help with that. Is there anything else I can help you with?`;
    return {
      content,
      metadata: { retrieved_documents: ["Relevant document excerpt."] },
      tool_calls: [],
    };
  }
}

function buildRetrievalTests(): RetrievalTestCase[] {
  return [
    {
      query: "What's your return policy for damaged items?",
      relevant_doc_ids: ["return-policy.md", "damaged-goods-policy.md"],
      irrelevant_doc_ids: ["pricing.md", "careers.md"],
      min_results_expected: 2,
    },
    {
      query: "How do I reset my password?",
      relevant_doc_ids: ["account-faq.md"],
      partially_relevant_doc_ids: ["security-policy.md"],
      irrelevant_doc_ids: ["shipping-info.md"],
      min_results_expected: 1,
    },
    {
      query: "Do you ship to Germany?",
      relevant_doc_ids: ["international-shipping.md"],
      irrelevant_doc_ids: ["domestic-shipping.md", "return-policy.md"],
      min_results_expected: 1,
    },
    {
      query: "Tell me about your company history",
      relevant_doc_ids: ["about-us.md", "company-history.md"],
      irrelevant_doc_ids: ["pricing.md", "api-docs.md"],
      min_results_expected: 1,
    },
    {
      query: "What's the capital of France?",
      relevant_doc_ids: [],
      min_results_expected: 0,
    },
  ];
}

function buildGenerationTests(): GenerationTestCase[] {
  return [
    {
      query: "What's your return policy?",
      expected_answer_contains: ["30 days", "original packaging", "receipt"],
      expected_answer_not_contains: ["60 days"],
      expected_sources: ["return-policy.md"],
    },
    {
      query: "How much does shipping cost?",
      expected_answer_contains: ["free shipping", "$50"],
      expected_sources: ["shipping-info.md"],
    },
    {
      query: "Compare the Pro and Enterprise plans",
      expected_answer_contains: ["Pro", "Enterprise", "$"],
      min_answer_length: 50,
    },
    {
      query: "Do you ship to Germany?",
      expected_answer_contains: ["Germany"],
      expected_sources: ["international-shipping.md"],
    },
  ];
}

function buildE2eTests(): EndToEndTestCase[] {
  return [
    {
      scenario: "Customer wants to return a damaged item",
      user_messages: ["I received a damaged item and want to return it.", "Order #12345."],
      expected_outcome: "resolved",
      max_turns_expected: 3,
    },
    {
      scenario: "Customer asks about shipping to Germany",
      user_messages: ["Do you ship to Germany?"],
      expected_outcome: "resolved",
      max_turns_expected: 2,
    },
    {
      scenario: "Customer asks about shipping cost",
      user_messages: ["How much does shipping cost?"],
      expected_outcome: "resolved",
      max_turns_expected: 2,
    },
  ];
}

function buildCorpus(degraded = false): Record<string, string[]> {
  const corpus: Record<string, string[]> = {
    "What's your return policy for damaged items?": [
      "return-policy.md",
      "damaged-goods-policy.md",
      "pricing.md",
    ],
    "How do I reset my password?": ["account-faq.md", "security-policy.md"],
    "Do you ship to Germany?": ["international-shipping.md", "domestic-shipping.md"],
    "Tell me about your company history": ["about-us.md", "company-history.md"],
    "What's the capital of France?": [],
  };
  if (degraded) {
    const out: Record<string, string[]> = {};
    let i = 0;
    for (const [k, v] of Object.entries(corpus)) {
      out[k] = i % 2 === 0 ? [...v].reverse() : v;
      i++;
    }
    return out;
  }
  return corpus;
}

async function main(): Promise<void> {
  console.log("=".repeat(60));
  console.log("AGENT EVALUATION FRAMEWORK — TypeScript Demo");
  console.log("=".repeat(60));

  const retriever = new DemoRetriever(buildCorpus());
  const agent = new DemoAgent();
  const judge = new LLMJudge();

  // Override _callJudge for demo (no real API key required)
  (judge as unknown as { _callJudge: () => Promise<Record<string, unknown>> })._callJudge =
    async () => ({
      is_faithful: true,
      is_relevant: true,
      is_complete: true,
      score: 4,
      unsupported_claims: [],
      off_topic_parts: [],
      missing_parts: [],
      explanation: "Demo stub.",
    });

  const retrievalEval = new RetrievalEvaluator(retriever, buildRetrievalTests());
  const generationEval = new GenerationEvaluator(agent, buildGenerationTests(), judge);
  const e2eEval = new EndToEndEvaluator(agent, buildE2eTests());

  const pipeline = new ContinuousEvaluationPipeline(
    null,
    retrievalEval,
    generationEval,
    e2eEval
  );

  console.log("\n[ Stage 1 ] Setting baseline…");
  await pipeline.setBaseline();

  console.log("\n[ Stage 2 ] Simulating retrieval regression…");
  const degradedRetriever = new DemoRetriever(buildCorpus(true));
  (pipeline as unknown as { retrievalEvaluator: RetrievalEvaluator }).retrievalEvaluator =
    new RetrievalEvaluator(degradedRetriever, buildRetrievalTests());

  const check = await pipeline.checkRegression();
  if (check.has_regressions) {
    console.log("\n❌  REGRESSIONS DETECTED:");
    for (const r of check.regressions) console.log(`  • ${r}`);
  } else {
    console.log("\n✅  No regressions detected.");
  }

  console.log("\n[ Stage 3 ] Full report:");
  const report = check.current;
  console.log(`  Retrieval  — Hit Rate: ${(report.retrieval.hit_rate * 100).toFixed(1)}%`);
  console.log(`  Generation — Pass Rate: ${(report.generation.overall_pass_rate * 100).toFixed(1)}%`);
  console.log(`  End-to-End — Task Success: ${(report.end_to_end.task_success_rate * 100).toFixed(1)}%`);
  console.log("\nDone.");
}

main().catch(console.error);
