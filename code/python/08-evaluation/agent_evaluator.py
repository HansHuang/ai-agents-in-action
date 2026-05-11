"""
agent_evaluator.py
==================
Three-level agent evaluation framework.

Evaluates AI agents across three dimensions:
  1. Retrieval   — Are we finding the right documents? (Hit Rate, Precision, Recall, MRR, NDCG)
  2. Generation  — Is the answer good? (Rule-based + LLM-as-judge)
  3. End-to-End  — Does the agent solve the user's problem? (Task Success Rate)

Additionally provides a ContinuousEvaluationPipeline for regression detection.

See: docs/08-evaluation-and-guardrails/01-evaluating-agents.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# LEVEL 1 — RETRIEVAL EVALUATION
# ===========================================================================


@dataclass
class RetrievalTestCase:
    """A single retrieval test case mapping a query to expected document IDs."""

    query: str
    relevant_doc_ids: list[str]
    partially_relevant_doc_ids: list[str] = None
    irrelevant_doc_ids: list[str] = None
    min_results_expected: int = 1


@dataclass
class RetrievalReport:
    """Aggregated retrieval evaluation metrics across all test cases."""

    hit_rate: float          # Fraction of queries where ≥1 relevant doc was retrieved
    precision_at_k: float    # Avg fraction of retrieved docs that are relevant
    recall_at_k: float       # Avg fraction of all relevant docs that were retrieved
    mrr: float               # Mean Reciprocal Rank
    ndcg_at_k: float         # Normalised Discounted Cumulative Gain
    total_queries: int
    queries_with_zero_results: int
    per_query: list[dict]

    def to_string(self) -> str:
        return (
            "\nRETRIEVAL EVALUATION REPORT\n"
            "============================\n"
            f"Total Queries: {self.total_queries}\n\n"
            f"Hit Rate:        {self.hit_rate:.2%}  (target: > 90%)\n"
            f"Precision@5:     {self.precision_at_k:.2%}  (target: > 70%)\n"
            f"Recall@5:        {self.recall_at_k:.2%}  (target: > 80%)\n"
            f"MRR:             {self.mrr:.2%}  (target: > 60%)\n"
            f"NDCG@5:          {self.ndcg_at_k:.2%}  (target: > 70%)\n\n"
            f"Queries with zero relevant results: {self.queries_with_zero_results}\n"
        )


class RetrievalEvaluator:
    """
    Evaluate retrieval quality with standard information-retrieval metrics.

    Args:
        retriever: Object with a ``search(query, k)`` method returning
                   a list of dicts, each containing an ``"id"`` key.
        test_cases: List of RetrievalTestCase instances.
    """

    def __init__(self, retriever: Any, test_cases: list[RetrievalTestCase]) -> None:
        self.retriever = retriever
        self.test_cases = test_cases

    def evaluate(self, k: int = 5) -> RetrievalReport:
        """Evaluate retrieval quality across all test cases at depth *k*."""
        results = []
        for test in self.test_cases:
            retrieved = self.retriever.search(test.query, k=k)
            retrieved_ids = [doc["id"] for doc in retrieved]
            metrics = self._calculate_metrics(test, retrieved_ids, k)
            results.append(metrics)
        return self._aggregate(results)

    def _calculate_metrics(
        self,
        test: RetrievalTestCase,
        retrieved_ids: list[str],
        k: int,
    ) -> dict:
        """Calculate all retrieval metrics for a single query."""
        relevant = set(test.relevant_doc_ids)
        retrieved_set = set(retrieved_ids[:k])

        # Hit Rate: was at least one relevant document retrieved?
        hit = 1 if relevant & retrieved_set else 0

        # Precision@K: what fraction of retrieved documents are relevant?
        precision = len(relevant & retrieved_set) / len(retrieved_set) if retrieved_set else 0.0

        # Recall@K: what fraction of all relevant documents were retrieved?
        recall = len(relevant & retrieved_set) / len(relevant) if relevant else 1.0

        # MRR: reciprocal rank of the first relevant document
        reciprocal_rank = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / (i + 1)
                break

        # NDCG@K
        ndcg = self._calculate_ndcg(test, retrieved_ids, k)

        return {
            "query": test.query,
            "hit": hit,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_k": ndcg,
            "relevant_found": len(relevant & retrieved_set),
            "relevant_total": len(relevant),
            "retrieved_ids": retrieved_ids,
        }

    def _calculate_ndcg(
        self,
        test: RetrievalTestCase,
        retrieved_ids: list[str],
        k: int,
    ) -> float:
        """
        NDCG with graded relevance:
          - relevant          → score 2
          - partially_relevant → score 1
          - everything else   → score 0
        """
        relevance_scores: dict[str, int] = {}
        for doc_id in test.relevant_doc_ids:
            relevance_scores[doc_id] = 2
        if test.partially_relevant_doc_ids:
            for doc_id in test.partially_relevant_doc_ids:
                relevance_scores.setdefault(doc_id, 1)

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0)
            dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed

        # IDCG — ideal DCG given the best possible ordering
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _aggregate(self, results: list[dict]) -> RetrievalReport:
        """Aggregate per-query metrics into a RetrievalReport."""
        n = len(results)
        return RetrievalReport(
            hit_rate=sum(r["hit"] for r in results) / n,
            precision_at_k=sum(r["precision_at_k"] for r in results) / n,
            recall_at_k=sum(r["recall_at_k"] for r in results) / n,
            mrr=sum(r["reciprocal_rank"] for r in results) / n,
            ndcg_at_k=sum(r["ndcg_at_k"] for r in results) / n,
            total_queries=n,
            queries_with_zero_results=sum(
                1
                for r in results
                if r["relevant_found"] == 0 and r["relevant_total"] > 0
            ),
            per_query=results,
        )


# ===========================================================================
# LEVEL 2 — GENERATION EVALUATION
# ===========================================================================


@dataclass
class GenerationTestCase:
    """A single generation test case."""

    query: str
    expected_answer_contains: list[str] = None
    expected_answer_not_contains: list[str] = None
    expected_sources: list[str] = None
    min_answer_length: int = 20
    max_answer_length: int = 2000
    reference_answer: str = None
    evaluation_criteria: str = None


@dataclass
class JudgeResult:
    """Result from the LLM-as-judge for a single evaluation dimension."""

    dimension: str
    passed: bool
    score: int  # 1–5
    issues: list[str]
    explanation: str


@dataclass
class GenerationReport:
    """Aggregated generation evaluation metrics."""

    overall_pass_rate: float
    contains_required_rate: float
    avoids_forbidden_rate: float
    faithfulness_pass_rate: Optional[float]
    relevance_pass_rate: Optional[float]
    completeness_pass_rate: Optional[float]
    total_queries: int
    per_query: list[dict]

    def to_string(self) -> str:
        lines = [
            "\nGENERATION EVALUATION REPORT",
            "==============================",
            f"Total Queries: {self.total_queries}",
            "",
            f"Overall Pass Rate:       {self.overall_pass_rate:.2%}",
            f"Contains Required:       {self.contains_required_rate:.2%}",
            f"Avoids Forbidden:        {self.avoids_forbidden_rate:.2%}",
        ]
        if self.faithfulness_pass_rate is not None:
            lines.append(f"Faithfulness (judge):    {self.faithfulness_pass_rate:.2%}")
        if self.relevance_pass_rate is not None:
            lines.append(f"Relevance (judge):       {self.relevance_pass_rate:.2%}")
        if self.completeness_pass_rate is not None:
            lines.append(f"Completeness (judge):    {self.completeness_pass_rate:.2%}")
        lines.append("")
        return "\n".join(lines)


class LLMJudge:
    """
    Use an LLM to evaluate the quality of agent responses.

    The judge runs at low temperature for consistent, reproducible scoring.
    Biases to be aware of: length preference, verbosity preference, and
    susceptibility to confident-sounding falsehoods.

    Args:
        model: The model identifier to use for judging (default: ``"gpt-4o"``).
    """

    FAITHFULNESS_PROMPT = (
        "You are evaluating whether an AI response is faithful "
        "to the provided source documents.\n\n"
        "Faithfulness means: Every factual claim in the response is directly supported by "
        "at least one of the source documents. The response does not add information not "
        "found in the sources.\n\n"
        "SOURCE DOCUMENTS:\n{source_documents}\n\n"
        "RESPONSE TO EVALUATE:\n{response}\n\n"
        "Evaluate the response for faithfulness. Output JSON:\n"
        '{{\n'
        '    "is_faithful": true/false,\n'
        '    "score": 1-5,\n'
        '    "unsupported_claims": ["claim1", "claim2"],\n'
        '    "explanation": "Brief explanation of your evaluation"\n'
        "}}"
    )

    RELEVANCE_PROMPT = (
        "You are evaluating whether an AI response is relevant "
        "to the user's question.\n\n"
        "Relevance means: The response directly addresses what the user asked. It does not "
        "go off-topic or provide unnecessary information.\n\n"
        "USER QUESTION:\n{user_question}\n\n"
        "RESPONSE TO EVALUATE:\n{response}\n\n"
        "Evaluate the response for relevance. Output JSON:\n"
        '{{\n'
        '    "is_relevant": true/false,\n'
        '    "score": 1-5,\n'
        '    "off_topic_parts": ["part1", "part2"],\n'
        '    "explanation": "Brief explanation"\n'
        "}}"
    )

    COMPLETENESS_PROMPT = (
        "You are evaluating whether an AI response completely "
        "answers the user's question.\n\n"
        "Completeness means: The response addresses ALL parts of the user's question. "
        "If the user asked multiple questions, all are answered. If the user asked for "
        "a comparison, both sides are covered.\n\n"
        "USER QUESTION:\n{user_question}\n\n"
        "RESPONSE TO EVALUATE:\n{response}\n\n"
        "Evaluate the response for completeness. Output JSON:\n"
        '{{\n'
        '    "is_complete": true/false,\n'
        '    "score": 1-5,\n'
        '    "missing_parts": ["unanswered question 1", "unanswered question 2"],\n'
        '    "explanation": "Brief explanation"\n'
        "}}"
    )

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    async def evaluate_faithfulness(
        self, response: str, source_documents: list[str]
    ) -> JudgeResult:
        """Evaluate whether *response* is faithful to *source_documents*."""
        sources_text = "\n\n---\n\n".join(
            f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(source_documents)
        )
        prompt = self.FAITHFULNESS_PROMPT.format(
            source_documents=sources_text[:10_000],
            response=response[:5_000],
        )
        result = await self._call_judge(prompt)
        return JudgeResult(
            dimension="faithfulness",
            passed=result.get("is_faithful", False),
            score=result.get("score", 1),
            issues=result.get("unsupported_claims", []),
            explanation=result.get("explanation", ""),
        )

    async def evaluate_relevance(
        self, response: str, user_question: str
    ) -> JudgeResult:
        """Evaluate whether *response* is relevant to *user_question*."""
        prompt = self.RELEVANCE_PROMPT.format(
            user_question=user_question,
            response=response[:5_000],
        )
        result = await self._call_judge(prompt)
        return JudgeResult(
            dimension="relevance",
            passed=result.get("is_relevant", False),
            score=result.get("score", 1),
            issues=result.get("off_topic_parts", []),
            explanation=result.get("explanation", ""),
        )

    async def evaluate_completeness(
        self, response: str, user_question: str
    ) -> JudgeResult:
        """Evaluate whether *response* completely answers *user_question*."""
        prompt = self.COMPLETENESS_PROMPT.format(
            user_question=user_question,
            response=response[:5_000],
        )
        result = await self._call_judge(prompt)
        return JudgeResult(
            dimension="completeness",
            passed=result.get("is_complete", False),
            score=result.get("score", 1),
            issues=result.get("missing_parts", []),
            explanation=result.get("explanation", ""),
        )

    async def _call_judge(self, prompt: str) -> dict:
        """
        Call the judge LLM and parse the JSON response.

        In production, replace this stub with a real OpenAI (or other)
        API call using ``response_format={"type": "json_object"}`` and
        ``temperature=0.1``.
        """
        # Stub: return a passing score for demo purposes.
        # Replace with: openai.chat.completions.create(...)
        return {
            "is_faithful": True,
            "is_relevant": True,
            "is_complete": True,
            "score": 4,
            "unsupported_claims": [],
            "off_topic_parts": [],
            "missing_parts": [],
            "explanation": "Stub judge — replace with real LLM call in production.",
        }


class GenerationEvaluator:
    """
    Evaluate generation quality using rule-based checks and LLM-as-judge.

    Args:
        agent: Object with an async ``run(query)`` method returning an object
               with ``.content`` and ``.metadata`` attributes.
        test_cases: List of GenerationTestCase instances.
        judge: Optional LLMJudge; created with defaults when omitted.
    """

    def __init__(
        self,
        agent: Any,
        test_cases: list[GenerationTestCase],
        judge: Optional[LLMJudge] = None,
    ) -> None:
        self.agent = agent
        self.test_cases = test_cases
        self.judge = judge or LLMJudge()

    async def evaluate(self) -> GenerationReport:
        """Evaluate generation quality across all test cases."""
        results = []
        for test in self.test_cases:
            response = await self.agent.run(test.query)
            rule_checks = self._rule_based_checks(test, response.content)

            judge_checks: dict[str, JudgeResult] = {}
            if self.judge:
                source_docs = response.metadata.get("retrieved_documents", [])
                judge_checks["faithfulness"] = await self.judge.evaluate_faithfulness(
                    response.content, source_docs
                )
                judge_checks["relevance"] = await self.judge.evaluate_relevance(
                    response.content, test.query
                )
                judge_checks["completeness"] = await self.judge.evaluate_completeness(
                    response.content, test.query
                )

            overall_pass = rule_checks.get("all_passed", False) and all(
                j.passed for j in judge_checks.values()
            )
            results.append(
                {
                    "query": test.query,
                    "response": response.content,
                    "rule_checks": rule_checks,
                    "judge_checks": {k: v for k, v in judge_checks.items()},
                    "overall_pass": overall_pass,
                }
            )
        return self._aggregate(results)

    def _rule_based_checks(self, test: GenerationTestCase, response: str) -> dict:
        """Perform deterministic checks on the response."""
        checks: dict[str, Any] = {}
        response_lower = response.lower()

        if test.expected_answer_contains:
            missing = [
                p for p in test.expected_answer_contains if p.lower() not in response_lower
            ]
            checks["contains_required"] = len(missing) == 0
            checks["missing_required"] = missing

        if test.expected_answer_not_contains:
            found = [
                p for p in test.expected_answer_not_contains if p.lower() in response_lower
            ]
            checks["avoids_forbidden"] = len(found) == 0
            checks["found_forbidden"] = found

        checks["length_ok"] = (
            test.min_answer_length <= len(response) <= test.max_answer_length
        )

        if test.expected_sources:
            checks["sources_cited"] = all(
                src.lower() in response_lower for src in test.expected_sources
            )

        checks["all_passed"] = all(
            v for k, v in checks.items()
            if k not in ("missing_required", "found_forbidden") and isinstance(v, bool)
        )
        return checks

    def _aggregate(self, results: list[dict]) -> GenerationReport:
        """Aggregate per-query results into a GenerationReport."""
        n = len(results)
        has_judge = bool(results and results[0].get("judge_checks"))

        def _pass_rate(key: str) -> Optional[float]:
            if not has_judge:
                return None
            return sum(
                1 for r in results if r["judge_checks"].get(key, JudgeResult("", False, 1, [], "")).passed
            ) / n

        return GenerationReport(
            overall_pass_rate=sum(1 for r in results if r["overall_pass"]) / n,
            contains_required_rate=sum(
                1 for r in results if r["rule_checks"].get("contains_required", True)
            ) / n,
            avoids_forbidden_rate=sum(
                1 for r in results if r["rule_checks"].get("avoids_forbidden", True)
            ) / n,
            faithfulness_pass_rate=_pass_rate("faithfulness"),
            relevance_pass_rate=_pass_rate("relevance"),
            completeness_pass_rate=_pass_rate("completeness"),
            total_queries=n,
            per_query=results,
        )


# ===========================================================================
# LEVEL 3 — END-TO-END EVALUATION
# ===========================================================================


@dataclass
class EndToEndTestCase:
    """A multi-turn end-to-end test scenario."""

    scenario: str
    user_messages: list[str]
    expected_outcome: str  # "resolved", "escalated", "information_provided"
    expected_tools_called: list[str] = None
    max_turns_expected: int = 5
    forbidden_behaviors: list[str] = None


@dataclass
class EndToEndReport:
    """Aggregated end-to-end evaluation results."""

    task_success_rate: float
    avg_turns_to_resolution: float
    total_scenarios: int
    per_scenario: list[dict]

    def to_string(self) -> str:
        lines = [
            "\nEND-TO-END EVALUATION REPORT",
            "=============================",
            f"Total Scenarios: {self.total_scenarios}",
            "",
            f"Task Success Rate:         {self.task_success_rate:.2%}  (target: > 85%)",
            f"Avg Turns to Resolution:   {self.avg_turns_to_resolution:.1f}",
            "",
        ]
        for r in self.per_scenario:
            status = "✅" if r["success"] else "❌"
            lines.append(
                f"  {status}  {r['scenario'][:60]}  "
                f"({r['turns_taken']} turn(s), outcome: {r['outcome']})"
            )
        lines.append("")
        return "\n".join(lines)


class EndToEndEvaluator:
    """
    Evaluate the agent on realistic multi-turn scenarios.

    Args:
        agent: Object with an async ``run(message, conversation_history)`` method.
        test_cases: List of EndToEndTestCase instances.
    """

    # Phrases that indicate the agent considers the task resolved.
    RESOLUTION_MARKERS = [
        "is there anything else",
        "i hope that helps",
        "your request has been",
        "i've completed",
        "would you like me to",
    ]

    def __init__(self, agent: Any, test_cases: list[EndToEndTestCase]) -> None:
        self.agent = agent
        self.test_cases = test_cases

    async def evaluate(self) -> EndToEndReport:
        """Run all end-to-end scenarios and collect results."""
        results = []
        for test in self.test_cases:
            conversation: list[dict] = []
            tools_called: list[str] = []
            outcome = "unknown"
            turns_taken = 0

            for i, user_message in enumerate(test.user_messages):
                response = await self.agent.run(
                    user_message, conversation_history=conversation
                )
                turns_taken = i + 1
                conversation.append({"role": "user", "content": user_message})
                conversation.append({"role": "assistant", "content": response.content})

                if hasattr(response, "tool_calls") and response.tool_calls:
                    tools_called.extend(tc.name for tc in response.tool_calls)

                if self._is_resolved(response, test):
                    outcome = "resolved"
                    break

            if outcome == "unknown":
                outcome = (
                    "unresolved"
                    if turns_taken >= test.max_turns_expected
                    else "incomplete"
                )

            results.append(
                {
                    "scenario": test.scenario,
                    "outcome": outcome,
                    "expected_outcome": test.expected_outcome,
                    "turns_taken": turns_taken,
                    "tools_called": tools_called,
                    "expected_tools": test.expected_tools_called,
                    "success": outcome == test.expected_outcome,
                }
            )

        n = len(results)
        return EndToEndReport(
            task_success_rate=sum(1 for r in results if r["success"]) / n,
            avg_turns_to_resolution=sum(r["turns_taken"] for r in results) / n,
            total_scenarios=n,
            per_scenario=results,
        )

    def _is_resolved(self, response: Any, test: EndToEndTestCase) -> bool:
        """Return True if the response contains a resolution marker."""
        content_lower = response.content.lower()
        return any(marker in content_lower for marker in self.RESOLUTION_MARKERS)


# ===========================================================================
# CONTINUOUS EVALUATION PIPELINE
# ===========================================================================


@dataclass
class FullEvaluationReport:
    """Combined report from all three evaluation levels."""

    retrieval: RetrievalReport
    generation: GenerationReport
    end_to_end: EndToEndReport

    def to_string(self) -> str:
        return (
            self.retrieval.to_string()
            + self.generation.to_string()
            + self.end_to_end.to_string()
        )

    def summary(self) -> str:
        return (
            f"Retrieval hit_rate={self.retrieval.hit_rate:.2%}  "
            f"Generation pass_rate={self.generation.overall_pass_rate:.2%}  "
            f"E2E success={self.end_to_end.task_success_rate:.2%}"
        )


@dataclass
class RegressionCheck:
    """Result of a regression check against a stored baseline."""

    has_regressions: bool
    regressions: list[str]
    baseline: FullEvaluationReport
    current: FullEvaluationReport

    def to_string(self) -> str:
        if not self.has_regressions:
            return "\n✅  No regressions detected. All metrics within acceptable range.\n"
        lines = ["\n❌  REGRESSIONS DETECTED\n" + "=" * 30]
        for r in self.regressions:
            lines.append(f"  • {r}")
        lines.append("")
        return "\n".join(lines)


class ContinuousEvaluationPipeline:
    """
    Run evaluation on every change and alert on regressions.

    Compares current metrics against a stored baseline and flags any
    metric that drops by more than 5 percentage points.

    Args:
        harness: The production harness (passed through for context).
        retrieval_evaluator: A configured RetrievalEvaluator.
        generation_evaluator: A configured GenerationEvaluator.
        end_to_end_evaluator: A configured EndToEndEvaluator.
    """

    REGRESSION_THRESHOLD = 0.05  # 5 percentage points

    def __init__(
        self,
        harness: Any,
        retrieval_evaluator: RetrievalEvaluator,
        generation_evaluator: GenerationEvaluator,
        end_to_end_evaluator: EndToEndEvaluator,
    ) -> None:
        self.harness = harness
        self.retrieval_evaluator = retrieval_evaluator
        self.generation_evaluator = generation_evaluator
        self.end_to_end_evaluator = end_to_end_evaluator
        self.baseline: Optional[FullEvaluationReport] = None

    async def set_baseline(self) -> None:
        """Run all evaluators and store the result as the performance baseline."""
        self.baseline = await self.run_all()
        logger.info("Baseline set: %s", self.baseline.summary())

    async def run_all(self) -> FullEvaluationReport:
        """Run all three evaluation levels and return a combined report."""
        retrieval = self.retrieval_evaluator.evaluate()
        generation = await self.generation_evaluator.evaluate()
        end_to_end = await self.end_to_end_evaluator.evaluate()
        return FullEvaluationReport(
            retrieval=retrieval,
            generation=generation,
            end_to_end=end_to_end,
        )

    async def check_regression(self) -> RegressionCheck:
        """
        Compare current performance against the baseline.

        Raises:
            ValueError: If no baseline has been set.
        """
        if self.baseline is None:
            raise ValueError("No baseline set. Call set_baseline() first.")

        current = await self.run_all()
        regressions: list[str] = []
        t = self.REGRESSION_THRESHOLD

        def _check(name: str, baseline_val: float, current_val: float) -> None:
            if current_val < baseline_val - t:
                regressions.append(
                    f"{name} dropped from {baseline_val:.2%} to {current_val:.2%}"
                )

        _check("Hit Rate", self.baseline.retrieval.hit_rate, current.retrieval.hit_rate)
        _check("MRR", self.baseline.retrieval.mrr, current.retrieval.mrr)
        _check("NDCG@5", self.baseline.retrieval.ndcg_at_k, current.retrieval.ndcg_at_k)
        _check(
            "Generation Pass Rate",
            self.baseline.generation.overall_pass_rate,
            current.generation.overall_pass_rate,
        )
        _check(
            "Task Success Rate",
            self.baseline.end_to_end.task_success_rate,
            current.end_to_end.task_success_rate,
        )

        return RegressionCheck(
            has_regressions=len(regressions) > 0,
            regressions=regressions,
            baseline=self.baseline,
            current=current,
        )


# ===========================================================================
# DEMO STUBS — minimal fake retriever and agent for self-contained demo
# ===========================================================================


class _DemoRetriever:
    """Fake retriever backed by a configurable document corpus."""

    def __init__(self, corpus: dict[str, list[str]], degraded: bool = False) -> None:
        # corpus maps query → list of doc IDs ranked by relevance
        self._corpus = corpus
        self._degraded = degraded  # Simulate lower quality for regression demo

    def search(self, query: str, k: int = 5) -> list[dict]:
        results = self._corpus.get(query, [])
        if self._degraded:
            # Shuffle results to simulate degraded retrieval quality
            results = list(results)
            random.shuffle(results)
        return [{"id": doc_id} for doc_id in results[:k]]


@dataclass
class _DemoResponse:
    content: str
    metadata: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)


class _DemoAgent:
    """Minimal fake agent that returns scripted responses."""

    RESPONSES = {
        "What's your return policy?": (
            "You may return items within 30 days in original packaging with a receipt. "
            "Returns are accepted for any reason. "
            "Source: return-policy.md  "
            "Is there anything else I can help you with?"
        ),
        "How much does shipping cost?": (
            "Shipping is free on orders over $50. Standard rates apply below that threshold. "
            "Source: shipping-info.md  "
            "Is there anything else I can help you with?"
        ),
        "Compare the Pro and Enterprise plans": (
            "The Pro plan starts at $29/month. The Enterprise plan starts at $299/month "
            "and includes SLA guarantees, dedicated support, and SSO. "
            "Is there anything else I can help you with?"
        ),
        "How do I reset my password?": (
            "Visit account settings and click 'Reset Password'. "
            "Is there anything else I can help you with?"
        ),
        "Do you ship to Germany?": (
            "Yes, we ship internationally including Germany. "
            "Source: international-shipping.md  "
            "Is there anything else I can help you with?"
        ),
        "Tell me about your company": (
            "We were founded in 2010 and are headquartered in San Francisco. "
            "Source: about-us.md  "
            "Is there anything else I can help you with?"
        ),
    }

    async def run(self, query: str, conversation_history: list = None) -> _DemoResponse:
        content = self.RESPONSES.get(
            query,
            f"I can help you with that. The answer to '{query}' is... "
            "Is there anything else I can help you with?",
        )
        return _DemoResponse(
            content=content,
            metadata={"retrieved_documents": ["Relevant document excerpt."]},
        )


# ===========================================================================
# DEMO
# ===========================================================================


def _build_retrieval_tests() -> list[RetrievalTestCase]:
    return [
        RetrievalTestCase(
            query="What's your return policy for damaged items?",
            relevant_doc_ids=["return-policy.md", "damaged-goods-policy.md"],
            irrelevant_doc_ids=["pricing.md", "careers.md"],
            min_results_expected=2,
        ),
        RetrievalTestCase(
            query="How do I reset my password?",
            relevant_doc_ids=["account-faq.md"],
            partially_relevant_doc_ids=["security-policy.md"],
            irrelevant_doc_ids=["shipping-info.md"],
            min_results_expected=1,
        ),
        RetrievalTestCase(
            query="Do you ship to Germany?",
            relevant_doc_ids=["international-shipping.md"],
            irrelevant_doc_ids=["domestic-shipping.md", "return-policy.md"],
            min_results_expected=1,
        ),
        RetrievalTestCase(
            query="Tell me about your company history",
            relevant_doc_ids=["about-us.md", "company-history.md"],
            irrelevant_doc_ids=["pricing.md", "api-docs.md"],
            min_results_expected=1,
        ),
        RetrievalTestCase(
            query="What's the capital of France?",
            relevant_doc_ids=[],  # Intentionally not in knowledge base
            min_results_expected=0,
        ),
        RetrievalTestCase(
            query="How long does shipping take?",
            relevant_doc_ids=["shipping-info.md", "domestic-shipping.md"],
            irrelevant_doc_ids=["return-policy.md"],
            min_results_expected=1,
        ),
        RetrievalTestCase(
            query="What payment methods do you accept?",
            relevant_doc_ids=["payment-methods.md"],
            irrelevant_doc_ids=["shipping-info.md"],
            min_results_expected=1,
        ),
        RetrievalTestCase(
            query="How do I cancel my subscription?",
            relevant_doc_ids=["subscription-faq.md", "account-faq.md"],
            min_results_expected=1,
        ),
        RetrievalTestCase(
            query="Are there student discounts?",
            relevant_doc_ids=["discounts.md"],
            irrelevant_doc_ids=["careers.md"],
            min_results_expected=1,
        ),
        RetrievalTestCase(
            query="What are your business hours?",
            relevant_doc_ids=["contact-us.md"],
            min_results_expected=1,
        ),
    ]


def _build_retrieval_corpus(degraded: bool = False) -> dict[str, list[str]]:
    """Map each test query to an ordered list of retrieved document IDs."""
    corpus = {
        "What's your return policy for damaged items?": [
            "return-policy.md",
            "damaged-goods-policy.md",
            "pricing.md",
        ],
        "How do I reset my password?": [
            "account-faq.md",
            "security-policy.md",
            "shipping-info.md",
        ],
        "Do you ship to Germany?": [
            "international-shipping.md",
            "domestic-shipping.md",
        ],
        "Tell me about your company history": [
            "about-us.md",
            "company-history.md",
            "pricing.md",
        ],
        "What's the capital of France?": [],  # Nothing relevant
        "How long does shipping take?": [
            "shipping-info.md",
            "domestic-shipping.md",
        ],
        "What payment methods do you accept?": ["payment-methods.md"],
        "How do I cancel my subscription?": [
            "subscription-faq.md",
            "account-faq.md",
        ],
        "Are there student discounts?": ["discounts.md"],
        "What are your business hours?": ["contact-us.md"],
    }
    if degraded:
        # Degrade quality by reversing the rank order for half the queries
        degraded_corpus = {}
        for i, (query, docs) in enumerate(corpus.items()):
            degraded_corpus[query] = list(reversed(docs)) if i % 2 == 0 else docs
        return degraded_corpus
    return corpus


def _build_generation_tests() -> list[GenerationTestCase]:
    return [
        GenerationTestCase(
            query="What's your return policy?",
            expected_answer_contains=["30 days", "original packaging", "receipt"],
            expected_answer_not_contains=["60 days", "no questions asked"],
            expected_sources=["return-policy.md"],
        ),
        GenerationTestCase(
            query="How much does shipping cost?",
            expected_answer_contains=["free shipping", "$50"],
            expected_sources=["shipping-info.md"],
        ),
        GenerationTestCase(
            query="Compare the Pro and Enterprise plans",
            expected_answer_contains=["Pro", "Enterprise", "$"],
            min_answer_length=100,
        ),
        GenerationTestCase(
            query="How do I reset my password?",
            expected_answer_contains=["account settings", "Reset Password"],
        ),
        GenerationTestCase(
            query="Do you ship to Germany?",
            expected_answer_contains=["Germany"],
            expected_sources=["international-shipping.md"],
        ),
        GenerationTestCase(
            query="What are your support hours?",
            expected_answer_contains=["hours"],
            min_answer_length=10,
        ),
        GenerationTestCase(
            query="How do I get a refund?",
            expected_answer_contains=["refund"],
            min_answer_length=20,
        ),
        GenerationTestCase(
            query="Do you have an API?",
            min_answer_length=10,
        ),
        GenerationTestCase(
            query="Tell me about your company",
            expected_answer_contains=["founded"],
            expected_sources=["about-us.md"],
        ),
        GenerationTestCase(
            query="What is your privacy policy?",
            min_answer_length=10,
        ),
    ]


def _build_e2e_tests() -> list[EndToEndTestCase]:
    return [
        EndToEndTestCase(
            scenario="Customer wants to return a damaged item",
            user_messages=[
                "I received a damaged item and want to return it.",
                "The order number is 12345.",
            ],
            expected_outcome="resolved",
            max_turns_expected=3,
        ),
        EndToEndTestCase(
            scenario="Customer asks about shipping to Germany",
            user_messages=["Do you ship to Germany?"],
            expected_outcome="resolved",
            max_turns_expected=2,
        ),
        EndToEndTestCase(
            scenario="Customer needs to reset password",
            user_messages=["I forgot my password. How do I reset it?"],
            expected_outcome="resolved",
            max_turns_expected=2,
        ),
        EndToEndTestCase(
            scenario="Customer compares subscription plans",
            user_messages=[
                "What's the difference between Pro and Enterprise?",
                "Which one includes dedicated support?",
            ],
            expected_outcome="resolved",
            max_turns_expected=3,
        ),
        EndToEndTestCase(
            scenario="Customer asks about shipping cost",
            user_messages=["How much does shipping cost?"],
            expected_outcome="resolved",
            max_turns_expected=2,
        ),
    ]


async def main() -> None:
    print("=" * 60)
    print("AGENT EVALUATION FRAMEWORK — DEMO")
    print("=" * 60)

    retrieval_tests = _build_retrieval_tests()
    generation_tests = _build_generation_tests()
    e2e_tests = _build_e2e_tests()

    # --- Stage 1: Baseline evaluation ---
    print("\n[ Stage 1 ] Running baseline evaluation…\n")

    retriever = _DemoRetriever(_build_retrieval_corpus(degraded=False))
    agent = _DemoAgent()

    retrieval_eval = RetrievalEvaluator(retriever, retrieval_tests)
    generation_eval = GenerationEvaluator(agent, generation_tests)
    e2e_eval = EndToEndEvaluator(agent, e2e_tests)

    pipeline = ContinuousEvaluationPipeline(
        harness=None,
        retrieval_evaluator=retrieval_eval,
        generation_evaluator=generation_eval,
        end_to_end_evaluator=e2e_eval,
    )

    await pipeline.set_baseline()
    print(pipeline.baseline.to_string())

    # --- Stage 2: Simulate a degraded retriever ---
    print("[ Stage 2 ] Simulating retrieval regression…\n")

    degraded_retriever = _DemoRetriever(_build_retrieval_corpus(degraded=True))
    pipeline.retrieval_evaluator = RetrievalEvaluator(degraded_retriever, retrieval_tests)

    regression = await pipeline.check_regression()
    print(regression.to_string())

    if regression.has_regressions:
        print("Baseline metrics:")
        print(f"  Hit Rate: {regression.baseline.retrieval.hit_rate:.2%}")
        print(f"  MRR:      {regression.baseline.retrieval.mrr:.2%}")
        print("\nCurrent metrics:")
        print(f"  Hit Rate: {regression.current.retrieval.hit_rate:.2%}")
        print(f"  MRR:      {regression.current.retrieval.mrr:.2%}")

    print("\n[ Stage 3 ] Full combined report\n")
    print(regression.current.to_string())


if __name__ == "__main__":
    asyncio.run(main())
