"""RAG pipeline evaluation framework.

Measures retrieval quality (hit rate, MRR, precision@k, recall@k) and
generation quality (faithfulness, relevance, groundedness) using
LLM-as-judge for the generation metrics.

See: docs/08-evaluation-and-guardrails/01-evaluating-agents.md
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from rag_pipeline import RAGPipeline
from simple_vector_store import SimpleVectorStore
from embedding_generator import EmbeddingGenerator

# ---------------------------------------------------------------------------
# Test-case dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievalTestCase:
    """A single retrieval evaluation instance.

    Attributes:
        query:            The search query.
        relevant_doc_ids: ``source`` IDs of documents that contain the answer.
    """

    query: str
    relevant_doc_ids: list[str]


@dataclass
class GenerationTestCase:
    """A single generation evaluation instance.

    Attributes:
        query:           The question posed to the pipeline.
        expected_facts:  Facts the answer MUST include.
        forbidden_facts: Facts the answer MUST NOT include.
    """

    query: str
    expected_facts: list[str]
    forbidden_facts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM-as-judge prompts
# ---------------------------------------------------------------------------

LLM_AS_JUDGE_FAITHFULNESS = """\
You are evaluating a RAG system's answer for faithfulness.
Faithfulness means the answer contains ONLY information from the provided documents.

Documents:
{documents}

Answer to evaluate:
{answer}

Does the answer contain any claims NOT supported by the documents?
Respond with valid JSON only:
{{
    "is_faithful": true or false,
    "hallucinated_claims": ["claim1", "claim2"],
    "explanation": "brief explanation"
}}
"""

LLM_AS_JUDGE_RELEVANCE = """\
You are evaluating whether a RAG system's answer addresses the question.

Question:
{question}

Answer:
{answer}

Does the answer directly address the question?
Respond with valid JSON only:
{{
    "is_relevant": true or false,
    "score": 0.0 to 1.0,
    "explanation": "brief explanation"
}}
"""

LLM_AS_JUDGE_GROUNDEDNESS = """\
You are checking whether an answer includes citations for its claims.

Answer:
{answer}

Count how many distinct factual claims are made and how many are cited with
a [Source: ...] marker.
Respond with valid JSON only:
{{
    "total_claims": integer,
    "cited_claims": integer,
    "groundedness_score": 0.0 to 1.0,
    "explanation": "brief explanation"
}}
"""


# ---------------------------------------------------------------------------
# RAGEvaluator
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """Evaluate RAG pipeline quality using standard metrics.

    Args:
        pipeline: A :class:`RAGPipeline` instance to evaluate.
        model:    OpenAI model to use for LLM-as-judge calls.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        model: str = "gpt-4o",
    ) -> None:
        self.pipeline = pipeline
        self.model = model
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Retrieval evaluation
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        test_cases: list[RetrievalTestCase],
        k: int = 5,
    ) -> dict:
        """Evaluate retrieval quality across *test_cases*.

        Metrics:
            - **hit_rate@k**: Fraction of queries where a relevant doc appears in top-k.
            - **mrr**: Mean Reciprocal Rank — how highly relevant docs rank on average.
            - **precision@k**: Fraction of top-k results that are relevant.
            - **recall@k**: Fraction of all relevant docs found in top-k.

        Args:
            test_cases: List of :class:`RetrievalTestCase` instances.
            k:          Number of results to retrieve per query.

        Returns:
            Dict with metric names as keys and float values.
        """
        if not test_cases:
            return {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}

        hits = 0
        reciprocal_ranks: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []

        embedder = self.pipeline.embedder
        vector_store = self.pipeline.vector_store

        for tc in test_cases:
            query_emb = embedder.embed(tc.query)
            results = vector_store.search(query_emb, k=k)
            retrieved_sources = [r["metadata"].get("source", "") for r in results]

            relevant_set = set(tc.relevant_doc_ids)

            # Hit rate
            if any(src in relevant_set for src in retrieved_sources):
                hits += 1

            # MRR — rank of first relevant result
            rr = 0.0
            for rank, src in enumerate(retrieved_sources, 1):
                if src in relevant_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # Precision@k
            relevant_in_top_k = sum(1 for s in retrieved_sources if s in relevant_set)
            precisions.append(relevant_in_top_k / k if k > 0 else 0.0)

            # Recall@k
            recalls.append(
                relevant_in_top_k / len(relevant_set) if relevant_set else 0.0
            )

        n = len(test_cases)
        return {
            "hit_rate": hits / n,
            "mrr": sum(reciprocal_ranks) / n,
            "precision_at_k": sum(precisions) / n,
            "recall_at_k": sum(recalls) / n,
            "total_queries": n,
            "k": k,
        }

    # ------------------------------------------------------------------
    # Generation evaluation
    # ------------------------------------------------------------------

    def evaluate_generation(
        self,
        test_cases: list[GenerationTestCase],
    ) -> dict:
        """Evaluate generation quality using LLM-as-judge.

        Metrics:
            - **faithfulness**: Does the answer stick to documents?
            - **relevance**: Does the answer address the question?
            - **groundedness**: Are claims cited?

        Args:
            test_cases: List of :class:`GenerationTestCase` instances.

        Returns:
            Aggregated metric scores plus per-case details.
        """
        if not test_cases:
            return {}

        case_results = []
        for tc in test_cases:
            response = self.pipeline.query(tc.query)
            answer = response.answer
            doc_texts = "\n---\n".join(c["text"] for c in response.retrieved_chunks)

            faithfulness = self._judge_faithfulness(doc_texts, answer)
            relevance = self._judge_relevance(tc.query, answer)
            groundedness = self._judge_groundedness(answer)

            # Check expected / forbidden facts (simple substring check)
            facts_present = [
                f for f in tc.expected_facts
                if f.lower() in answer.lower()
            ]
            facts_missing = [
                f for f in tc.expected_facts
                if f.lower() not in answer.lower()
            ]
            forbidden_present = [
                f for f in tc.forbidden_facts
                if f.lower() in answer.lower()
            ]

            case_results.append({
                "query": tc.query,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "groundedness": groundedness,
                "expected_facts_found": facts_present,
                "expected_facts_missing": facts_missing,
                "forbidden_facts_found": forbidden_present,
            })

        # Aggregate
        faithful_scores = [
            1.0 if r["faithfulness"].get("is_faithful", False) else 0.0
            for r in case_results
        ]
        relevance_scores = [
            r["relevance"].get("score", 0.0) for r in case_results
        ]
        groundedness_scores = [
            r["groundedness"].get("groundedness_score", 0.0) for r in case_results
        ]

        return {
            "faithfulness": sum(faithful_scores) / len(faithful_scores),
            "relevance": sum(relevance_scores) / len(relevance_scores),
            "groundedness": sum(groundedness_scores) / len(groundedness_scores),
            "total_cases": len(case_results),
            "case_details": case_results,
        }

    # ------------------------------------------------------------------
    # Pipeline comparison
    # ------------------------------------------------------------------

    def compare_pipelines(
        self,
        pipelines: list[RAGPipeline],
        test_cases: list[RetrievalTestCase],
        labels: Optional[list[str]] = None,
        k: int = 5,
    ) -> dict:
        """Compare multiple RAG pipelines on the same test cases.

        Args:
            pipelines: List of :class:`RAGPipeline` instances.
            test_cases: Shared test cases for all pipelines.
            labels:    Human-readable names for each pipeline.
            k:         Number of results per query.

        Returns:
            Dict mapping pipeline labels to their metric dicts.
        """
        labels = labels or [f"pipeline_{i}" for i in range(len(pipelines))]
        results = {}
        original = self.pipeline
        for label, pipeline in zip(labels, pipelines):
            self.pipeline = pipeline
            results[label] = self.evaluate_retrieval(test_cases, k=k)
        self.pipeline = original
        return results

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, results: dict) -> str:
        """Generate a human-readable evaluation report from *results*.

        Args:
            results: Dict returned by :meth:`evaluate_retrieval` or
                     :meth:`evaluate_generation`.

        Returns:
            Formatted multi-line string.
        """
        lines = ["=" * 60, "RAG EVALUATION REPORT", "=" * 60, ""]

        # Retrieval metrics
        retrieval_keys = {"hit_rate", "mrr", "precision_at_k", "recall_at_k"}
        if retrieval_keys & set(results):
            lines.append("RETRIEVAL METRICS")
            lines.append("-" * 30)
            lines.append(f"  Hit Rate@{results.get('k', '?')}: {results.get('hit_rate', 0):.3f}")
            lines.append(f"  MRR:           {results.get('mrr', 0):.3f}")
            lines.append(f"  Precision@{results.get('k', '?')}: {results.get('precision_at_k', 0):.3f}")
            lines.append(f"  Recall@{results.get('k', '?')}:    {results.get('recall_at_k', 0):.3f}")
            lines.append(f"  Total queries: {results.get('total_queries', '?')}")
            lines.append("")

        # Generation metrics
        gen_keys = {"faithfulness", "relevance", "groundedness"}
        if gen_keys & set(results):
            lines.append("GENERATION METRICS (LLM-as-judge)")
            lines.append("-" * 30)
            lines.append(f"  Faithfulness:  {results.get('faithfulness', 0):.3f}")
            lines.append(f"  Relevance:     {results.get('relevance', 0):.3f}")
            lines.append(f"  Groundedness:  {results.get('groundedness', 0):.3f}")
            lines.append(f"  Total cases:   {results.get('total_cases', '?')}")
            lines.append("")

            # Per-case details
            for case in results.get("case_details", []):
                lines.append(f"  Q: {case['query'][:60]}")
                faith = "✓" if case["faithfulness"].get("is_faithful") else "✗"
                lines.append(f"     Faithful: {faith}  |  Relevance: {case['relevance'].get('score', 0):.2f}")
                if case["expected_facts_missing"]:
                    lines.append(f"     Missing facts: {case['expected_facts_missing']}")
                if case["forbidden_facts_found"]:
                    lines.append(f"     FORBIDDEN facts present: {case['forbidden_facts_found']}")
                lines.append("")

        # Pipeline comparison
        if all(isinstance(v, dict) for v in results.values()):
            lines.append("PIPELINE COMPARISON")
            lines.append("-" * 30)
            header = f"  {'Pipeline':<20} {'HitRate':>8} {'MRR':>8} {'Prec@k':>8} {'Recall@k':>10}"
            lines.append(header)
            lines.append("  " + "-" * 48)
            for label, metrics in results.items():
                if isinstance(metrics, dict) and "hit_rate" in metrics:
                    lines.append(
                        f"  {label:<20} "
                        f"{metrics['hit_rate']:>8.3f} "
                        f"{metrics['mrr']:>8.3f} "
                        f"{metrics['precision_at_k']:>8.3f} "
                        f"{metrics['recall_at_k']:>10.3f}"
                    )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM-as-judge helpers
    # ------------------------------------------------------------------

    def _judge(self, prompt: str) -> dict:
        """Call the LLM judge and parse JSON response."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "failed to parse judge response", "raw": content}

    def _judge_faithfulness(self, documents: str, answer: str) -> dict:
        prompt = LLM_AS_JUDGE_FAITHFULNESS.format(
            documents=documents, answer=answer
        )
        return self._judge(prompt)

    def _judge_relevance(self, question: str, answer: str) -> dict:
        prompt = LLM_AS_JUDGE_RELEVANCE.format(question=question, answer=answer)
        return self._judge(prompt)

    def _judge_groundedness(self, answer: str) -> dict:
        prompt = LLM_AS_JUDGE_GROUNDEDNESS.format(answer=answer)
        return self._judge(prompt)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    from document_chunker import DocumentChunker

    print("=" * 70)
    print("RAG EVALUATOR DEMO")
    print("=" * 70)

    # Build pipeline with sample docs
    embedder = EmbeddingGenerator(model="text-embedding-3-small")
    vector_store = SimpleVectorStore()
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        model="gpt-4o",
        chunk_size=200,
        overlap=40,
        retrieval_k=4,
        similarity_threshold=0.5,
    )

    docs = {
        "return-policy.md": (
            "Our return policy: return any product within 30 days of purchase. "
            "Damaged items may be returned with photo evidence. "
            "Refunds processed within 5-7 business days."
        ),
        "shipping-info.md": (
            "Standard shipping: 3-5 business days, $4.99 (free over $50). "
            "Express shipping: $14.99, 1-2 business days. "
            "Overnight: $29.99, next-day if ordered before 2 PM EST."
        ),
        "faq.md": (
            "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. "
            "All transactions use 256-bit TLS encryption. "
            "Orders can be modified within 1 hour of placement."
        ),
    }

    for source, text in docs.items():
        pipeline.ingest_text(text, metadata={"source": source})

    evaluator = RAGEvaluator(pipeline, model="gpt-4o")

    # --- Retrieval evaluation -------------------------------------------------
    print("\n--- Retrieval Evaluation ---")
    retrieval_cases = [
        RetrievalTestCase("return policy", ["return-policy.md"]),
        RetrievalTestCase("shipping costs", ["shipping-info.md"]),
        RetrievalTestCase("payment methods", ["faq.md"]),
        RetrievalTestCase("express delivery options", ["shipping-info.md"]),
    ]

    retrieval_results = evaluator.evaluate_retrieval(retrieval_cases, k=3)
    print(evaluator.generate_report(retrieval_results))

    # --- Generation evaluation ------------------------------------------------
    print("\n--- Generation Evaluation ---")
    gen_cases = [
        GenerationTestCase(
            query="What is the return window?",
            expected_facts=["30 days"],
            forbidden_facts=["60 days", "90 days"],
        ),
        GenerationTestCase(
            query="How much does express shipping cost?",
            expected_facts=["$14.99"],
        ),
    ]

    gen_results = evaluator.evaluate_generation(gen_cases)
    print(evaluator.generate_report(gen_results))

    # --- Pipeline comparison --------------------------------------------------
    print("\n--- Pipeline Comparison (chunk_size effect) ---")
    vs2 = SimpleVectorStore()
    pipeline2 = RAGPipeline(
        vector_store=vs2,
        embedder=embedder,
        model="gpt-4o",
        chunk_size=100,   # smaller chunks
        overlap=20,
        retrieval_k=4,
        similarity_threshold=0.5,
    )
    for source, text in docs.items():
        pipeline2.ingest_text(text, metadata={"source": source})

    comparison = evaluator.compare_pipelines(
        pipelines=[pipeline, pipeline2],
        test_cases=retrieval_cases,
        labels=["chunk_size=200", "chunk_size=100"],
        k=3,
    )
    print(evaluator.generate_report(comparison))


if __name__ == "__main__":
    _run_demo()
