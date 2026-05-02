"""Advanced retrieval techniques for RAG pipelines.

Implements three strategies that improve over naive nearest-neighbour search:

- **HyDE** (Hypothetical Document Embeddings): generate a hypothetical answer,
  then search with it — answers are more similar to documents than short
  questions are.
- **Multi-query**: rephrase the question several ways, retrieve for each, and
  merge unique results.
- **Decompose-and-retrieve**: break a complex question into sub-questions and
  retrieve for each independently.
- **Contextual retrieve**: incorporate prior conversation turns to enrich the
  search query.

See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
"""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from embedding_generator import EmbeddingGenerator
from simple_vector_store import SimpleVectorStore


class AdvancedRetriever:
    """Advanced retrieval techniques that improve RAG accuracy.

    Args:
        vector_store: Pre-populated :class:`SimpleVectorStore`.
        embedder:     :class:`EmbeddingGenerator` — same model used at ingest time.
        model:        OpenAI chat model for query generation steps.
    """

    def __init__(
        self,
        vector_store: SimpleVectorStore,
        embedder: EmbeddingGenerator,
        model: str = "gpt-4o",
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.model = model
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Standard baseline
    # ------------------------------------------------------------------

    def standard_retrieve(self, question: str, k: int = 5) -> list[dict]:
        """Basic nearest-neighbour retrieval (baseline for comparison).

        Args:
            question: Natural-language query.
            k:        Number of results to return.

        Returns:
            List of ``{"text", "score", "metadata"}`` dicts.
        """
        embedding = self.embedder.embed(question)
        return self.vector_store.search(embedding, k=k)

    # ------------------------------------------------------------------
    # HyDE
    # ------------------------------------------------------------------

    def hyde_retrieve(self, question: str, k: int = 5) -> list[dict]:
        """Hypothetical Document Embeddings retrieval.

        1. Ask the LLM to write a hypothetical answer to *question*.
        2. Embed the hypothetical answer (not the question).
        3. Search the vector store with that embedding.

        The hypothesis is not shown to the user — it only guides retrieval.
        Useful when user queries are short or use different vocabulary than
        the indexed documents.

        Args:
            question: Natural-language query.
            k:        Number of results to return.

        Returns:
            List of ``{"text", "score", "metadata"}`` dicts.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write a detailed, factual answer to the following question. "
                        f"Use the style of a technical document or FAQ entry.\n\n"
                        f"Question: {question}"
                    ),
                }
            ],
            temperature=0.3,
        )
        hypothetical_answer = response.choices[0].message.content or ""
        hypothesis_embedding = self.embedder.embed(hypothetical_answer)
        results = self.vector_store.search(hypothesis_embedding, k=k)
        for r in results:
            r["_retrieval_method"] = "hyde"
        return results

    # ------------------------------------------------------------------
    # Multi-query
    # ------------------------------------------------------------------

    def multi_query_retrieve(
        self,
        question: str,
        n_queries: int = 3,
        k_per_query: int = 3,
    ) -> list[dict]:
        """Generate multiple search queries, retrieve for each, deduplicate.

        Different phrasings find different relevant documents. Merging results
        expands coverage.

        Args:
            question:    Original user question.
            n_queries:   Number of alternative queries to generate.
            k_per_query: Results per generated query.

        Returns:
            Deduplicated list of results, sorted by score descending.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Generate exactly {n_queries} alternative search queries for the "
                        f"following question. Each query should use different vocabulary and "
                        f"phrasing to maximise retrieval coverage.\n\n"
                        f"Original question: {question}\n\n"
                        f'Output a JSON array of strings, e.g. ["query1", "query2"].'
                    ),
                }
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "[]"
        try:
            parsed = json.loads(raw)
            # Handle both {"queries": [...]} and bare [...]
            if isinstance(parsed, list):
                queries = parsed
            else:
                queries = next(
                    (v for v in parsed.values() if isinstance(v, list)), []
                )
        except (json.JSONDecodeError, StopIteration):
            queries = [question]

        # Retrieve for each query, deduplicate by text
        seen: dict[str, dict] = {}
        for q in queries[:n_queries]:
            emb = self.embedder.embed(q)
            results = self.vector_store.search(emb, k=k_per_query)
            for r in results:
                if r["text"] not in seen or r["score"] > seen[r["text"]]["score"]:
                    r["_retrieval_method"] = "multi_query"
                    seen[r["text"]] = r

        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    # ------------------------------------------------------------------
    # Decompose-and-retrieve
    # ------------------------------------------------------------------

    def decompose_and_retrieve(
        self,
        complex_question: str,
        k: int = 5,
    ) -> list[dict]:
        """Break a complex question into sub-questions, retrieve for each.

        Useful for comparison questions ("Compare X and Y") or multi-hop
        queries that require information from different documents.

        Args:
            complex_question: A question that may span multiple topics.
            k:                Total number of results to return.

        Returns:
            Deduplicated, merged results from all sub-question retrievals.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Break the following question into simple, focused sub-questions. "
                        "Each sub-question should be answerable from a single document.\n\n"
                        f"Question: {complex_question}\n\n"
                        'Output a JSON array of strings, e.g. {"sub_questions": ["q1", "q2"]}.'
                    ),
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                sub_questions = parsed
            else:
                sub_questions = next(
                    (v for v in parsed.values() if isinstance(v, list)),
                    [complex_question],
                )
        except (json.JSONDecodeError, StopIteration):
            sub_questions = [complex_question]

        seen: dict[str, dict] = {}
        for sq in sub_questions:
            emb = self.embedder.embed(sq)
            results = self.vector_store.search(emb, k=3)
            for r in results:
                if r["text"] not in seen or r["score"] > seen[r["text"]]["score"]:
                    r["_retrieval_method"] = "decompose"
                    r["_sub_question"] = sq
                    seen[r["text"]] = r

        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:k]

    # ------------------------------------------------------------------
    # Contextual retrieve
    # ------------------------------------------------------------------

    def contextual_retrieve(
        self,
        question: str,
        conversation_history: list[dict],
        k: int = 5,
    ) -> list[dict]:
        """Use conversation history to enrich the retrieval query.

        Prior turns may contain pronouns or implicit references that make
        the current question ambiguous. This method resolves the question
        against the history before searching.

        Args:
            question:             The current user question.
            conversation_history: List of ``{"role": ..., "content": ...}`` dicts.
            k:                    Number of results to return.

        Returns:
            List of ``{"text", "score", "metadata"}`` dicts.
        """
        if not conversation_history:
            return self.standard_retrieve(question, k=k)

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in conversation_history[-6:]   # last 3 exchanges
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Given the following conversation, rewrite the last question "
                        "as a fully self-contained search query. "
                        "Resolve all pronouns and implicit references.\n\n"
                        f"Conversation:\n{history_text}\n\n"
                        f"Last question: {question}\n\n"
                        "Output only the rewritten query, nothing else."
                    ),
                }
            ],
            temperature=0,
        )
        enriched_query = (response.choices[0].message.content or question).strip()
        emb = self.embedder.embed(enriched_query)
        results = self.vector_store.search(emb, k=k)
        for r in results:
            r["_retrieval_method"] = "contextual"
            r["_enriched_query"] = enriched_query
        return results

    # ------------------------------------------------------------------
    # Compare methods
    # ------------------------------------------------------------------

    def compare_methods(
        self,
        question: str,
        k: int = 5,
    ) -> dict:
        """Run all retrieval methods on *question* and compare results.

        Args:
            question: Natural-language query.
            k:        Number of results per method.

        Returns:
            Dict with keys ``"standard"``, ``"hyde"``, ``"multi_query"``,
            ``"decompose"`` mapping to their respective result lists, plus
            ``"overlap_analysis"`` showing which texts appeared in multiple methods.
        """
        standard = self.standard_retrieve(question, k=k)
        hyde = self.hyde_retrieve(question, k=k)
        multi = self.multi_query_retrieve(question, k_per_query=k)
        decompose = self.decompose_and_retrieve(question, k=k)

        methods = {
            "standard": standard,
            "hyde": hyde,
            "multi_query": multi,
            "decompose": decompose,
        }

        # Overlap analysis
        text_to_methods: dict[str, list[str]] = {}
        for method_name, results in methods.items():
            for r in results:
                text_to_methods.setdefault(r["text"], []).append(method_name)

        overlap = {
            text[:80]: found_in
            for text, found_in in text_to_methods.items()
            if len(found_in) > 1
        }
        unique = {
            method_name: len([
                r for r in results
                if len(text_to_methods.get(r["text"], [])) == 1
            ])
            for method_name, results in methods.items()
        }

        return {
            **methods,
            "overlap_analysis": overlap,
            "unique_results_per_method": unique,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    print("=" * 70)
    print("ADVANCED RETRIEVER DEMO")
    print("=" * 70)

    # Build knowledge base
    embedder = EmbeddingGenerator(model="text-embedding-3-small")
    vector_store = SimpleVectorStore()

    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        chunk_size=200,
        overlap=30,
    )

    DOCS = {
        "return-policy.md": (
            "Return Policy: Customers may return items within 30 days of purchase. "
            "Damaged goods require photo documentation and are refunded within 7 days. "
            "Digital products are non-refundable."
        ),
        "shipping-info.md": (
            "Shipping: Standard delivery 3-5 days at $4.99 (free over $50). "
            "Express 1-2 days at $14.99. Overnight next-day at $29.99 before 2 PM. "
            "International shipping available to 40+ countries."
        ),
        "faq.md": (
            "FAQ: Payment methods: Visa, Mastercard, PayPal, Apple Pay. "
            "Price matching available within 7 days. "
            "Order modifications allowed within 1 hour of placement."
        ),
        "products.md": (
            "Products: WidgetPro 3000 at $199.99 with 2-year warranty. "
            "WidgetLite at $49.99 with 1-year warranty. "
            "Accessories: cables $9.99, cases $24.99, extended warranty $29.99/year."
        ),
    }
    for source, text in DOCS.items():
        pipeline.ingest_text(text, metadata={"source": source})

    retriever = AdvancedRetriever(vector_store, embedder, model="gpt-4o")

    # --- Compare methods on one query -----------------------------------------
    question = "How long does it take to get a refund for a damaged product?"
    print(f"\nQuery: {question!r}")
    print("-" * 70)

    comparison = retriever.compare_methods(question, k=3)

    for method in ("standard", "hyde", "multi_query", "decompose"):
        results = comparison[method]
        print(f"\n{method.upper()} ({len(results)} result(s)):")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['metadata'].get('source', '?')} — {r['text'][:80]!r}")

    print("\nOverlap (texts found by multiple methods):")
    for snippet, methods in comparison["overlap_analysis"].items():
        print(f"  {methods} → {snippet!r}")

    print("\nUnique results per method:")
    for method, count in comparison["unique_results_per_method"].items():
        print(f"  {method}: {count} unique")

    # --- Decompose a comparison question ------------------------------------
    print("\n" + "=" * 70)
    comparison_q = "Compare the WidgetPro 3000 and the WidgetLite in terms of price and warranty."
    print(f"Decompose query: {comparison_q!r}")
    decomposed = retriever.decompose_and_retrieve(comparison_q, k=4)
    print(f"Found {len(decomposed)} result(s) via decomposition:")
    for r in decomposed:
        print(
            f"  [{r['score']:.3f}] sub-q={r.get('_sub_question', '?')[:50]!r} "
            f"| {r['metadata'].get('source', '?')}"
        )

    # --- Contextual retrieval -------------------------------------------------
    print("\n" + "=" * 70)
    history = [
        {"role": "user", "content": "Tell me about the WidgetPro 3000."},
        {"role": "assistant", "content": "The WidgetPro 3000 costs $199.99 with a 2-year warranty."},
    ]
    followup = "What accessories are available for it?"
    print(f"Contextual query: {followup!r} (with history)")
    ctx_results = retriever.contextual_retrieve(followup, history, k=3)
    for r in ctx_results:
        print(
            f"  [{r['score']:.3f}] enriched_query={r.get('_enriched_query', '?')[:60]!r} "
            f"| {r['metadata'].get('source', '?')}"
        )


if __name__ == "__main__":
    _run_demo()
