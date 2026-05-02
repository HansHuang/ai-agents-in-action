"""Complete Retrieval-Augmented Generation (RAG) pipeline.

Four-phase pipeline:
    1. INGEST  — Load → Chunk → Embed → Store
    2. RETRIEVE — Embed query → Search → Filter by threshold
    3. AUGMENT  — Build prompt with retrieved context
    4. GENERATE — Call LLM with augmented prompt

See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from document_chunker import DocumentChunker
from embedding_generator import EmbeddingGenerator
from simple_vector_store import SimpleVectorStore

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided documents.

Rules:
1. Answer ONLY using information from the documents below.
2. If the documents don't contain the answer, say exactly: \
"I don't have information about that in my knowledge base."
3. Cite sources using [Source: filename] format.
4. If multiple documents are relevant, synthesize information from all of them.
5. If documents contain conflicting information, note the conflict and cite both sources.
6. Do not use any knowledge outside the provided documents.

Documents:
{document_context}

When answering, structure your response as:
1. Direct answer to the question
2. Supporting details from the documents
3. Source citations
"""

CITATION_SUFFIX = (
    "\n\nIMPORTANT: You MUST cite every factual claim with [Source: <filename>]. "
    "Do not make any statement without an explicit citation."
)

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".text"}


# ---------------------------------------------------------------------------
# RAGResponse
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    """Structured result returned by :meth:`RAGPipeline.query`."""

    answer: str
    sources: list[str]
    retrieved_chunks: list[dict]
    similarity_scores: list[float]
    tokens_used: int
    pipeline_steps: list[str]


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline.

    Phases:
        1. INGEST:    Load → Chunk → Embed → Store
        2. RETRIEVE:  Embed query → Search → Filter by threshold
        3. AUGMENT:   Build prompt with retrieved context
        4. GENERATE:  Call LLM with augmented prompt

    Args:
        vector_store:         Pre-created :class:`SimpleVectorStore`.
        embedder:             Pre-created :class:`EmbeddingGenerator`.
        model:                OpenAI chat model name.
        chunk_size:           Target chunk size in tokens.
        overlap:              Token overlap between adjacent chunks.
        retrieval_k:          Default number of results to return.
        similarity_threshold: Default minimum cosine-similarity score.
    """

    def __init__(
        self,
        vector_store: SimpleVectorStore,
        embedder: EmbeddingGenerator,
        model: str = "gpt-4o",
        chunk_size: int = 256,
        overlap: int = 50,
        retrieval_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retrieval_k = retrieval_k
        self.similarity_threshold = similarity_threshold

        self._chunker = DocumentChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            strategy="semantic",
        )
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Phase 1 – Ingest
    # ------------------------------------------------------------------

    def ingest_directory(self, directory: str) -> dict:
        """Load, chunk, embed, and store all documents in *directory*.

        Args:
            directory: Path to a directory containing text files.

        Returns:
            ``{"documents_processed": int, "chunks_created": int, "errors": list}``
        """
        directory_path = Path(directory)
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory!r}")

        documents_processed = 0
        chunks_created = 0
        errors: list[str] = []

        for file_path in sorted(directory_path.iterdir()):
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
                chunks_created += self.ingest_text(
                    text,
                    metadata={"source": file_path.name, "path": str(file_path)},
                )
                documents_processed += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{file_path.name}: {exc}")

        return {
            "documents_processed": documents_processed,
            "chunks_created": chunks_created,
            "errors": errors,
        }

    def ingest_text(self, text: str, metadata: Optional[dict] = None) -> int:
        """Ingest a single text document.

        Chunks the text, embeds each chunk, and stores them in the vector
        store.  All chunks share the same *metadata*.

        Args:
            text:     Raw document text.
            metadata: Optional key-value pairs (e.g. ``{"source": "faq.md"}``).

        Returns:
            Number of chunks created and stored.
        """
        metadata = metadata or {}
        chunks = self._chunker.chunk(text)

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            self.vector_store.add(
                text=chunk.text,
                embedding=embedding,
                metadata=chunk_meta,
            )

        return len(chunks)

    # ------------------------------------------------------------------
    # Phase 2 – Retrieve
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        question: str,
        k: int,
        threshold: float,
        steps: list[str],
    ) -> list[dict]:
        """Embed the query and search the vector store."""
        steps.append("RETRIEVE: embedding query")
        query_embedding = self.embedder.embed(question)

        steps.append(f"RETRIEVE: searching vector store (k={k}, threshold={threshold})")
        results = self.vector_store.search_with_threshold(
            query_embedding=query_embedding,
            threshold=threshold,
            k=k,
        )
        steps.append(f"RETRIEVE: found {len(results)} chunk(s) above threshold")
        return results

    # ------------------------------------------------------------------
    # Phase 3 – Augment
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        retrieved: list[dict],
        extra_system_suffix: str = "",
    ) -> list[dict]:
        """Build the system+user message list for the LLM call."""
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            source = doc["metadata"].get("source", "unknown")
            context_parts.append(f"[Document {i} — Source: {source}]\n{doc['text']}")

        document_context = "\n\n---\n\n".join(context_parts)
        system_prompt = RAG_SYSTEM_PROMPT.format(document_context=document_context)
        if extra_system_suffix:
            system_prompt += extra_system_suffix

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    # ------------------------------------------------------------------
    # Phase 4 – Generate (internal helper)
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[dict],
        steps: list[str],
    ) -> tuple[str, int]:
        """Call the LLM and return (answer_text, tokens_used)."""
        steps.append(f"GENERATE: calling {self.model}")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        steps.append(f"GENERATE: received {tokens} total tokens")
        return answer, tokens

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RAGResponse:
        """Answer *question* using the RAG pipeline.

        Args:
            question:  Natural-language question.
            k:         Override the default number of retrieved chunks.
            threshold: Override the default similarity threshold.

        Returns:
            :class:`RAGResponse` with answer, sources, scores, and debug steps.
        """
        k = k if k is not None else self.retrieval_k
        threshold = threshold if threshold is not None else self.similarity_threshold
        steps: list[str] = []

        steps.append("INGEST: (already complete)")
        results = self._retrieve(question, k=k, threshold=threshold, steps=steps)

        if not results:
            steps.append("AUGMENT: no results above threshold — short-circuit")
            return RAGResponse(
                answer="I don't have information about that in my knowledge base.",
                sources=[],
                retrieved_chunks=[],
                similarity_scores=[],
                tokens_used=0,
                pipeline_steps=steps,
            )

        steps.append(f"AUGMENT: building prompt with {len(results)} document(s)")
        messages = self._build_messages(question, results)

        answer, tokens = self._generate(messages, steps)

        sources = list(dict.fromkeys(
            r["metadata"].get("source", "unknown") for r in results
        ))
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=[
                {"text": r["text"], "score": r["score"], "metadata": r["metadata"]}
                for r in results
            ],
            similarity_scores=[r["score"] for r in results],
            tokens_used=tokens,
            pipeline_steps=steps,
        )

    def query_with_citations(
        self,
        question: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RAGResponse:
        """Same as :meth:`query` but enforces citation format in the answer."""
        k = k if k is not None else self.retrieval_k
        threshold = threshold if threshold is not None else self.similarity_threshold
        steps: list[str] = []

        steps.append("INGEST: (already complete)")
        results = self._retrieve(question, k=k, threshold=threshold, steps=steps)

        if not results:
            steps.append("AUGMENT: no results above threshold — short-circuit")
            return RAGResponse(
                answer="I don't have information about that in my knowledge base.",
                sources=[],
                retrieved_chunks=[],
                similarity_scores=[],
                tokens_used=0,
                pipeline_steps=steps,
            )

        steps.append(f"AUGMENT: building citation-enforced prompt with {len(results)} document(s)")
        messages = self._build_messages(question, results, extra_system_suffix=CITATION_SUFFIX)

        answer, tokens = self._generate(messages, steps)

        sources = list(dict.fromkeys(
            r["metadata"].get("source", "unknown") for r in results
        ))
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=[
                {"text": r["text"], "score": r["score"], "metadata": r["metadata"]}
                for r in results
            ],
            similarity_scores=[r["score"] for r in results],
            tokens_used=tokens,
            pipeline_steps=steps,
        )

    def batch_query(self, questions: list[str]) -> list[RAGResponse]:
        """Answer multiple questions.

        Each question is processed independently via :meth:`query`.

        Args:
            questions: List of natural-language questions.

        Returns:
            List of :class:`RAGResponse` objects in the same order.
        """
        return [self.query(q) for q in questions]

    def add_document(self, text: str, metadata: Optional[dict] = None) -> None:
        """Add a single document to the knowledge base (incremental ingest).

        Args:
            text:     Raw document text.
            metadata: Optional metadata dict.
        """
        self.ingest_text(text, metadata=metadata)

    def remove_document(self, source_id: str) -> int:
        """Remove all chunks whose ``metadata["source"]`` matches *source_id*.

        Args:
            source_id: The ``source`` value in chunk metadata.

        Returns:
            Number of chunks removed.
        """
        ids_to_remove = [
            doc["id"]
            for doc in self.vector_store._documents
            if doc.get("metadata", {}).get("source") == source_id
        ]
        for doc_id in ids_to_remove:
            self.vector_store.delete(doc_id)
        return len(ids_to_remove)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """Demonstrate the full RAG pipeline with sample company documents."""

    # --- Sample in-memory documents ------------------------------------------
    SAMPLE_DOCS = {
        "return-policy.md": """\
# Return Policy

Our return policy allows customers to return any product within 30 days of purchase.
Items must be in their original condition with all packaging intact.
Damaged items may be returned within 30 days of delivery with photo evidence of the damage.
Digital downloads are non-refundable once downloaded.
To initiate a return, contact support@example.com with your order number and reason.
Refunds are processed within 5-7 business days after we receive the returned item.
""",
        "shipping-info.md": """\
# Shipping Information

Standard shipping takes 3-5 business days and costs $4.99 for orders under $50.
Free standard shipping is available on all orders over $50.
Express shipping is available for $14.99 and delivers within 1-2 business days.
Overnight shipping costs $29.99 and guarantees next-day delivery if ordered before 2 PM EST.
International shipping is available to 40+ countries; rates and times vary by destination.
Orders are processed within 1 business day of being placed.
""",
        "faq.md": """\
# Frequently Asked Questions

Q: Can I change my order after placing it?
A: Orders can be modified within 1 hour of placement by contacting support@example.com.

Q: Do you offer price matching?
A: Yes, we match prices from authorised retailers. Submit a price-match request within 7 days of purchase.

Q: What payment methods do you accept?
A: We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.

Q: Is my payment information secure?
A: Yes, all transactions use 256-bit TLS encryption. We never store raw card numbers.
""",
        "products.md": """\
# Product Catalogue

## WidgetPro 3000
The WidgetPro 3000 is our flagship product priced at $199.99.
It includes a 2-year manufacturer warranty and free tech support for the first year.
Compatible with Windows 10/11, macOS 12+, and Linux (Ubuntu 22.04+).

## WidgetLite
The WidgetLite is our budget option at $49.99 with a 1-year warranty.
It supports Windows 10/11 and macOS 12+ only.

## WidgetPro Accessories
Replacement cables are $9.99. Carrying cases are $24.99. Extended warranties add $29.99/year.
""",
    }

    print("=" * 70)
    print("RAG PIPELINE DEMO")
    print("=" * 70)

    # --- Build pipeline -------------------------------------------------------
    embedder = EmbeddingGenerator(model="text-embedding-3-small")
    vector_store = SimpleVectorStore()
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        model="gpt-4o",
        chunk_size=200,
        overlap=40,
        retrieval_k=4,
        similarity_threshold=0.6,
    )

    # --- Ingest ---------------------------------------------------------------
    print("\n--- Ingesting sample documents ---")
    for source_name, text in SAMPLE_DOCS.items():
        n = pipeline.ingest_text(text, metadata={"source": source_name})
        print(f"  {source_name}: {n} chunk(s)")
    print(f"  Total stored: {vector_store.count()} chunks")

    # --- Queries --------------------------------------------------------------
    QUERIES = [
        (
            "1. Simple factual query",
            "How long do I have to return a damaged item?",
            "query",
        ),
        (
            "2. Multi-document synthesis",
            "If I order a $60 item and want it tomorrow, what are my shipping options and total cost?",
            "query",
        ),
        (
            "3. No relevant documents (should say I don't know)",
            "What is the capital of France and its population?",
            "query",
        ),
        (
            "4. Citation-enforced query",
            "What is the price of the WidgetPro 3000 and what warranty does it include?",
            "query_with_citations",
        ),
        (
            "5. Below-threshold query (high threshold override)",
            "What is quantum entanglement?",
            "high_threshold",
        ),
    ]

    for label, question, mode in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"Q: {question}")
        print("-" * 70)

        if mode == "high_threshold":
            response = pipeline.query(question, threshold=0.95)
        elif mode == "query_with_citations":
            response = pipeline.query_with_citations(question)
        else:
            response = pipeline.query(question)

        print("Pipeline steps:")
        for step in response.pipeline_steps:
            print(f"  → {step}")

        print(f"\nAnswer:\n{response.answer}")
        print(f"\nSources: {response.sources}")
        print(f"Scores:  {[round(s, 3) for s in response.similarity_scores]}")
        print(f"Tokens:  {response.tokens_used}")


if __name__ == "__main__":
    _run_demo()
