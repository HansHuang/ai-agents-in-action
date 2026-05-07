"""LangChain RAG pipeline — Chapter 03's from-scratch RAG, now with LangChain.

Builds the identical RAG pipeline from code/python/04-rag-pipeline/rag_pipeline.py
using LangChain's document loaders, vector store, and retrieval chain.

Includes:
  • :class:`LangChainRAGPipeline` — full LangChain implementation
  • :func:`compare_pipelines`     — side-by-side metrics vs. from-scratch RAG
  • :meth:`extract_to_custom`     — four-step incremental extraction from
                                    LangChain back to zero-dependency code

Run:
    python langchain_rag_pipeline.py

See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional LangChain imports (fail gracefully)
# ---------------------------------------------------------------------------

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Sample documents (used when no docs_directory is provided)
# ---------------------------------------------------------------------------

SAMPLE_DOCS: list[dict] = [
    {
        "filename": "rag_architecture.md",
        "text": (
            "Retrieval-Augmented Generation (RAG) has four phases: "
            "Ingest (load, chunk, embed, store), "
            "Retrieve (embed query, search), "
            "Augment (build prompt with retrieved context), "
            "Generate (call LLM with augmented prompt). "
            "RAG reduces hallucination by grounding the LLM in real documents."
        ),
    },
    {
        "filename": "vector_databases.md",
        "text": (
            "Vector databases store high-dimensional embeddings and support "
            "approximate nearest-neighbour (ANN) search. Popular options: "
            "Qdrant, Pinecone, Weaviate, Chroma, and FAISS. "
            "FAISS is Facebook's open-source library — CPU-only but very fast for small datasets. "
            "Qdrant and Pinecone offer managed cloud hosting with filtering."
        ),
    },
    {
        "filename": "langchain_overview.md",
        "text": (
            "LangChain is an open-source framework for building LLM applications. "
            "It provides 700+ integrations, a chain composition API (LCEL), "
            "and LangSmith for observability. "
            "LangGraph extends LangChain with stateful, graph-based workflows. "
            "The API stabilised significantly from version 0.3 onwards in 2024."
        ),
    },
    {
        "filename": "python_history.md",
        "text": (
            "Python was created by Guido van Rossum and first released in 1991. "
            "It emphasises readability and uses significant indentation. "
            "Python 3.0 was released in 2008 and is not backward-compatible with Python 2. "
            "Python became the dominant language for AI/ML by 2020."
        ),
    },
    {
        "filename": "agent_loops.md",
        "text": (
            "An AI agent loop repeatedly calls an LLM, parses tool calls, "
            "executes those tools, and feeds results back until the LLM gives a final answer. "
            "The loop has four parts: perceive, think, act, observe. "
            "The ReAct pattern (Reasoning + Acting) is the most common agent architecture."
        ),
    },
]

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RAGResult:
    """Structured result from a RAG query."""

    answer: str
    sources: list[str]
    retrieved_count: int
    tokens_used: int
    elapsed_ms: float


@dataclass
class PipelineComparison:
    """Side-by-side comparison of LangChain vs from-scratch RAG."""

    query: str
    langchain_answer: str
    scratch_answer: str
    langchain_sources: list[str]
    scratch_sources: list[str]
    langchain_elapsed_ms: float
    scratch_elapsed_ms: float
    langchain_tokens: int
    scratch_tokens: int


# ---------------------------------------------------------------------------
# LangChainRAGPipeline
# ---------------------------------------------------------------------------


class LangChainRAGPipeline:
    """RAG pipeline built with LangChain.

    Functionally identical to the from-scratch RAGPipeline in
    code/python/04-rag-pipeline/rag_pipeline.py, but uses:
      • LangChain document loaders   (Phase 1 — Ingest)
      • LangChain FAISS vector store (Phase 1 — Store / Phase 2 — Retrieve)
      • LangChain create_retrieval_chain (Phase 3 — Augment + Phase 4 — Generate)

    Args:
        docs_directory: Path to a directory of .md/.txt files to ingest.
                        Inline :data:`SAMPLE_DOCS` are used when None.
        model:          OpenAI chat model name.
        retrieval_k:    Number of documents to retrieve per query.
    """

    def __init__(
        self,
        docs_directory: Optional[str] = None,
        model: str = "gpt-4o",
        retrieval_k: int = 3,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required. Install with:\n"
                "  pip install langchain langchain-openai langchain-community faiss-cpu"
            )
        self.docs_directory = docs_directory
        self.model = model
        self.retrieval_k = retrieval_k
        self.llm = ChatOpenAI(model=model)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store: Optional[FAISS] = None
        self.chain = None
        self._documents_count = 0

    # ------------------------------------------------------------------
    # Phase 1 – Ingest
    # ------------------------------------------------------------------

    def ingest(self) -> int:
        """Load, chunk (via LangChain's text splitter), embed, and store docs.

        Uses :attr:`docs_directory` if set; otherwise writes :data:`SAMPLE_DOCS`
        to a temp directory and loads from there.

        Returns:
            Number of document chunks stored.
        """
        docs: list[Document] = []

        if self.docs_directory:
            loader = DirectoryLoader(
                self.docs_directory,
                glob="**/*.{md,txt}",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=False,
            )
            docs = loader.load()
        else:
            # Use inline sample documents
            for sample in SAMPLE_DOCS:
                docs.append(
                    Document(
                        page_content=sample["text"],
                        metadata={"source": sample["filename"]},
                    )
                )

        # Build FAISS index — LangChain handles chunking internally when
        # using from_documents with a text splitter, but for comparability
        # with the from-scratch pipeline we pass whole documents here.
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

        # Build the retrieval chain
        self._build_chain()
        self._documents_count = len(docs)
        return len(docs)

    def _build_chain(self) -> None:
        """Assemble the LangChain retrieval chain."""
        if self.vector_store is None:
            raise RuntimeError("Call ingest() before querying.")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer using ONLY the provided context. "
                    "If the context doesn't contain the answer, say 'I don't have information "
                    "about that in my knowledge base.' Cite sources as [Source: filename].\n\n"
                    "Context:\n{context}",
                ),
                ("human", "{input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(retriever, document_chain)

    # ------------------------------------------------------------------
    # Phase 2–4 — Retrieve, Augment, Generate (all in one chain call)
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """Answer *question* using the ingested knowledge base.

        Args:
            question: Natural-language question.

        Returns:
            :class:`RAGResult` with answer, sources, token usage, and latency.

        Raises:
            RuntimeError: If :meth:`ingest` has not been called.
        """
        if self.chain is None:
            raise RuntimeError("Call ingest() before querying.")

        start = time.monotonic()
        response = self.chain.invoke({"input": question})
        elapsed = (time.monotonic() - start) * 1000

        answer = response.get("answer", "")
        context_docs: list[Document] = response.get("context", [])
        sources = list({
            doc.metadata.get("source", "unknown") for doc in context_docs
        })

        return RAGResult(
            answer=answer,
            sources=sources,
            retrieved_count=len(context_docs),
            tokens_used=0,  # LangChain's chain doesn't surface token counts easily
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Extraction path: incremental migration from LangChain → custom code
    # ------------------------------------------------------------------

    def extract_to_custom(self) -> str:
        """Show the four-step extraction from LangChain to zero-dependency code.

        Each step replaces one LangChain component with custom Python.
        Returns a string showing the diff at each step.

        Returns:
            Multi-line string describing each extraction step.
        """
        return """\
Extraction Path: LangChain → Custom Code
─────────────────────────────────────────

STEP 1 — Replace create_retrieval_chain with custom prompt assembly
  BEFORE (LangChain):
    response = self.chain.invoke({"input": question})
    answer = response["answer"]

  AFTER (custom):
    docs = retriever.get_relevant_documents(question)
    context = "\\n\\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": f"Answer using:\\n{context}"},
        {"role": "user",   "content": question},
    ]
    response = openai_client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content

  BENEFIT: You control prompt format, citation style, and token budget.

─────────────────────────────────────────
STEP 2 — Replace LangChain FAISS vector store with direct client
  BEFORE (LangChain):
    self.vector_store = FAISS.from_documents(docs, self.embeddings)
    results = self.vector_store.similarity_search(query, k=3)

  AFTER (custom SimpleVectorStore):
    from simple_vector_store import SimpleVectorStore
    self.store = SimpleVectorStore()
    for doc in docs:
        emb = openai_client.embeddings.create(input=doc.text, model=embed_model)
        self.store.add(text=doc.text, embedding=emb.data[0].embedding, metadata=doc.metadata)
    results = self.store.search(query_embedding, k=3)

  BENEFIT: No FAISS install required; portable across environments.

─────────────────────────────────────────
STEP 3 — Replace LangChain document loaders with custom loaders
  BEFORE (LangChain):
    loader = DirectoryLoader(path, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

  AFTER (custom):
    docs = []
    for path in Path(directory).rglob("*.md"):
        docs.append({"text": path.read_text(), "source": path.name})

  BENEFIT: Zero extra dependencies; supports any file type with 2 lines of code.

─────────────────────────────────────────
STEP 4 — Complete extraction (zero LangChain dependencies)
  At this point your pipeline is the same RAGPipeline from Chapter 03:
    • document_chunker.py    — custom chunking
    • embedding_generator.py — OpenAI embeddings via SDK
    • simple_vector_store.py — pure-Python cosine similarity
    • rag_pipeline.py        — your orchestration, no framework

  RESULT: The final pipeline is ~200 lines of Python that you fully own,
  understand, and can debug without reading LangChain's source code.

─────────────────────────────────────────
WHEN TO STAY ON LANGCHAIN:
  • You need PDF, Word, or web loader support fast → LangChain loaders save days
  • You're using Chroma/Pinecone/Weaviate → the connectors are well-tested
  • Your team knows LangChain → don't migrate without a reason

WHEN TO EXTRACT:
  • The framework obscures a bug you can't diagnose
  • You need fine-grained control over prompt format or token budget
  • You're deploying to an environment where pip install langchain is painful
"""


# ---------------------------------------------------------------------------
# Comparison with from-scratch RAG pipeline
# ---------------------------------------------------------------------------


def compare_pipelines(
    queries: Optional[list[str]] = None,
) -> list[PipelineComparison]:
    """Run *queries* through both pipelines and return comparison metrics.

    Args:
        queries: List of questions to ask. Uses ten default questions if None.

    Returns:
        List of :class:`PipelineComparison` results, one per query.
    """
    if queries is None:
        queries = [
            "What are the four phases of a RAG pipeline?",
            "What is FAISS and when would you use it?",
            "How does LangChain's LCEL work?",
            "When was Python first released?",
            "What is the ReAct agent pattern?",
            "What vector databases are popular?",
            "What does RAG stand for?",
            "What is LangGraph used for?",
            "How is Python 3 different from Python 2?",
            "What are the parts of an agent loop?",
        ]

    results: list[PipelineComparison] = []

    # LangChain pipeline
    lc_pipeline = LangChainRAGPipeline()
    lc_pipeline.ingest()

    # From-scratch pipeline (optional — falls back gracefully)
    scratch_pipeline = None
    scratch_dir = Path(__file__).parent.parent / "04-rag-pipeline"
    if scratch_dir.is_dir():
        sys.path.insert(0, str(scratch_dir))
        try:
            from rag_pipeline import RAGPipeline  # type: ignore
            from simple_vector_store import SimpleVectorStore  # type: ignore
            from embedding_generator import EmbeddingGenerator  # type: ignore

            store = SimpleVectorStore()
            embedder = EmbeddingGenerator()
            scratch_pipeline = RAGPipeline(
                vector_store=store, embedder=embedder, model="gpt-4o"
            )
            # Ingest the same sample docs
            for sample in SAMPLE_DOCS:
                scratch_pipeline.ingest_text(
                    sample["text"], metadata={"source": sample["filename"]}
                )
        except Exception:
            pass

    for question in queries:
        lc_result = lc_pipeline.query(question)

        if scratch_pipeline is not None:
            try:
                t0 = time.monotonic()
                sc = scratch_pipeline.query(question)
                sc_elapsed = (time.monotonic() - t0) * 1000
                scratch_answer = sc.answer
                scratch_sources = sc.sources
                scratch_tokens = sc.tokens_used
            except Exception:
                scratch_answer = "(from-scratch pipeline unavailable)"
                scratch_sources = []
                scratch_tokens = 0
                sc_elapsed = 0.0
        else:
            scratch_answer = "(from-scratch pipeline not on sys.path)"
            scratch_sources = []
            scratch_tokens = 0
            sc_elapsed = 0.0

        results.append(
            PipelineComparison(
                query=question,
                langchain_answer=lc_result.answer,
                scratch_answer=scratch_answer,
                langchain_sources=lc_result.sources,
                scratch_sources=scratch_sources,
                langchain_elapsed_ms=lc_result.elapsed_ms,
                scratch_elapsed_ms=sc_elapsed,
                langchain_tokens=lc_result.tokens_used,
                scratch_tokens=scratch_tokens,
            )
        )

    if scratch_dir.is_dir() and str(scratch_dir) in sys.path:
        sys.path.remove(str(scratch_dir))

    return results


def print_comparison_matrix(comparisons: list[PipelineComparison]) -> None:
    """Print a formatted comparison matrix for *comparisons*."""
    print("\n" + "=" * 80)
    print("RAG PIPELINE COMPARISON MATRIX")
    print("=" * 80)
    print(f"{'#':<3} {'Query (truncated)':<35} {'LC (ms)':>8} {'SC (ms)':>8} {'LC sources':>20}")
    print("-" * 80)

    for i, c in enumerate(comparisons, 1):
        query_short = c.query[:33] + ".." if len(c.query) > 35 else c.query
        lc_src = ", ".join(c.langchain_sources[:2]) or "—"
        print(
            f"{i:<3} {query_short:<35} "
            f"{c.langchain_elapsed_ms:>8.0f} "
            f"{c.scratch_elapsed_ms:>8.0f} "
            f"{lc_src:>20}"
        )

    print("=" * 80)
    avg_lc = sum(c.langchain_elapsed_ms for c in comparisons) / len(comparisons)
    avg_sc = sum(c.scratch_elapsed_ms for c in comparisons) / len(comparisons)
    print(f"{'Average':>41} {avg_lc:>8.0f} {avg_sc:>8.0f}")
    print()


def show_pdf_extension(lc_pipeline: LangChainRAGPipeline) -> None:
    """Show how to add PDF support: LangChain vs custom code.

    Adding a new document type is a common maintenance task.
    """
    print("\nAdding PDF support:")
    print()
    print("  LangChain approach (1 line of change):")
    print("    from langchain_community.document_loaders import PyPDFLoader")
    print("    loader = PyPDFLoader('document.pdf')")
    print("    docs = loader.load_and_split()")
    print()
    print("  From-scratch approach (5 lines, pypdf only):")
    print("    import pypdf")
    print("    reader = pypdf.PdfReader('document.pdf')")
    print("    text = ' '.join(page.extract_text() for page in reader.pages)")
    print("    scratch_pipeline.ingest_text(text, metadata={'source': 'document.pdf'})")
    print()
    print("  Verdict: LangChain saves a few lines but adds a dependency.")
    print("  For > 5 file types, LangChain's loaders justify the install.")


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the demo: ingest documents, compare both pipelines, show PDF path."""
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not installed. Install with:")
        print("  pip install langchain langchain-openai langchain-community faiss-cpu")
        return

    print("LangChain RAG Pipeline Demo")
    print("=" * 50)

    # Ingest
    pipeline = LangChainRAGPipeline()
    doc_count = pipeline.ingest()
    print(f"Ingested {doc_count} documents into FAISS vector store.\n")

    # Single query demo
    question = "What are the four phases of a RAG pipeline?"
    print(f"Query: {question!r}")
    result = pipeline.query(question)
    print(f"Answer: {result.answer[:200]}")
    print(f"Sources: {result.sources}")
    print(f"Elapsed: {result.elapsed_ms:.0f} ms\n")

    # Extraction path
    print(pipeline.extract_to_custom())

    # PDF extension demo (no actual PDF needed — shows the pattern)
    show_pdf_extension(pipeline)

    # Full comparison matrix (calls the real API — skipped in CI)
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_COMPARISON"):
        comparisons = compare_pipelines()
        print_comparison_matrix(comparisons)


if __name__ == "__main__":
    main()
