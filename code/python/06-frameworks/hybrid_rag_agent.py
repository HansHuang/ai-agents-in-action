"""Hybrid RAG agent: LangChain for commodity, custom code for agent logic.

Architecture:
    FRAMEWORK (commodity parts):
        • DirectoryLoader   — load .md files from a directory
        • OpenAIEmbeddings  — embed documents and queries
        • Chroma / FAISS    — vector storage and ANN search

    YOUR CODE (differentiated parts):
        • Agent loop        — you control the orchestration
        • ContextAssembler  — you control prompt quality
        • MemoryManager     — you control conversation budget
        • TokenTracker      — you control cost visibility

The demo shows:
    1. Ingest a directory of .md files via LangChain
    2. Query the hybrid agent
    3. Swap the vector store from FAISS to SimpleVectorStore (zero logic changes)
    4. Report custom-code lines vs. framework-code lines

See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
"""

from __future__ import annotations

import os
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional LangChain imports
# ---------------------------------------------------------------------------

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document

    try:
        from langchain_community.vectorstores import FAISS as LCVectorStore
        _VECTOR_BACKEND = "FAISS"
    except ImportError:
        from langchain_community.vectorstores import Chroma as LCVectorStore  # type: ignore
        _VECTOR_BACKEND = "Chroma"

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    _VECTOR_BACKEND = "none"

_LLM_MODEL = "gpt-4o-mini"
_EMBED_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class HybridResult:
    """Structured result from :meth:`HybridRAGAgent.query`."""

    answer: str
    sources: list[str]
    tokens: int
    retrieval_ms: float
    generation_ms: float


# ---------------------------------------------------------------------------
# VectorStoreProtocol — your interface, not LangChain's
# ---------------------------------------------------------------------------


class VectorStoreProtocol(Protocol):
    """Minimal interface your agent depends on.

    Both the LangChain-backed store and the custom SimpleVectorStore satisfy
    this protocol, making the swap a one-liner.
    """

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k results as dicts with 'text' and 'metadata' keys."""
        ...


# ---------------------------------------------------------------------------
# LangChainVectorStore — wraps LangChain behind your protocol
# ---------------------------------------------------------------------------


class LangChainVectorStore:
    """Wraps a LangChain vector store behind :class:`VectorStoreProtocol`.

    The agent never imports LangChain directly — only this wrapper does.
    Replacing the backend (FAISS → Chroma → Qdrant) touches only this class.
    """

    def __init__(self, lc_store) -> None:
        self._store = lc_store
        self.backend_name: str = _VECTOR_BACKEND

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        docs = self._store.similarity_search(query, k=k)
        return [
            {"text": d.page_content, "metadata": d.metadata}
            for d in docs
        ]

    @classmethod
    def from_documents(
        cls,
        documents: list["Document"],
        embeddings,
    ) -> "LangChainVectorStore":
        store = LCVectorStore.from_documents(documents, embeddings)
        return cls(store)


# ---------------------------------------------------------------------------
# SimpleVectorStore — pure Python, no framework dependency
# ---------------------------------------------------------------------------


class SimpleVectorStore:
    """In-memory cosine-similarity vector store (no framework required).

    Drop-in replacement for :class:`LangChainVectorStore`.  Satisfies
    :class:`VectorStoreProtocol` exactly.
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client
        self._docs: list[dict] = []
        self._embeddings: list[list[float]] = []
        self.backend_name = "SimpleVectorStore"

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[dict]) -> int:
        """Embed and store *docs* (list of {text, metadata} dicts)."""
        texts = [d["text"] for d in docs]
        resp = self._client.embeddings.create(model=_EMBED_MODEL, input=texts)
        for doc, item in zip(docs, resp.data):
            self._docs.append(doc)
            self._embeddings.append(item.embedding)
        return len(docs)

    # ------------------------------------------------------------------
    # Retrieval (satisfies VectorStoreProtocol)
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        if not self._docs:
            return []
        resp = self._client.embeddings.create(model=_EMBED_MODEL, input=[query])
        q_emb = np.array(resp.data[0].embedding)
        matrix = np.array(self._embeddings)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_emb)
        scores = matrix @ q_emb / np.where(norms == 0, 1, norms)
        top_k = int(min(k, len(self._docs)))
        indices = np.argsort(scores)[::-1][:top_k]
        return [self._docs[i] for i in indices]


# ---------------------------------------------------------------------------
# ContextAssembler — YOUR logic, no framework dependency
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Assemble a system prompt from retrieved documents.

    You own this logic.  Frameworks cannot improve on it because improving it
    requires understanding your domain, your users, and your quality bar.
    """

    _TEMPLATES: dict[str, str] = {
        "rag_query": (
            "Answer the question using ONLY the documents below.\n"
            "If the answer is not in the documents, say so explicitly.\n"
            "Cite sources with [Source: filename].\n\n"
            "{documents}\n"
        ),
    }

    def assemble(
        self,
        template: str,
        variables: dict,
        sources: dict,
    ) -> str:
        """Return a fully-assembled system prompt.

        Args:
            template: Key from :attr:`_TEMPLATES`.
            variables: Additional format variables (e.g. question).
            sources:   Named document lists, e.g. ``{"retrieved_docs": [...]}``.
        """
        doc_blocks = []
        for docs in sources.values():
            for doc in docs:
                src = doc.get("metadata", {}).get("source", "unknown")
                text = doc.get("text", "")
                doc_blocks.append(f"[Source: {src}]\n{text}")

        rendered_docs = "\n\n".join(doc_blocks) if doc_blocks else "(no documents retrieved)"
        tmpl = self._TEMPLATES[template]
        return tmpl.format(documents=rendered_docs, **variables)


# ---------------------------------------------------------------------------
# MemoryManager — YOUR logic, no framework dependency
# ---------------------------------------------------------------------------


class MemoryManager:
    """Maintain a sliding window of conversation messages.

    Keeps the last *max_turns* user+assistant pairs so the agent remembers
    recent context without blowing the token budget.
    """

    def __init__(self, max_turns: int = 5) -> None:
        self._history: list[dict] = []
        self._max_turns = max_turns

    def get_messages(self, system_prompt: str, user_message: str) -> list[dict]:
        """Return the full message list for the next LLM call."""
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history[-(self._max_turns * 2):])
        messages.append({"role": "user", "content": user_message})
        return messages

    def record(self, user_message: str, assistant_reply: str) -> None:
        """Append a turn to the conversation history."""
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": assistant_reply})


# ---------------------------------------------------------------------------
# TokenTracker — YOUR logic, no framework dependency
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:
    """Accumulate token usage across all LLM calls in a session."""

    def __init__(self) -> None:
        self._total = TokenUsage()
        self._calls: int = 0

    def record(self, usage) -> None:
        """Accept an OpenAI usage object or a :class:`TokenUsage` instance."""
        if hasattr(usage, "prompt_tokens"):
            self._total.prompt_tokens += usage.prompt_tokens or 0
            self._total.completion_tokens += usage.completion_tokens or 0
        self._calls += 1

    @property
    def totals(self) -> TokenUsage:
        return self._total

    @property
    def call_count(self) -> int:
        return self._calls


# ===========================================================================
# HybridRAGAgent — the main class
# ===========================================================================


class HybridRAGAgent:
    """RAG agent that uses LangChain for commodity ops, custom code for logic.

    Args:
        docs_directory: Path to a directory of ``.md`` / ``.txt`` files.
        llm_client:     An :class:`openai.OpenAI` client (injected, not created
                        inside the agent — testable and swappable).
        use_langchain:  If True (default) and LangChain is installed, use
                        LangChain for loading and embedding.  Set to False to
                        force the pure-custom path.
    """

    def __init__(
        self,
        docs_directory: str,
        llm_client: OpenAI,
        use_langchain: bool = True,
    ) -> None:
        self._docs_dir = Path(docs_directory)
        self._llm = llm_client
        self._use_langchain = use_langchain and LANGCHAIN_AVAILABLE

        # Framework: embeddings (only when using LangChain path)
        self._lc_embeddings = OpenAIEmbeddings(model=_EMBED_MODEL) if self._use_langchain else None

        # Your code: the important parts
        self._context_assembler = ContextAssembler()
        self._memory_manager = MemoryManager(max_turns=5)
        self._token_tracker = TokenTracker()

        # Populated by ingest()
        self._vector_store: Optional[VectorStoreProtocol] = None
        self._ingested_count: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self) -> int:
        """Load, embed, and store all documents in *docs_directory*.

        Uses LangChain's DirectoryLoader when available; falls back to a
        simple recursive glob otherwise.

        Returns:
            Number of documents ingested.
        """
        if self._use_langchain:
            return self._ingest_with_langchain()
        return self._ingest_custom()

    def _ingest_with_langchain(self) -> int:
        """Framework does the heavy lifting."""
        loader = DirectoryLoader(
            str(self._docs_dir),
            glob="**/*.{md,txt}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        lc_docs = loader.load()
        if not lc_docs:
            lc_docs = self._load_inline_docs()
        self._vector_store = LangChainVectorStore.from_documents(
            lc_docs, self._lc_embeddings
        )
        self._ingested_count = len(lc_docs)
        return self._ingested_count

    def _ingest_custom(self) -> int:
        """Pure-custom ingestion path (no LangChain)."""
        docs: list[dict] = []
        for path in self._docs_dir.rglob("*"):
            if path.suffix in {".md", ".txt"} and path.is_file():
                text = path.read_text(encoding="utf-8")
                docs.append({"text": text, "metadata": {"source": path.name}})

        if not docs:
            docs = [
                {"text": d["text"], "metadata": {"source": d["source"]}}
                for d in _INLINE_DOCS
            ]

        store = SimpleVectorStore(self._llm)
        store.add_documents(docs)
        self._vector_store = store
        self._ingested_count = len(docs)
        return self._ingested_count

    def _load_inline_docs(self) -> list["Document"]:
        """Return inline demo documents when the directory has no .md files."""
        return [
            Document(
                page_content=d["text"],
                metadata={"source": d["source"]},
            )
            for d in _INLINE_DOCS
        ]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> HybridResult:
        """Run the full RAG pipeline for *question*.

        Your code controls every step. The framework is just a tool.

        Args:
            question: The user's natural-language question.

        Returns:
            A :class:`HybridResult` with the answer and metadata.
        """
        if self._vector_store is None:
            raise RuntimeError("Call ingest() before query().")

        # 1. Retrieve (framework or custom does this well)
        t_ret = time.perf_counter()
        retrieved = self._vector_store.similarity_search(question, k=5)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # 2. Your context assembly logic (you control quality)
        context = self._context_assembler.assemble(
            template="rag_query",
            variables={"question": question},
            sources={"retrieved_docs": retrieved},
        )

        # 3. Your memory management (you control budget)
        messages = self._memory_manager.get_messages(
            system_prompt=context,
            user_message=question,
        )

        # 4. Your LLM call (through your abstraction — you control provider)
        t_gen = time.perf_counter()
        resp = self._llm.chat.completions.create(model=_LLM_MODEL, messages=messages)
        generation_ms = (time.perf_counter() - t_gen) * 1000

        answer = resp.choices[0].message.content or ""

        # 5. Your token tracking (you control cost visibility)
        self._token_tracker.record(resp.usage)

        # Record this turn in memory
        self._memory_manager.record(question, answer)

        return HybridResult(
            answer=answer,
            sources=list({r.get("metadata", {}).get("source", "") for r in retrieved}),
            tokens=resp.usage.total_tokens if resp.usage else 0,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    # ------------------------------------------------------------------
    # Vector store swap (the key demonstration)
    # ------------------------------------------------------------------

    def swap_vector_store(self, new_store: VectorStoreProtocol) -> None:
        """Replace the vector store backend with zero logic changes.

        The agent loop, context assembly, memory management, and token
        tracking are all completely unaffected by this swap.

        Args:
            new_store: Any object satisfying :class:`VectorStoreProtocol`.
        """
        self._vector_store = new_store

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def token_summary(self) -> TokenUsage:
        """Accumulated token usage across all queries in this session."""
        return self._token_tracker.totals

    @property
    def ingested_count(self) -> int:
        return self._ingested_count

    @property
    def vector_backend(self) -> str:
        if self._vector_store is None:
            return "none"
        return getattr(self._vector_store, "backend_name", type(self._vector_store).__name__)


# ---------------------------------------------------------------------------
# Inline demo documents (used when the docs directory is empty/missing)
# ---------------------------------------------------------------------------

_INLINE_DOCS = [
    {
        "source": "rag_phases.md",
        "text": (
            "RAG has four phases: Ingest (load, chunk, embed, store), "
            "Retrieve (embed query, search), Augment (build prompt), "
            "Generate (call LLM with augmented prompt)."
        ),
    },
    {
        "source": "vector_databases.md",
        "text": (
            "Vector databases store embeddings and support approximate "
            "nearest-neighbour search.  Popular choices: Qdrant, Pinecone, "
            "Chroma, FAISS, Weaviate."
        ),
    },
    {
        "source": "agent_loop.md",
        "text": (
            "An agent loop: perceive input, think (call LLM), act (execute tools), "
            "observe results.  Repeat until a final answer is produced."
        ),
    },
]


# ---------------------------------------------------------------------------
# Line-count reporter
# ---------------------------------------------------------------------------


def _count_custom_vs_framework_lines() -> tuple[int, int]:
    """Return (custom_lines, framework_lines) by inspecting this module."""
    import inspect, sys

    source = Path(__file__).read_text(encoding="utf-8")
    lines = source.splitlines()

    framework_markers = {
        "LangChainVectorStore",
        "DirectoryLoader",
        "OpenAIEmbeddings",
        "from langchain",
        "import langchain",
        "LCVectorStore",
    }
    custom_lines = framework_lines = 0
    in_framework_class = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        is_fw = any(m in stripped for m in framework_markers)
        if is_fw:
            framework_lines += 1
        else:
            custom_lines += 1

    return custom_lines, framework_lines


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo(docs_dir: str = ".") -> None:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("\n" + "=" * 65)
    print("  HYBRID RAG AGENT DEMO")
    print("=" * 65)

    # ---- Phase 1: Ingest with LangChain --------------------------------
    print(f"\n[1] Ingesting documents from '{docs_dir}' using LangChain ({_VECTOR_BACKEND})...")
    agent = HybridRAGAgent(docs_dir, client, use_langchain=True)
    n = agent.ingest()
    print(f"    Ingested {n} documents. Vector backend: {agent.vector_backend}")

    # ---- Phase 2: Query ------------------------------------------------
    question = "What are the four phases of a RAG pipeline?"
    print(f"\n[2] Query: '{question}'")
    result = agent.query(question)
    print(f"    Answer: {textwrap.fill(result.answer, 60)}")
    print(f"    Sources: {result.sources}")
    print(f"    Tokens : {result.tokens}  |  Retrieval: {result.retrieval_ms:.0f}ms  |  Generation: {result.generation_ms:.0f}ms")

    # ---- Phase 3: Swap vector store ------------------------------------
    print("\n[3] Swapping vector store: LangChain FAISS → SimpleVectorStore ...")
    custom_store = SimpleVectorStore(client)
    custom_store.add_documents([
        {"text": d["text"], "metadata": {"source": d["source"]}} for d in _INLINE_DOCS
    ])
    agent.swap_vector_store(custom_store)
    print(f"    Vector backend is now: {agent.vector_backend}")

    print(f"\n[4] Same query after swap: '{question}'")
    result2 = agent.query(question)
    print(f"    Answer: {textwrap.fill(result2.answer, 60)}")
    print("    Agent logic unchanged — only storage backend changed.")

    # ---- Phase 4: Line count report ------------------------------------
    custom, fw = _count_custom_vs_framework_lines()
    print("\n[5] Code composition:")
    print(f"    Custom code lines : {custom}")
    print(f"    Framework lines   : {fw}")
    print(f"    Framework ratio   : {fw / (custom + fw) * 100:.0f}%  (only commodity ops)")

    # ---- Phase 5: Breaking-change resilience ---------------------------
    print("\n[6] What breaks if LangChain has a breaking change?")
    print("    • LangChainVectorStore.from_documents() — isolated in one class")
    print("    • DirectoryLoader usage — isolated in _ingest_with_langchain()")
    print("    • OpenAIEmbeddings — isolated in __init__()")
    print("    • Agent loop, context assembly, memory → UNAFFECTED")
    print("    Fix scope: update 3 isolated methods, not the whole agent.")

    print(f"\nTotal tokens this session: {agent.token_summary.total}")
    print("=" * 65)


if __name__ == "__main__":
    import sys

    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    _demo(docs_dir)
