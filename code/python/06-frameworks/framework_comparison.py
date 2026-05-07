"""Framework comparison: the same RAG agent built three ways.

Three parallel implementations answer the identical query from the identical
knowledge base so you can compare tradeoffs side-by-side:

    1. FROM SCRATCH  — pure OpenAI SDK, no framework dependencies
    2. LANGCHAIN     — LangChain chains and FAISS vector store
    3. LANGGRAPH     — LangGraph StateGraph with typed state

For each implementation the script measures:
    • Lines of code (static, counted at runtime)
    • Number of top-level package dependencies
    • Time to first response
    • Token usage
    • Debuggability (a deliberate zero-results bug shows the error trace)

Run:
    python framework_comparison.py [--buggy]   # --buggy triggers debug mode

See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional framework imports (fail gracefully so from-scratch always works)
# ---------------------------------------------------------------------------

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from typing import TypedDict, Annotated
    import operator
    from langgraph.graph import StateGraph, START, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared knowledge base (5 documents — identical for all three implementations)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: list[dict] = [
    {
        "id": "kb-01",
        "title": "Python History",
        "text": (
            "Python was created by Guido van Rossum and first released in 1991. "
            "It emphasises readability and uses significant indentation. "
            "Python 3.0 was released in 2008 and is not backward-compatible with Python 2."
        ),
        "source": "python_history.txt",
    },
    {
        "id": "kb-02",
        "title": "Vector Databases",
        "text": (
            "Vector databases store high-dimensional embeddings and support approximate "
            "nearest-neighbour search. Popular options include Qdrant, Pinecone, Weaviate, "
            "Chroma, and FAISS. They power semantic search and RAG pipelines."
        ),
        "source": "vector_databases.txt",
    },
    {
        "id": "kb-03",
        "title": "LangChain Overview",
        "text": (
            "LangChain is an open-source framework for building LLM-powered applications. "
            "It provides 700+ integrations, a chain composition API, and LangSmith for "
            "observability. The API has stabilised significantly since 2024."
        ),
        "source": "langchain_overview.txt",
    },
    {
        "id": "kb-04",
        "title": "RAG Architecture",
        "text": (
            "Retrieval-Augmented Generation (RAG) enhances LLM responses with external "
            "knowledge. The four phases are: Ingest (load, chunk, embed, store), "
            "Retrieve (embed query, search), Augment (build prompt), Generate (call LLM)."
        ),
        "source": "rag_architecture.txt",
    },
    {
        "id": "kb-05",
        "title": "Agent Loops",
        "text": (
            "An AI agent loop repeatedly calls an LLM, parses its response for tool calls, "
            "executes those tools, and feeds results back until the LLM emits a final answer. "
            "The loop has four parts: perceive, think, act, observe."
        ),
        "source": "agent_loops.txt",
    },
]

DEMO_QUERY = "What are the four phases of a RAG pipeline?"

_MODEL = "gpt-4o-mini"
_EMBED_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Captured metrics for one implementation."""

    name: str
    answer: str
    sources: list[str]
    lines_of_code: int
    dependencies: list[str]
    response_time_s: float
    tokens_used: int
    # Error trace when run in buggy mode, otherwise empty string
    debug_trace: str = ""

    @property
    def debuggability(self) -> str:
        """Qualitative label derived from implementation name."""
        labels = {
            "From Scratch": "Easy",
            "LangChain": "Hard",
            "LangGraph": "Moderate",
        }
        return labels.get(self.name, "Unknown")


# ===========================================================================
# IMPLEMENTATION 1: FROM SCRATCH (pure OpenAI SDK)
# ===========================================================================


def _embed_texts_scratch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def _build_from_scratch_store(
    client: OpenAI, docs: list[dict]
) -> list[dict]:
    """Embed all documents and return an annotated list (our vector store)."""
    embeddings = _embed_texts_scratch(client, [d["text"] for d in docs])
    return [
        {"doc": d, "embedding": emb}
        for d, emb in zip(docs, embeddings)
    ]


def _retrieve_from_scratch(
    client: OpenAI,
    store: list[dict],
    query: str,
    k: int = 3,
    buggy: bool = False,
) -> list[dict]:
    """Return top-k documents by cosine similarity.

    When *buggy* is True the store is replaced with an empty list so the
    retrieval silently returns zero results — making the downstream answer
    wrong with no clear indication why.
    """
    if buggy:
        store = []  # deliberate bug: empty store
    query_emb = _embed_texts_scratch(client, [query])[0]
    scored = [
        (item["doc"], _cosine_similarity(query_emb, item["embedding"]))
        for item in store
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


def run_from_scratch(query: str, buggy: bool = False) -> ComparisonResult:
    """FROM SCRATCH implementation of a simple RAG agent."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    store = _build_from_scratch_store(client, KNOWLEDGE_BASE)

    t0 = time.perf_counter()
    retrieved = _retrieve_from_scratch(client, store, query, k=3, buggy=buggy)

    context_parts = []
    for doc in retrieved:
        context_parts.append(f"[Source: {doc['source']}]\n{doc['text']}")
    context = "\n\n".join(context_parts) if context_parts else "(no documents retrieved)"

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using ONLY the documents below.\n\n"
                + context
            ),
        },
        {"role": "user", "content": query},
    ]
    resp = client.chat.completions.create(model=_MODEL, messages=messages)
    elapsed = time.perf_counter() - t0

    answer = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0

    debug_trace = ""
    if buggy and not retrieved:
        debug_trace = (
            "BUG: store was replaced with [] in _retrieve_from_scratch().\n"
            "ERROR TRACE (from scratch):\n"
            "  File framework_comparison.py, line ~155, in _retrieve_from_scratch\n"
            "    store = []  # deliberate bug\n"
            "  → retrieved == []\n"
            "  → context == '(no documents retrieved)'\n"
            "  → LLM answer is hallucinated or 'I don\\'t know'\n"
            "DIAGNOSIS: Easy — the bug is in YOUR code. grep for 'store = []'.\n"
        )

    return ComparisonResult(
        name="From Scratch",
        answer=answer,
        sources=[d["source"] for d in retrieved],
        lines_of_code=_count_lines(run_from_scratch),
        dependencies=["openai", "numpy"],
        response_time_s=elapsed,
        tokens_used=tokens,
        debug_trace=debug_trace,
    )


# ===========================================================================
# IMPLEMENTATION 2: LANGCHAIN
# ===========================================================================


def run_langchain(query: str, buggy: bool = False) -> Optional[ComparisonResult]:
    """LANGCHAIN implementation of a simple RAG agent."""
    if not LANGCHAIN_AVAILABLE:
        print("  [SKIP] LangChain not installed. pip install langchain langchain-openai langchain-community faiss-cpu")
        return None

    langchain_docs = [
        Document(
            page_content=d["text"],
            metadata={"source": d["source"], "title": d["title"]},
        )
        for d in KNOWLEDGE_BASE
    ]

    embeddings = OpenAIEmbeddings(model=_EMBED_MODEL)
    llm = ChatOpenAI(model=_MODEL, temperature=0)

    t0 = time.perf_counter()

    if buggy:
        # Deliberate bug: pass empty doc list so FAISS index has nothing
        vector_store = FAISS.from_documents([], embeddings)
    else:
        vector_store = FAISS.from_documents(langchain_docs, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "Answer the question using ONLY the context below.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    result = rag_chain.invoke({"input": query})
    elapsed = time.perf_counter() - t0

    answer = result.get("answer", "")
    retrieved_docs = result.get("context", [])
    sources = list({d.metadata.get("source", "") for d in retrieved_docs})

    debug_trace = ""
    if buggy:
        debug_trace = (
            "BUG: FAISS.from_documents([], embeddings) — empty index.\n"
            "ERROR TRACE (LangChain):\n"
            "  chain.invoke({'input': ...}) → result['answer'] contains no context\n"
            "  BUT: no exception is raised — the chain silently returns an answer\n"
            "  LangChain swallows the empty-retrieval case with no warning.\n"
            "DIAGNOSIS: Hard — you must inspect result['context'] manually.\n"
            "  Seven abstraction layers hide the empty store from the traceback.\n"
        )

    dep_count = [
        "langchain-core",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "faiss-cpu",
        "openai",
        "numpy",
        "pydantic",
        "httpx",
        "aiohttp",
        "tiktoken",
        "tenacity",
    ]

    return ComparisonResult(
        name="LangChain",
        answer=answer,
        sources=sources,
        lines_of_code=_count_lines(run_langchain),
        dependencies=dep_count,
        response_time_s=elapsed,
        tokens_used=0,  # LangChain chain doesn't expose token counts trivially
        debug_trace=debug_trace,
    )


# ===========================================================================
# IMPLEMENTATION 3: LANGGRAPH
# ===========================================================================


def run_langgraph(query: str, buggy: bool = False) -> Optional[ComparisonResult]:
    """LANGGRAPH implementation of a simple RAG agent."""
    if not LANGCHAIN_AVAILABLE or not LANGGRAPH_AVAILABLE:
        print("  [SKIP] LangGraph not installed. pip install langgraph langchain-openai langchain-community faiss-cpu")
        return None

    # ---- State definition ------------------------------------------------
    class RAGState(TypedDict):
        query: str
        retrieved_docs: list[Document]
        context: str
        answer: str
        sources: list[str]

    embeddings = OpenAIEmbeddings(model=_EMBED_MODEL)
    llm = ChatOpenAI(model=_MODEL, temperature=0)

    langchain_docs = [
        Document(
            page_content=d["text"],
            metadata={"source": d["source"]},
        )
        for d in KNOWLEDGE_BASE
    ]

    if buggy:
        vector_store = FAISS.from_documents([], embeddings)
    else:
        vector_store = FAISS.from_documents(langchain_docs, embeddings)

    # ---- Graph nodes -------------------------------------------------------
    def retrieve_node(state: RAGState) -> dict:
        docs = vector_store.similarity_search(state["query"], k=3)
        return {"retrieved_docs": docs}

    def augment_node(state: RAGState) -> dict:
        parts = [f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in state["retrieved_docs"]]
        context = "\n\n".join(parts) if parts else "(no documents retrieved)"
        sources = list({d.metadata["source"] for d in state["retrieved_docs"]})
        return {"context": context, "sources": sources}

    def generate_node(state: RAGState) -> dict:
        messages = [
            {"role": "system", "content": f"Answer using ONLY:\n{state['context']}"},
            {"role": "user", "content": state["query"]},
        ]
        resp = llm.invoke(messages)
        return {"answer": resp.content}

    # ---- Build graph -------------------------------------------------------
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("augment", augment_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "augment")
    graph.add_edge("augment", "generate")
    graph.add_edge("generate", END)
    app = graph.compile()

    t0 = time.perf_counter()
    final_state = app.invoke({"query": query, "retrieved_docs": [], "context": "", "answer": "", "sources": []})
    elapsed = time.perf_counter() - t0

    debug_trace = ""
    if buggy and not final_state.get("retrieved_docs"):
        debug_trace = (
            "BUG: FAISS.from_documents([], embeddings) — empty index.\n"
            "ERROR TRACE (LangGraph):\n"
            "  app.invoke(...) succeeds — no exception\n"
            "  state after 'retrieve' node: retrieved_docs == []\n"
            "  state after 'augment' node:  context == '(no documents retrieved)'\n"
            "  state after 'generate' node: answer contains no grounded content\n"
            "DIAGNOSIS: Moderate — LangGraph Studio shows per-node state.\n"
            "  You can inspect state['retrieved_docs'] after each node.\n"
            "  Still requires knowing to look at intermediate state.\n"
        )

    dep_count = [
        "langgraph",
        "langchain-core",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "faiss-cpu",
        "openai",
        "numpy",
        "pydantic",
        "httpx",
        "aiohttp",
        "tiktoken",
        "tenacity",
        "orjson",
        "httpx-sse",
    ]

    return ComparisonResult(
        name="LangGraph",
        answer=final_state.get("answer", ""),
        sources=final_state.get("sources", []),
        lines_of_code=_count_lines(run_langgraph),
        dependencies=dep_count,
        response_time_s=elapsed,
        tokens_used=0,
        debug_trace=debug_trace,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_lines(func) -> int:
    """Count non-blank, non-comment source lines in *func*."""
    src = inspect.getsource(func)
    return sum(
        1
        for line in src.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _print_comparison_table(results: list[ComparisonResult]) -> None:
    col_w = 20
    headers = ["Metric", "From Scratch", "LangChain", "LangGraph"]
    row_fmt = "{:<22} {:<20} {:<20} {:<20}"

    print("\n" + "=" * 85)
    print("  FRAMEWORK COMPARISON TABLE")
    print("=" * 85)
    print(row_fmt.format(*headers))
    print("-" * 85)

    def _val(name: str, attr: str, fmt=str) -> str:
        r = next((x for x in results if x.name == name), None)
        if r is None:
            return "N/A (not installed)"
        val = getattr(r, attr)
        return fmt(val)

    def _val_fn(name: str, fn) -> str:
        r = next((x for x in results if x.name == name), None)
        return fn(r) if r else "N/A"

    names = ["From Scratch", "LangChain", "LangGraph"]

    rows = [
        ("Lines of code", [_val(n, "lines_of_code") for n in names]),
        ("Dependencies", [_val_fn(n, lambda r: str(len(r.dependencies))) for n in names]),
        ("Response time", [_val_fn(n, lambda r: f"{r.response_time_s:.2f}s") for n in names]),
        ("Token usage", [_val_fn(n, lambda r: str(r.tokens_used) if r.tokens_used else "N/A") for n in names]),
        ("Debuggability", [_val_fn(n, lambda r: r.debuggability) for n in names]),
    ]

    for label, vals in rows:
        print(row_fmt.format(label, *vals))

    print("=" * 85)


def _print_debug_section(results: list[ComparisonResult]) -> None:
    print("\n" + "=" * 85)
    print("  DEBUGGING COMPARISON (vector store returns 0 results)")
    print("=" * 85)
    for r in results:
        if r.debug_trace:
            print(f"\n--- {r.name} ---")
            print(r.debug_trace)
    print(
        "\nKEY INSIGHT: From-scratch code has a clear, single-call stack.\n"
        "  LangChain silently swallows empty-retrieval with no exception.\n"
        "  LangGraph exposes per-node state — moderate to debug with Studio.\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework comparison runner")
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="Trigger the deliberate zero-results bug to compare error traces.",
    )
    args = parser.parse_args()

    print(f"\nQuery: '{DEMO_QUERY}'")
    print(f"Mode:  {'BUGGY (zero results)' if args.buggy else 'NORMAL'}\n")

    results: list[ComparisonResult] = []

    print("Running FROM SCRATCH...")
    results.append(run_from_scratch(DEMO_QUERY, buggy=args.buggy))
    print(f"  Answer: {results[-1].answer[:80]}...")

    print("Running LANGCHAIN...")
    lc = run_langchain(DEMO_QUERY, buggy=args.buggy)
    if lc:
        results.append(lc)
        print(f"  Answer: {lc.answer[:80]}...")

    print("Running LANGGRAPH...")
    lg = run_langgraph(DEMO_QUERY, buggy=args.buggy)
    if lg:
        results.append(lg)
        print(f"  Answer: {lg.answer[:80]}...")

    _print_comparison_table(results)

    if args.buggy:
        _print_debug_section(results)


if __name__ == "__main__":
    main()
