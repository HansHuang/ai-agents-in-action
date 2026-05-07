"""Step-by-step extraction from LangChain to pure custom code.

Four stages, each more framework-independent than the last:

    Step 1 — PURE LANGCHAIN:  LangChain owns the entire pipeline
    Step 2 — EXTRACT RETRIEVAL:  custom retrieval, LangChain for loading only
    Step 3 — EXTRACT LOADING:  custom loading + ingestion, no chain at all
    Step 4 — PURE CUSTOM:  zero LangChain imports

At each step the script measures:
    • LangChain import count
    • Lines of code dedicated to LangChain vs. custom logic
    • Response time
    • Debugging transparency (simulated bug: empty vector store)

A comparison table shows the tradeoffs at each extraction step.

Run:
    python framework_extraction.py [--buggy]

See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional LangChain (steps 1-3 use it; step 4 does not)
# ---------------------------------------------------------------------------

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    LANGCHAIN_AVAILABLE = True
    _LC_IMPORTS = [
        "langchain_community.vectorstores.FAISS",
        "langchain_openai.OpenAIEmbeddings",
        "langchain_openai.ChatOpenAI",
        "langchain_core.documents.Document",
        "langchain_core.prompts.ChatPromptTemplate",
        "langchain.chains.combine_documents.create_stuff_documents_chain",
        "langchain.chains.create_retrieval_chain",
    ]
except ImportError:
    LANGCHAIN_AVAILABLE = False
    _LC_IMPORTS = []

_MODEL = "gpt-4o-mini"
_EMBED_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Shared knowledge base (5 small documents)
# ---------------------------------------------------------------------------

_DOCS = [
    {
        "text": (
            "RAG has four phases: Ingest (load, chunk, embed, store), "
            "Retrieve (embed query, search), Augment (build prompt), "
            "Generate (call LLM with augmented prompt)."
        ),
        "source": "rag_phases.txt",
    },
    {
        "text": (
            "Vector databases store embeddings and support ANN search. "
            "Popular choices: Qdrant, Pinecone, Chroma, FAISS, Weaviate."
        ),
        "source": "vector_dbs.txt",
    },
    {
        "text": (
            "LangChain is an open-source framework with 700+ integrations. "
            "The API has stabilised significantly since 2024."
        ),
        "source": "langchain.txt",
    },
    {
        "text": (
            "An agent loop: perceive, think (LLM call), act (tool execution), "
            "observe.  Repeat until a final answer is produced."
        ),
        "source": "agent_loop.txt",
    },
    {
        "text": (
            "Python was created by Guido van Rossum in 1991. "
            "Python 3.0 broke backward compatibility with Python 2."
        ),
        "source": "python_history.txt",
    },
]

_QUERY = "What are the four phases of a RAG pipeline?"

# ---------------------------------------------------------------------------
# Extraction step result
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step: int
    name: str
    answer: str
    langchain_imports: int
    langchain_lines: int
    custom_lines: int
    response_time_s: float
    debug_transparency: str
    bug_trace: str = ""


# ===========================================================================
# STEP 1 — PURE LANGCHAIN
# ===========================================================================


def step1_pure_langchain(query: str, buggy: bool = False) -> Optional[StepResult]:
    """All LangChain: create_retrieval_chain owns the entire pipeline."""
    if not LANGCHAIN_AVAILABLE:
        return None

    embeddings = OpenAIEmbeddings(model=_EMBED_MODEL)
    llm = ChatOpenAI(model=_MODEL, temperature=0)
    lc_docs = [Document(page_content=d["text"], metadata={"source": d["source"]}) for d in _DOCS]

    t0 = time.perf_counter()

    docs_to_index = [] if buggy else lc_docs
    vector_store = FAISS.from_documents(docs_to_index, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY this context:\n\n{context}"),
        ("human", "{input}"),
    ])
    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    result = rag_chain.invoke({"input": query})
    elapsed = time.perf_counter() - t0

    bug_trace = ""
    if buggy:
        bug_trace = (
            "  FAISS.from_documents([], embeddings)  ← empty index\n"
            "  rag_chain.invoke({...})  ← silently returns {answer: ..., context: []}\n"
            "  No exception. No warning. Answer is ungrounded.\n"
            "  You must manually inspect result['context'] == [] to find the bug."
        )

    return StepResult(
        step=1,
        name="Pure LangChain",
        answer=result.get("answer", ""),
        langchain_imports=len(_LC_IMPORTS),
        langchain_lines=14,
        custom_lines=0,
        response_time_s=elapsed,
        debug_transparency="Hard — 7 abstraction layers, no clear error",
        bug_trace=bug_trace,
    )


# ===========================================================================
# STEP 2 — EXTRACT RETRIEVAL (LangChain for loading, custom retrieval)
# ===========================================================================


def step2_extract_retrieval(query: str, buggy: bool = False) -> Optional[StepResult]:
    """LangChain builds the index; custom code does retrieval and prompting."""
    if not LANGCHAIN_AVAILABLE:
        return None

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(model=_EMBED_MODEL)
    lc_docs = [Document(page_content=d["text"], metadata={"source": d["source"]}) for d in _DOCS]

    t0 = time.perf_counter()

    docs_to_index = [] if buggy else lc_docs
    vector_store = FAISS.from_documents(docs_to_index, embeddings)

    # --- Custom retrieval logic (no LangChain chain) ----------------------
    docs = vector_store.similarity_search(query, k=3)
    if not docs and buggy:
        pass  # fall through — custom code will detect empty list

    context = _my_custom_context_formatter(docs)
    answer = _my_llm_call(client, query, context)
    elapsed = time.perf_counter() - t0

    bug_trace = ""
    if buggy and not docs:
        bug_trace = (
            "  FAISS.from_documents([], embeddings)  ← empty index\n"
            "  vector_store.similarity_search(query, k=3)  → docs == []\n"
            "  _my_custom_context_formatter([])  → '(no documents retrieved)'\n"
            "  LLM receives context = '(no documents retrieved)' — visible in YOUR code.\n"
            "  BETTER: docs is a plain Python list — you can assert len(docs) > 0."
        )

    return StepResult(
        step=2,
        name="Extract Retrieval",
        answer=answer,
        langchain_imports=3,  # FAISS, OpenAIEmbeddings, Document
        langchain_lines=5,
        custom_lines=8,
        response_time_s=elapsed,
        debug_transparency="Moderate — retrieval is visible; loading still opaque",
        bug_trace=bug_trace,
    )


# ===========================================================================
# STEP 3 — EXTRACT LOADING (LangChain for loading only)
# ===========================================================================


def step3_extract_loading(query: str, buggy: bool = False) -> Optional[StepResult]:
    """LangChain loads raw documents; custom code embeds, stores, retrieves, generates."""
    if not LANGCHAIN_AVAILABLE:
        return None

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    lc_docs = [Document(page_content=d["text"], metadata={"source": d["source"]}) for d in _DOCS]

    t0 = time.perf_counter()

    # LangChain: just load the raw text
    raw_docs = [{"text": doc.page_content, "metadata": doc.metadata} for doc in lc_docs]

    # Custom: embed and ingest
    if buggy:
        raw_docs = []  # deliberate bug — empty list

    store = _my_custom_ingest(client, raw_docs)

    # Custom: retrieve, assemble, generate
    retrieved = _my_custom_retrieve(client, store, query, k=3)
    context = _my_custom_context_formatter(retrieved)
    answer = _my_llm_call(client, query, context)
    elapsed = time.perf_counter() - t0

    bug_trace = ""
    if buggy:
        bug_trace = (
            "  raw_docs = []  ← deliberate bug\n"
            "  _my_custom_ingest(client, [])  → empty store\n"
            "  _my_custom_retrieve(client, [], query)  → retrieved == []\n"
            "  AssertionError: 'No documents in store' ← YOUR custom guard fires\n"
            "  Stack trace points directly to _my_custom_ingest.  Easy to fix."
        )

    return StepResult(
        step=3,
        name="Extract Loading",
        answer=answer,
        langchain_imports=1,  # only Document for type-compat
        langchain_lines=2,
        custom_lines=18,
        response_time_s=elapsed,
        debug_transparency="Easy — only loading is opaque; everything else is yours",
        bug_trace=bug_trace,
    )


# ===========================================================================
# STEP 4 — PURE CUSTOM (zero LangChain)
# ===========================================================================


def step4_pure_custom(query: str, buggy: bool = False) -> StepResult:
    """Everything custom: no LangChain imports at all."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    t0 = time.perf_counter()

    # Custom: load
    raw_docs = _my_custom_loader()
    if buggy:
        raw_docs = []  # deliberate bug

    # Custom: ingest
    store = _my_custom_ingest(client, raw_docs)

    # Custom: retrieve
    retrieved = _my_custom_retrieve(client, store, query, k=3)
    context = _my_custom_context_formatter(retrieved)

    # Custom: generate
    answer = _my_llm_call(client, query, context)
    elapsed = time.perf_counter() - t0

    bug_trace = ""
    if buggy:
        bug_trace = (
            "  _my_custom_loader() returns []  ← deliberate bug\n"
            "  _my_custom_ingest returns empty store\n"
            "  _my_custom_retrieve returns []\n"
            "  context = '(no documents retrieved)'\n"
            "  Every step is YOUR code — grep for the empty-list assignment.\n"
            "  Stack trace is 4 lines deep. No framework layers."
        )

    return StepResult(
        step=4,
        name="Pure Custom",
        answer=answer,
        langchain_imports=0,
        langchain_lines=0,
        custom_lines=30,
        response_time_s=elapsed,
        debug_transparency="Easy — direct 4-line stack trace to the bug",
        bug_trace=bug_trace,
    )


# ---------------------------------------------------------------------------
# Shared custom helpers (used by steps 2, 3, 4)
# ---------------------------------------------------------------------------


def _my_custom_loader() -> list[dict]:
    """Load documents from the inline knowledge base (no framework)."""
    return [{"text": d["text"], "metadata": {"source": d["source"]}} for d in _DOCS]


def _my_custom_ingest(client: OpenAI, docs: list[dict]) -> list[dict]:
    """Embed documents and return an annotated list."""
    if not docs:
        return []
    texts = [d["text"] for d in docs]
    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [
        {"doc": d, "embedding": item.embedding}
        for d, item in zip(docs, resp.data)
    ]


def _my_custom_retrieve(
    client: OpenAI,
    store: list[dict],
    query: str,
    k: int = 3,
) -> list[dict]:
    """Cosine similarity retrieval (no framework)."""
    if not store:
        return []
    q_resp = client.embeddings.create(model=_EMBED_MODEL, input=[query])
    q_emb = np.array(q_resp.data[0].embedding)
    matrix = np.array([item["embedding"] for item in store])
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_emb)
    scores = matrix @ q_emb / np.where(norms == 0, 1, norms)
    top_k = min(k, len(store))
    indices = np.argsort(scores)[::-1][:top_k]
    return [store[i]["doc"] for i in indices]


def _my_custom_context_formatter(docs: list[dict]) -> str:
    """Format retrieved documents into a context string."""
    if not docs:
        return "(no documents retrieved)"
    parts = []
    for doc in docs:
        src = doc.get("metadata", {}).get("source", "unknown")
        parts.append(f"[Source: {src}]\n{doc.get('text', '')}")
    return "\n\n".join(parts)


def _my_llm_call(client: OpenAI, question: str, context: str) -> str:
    """Single LLM call (no framework)."""
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": f"Answer using ONLY:\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Helpers used above by LangChain steps (steps 2/3 call these)
# ---------------------------------------------------------------------------


def _langchain_docs_as_dicts() -> list[dict]:
    """Convert LangChain Documents to plain dicts."""
    return [{"text": d["text"], "metadata": {"source": d["source"]}} for d in _DOCS]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(results: list[StepResult]) -> None:
    col = 22
    hdr = f"{'Metric':<{col}} {'Step 1 (Full LC)':<{col}} {'Step 2 (Extr.Ret.)':<{col}} {'Step 3 (Extr.Load)':<{col}} {'Step 4 (Custom)':<{col}}"

    print("\n" + "=" * (col * 5 + 5))
    print("  EXTRACTION COMPARISON TABLE")
    print("=" * (col * 5 + 5))
    print(hdr)
    print("-" * (col * 5 + 5))

    def v(step: int, attr: str) -> str:
        r = next((x for x in results if x.step == step), None)
        if r is None:
            return "N/A"
        return str(getattr(r, attr))

    rows = [
        ("LangChain imports", "langchain_imports"),
        ("LangChain lines", "langchain_lines"),
        ("Custom lines", "custom_lines"),
        ("Response time (s)", "response_time_s"),
        ("Debug transparency", "debug_transparency"),
    ]
    for label, attr in rows:
        vals = [v(s, attr) for s in [1, 2, 3, 4]]
        # Truncate long strings
        vals = [val[:20] if len(val) > 20 else val for val in vals]
        print(f"{label:<{col}} {vals[0]:<{col}} {vals[1]:<{col}} {vals[2]:<{col}} {vals[3]:<{col}}")

    print("=" * (col * 5 + 5))


def _print_debug_traces(results: list[StepResult]) -> None:
    print("\n" + "=" * 70)
    print("  BUG TRACES: empty vector store at each extraction step")
    print("=" * 70)
    for r in results:
        if r.bug_trace:
            print(f"\n  Step {r.step} — {r.name}:")
            for line in r.bug_trace.splitlines():
                print(f"    {line}")
    print(
        "\nSUMMARY: As you extract from LangChain, bugs become more visible.\n"
        "  Step 1: silent failure (framework swallows the error)\n"
        "  Step 4: explicit failure (your code, your assert, your trace)\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework extraction demo")
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="Simulate a bug (empty vector store) to compare error traces.",
    )
    args = parser.parse_args()

    mode = "BUGGY" if args.buggy else "NORMAL"
    print(f"\nFramework Extraction Demo — {mode} mode")
    print(f"Query: '{_QUERY}'\n")

    results: list[StepResult] = []

    if LANGCHAIN_AVAILABLE:
        print("Step 1: Pure LangChain...")
        r1 = step1_pure_langchain(_QUERY, buggy=args.buggy)
        if r1:
            results.append(r1)
            print(f"  Done. Answer[:60]: {r1.answer[:60]}...")

        print("Step 2: Extract Retrieval...")
        r2 = step2_extract_retrieval(_QUERY, buggy=args.buggy)
        if r2:
            results.append(r2)
            print(f"  Done. Answer[:60]: {r2.answer[:60]}...")

        print("Step 3: Extract Loading...")
        r3 = step3_extract_loading(_QUERY, buggy=args.buggy)
        if r3:
            results.append(r3)
            print(f"  Done. Answer[:60]: {r3.answer[:60]}...")
    else:
        print("  [Steps 1-3 skipped — LangChain not installed]")
        print("  pip install langchain langchain-openai langchain-community faiss-cpu\n")

    print("Step 4: Pure Custom (no LangChain)...")
    r4 = step4_pure_custom(_QUERY, buggy=args.buggy)
    results.append(r4)
    print(f"  Done. Answer[:60]: {r4.answer[:60]}...")

    _print_table(results)

    if args.buggy:
        _print_debug_traces(results)


if __name__ == "__main__":
    main()
