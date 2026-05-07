# Context Compression and Filtering

## What You'll Learn
- Why more context is often worse: the dilution problem
- Relevance filtering: keeping only what the query actually needs
- Extractive compression: pulling key sentences, not summarizing everything
- LLM-based summarization vs. heuristic truncation: when to use each
- Reranking: using a better model to reorder retrieval results
- The compression budget: how much information survives each stage

## Prerequisites
- [Dynamic Prompt Assembly](02-dynamic-prompt-assembly.md) — assembling context from multiple sources
- [The Context Window as a Resource](01-the-context-window-as-a-resource.md) — the budget you're fitting into
- [RAG from Scratch](../03-memory-and-retrieval/03-rag-from-scratch.md) — RAG is the biggest context contributor

---

## The Paradox of More Context

Your RAG pipeline retrieves 20 chunks. Your agent has 5 tool outputs. Your conversation history spans 15 turns. You have 60,000 tokens of available context budget. So you stuff everything in.

**This makes answers worse.**

Research consistently shows:
- Adding more *relevant* context improves answers — up to a point
- Adding *irrelevant* context degrades answers, even if relevant context is also present
- The model's ability to distinguish signal from noise degrades as context grows
- Performance peaks at 5-10 highly relevant chunks, not 20 mixed-quality chunks

The solution isn't more context. It's **better context.**

---

## The Filtering Pipeline

Think of context assembly as a funnel:

```
Raw Sources (100,000 tokens)
    │
    ▼
┌─────────────────────────┐
│ 1. RELEVANCE FILTER     │  Remove clearly irrelevant content
│    (threshold: > 0.6)   │  "Is this even about the right topic?"
└───────────┬─────────────┘
            │  80,000 tokens
            ▼
┌─────────────────────────┐
│ 2. QUALITY FILTER       │  Remove low-quality content
│    (dedup, noise rem.)  │  "Is this information reliable and clear?"
└───────────┬─────────────┘
            │  50,000 tokens
            ▼
┌─────────────────────────┐
│ 3. RERANK               │  Reorder by a better relevance model
│    (cross-encoder)      │  "What's the TRUE order of relevance?"
└───────────┬─────────────┘
            │  50,000 tokens (reordered)
            ▼
┌─────────────────────────┐
│ 4. EXTRACT / COMPRESS   │  Keep key sentences, discard filler
│    (extractive summary) │  "What are the essential facts?"
└───────────┬─────────────┘
            │  15,000 tokens
            ▼
┌─────────────────────────┐
│ 5. BUDGET ENFORCEMENT   │  Fit within allocated zone
│    (truncate if needed) │  "Does this fit in the dynamic context zone?"
└───────────┬─────────────┘
            │  12,000 tokens (fits budget)
            ▼
      Final Context
```

Each stage removes noise and preserves signal. The goal isn't to minimize tokens — it's to maximize **information density.**

---

## Stage 1: Relevance Filtering

The first and highest-impact filter. Remove anything that isn't about the user's query.

### Threshold-Based Filtering

```python
def filter_by_relevance(documents: list[dict], 
                        threshold: float = 0.6) -> list[dict]:
    """
    Remove documents below a similarity threshold.
    
    A threshold of 0.6 means:
    - 0.8-1.0: Highly relevant (keep)
    - 0.6-0.8: Moderately relevant (keep, but flag)
    - 0.4-0.6: Weakly relevant (discard unless few results)
    - 0.0-0.4: Irrelevant (always discard)
    """
    filtered = [doc for doc in documents if doc["score"] >= threshold]
    
    # Safety: if filtering removes everything, keep top 3
    if not filtered and documents:
        logging.warning(f"Threshold {threshold} filtered all documents. "
                       f"Keeping top 3.")
        filtered = documents[:3]
    
    logging.info(f"Relevance filter: {len(documents)} → {len(filtered)} "
                f"(threshold: {threshold})")
    return filtered
```

### Adaptive Thresholding

A fixed threshold of 0.6 doesn't work for all queries. Some queries have many highly relevant documents. Others have none.

```python
def adaptive_threshold(documents: list[dict], 
                       min_results: int = 3,
                       max_results: int = 10) -> float:
    """
    Find the threshold that keeps between min_results and max_results.
    
    Strategy:
    1. Start at 0.8 (very strict)
    2. If fewer than min_results, lower threshold by 0.1
    3. Repeat until we have enough results
    4. If more than max_results, raise threshold
    """
    scores = sorted([doc["score"] for doc in documents], reverse=True)
    
    threshold = 0.8
    while threshold >= 0.3:
        count = sum(1 for s in scores if s >= threshold)
        if count >= min_results:
            return threshold
        threshold -= 0.1
    
    return 0.3  # Fallback: very lenient

def filter_adaptive(documents: list[dict],
                    min_results: int = 3,
                    max_results: int = 10) -> list[dict]:
    """Filter with an adaptive threshold."""
    threshold = adaptive_threshold(documents, min_results, max_results)
    filtered = [doc for doc in documents if doc["score"] >= threshold]
    
    # Cap at max_results
    return filtered[:max_results]
```

---

## Stage 2: Quality and Redundancy Filtering

After relevance, filter for quality. Remove noise, duplicates, and contradictory information.

### Deduplication

```python
def deduplicate(documents: list[dict],
               threshold: float = 0.9) -> list[dict]:
    """
    Remove near-duplicate documents using character 3-gram Jaccard similarity.
    Two chunks with Jaccard >= threshold → keep the first one encountered.

    Why Jaccard instead of embedding cosine?
    - No embedding call required — fast and offline-safe
    - Character n-grams catch copy-pasted text reliably
    - Threshold 0.9 works well for long chunks; use ~0.6 for short snippets
    """
    def ngrams(text: str, n: int = 3) -> set[str]:
        t = text.lower()
        return {t[i:i+n] for i in range(len(t) - n + 1)}

    def jaccard(a: str, b: str) -> float:
        sa, sb = ngrams(a), ngrams(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    kept: list[dict] = []
    for doc in documents:
        is_duplicate = any(
            jaccard(doc["text"], k["text"]) >= threshold
            for k in kept
        )
        if not is_duplicate:
            kept.append(doc)

    logging.info(f"Dedup: {len(documents)} → {len(kept)} (threshold: {threshold})")
    return kept
```

### Information Density Scoring

Not all chunks are equally informative. A chunk full of boilerplate is less valuable than a chunk packed with facts.

```python
def score_information_density(text: str) -> float:
    """
    Estimate how information-dense a chunk is.
    
    Heuristics:
    - High ratio of named entities to total words → more facts
    - High ratio of numbers/dates → specific data
    - Low ratio of stop words → less filler
    - Presence of bullet points or tables → structured information
    """
    words = text.split()
    if not words:
        return 0.0
    
    # Count informative elements
    named_entities = count_named_entities(text)
    numbers = sum(1 for w in words if w.replace(',', '.').replace('%', '').isdigit())
    stop_words = sum(1 for w in words if w.lower() in STOP_WORDS)
    
    # Score: higher = more information-dense
    entity_ratio = named_entities / len(words)
    number_ratio = numbers / len(words)
    filler_penalty = stop_words / len(words)
    
    return (entity_ratio * 0.4 + number_ratio * 0.3 + (1 - filler_penalty) * 0.3)

def filter_by_density(documents: list[dict], 
                      min_density: float = 0.1) -> list[dict]:
    """Remove chunks that are mostly filler."""
    scored = [(doc, score_information_density(doc["text"])) for doc in documents]
    
    filtered = []
    for doc, density in scored:
        if density >= min_density:
            doc["density_score"] = density
            filtered.append(doc)
        else:
            logging.debug(f"Low density ({density:.2f}): '{doc['text'][:80]}...'")
    
    logging.info(f"Density filter: {len(documents)} → {len(filtered)} "
                f"(min: {min_density})")
    return filtered
```

---

## Stage 3: Reranking

Embedding similarity (cosine score) is fast but imprecise. A cross-encoder reads the query and document together and produces a true relevance score.

### Cross-Encoder Reranking

```python
def rerank_with_cross_encoder(query: str, documents: list[dict], 
                              top_k: int = 10) -> list[dict]:
    """
    Rerank documents using a cross-encoder model.
    
    Embedding similarity: "Does this document live near the query in vector space?"
    Cross-encoder: "Reading this query and this document side by side, 
                    how relevant are they to each other?"
    
    Cross-encoders are 10-100x slower than embedding search but much more accurate.
    Strategy: retrieve 50 with embeddings, rerank top 50 with cross-encoder, keep top 10.
    """
    from sentence_transformers import CrossEncoder
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Create query-document pairs
    pairs = [(query, doc["text"]) for doc in documents]
    
    # Get true relevance scores
    scores = model.predict(pairs)
    
    # Attach scores and sort
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)
    
    documents.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    logging.info(f"Reranked {len(documents)} documents, keeping top {top_k}")
    return documents[:top_k]
```

### LLM-Based Reranking

When a cross-encoder isn't available, use a fast LLM:

```python
def rerank_with_llm(query: str, documents: list[dict], 
                    top_k: int = 5) -> list[dict]:
    """
    Use an LLM to rerank documents.
    More expensive than cross-encoder but can handle nuanced relevance.
    """
    # Format documents for the LLM
    doc_list = "\n\n".join([
        f"[{i}] {doc['text'][:200]}..."  # First 200 chars as preview
        for i, doc in enumerate(documents)
    ])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast, cheap model for reranking
        messages=[{
            "role": "user",
            "content": f"""Query: {query}

Documents:
{doc_list}

Rank these documents by relevance to the query. Return the indices of the 
top {top_k} most relevant documents as a JSON array: [3, 7, 1, ...]"""
        }],
        response_format={"type": "json_object"}
    )
    
    ranked_indices = json.loads(response.choices[0].message.content)["indices"]
    
    return [documents[i] for i in ranked_indices if i < len(documents)]
```

---

## Stage 4: Extractive Compression

After filtering and reranking, you have the right documents in the right order. Now compress them — keep the key sentences, discard the rest.

### Extractive Summarization

Unlike abstractive summarization (which rewrites), extractive summarization selects the most important sentences verbatim. This preserves factual accuracy.

```python
def extract_key_sentences(text: str, query: str, 
                         max_sentences: int = 5) -> str:
    """
    Extract the sentences most relevant to the query.
    Preserves original wording — no hallucination risk.
    """
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text
    
    # Score each sentence by relevance to query
    query_embedding = embedder.embed(query)
    sentence_embeddings = [embedder.embed(s) for s in sentences]
    
    scored = [
        (sentence, cosine_similarity(query_embedding, emb))
        for sentence, emb in zip(sentences, sentence_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top sentences, preserve original order
    top_sentences = {s for s, _ in scored[:max_sentences]}
    kept = [s for s in sentences if s in top_sentences]
    
    return " ".join(kept)

def compress_documents(documents: list[dict], query: str,
                       max_tokens_per_doc: int = 200) -> list[dict]:
    """
    Compress each document to its most query-relevant sentences.
    """
    compressed = []
    for doc in documents:
        full_text = doc["text"]
        
        if count_tokens(full_text) <= max_tokens_per_doc:
            compressed.append(doc)
            continue
        
        # Extract key sentences
        extracted = extract_key_sentences(
            full_text, query, 
            max_sentences=max_tokens_per_doc // 30  # ~30 tokens per sentence
        )
        
        compressed.append({
            **doc,
            "text": extracted,
            "compressed": True,
            "original_tokens": count_tokens(full_text),
            "compressed_tokens": count_tokens(extracted)
        })
    
    total_original = sum(d.get("original_tokens", count_tokens(d["text"])) for d in compressed)
    total_compressed = sum(count_tokens(d["text"]) for d in compressed)
    
    logging.info(f"Compression: {total_original} → {total_compressed} tokens "
                f"({(1 - total_compressed/total_original)*100:.0f}% reduction)")
    
    return compressed
```

### LLM-Based Compression (for Critical Context)

When extractive compression isn't enough, use an LLM to summarize:

```python
def compress_with_llm(text: str, query: str, max_tokens: int = 150) -> str:
    """
    Use an LLM to compress a document while preserving query-relevant facts.
    More expensive but handles complex information better.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Summarize this document in {max_tokens} tokens or fewer.
Focus on information relevant to: {query}

Preserve:
- Specific numbers, dates, and names
- Key facts and findings
- Actionable information

Discard:
- Introductory phrases
- Redundant explanations
- Tangential details

Document:
{text}"""
        }],
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content
```

> **Important:** LLM-based compression can introduce hallucinations. Always verify critical facts against the original. Extractive compression is safer for regulated industries.

---

## Stage 5: The Complete Compression Pipeline

```python
from dataclasses import dataclass, field

@dataclass
class CompressionConfig:
    target_tokens:      int   = 12_000
    min_results:        int   = 3
    max_results:        int   = 15
    dedup_threshold:    float = 0.9
    min_density:        float = 0.1
    rerank_method:      str   = "none"      # "embedding" | "none"
    compress_method:    str   = "extractive"  # "extractive" | "none"
    adaptive_threshold: bool  = True
    fixed_threshold:    float = 0.6

@dataclass
class CompressionAudit:
    stages:         list[dict]  # [{name, doc_count, token_count}, ...]
    initial_docs:   int
    initial_tokens: int

    def report(self) -> str:
        """Return a formatted ASCII table of all pipeline stages."""
        lines = [f"{'Stage':<22}  {'Docs':>5}  {'Tokens':>9}  {'% Remaining':>12}"]
        lines.append("\u2500" * 56)
        for s in self.stages:
            pct = s["token_count"] / max(self.initial_tokens, 1) * 100
            lines.append(f"{s['name']:<22}  {s['doc_count']:>5}  "
                         f"{s['token_count']:>9,}  {pct:>11.0f}%")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"initial_docs": self.initial_docs,
                "initial_tokens": self.initial_tokens,
                "stages": self.stages}

@dataclass
class CompressionResult:
    documents: list[dict]
    stats:     dict                  # stage name → {docs, tokens}
    audit:     CompressionAudit


class ContextCompressor:
    """
    Five-stage context compression pipeline.

    Usage::

        compressor = ContextCompressor(embedder=KeywordEmbedder())
        result = compressor.compress(
            query="How do I return a damaged item?",
            documents=docs,
            target_tokens=8_000,
        )
        print(result.audit.report())   # ASCII audit table
        print(result.documents)        # filtered, compressed docs
    """

    def __init__(self, embedder, cross_encoder=None):
        self.embedder      = embedder
        self.cross_encoder = cross_encoder   # optional; see note below

    def compress(self, query: str, documents: list[dict],
                 target_tokens: int = 12_000,
                 config: CompressionConfig | None = None) -> CompressionResult:
        """
        Run the full five-stage pipeline and return a CompressionResult.
        """
        cfg    = config or CompressionConfig(target_tokens=target_tokens)
        stages = []

        def snap(name: str) -> None:
            stages.append({"name": name,
                           "doc_count":   len(documents),
                           "token_count": sum(count_tokens(d["text"]) for d in documents)})

        snap("Raw Input")

        # Stage 1
        documents = self.filter_by_relevance(
            documents,
            threshold=None if cfg.adaptive_threshold else cfg.fixed_threshold,
            min_results=cfg.min_results,
            max_results=cfg.max_results,
        )
        snap("Relevance Filter")

        # Stage 2
        documents = self.filter_by_quality(
            documents, cfg.dedup_threshold, cfg.min_density
        )
        snap("Quality Filter")

        # Stage 3
        if cfg.rerank_method != "none":
            documents = self.rerank(query, documents, top_k=cfg.max_results)
        snap(f"Rerank ({cfg.rerank_method})")

        # Stage 4
        if cfg.compress_method != "none":
            tokens_per_doc = cfg.target_tokens // max(len(documents), 1)
            documents = self.compress_documents(
                query, documents, max_tokens_per_doc=max(tokens_per_doc, 50)
            )
        snap("Extract/Compress")

        # Stage 5
        documents = self.enforce_budget(documents, cfg.target_tokens)
        snap("Budget Enforcement")

        audit = CompressionAudit(
            stages=stages,
            initial_docs=stages[0]["doc_count"],
            initial_tokens=stages[0]["token_count"],
        )
        return CompressionResult(
            documents=documents,
            stats={s["name"]: {"docs": s["doc_count"], "tokens": s["token_count"]}
                   for s in stages},
            audit=audit,
        )
```

> **Note on cross-encoder fallback:** When `cross_encoder=None`, Stage 3 uses embedding cosine similarity for reranking — the same embedder passed to `__init__`. This is weaker than a true cross-encoder but avoids an external model dependency. For production accuracy, use `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`.

> **Code Reference:** [Python](../../code/python/05-context-assembly/) · [Node.js](../../code/nodejs/05-context-assembly/) · [Go](../../code/go/05-context-assembly/)  
> The context assembly implementations include the full `ContextCompressor` with all five filtering stages.

---

## The Compression Budget

Track how much information survives each stage:

In the implementation, the audit is built into `CompressionAudit`, which is returned as part of every `CompressionResult`:

```python
result = compressor.compress(query, documents, target_tokens=12_000)

# Formatted ASCII table
print(result.audit.report())

# Stage               Docs     Tokens  % Remaining
# ──────────────────────────────────────────────────────────
# Raw Input              50    125,000         100%
# Relevance Filter       22     68,000          54%
# Quality Filter         18     54,000          43%
# Rerank (none)          18     54,000          43%
# Extract/Compress       18     14,200          11%
# Budget Enforcement     10     12,000          10%

# Machine-readable dict (for logging/metrics)
print(result.audit.to_dict())
```

This audit is invaluable for tuning. If your relevance filter removes 80% of documents, your retrieval is bringing in too much noise. If your compression stage removes 90% of tokens, your chunks are too verbose.

---

## Common Pitfalls

- **"I skip filtering and just send everything"**: You're paying for tokens that make your answers worse. Relevance filtering is the highest-ROI optimization in RAG.
- **"My threshold is too aggressive and I filter everything"**: An empty context is worse than a noisy context. Always have a fallback — if filtering removes all documents, keep the top 3.
- **"I use LLM summarization for everything"**: LLM summarization can hallucinate. Use extractive compression (key sentence selection) for factual content. Reserve LLM summarization for narrative text where rewording is safe.
- **"I rerank with embeddings again"**: Reranking with the same embedding model that did the initial retrieval is pointless. You get the same order. Use a cross-encoder or a different model.
- **"I never audit my compression pipeline"**: You can't improve what you don't measure. Log the token count at each stage. You'll find stages that remove too much or too little.

## What's Next

You can now filter and compress context to maximize information density. Next: managing context across long, multi-turn conversations — maintaining continuity without overflowing the window.
→ [Multi-Turn Context Management](04-multi-turn-context-management.md)