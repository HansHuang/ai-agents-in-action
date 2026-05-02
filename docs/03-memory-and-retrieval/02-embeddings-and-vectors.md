# Embeddings and Vectors

## What You'll Learn
- What embeddings are: turning text into numbers the model understands
- Why embeddings matter: similarity search, clustering, and classification
- The embedding workflow: tokenize → embed → store → query → compare
- Choosing an embedding model: dimensions, cost, and language support
- Cosine similarity and other distance metrics
- How embeddings enable RAG (next chapter)

## Prerequisites
- [Short-Term Memory](01-short-term-memory.md) — why the context window needs external memory
- [How LLMs Actually Work](../01-foundations/01-how-llms-work.md) — tokens and model internals
- High school math: vectors, dot products, basic geometry

---

## The Problem: "Find Similar Things"

A user asks your agent: *"What's our return policy for damaged items?"*

Your company has 500 support documents. One of them contains the answer. How does the agent find it?

You can't:
- Search by keyword: "damaged" might not appear in the document (it says "defective")
- Search by category: the document is tagged "shipping" but the question is about "returns"
- Send all 500 documents: they won't fit in the context window
- Use an LLM to read all 500: too slow, too expensive

You need to find documents by **meaning**, not keywords. This is what embeddings do.

---

## What Is an Embedding?

An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meanings have similar vectors.

```
"The cat sat on the mat"       → [0.02, -0.14, 0.38, ..., -0.09]  (1536 numbers)
"The feline rested on the rug" → [0.03, -0.12, 0.36, ..., -0.08]  (very similar vector)
"The stock market crashed"     → [-0.41, 0.27, -0.15, ..., 0.33]  (completely different)
```

An embedding model takes text and outputs a fixed-size vector. The same model always produces the same vector dimension (e.g., 1536 for OpenAI's `text-embedding-3-small`, 3072 for `text-embedding-3-large`).

### The Key Insight

Embeddings map **semantic similarity** to **geometric proximity**. If two texts mean similar things, their vectors point in similar directions. You can measure this with math.

---

## How Embeddings Work (The 30-Second Version)

1. **Tokenize** the text into tokens
2. **Pass tokens** through a neural network trained to predict which passages are related
3. **Extract the vector** from the network's last hidden layer
4. **Output is unit-normalized** — the model outputs a vector of length 1, so you can compare angles directly without worrying about magnitude

You don't need to understand the neural network. You need to understand: **same meaning = similar vector direction.**

---

## Generating Embeddings

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Convert text to an embedding vector."""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# Example
text1 = "The cat sat on the mat"
text2 = "A feline rested on a rug"
text3 = "Stock prices fell sharply today"

embedding1 = get_embedding(text1)  # 1536 floats
embedding2 = get_embedding(text2)  # 1536 floats
embedding3 = get_embedding(text3)  # 1536 floats
```

**Important:** Always use the same embedding model for all your data. Mixing models (e.g., embedding documents with `text-embedding-3-small` but queries with `text-embedding-ada-002`) produces nonsense — the vectors don't live in the same space.

---

## Measuring Similarity: Cosine Similarity

The standard metric for comparing embeddings is **cosine similarity**. It measures the angle between two vectors, ignoring their magnitude.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

- **1.0** = vectors point in the exact same direction (identical meaning)
- **0.0** = vectors are perpendicular (unrelated)
- **-1.0** = vectors point in opposite directions (opposite meaning, rare in practice)

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_array = np.array(a)
    b_array = np.array(b)
    return np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))

# Compare our texts
sim_1_2 = cosine_similarity(embedding1, embedding2)  # ~0.85 (similar meaning)
sim_1_3 = cosine_similarity(embedding1, embedding3)  # ~0.12 (different meaning)

print(f"Cats vs Feline: {sim_1_2:.3f}")  # 0.85
print(f"Cats vs Stocks: {sim_1_3:.3f}")  # 0.12
```

### Other Distance Metrics

| Metric | Formula | When to Use |
|:---|:---|:---|
| **Cosine Similarity** | cos(θ) = A·B / (|A|×|B|) | Default choice. Ignores vector magnitude. |
| **Euclidean Distance** | √(Σ(Aᵢ - Bᵢ)²) | When magnitude matters (not typical for text) |
| **Dot Product** | A·B | Faster than cosine if vectors are already normalized |

For text embeddings, **cosine similarity is the standard.** If your embedding model normalizes vectors (most do), dot product gives the same ordering.

---

## Choosing an Embedding Model

| Model | Dimensions | Max Input | Cost (per 1M tokens) | Best For |
|:---|:---|:---|:---|:---|
| OpenAI `text-embedding-3-small` | 512 or 1536 | 8,191 | $0.02 | Cost-effective, good enough |
| OpenAI `text-embedding-3-large` | 256, 768, 1024, 3072 | 8,191 | $0.13 | Higher accuracy |
| Cohere `embed-english-v3` | 1,024 | 512 tokens per chunk | Varies | Long documents |
| Voyage AI `voyage-2` | 1,024 | 4,000 | $0.02 | Multilingual, retrieval-optimized |
| Open source (`bge-large-en`, `gte-large`) | 1,024 | 512 | Free (self-host) | Privacy, customization |

### Dimensions Trade-Off

More dimensions = more information preserved = better accuracy. But:
- **Higher storage costs:** 3072 floats vs 512 floats per vector
- **Slower similarity search:** More dimensions = more computation
- **Diminishing returns:** 1536 → 3072 might give you 2% better retrieval, not 2x

`text-embedding-3-small` at 512 dimensions is the right default. Upgrade when you have a proven accuracy problem, not before.

> **Code Reference:** [Python](../../code/python/04-rag-pipeline/) · [Node.js](../../code/nodejs/04-rag-pipeline/) · [Go](../../code/go/04-rag-pipeline/)  
> The RAG pipeline implementations include embedding generation with configurable model selection and dimension reduction.

---

## Chunking: Breaking Documents into Embeddable Pieces

Embedding models have input limits. You can't embed an entire book at once. You must break documents into **chunks** — smaller pieces that each get their own embedding.

### Why Chunking Matters

A chunk that's too large:
- Exceeds the model's input limit
- Mixed topics (a chunk about "returns" and "shipping" and "pricing" — what's this chunk "about"?)
- The embedding represents the average meaning, not any specific meaning

A chunk that's too small:
- Lacks context ("Click the button" — what button? Where?)
- The user's query might not match this tiny fragment
- Too many chunks to search

### Chunking Strategies

**Fixed-Size Chunking (Simplest)**
```python
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")  # matches text-embedding-3-* models

def chunk_by_tokens(text: str, chunk_size: int = 256, overlap: int = 50) -> list[str]:
    """Split text into chunks of roughly equal token count, with overlap."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start = end - overlap  # Overlap preserves context across boundaries
    return chunks
```

**Semantic Chunking (Smarter)**
```python
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def chunk_by_semantics(text: str, max_chunk_size: int = 512) -> list[str]:
    """Split at natural boundaries: paragraphs, sections, sentences."""
    # Split by double newlines first (paragraphs)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        candidate = current_chunk + "\n\n" + para if current_chunk else para
        if count_tokens(candidate) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk = candidate
    
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
```

**Hierarchical Chunking (Most Robust)**
```python
# Break a document into:
# - Large chunks (parent): broader context
# - Small chunks (child): specific details
# On retrieval, you might return the matching child's parent for full context

document_structure = {
    "title": "Return Policy",
    "sections": [
        {"heading": "Damaged Items", "content": "...", 
         "chunks": [chunk1, chunk2]},
        {"heading": "Refund Timeline", "content": "...",
         "chunks": [chunk3, chunk4]}
    ]
}
```

### The Overlap Rule

Set overlap to 10–20% of `chunk_size` in tokens (e.g., 25–50 tokens for a 256-token chunk). If a critical sentence is split across a chunk boundary, the overlap ensures it appears in full in at least one chunk.

```
Chunk 1: "...the customer must return the item within 30 days of | purchase..."
Chunk 2: "...within 30 days of | purchase. A receipt is required for all..."
```

The `|` marks the boundary. Without overlap, the sentence "within 30 days of purchase" is broken. With 50-token overlap, it appears completely in Chunk 2.

---

## The Embedding Workflow

Every RAG system follows this pipeline:

```
1. INGEST
   Documents → Clean → Chunk → Embed → Store
   
2. QUERY
   User question → Clean → Embed → Search → Get top K chunks
   
3. AUGMENT
   Original question + Retrieved chunks → Prompt
   
4. GENERATE
   Prompt → LLM → Answer with citations
```

The next chapter covers the full RAG pipeline. For now, understand: **embeddings are how you find relevant chunks. The vector database is where you store them.**

---

## Storing Embeddings

You need somewhere to put your vectors and search them efficiently.

### Option 1: In-Memory (Prototyping Only)
```python
class SimpleVectorStore:
    def __init__(self):
        self.vectors = []  # List of (text, embedding) tuples
    
    def add(self, text: str, embedding: list[float]) -> None:
        self.vectors.append({"text": text, "embedding": embedding})
    
    def search(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        """Brute force search. O(n) — doesn't scale."""
        scored = [
            {"text": item["text"], 
             "score": cosine_similarity(query_embedding, item["embedding"])}
            for item in self.vectors
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
```

This works for 100 documents. It doesn't work for 100,000.

### Option 2: Vector Database (Production)

Vector databases index embeddings for fast approximate nearest neighbor (ANN) search. They trade a tiny amount of accuracy for massive speed gains.

| Database | Type | Best For |
|:---|:---|:---|
| **Chroma** | Embedded, open source | Prototyping, small datasets |
| **Qdrant** | Standalone or embedded | Fast, filtering, production |
| **Pinecone** | Managed cloud | Zero-ops, large scale |
| **Weaviate** | Open source or cloud | Hybrid search (vector + keyword) |
| **pgvector** | PostgreSQL extension | If you already use Postgres |
| **Milvus** | Open source, distributed | Billion-scale vectors |

> Deep dive into vector databases: [Vector Databases](../05-the-tool-ecosystem/02-vector-databases.md)

---

## When Embeddings Fail

Embeddings are powerful but not magic. They fail when:

| Failure Mode | Example | Fix |
|:---|:---|:---|
| **Keywords matter more than meaning** | "Error code 503" vs "server error" — a keyword search finds 503, an embedding search might not | Hybrid search (keyword + vector) |
| **Exact match required** | "Invoice #INV-2024-0782" — embeddings won't match this exact string | Keyword search for structured IDs |
| **Negation** | "Not suitable for outdoor use" — embeddings often ignore negation | Larger chunks with more context |
| **Rare terms** | Technical jargon, product codes — the embedding model hasn't seen these | Fine-tuned embedding model |
| **Multiple topics in one chunk** | A chunk covers 3 different topics, the embedding is a blurry average | Smaller chunks, better chunking |

---

## Common Pitfalls

- **"I use different embedding models for documents and queries"**: The vectors live in different spaces. Cosine similarity between them is meaningless. Use one model for everything.
- **"My chunks are too large"**: A 4,000-token chunk containing 5 topics has an embedding that represents none of them well. Aim for 256-512 tokens per chunk.
- **"My chunks have no overlap"**: Critical information gets cut at boundaries. 10-20% overlap is cheap insurance.
- **"I re-embed everything on every query"**: Embeddings are computed once per document, cached, and reused. Re-embedding on every query burns money with no benefit.
- **"I mix embedding models"**: Switching models — even between versions like `ada-002` → `text-embedding-3-small` — means old and new vectors live in different spaces. Cosine similarity between them is meaningless. Lock in your model before indexing and re-embed the entire corpus if you ever change it. (Note: OpenAI's `text-embedding-3-*` models use Matryoshka Representation Learning, so a 512-dim vector from the same model is comparable to a 1536-dim one from the same model — but cross-model comparisons are never valid.)

## What's Next

You can now turn text into searchable vectors. Next: the complete RAG pipeline — ingesting documents, retrieving relevant chunks, and augmenting the LLM's context to produce grounded answers.
→ [RAG from Scratch](03-rag-from-scratch.md)