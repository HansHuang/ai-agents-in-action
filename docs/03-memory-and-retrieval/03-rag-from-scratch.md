# RAG from Scratch

## What You'll Learn
- The complete RAG pipeline: Ingest → Retrieve → Augment → Generate
- Building a RAG system without frameworks — every line from scratch
- Chunking strategies that actually work in production
- The retrieval-augmented generation loop in detail
- How RAG and agents combine: the agent decides when to retrieve
- Evaluating retrieval quality: hit rate, MRR, and precision@k

## Prerequisites
- [Embeddings and Vectors](02-embeddings-and-vectors.md) — how text becomes searchable
- [Short-Term Memory](01-short-term-memory.md) — context window management
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — RAG as a tool in the agent loop

---

## What Is RAG?

**R**etrieval-**A**ugmented **G**eneration. A pattern that gives LLMs access to external knowledge without fine-tuning.

Instead of hoping the model memorized your company's return policy during training, you:

1. **Ingest** your documents (chunk them, embed them, store them)
2. **Retrieve** the most relevant chunks when a user asks a question — and filter by similarity threshold
3. **Augment** the prompt with those chunks
4. **Generate** a grounded answer based on the retrieved context

```
User: "What's our return policy for damaged items?"
     │
     ▼
┌─────────────────┐
│ 1. EMBED QUERY  │  "What's our return policy for damaged items?"
│    → vector     │  → [0.02, -0.14, 0.38, ...]
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. SEARCH       │  cosine_similarity(query, doc_chunks)
│    vector DB    │  → Top K results, filtered by threshold ≥ 0.7
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. AUGMENT      │  System: "Answer using only the provided documents."
│    prompt       │  Documents: [chunk1, chunk2, chunk3]
│                 │  User: "What's our return policy for damaged items?"
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. GENERATE     │  LLM: "For damaged items, you can return them within
│    answer       │   30 days of delivery. Include photos of the damage..."
└─────────────────┘
```

> **Code Reference:** [Python](../../code/python/04-rag-pipeline/) · [Node.js](../../code/nodejs/04-rag-pipeline/) · [Go](../../code/go/04-rag-pipeline/)  
> The RAG pipeline folder contains the complete implementation with a working demo that ingests sample documents and answers queries.

---

## Phase 1: Ingest

Turn your documents into searchable vectors.

### Step 1.1: Load Documents

```python
def load_documents(directory: str) -> list[dict]:
    """Load all text files from a directory with metadata."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(('.txt', '.md', '.rst')):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                text = f.read()
            documents.append({
                "id": filename,
                "text": text,
                "metadata": {
                    "source": filename,
                    "type": filename.split('.')[-1],
                    "path": filepath
                }
            })
    return documents
```

### Step 1.2: Chunk Documents

Use the chunking strategies from the previous chapter. For RAG, semantic chunking with overlap is the default:

```python
chunker = DocumentChunker(
    chunk_size=256,    # Tokens per chunk
    overlap=50,        # Overlap between chunks
    strategy="semantic"  # Split at paragraph boundaries
)

all_chunks = []
for doc in documents:
    chunks = chunker.chunk(doc["text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"{doc['id']}_chunk_{i}",
            "text": chunk.text,
            "metadata": {
                **doc["metadata"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        })
```

### Step 1.3: Generate Embeddings

```python
embedder = EmbeddingGenerator(model="text-embedding-3-small")

for chunk in all_chunks:
    chunk["embedding"] = embedder.embed(chunk["text"])
```

### Step 1.4: Store in Vector Database

```python
vector_store = SimpleVectorStore()

for chunk in all_chunks:
    vector_store.add(
        text=chunk["text"],
        embedding=chunk["embedding"],
        metadata=chunk["metadata"]
    )

# For production, use a real vector database:
# vector_store = QdrantClient(...)
# vector_store = PineconeClient(...)
# vector_store = WeaviateClient(...)
```

The ingest phase runs once (or periodically, as documents change). The retrieval phase runs on every query.

---

## Phase 2: Retrieve

Given a user query, find the most relevant chunks.

```python
def retrieve(query: str, vector_store, embedder, k: int = 5) -> list[dict]:
    """
    Retrieve the k most relevant chunks for a query.
    
    Returns:
        [{"text": "...", "score": 0.92, "metadata": {...}}, ...]
    """
    # Embed the query (same model as documents)
    query_embedding = embedder.embed(query)
    
    # Search the vector store
    results = vector_store.search(
        query_embedding=query_embedding,
        k=k
    )
    
    return results
```

### Retrieval Quality: Not All Results Are Worth Using

A similarity score of 0.5 means the chunk is weakly related. Including it in the prompt adds noise, not signal.

```python
def retrieve_with_threshold(
        query: str,
        vector_store,
        embedder,
        k: int = 5,
        threshold: float = 0.7,
) -> list[dict]:
    """Retrieve but only return results above a similarity threshold."""
    query_embedding = embedder.embed(query)
    results = vector_store.search(query_embedding, k=k)
    return [r for r in results if r["score"] >= threshold]

# Fallback: if no results meet the threshold, tell the user you don't know
def retrieve_or_empty(
        query: str,
        vector_store,
        embedder,
        k: int = 5,
        threshold: float = 0.7,
) -> list[dict]:
    results = retrieve_with_threshold(query, vector_store, embedder, k, threshold)
    if not results:
        return []  # Agent will say "I don't have information about that"
    return results
```

---

## Phase 3: Augment

Build the prompt that includes retrieved context.

### The Augmentation Prompt Template

```python
RAG_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based on the provided documents.

Rules:
- Answer ONLY using information from the documents below.
- If the documents don't contain the answer, say "I don't have information about that in my knowledge base."
- Cite the source document when providing information.
- If multiple documents are relevant, synthesize information from all of them.
- Do not use any knowledge outside the provided documents.

Documents:
{document_context}
"""

def build_rag_prompt(query: str, retrieved_docs: list[dict]) -> tuple[str, list[dict]]:
    """
    Build the system prompt and messages for a RAG query.
    
    Returns: (system_prompt, messages)
    """
    # Build document context string
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc["metadata"].get("source", "Unknown")
        context_parts.append(
            f"[Document {i+1} - Source: {source}]\n{doc['text']}"
        )
    
    document_context = "\n\n---\n\n".join(context_parts)
    
    # Build system prompt
    system_prompt = RAG_SYSTEM_PROMPT.format(document_context=document_context)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    return messages
```

### The Citation Pattern

Teach the model to cite sources:

```markdown
When answering, cite the source document in brackets.
Example: "You can return damaged items within 30 days [Source: return-policy.md]."

If information comes from multiple documents, cite each one:
"Standard shipping takes 3-5 days [Source: shipping-info.md]. 
Express shipping is 1-2 days [Source: shipping-info.md]. 
Free shipping applies to orders over $50 [Source: promotions.md]."
```

---

## Phase 4: Generate

Call the LLM with the augmented prompt.

```python
def generate_rag_response(query: str, retrieved_docs: list[dict]) -> str:
    """Generate an answer using retrieved documents."""
    
    # Build the augmented prompt
    messages = build_rag_prompt(query, retrieved_docs)
    
    # Call the LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3  # Lower temperature for factual responses
    )
    
    return response.choices[0].message.content
```

---

## The Complete RAG Pipeline

Putting all four phases together:

```python
class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.
    Ingest → Retrieve → Augment → Generate
    """
    
    def __init__(self, vector_store, embedder, model: str = "gpt-4o"):
        self.vector_store = vector_store
        self.embedder = embedder
        self.model = model
    
    def ingest_documents(self, directory: str) -> int:
        """Load, chunk, embed, and store all documents in a directory."""
        documents = load_documents(directory)
        chunker = DocumentChunker(chunk_size=256, overlap=50, 
                                  strategy="semantic")
        
        total_chunks = 0
        for doc in documents:
            chunks = chunker.chunk(doc["text"])
            for i, chunk in enumerate(chunks):
                embedding = self.embedder.embed(chunk.text)
                self.vector_store.add(
                    text=chunk.text,
                    embedding=embedding,
                    metadata={
                        "source": doc["id"],
                        "chunk_index": i
                    }
                )
                total_chunks += 1
        
        return total_chunks
    
    def query(self, question: str, k: int = 5, 
              threshold: float = 0.7) -> dict:
        """
        Answer a question using RAG.
        
        Returns: {
            "answer": "...",
            "sources": ["doc1.md", "doc2.md"],
            "retrieved_chunks": [...],
            "similarity_scores": [0.92, 0.87]
        }
        """
        # Phase 2: Retrieve
        results = retrieve_with_threshold(
            question, self.vector_store, self.embedder, k, threshold
        )
        
        if not results:
            return {
                "answer": "I don't have information about that in my knowledge base.",
                "sources": [],
                "retrieved_chunks": [],
                "similarity_scores": []
            }
        
        # Phase 3: Augment
        messages = build_rag_prompt(question, results)
        
        # Phase 4: Generate
        response = self._self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": list(set(
                r["metadata"]["source"] for r in results
            )),
            "retrieved_chunks": [r["text"] for r in results],
            "similarity_scores": [r["score"] for r in results]
        }
```

---

## RAG as an Agent Tool

RAG isn't an alternative to agents. It's a **tool** the agent uses.

```python
def create_rag_tool(rag_pipeline: RAGPipeline):
    """Create a tool definition that wraps the RAG pipeline."""
    return {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the company knowledge base for information. "
                          "Use this when the user asks about policies, procedures, "
                          "or any company-specific information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific. "
                                      "Example: 'return policy for damaged electronics'"
                    }
                },
                "required": ["query"]
            }
        }
    }

# Agent uses RAG like any other tool:
def agent_rag_search(query: str) -> str:
    """Agent-callable RAG tool."""
    result = rag_pipeline.query(query)
    return json.dumps({
        "answer": result["answer"],
        "sources": result["sources"]
    })
```

Now your agent can decide *when* to retrieve. It might answer "What's 2+2?" directly, but call `search_knowledge_base` for "What's our return policy?" This is the agent-RAG synergy.

---

## Evaluating RAG Quality

You can't just eyeball RAG answers. You need metrics.

### Retrieval Metrics

| Metric | What It Measures | Formula |
|:---|:---|:---|
| **Hit Rate** | How often the correct chunk is in the top K results | (queries where correct chunk found) / (total queries) |
| **MRR** (Mean Reciprocal Rank) | How high the correct chunk ranks | (1/rank) averaged across queries |
| **Precision@K** | What fraction of top K results are relevant | (relevant in top K) / K |

```python
def evaluate_retrieval(pipeline: RAGPipeline, 
                       test_queries: list[dict]) -> dict:
    """
    Evaluate retrieval quality.
    
    test_queries = [
        {
            "query": "What's the return policy?",
            "relevant_doc_ids": ["return-policy.md"]
        },
        ...
    ]
    """
    hits = 0
    reciprocal_ranks = []
    
    for test in test_queries:
        results = retrieve(test["query"], pipeline.vector_store, 
                          pipeline.embedder, k=10)
        retrieved_ids = [r["metadata"]["source"] for r in results]
        
        # Hit rate
        if any(doc_id in retrieved_ids for doc_id in test["relevant_doc_ids"]):
            hits += 1
        
        # Reciprocal rank
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in test["relevant_doc_ids"]:
                reciprocal_ranks.append(1 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0)
    
    return {
        "hit_rate": hits / len(test_queries),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "total_queries": len(test_queries)
    }
```

### Generation Metrics

| Metric | What It Measures | How | Score Range |
|:---|:---|:---|:---|
| **Faithfulness** | Does the answer stick to the documents? | LLM-as-judge: "Does this answer contain claims not in the documents?" | 0 = hallucinated, 1 = fully grounded |
| **Relevance** | Does the answer address the question? | LLM-as-judge: "Does this answer directly address the question?" | 0–1 float score |
| **Groundedness** | Are claims cited? | Ratio of cited claims to total claims | cited_claims / total_claims |

> Evaluation is covered in depth in [Evaluating Agents](../08-evaluation-and-guardrails/01-evaluating-agents.md).

---

## RAG Patterns: Beyond Basic Retrieval

### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer first, then use *that* as the search query. The hypothetical answer is closer to document content than a short question.

```python
def hyde_retrieve(question: str, vector_store, embedder, k: int = 5) -> list[dict]:
    """Generate a hypothetical answer, then search with it."""
    # Step 1: Generate a hypothetical answer
    hypothetical = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Write a detailed answer to this question: {question}"
        }]
    ).choices[0].message.content
    
    # Step 2: Use the hypothetical answer as the search query
    return retrieve(hypothetical, vector_store, embedder, k=k)
```

HyDE helps when user queries are short or use different vocabulary than your documents.

> **Cost tradeoff**: HyDE makes one extra LLM call per query to generate the hypothesis. Use it when retrieval quality matters more than latency, not on every query by default.

### Multi-Query Retrieval

Generate multiple search queries from one user question, retrieve for each, and deduplicate.

```python
def multi_query_retrieve(question: str, vector_store, embedder, 
                         n_queries: int = 3, k: int = 3) -> list[dict]:
    """Generate multiple search queries for better coverage."""
    # Generate alternative queries
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Generate {n_queries} different search queries to find "
                       f"information about: {question}\n"
                       f'Output a JSON array of strings only, e.g. ["query1", "query2"].'
        }],
        response_format={"type": "json_object"},
    )
    queries = json.loads(response.choices[0].message.content)
    
    # Retrieve for each query, collect unique results
    all_results = {}
    for query in queries:
        for result in retrieve(query, vector_store, embedder, k=k):
            if result["text"] not in all_results:
                all_results[result["text"]] = result
    
    # Sort by score, return top K
    return sorted(all_results.values(), 
                  key=lambda x: x["score"], reverse=True)[:k]
```

---

## Common Pitfalls

- **"I chunk everything the same way"**: Different document types need different chunking. API docs: chunk by endpoint. FAQ: chunk by Q&A pair. Articles: chunk by section. One size doesn't fit all.
- **"I use the same embedding model forever"**: Switching embedding models requires re-embedding every document. Lock in your model before going to production. Pin the model name in a config constant (e.g. `EMBEDDING_MODEL = "text-embedding-3-small"`) and treat it like a database schema — a change requires a migration.
- **"I stuff all retrieved chunks into the prompt"**: Retrieval returns 10 chunks, but 3 have a score of 0.5. Those 3 are noise. Use a similarity threshold or dynamically select the top K based on score gaps.
- **"I don't tell the model to say 'I don't know'"**: Without explicit instructions, the model will hallucinate an answer when the documents don't contain the information. Always include: "If the answer is not in the documents, say so."
- **"My RAG pipeline has no evaluation"**: "It looks right" isn't evaluation. Build a test set of 20-50 queries with known relevant documents. Measure hit rate and MRR. Track them over time.
- **"I treat RAG as a separate system, not an agent tool"**: RAG is retrieval. The agent decides when to retrieve. Don't force every query through RAG — let the agent route simple questions directly.

## What's Next

You can now build a complete RAG system from scratch. Next: context engineering — treating the context window as a managed resource and dynamically assembling the right information for each LLM call.
→ [The Context Window as a Resource](../04-context-engineering/01-the-context-window-as-a-resource.md)