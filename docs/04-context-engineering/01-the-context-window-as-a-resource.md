# The Context Window as a Scarce Resource

## What You'll Learn
- Why the context window is the most important budget in AI engineering
- The 3-zone model: system prompt, dynamic context, conversation history
- Token budgeting: how to allocate every token intentionally
- When to summarize, when to truncate, and when to expand
- Cost modeling for context: every token has a dollar value
- The needle-in-a-haystack problem and how context design mitigates it

## Prerequisites
- [Short-Term Memory](../03-memory-and-retrieval/01-short-term-memory.md) — the message list and its limits
- [RAG from Scratch](../03-memory-and-retrieval/03-rag-from-scratch.md) — retrieval fills the context window
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — the agent loop consumes context

---

## The Context Window Is Your Budget

Every LLM has a fixed context window. GPT-4o gives you 128K tokens. Claude gives you 200K. Gemini gives you 1M. These numbers keep growing, but the engineering principle never changes:

> **The context window is a finite resource. Every token you put in has a cost — in dollars, latency, and attention.**

Think of the context window like RAM. More is better, but you still can't load everything. And unlike RAM, you pay per byte on every access.

### What Each Token Costs

| Cost Type | Impact | Example (128K context) |
|:---|:---|:---|
| **Financial** | You pay for input tokens on every API call | 128K input tokens × $2.50/1M = $0.32 per call |
| **Latency** | More tokens = slower response | 128K input might take 5-10 seconds to process |
| **Attention** | The model's "focus" is diluted across tokens | Important instructions buried in a 100K context get forgotten |
| **Opportunity** | Tokens used now can't be used for other things | A verbose system prompt steals space from conversation history |

A 128K context window isn't "free space." It's a budget you allocate every single call.

---

## The 3-Zone Model

Every context window has three zones competing for space:

```
┌────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW                          │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ ZONE 1: SYSTEM PROMPT                                │ │
│  │ Fixed cost. Paid every call.                         │ │
│  │ Contains: role, rules, format, tool instructions     │ │
│  │ Typical allocation: 2-5% of window (2,500-6,500 tok) │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ ZONE 2: DYNAMIC CONTEXT                              │ │
│  │ Varies per request. Assembled at runtime.            │ │
│  │ Contains: RAG results, tool outputs, reference data  │ │
│  │ Typical allocation: 40-60% of window                 │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ ZONE 3: CONVERSATION HISTORY                         │ │
│  │ Grows with every turn.                               │ │
│  │ Contains: previous messages, tool calls, results     │ │
│  │ Typical allocation: whatever remains                 │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ BUFFER: Reserved for the model's response            │ │
│  │ Typical allocation: 10-20% of window                 │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### Zone 1: System Prompt — Your Fixed Tax

Every API call sends the system prompt. A 5,000-token system prompt on a 100-turn conversation costs you 500,000 tokens just for the system prompt. That's $1.25 at GPT-4o pricing — just for repeating instructions the model already knows.

**Budgeting principle:** The system prompt should be the smallest possible text that constrains the model effectively.

```python
# Bad: 2,000 tokens of personality and examples
system_prompt = """
You are a helpful, friendly, knowledgeable assistant. You always greet the
user warmly. You remember their preferences. You are patient and thorough.
[500 more words of personality description...]

Here are examples of how you should respond:
[1,000 tokens of few-shot examples...]
"""

# Good: 200 tokens of constraints
system_prompt = """
You are a customer support agent. Answer using knowledge base results.
If unsure, say you'll escalate. Format: direct answer, then sources.
"""
```

**Audit your system prompt.** If it's over 1,000 tokens, every word needs to justify its existence. Move examples and detailed instructions to dynamic context — include them only when needed.

### Zone 2: Dynamic Context — Your Variable Investment

This zone contains everything assembled at query time: RAG results, tool outputs, user-specific data, reference materials. Unlike the system prompt, this is an *investment* — you're spending tokens hoping for a better answer.

**Budgeting principle:** More context is not always better. The model's ability to use context degrades with volume.

Research consistently shows:
- **Needle-in-a-haystack tests:** Models perform best when the relevant information falls in the 20–60% range of the context. Performance drops near the very start (primacy decay) and near the end (recency competition with fresh turns). The golden middle is the sweet spot.
- **Context dilution:** Adding more irrelevant context makes the model *less* likely to find and use the relevant parts.
- **The paradox of RAG:** Retrieving more chunks can make answers worse if the chunks aren't highly relevant.

```python
# Bad: Stuff everything retrieved
retrieved = vector_store.search(query_embedding, k=20)  # All 20 results
context = "\n".join([r["text"] for r in retrieved])

# Good: Threshold-filtered, ranked, and scoped
retrieved = vector_store.search_with_threshold(
    query_embedding, threshold=0.75, k=10
)
# Only include chunks that are meaningfully similar
# Sort by score, provide the top 5 with highest relevance
context = format_context(retrieved[:5])
```

### Zone 3: Conversation History — Your Growing Liability

Every turn adds to the conversation. Long conversations create long histories. Without management, you hit the ceiling.

**Budgeting principle:** Old conversation history has diminishing returns. Summarize or truncate aggressively.

This is covered in detail in [Short-Term Memory](../03-memory-and-retrieval/01-short-term-memory.md). The key insight for context engineering: **the sliding window is your default.**

---

## Token Budgeting in Practice

Design your context allocation before writing code:

### The Default Budget (128K Window)

| Zone | Allocation | Tokens | Purpose |
|:---|:---|:---|:---|
| System Prompt | 2% | 2,560 | Core instructions |
| Tool Definitions | 5% | 6,400 | Available tools |
| Dynamic Context | 45% | 57,600 | RAG results, tool outputs |
| Conversation History | 33% | 42,240 | Previous turns |
| Response Buffer | 15% | 19,200 | Model's output |
| **Total** | **100%** | **128,000** | |

### The Agent Budget (Higher Tool Density)

Agents use more tools, so tool definitions take more space:

| Zone | Allocation | Tokens | Purpose |
|:---|:---|:---|:---|
| System Prompt | 2% | 2,560 | Agent constitution |
| Tool Definitions | 8% | 10,240 | 15-20 tools |
| Dynamic Context | 30% | 38,400 | Tool outputs |
| Conversation History | 45% | 57,600 | Agent loop history |
| Response Buffer | 15% | 19,200 | |
| **Total** | **100%** | **128,000** | |

### The RAG Budget (Higher Document Density)

RAG queries pack in more reference material:

| Zone | Allocation | Tokens | Purpose |
|:---|:---|:---|:---|
| System Prompt | 1% | 1,280 | Minimal instructions |
| Tool Definitions | 1% | 1,280 | Just the RAG tool |
| Dynamic Context | 65% | 83,200 | Retrieved documents |
| Conversation History | 18% | 23,040 | Brief history |
| Response Buffer | 15% | 19,200 | |
| **Total** | **100%** | **128,000** | |

Adjust these ratios based on your application. But define them explicitly. "Whatever fits" is not a strategy.

---

## Implementing Context Budget Enforcement

```python
@dataclass
class ContextBudget:
    """Define and enforce token allocation across context zones."""
    total_tokens: int = 128000
    system_prompt_pct: float = 0.02
    tool_definitions_pct: float = 0.05
    dynamic_context_pct: float = 0.45
    history_pct: float = 0.33
    response_buffer_pct: float = 0.15
    
    def allocate(self) -> dict:
        """Get token allocations for each zone."""
        return {
            "system_prompt": int(self.total_tokens * self.system_prompt_pct),
            "tool_definitions": int(self.total_tokens * self.tool_definitions_pct),
            "dynamic_context": int(self.total_tokens * self.dynamic_context_pct),
            "history": int(self.total_tokens * self.history_pct),
            "response_buffer": int(self.total_tokens * self.response_buffer_pct),
        }
    
    def enforce(self, messages: list[dict], 
                dynamic_context: str = "",
                tool_definitions: list[dict] = None) -> list[dict]:
        """
        Enforce the budget on a set of messages.
        
        1. Count tokens in each zone
        2. If any zone exceeds its allocation, compress or truncate
        3. Return messages that fit within the budget
        """
        allocation = self.allocate()
        
        # Measure current usage
        system_tokens = count_tokens([messages[0]])
        history_tokens = count_tokens(messages[1:])
        context_tokens = count_tokens([{"role": "system", "content": dynamic_context}])
        tool_tokens = count_tokens_for_tools(tool_definitions or [])
        
        # Enforce limits
        if history_tokens > allocation["history"]:
            messages = self._compress_history(messages, allocation["history"])
        
        if context_tokens > allocation["dynamic_context"]:
            dynamic_context = self._compress_context(dynamic_context, 
                                                     allocation["dynamic_context"])
        
        return messages, dynamic_context
    
    def _compress_history(self, messages, max_tokens):
        """Apply sliding window or summarization to fit budget."""
        ...
    
    def _compress_context(self, context, max_tokens):
        """Truncate or re-rank dynamic context to fit budget."""
        ...

# Usage
budget = ContextBudget(
    total_tokens=128000,
    dynamic_context_pct=0.50,  # Increase for RAG-heavy app
    history_pct=0.28           # Reduce accordingly
)

messages, context = budget.enforce(messages, 
                                   dynamic_context=rag_results,
                                   tool_definitions=tools)
```

> **Code Reference:** [Python](../../code/python/05-context-assembly/) · [Node.js](../../code/nodejs/05-context-assembly/) · [Go](../../code/go/05-context-assembly/)  
> The context assembly implementations include a `ContextBudget` class with zone allocation, enforcement, and overflow handling.

---

## When to Expand vs. Compress

Context engineering is about making intentional decisions for each piece of information:

| Decision | When | How |
|:---|:---|:---|
| **Include in full** | Critical for the current task | Add verbatim to dynamic context |
| **Summarize** | Relevant but not critical | Compress with an LLM, include summary |
| **Reference** | Might be needed later | Include a one-line description, fetch full text if needed |
| **Exclude** | Not relevant to current turn | Don't include at all |

```python
def prioritize_context(items: list[dict], query: str, budget_tokens: int) -> list[dict]:
    """
    Given a set of potential context items and a token budget,
    decide what to include, summarize, reference, or exclude.
    """
    # Score items by relevance
    scored = [(item, score_relevance(item, query)) for item in items]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    included = []
    tokens_used = 0
    
    for item, score in scored:
        item_tokens = count_tokens([item])
        
        if score > 0.85 and tokens_used + item_tokens < budget_tokens:
            # High relevance + fits in budget → include in full
            included.append({"type": "full", "content": item})
            tokens_used += item_tokens
        
        elif score > 0.6 and tokens_used + 200 < budget_tokens:
            # Medium relevance → include a summary
            summary = summarize_chunk(item, max_tokens=150)
            included.append({"type": "summary", "content": summary})
            tokens_used += 200
        
        elif score > 0.4:
            # Low relevance → one-line reference
            included.append({
                "type": "reference",
                "content": f"See: {item['metadata']['source']}"
            })
            tokens_used += 30
        
        # Score < 0.4 → exclude entirely
    
    return included
```

---

## The Needle-in-a-Haystack Problem

The "needle-in-a-haystack" test places a single fact ("the needle") somewhere in a long context ("the haystack") and asks the model to retrieve it. Results consistently show:

**Models perform worst when the needle is:**
- In the first ~10% of the context (primacy zone — overwritten by system-prompt attention)
- In the last ~20% (recency zone — crowded out by the most recent user turn)
- In contexts that are uniformly dense (no structural cues)

**Models perform best when the needle is:**
- 20–60% into the context (the "golden middle" — peak model attention)
- In a structurally distinct section (heading, bullet, table)
- Repeated or referenced multiple times

### Context Design for Needle Retrieval

1. **Put critical information in the middle.** Don't lead with it. Don't save it for last.
2. **Use structure.** Headings, bullet points, and tables create "landmarks" the model can navigate.
3. **Repeat key facts.** A fact mentioned once in a 100K context might be missed. Mentioned three times with different phrasing, it's found.
4. **Separate distinct topics.** A dense wall of text buries everything. Use clear section breaks.

```python
def structure_context_for_retrieval(documents: list[dict]) -> str:
    """
    Build a context string optimized for model attention.
    """
    parts = []
    
    # Lead with a table of contents (helps model build a mental map)
    parts.append("## Context Overview\n")
    for i, doc in enumerate(documents):
        source = doc["metadata"]["source"]
        parts.append(f"- Section {i+1}: {source}")
    
    parts.append("\n---\n")
    
    # Present documents with clear structural markers
    for i, doc in enumerate(documents):
        parts.append(f"## [{i+1}] {doc['metadata']['source']}\n")
        parts.append(doc["text"])
        parts.append(f"\n[End Section {i+1}]\n")
        parts.append("---\n")
    
    return "\n".join(parts)
```

---

## Context Economics: What a Token Costs

Understanding the financial cost makes budgeting real:

| Model | Input per 1M tokens | Output per 1M tokens | 128K context cost (input only) |
|:---|:---|:---|:---|
| GPT-4o | $2.50 | $10.00 | $0.32 |
| GPT-4o-mini | $0.15 | $0.60 | $0.019 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $0.38 |
| Claude 3 Haiku | $0.25 | $1.25 | $0.032 |
| Gemini 1.5 Pro | $3.50 | $10.50 | $0.45 |
| Gemini 1.5 Flash | $0.075 | $0.30 | $0.010 |

> Prices as of mid-2025. Check provider pricing pages for the latest figures.

A 5,000-token system prompt on 3,000 calls/day = 15 M tokens/day just for system prompts. That's $37.50/day with GPT-4o — just for repeating instructions. Cut it to 500 tokens and save $33.75/day, over **$12,000/year**, with zero change to your model or infrastructure.

> **The most cost-effective optimization in AI engineering is shortening your system prompt.** No model upgrade, no infrastructure change. Just delete words.

---

## Common Pitfalls

- **"I have a 128K window, so I use all 128K"**: Model attention is not uniform. Performance degrades with density. Start with the smallest context that fully answers the question; only add more when you can measure a quality improvement.
- **"Detailed instructions belong in the system prompt"**: Universal constraints go in the system prompt. Edge-case instructions belong in dynamic context — loaded only when that edge case is active. Audit your system prompt with `TokenCostCalculator.audit_context()` to find candidates for relocation.
- **"I don't track token usage per zone"**: You can't optimise what you don't measure. Use `ContextBudget.enforce()` to instrument token counts per zone on every call. Review the audit logs weekly — surprises are common.
- **"I retrieve 20 chunks because more is better"**: Quality decays with quantity. Score chunks with a similarity threshold (e.g. ≥ 0.75); 5 high-scoring chunks outperform 20 mixed-relevance ones every time.
- **"I'll add history compression when I need it"**: Every production conversation eventually hits the limit. Implement sliding-window truncation or summarisation before launch. Retrofitting it under pressure causes data-loss bugs.

## What's Next

You can now think of the context window as a budget to be managed, with distinct zones that each warrant their own enforcement policy. Next: dynamically assembling the right context for each request — prompt templates, conditional sections, and context injection.
→ [Dynamic Prompt Assembly](02-dynamic-prompt-assembly.md)

For hands-on implementation, the code examples in this chapter live in:
→ [code/python/05-context-assembly/](../../code/python/05-context-assembly/) · [Node.js](../../code/nodejs/05-context-assembly/) · [Go](../../code/go/05-context-assembly/)