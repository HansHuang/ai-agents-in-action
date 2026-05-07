# 05 — Context Assembly

Demonstrates the context window as a **scarce resource to be budgeted**, not a
pipe to be filled — and prompts as **dynamic, version-controlled artifacts**
that adapt to every request.

## What this chapter shows

| Theme | Module |
|---|---|
| Zone-based token budgeting | `context_budget.py` |
| Attention-aware structuring | `context_optimizer.py` |
| Multi-source dynamic assembly | `context_assembler.py` |
| Cost modelling & waste detection | `token_cost_calculator.py` |
| **Dynamic prompt assembly** | `prompt_assembler.py` |
| **Version-controlled YAML templates** | `prompt_library.py` |
| **Condition DSL for prompt sections** | `condition_engine.py` |
| **Context compression pipeline** | `context_compressor.py` |
| **Information-density scoring** | `density_analyzer.py` |
| **Extractive sentence compression** | `extractive_summarizer.py` |

## Files

| File | Purpose |
|---|---|
| `context_budget.py` | `ContextBudget` — allocate tokens across zones, enforce limits, audit compression |
| `context_optimizer.py` | `ContextOptimizer` — structure docs for attention, deduplicate, place needles |
| `context_assembler.py` | `ContextAssembler` — assemble RAG docs, tool results, profile, and summary into one context |
| `token_cost_calculator.py` | `TokenCostCalculator` — calculate call cost, compare models, detect prompt waste |
| `prompt_assembler.py` | `PromptAssembler` — templates, conditional sections, multi-source injection, budget enforcement |
| `prompt_library.py` | `PromptLibrary` — YAML-based version-controlled template management with hot-reload |
| `condition_engine.py` | `ConditionEngine` — safe DSL evaluator for `"plan == 'premium' AND country == 'EU'"` |
| `prompts/` | Example YAML templates: `support_base`, `support_billing`, `support_technical` |
| `context_compressor.py` | `ContextCompressor` — five-stage pipeline: relevance → dedup → rerank → extract → budget |
| `density_analyzer.py` | `InformationDensityAnalyzer` — score text for fact density, structure, and filler ratio |
| `extractive_summarizer.py` | `ExtractiveSummarizer` — pick query-relevant sentences verbatim (zero hallucination risk) |
| `test_context_engineering.py` | 56 offline tests covering the four context modules |
| `test_prompt_assembly.py` | 50 offline tests covering the three prompt assembly modules |
| `test_compression.py` | 28 offline tests covering the three compression modules |

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

### Context budget and assembly (chapter 01)

```python
from context_budget import ContextBudget
from context_assembler import ContextAssembler

# 1. Create a budget for an 8K-token model call
budget = ContextBudget(total_tokens=8_000, model="gpt-4o-mini")

# 2. Assemble context from multiple sources
asm = ContextAssembler(budget)
result = asm.assemble(
    template="You are a support agent for $company.",
    variables={"company": "Acme Corp"},
    retrieved_docs=[{"text": "Return policy: 30 days.", "metadata": {"source": "kb"}}],
    user_profile={"name": "Alice", "plan": "Pro"},
)

print(result.sources_included)   # ["rag", "profile"]
print(result.token_breakdown)    # {"template": 12, "rag": 45, "profile": 8}
```

### Dynamic prompt assembly (chapter 02)

```python
from prompt_assembler import PromptAssembler, format_rag_results, format_user_profile

assembler = PromptAssembler()

# 1. Register a template
assembler.register_template("support", """
You are a support agent for {company}.
{sections}
{context}
""".strip())

# 2. Register conditional sections
assembler.register_section(
    "premium_user",
    "- Provide priority service to this premium customer.",
    condition=lambda v: v.get("plan") == "premium",
)

# 3. Register context source formatters
assembler.register_source_formatter("rag",  format_rag_results,  priority=3)
assembler.register_source_formatter("profile", format_user_profile, priority=2)

# 4. Assemble — sections and sources adapt to the variables
prompt = assembler.assemble(
    "support",
    {"company": "Acme Corp", "plan": "premium"},
    context_sources={"rag": rag_docs, "profile": {"name": "Alice", "plan": "Premium"}},
)
```

### YAML template library

```python
from prompt_library import PromptLibrary

library = PromptLibrary("prompts/")

result = library.render("support_base", {
    "role":            "support agent",
    "company_name":    "Acme Corp",
    "customer_name":   "Alice",
    "customer_plan":   "premium",
    "customer_region": "EU",
    "guidelines":      "Be concise and cite sources.",
    "context":         "(RAG results here)",
})

print(result.template_version)    # "1.2.0"
print(result.sections_included)   # ["premium_experience", "gdpr_notice"]
print(result.token_count)         # 312
```

### Context compression pipeline (chapter 03)

```python
from context_compressor import ContextCompressor, CompressionConfig
from extractive_summarizer import KeywordEmbedder

compressor = ContextCompressor(KeywordEmbedder())

docs = [
    {"text": "Damaged items must be reported within 48 hours with photos.", "score": 0.92},
    {"text": "Standard returns are accepted within 30 days of purchase.",     "score": 0.65},
    {"text": "We offer free shipping on orders over $50.",                     "score": 0.30},
    {"text": "Our headquarters is located in Austin, Texas.",                   "score": 0.08},
]

result = compressor.compress(
    query="How do I return a damaged item?",
    documents=docs,
    target_tokens=500,
    config=CompressionConfig(min_results=2),
)

print(result.audit.report())   # ASCII table of all five stages
print(len(result.documents))   # reduced, query-relevant set
```

> **Key insight:** Better context beats more context. Filter ruthlessly.
> Every irrelevant chunk wastes attention budget and degrades answer quality.

### Condition DSL

```python
from condition_engine import ConditionEngine

engine = ConditionEngine()

engine.evaluate("plan == 'premium'",                        {"plan": "premium"})  # True
engine.evaluate("country in ['DE', 'FR']",                  {"country": "US"})    # False
engine.evaluate("plan == 'premium' AND country == 'EU'",    {"plan": "premium", "country": "EU"})  # True

print(engine.explain("plan == 'premium' AND score > 0.5", {"plan": "premium", "score": 0.85}))
```

## Assembly pipeline

```
Template (YAML or registered string)
    │
    ├─ evaluate conditions → include matching sections
    │
    ├─ format context sources (RAG, profile, tools, history)
    │       └─ sort by priority, enforce per-source token limit
    │
    ├─ fill {placeholders}
    │
    ├─ enforce total token budget → drop low-priority sources
    │
    └─ return assembled prompt string
```

## Compression pipeline

```
Raw retrieved documents
    │
    ├─ 1. Relevance filter  → drop score < adaptive threshold (keeps ≥ min_results)
    │
    ├─ 2. Quality filter   → remove near-duplicates (3-gram Jaccard) + low-density chunks
    │
    ├─ 3. Rerank           → re-score by embedding cosine similarity (optional)
    │
    ├─ 4. Extract/Compress → keep only query-relevant sentences verbatim
    │
    ├─ 5. Budget enforce   → drop lowest-scoring docs until total ≤ target_tokens
    │
    └─ CompressionResult   → documents + stats + CompressionAudit
```

> **Key insight:** Templates are code. Version them, test them, review them.
> A prompt change that degrades output quality is a production incident.
> Treat it like one.

## Architecture (context budget modules)

The five budget zones and their default allocations:

| Zone | Default |
|---|---|
| `system_prompt` | 2 % |
| `tool_definitions` | 5 % |
| `dynamic_context` | 45 % |
| `conversation_history` | 33 % |
| `response_buffer` | 15 % |

---

## Multi-turn conversation management

Four additional modules implement the full lifecycle of long-running, multi-turn
conversations — the focus of Chapter 4 §4 in the documentation.

### What this shows

| Theme | Module |
|---|---|
| Persistent state across context truncation | `state_manager.py` |
| Incremental layered summarisation | `progressive_summarizer.py` |
| Session lifecycle, branching, persistence | `session_manager.py` |
| Health diagnosis and recovery interventions | `recovery_manager.py` |

### Files

| File | Purpose |
|---|---|
| `state_manager.py` | `ConversationState` + `StateManager` — goal tracking, user info extraction, drift detection, recovery actions |
| `progressive_summarizer.py` | `ProgressiveSummarizer` — verbatim ring-buffer + 3 compressed layers, cascaded overflow |
| `session_manager.py` | `Session` + `SessionManager` — TTL expiry, branching, reset, JSON persistence |
| `recovery_manager.py` | `GoalDetector` + `RecoveryManager` — issue diagnosis, targeted prompt interventions |
| `test_multi_turn.py` | 57 offline tests covering all four modules |

### Quick start

```python
from session_manager import SessionManager

mgr = SessionManager(ttl_minutes=30)
session = mgr.get_session("alice")

# Set the conversation goal
session.state_manager.set_goal(
    "Book a round-trip to London",
    ["choose dates", "select seat", "confirm payment"],
)

# Normal conversation loop
session.add_user_message("I'd like to fly June 15 and return June 22.")
session.add_agent_message("Searching June 15–22 flights from your location.")
session.state_manager.mark_subtask_complete("choose dates")

# Build a token-budgeted message list for the next LLM call
messages = session.build_messages_for_llm(
    system_prompt="You are a travel booking assistant.",
    max_tokens=8_000,
)

# State and summaries are automatically injected into the system prompt
print(messages[0]["content"])   # augmented system prompt with state context
```

```python
from recovery_manager import RecoveryManager

recovery = RecoveryManager(session.state_manager)
issues = recovery.diagnose()
if issues:
    intervention = recovery.get_intervention(issues)
    # Prepend intervention to the next system prompt or inject as a message
```

### Architecture

The modules form a layered stack, each building on the one below:

```
SessionManager           ← session lifecycle, expiry, branching, persistence
    └─ Session
        ├─ StateManager  ← goal + user info; survives message truncation
        └─ ProgressiveSummarizer  ← verbatim ring-buffer + 3 compression layers
                                    older turns → Layer 1 (detailed)
                                    older still → Layer 2 (compressed)
                                    oldest      → Layer 3 (archival)

RecoveryManager          ← diagnoses drift / frustration / stalemate
    └─ GoalDetector      ← LLM-assisted or heuristic goal/completion detection
```

> **Key insight:** Never re-summarise on every turn.  
> A ring-buffer of verbatim recent turns plus incrementally-updated compressed
> layers is both cheaper and more faithful than reprocessing the entire history
> each time.

### Running the tests

```bash
pytest test_multi_turn.py -v
# or combined with the rest of the chapter tests:
pytest test_context_engineering.py test_prompt_assembly.py test_compression.py test_multi_turn.py -v
```

## Key insight

> The most cost-effective optimisation is **shortening your system prompt**.  
> A 500-token system prompt repeated across 10 000 daily calls costs as much as
> 5 million extra tokens per day — before a single user message is sent.

`TokenCostCalculator.optimize_system_prompt()` and `audit_context()` automate
the detection and measurement of that waste.

## Running the tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tiktoken pytest pyyaml
pytest test_context_engineering.py test_prompt_assembly.py test_compression.py -v
```

## See also

- Context budget concepts: [docs/04-context-engineering/01-the-context-window-as-a-resource.md](../../../../docs/04-context-engineering/01-the-context-window-as-a-resource.md)
- Dynamic prompt assembly concepts: [docs/04-context-engineering/02-dynamic-prompt-assembly.md](../../../../docs/04-context-engineering/02-dynamic-prompt-assembly.md)
- Context compression concepts: [docs/04-context-engineering/03-context-compression-and-filtering.md](../../../../docs/04-context-engineering/03-context-compression-and-filtering.md)
- Multi-turn context management: [docs/04-context-engineering/04-multi-turn-context-management.md](../../../../docs/04-context-engineering/04-multi-turn-context-management.md)
- Node.js / TypeScript implementation: [code/nodejs/05-context-assembly/](../../nodejs/05-context-assembly/)
- Go implementation: [code/go/05-context-assembly/](../../go/05-context-assembly/)
