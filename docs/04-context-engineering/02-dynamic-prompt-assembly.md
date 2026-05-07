# Dynamic Prompt Assembly

## What You'll Learn
- Why static prompts fail at scale: one size doesn't fit all queries
- Prompt templates: separating structure from content
- Conditional prompt sections: include instructions only when needed
- Multi-source context injection: RAG results, user profiles, tool outputs
- The template engine pattern: build prompts programmatically
- Version-controlling prompts alongside code

## Prerequisites
- [The Context Window as a Resource](01-the-context-window-as-a-resource.md) — the budget you're assembling into
- [Prompt Engineering](../01-foundations/02-prompt-engineering.md) — prompt fundamentals
- [RAG from Scratch](../03-memory-and-retrieval/03-rag-from-scratch.md) — RAG is a primary context source

---

## The Problem with Static Prompts

A static prompt is written once and used for every request:

```python
# Static prompt — same for everyone
SYSTEM_PROMPT = """
You are a customer support agent for Acme Corp.
You handle orders, returns, shipping, and billing.
Always be polite and professional.
If you don't know, escalate to a human agent.
"""
```

This works for 10 users. It fails at scale because:

- **A billing question** doesn't need shipping instructions
- **A premium customer** should get different treatment than a free-tier user
- **A returning user** already knows the basics — repeating them wastes tokens
- **A user in Germany** needs different return policy information
- **A user at 2 AM** might be frustrated and needs a different tone

Static prompts treat every request as identical. Dynamic prompts adapt.

---

## The Template Engine Pattern

Separate the prompt's **structure** from its **content**. The template is code. The content is data.

```python
# Template: structure (lives in code, version-controlled)
# Uses {variable} placeholders — consistent with str.format_map()
SUPPORT_TEMPLATE = """
You are a customer support agent for {company_name}.

## Your Role
{role_description}

## Current Customer
Name: {customer_name}
Plan: {customer_plan}
Location: {customer_location}

## Relevant Policies
{policy_context}

## Conversation History Summary
{conversation_summary}

## Instructions
- Answer using the policies above.
- {tone_instruction}
- If unsure, {escalation_instruction}
- Response language: {language}
"""

# Data: content (assembled at runtime)
prompt = SUPPORT_TEMPLATE.format_map({
    "company_name":       "Acme Corp",
    "role_description":   "Handle billing inquiries. Do NOT handle technical support.",
    "customer_name":      user.name,
    "customer_plan":      user.plan,
    "customer_location":  user.country,
    "policy_context":     rag_results,
    "conversation_summary": memory.get_summary(),
    "tone_instruction":   "Be especially patient and thorough." if user.is_frustrated else "Be concise.",
    "escalation_instruction": "transfer to billing-dept@acme.com",
    "language":           "German" if user.country == "DE" else "English",
})
```

The key insight: **the template is deterministic. The variables make it dynamic.** You can test the template independently of the data.

---

## Conditional Prompt Sections

Not every instruction applies to every request. Conditional sections include instructions only when relevant.

### Pattern 1: Boolean Conditions

This approach is simple and readable — but tightly couples prompt logic to
application code and doesn't version-control cleanly. Use it for simple cases.
For templates shared across teams or stored in YAML (see below), Pattern 2 is
more maintainable.

```python
def build_instructions(user, query_type):
    instructions = []
    
    # Always include
    instructions.append("Answer using the provided policies.")
    
    # Conditional: only for billing questions
    if query_type == "billing":
        instructions.append("Show exact amounts in the customer's local currency.")
        instructions.append("Include payment due dates if applicable.")
    
    # Conditional: only for premium customers
    if user.plan == "premium":
        instructions.append("This is a premium customer. Offer priority support.")
        instructions.append("Mention their dedicated account manager: {user.account_manager}")
    
    # Conditional: only for specific countries
    if user.country in ["DE", "FR", "ES"]:
        instructions.append("Include GDPR data processing notice.")
    
    # Conditional: only if the user seems frustrated
    if user.sentiment == "frustrated":
        instructions.append("Acknowledge their frustration before providing the answer.")
        instructions.append("Offer a goodwill discount code if appropriate.")
    
    return "\n".join(f"- {i}" for i in instructions)
```

### Pattern 2: Tiered Template Sections

```python
TEMPLATE_SECTIONS = {
    "base": """
You are a {role} for {company}.
Answer questions using the provided context.
""",
    "with_tools": """
You have access to the following tools:
{tool_descriptions}

Use them when needed. Always verify results before responding.
""",
    "with_rag": """
The following documents are relevant to this query:
{rag_context}

Base your answer on these documents. Cite sources.
""",
    "premium_experience": """
This is a premium customer. Provide white-glove service.
- Address them by name: {customer_name}
- Offer proactive suggestions
- Mention their dedicated support line: {premium_support_number}
""",
    "multi_turn": """
Previous conversation summary:
{conversation_summary}

The user may refer to previous topics. Maintain continuity.
"""
}

def assemble_prompt(sections_needed: list[str], variables: dict) -> str:
    """Assemble a prompt from the required sections only."""
    parts = []
    for section in sections_needed:
        if section in TEMPLATE_SECTIONS:
            parts.append(TEMPLATE_SECTIONS[section].format(**variables))
    return "\n".join(parts)

# Usage
sections = ["base", "with_rag"]
if user.plan == "premium":
    sections.append("premium_experience")
if is_multi_turn:
    sections.append("multi_turn")

prompt = assemble_prompt(sections, {
    "role": "support agent",
    "company": "Acme Corp",
    "rag_context": rag_results,
    "customer_name": user.name,
    "premium_support_number": "+1-555-PREMIUM",
    "conversation_summary": memory.get_summary()
})
```

---

## Multi-Source Context Injection

A production prompt pulls context from multiple sources. Each source needs its own formatting and priority.

```python
class PromptAssembler:
    """
    Assemble prompts from multiple context sources.
    Each source has a format template and priority.
    """
    
    def __init__(self, base_template: str):
        self.base_template = base_template
        self.sources = {}  # name -> ContextSource
    
    def register_source(self, name: str, formatter: callable, 
                        priority: int, max_tokens: int = None):
        """Register a context source with formatting rules."""
        self.sources[name] = ContextSource(
            name=name,
            formatter=formatter,
            priority=priority,
            max_tokens=max_tokens
        )
    
    def assemble(self, available_sources: dict) -> str:
        """
        Assemble the prompt with all available context.
        
        available_sources = {
            "rag_results": [...],
            "user_profile": {...},
            "tool_outputs": [...],
            "conversation_summary": "..."
        }
        """
        # Collect context from available sources
        context_sections = []
        tokens_used = 0
        
        # Sort sources by priority (highest first)
        active_sources = [
            self.sources[name] 
            for name in available_sources 
            if name in self.sources
        ]
        active_sources.sort(key=lambda s: s.priority, reverse=True)
        
        for source in active_sources:
            data = available_sources[source.name]
            formatted = source.formatter(data)
            context_tokens = count_tokens(formatted)
            
            # Apply max_tokens limit per source
            if source.max_tokens and context_tokens > source.max_tokens:
                formatted = truncate_tokens(formatted, source.max_tokens)
                context_tokens = source.max_tokens
            
            context_sections.append({
                "name": source.name,
                "content": formatted,
                "tokens": context_tokens
            })
            tokens_used += context_tokens
        
        # Render base template with assembled context
        context_block = "\n\n".join(
            f"## {s['name']}\n{s['content']}" 
            for s in context_sections
        )
        
        return self.base_template.format(
            context=context_block,
            context_sources=list(available_sources.keys()),
            context_token_count=tokens_used
        )
```

### Source Formatters

Each context source needs a formatter that shapes raw data into prompt-ready text:

```python
def format_rag_results(documents: list[dict]) -> str:
    """Format retrieved documents for prompt insertion."""
    parts = []
    for i, doc in enumerate(documents):
        source = doc["metadata"]["source"]
        score = doc["score"]
        parts.append(
            f"[Document {i+1}] Source: {source} (Relevance: {score:.0%})\n"
            f"{doc['text']}"
        )
    return "\n\n---\n\n".join(parts)

def format_user_profile(profile: dict) -> str:
    """Format user profile data for prompt insertion."""
    return f"""
Customer since: {profile.get('member_since', 'Unknown')}
Plan: {profile.get('plan', 'Free')}
Recent orders: {len(profile.get('recent_orders', []))}
Open tickets: {len(profile.get('open_tickets', []))}
Preferences: {profile.get('preferences', 'None specified')}
""".strip()

def format_tool_results(results: list[dict]) -> str:
    """Format tool execution results for prompt insertion."""
    parts = []
    for result in results:
        status = "✓" if result["success"] else "✗"
        parts.append(f"{status} {result['tool_name']}: {result['summary']}")
    return "\n".join(parts)

def format_conversation_summary(summary: str) -> str:
    """Format conversation summary for prompt insertion."""
    return f"Previous conversation: {summary}"
```

---

## The Complete Assembly Pipeline

```
┌─────────────────────────────────────────────────────┐
│               PROMPT ASSEMBLY PIPELINE              │
│                                                     │
│  1. SELECT TEMPLATE                                 │
│     Choose base template based on query type        │
│     (support, sales, technical, general)            │
│                                                     │
│  2. DETERMINE SECTIONS                              │
│     Based on: user tier, query complexity,          │
│     available data, conversation state              │
│                                                     │
│  3. GATHER CONTEXT                                  │
│     RAG results, user profile, tool outputs,        │
│     conversation history, business rules            │
│                                                     │
│  4. FORMAT & PRIORITIZE                             │
│     Apply source formatters, sort by priority,      │
│     enforce per-source token limits                 │
│                                                     │
│  5. ASSEMBLE                                        │
│     Fill template, inject context, apply            │
│     conditional sections                            │
│                                                     │
│  6. ENFORCE BUDGET                                  │
│     Run through ContextBudget, compress if needed   │
│                                                     │
│  7. OPTIMIZE STRUCTURE                              │
│     Apply attention optimization from Chapter 01    │
│                                                     │
│  8. RETURN                                          │
│     Final prompt, ready for LLM call                │
└─────────────────────────────────────────────────────┘
```

> **Step 7 — Attention optimization** means reordering sections so critical
> instructions land in the 20–60% "golden middle" of the context window, where
> attention scores are highest. See `ContextOptimizer.reorder_for_attention()`
> in [01-the-context-window-as-a-resource.md](01-the-context-window-as-a-resource.md).

> **Code Reference:** [Python](../../code/python/05-context-assembly/) · [Node.js](../../code/nodejs/05-context-assembly/) · [Go](../../code/go/05-context-assembly/)  
> The context assembly implementations include the full pipeline with template engine, multi-source injection, and budget enforcement.

---

## Prompt Version Control

Templates are code. Treat them like code.

### Store Templates Alongside Code

```
project/
├── src/
│   ├── agent.py
│   └── ...
├── prompts/
│   ├── support/
│   │   ├── base.yaml          # Base support template
│   │   ├── billing.yaml       # Billing-specific additions
│   │   └── premium.yaml       # Premium customer additions
│   ├── sales/
│   │   └── product_inquiry.yaml
│   └── templates.yaml         # Shared template fragments
└── tests/
    └── test_prompts.py        # Prompt tests
```

### YAML Template Format

```yaml
# prompts/support/base.yaml
name: support_base
version: 2.3.0
description: Base support agent template

template: |
  You are a customer support agent for {company_name}.
  
  Role: {role_description}
  
  Guidelines:
  {guidelines}
  
  Context:
  {context}
  
  Current customer: {customer_name} ({customer_plan} plan)

sections:
  billing:
    condition: "query_type == 'billing'"   # simple DSL: ==, !=, in, not_in, >, contains, AND/OR
    content: |
      - Show amounts in {currency}
      - Include payment due dates
      - Offer payment plan options for amounts over $100
  
  premium:
    condition: "user.plan in ['premium', 'enterprise']"
    content: |
      - Address customer by name
      - Offer proactive solutions
      - Mention dedicated support line: {premium_line}
  
  international:
    condition: "user.country not_in ['US', 'CA']"
    content: |
      - Include international shipping information
      - Mention customs and duties if applicable
```

Conditions use a safe DSL (no `eval`): `field == 'value'`, `field in [list]`,
`field not_in [list]`, `score > 0.7`, `token exists`, and compound `AND`/`OR`.
See `condition_engine.py` for the full grammar and `explain()` for debugging
"why wasn't this section included?"

### Loading and Version Tracking

```python
class PromptLibrary:
    """Version-controlled prompt template management."""
    
    def __init__(self, prompts_dir: str = "prompts/"):
        self.prompts_dir = prompts_dir
        self.templates = {}
        self.load_all()
    
    def load_all(self):
        """Load all YAML templates from the prompts directory."""
        for filepath in Path(self.prompts_dir).rglob("*.yaml"):
            template = yaml.safe_load(filepath.read_text())
            self.templates[template["name"]] = template
    
    def get_template(self, name: str, version: str = None) -> PromptTemplate:
        """Get a template by name. Log version for observability."""
        template = self.templates[name]
        logging.info(f"Loading prompt template: {name} v{template['version']}")
        return PromptTemplate(template)
    
    def render(self, name: str, variables: dict, 
               active_sections: list[str] = None) -> str:
        """Render a template with variables and conditional sections."""
        template = self.get_template(name)
        
        # Determine which sections to include
        if active_sections is None:
            active_sections = self._evaluate_conditions(template, variables)
        
        # Render base template
        rendered = template.render_base(variables)
        
        # Add active sections
        for section_name in active_sections:
            if section_name in template.sections:
                rendered += "\n" + template.sections[section_name].render(variables)
        
        return rendered
```

This approach gives you:
- **Git history** for every prompt change
- **Code review** for prompt modifications
- **Rollback** capability when a prompt change degrades performance
- **A/B testing** by deploying different template versions
- **Observability** by logging which template version generated each response

In the reference implementation, `render()` returns a `RenderedPrompt` dataclass
that carries version metadata alongside the text: `template_name`, `template_version`,
`sections_included`, `variables_used`, and `token_count`. Log or trace this object
instead of the raw string to make "which prompt fired" visible in your observability
stack.

---

## Common Pitfalls

- **"I have one massive template with 50 conditional sections"**: You've built a configuration nightmare. Split into domain-specific templates. A billing template and a shipping template are easier to maintain than one mega-template.
- **"My template variables are undefined at runtime"**: The template references `{customer_name}` but the variable isn't passed. Validate at load time using `PromptAssembler.get_available_variables()` or `PromptLibrary.validate_all()` — both report every `{placeholder}` so you can cross-check against your variable-provision logic before the first request hits.
- **"I version my code but not my prompts"**: Prompt changes cause production incidents more often than code changes. Prompts need the same version control, code review, and rollback capability as any other code.
- **"I include all context sources even when empty"**: An empty "Conversation History" section adds headers and formatting tokens with no value. Conditionally include sections only when they have content.
- **"My template is 5,000 tokens before any variables are filled"**: The template itself is consuming your context budget. Move verbose instructions to conditional sections that load only when needed.

## What's Next

You can now assemble prompts dynamically from multiple sources. Next: when the assembled context still doesn't fit — compression and filtering techniques that preserve signal while reducing noise.
→ [Context Compression and Filtering](03-context-compression-and-filtering.md)