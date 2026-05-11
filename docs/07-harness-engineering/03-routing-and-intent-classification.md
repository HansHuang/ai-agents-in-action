# Routing and Intent Classification

## What You'll Learn
- Why "one agent handles everything" fails at scale
- The router pattern: classifying requests and directing them to specialized handlers
- Deterministic vs. LLM-based classification: when to use each
- Building a multi-path agent system: chat, RAG, agent loop, and human escalation
- Intent hierarchies: from coarse-grained routing to fine-grained parameter extraction
- Measuring routing accuracy and handling misclassifications gracefully

## Prerequisites
- [Input Guardrails and Validation](02-input-guardrails-and-validation.md) — routing happens after input is validated
- [The Harness Mindset](01-the-harness-mindset.md) — routing is a deterministic harness decision
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — the agent loop is one possible route

---

## Why One Agent Isn't Enough

A single ReAct agent with all tools loaded can handle anything. In theory. In practice, it can't.

| Problem | Example |
|:---|:---|
| **Tool overload** | An agent with 50 tools gets confused about which to use |
| **Prompt bloat** | Instructions for handling billing, support, sales, and technical questions all in one system prompt |
| **Cost inefficiency** | A simple "hello" goes through the full agent loop with all tool definitions loaded |
| **Latency degradation** | Every request pays the overhead of tool definitions and complex system prompts |
| **Safety boundaries** | The agent that can look up orders shouldn't also have access to delete them |
| **Specialization** | A support agent needs empathy. A code agent needs technical precision. One prompt can't optimize for both |

The solution: **route requests to specialized handlers.**

```
User: "What's the weather in Tokyo?"
     │
     ▼
┌─────────────────┐
│    ROUTER       │
│  Classify intent │
└────────┬────────┘
         │
    ┌────┼────────┬──────────┐
    ▼    ▼        ▼          ▼
┌──────┐┌──────┐┌──────┐┌──────┐
│Simple││ RAG  ││Agent ││Human │
│ Chat ││ Path ││ Loop ││Escal.│
└──────┘└──────┘└──────┘└──────┘
  Fast    Docs    Tools    Complex
  Cheap   Search  Actions  Issues
```

---

## The Router Pattern

A router is a classifier that decides which handler should process a request.

### Deterministic vs. LLM-Based Routing

| | Deterministic | LLM-Based |
|:---|:---|:---|
| **How it works** | Keyword matching, regex, rules | LLM classifies the intent |
| **Speed** | <1ms | 200-800ms |
| **Cost** | Free | Small token cost per classification |
| **Accuracy** | High for clear patterns, low for ambiguous | High for nuanced requests |
| **Maintainability** | Rules proliferate, hard to update | Prompt can be updated easily |
| **Best for** | Clear-cut distinctions | Nuanced, context-dependent routing |

**The best routers use both.** Deterministic rules catch the obvious cases fast. LLM classification handles the ambiguous remainder.

---

## Building a Deterministic Router

Start with the cases you can catch without an LLM:

```python
class DeterministicRouter:
    """
    Route requests using fast, deterministic rules.
    Handles ~70-80% of cases instantly.
    """
    
    # Clear patterns for common intents
    PATTERNS = {
        "greeting": [
            r'^(hi|hello|hey|good morning|good evening|yo|sup)\b',
            r'^(how are you|how\'s it going|what\'s up)\b',
        ],
        "goodbye": [
            r'\b(bye|goodbye|see you|talk later|farewell)\b',
        ],
        "thanks": [
            r'\b(thanks|thank you|thx|appreciate it|grateful)\b',
        ],
        "reset": [
            r'\b(start over|reset|clear|new conversation|forget everything|fresh start)\b',
        ],
        "help": [
            r'\b(what can you do|help me|capabilities|features|how do (I|you))\b',
        ],
        "weather": [
            r'\b(weather|temperature|forecast|humidity|rain|sunny|cloudy|snow)\b',
        ],
        "stock": [
            r'\b(stock|market|price|ticker|nasdaq|dow|s&p|invest|share price)\b',
        ],
        "order_lookup": [
            r'\b(order|tracking|shipment|delivery|where is my|status of)\b.*\b(order|package|item|number)\b',
        ],
        "return_request": [
            r'\b(return|refund|exchange|money back|send back|cancel order)\b',
        ],
        "billing": [
            r'\b(bill|invoice|charge|payment|subscription|receipt|pricing|cost)\b',
        ],
        "technical_support": [
            r'\b(not working|broken|error|bug|crash|down|failed|issue|problem)\b',
        ],
        "account": [
            r'\b(account|login|password|profile|settings|email change|update.*info)\b',
        ],
    }
    
    def __init__(self, patterns: dict[str, list[str]] = None):
        self.patterns = patterns or self.PATTERNS
        self.compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.patterns.items()
        }
    
    def classify(self, user_input: str) -> RouteResult | None:
        """
        Try to classify the input deterministically.
        Returns None if no pattern matches (needs LLM classification).
        """
        matches = []
        
        for intent, patterns in self.compiled.items():
            for pattern in patterns:
                if pattern.search(user_input):
                    matches.append((intent, pattern))
        
        if not matches:
            return None  # Defer to LLM classifier
        
        # If multiple intents match, use the one with the longest pattern match
        # (longer patterns are more specific)
        best_intent, best_pattern = max(
            matches,
            key=lambda m: len(m[1].pattern)
        )
        
        return RouteResult(
            intent=best_intent,
            confidence=0.85 if len(matches) == 1 else 0.65,
            method="deterministic",
            matched_pattern=best_pattern.pattern,
        )
```

---

## Building an LLM-Based Router

For requests that don't match deterministic patterns, use a fast, cheap LLM:

```python
class LLMRouter:
    """
    Route ambiguous requests using an LLM classifier.
    Uses a fast, cheap model (GPT-4o-mini, Claude Haiku).
    """
    
    ROUTING_PROMPT = """You are a request classifier. Analyze the user's message
and determine its primary intent.

Available routes:
- simple_chat: Casual conversation, greetings, general questions not requiring tools
- knowledge_question: Questions answerable from our knowledge base (policies, docs, FAQs)
- agent_task: Requests requiring tool use (lookups, calculations, multi-step tasks)
- human_escalation: User explicitly asks for a human, or the request is too complex
- support_request: Customer support issues, complaints, problems with products
- out_of_scope: Requests we cannot or should not handle

Classification rules:
- If the user just says "hi" or makes small talk → simple_chat
- If the user asks about policies, procedures, documentation → knowledge_question
- If the user wants you to DO something (look up, calculate, book, create) → agent_task
- If the user is frustrated and demands a person → human_escalation
- If the user reports a problem with a product/service → support_request
- If the request is inappropriate or impossible → out_of_scope

Output ONLY a JSON object:
{
    "intent": "<intent_name>",
    "confidence": 0.0-1.0,
    "reasoning": "<one sentence explanation>",
    "extracted_params": {
        "order_number": null,
        "product_name": null,
        "issue_type": null
    }
}
"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
    
    async def classify(self, user_input: str,
                      conversation_history: list[dict] = None) -> RouteResult:
        """Classify using LLM."""
        
        messages = [
            {"role": "system", "content": self.ROUTING_PROMPT},
        ]
        
        # Include recent conversation for context
        if conversation_history:
            recent = conversation_history[-4:]  # Last 4 messages
            summary = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {str(m.get('content', ''))[:200]}"
                for m in recent
            )
            messages.append({
                "role": "user",
                "content": f"Recent conversation:\n{summary}\n\nClassify this message: {user_input}"
            })
        else:
            messages.append({"role": "user", "content": user_input})
        
        response = await llm.chat(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent classification
        )
        
        result = json.loads(response.content)
        
        return RouteResult(
            intent=result["intent"],
            confidence=result["confidence"],
            method="llm",
            reasoning=result.get("reasoning"),
            extracted_params=result.get("extracted_params"),
        )
```

---

## The Hybrid Router: Best of Both

Combine deterministic and LLM routing:

```python
class HybridRouter:
    """
    Two-stage router: deterministic first, LLM for ambiguous cases.
    """
    
    def __init__(self):
        self.deterministic = DeterministicRouter()
        self.llm = LLMRouter()
        self.metrics = RouterMetrics()
    
    async def route(self, user_input: str,
                   conversation_history: list[dict] = None) -> RouteResult:
        """Route a request. Deterministic first, LLM fallback."""
        
        # Stage 1: Try deterministic
        result = self.deterministic.classify(user_input)
        
        if result and result.confidence > 0.8:
            self.metrics.record("deterministic", result.intent)
            return result
        
        # Stage 2: LLM for ambiguous or low-confidence cases
        llm_result = await self.llm.classify(
            user_input, conversation_history
        )
        
        # Use deterministic result if LLM confidence is lower
        if result and result.confidence > llm_result.confidence:
            self.metrics.record("deterministic_fallback", result.intent)
            return result
        
        self.metrics.record("llm", llm_result.intent)
        return llm_result
    
    def get_metrics(self) -> dict:
        """Get routing statistics."""
        return self.metrics.summary()

class RouterMetrics:
    """Track routing performance."""
    
    def __init__(self):
        self.total = 0
        self.by_method = defaultdict(int)
        self.by_intent = defaultdict(int)
        self.latencies = []
    
    def record(self, method: str, intent: str, latency_ms: float = 0):
        self.total += 1
        self.by_method[method] += 1
        self.by_intent[intent] += 1
        self.latencies.append(latency_ms)
    
    def summary(self) -> dict:
        return {
            "total_routed": self.total,
            "deterministic_rate": self.by_method.get("deterministic", 0) / max(self.total, 1),
            "llm_fallback_rate": self.by_method.get("llm", 0) / max(self.total, 1),
            "intent_distribution": dict(self.by_intent),
            "avg_latency_ms": sum(self.latencies) / max(len(self.latencies), 1),
        }
```

---

## Mapping Intents to Handlers

Once classified, each intent maps to a handler:

```python
class HandlerRegistry:
    """
    Map intents to handler functions.
    Each handler has different characteristics.
    """
    
    def __init__(self):
        self.handlers: dict[str, RouteHandler] = {}
    
    def register(self, intent: str, handler: callable, 
                config: HandlerConfig) -> None:
        self.handlers[intent] = RouteHandler(
            handler=handler,
            config=config,
        )
    
    def get_handler(self, intent: str) -> "RouteHandler":
        """Get handler for intent. Falls back to default if intent unknown."""
        if intent in self.handlers:
            return self.handlers[intent]
        
        # Unknown intent → use simple chat as safe default
        logger.warning(f"Unknown intent '{intent}', falling back to simple_chat")
        return self.handlers.get("simple_chat")

@dataclass
class HandlerConfig:
    """Configuration for a route handler."""
    model: str = "gpt-4o-mini"      # Which model to use
    max_tokens: int = 1024           # Response limit
    temperature: float = 0.7         # Creativity level
    timeout_seconds: int = 30        # Max execution time
    requires_tools: bool = False     # Load tool definitions?
    requires_rag: bool = False       # Perform retrieval?
    requires_approval: bool = False  # Human approval needed?
    cost_budget: float = 0.01        # Max cost per request

class RouteHandler:
    """A handler with its configuration."""
    handler: callable
    config: HandlerConfig

# Example handler registry
registry = HandlerRegistry()

registry.register(
    intent="simple_chat",
    handler=simple_chat_handler,
    config=HandlerConfig(
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.7,
        timeout_seconds=15,
        cost_budget=0.001,
    )
)

registry.register(
    intent="knowledge_question",
    handler=rag_handler,
    config=HandlerConfig(
        model="gpt-4o",
        max_tokens=2048,
        temperature=0.3,
        timeout_seconds=45,
        requires_rag=True,
        cost_budget=0.05,
    )
)

registry.register(
    intent="agent_task",
    handler=agent_loop_handler,
    config=HandlerConfig(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0.2,
        timeout_seconds=120,
        requires_tools=True,
        requires_approval=True,  # For high-stakes actions
        cost_budget=0.25,
    )
)

registry.register(
    intent="human_escalation",
    handler=escalation_handler,
    config=HandlerConfig(
        timeout_seconds=300,
        cost_budget=0.0,  # No LLM cost, just routing
    )
)

registry.register(
    intent="out_of_scope",
    handler=out_of_scope_handler,
    config=HandlerConfig(
        model="gpt-4o-mini",
        max_tokens=256,
        temperature=0.5,
        timeout_seconds=10,
        cost_budget=0.001,
    )
)
```

---

## Handler Implementations

### Simple Chat Handler

```python
async def simple_chat_handler(user_input: str, 
                              conversation_history: list[dict],
                              config: HandlerConfig) -> HandlerResponse:
    """Handle casual conversation. Fast, cheap, no tools."""
    
    messages = [
        {"role": "system", "content": "You are a friendly assistant. Keep responses brief."},
        *conversation_history[-6:],  # Only last 6 messages for chat
        {"role": "user", "content": user_input},
    ]
    
    response = await llm.chat(
        messages=messages,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )
    
    return HandlerResponse(
        content=response.content,
        handler_used="simple_chat",
        tokens_used=response.token_usage["total_tokens"],
        cost=calculate_cost(response.token_usage, config.model),
    )
```

### RAG Handler

```python
async def rag_handler(user_input: str,
                     conversation_history: list[dict],
                     config: HandlerConfig) -> HandlerResponse:
    """Handle knowledge questions with retrieval-augmented generation."""
    
    # Retrieve relevant documents
    query_embedding = await embedder.embed(user_input)
    documents = await vector_db.search(query_embedding, k=5, threshold=0.7)
    
    if not documents:
        # No relevant documents found — could re-route to agent
        return HandlerResponse(
            content="I couldn't find information about that in my knowledge base. "
                   "Let me try a different approach...",
            handler_used="rag",
            metadata={"documents_found": 0},
        )
    
    # Build RAG prompt
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.text}"
        for doc in documents
    )
    
    messages = [
        {
            "role": "system",
            "content": f"""Answer using only the provided documents.
If the answer is not in the documents, say so.

Documents:
{context}"""
        },
        {"role": "user", "content": user_input},
    ]
    
    response = await llm.chat(
        messages=messages,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )
    
    return HandlerResponse(
        content=response.content,
        handler_used="rag",
        tokens_used=response.token_usage["total_tokens"],
        cost=calculate_cost(response.token_usage, config.model),
        metadata={
            "documents_found": len(documents),
            "sources": [doc.metadata["source"] for doc in documents],
            "similarity_scores": [doc.score for doc in documents],
        },
    )
```

### Agent Loop Handler

```python
async def agent_loop_handler(user_input: str,
                            conversation_history: list[dict],
                            config: HandlerConfig) -> HandlerResponse:
    """Handle complex tasks with the full agent loop."""
    
    agent = Agent(
        llm_provider=LLMFactory.create(config.model),
        tools=load_tools_for_intent("agent_task"),
        max_iterations=10,
        timeout_seconds=config.timeout_seconds,
    )
    
    # Check if any tool calls require approval
    if config.requires_approval:
        agent.on_tool_call = approval_check
    
    result = await agent.run(user_input)
    
    return HandlerResponse(
        content=result.answer,
        handler_used="agent_loop",
        tokens_used=result.total_tokens,
        cost=result.total_cost,
        metadata={
            "iterations": result.iterations,
            "tools_called": result.tool_calls_made,
            "required_approval": result.approvals_required,
        },
    )
```

---

## Re-Routing on Failure

A handler might fail or produce a low-confidence result. The router should be able to escalate:

```python
class EscalatingRouter:
    """
    Router that can re-route requests when handlers fail.
    """
    
    ESCALATION_PATHS = {
        "simple_chat": ["knowledge_question"],  # If chat can't answer, try RAG
        "knowledge_question": ["agent_task"],     # If docs don't help, try agent
        "agent_task": ["human_escalation"],       # If agent can't solve, escalate
    }
    
    def __init__(self, router: HybridRouter, registry: HandlerRegistry):
        self.router = router
        self.registry = registry
    
    async def handle(self, user_input: str,
                    conversation_history: list[dict] = None) -> HandlerResponse:
        """Handle a request with automatic escalation on failure."""
        
        intent = await self.router.route(user_input, conversation_history)
        handler = self.registry.get_handler(intent.intent)
        
        try:
            response = await handler.handler(
                user_input, conversation_history, handler.config
            )
            
            # Check if response should trigger escalation
            if self._should_escalate(response, intent):
                return await self._escalate(
                    intent.intent, user_input, conversation_history, response
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Handler '{intent.intent}' failed: {e}")
            return await self._escalate(
                intent.intent, user_input, conversation_history, None, str(e)
            )
    
    def _should_escalate(self, response: HandlerResponse, 
                        intent: RouteResult) -> bool:
        """Check if response quality warrants escalation."""
        # No documents found in RAG
        if response.metadata and response.metadata.get("documents_found") == 0:
            return True
        
        # Agent couldn't complete
        if response.metadata and response.metadata.get("iterations", 0) >= 10:
            return True
        
        # Response contains uncertainty markers
        uncertainty_phrases = [
            "I'm not sure", "I don't know", "I cannot",
            "I'm unable", "I don't have enough information"
        ]
        if any(phrase in response.content for phrase in uncertainty_phrases):
            return True
        
        return False
    
    async def _escalate(self, original_intent: str,
                       user_input: str,
                       conversation_history: list[dict],
                       previous_response: HandlerResponse = None,
                       error: str = None) -> HandlerResponse:
        """Escalate to the next handler in the chain."""
        
        escalation_path = self.ESCALATION_PATHS.get(original_intent, ["human_escalation"])
        
        for next_intent in escalation_path:
            handler = self.registry.get_handler(next_intent)
            
            # Add context about why we're re-routing
            augmented_input = user_input
            if previous_response:
                augmented_input = (
                    f"[Previous attempt using {original_intent} did not fully "
                    f"resolve this. Response was: '{previous_response.content[:200]}...']"
                    f"\n\nOriginal request: {user_input}"
                )
            
            try:
                response = await handler.handler(
                    augmented_input, conversation_history, handler.config
                )
                
                # Add escalation metadata
                response.metadata["escalated_from"] = original_intent
                response.metadata["escalation_reason"] = error or "low_confidence"
                
                return response
            except Exception as e:
                logger.error(f"Escalation to '{next_intent}' also failed: {e}")
                continue
        
        # All escalations failed
        return HandlerResponse(
            content="I apologize, but I'm having trouble processing your request. "
                   "A human team member will follow up with you shortly.",
            handler_used="escalation_fallback",
            metadata={"escalation_chain_exhausted": True},
        )
```

---

## Intent Hierarchies

Simple routing works for distinct categories. For more complex systems, use intent hierarchies:

```python
class HierarchicalRouter:
    """
    Multi-level routing: coarse intent → fine intent → parameter extraction.
    """
    
    # Level 1: Coarse intent (which major category?)
    COARSE_INTENTS = {
        "conversation": ["simple_chat", "greeting", "goodbye", "thanks"],
        "information": ["knowledge_question", "fact_lookup", "comparison"],
        "action": ["agent_task", "purchase", "booking", "modification"],
        "support": ["support_request", "complaint", "bug_report", "human_escalation"],
    }
    
    # Level 2: Fine intent (what specifically?)
    FINE_INTENTS = {
        "agent_task": {
            "description": "Multi-step task requiring tools",
            "sub_intents": ["order_lookup", "return_request", "account_update", "data_export"],
        },
        "support_request": {
            "description": "Customer needs help with an issue",
            "sub_intents": ["product_defect", "shipping_issue", "billing_dispute", "account_access"],
        },
    }
    
    async def classify_detailed(self, user_input: str) -> DetailedRoute:
        """
        Classify with increasing specificity:
        Level 1: conversation, information, action, support
        Level 2: specific intent within category
        Level 3: extracted parameters (order number, product, issue type)
        """
        
        # Level 1: Coarse classification (deterministic first)
        coarse = self.deterministic.classify_coarse(user_input)
        if not coarse or coarse.confidence < 0.7:
            coarse = await self.llm.classify_coarse(user_input)
        
        # Level 2: Fine classification (LLM)
        fine = await self.llm.classify_fine(
            user_input, 
            coarse_category=coarse.intent
        )
        
        # Level 3: Parameter extraction (LLM)
        params = await self.llm.extract_parameters(
            user_input,
            intent=fine.intent
        )
        
        return DetailedRoute(
            coarse_intent=coarse.intent,
            fine_intent=fine.intent,
            extracted_params=params,
            confidence=fine.confidence,
        )
```

---

## Measuring Routing Accuracy

You can't improve routing without measuring it:

```python
class RoutingEvaluator:
    """
    Evaluate routing accuracy with labeled test data.
    """
    
    def __init__(self, router: HybridRouter):
        self.router = router
    
    async def evaluate(self, 
                      test_cases: list[RoutingTestCase]) -> RoutingReport:
        """Evaluate router against labeled test cases."""
        
        results = []
        
        for test in test_cases:
            result = await self.router.route(test.user_input)
            
            is_correct = result.intent == test.expected_intent
            results.append(EvaluationResult(
                input=test.user_input,
                expected=test.expected_intent,
                predicted=result.intent,
                correct=is_correct,
                method=result.method,
                confidence=result.confidence,
            ))
        
        return self._generate_report(results)
    
    def _generate_report(self, results: list) -> RoutingReport:
        """Generate accuracy report with per-intent breakdown."""
        
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        
        # Per-intent accuracy
        by_intent = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            by_intent[r.expected]["total"] += 1
            if r.correct:
                by_intent[r.expected]["correct"] += 1
        
        # Most common misclassifications
        misclassifications = defaultdict(int)
        for r in results:
            if not r.correct:
                misclassifications[f"{r.expected} → {r.predicted}"] += 1
        
        return RoutingReport(
            overall_accuracy=correct / total,
            total_cases=total,
            by_intent={
                intent: stats["correct"] / max(stats["total"], 1)
                for intent, stats in by_intent.items()
            },
            top_misclassifications=sorted(
                misclassifications.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            deterministic_rate=sum(1 for r in results if r.method == "deterministic") / total,
            avg_confidence_correct=sum(r.confidence for r in results if r.correct) / max(correct, 1),
            avg_confidence_incorrect=sum(r.confidence for r in results if not r.correct) / max(total - correct, 1),
        )

@dataclass
class RoutingTestCase:
    user_input: str
    expected_intent: str
    description: str = ""

@dataclass
class RoutingReport:
    overall_accuracy: float
    total_cases: int
    by_intent: dict[str, float]
    top_misclassifications: list[tuple[str, int]]
    deterministic_rate: float
    avg_confidence_correct: float
    avg_confidence_incorrect: float
```

---

## The Complete Request Flow

With routing integrated into the harness:

```python
class HarnessWithRouting:
    """
    Complete harness: input guardrails → routing → handler → output guardrails.
    """
    
    def __init__(self):
        self.input_guardrails = InputGuardrailPipeline()
        self.router = HybridRouter()
        self.handlers = HandlerRegistry()
        self.escalating_router = EscalatingRouter(self.router, self.handlers)
        self.output_guardrails = OutputGuardrailPipeline()
    
    async def process(self, user_input: str,
                     user_id: str,
                     conversation_history: list[dict] = None) -> FinalResponse:
        """Process a user request through the complete harness."""
        
        # Phase 1: Input guardrails
        guardrail_result = await self.input_guardrails.process(
            user_input, user_id, conversation_history
        )
        if not guardrail_result.passed:
            return FinalResponse(
                content=guardrail_result.rejection_reason,
                status="rejected",
                rejection_layer=guardrail_result.rejection_layer,
            )
        
        cleaned_input = guardrail_result.cleaned_input
        
        # Phase 2: Route
        route_result = await self.router.route(
            cleaned_input, conversation_history
        )
        
        # Phase 3: Handle (with escalation)
        handler_response = await self.escalating_router.handle(
            cleaned_input, conversation_history
        )
        
        # Phase 4: Output guardrails
        output_guard_result = await self.output_guardrails.validate(
            handler_response.content
        )
        if not output_guard_result.passed:
            return FinalResponse(
                content="I'm unable to provide that response. Please rephrase your request.",
                status="blocked",
                rejection_layer="output_guardrails",
            )
        
        return FinalResponse(
            content=handler_response.content,
            status="success",
            route=route_result.intent,
            handler=handler_response.handler_used,
            metadata=handler_response.metadata,
        )
```

> **Code Reference:** [Python](../../code/python/07-harness/) · [Node.js](../../code/nodejs/07-harness/) · [Go](../../code/go/07-harness/)  
> The harness implementations include the complete routing system with deterministic, LLM, and hybrid routers.

---

## Common Pitfalls

- **"I use LLM routing for every request"**: A simple "hello" doesn't need an API call to classify. Deterministic rules handle 70-80% of cases instantly and free.
- **"My router has no fallback for unknown intents"**: Every router needs a default handler. "Simple chat" is a safe default — it can always respond, even if it can't solve the problem.
- **"I don't re-route when a handler fails"**: A handler might not find the answer. That's not a failure — it's a signal to try the next handler. Build escalation paths.
- **"My intent categories overlap"**: If "order_lookup" and "return_request" both match "Where's my order?", your categories are too fuzzy. Make them mutually exclusive or use a hierarchy.
- **"I don't measure routing accuracy"**: A router making wrong decisions silently sends users to the wrong handler. Build a test set and measure accuracy. Track misclassifications over time.
- **"My routing prompt is too verbose"**: The routing LLM call should be fast. Use a cheap model. Keep the prompt tight. Remember: you're paying for this on every ambiguous request.

## What's Next

Requests are now routed to the right handler. Next: making those handlers resilient — retry logic, exponential backoff, and circuit breakers for every external call.
→ [Retry, Fallback, and Circuit Breakers](04-retry-fallback-and-circuit-breakers.md)