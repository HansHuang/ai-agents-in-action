# The Harness Mindset

## What You'll Learn
- Why LLMs are components, not applications — and what that means for engineering
- The harness as a deterministic control system around a probabilistic core
- The five principles of harness engineering: distrust, timeout, fallback, validate, observe
- How harness engineering parallels site reliability engineering (SRE)
- The harness as a state machine: every LLM interaction is a state transition
- Why most agent failures are harness failures, not model failures

## Prerequisites
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — the agent loop the harness wraps around
- [Agent Observability](../05-the-tool-ecosystem/03-agent-observability.md) — the harness is observable by design
- Everything before this chapter — harness engineering ties together every concept in this repo

---

## The Central Insight

Every chapter before this one taught you how to make agents work. This chapter teaches you how to make agents work **reliably**.

The difference is the harness.

> A system that works 95% of the time is not a 95%-reliable system — it is an unreliable system with a 5% defect rate. The harness is what turns that 5% into a handled state rather than an unhandled failure.

```
┌─────────────────────────────────────────────────────────┐
│                     HARNESS                             │
│  (deterministic control system)                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │                 INPUT GUARDRAILS                    │ │
│  │  - Validate inputs before they reach the agent      │ │
│  │  - Reject malicious, malformed, or out-of-scope    │ │
│  │  - Normalize and sanitize                          │ │
│  └──────────────────────┬─────────────────────────────┘ │
│                         ▼                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │                    ROUTER                          │ │
│  │  - Classify the request                            │ │
│  │  - Route to appropriate handler/path               │ │
│  │  - Simple chat? Direct LLM. Complex? Agent loop.   │ │
│  └──────────────────────┬─────────────────────────────┘ │
│                         ▼                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │                                                      │ │
│  │              ┌──────────────┐                        │ │
│  │              │    AGENT     │                        │ │
│  │              │ (probabilistic│                       │ │
│  │              │    core)     │                        │ │
│  │              └──────────────┘                        │ │
│  │                                                      │ │
│  └──────────────────────┬─────────────────────────────┘ │
│                         ▼                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │                 OUTPUT GUARDRAILS                   │ │
│  │  - Validate the agent's response                   │ │
│  │  - Check for hallucinations, PII leaks, toxicity   │ │
│  │  - Enforce format and schema                       │ │
│  └──────────────────────┬─────────────────────────────┘ │
│                         ▼                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │              HUMAN-IN-THE-LOOP                     │ │
│  │  - Conditional approval for high-stakes actions     │ │
│  │  - "The agent wants to send this email. Approve?"  │ │
│  └──────────────────────┬─────────────────────────────┘ │
│                         ▼                                │
│                    Response                              │
└─────────────────────────────────────────────────────────┘
```

The agent is the probabilistic core. The harness is everything around it — and the harness is entirely deterministic.

---

## LLMs Are Components, Not Applications

This is the foundational principle of harness engineering:

> **An LLM is more like a database than a backend.**

You don't trust a database query. You:
- Validate the input (SQL injection prevention)
- Set a timeout (slow query detection)
- Check the result (data integrity)
- Have a fallback (read replica)
- Log everything (audit trail)

You should do the same with LLMs. An LLM is a component with:
- A defined interface (messages in, text out)
- Known failure modes (hallucination, timeout, malformed output)
- Performance characteristics (latency percentiles, token limits)
- A cost profile (per-token pricing)

Treating an LLM as an application — trusting its output, assuming it will always respond, not planning for failure — is the root cause of most production AI incidents.

**This mental model is the unlock.** Once you treat the LLM as a component, every reliability pattern from your existing infrastructure knowledge applies directly.

---

## The Five Principles of Harness Engineering

### Principle 1: Never Trust the Model's Output Without Validation

The model is a token predictor. It doesn't "know" things. It predicts plausible sequences of tokens. Sometimes those sequences are factual. Sometimes they're not.

```python
# DON'T: Trust the model's output
def untrusted_agent(user_input):
    response = llm.chat(messages)
    send_email(response.content)  # What if the model hallucinated the recipient?
    update_database(response.tool_calls)  # What if it called the wrong function?

# DO: Validate everything
def harnessed_agent(user_input):
    # Input guard
    if not is_valid_input(user_input):
        return "Invalid request."
    
    # Agent execution with timeout
    try:
        response = agent.run(user_input, timeout=30)
    except TimeoutError:
        return "Request timed out. Please try again."
    
    # Output guard
    if not is_safe_output(response):
        log_safety_violation(response)
        return "I cannot provide that response."
    
    # Schema validation for tool calls
    if response.tool_calls:
        for call in response.tool_calls:
            if not validate_tool_call(call):
                log_invalid_tool_call(call)
                return "An error occurred processing your request."
    
    return response
```

### Principle 2: Every LLM Call Needs a Timeout

LLMs are slow. Sometimes they're very slow. Sometimes they hang indefinitely. Every call needs a deadline.

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds: int):
    """Context manager that raises TimeoutError after `seconds`."""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Usage
try:
    with timeout(30):
        response = llm.chat(messages)
except TimeoutError:
    response = fallback_response(messages)
    log_timeout(messages)
```

For async code, use `asyncio.wait_for`:

```python
try:
    response = await asyncio.wait_for(
        llm.chat_async(messages),
        timeout=30.0
    )
except asyncio.TimeoutError:
    response = await fallback_llm.chat_async(messages)
```

Timeouts should be set at multiple levels:

| Level | Scope | Recommended range |
|:---|:---|:---|
| LLM call | Single API request | 30–60 s |
| Agent step | One iteration of the agent loop | 10–20 s |
| Tool execution | Single tool call | 5–30 s |
| Total request | End-to-end user request | 2–5 min |

Each level is independent. A 60-second LLM timeout inside a 5-minute total timeout means the agent can still retry after a slow call without breaking the user's experience.

### Principle 3: Every LLM Call Needs a Fallback Path

APIs fail. Rate limits get hit. Models return errors. The harness must have an answer for every failure mode.

> The failure mode you don't plan for is the one that pages you at 3 AM.

```python
class FallbackChain:
    """Try providers in order until one succeeds."""
    
    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
    
    def chat(self, messages, **kwargs) -> LLMResponse:
        errors = []
        
        for provider in self.providers:
            try:
                return provider.chat(messages, **kwargs)
            except RateLimitError:
                errors.append(f"{provider.name}: rate limited")
                continue
            except TimeoutError:
                errors.append(f"{provider.name}: timeout")
                continue
            except Exception as e:
                errors.append(f"{provider.name}: {e}")
                continue
        
        # All providers failed
        raise AllProvidersFailedError(errors)

# Configure your fallback chain
primary = OpenAIProvider(model="gpt-4o")
secondary = AnthropicProvider(model="claude-3-5-sonnet")
tertiary = OpenAIProvider(model="gpt-4o-mini")  # Cheaper, always available

llm = FallbackChain([primary, secondary, tertiary])
```

Fallback paths should degrade gracefully:

| Level | Provider | When activated |
|:---|:---|:---|
| Primary | Best model (e.g. gpt-4o) | Normal operation |
| Secondary | Equivalent, different provider (e.g. claude-3-5-sonnet) | Provider outage |
| Tertiary | Cheaper, same provider (e.g. gpt-4o-mini) | Rate limit on primary |
| Static | Pre-computed or cached response | Complete LLM outage |
| Error | Clear message to user | Nothing worked |

Do not use the same provider for primary and secondary — that defeats the purpose. Provider diversity is the whole point.

### Principle 4: The Harness Is Deterministic; the Model Is Probabilistic

Everything in the harness should be predictable, testable, and auditable:

```python
# The harness: deterministic logic
def validate_email_output(response: str) -> bool:
    """Check if the response contains an email. Deterministic regex."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(email_pattern, response)
    
    # Check against allowlist
    for email in matches:
        if email not in ALLOWED_EMAIL_DOMAINS:
            return False
    
    return True

# The harness: deterministic routing
def route_request(user_input: str) -> str:
    """Route to the correct handler. Deterministic classification."""
    if any(phrase in user_input.lower() for phrase in RESET_PHRASES):
        return "reset"
    if any(phrase in user_input.lower() for phrase in BILLING_PHRASES):
        return "billing"
    if len(user_input.split()) < 3:
        return "simple_chat"
    return "agent"

# The model: probabilistic reasoning
# This is the only part that's non-deterministic
response = llm.chat(messages)
```

The harness should never call an LLM to make a decision that can be made deterministically. Classification, routing, validation, and safety checks should be deterministic wherever possible.

### Principle 5: Observability Is Not Optional

Every decision the harness makes must be logged. Every validation, every timeout, every fallback, every rejection.

```python
class HarnessLogger:
    """Structured logging for harness operations."""
    
    def log_input_validation(self, user_input: str, 
                            result: str, reason: str = None) -> None:
        """Log input guardrail decisions."""
        logger.info(json.dumps({
            "event": "input_validation",
            "result": result,  # "passed", "rejected", "sanitized"
            "reason": reason,
            "input_length": len(user_input),
            "timestamp": time.time()
        }))
    
    def log_route_decision(self, user_input: str, 
                          route: str, confidence: float) -> None:
        """Log routing decisions."""
        logger.info(json.dumps({
            "event": "route_decision",
            "route": route,
            "confidence": confidence,
            "input_preview": user_input[:100],
            "timestamp": time.time()
        }))
    
    def log_timeout(self, operation: str, timeout_s: int,
                   messages_count: int) -> None:
        """Log timeout events."""
        logger.warning(json.dumps({
            "event": "timeout",
            "operation": operation,
            "timeout_seconds": timeout_s,
            "messages_count": messages_count,
            "timestamp": time.time()
        }))
    
    def log_fallback(self, from_provider: str, to_provider: str,
                    reason: str) -> None:
        """Log fallback activations."""
        logger.warning(json.dumps({
            "event": "fallback",
            "from_provider": from_provider,
            "to_provider": to_provider,
            "reason": reason,
            "timestamp": time.time()
        }))
    
    def log_output_validation(self, response: str,
                             result: str, violations: list[str] = None) -> None:
        """Log output guardrail decisions."""
        logger.info(json.dumps({
            "event": "output_validation",
            "result": result,  # "passed", "blocked", "redacted"
            "violations": violations,
            "response_length": len(response),
            "timestamp": time.time()
        }))
    
    def log_human_approval(self, action: str, approved: bool,
                          approver: str, reason: str = None) -> None:
        """Log human-in-the-loop decisions."""
        logger.info(json.dumps({
            "event": "human_approval",
            "action": action,
            "approved": approved,
            "approver": approver,
            "reason": reason,
            "timestamp": time.time()
        }))
```

---

## The Harness as a State Machine

A harness can be modeled as a finite state machine. Every LLM interaction is a state transition with defined failure modes:

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ VALIDATE │──────────┐ (invalid)
                    │  INPUT   │          │
                    └────┬─────┘     ┌────▼─────┐
                         │ (valid)   │  REJECT  │──► END
                    ┌────▼─────┐     └──────────┘
                    │  ROUTE   │
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │ SIMPLE │ │ AGENT  │ │  RAG   │
         │  CHAT  │ │  LOOP  │ │  PATH  │
         └───┬────┘ └───┬────┘ └───┬────┘
             │          │          │
             └──────────┼──────────┘
                        │
                   ┌────▼─────┐
                   │ VALIDATE │──────────┐ (invalid)
                   │  OUTPUT  │          │
                   └────┬─────┘     ┌────▼─────┐
                        │ (valid)   │  REWRITE │──► VALIDATE OUTPUT
                   ┌────▼─────┐     │  RETRY   │
                   │  HUMAN   │     └──────────┘
                   │ APPROVAL │
                   └────┬─────┘
                        │
                   ┌────▼─────┐
                   │ RESPOND  │
                   └──────────┘
```

Each state has:
- **Happy path**: Normal transition to the next state
- **Error path**: Transition to error handling
- **Timeout path**: Transition to timeout handling
- **Retry path**: Transition back to the same state (with count)

```python
class HarnessStateMachine:
    """A harness implemented as an explicit state machine."""
    
    def __init__(self):
        self.state = "start"
        self.retry_counts = {}
        self.max_retries = {"llm_call": 3, "tool_execution": 2}
    
    async def process(self, user_input: str) -> dict:
        self.state = "validate_input"
        
        while self.state != "end":
            if self.state == "validate_input":
                user_input = await self._validate_input(user_input)
            
            elif self.state == "route":
                handler = await self._route(user_input)
            
            elif self.state == "execute":
                response = await self._execute(handler, user_input)
            
            elif self.state == "validate_output":
                response = await self._validate_output(response)
            
            elif self.state == "human_approval":
                response = await self._human_approval(response)
            
            elif self.state == "respond":
                return response
            
            elif self.state == "reject":
                return {"error": "Request rejected", "reason": self.reject_reason}
            
            elif self.state == "timeout":
                return {"error": "Request timed out"}
        
        return response
    
    def _transition(self, new_state: str, reason: str = None) -> None:
        logger.info(f"State transition: {self.state} → {new_state}" + 
                   (f" ({reason})" if reason else ""))
        self.state = new_state
```

---

## Most Agent Failures Are Harness Failures

When an agent behaves badly in production, it is rarely because the model was "wrong." The model did exactly what a token predictor does: predict a plausible next token. It is the harness that failed to constrain the outcome.

This framing is important for incident response. Asking "why did the model do that?" is usually unproductive. Asking "what harness check should have caught this?" leads to actionable fixes.

| Symptom | Model Failure? | Harness Failure? |
|:---|:---|:---|
| Agent calls wrong tool with made-up parameters | Model hallucinated the tool call | No schema validation on tool calls |
| Agent returns offensive content | Model generated toxic text | No output content filter |
| User gets a timeout error | Model was slow | No timeout set, no fallback provider |
| Agent sends email to wrong person | Model hallucinated email address | No allowlist validation on recipients |
| Agent loops forever calling the same tool | Model got stuck in a loop | No max iterations, no loop detection |
| Agent exposes system prompt | Model regurgitated instructions | No output scanning for prompt leakage |
| Costs spike unexpectedly | Model used too many tokens | No per-request token budget |
| Agent approves a risky action | Model made a bad decision | No human-in-the-loop for high-stakes actions |

The model will fail. That's not a bug — it's the nature of probabilistic systems. The harness exists to catch those failures before they reach the user.

---

## The SRE Parallel

Harness engineering is to AI agents what Site Reliability Engineering (SRE) is to distributed systems.

| SRE Concept | Harness Equivalent |
|:---|:---|
| **Service Level Objective (SLO)** | Accuracy target, latency target, cost target |
| **Error budget** | Acceptable failure rate before paging |
| **Circuit breaker** | Stop calling a failing model, fall back |
| **Retry with backoff** | Exponential backoff on rate limits |
| **Canary deployment** | Test new prompts on 1% of traffic |
| **Incident response** | Runbook for agent misbehavior |
| **Blameless postmortem** | Analyze agent failures to improve harness |
| **Monitoring and alerting** | Dashboards for cost, latency, error rate |

If your team has SRE expertise, harness engineering will feel familiar. It's the same principles applied to a new type of component.

---

## The Harness Engineering Checklist

Before deploying an agent to production, verify every item:

### Input Guardrails
- [ ] Input length limits enforced
- [ ] PII detection and redaction on input
- [ ] Prompt injection detection
- [ ] Content policy enforcement
- [ ] Rate limiting per user

### Execution Guardrails
- [ ] Timeout on every LLM call
- [ ] Timeout on every tool execution
- [ ] Max iterations on agent loop
- [ ] Fallback provider configured
- [ ] Token budget per request
- [ ] Cost budget per user per day

### Output Guardrails
- [ ] Schema validation on tool calls
- [ ] Content safety filtering
- [ ] PII detection on output
- [ ] Hallucination detection (basic)
- [ ] Response length limits

### Operational Guardrails
- [ ] Structured logging for every harness decision
- [ ] Metrics: latency, tokens, cost, error rate
- [ ] Alerting on anomaly detection
- [ ] Human-in-the-loop for high-stakes actions
- [ ] Kill switch for emergency shutdown

---

## Common Pitfalls

- **"I trust the model to validate its own output"**: The model cannot validate itself — it is the thing being validated. Use deterministic regex, schema checks, and allowlists for safety-critical validation. Reserve LLM-as-judge for subjective quality checks only.
- **"I added a timeout but forgot to handle the exception"**: A timeout that crashes the process is not a timeout — it is a deferred crash. Every `try` block around a timeout needs a fallback that returns something useful to the user.
- **"My fallback is the same model with a different name"**: Using `gpt-4o` as primary and `gpt-4o` as fallback provides zero resilience against an OpenAI outage. Provider diversity is the entire value of a fallback chain.
- **"I log everything but never look at the logs"**: Logs without dashboards and alerts are archaeology, not observability. Define the three metrics you would page on, set alert thresholds before you deploy, and review anomalies weekly.
- **"I deployed without a kill switch"**: A feature flag that disables the agent entirely is the minimum viable safety mechanism. Without it, a misbehaving agent at 3 AM requires a code deployment to stop.
- **"I treat the harness as an afterthought"**: Retrofitting a harness onto an existing agent is significantly harder than building it in from the start. Treat the harness as part of the initial design, not a polish step after the demo works.

## What's Next

You understand the harness philosophy. Next: implementing the first line of defence — the checks that catch problems before they ever reach the agent.
→ [Input Guardrails and Validation](02-input-guardrails-and-validation.md)

Then: routing and intent classification — the deterministic layer that decides which handler receives the validated request.
→ [Routing and Intent Classification](03-routing-and-intent-classification.md)