# Agent Observability

## What You'll Learn
- Why agent debugging is harder than traditional software debugging
- The three pillars: tracing, logging, and metrics
- Building an observability pipeline: what to capture at every step
- Token accounting: tracking cost per user, per session, per task
- Decision tracing: reconstructing why the agent did what it did
- Production monitoring: detecting drift, failures, and anomalies

## Prerequisites
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — the agent loop you're observing
- [Model Providers](01-model-providers.md) — different models, different costs
- [Context Engineering](../04-context-engineering/01-the-context-window-as-a-resource.md) — context is what you're tracing

---

## The Observability Crisis

You deploy your agent. A user reports: *"It gave me the wrong answer."*

In traditional software, you'd check the logs, find the error, fix the bug. With an agent, you face a cascade of questions:

- What was the user's exact input?
- What was the full conversation history at that point?
- Which tools did the agent call? What did they return?
- What was the model's reasoning? Which model was used?
- Was it a planning failure, a tool failure, or a generation failure?
- Did the context window overflow and truncate critical information?
- How many tokens did this conversation consume? What did it cost?

Without observability, the answer to every question is: *"I don't know."*

---

## The Three Pillars

| Pillar | What It Captures | Answers the Question |
|:---|:---|:---|
| **Tracing** | The full path of a single request through the system | "What happened during this specific interaction?" |
| **Logging** | Structured events at each step | "What was the state at this point in time?" |
| **Metrics** | Aggregated measurements over time | "Is the system healthy? Are there trends?" |

An observable agent implements all three.

---

## Tracing: The Agent Decision Tree

Every agent interaction is a tree of decisions. A trace captures that tree.

```
User Query: "Compare Apple and Microsoft stock"
│
├── [LLM Call #1] Model: gpt-4o | Tokens: 2,340 | Latency: 1.2s
│   └── Decision: Call get_stock_price("AAPL")
│       │
│       ├── [Tool Call] get_stock_price("AAPL") | Latency: 0.3s
│       │   └── Result: {"price": 182.52, "change": "+1.2%"}
│       │
│       ├── [LLM Call #2] Model: gpt-4o | Tokens: 2,890 | Latency: 1.5s
│       │   └── Decision: Call get_stock_price("MSFT")
│       │       │
│       │       ├── [Tool Call] get_stock_price("MSFT") | Latency: 0.2s
│       │       │   └── Result: {"price": 428.74, "change": "-0.3%"}
│       │       │
│       │       └── [LLM Call #3] Model: gpt-4o | Tokens: 3,100 | Latency: 1.8s
│       │           └── Final Answer: "Apple is at $182.52 (+1.2%)..."
│       │
│       └── Total: 3 LLM calls, 2 tool calls, 8,330 tokens, 5.0s
```

### Implementing Tracing

```python
import time
import uuid
from dataclasses import dataclass, field

@dataclass
class Trace:
    """A single request trace through the agent system."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""
    user_id: str | None = None      # for query_traces() filtering
    session_id: str | None = None   # for per-session cost tracking
    spans: list["Span"] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def add_span(self, span: "Span") -> None:
        self.spans.append(span)
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "user_query": self.user_query,
            "total_duration_ms": (time.time() - self.start_time) * 1000,
            "total_llm_calls": sum(1 for s in self.spans if s.type == "llm_call"),
            "total_tool_calls": sum(1 for s in self.spans if s.type == "tool_call"),
            "total_tokens": sum(s.tokens_used or 0 for s in self.spans),
            "total_cost": sum(s.cost or 0 for s in self.spans),
            "spans": [s.to_dict() for s in self.spans]
        }

@dataclass
class Span:
    """A single operation within a trace."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str = None
    type: str = ""  # "llm_call", "tool_call", "retrieval", "planning"
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = None
    input_data: dict = None
    output_data: dict = None
    tokens_used: int = None
    cost: float = None
    model: str = None
    status: str = "running"  # updated to "success" or "error" on finish()
    error_message: str = None
    
    def finish(self, output_data: dict = None, status: str = "success",
               error_message: str = None) -> None:
        self.end_time = time.time()
        self.output_data = output_data
        self.status = status
        self.error_message = error_message
    
    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "type": self.type,
            "name": self.name,
            "duration_ms": ((self.end_time or time.time()) - self.start_time) * 1000,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "model": self.model,
            "status": self.status,
            "error_message": self.error_message
        }

class ObservableAgent:
    """An agent that produces traces."""
    
    def __init__(self, llm_provider, tools, tracer=None):
        self.llm = llm_provider
        self.tools = tools
        self.tracer = tracer or TraceCollector()
    
    def run(self, user_input: str) -> dict:
        trace = Trace(user_query=user_input)
        
        try:
            # Span: Planning
            plan_span = Span(type="planning", name="generate_plan")
            trace.add_span(plan_span)
            
            plan = self._generate_plan(user_input)
            plan_span.finish(output_data={"plan": plan})
            
            # Span: Execution (with child spans for each step)
            exec_span = Span(type="execution", name="execute_plan",
                           parent_span_id=plan_span.span_id)
            trace.add_span(exec_span)
            
            results = []
            for step in plan:
                step_span = Span(type="tool_call", name=step["tool_name"],
                               parent_span_id=exec_span.span_id)
                trace.add_span(step_span)
                
                result = self._execute_step(step)
                step_span.finish(output_data=result)
                results.append(result)
            
            exec_span.finish(output_data={"steps_completed": len(results)})
            
            # Span: Generation
            gen_span = Span(type="llm_call", name="generate_answer",
                          parent_span_id=exec_span.span_id)
            trace.add_span(gen_span)
            
            answer = self._generate_answer(user_input, results)
            gen_span.finish(output_data={"answer_length": len(answer)})
            
            # Export trace
            self.tracer.export(trace)
            
            return {"answer": answer, "trace_id": trace.trace_id}
            
        except Exception as e:
            # Ensure trace is exported even on failure
            trace.spans[-1].finish(status="error", error_message=str(e))
            self.tracer.export(trace)
            raise
```

> **Code Reference:** [Python](../../code/python/05-the-tool-ecosystem/observability/) · [Node.js](../../code/nodejs/05-the-tool-ecosystem/observability/) · [Go](../../code/go/05-the-tool-ecosystem/observability/)  
> The observability implementations include the full tracing, logging, and metrics pipeline.

---

## Logging: Structured Events at Every Step

Logs answer: *"What was the state at this point?"*

### What to Log

```python
import logging
import json

class AgentLogger:
    """Structured logging for agent operations."""
    
    def __init__(self, logger_name: str = "agent"):
        self.logger = logging.getLogger(logger_name)
    
    def log_llm_call(self, trace_id: str, span_id: str,
                     model: str, messages_count: int,
                     estimated_tokens: int) -> None:
        """Log an LLM call attempt."""
        self.logger.info(json.dumps({
            "event": "llm_call_start",
            "trace_id": trace_id,
            "span_id": span_id,
            "model": model,
            "messages_count": messages_count,
            "estimated_tokens": estimated_tokens,
            "timestamp": time.time()
        }))
    
    def log_llm_response(self, response, trace_id: str, 
                         span_id: str, latency_ms: float) -> None:
        """Log an LLM response."""
        self.logger.info(json.dumps({
            "event": "llm_call_complete",
            "trace_id": trace_id,
            "span_id": span_id,
            "model": response.model,
            "input_tokens": response.token_usage.get("prompt_tokens", 0),
            "output_tokens": response.token_usage.get("completion_tokens", 0),
            "finish_reason": response.finish_reason,
            "has_tool_calls": response.tool_calls is not None,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        }))
    
    def log_tool_call(self, trace_id: str, span_id: str,
                      tool_name: str, params_summary: str) -> None:
        """Log a tool execution. Pass a pre-sanitised params_summary — never
        log raw params directly as they may contain PII or secrets."""
        self.logger.info(json.dumps({
            "event": "tool_call_start",
            "trace_id": trace_id,
            "span_id": span_id,
            "tool_name": tool_name,
            "params_summary": params_summary,
            "timestamp": time.time()
        }))
    
    def log_tool_result(self, tool_name: str, success: bool,
                        result_summary: str, latency_ms: float,
                        trace_id: str, span_id: str) -> None:
        """Log a tool result."""
        self.logger.info(json.dumps({
            "event": "tool_result",
            "trace_id": trace_id,
            "span_id": span_id,
            "tool_name": tool_name,
            "success": success,
            "result_summary": result_summary[:200],  # Truncate long results
            "latency_ms": latency_ms,
            "timestamp": time.time()
        }))
    
    def log_context_truncation(self, original_tokens: int,
                               truncated_tokens: int,
                               strategy: str,
                               trace_id: str) -> None:
        """Log when context was truncated."""
        self.logger.warning(json.dumps({
            "event": "context_truncation",
            "trace_id": trace_id,
            "original_tokens": original_tokens,
            "truncated_tokens": truncated_tokens,
            "tokens_removed": original_tokens - truncated_tokens,
            "strategy": strategy,
            "timestamp": time.time()
        }))
    
    def log_error(self, error: Exception, context: dict,
                  trace_id: str) -> None:
        """Log an error with full context."""
        self.logger.error(json.dumps({
            "event": "agent_error",
            "trace_id": trace_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": time.time()
        }))
```

### Log Levels for Agents

| Level | What to Log | Example |
|:---|:---|:---|
| **DEBUG** | Full message content, tool parameters, raw LLM responses | "Messages sent: [{role: 'user', content: '...'}]" |
| **INFO** | Trace lifecycle, LLM call summaries, tool call summaries | "LLM call complete: 2,340 tokens, gpt-4o, 1.2s" |
| **WARNING** | Context truncation, retry attempts, fallback activation | "Context truncated: 85,000 → 65,000 tokens (sliding_window)" |
| **ERROR** | Failed LLM calls, tool execution failures, validation errors | "Tool 'get_weather' failed: API timeout after 3 retries" |
| **CRITICAL** | Data corruption, security events, complete system failures | "API key invalid. All agent operations halted." |

---

## Metrics: The System Health Dashboard

Metrics answer: *"Is the system healthy?"*

### Essential Agent Metrics

```python
class AgentMetrics:
    """Collect and export agent performance metrics."""
    
    def __init__(self):
        self.metrics: dict[str, list] = {
            "llm_call_latency_ms": [],
            "tool_call_latency_ms": [],
            "total_request_latency_ms": [],
            "tokens_per_request": [],
            "cost_per_request": [],
            "tool_calls_per_request": [],
            "llm_calls_per_request": [],
            "error_rate": [],  # 0 or 1 per request
            "context_truncation_rate": [],
        }
    
    def record_request(self, trace: Trace) -> None:
        """Record all metrics from a completed trace."""
        self.metrics["total_request_latency_ms"].append(
            (trace.spans[-1].end_time - trace.start_time) * 1000
        )
        self.metrics["tokens_per_request"].append(
            sum(s.tokens_used or 0 for s in trace.spans)
        )
        self.metrics["cost_per_request"].append(
            sum(s.cost or 0 for s in trace.spans)
        )
        self.metrics["tool_calls_per_request"].append(
            sum(1 for s in trace.spans if s.type == "tool_call")
        )
        self.metrics["llm_calls_per_request"].append(
            sum(1 for s in trace.spans if s.type == "llm_call")
        )
        self.metrics["error_rate"].append(
            1 if any(s.status == "error" for s in trace.spans) else 0
        )
    
    def get_summary(self, window_minutes: int = 60) -> dict:
        """Get a summary of recent metrics."""
        return {
            "requests_per_minute": self._rate("total_request_latency_ms", window_minutes),
            "avg_latency_ms": self._avg("total_request_latency_ms"),
            "p95_latency_ms": self._percentile("total_request_latency_ms", 95),
            "avg_tokens_per_request": self._avg("tokens_per_request"),
            "avg_cost_per_request": self._avg("cost_per_request"),
            "total_cost": sum(self.metrics["cost_per_request"]),
            "error_rate": self._avg("error_rate") * 100,  # As percentage
            "avg_llm_calls_per_request": self._avg("llm_calls_per_request"),
            "avg_tool_calls_per_request": self._avg("tool_calls_per_request"),
        }
    
    def detect_anomalies(self) -> list[str]:
        """Detect anomalous patterns in recent metrics."""
        alerts = []
        
        # Error rate spike
        recent_errors = self.metrics["error_rate"][-100:]
        if sum(recent_errors) / len(recent_errors) > 0.1:  # >10% error rate
            alerts.append(f"High error rate: {sum(recent_errors)/len(recent_errors)*100:.1f}%")
        
        # Latency spike
        recent_latency = self.metrics["total_request_latency_ms"][-100:]
        avg_latency = sum(recent_latency) / len(recent_latency)
        if avg_latency > 10000:  # >10 seconds
            alerts.append(f"High latency: {avg_latency/1000:.1f}s average")
        
        # Cost spike
        recent_cost = self.metrics["cost_per_request"][-100:]
        avg_cost = sum(recent_cost) / len(recent_cost)
        if avg_cost > 0.50:  # >$0.50 per request
            alerts.append(f"High cost per request: ${avg_cost:.2f}")
        
        return alerts
```

---

## Token Accounting: Who Spent What?

Tokens are money. Track them per user, per session, per task.

```python
class TokenAccountant:
    """Track token usage and cost across dimensions."""
    
    def __init__(self, pricing: dict = None):
        self.pricing = pricing or DEFAULT_PRICING
        self.usage_by_user: dict[str, list] = {}
        self.usage_by_session: dict[str, list] = {}
        self.usage_by_model: dict[str, list] = {}
    
    def record(self, trace: Trace, user_id: str, 
               session_id: str) -> None:
        """Record token usage from a trace."""
        usage = {
            "trace_id": trace.trace_id,
            "timestamp": time.time(),
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        }
        
        for span in trace.spans:
            if span.type == "llm_call" and span.tokens_used:
                usage["model"] = span.model
                usage["input_tokens"] += span.input_tokens or 0
                usage["output_tokens"] += span.output_tokens or 0
                usage["cost"] += span.cost or 0  # pre-computed in the span
        
        if user_id not in self.usage_by_user:
            self.usage_by_user[user_id] = []
        self.usage_by_user[user_id].append(usage)
        
        if session_id not in self.usage_by_session:
            self.usage_by_session[session_id] = []
        self.usage_by_session[session_id].append(usage)
        
        model_key = usage["model"] or "unknown"
        if model_key not in self.usage_by_model:
            self.usage_by_model[model_key] = []
        self.usage_by_model[model_key].append(usage)
    
    def get_user_cost(self, user_id: str, 
                      since_days: int = 30) -> float:
        """Get total cost for a user."""
        cutoff = time.time() - (since_days * 86400)
        total = sum(
            u["cost"] for u in self.usage_by_user.get(user_id, [])
            if u["timestamp"] >= cutoff
        )
        return total
    
    def get_daily_report(self) -> dict:
        """Generate a daily cost report."""
        today = time.time() - 86400
        recent = [
            u for user_list in self.usage_by_user.values()
            for u in user_list if u["timestamp"] >= today
        ]
        
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_requests": len(recent),
            "total_tokens": sum(u["input_tokens"] + u["output_tokens"] for u in recent),
            "total_cost": sum(u["cost"] for u in recent),
            "unique_users": len(self.usage_by_user),
            "by_model": {
                model: {
                    "requests": len(usages),
                    "total_tokens": sum(u["input_tokens"] + u["output_tokens"] for u in usages),
                    "total_cost": sum(u["cost"] for u in usages)
                }
                for model, usages in self.usage_by_model.items()
            }
        }
```

---

## Decision Tracing: Why Did the Agent Do That?

The hardest question to answer. Capture the model's reasoning at each step.

```python
class DecisionTracer:
    """Capture and explain agent decisions."""
    
    def __init__(self):
        self.decisions: list[AgentDecision] = []
    
    def capture_decision(self, step_type: str, context: dict,
                        options: list[str], chosen: str,
                        reasoning: str) -> AgentDecision:
        """Record an agent decision with context."""
        decision = AgentDecision(
            timestamp=time.time(),
            step_type=step_type,
            context=context,
            options=options,
            chosen=chosen,
            reasoning=reasoning
        )
        self.decisions.append(decision)
        return decision
    
    def explain_path(self, trace_id: str = None) -> str:
        """Generate a human-readable explanation of the agent's decisions."""
        lines = ["# Agent Decision Trail\n"]
        
        for i, decision in enumerate(self.decisions):
            lines.append(f"## Step {i+1}: {decision.step_type}")
            lines.append(f"**Context:** {decision.context}")
            lines.append(f"**Options considered:** {', '.join(decision.options)}")
            lines.append(f"**Chosen:** {decision.chosen}")
            lines.append(f"**Reasoning:** {decision.reasoning}")
            lines.append("")
        
        return "\n".join(lines)
    
    def debug_incorrect_decision(self, expected: str) -> str:
        """Given an expected decision, find where the agent diverged."""
        for i, decision in enumerate(self.decisions):
            if expected in decision.options and decision.chosen != expected:
                return (
                    f"Agent chose '{decision.chosen}' instead of '{expected}' "
                    f"at step {i+1} ({decision.step_type}).\n"
                    f"Reasoning was: {decision.reasoning}\n"
                    f"Context was: {decision.context}"
                )
        return "No divergence found — agent may have made the correct choice."

@dataclass
class AgentDecision:
    timestamp: float
    step_type: str
    context: dict
    options: list[str]
    chosen: str
    reasoning: str
```

---

## Production Monitoring

### Health Check Endpoint

```python
class AgentHealthCheck:
    """Monitor agent system health."""
    
    def __init__(self, agent, metrics: AgentMetrics,
                 vector_db=None, llm_provider=None):
        self.agent = agent
        self.metrics = metrics
        self.vector_db = vector_db
        self.llm_provider = llm_provider
    
    def check(self) -> dict:
        """Run a comprehensive health check."""
        checks = {}
        
        # LLM connectivity
        try:
            self.llm_provider.chat([{"role": "user", "content": "ping"}], max_tokens=1)
            checks["llm_connection"] = "healthy"
        except Exception as e:
            checks["llm_connection"] = f"unhealthy: {e}"
        
        # Vector DB connectivity
        if self.vector_db:
            try:
                self.vector_db.count()
                checks["vector_db"] = "healthy"
            except Exception as e:
                checks["vector_db"] = f"unhealthy: {e}"
        
        # Recent metrics
        summary = self.metrics.get_summary(window_minutes=5)
        checks["recent_error_rate"] = f"{summary['error_rate']:.1f}%"
        checks["recent_latency_p95"] = f"{summary['p95_latency_ms']:.0f}ms"
        
        # Anomalies
        anomalies = self.metrics.detect_anomalies()
        checks["anomalies"] = anomalies if anomalies else "none"
        
        # Overall status
        has_critical = any("unhealthy" in v for v in checks.values())
        checks["overall"] = "unhealthy" if has_critical else "healthy"
        
        return checks
```

---

## Tools of the Trade

| Tool | Type | Best For |
|:---|:---|:---|
| **LangSmith** | Tracing + Evaluation | LangChain users, full-featured |
| **Arize Phoenix** | Tracing + Monitoring | Open-source, OpenTelemetry-native |
| **Weights & Biases Weave** | Experiment + LLM Tracing | Prompt iteration, model comparison |
| **LangFuse** | Tracing | Open-source, self-hostable |
| **Helicone** | Proxy + Analytics | Gateway-based observability (zero code change) |
| **OpenTelemetry** | Standard | Vendor-neutral; works with Jaeger, Grafana Tempo, Datadog |
| **Structured Logging** (JSON) | Logging | Always — it's free and essential |

---

## Common Pitfalls

- **"I log raw API keys or user data"**: Logs are often stored in plaintext. Never log API keys, passwords, or PII. Redact sensitive fields.
- **"I have traces but no way to query them"**: 10,000 traces are useless if you can't find the one where the agent failed. Index traces by trace_id, user_id, and error status.
- **"I measure latency but not token usage"**: Latency tells you *when* something is slow. Token usage tells you *why*. They're related — more tokens = more latency = more cost.
- **"My metrics look fine but users are unhappy"**: Aggregate metrics hide individual failures. A 99% success rate means 1 in 100 users has a bad experience. Trace individual failures.
- **"I don't alert on anomalies"**: If your cost per request doubles overnight, you want to know immediately. Set up alerts for cost, latency, and error rate.
- **"I mix input and output token counts"**: Input tokens (your prompt) and output tokens (the model's reply) have different per-token prices on every model. Store them separately in every span, or your cost calculations will be wrong.

## What's Next

You can now see inside your agent's decision-making. Next: the Model Context Protocol (MCP) — a standard for connecting agents to tools and data sources across providers.
→ [MCP Protocol](04-mcp-protocol.md)