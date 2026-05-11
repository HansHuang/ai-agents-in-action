# Deployment Strategies

## What You'll Learn
- Why deploying an AI agent is fundamentally different from deploying a traditional API
- Deployment architectures: serverless, containerized, and hybrid
- Streaming vs. batch: when to use each pattern
- Gradual rollout: feature flags, canary deployments, and A/B testing
- Infrastructure considerations: GPU provisioning, rate limiting, and cost controls
- Multi-region deployment for latency-sensitive AI applications
- Rollback strategies: when (not if) you need to revert

## Prerequisites
- [Building a Reliable Harness](../07-harness-engineering/01-the-harness-mindset.md) — the harness you're deploying
- [Model Providers](../05-the-tool-ecosystem/01-model-providers.md) — understanding deployment options for different providers
- [Agent Observability](../05-the-tool-ecosystem/03-agent-observability.md) — monitoring what you deploy

**Code samples:** [Python](../../code/python/09-deployment/) · [Node.js](../../code/nodejs/09-deployment/) · [Go](../../code/go/09-deployment/)

---

## Why AI Agent Deployment Is Different

You've deployed APIs before. You know about containers, load balancers, and CI/CD pipelines. AI agents add new challenges:

| Traditional API | AI Agent |
|:---|:---|
| Predictable latency (10-50ms) | Variable latency (500ms-30s) |
| Deterministic responses | Probabilistic responses |
| Fixed resource requirements | Bursty GPU/LLM usage |
| Simple health checks (200 OK) | Complex health checks (is the agent behaving correctly?) |
| Rollback = revert code | Rollback = revert code + prompt + model + config |
| Cost per request is negligible | Cost per request is measurable and variable |
| Stateless by design | Stateful conversations |
| One service to deploy | Multiple services (agent, vector DB, LLM provider, tools) |

Deployment for AI agents requires new patterns.

---

## Deployment Architectures

### Architecture 1: Serverless (Simplest)

Best for: Prototypes, low-traffic applications, bursty workloads.

```
┌─────────────────────────────────────────────────────────┐
│                    SERVERLESS                            │
│                                                          │
│  User ──▶ API Gateway ──▶ Lambda/Cloud Function         │
│                              │                           │
│                              ├──▶ LLM API (OpenAI)       │
│                              ├──▶ Vector DB (Pinecone)   │
│                              └──▶ Tool APIs              │
│                                                          │
│  Pros: Zero ops, auto-scaling, pay-per-use              │
│  Cons: Cold starts, timeout limits, stateless           │
└─────────────────────────────────────────────────────────┘
```

```python
# AWS Lambda example (simplified)
import json
from agent import ProductionHarness

# Initialize outside handler for connection reuse
harness = None

def lambda_handler(event, context):
    global harness
    
    # Lazy initialization (cold start optimization)
    if harness is None:
        harness = ProductionHarness(HarnessConfig.from_env())
    
    user_input = json.loads(event['body'])['message']
    user_id = event['requestContext']['authorizer']['userId']
    
    # Process with timeout awareness
    response = await asyncio.wait_for(
        harness.process(user_input, user_id=user_id),
        timeout=25  # Lambda has 30s timeout, leave margin
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({'response': response.content}),
        'headers': {'Content-Type': 'application/json'},
    }
```

**Serverless caveats for AI agents:**
- **Cold starts**: Initialization (loading models, connecting to databases) can take seconds. Use provisioned concurrency for latency-sensitive applications.
- **Timeout limits**: AWS Lambda has a 15-minute maximum. Agent conversations that exceed this need a different architecture.
- **Connection pooling**: Serverless functions can't maintain persistent connections to LLM providers efficiently. Use a connection proxy or managed service.
- **State management**: Conversations must be persisted externally (DynamoDB, Redis). The function itself is stateless.

### Architecture 2: Containerized (Most Flexible)

Best for: Production applications, complex workflows, stateful conversations.

```
┌─────────────────────────────────────────────────────────┐
│                    CONTAINERIZED                         │
│                                                          │
│  User ──▶ Load Balancer ──▶ Agent Service (K8s/ECS)     │
│                              │                           │
│                              ├──▶ LLM API (OpenAI)       │
│                              ├──▶ LLM API (Anthropic)    │
│                              ├──▶ Vector DB (Qdrant)     │
│                              ├──▶ Redis (Session State)  │
│                              ├──▶ PostgreSQL (Data)      │
│                              └──▶ Tool Services          │
│                                                          │
│  Pros: Full control, stateful, no timeouts               │
│  Cons: Ops overhead, scaling complexity                 │
└─────────────────────────────────────────────────────────┘
```

```python
# FastAPI service example
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from agent import ProductionHarness
import asyncio

app = FastAPI()
harness = ProductionHarness(HarnessConfig.from_env())

@app.post("/agent/chat")
async def chat(request: ChatRequest):
    """Synchronous chat endpoint."""
    try:
        response = await asyncio.wait_for(
            harness.process(
                user_input=request.message,
                user_id=request.user_id,
                session_id=request.session_id,
                conversation_history=request.history,
            ),
            timeout=120  # 2 minute timeout for agent tasks
        )
        
        return ChatResponse(
            content=response.content,
            trace_id=response.trace_id,
            tokens_used=response.tokens_used,
            cost=response.cost,
        )
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/agent/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    
    async def event_stream():
        async for event in harness.process_stream(
            user_input=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
        ):
            yield f"data: {json.dumps(event)}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    health_status = harness.get_health()
    
    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown."""
    await harness.shutdown()
```

**Containerized best practices:**
- **Readiness probes**: Don't send traffic until the harness is fully initialized
- **Liveness probes**: Restart if the agent loop hangs
- **Graceful shutdown**: Complete in-flight requests before shutting down
- **Resource limits**: Set CPU and memory limits appropriate for your workload
- **Horizontal scaling**: Scale based on request queue depth, not just CPU

### Architecture 3: Hybrid (Best of Both)

Best for: High-traffic applications with bursty patterns.

```
┌─────────────────────────────────────────────────────────┐
│                       HYBRID                             │
│                                                          │
│  User ──▶ API Gateway ──▶ Router Lambda                 │
│                              │                           │
│                    ┌─────────┼─────────┐                │
│                    ▼         ▼         ▼                │
│              Simple Chat  RAG Query  Agent Task         │
│              (Lambda)     (Lambda)   (Container)        │
│                                         │               │
│                                         ▼               │
│                                   Agent Service         │
│                                   (Long-running)        │
│                                                          │
│  Pros: Cost-optimized, fast for simple queries          │
│  Cons: Complexity, routing logic needed                 │
└─────────────────────────────────────────────────────────┘
```

The hybrid approach from [Routing and Intent Classification](../07-harness-engineering/03-routing-and-intent-classification.md) extends to infrastructure:
- **Simple chat** → Fast, cheap Lambda with gpt-4o-mini
- **RAG queries** → Lambda with vector DB connection
- **Complex agent tasks** → Containerized service with full agent loop
- **Long-running tasks** → Queue-based worker with async processing

---

## Streaming Deployment

Streaming changes your deployment architecture significantly.

### Streaming Infrastructure Requirements

```python
class StreamingDeployment:
    """
    Deployment configuration optimized for streaming AI responses.
    """
    
    def __init__(self):
        self.config = {
            # Connection management
            "max_concurrent_streams": 1000,
            "stream_timeout_seconds": 300,  # 5 minutes max stream
            "keepalive_interval_seconds": 15,  # Prevent proxy timeout
            
            # Buffering
            "disable_proxy_buffering": True,  # nginx: proxy_buffering off
            "disable_compression": True,  # Stream chunks individually
            
            # Backpressure
            "max_queue_size": 100,  # Max pending chunks per connection
            "drop_on_backpressure": False,  # Don't drop, slow down producer
            
            # Resource limits
            "max_tokens_per_stream": 4096,
            "cost_limit_per_stream": 0.50,
        }
```

**Streaming-specific concerns:**
- **Proxy buffering**: Nginx and similar proxies buffer responses. Disable buffering for streaming endpoints (`proxy_buffering off;` and the `X-Accel-Buffering: no` response header).
- **Connection limits**: Streaming connections are long-lived. Your server needs to handle many concurrent open connections.
- **Backpressure**: If the LLM generates tokens faster than the client receives them, you need a buffer. If the buffer fills, slow the producer — do not drop chunks silently.
- **Reconnection**: Clients disconnect. Use the SSE `retry:` field and echo back `Last-Event-ID` so clients resume mid-stream without re-sending the full request:

```python
async def event_stream(request: ChatRequest, last_event_id: str | None):
    # Resume from last checkpoint if client reconnected
    start_token = int(last_event_id) + 1 if last_event_id else 0

    event_id = start_token
    yield "retry: 3000\n"  # Ask client to reconnect after 3 s if dropped

    async for chunk in harness.process_stream(request.message, start_token=start_token):
        yield f"id: {event_id}\ndata: {json.dumps(chunk)}\n\n"
        event_id += 1
```

- **Keepalive comments**: Send an SSE comment (`: keepalive`) every 15 seconds so intermediate proxies don't close idle connections before the LLM finishes.

---

## Gradual Rollout

Never deploy to 100% of users at once.

### Feature Flags

```python
class DeploymentManager:
    """
    Manage gradual rollout of new agent versions and features.
    """
    
    def __init__(self, feature_flag_service):
        self.flags = feature_flag_service
    
    def get_agent_version(self, user_id: str) -> str:
        """
        Determine which agent version a user should get.
        
        Rollout stages:
        1. Internal team (100% of internal users)
        2. 1% of external users (canary)
        3. 5% of external users (extended canary)
        4. 25% of external users (beta)
        5. 100% of external users (full rollout)
        """
        if self.flags.is_internal_user(user_id):
            return self.flags.get_string("agent_version_internal", "v3.2.1")
        
        rollout_pct = self.flags.get_int("agent_v3_rollout_pct", 0)
        
        if self._user_in_rollout_group(user_id, rollout_pct):
            return "v3.2.1"  # New version
        else:
            return "v3.1.0"  # Stable version
    
    def _user_in_rollout_group(self, user_id: str, percentage: int) -> bool:
        """Deterministic rollout group assignment."""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        return (hash_value % 100) < percentage
    
    def promote_rollout(self, from_pct: int, to_pct: int):
        """
        Promote the rollout percentage.
        Monitor metrics between each promotion.
        """
        logger.info(f"Promoting agent v3.2.1 rollout: {from_pct}% → {to_pct}%")
        self.flags.set_int("agent_v3_rollout_pct", to_pct)
```

### Canary Deployment

```python
class CanaryDeployer:
    """
    Canary deployment for AI agents.
    Gradually shifts traffic to new version while monitoring for regressions.
    """
    
    def __init__(self, stable_harness: ProductionHarness,
                canary_harness: ProductionHarness,
                metrics: HarnessMetrics):
        self.stable = stable_harness
        self.canary = canary_harness
        self.metrics = metrics
        self.canary_pct = 5  # Start at 5%
    
    async def process(self, user_input: str, user_id: str, **kwargs):
        """Route to canary or stable based on rollout percentage."""
        
        if self._is_canary_user(user_id):
            response = await self.canary.process(user_input, user_id=user_id, **kwargs)
            response.metadata["version"] = "canary"
        else:
            response = await self.stable.process(user_input, user_id=user_id, **kwargs)
            response.metadata["version"] = "stable"
        
        return response
    
    def evaluate_canary(self) -> CanaryEvaluation:
        """
        Compare canary metrics against stable.
        
        Checks:
        - Error rate (canary should not be significantly higher)
        - Latency P95 (canary should not be significantly slower)
        - Cost per request (canary should not be significantly more expensive)
        - Task success rate (canary should not be significantly lower)
        - User satisfaction (canary should not have significantly more complaints)
        """
        stable_metrics = self.metrics.get_metrics_for_version("stable", window_minutes=60)
        canary_metrics = self.metrics.get_metrics_for_version("canary", window_minutes=60)
        
        issues = []
        
        # Error rate check
        if canary_metrics.error_rate > stable_metrics.error_rate * 1.5:
            issues.append(f"Error rate: stable={stable_metrics.error_rate:.2%}, canary={canary_metrics.error_rate:.2%}")
        
        # Latency check
        if canary_metrics.p95_latency > stable_metrics.p95_latency * 1.2:
            issues.append(f"P95 latency: stable={stable_metrics.p95_latency:.1f}s, canary={canary_metrics.p95_latency:.1f}s")
        
        # Cost check
        if canary_metrics.avg_cost > stable_metrics.avg_cost * 1.2:
            issues.append(f"Avg cost: stable=${stable_metrics.avg_cost:.3f}, canary=${canary_metrics.avg_cost:.3f}")
        
        return CanaryEvaluation(
            canary_pct=self.canary_pct,
            has_issues=len(issues) > 0,
            issues=issues,
            stable_metrics=stable_metrics,
            canary_metrics=canary_metrics,
            recommendation=self._generate_recommendation(issues),
        )
    
    def _generate_recommendation(self, issues: list[str]) -> str:
        """Generate a recommendation based on canary evaluation."""
        if not issues:
            return "Canary is healthy. Consider increasing rollout percentage."
        elif len(issues) <= 1:
            return "Minor issues detected. Monitor for another hour before promoting."
        else:
            return "Significant issues detected. Halt rollout and investigate."

@dataclass
class CanaryEvaluation:
    canary_pct: int
    has_issues: bool
    issues: list[str]
    stable_metrics: dict
    canary_metrics: dict
    recommendation: str
```

### Rollout Playbook

```
ROLLOUT PLAYBOOK
================

Stage 1: Internal Testing (Day 0)
- Deploy to internal/staging environment
- Run full evaluation suite
- Run red team assessment
- All tests must pass before proceeding

Stage 2: 1% Canary (Day 1-2)
- Deploy to 1% of external users
- Monitor for 24 hours minimum
- Check: error rate, latency, cost, safety blocks
- Pause rollout if any metric degrades >20%

Stage 3: 5% Extended Canary (Day 3-4)
- Increase to 5% of users
- Monitor for 48 hours
- Run A/B test on task success rate
- Pause if success rate drops >5%

Stage 4: 25% Beta (Day 5-7)
- Increase to 25% of users
- Monitor for 72 hours
- Collect user feedback
- Address any issues before proceeding

Stage 5: 100% Full Rollout (Day 8)
- Deploy to all users
- Continue monitoring for 1 week
- Keep previous version ready for rollback

ROLLBACK TRIGGERS (at any stage):
- Error rate exceeds 2x baseline
- Safety block rate spikes >3x
- User complaints increase >50%
- Cost per request exceeds 2x baseline
- Critical security vulnerability discovered

ROLLBACK PROCESS:
1. Set feature flag to 0% for new version
2. All traffic returns to stable version
3. Investigate root cause
4. Fix and restart rollout from Stage 1
```

---

## Cost Control in Production

AI agents cost money. Production deployment requires cost controls.

```python
class ProductionCostController:
    """
    Enforce cost controls in production.
    """
    
    def __init__(self, config: CostConfig):
        self.config = config
        self.daily_costs: dict[str, float] = {}  # Per-user daily cost
        self.total_daily_cost: float = 0.0
        self.alert_thresholds = {
            "warning": 0.7,   # 70% of budget
            "critical": 0.9,  # 90% of budget
            "shutdown": 1.0,  # 100% of budget
        }
    
    def check_budget(self, user_id: str, estimated_cost: float) -> BudgetCheck:
        """
        Check if a request fits within the budget.
        
        Returns: BudgetCheck with allowed/rejected and reason.
        """
        user_daily = self.daily_costs.get(user_id, 0.0)
        
        # Per-user budget check
        if self.config.user_daily_budget:
            if user_daily + estimated_cost > self.config.user_daily_budget:
                return BudgetCheck(
                    allowed=False,
                    reason=f"Daily budget of ${self.config.user_daily_budget:.2f} exceeded. "
                           f"Current: ${user_daily:.2f}",
                )
        
        # Total daily budget check
        if self.config.total_daily_budget:
            if self.total_daily_cost + estimated_cost > self.config.total_daily_budget:
                return BudgetCheck(
                    allowed=False,
                    reason="Service temporarily unavailable due to high demand. "
                           "Please try again later.",
                )
        
        return BudgetCheck(allowed=True)
    
    def record_cost(self, user_id: str, cost: float):
        """Record actual cost after request completion."""
        self.daily_costs[user_id] = self.daily_costs.get(user_id, 0.0) + cost
        self.total_daily_cost += cost
        
        # Check alert thresholds
        if self.config.total_daily_budget:
            pct_used = self.total_daily_cost / self.config.total_daily_budget
            
            if pct_used >= self.alert_thresholds["shutdown"]:
                self._trigger_alert("critical", f"Daily budget exhausted: ${self.total_daily_cost:.2f}")
            elif pct_used >= self.alert_thresholds["critical"]:
                self._trigger_alert("warning", f"Daily budget at {pct_used:.0%}: ${self.total_daily_cost:.2f}")
            elif pct_used >= self.alert_thresholds["warning"]:
                self._trigger_alert("info", f"Daily budget at {pct_used:.0%}: ${self.total_daily_cost:.2f}")
    
    def get_cost_report(self) -> dict:
        """Generate cost report."""
        return {
            "total_daily_cost": self.total_daily_cost,
            "daily_budget": self.config.total_daily_budget,
            "budget_remaining": (self.config.total_daily_budget or 0) - self.total_daily_cost,
            "pct_used": self.total_daily_cost / (self.config.total_daily_budget or 1),
            "top_users": sorted(
                self.daily_costs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "user_count": len(self.daily_costs),
            "avg_cost_per_user": self.total_daily_cost / max(len(self.daily_costs), 1),
        }

@dataclass
class CostConfig:
    user_daily_budget: float = 10.0    # Per-user daily limit
    total_daily_budget: float = 1000.0  # Total daily limit
    max_cost_per_request: float = 1.0   # Single request limit
    free_tier_daily_budget: float = 0.50  # Free users get less
```

---

## Multi-Region Deployment

LLM APIs have regional availability. Deploy close to your users and your LLM providers.

```python
class MultiRegionDeployer:
    """
    Route users to the nearest available region.
    """
    
    REGIONS = {
        "us-east": {
            "llm_provider": "openai",  # OpenAI primary region
            "fallback_provider": "anthropic",
            "vector_db_endpoint": "https://us-east.qdrant.example.com",
            "latency_to_provider_ms": 50,
        },
        "eu-west": {
            "llm_provider": "openai",  # OpenAI EU region
            "fallback_provider": "anthropic",
            "vector_db_endpoint": "https://eu-west.qdrant.example.com",
            "latency_to_provider_ms": 80,
        },
        "ap-southeast": {
            "llm_provider": "anthropic",  # Anthropic has better AP latency
            "fallback_provider": "openai",
            "vector_db_endpoint": "https://ap-se.qdrant.example.com",
            "latency_to_provider_ms": 120,
        },
    }
    
    def get_region(self, user_ip: str, user_preferences: dict = None) -> str:
        """
        Determine the best region for a user.
        
        Factors:
        - Geographic proximity (latency to our service)
        - LLM provider latency from that region
        - Data residency requirements (GDPR, etc.)
        - User preferences (if they've selected a region)
        """
        # Data residency: EU users must stay in EU
        if self._is_eu_user(user_ip):
            return "eu-west"
        
        # Geographic routing
        region = self._geo_route(user_ip)
        
        # Check if preferred region is healthy
        if not self._is_region_healthy(region):
            return self._get_nearest_healthy_region(region)
        
        return region
    
    def _is_eu_user(self, ip: str) -> bool:
        """Check if IP is from the EU (GDPR requirement)."""
        ...
    
    def _geo_route(self, ip: str) -> str:
        """Route to nearest geographic region."""
        ...
    
    def _is_region_healthy(self, region: str) -> bool:
        """Check if a region is healthy (circuit breaker not open)."""
        ...
```

---

## Rollback Strategies

Rollback for AI agents is more complex than reverting code.

### The AI Rollback Checklist

Items are always executed **fastest-first** (sorted by `time_seconds`). Quick wins like reverting a config or prompt happen in seconds, limiting blast radius while the slower code and document rollbacks proceed.

```python
class RollbackManager:
    """
    Manage rollbacks for AI agent deployments.
    """
    
    def __init__(self):
        self.rollback_items = [
            RollbackItem(
                name="code",
                description="Revert application code to previous version",
                method=self._rollback_code,
                time_seconds=60,
            ),
            RollbackItem(
                name="model",
                description="Revert to previous model version (e.g., gpt-4o → gpt-4o-mini)",
                method=self._rollback_model,
                time_seconds=30,
            ),
            RollbackItem(
                name="prompt",
                description="Revert system prompt to previous version",
                method=self._rollback_prompt,
                time_seconds=10,
            ),
            RollbackItem(
                name="config",
                description="Revert harness configuration to previous version",
                method=self._rollback_config,
                time_seconds=10,
            ),
            RollbackItem(
                name="tools",
                description="Revert tool definitions/implementations to previous version",
                method=self._rollback_tools,
                time_seconds=30,
            ),
            RollbackItem(
                name="documents",
                description="Revert knowledge base to previous version",
                method=self._rollback_documents,
                time_seconds=300,  # Re-embedding takes time
            ),
        ]
    
    async def rollback(self, reason: str, items: list[str] = None) -> RollbackResult:
        """
        Execute a rollback.
        
        Args:
            reason: Why are we rolling back?
            items: Specific items to rollback (None = rollback everything)
        """
        logger.warning(f"ROLLBACK INITIATED: {reason}")
        
        to_rollback = self.rollback_items
        if items:
            to_rollback = [r for r in self.rollback_items if r.name in items]
        
        results = []
        for item in sorted(to_rollback, key=lambda r: r.time_seconds):
            logger.info(f"Rolling back: {item.name}...")
            try:
                await item.method()
                results.append(RollbackItemResult(name=item.name, success=True))
                logger.info(f"Rollback complete: {item.name}")
            except Exception as e:
                results.append(RollbackItemResult(name=item.name, success=False, error=str(e)))
                logger.error(f"Rollback failed: {item.name}: {e}")
        
        return RollbackResult(
            reason=reason,
            items=results,
            total_time_seconds=sum(r.time_seconds for r in to_rollback),
            success=all(r.success for r in results),
        )
    
    async def _rollback_code(self):
        """Revert to previous git commit."""
        ...
    
    async def _rollback_model(self):
        """Switch back to previous model version."""
        ...
    
    async def _rollback_prompt(self):
        """Revert system prompt to previous version."""
        ...
```

---

## The Deployment Checklist

Before deploying to production, verify every item:

### Pre-Deployment
- [ ] All evaluation tests pass (retrieval, generation, end-to-end)
- [ ] Red team assessment shows block rate > 95%
- [ ] Safety regression suite passes
- [ ] Load testing completed (target throughput achieved)
- [ ] Cost estimates validated against budget
- [ ] Rollback plan documented and tested
- [ ] Monitoring dashboards configured
- [ ] Alerts configured for critical metrics
- [ ] Runbook updated with new deployment procedures

### Deployment
- [ ] Feature flags configured for gradual rollout
- [ ] Canary deployment starts at 1%
- [ ] Monitor canary for minimum 24 hours
- [ ] No significant regressions in error rate, latency, or cost
- [ ] Gradual promotion through rollout stages
- [ ] Previous version kept warm for instant rollback

### Post-Deployment
- [ ] Monitor for 1 week at full traffic
- [ ] Review cost trends daily
- [ ] Review safety blocks weekly
- [ ] Collect user feedback
- [ ] Archive previous version after 2 weeks of stability

---

## Common Pitfalls

- **"I deployed to 100% on Friday at 5 PM"**: Never deploy significant changes before weekends or holidays. You won't be around to fix issues. Deploy on Tuesday morning.
- **"I didn't test with production-scale concurrency"**: Your agent works with 10 concurrent users. It crashes with 1,000. Load test before production.
- **"I forgot about cold starts"**: Serverless functions have cold starts. Your agent's initialization might take 10 seconds. Users won't wait that long.
- **"I didn't set up cost alerts"**: You check the billing dashboard a week later and discover you spent $5,000 instead of $500. Set daily cost alerts.
- **"My rollback plan is 'git revert and redeploy'"**: Reverting code isn't enough. You might need to revert the model, prompt, config, and knowledge base too.
- **"I treat all regions the same"**: LLM API latency varies significantly by region. Deploy where your LLM provider has the best performance.
- **"I rolled back code but left the new prompt"**: Rollback artifacts must be consistent. A v3.1 code base running a v3.2 system prompt can produce unexpected behaviour — always rollback the full slice (code + prompt + config together).
- **"My canary only tested happy-path queries"**: Canary deployments catch regressions on average traffic. Edge cases — very long conversations, tool-heavy tasks, adversarial inputs — need a dedicated evaluation suite that runs before every promotion.

## What's Next

You can now deploy an AI agent to production with confidence. The final chapter: the 12-factor agent — design principles for building agents that thrive in production.
→ [The 12-Factor Agent](02-the-12-factor-agent.md)
