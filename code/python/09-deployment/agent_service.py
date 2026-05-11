"""
FastAPI Agent Service
=====================
Production-ready FastAPI service for deploying an AI agent.

Endpoints:
  POST /agent/chat          — synchronous chat
  POST /agent/chat/stream   — Server-Sent Events streaming
  GET  /health              — comprehensive health status
  GET  /metrics             — Prometheus-format metrics

Reference: docs/09-from-dev-to-production/01-deployment-strategies.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Settings (environment-based, validated by Pydantic)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    # LLM providers
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Service behaviour
    default_model: str = Field(default="gpt-4o-mini", alias="DEFAULT_MODEL")
    request_timeout_seconds: int = Field(default=120, alias="REQUEST_TIMEOUT_SECONDS")
    stream_timeout_seconds: int = Field(default=300, alias="STREAM_TIMEOUT_SECONDS")

    # Rate limiting
    global_rate_limit_rps: int = Field(default=1000, alias="GLOBAL_RATE_LIMIT_RPS")
    per_user_rate_limit_rpm: int = Field(default=60, alias="PER_USER_RATE_LIMIT_RPM")

    # Cost limits
    daily_budget_usd: float = Field(default=1000.0, alias="DAILY_BUDGET_USD")
    user_daily_budget_usd: float = Field(default=10.0, alias="USER_DAILY_BUDGET_USD")

    # CORS
    allowed_origins: list[str] = Field(
        default=["*"], alias="ALLOWED_ORIGINS"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"populate_by_name": True}


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format='{"time": "%(asctime)s", "level": "%(levelname)s", '
           '"logger": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger("agent_service")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32_000)
    user_id: str = Field(..., min_length=1, max_length=256)
    session_id: str | None = None
    history: list[dict] | None = None
    metadata: dict | None = None

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class ChatResponse(BaseModel):
    content: str
    trace_id: str
    tokens_used: int
    cost: float
    route: str
    handler_used: str
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Stubs (replace with real harness, metrics, circuit-breakers, etc.)
# ---------------------------------------------------------------------------

class AgentHarnessStub:
    """
    Minimal stand-in for a ProductionHarness.
    Replace with the real harness from code/python/07-harness/.
    """

    async def process(self, user_input: str, *, user_id: str,
                      session_id: str | None = None,
                      conversation_history: list | None = None,
                      **kwargs):
        await asyncio.sleep(0.1)  # Simulate LLM latency
        return type("AgentResponse", (), {
            "content": f"This is a response to: {user_input}",
            "trace_id": str(uuid.uuid4()),
            "tokens_used": 150,
            "cost": 0.002,
            "route": "direct_llm",
            "handler_used": "default",
            "metadata": {},
        })()

    async def process_stream(self, user_input: str, *, user_id: str,
                             session_id: str | None = None,
                             **kwargs) -> AsyncGenerator[dict, None]:
        trace_id = str(uuid.uuid4())
        tokens = f"This is a streaming response to: {user_input}".split()
        for i, token in enumerate(tokens):
            await asyncio.sleep(0.02)
            yield {"type": "token", "data": token + (" " if i < len(tokens) - 1 else "")}

        yield {
            "type": "complete",
            "data": {
                "trace_id": trace_id,
                "tokens_used": len(tokens),
                "cost": round(len(tokens) * 0.000002, 6),
                "route": "direct_llm",
                "handler_used": "default",
            },
        }

    def get_health(self) -> dict:
        return {
            "status": "healthy",
            "llm_connectivity": True,
            "vector_db_connectivity": True,
            "circuit_breakers": {"openai": "closed", "anthropic": "closed"},
        }

    async def shutdown(self):
        logger.info("Harness shutdown complete.")


# Simple in-memory rate limiter (use Redis token-bucket in production)
class RateLimiter:
    def __init__(self, per_user_rpm: int, global_rps: int):
        self._per_user_rpm = per_user_rpm
        self._global_rps = global_rps
        self._user_windows: dict[str, list[float]] = {}
        self._global_window: list[float] = []

    def check(self, user_id: str) -> tuple[bool, int]:
        """
        Returns (allowed, remaining_requests).
        Removes expired timestamps before checking.
        """
        now = time.time()
        cutoff_minute = now - 60
        cutoff_second = now - 1

        # Per-user rate limit
        window = self._user_windows.setdefault(user_id, [])
        window[:] = [t for t in window if t > cutoff_minute]
        if len(window) >= self._per_user_rpm:
            return False, 0
        window.append(now)
        remaining = self._per_user_rpm - len(window)

        # Global rate limit
        self._global_window[:] = [t for t in self._global_window if t > cutoff_second]
        if len(self._global_window) >= self._global_rps:
            return False, 0
        self._global_window.append(now)

        return True, remaining


# Simple in-memory cost tracker (replace with Redis in production)
class CostTracker:
    def __init__(self, daily_budget: float, user_daily_budget: float):
        self._daily_budget = daily_budget
        self._user_budget = user_daily_budget
        self._total: float = 0.0
        self._by_user: dict[str, float] = {}

    def record(self, user_id: str, cost: float):
        self._total += cost
        self._by_user[user_id] = self._by_user.get(user_id, 0.0) + cost

    @property
    def total_today(self) -> float:
        return self._total

    @property
    def budget_pct(self) -> float:
        return self._total / max(self._daily_budget, 0.01)

    def is_over_budget(self) -> bool:
        return self._total >= self._daily_budget

    def user_is_over_budget(self, user_id: str) -> bool:
        return self._by_user.get(user_id, 0.0) >= self._user_budget


# Simple metrics counters (replace with prometheus_client in production)
class Metrics:
    def __init__(self):
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.latencies: list[float] = []
        self._start = time.time()

    def record_request(self, latency: float, tokens: int, cost: float,
                       error: bool = False):
        self.request_count += 1
        self.latencies.append(latency)
        self.total_tokens += tokens
        self.total_cost += cost
        if error:
            self.error_count += 1

    def prometheus_text(self) -> str:
        uptime = time.time() - self._start
        lats_sorted = sorted(self.latencies)
        n = len(lats_sorted)

        def pct(p: float) -> float:
            if not lats_sorted:
                return 0.0
            return lats_sorted[min(int(n * p), n - 1)]

        lines = [
            "# HELP agent_requests_total Total number of agent requests",
            "# TYPE agent_requests_total counter",
            f"agent_requests_total {self.request_count}",
            "# HELP agent_errors_total Total number of errors",
            "# TYPE agent_errors_total counter",
            f"agent_errors_total {self.error_count}",
            "# HELP agent_tokens_total Total tokens consumed",
            "# TYPE agent_tokens_total counter",
            f"agent_tokens_total {self.total_tokens}",
            "# HELP agent_cost_usd_total Total cost in USD",
            "# TYPE agent_cost_usd_total counter",
            f"agent_cost_usd_total {self.total_cost:.6f}",
            "# HELP agent_latency_seconds Request latency",
            "# TYPE agent_latency_seconds summary",
            f'agent_latency_seconds{{quantile="0.5"}} {pct(0.50):.4f}',
            f'agent_latency_seconds{{quantile="0.95"}} {pct(0.95):.4f}',
            f'agent_latency_seconds{{quantile="0.99"}} {pct(0.99):.4f}',
            f"agent_latency_seconds_count {n}",
            "# HELP agent_uptime_seconds Service uptime",
            "# TYPE agent_uptime_seconds gauge",
            f"agent_uptime_seconds {uptime:.1f}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

harness: AgentHarnessStub | None = None
rate_limiter: RateLimiter | None = None
cost_tracker: CostTracker | None = None
metrics: Metrics | None = None
_startup_complete = False


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global harness, rate_limiter, cost_tracker, metrics, _startup_complete

    logger.info("Starting agent service …")
    harness = AgentHarnessStub()
    rate_limiter = RateLimiter(
        per_user_rpm=settings.per_user_rate_limit_rpm,
        global_rps=settings.global_rate_limit_rps,
    )
    cost_tracker = CostTracker(
        daily_budget=settings.daily_budget_usd,
        user_daily_budget=settings.user_daily_budget_usd,
    )
    metrics = Metrics()
    _startup_complete = True
    logger.info("Agent service ready.")

    yield  # Application runs here

    logger.info("Shutting down agent service …")
    _startup_complete = False
    if harness:
        await harness.shutdown()
    logger.info("Agent service stopped.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Agent Service",
    description="Production FastAPI service for AI agent deployment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to every request and response."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Structured JSON logging for every request."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "request completed",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "request_id": getattr(request.state, "request_id", ""),
        },
    )
    return response


# ---------------------------------------------------------------------------
# Helper: check service ready
# ---------------------------------------------------------------------------

def _require_ready() -> None:
    """Raise 503 if the service is not yet initialised."""
    if not _startup_complete or harness is None:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Please retry shortly.",
        )


def _check_rate_limit(user_id: str) -> int:
    """Check rate limit; raise 429 if exceeded. Returns remaining count."""
    allowed, remaining = rate_limiter.check(user_id)  # type: ignore[union-attr]
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": "60"},
        )
    return remaining


def _check_cost_budget(user_id: str) -> None:
    """Raise 503 if cost budget is exhausted."""
    if cost_tracker.is_over_budget():  # type: ignore[union-attr]
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to high demand.",
        )
    if cost_tracker.user_is_over_budget(user_id):  # type: ignore[union-attr]
        raise HTTPException(
            status_code=429,
            detail="Daily usage limit reached.",
            headers={"Retry-After": "3600"},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request) -> Response:
    """
    Synchronous chat endpoint.

    Accepts a ChatRequest, processes it through the agent harness,
    and returns a ChatResponse. Enforces rate limiting, budget checks,
    and a 120-second timeout.
    """
    _require_ready()
    remaining = _check_rate_limit(request_body.user_id)
    _check_cost_budget(request_body.user_id)

    start = time.time()
    try:
        agent_response = await asyncio.wait_for(
            harness.process(  # type: ignore[union-attr]
                user_input=request_body.message,
                user_id=request_body.user_id,
                session_id=request_body.session_id,
                conversation_history=request_body.history or [],
            ),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        metrics.record_request(  # type: ignore[union-attr]
            time.time() - start, 0, 0.0, error=True
        )
        raise HTTPException(status_code=504, detail="Request timed out.")
    except Exception as exc:
        latency = time.time() - start
        metrics.record_request(latency, 0, 0.0, error=True)  # type: ignore[union-attr]
        logger.exception("Agent error for user %s: %s", request_body.user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    latency = time.time() - start
    cost = getattr(agent_response, "cost", 0.0)
    tokens = getattr(agent_response, "tokens_used", 0)

    cost_tracker.record(request_body.user_id, cost)  # type: ignore[union-attr]
    metrics.record_request(latency, tokens, cost)  # type: ignore[union-attr]

    response_data = ChatResponse(
        content=agent_response.content,
        trace_id=getattr(agent_response, "trace_id", str(uuid.uuid4())),
        tokens_used=tokens,
        cost=cost,
        route=getattr(agent_response, "route", "direct_llm"),
        handler_used=getattr(agent_response, "handler_used", "default"),
        metadata=getattr(agent_response, "metadata", None),
    )

    return JSONResponse(
        content=response_data.model_dump(),
        headers={
            "X-RateLimit-Remaining": str(remaining),
            "X-Cost-This-Request": f"{cost:.6f}",
        },
    )


@app.post("/agent/chat/stream")
async def chat_stream(request_body: ChatRequest, request: Request) -> StreamingResponse:
    """
    Streaming chat endpoint using Server-Sent Events.

    Emits a stream of JSON-encoded events:
      {"type": "token",       "data": "<text fragment>"}
      {"type": "tool_call",   "data": {"name": ..., "args": ...}}
      {"type": "tool_result", "data": {"name": ..., "result": ...}}
      {"type": "complete",    "data": {"trace_id": ..., "tokens_used": ..., ...}}
      {"type": "error",       "data": "<error message>"}

    Keepalive comments (`: keepalive`) are emitted every 15 seconds.
    """
    _require_ready()
    _check_rate_limit(request_body.user_id)
    _check_cost_budget(request_body.user_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        last_keepalive = time.time()
        try:
            async for event in harness.process_stream(  # type: ignore[union-attr]
                user_input=request_body.message,
                user_id=request_body.user_id,
                session_id=request_body.session_id,
            ):
                # Keepalive comment to prevent proxy timeout
                now = time.time()
                if now - last_keepalive >= 15:
                    yield ": keepalive\n\n"
                    last_keepalive = now

                yield f"data: {json.dumps(event)}\n\n"

                # Record cost when stream completes
                if event.get("type") == "complete":
                    data = event.get("data", {})
                    cost = data.get("cost", 0.0)
                    tokens = data.get("tokens_used", 0)
                    cost_tracker.record(request_body.user_id, cost)  # type: ignore
                    metrics.record_request(  # type: ignore
                        time.time() - last_keepalive, tokens, cost
                    )

        except asyncio.CancelledError:
            # Client disconnected
            logger.info("Stream cancelled for user %s", request_body.user_id)
        except Exception as exc:
            logger.exception("Stream error for user %s: %s",
                             request_body.user_id, exc)
            yield f"data: {json.dumps({'type': 'error', 'data': 'Stream error.'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
            "X-Request-ID": getattr(request.state, "request_id", ""),
        },
    )


@app.get("/health")
async def health() -> Response:
    """
    Comprehensive health status endpoint.

    Returns HTTP 200 if healthy or degraded, HTTP 503 if unhealthy.
    """
    if not _startup_complete or harness is None:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "Service is initialising."},
        )

    harness_health = harness.get_health()
    cost_report = {
        "cost_today_usd": cost_tracker.total_today if cost_tracker else 0.0,
        "budget_pct": cost_tracker.budget_pct if cost_tracker else 0.0,
    }
    m = metrics
    error_rate = (
        m.error_count / max(m.request_count, 1) if m else 0.0
    )

    status_detail = {
        "status": harness_health.get("status", "unknown"),
        "llm_connectivity": harness_health.get("llm_connectivity", False),
        "vector_db_connectivity": harness_health.get("vector_db_connectivity", False),
        "circuit_breakers": harness_health.get("circuit_breakers", {}),
        "cost": cost_report,
        "error_rate": round(error_rate, 4),
        "active_alerts": (
            ["budget_critical"] if cost_report["budget_pct"] >= 0.9 else []
        ),
    }

    http_status = 200 if harness_health.get("status") == "healthy" else 503
    return JSONResponse(content=status_detail, status_code=http_status)


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus-format metrics endpoint."""
    if not metrics:
        return Response("# No metrics yet\n", media_type="text/plain; version=0.0.4")
    return Response(
        metrics.prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# Explicit readiness and liveness probes (separate from /health for k8s)
@app.get("/ready")
async def readiness() -> Response:
    """Readiness probe: is the service ready to accept traffic?"""
    if not _startup_complete:
        raise HTTPException(status_code=503, detail="Not ready.")
    return JSONResponse({"ready": True})


@app.get("/live")
async def liveness() -> Response:
    """Liveness probe: is the service still functioning?"""
    return JSONResponse({"alive": True})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(status_code=404, content={"detail": "Not found."})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> Response:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_service:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
        access_log=False,  # Handled by logging_middleware
    )
