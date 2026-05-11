"""
Deployment Manager for AI Agents in Production
===============================================
Demonstrates: gradual rollout, canary deployment, multi-region routing,
cost control, rollback management, and deployment health checking.

Reference: docs/09-from-dev-to-production/01-deployment-strategies.md
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supporting stubs (replace with real harness / metrics in production)
# ---------------------------------------------------------------------------

class ProductionHarness:
    """Minimal stub so the module runs without a full harness implementation."""

    def __init__(self, version: str = "stable"):
        self.version = version

    async def process(self, user_input: str, *, user_id: str, **kwargs):
        await asyncio.sleep(0.01)
        return type("Response", (), {
            "content": f"[{self.version}] Response to: {user_input}",
            "metadata": {},
        })()


class HarnessMetrics:
    """Minimal stub for version-keyed metrics."""

    def __init__(self):
        self._data: dict[str, dict] = {}

    def record(self, version: str, *, error: bool = False,
               latency: float = 1.0, cost: float = 0.02,
               safety_blocked: bool = False, task_success: bool = True):
        m = self._data.setdefault(version, {
            "requests": 0, "errors": 0, "latencies": [],
            "costs": [], "safety_blocks": 0, "task_successes": 0,
        })
        m["requests"] += 1
        m["errors"] += int(error)
        m["latencies"].append(latency)
        m["costs"].append(cost)
        m["safety_blocks"] += int(safety_blocked)
        m["task_successes"] += int(task_success)

    def get_metrics_for_version(self, version: str, window_minutes: int = 60) -> dict:
        m = self._data.get(version, {})
        if not m or m["requests"] == 0:
            return {
                "error_rate": 0.0, "p50_latency": 1.0, "p95_latency": 2.0,
                "p99_latency": 3.0, "avg_cost": 0.02,
                "safety_block_rate": 0.0, "task_success_rate": 1.0,
                "user_satisfaction": 0.9,
            }
        lats = sorted(m["latencies"])
        n = len(lats)
        return {
            "error_rate": m["errors"] / m["requests"],
            "p50_latency": lats[int(n * 0.50)],
            "p95_latency": lats[int(n * 0.95)],
            "p99_latency": lats[int(n * 0.99)],
            "avg_cost": sum(m["costs"]) / len(m["costs"]),
            "safety_block_rate": m["safety_blocks"] / m["requests"],
            "task_success_rate": m["task_successes"] / m["requests"],
            "user_satisfaction": 0.85,
        }


class FeatureFlagService:
    """In-memory feature-flag service (replace with LaunchDarkly, etc.)."""

    def __init__(self):
        self._flags: dict[str, Any] = {}
        self._internal_users: set[str] = {"internal-001", "internal-002", "dev-team"}

    def is_internal_user(self, user_id: str) -> bool:
        return user_id in self._internal_users

    def get_string(self, key: str, default: str = "") -> str:
        return str(self._flags.get(key, default))

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self._flags.get(key, default))

    def set_int(self, key: str, value: int):
        self._flags[key] = value

    def set_string(self, key: str, value: str):
        self._flags[key] = value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CanaryEvaluation:
    canary_pct: int
    has_issues: bool
    issues: list[str]
    stable_metrics: dict
    canary_metrics: dict
    recommendation: str


@dataclass
class RollbackItem:
    name: str
    description: str
    method: Callable
    time_seconds: int


@dataclass
class RollbackItemResult:
    name: str
    success: bool
    error: str | None = None


@dataclass
class RollbackResult:
    reason: str
    items: list[RollbackItemResult]
    total_time_seconds: int
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class BudgetCheck:
    allowed: bool
    reason: str | None = None
    current_user_cost: float = 0.0
    user_budget: float = 0.0


@dataclass
class CostConfig:
    user_daily_budget: float = 10.0
    total_daily_budget: float = 1000.0
    max_cost_per_request: float = 1.0
    free_tier_daily_budget: float = 0.50
    enterprise_daily_budget: float = 50.0


@dataclass
class HealthReport:
    overall: str  # "healthy" | "degraded" | "unhealthy"
    checks: dict[str, bool]
    details: dict[str, str]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 1. Deployment Manager
# ---------------------------------------------------------------------------

class DeploymentManager:
    """
    Manages gradual rollout of new agent versions.

    Rollout stages:
      Stage 0: Internal only (0% external)
      Stage 1: 1% canary (monitor 24h)
      Stage 2: 5% extended canary (monitor 48h)
      Stage 3: 25% beta (monitor 72h)
      Stage 4: 100% full rollout
    """

    ROLLOUT_STAGES = [0, 1, 5, 25, 100]

    def __init__(self, feature_flag_service: FeatureFlagService):
        self.flags = feature_flag_service
        # Default: no external rollout
        self.flags.set_int("canary_rollout_pct", 0)
        self.flags.set_string("stable_version", "v3.1.0")
        self.flags.set_string("canary_version", "v3.2.1")
        self.flags.set_string("internal_version", "v3.2.1")
        self._halted: bool = False
        self._halt_reason: str | None = None

    def get_agent_version(self, user_id: str) -> str:
        """
        Determine which agent version a user should receive.

        Resolution order:
          1. Internal users always get the internal (latest) version.
          2. If rollout is halted, everyone gets stable.
          3. Otherwise, hash user_id to determine canary bucket.
        """
        if self.flags.is_internal_user(user_id):
            return self.flags.get_string("internal_version", "v3.2.1")

        if self._halted:
            return self.flags.get_string("stable_version", "v3.1.0")

        rollout_pct = self.flags.get_int("canary_rollout_pct", 0)
        if self._user_in_rollout_group(user_id, rollout_pct):
            return self.flags.get_string("canary_version", "v3.2.1")

        return self.flags.get_string("stable_version", "v3.1.0")

    def _user_in_rollout_group(self, user_id: str, percentage: int) -> bool:
        """Deterministic assignment: MD5(user_id) mod 100 < percentage."""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        return (hash_value % 100) < percentage

    def promote_rollout(self, from_pct: int, to_pct: int) -> None:
        """Increase canary rollout from from_pct to to_pct."""
        if to_pct not in self.ROLLOUT_STAGES:
            raise ValueError(f"to_pct must be one of {self.ROLLOUT_STAGES}")
        current = self.flags.get_int("canary_rollout_pct", 0)
        if current != from_pct:
            raise ValueError(f"Current rollout is {current}%, not {from_pct}%")
        canary = self.flags.get_string("canary_version")
        logger.info("Promoting %s rollout: %d%% → %d%%", canary, from_pct, to_pct)
        self.flags.set_int("canary_rollout_pct", to_pct)
        self._halted = False

    def halt_rollout(self, reason: str) -> None:
        """Immediately route all external users back to stable."""
        logger.warning("ROLLOUT HALTED: %s", reason)
        self._halted = True
        self._halt_reason = reason
        self.flags.set_int("canary_rollout_pct", 0)

    def get_rollout_status(self) -> dict:
        """Current rollout state."""
        return {
            "stable_version": self.flags.get_string("stable_version"),
            "canary_version": self.flags.get_string("canary_version"),
            "internal_version": self.flags.get_string("internal_version"),
            "canary_rollout_pct": self.flags.get_int("canary_rollout_pct"),
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }


# ---------------------------------------------------------------------------
# 2. Canary Deployer
# ---------------------------------------------------------------------------

class CanaryDeployer:
    """
    Routes requests to stable or canary harness and evaluates canary health.
    """

    def __init__(self, stable_harness: ProductionHarness,
                 canary_harness: ProductionHarness,
                 metrics: HarnessMetrics,
                 deployment_manager: DeploymentManager):
        self.stable = stable_harness
        self.canary = canary_harness
        self.metrics = metrics
        self.dm = deployment_manager

    async def process(self, user_input: str, user_id: str, **kwargs):
        """Route to canary or stable based on current rollout."""
        version = self.dm.get_agent_version(user_id)
        canary_version = self.dm.flags.get_string("canary_version")

        if version == canary_version:
            response = await self.canary.process(user_input, user_id=user_id, **kwargs)
            response.metadata["version"] = "canary"
            self.metrics.record("canary", latency=1.0, cost=0.025)
        else:
            response = await self.stable.process(user_input, user_id=user_id, **kwargs)
            response.metadata["version"] = "stable"
            self.metrics.record("stable", latency=0.9, cost=0.020)

        return response

    def evaluate_canary(self) -> CanaryEvaluation:
        """Compare canary vs stable across 5 dimensions."""
        stable = self.metrics.get_metrics_for_version("stable", window_minutes=60)
        canary = self.metrics.get_metrics_for_version("canary", window_minutes=60)
        pct = self.dm.flags.get_int("canary_rollout_pct")

        issues: list[str] = []

        if not self._check_error_rate(stable, canary):
            issues.append(
                f"Error rate: stable={stable['error_rate']:.2%}, "
                f"canary={canary['error_rate']:.2%}"
            )
        if not self._check_latency(stable, canary):
            issues.append(
                f"P95 latency: stable={stable['p95_latency']:.1f}s, "
                f"canary={canary['p95_latency']:.1f}s"
            )
        if not self._check_cost(stable, canary):
            issues.append(
                f"Avg cost: stable=${stable['avg_cost']:.3f}, "
                f"canary={canary['avg_cost']:.3f}"
            )
        if not self._check_safety(stable, canary):
            issues.append(
                f"Safety block rate: stable={stable['safety_block_rate']:.2%}, "
                f"canary={canary['safety_block_rate']:.2%}"
            )
        if not self._check_task_success(stable, canary):
            issues.append(
                f"Task success rate: stable={stable['task_success_rate']:.2%}, "
                f"canary={canary['task_success_rate']:.2%}"
            )

        return CanaryEvaluation(
            canary_pct=pct,
            has_issues=len(issues) > 0,
            issues=issues,
            stable_metrics=stable,
            canary_metrics=canary,
            recommendation=self._generate_recommendation(issues),
        )

    def _check_error_rate(self, stable: dict, canary: dict) -> bool:
        """Canary error rate must be ≤ 1.5× stable."""
        if stable["error_rate"] == 0:
            return canary["error_rate"] == 0
        return canary["error_rate"] <= stable["error_rate"] * 1.5

    def _check_latency(self, stable: dict, canary: dict) -> bool:
        """Canary P95 latency must be ≤ 1.2× stable."""
        return canary["p95_latency"] <= stable["p95_latency"] * 1.2

    def _check_cost(self, stable: dict, canary: dict) -> bool:
        """Canary avg cost must be ≤ 1.2× stable."""
        return canary["avg_cost"] <= stable["avg_cost"] * 1.2

    def _check_safety(self, stable: dict, canary: dict) -> bool:
        """Canary safety block rate must be ≤ 1.5× stable."""
        if stable["safety_block_rate"] == 0:
            return canary["safety_block_rate"] == 0
        return canary["safety_block_rate"] <= stable["safety_block_rate"] * 1.5

    def _check_task_success(self, stable: dict, canary: dict) -> bool:
        """Canary task success rate must be ≥ 0.95× stable."""
        return canary["task_success_rate"] >= stable["task_success_rate"] * 0.95

    def _generate_recommendation(self, issues: list[str]) -> str:
        if not issues:
            return "Canary is healthy. Consider increasing rollout percentage."
        if len(issues) == 1:
            return "Minor issues detected. Monitor for another hour before promoting."
        return "Significant issues detected. Halt rollout and investigate."


# ---------------------------------------------------------------------------
# 3. Streaming Deployment
# ---------------------------------------------------------------------------

class StreamingDeployment:
    """
    Configuration and helpers for streaming (SSE) AI responses.
    """

    def __init__(self):
        self.max_concurrent_streams: int = 1000
        self.stream_timeout_seconds: int = 300
        self.keepalive_interval_seconds: int = 15
        self.disable_proxy_buffering: bool = True
        self.disable_compression: bool = True
        self.max_queue_size: int = 100
        self.max_tokens_per_stream: int = 4096
        self.cost_limit_per_stream: float = 0.50

        self._active_streams: dict[str, float] = {}   # conn_id → start_time
        self._dropped_connections: int = 0
        self._completed_connections: int = 0

    def configure_streaming_response(self, response: Any) -> dict:
        """
        Return the headers dict required for a streaming SSE response.

        In a FastAPI/Starlette app, pass these as the `headers` argument of
        StreamingResponse.
        """
        return {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
            "Transfer-Encoding": "chunked",
        }

    async def handle_backpressure(self, queue: asyncio.Queue, max_size: int) -> None:
        """
        Slow the producer when the consumer queue is near capacity.
        Waits until there is space before returning to the caller.
        """
        while queue.qsize() >= max_size:
            await asyncio.sleep(0.05)

    def handle_disconnect(self, connection_id: str) -> None:
        """Remove a disconnected stream from the active set."""
        if connection_id in self._active_streams:
            del self._active_streams[connection_id]
            self._dropped_connections += 1
            logger.info("Stream disconnected: %s", connection_id)

    def register_stream(self, connection_id: str) -> None:
        """Register a new active stream."""
        self._active_streams[connection_id] = time.time()

    def complete_stream(self, connection_id: str) -> None:
        """Mark a stream as cleanly completed."""
        if connection_id in self._active_streams:
            del self._active_streams[connection_id]
            self._completed_connections += 1

    def get_streaming_metrics(self) -> dict:
        """Current streaming health snapshot."""
        now = time.time()
        durations = [now - start for start in self._active_streams.values()]
        return {
            "active_streams": len(self._active_streams),
            "dropped_connections": self._dropped_connections,
            "completed_connections": self._completed_connections,
            "avg_stream_duration_seconds": (
                sum(durations) / len(durations) if durations else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# 4. Production Cost Controller
# ---------------------------------------------------------------------------

class ProductionCostController:
    """
    Enforce per-user and total daily budget limits.

    Alert thresholds:
      warning  → 70%
      critical → 90%
      shutdown → 100%
    """

    def __init__(self, config: CostConfig):
        self.config = config
        self._user_daily_costs: dict[str, float] = {}
        self._total_daily_cost: float = 0.0
        self._free_tier_users: set[str] = set()
        self._enterprise_users: set[str] = set()
        self._alert_thresholds = {
            "warning": 0.70,
            "critical": 0.90,
            "shutdown": 1.00,
        }
        self._last_alert_level: str | None = None

    # -- Registration helpers ------------------------------------------------

    def register_free_tier(self, user_id: str) -> None:
        self._free_tier_users.add(user_id)

    def register_enterprise(self, user_id: str) -> None:
        self._enterprise_users.add(user_id)

    # -- Budget helpers ------------------------------------------------------

    def _user_budget(self, user_id: str) -> float:
        if user_id in self._enterprise_users:
            return self.config.enterprise_daily_budget
        if user_id in self._free_tier_users:
            return self.config.free_tier_daily_budget
        return self.config.user_daily_budget

    def _check_user_budget(self, user_id: str, estimated: float) -> bool:
        current = self._user_daily_costs.get(user_id, 0.0)
        return current + estimated <= self._user_budget(user_id)

    def _check_total_budget(self, estimated: float) -> bool:
        return self._total_daily_cost + estimated <= self.config.total_daily_budget

    def _check_request_limit(self, estimated: float) -> bool:
        return estimated <= self.config.max_cost_per_request

    # -- Public interface -----------------------------------------------------

    def check_budget(self, user_id: str, estimated_cost: float) -> BudgetCheck:
        """Pre-request budget check. Returns immediately if any limit is exceeded."""
        current_user = self._user_daily_costs.get(user_id, 0.0)
        user_budget = self._user_budget(user_id)

        if not self._check_request_limit(estimated_cost):
            return BudgetCheck(
                allowed=False,
                reason=(
                    f"Request estimated cost ${estimated_cost:.3f} exceeds "
                    f"per-request limit ${self.config.max_cost_per_request:.2f}."
                ),
                current_user_cost=current_user,
                user_budget=user_budget,
            )

        if not self._check_user_budget(user_id, estimated_cost):
            return BudgetCheck(
                allowed=False,
                reason=(
                    f"Daily budget of ${user_budget:.2f} exceeded. "
                    f"Current: ${current_user:.2f}."
                ),
                current_user_cost=current_user,
                user_budget=user_budget,
            )

        if not self._check_total_budget(estimated_cost):
            return BudgetCheck(
                allowed=False,
                reason="Service temporarily unavailable due to high demand. "
                       "Please try again later.",
                current_user_cost=current_user,
                user_budget=user_budget,
            )

        return BudgetCheck(
            allowed=True,
            current_user_cost=current_user,
            user_budget=user_budget,
        )

    def record_cost(self, user_id: str, cost: float) -> None:
        """Record actual cost after request completion and fire alerts."""
        self._user_daily_costs[user_id] = (
            self._user_daily_costs.get(user_id, 0.0) + cost
        )
        self._total_daily_cost += cost
        self._check_and_fire_alerts()

    def _check_and_fire_alerts(self) -> None:
        budget = self.config.total_daily_budget
        if budget <= 0:
            return
        pct = self._total_daily_cost / budget
        if pct >= self._alert_thresholds["shutdown"]:
            self._trigger_alert("critical",
                f"Daily budget exhausted: ${self._total_daily_cost:.2f}")
        elif pct >= self._alert_thresholds["critical"]:
            self._trigger_alert("warning",
                f"Daily budget at {pct:.0%}: ${self._total_daily_cost:.2f}")
        elif pct >= self._alert_thresholds["warning"]:
            self._trigger_alert("info",
                f"Daily budget at {pct:.0%}: ${self._total_daily_cost:.2f}")

    def _trigger_alert(self, level: str, message: str) -> None:
        if level != self._last_alert_level:
            logger.warning("[COST ALERT %s] %s", level.upper(), message)
            self._last_alert_level = level

    def get_cost_report(self) -> dict:
        """Daily cost summary."""
        budget = self.config.total_daily_budget or 1.0
        top_users = sorted(
            self._user_daily_costs.items(), key=lambda x: x[1], reverse=True
        )[:10]
        user_count = len(self._user_daily_costs)
        return {
            "total_daily_cost": self._total_daily_cost,
            "daily_budget": self.config.total_daily_budget,
            "budget_remaining": self.config.total_daily_budget - self._total_daily_cost,
            "pct_used": self._total_daily_cost / budget,
            "top_users": top_users,
            "user_count": user_count,
            "avg_cost_per_user": (
                self._total_daily_cost / max(user_count, 1)
            ),
        }

    def reset_daily_costs(self) -> None:
        """Called at midnight to reset daily counters."""
        self._user_daily_costs.clear()
        self._total_daily_cost = 0.0
        self._last_alert_level = None
        logger.info("Daily costs reset.")


# ---------------------------------------------------------------------------
# 5. Multi-Region Deployer
# ---------------------------------------------------------------------------

class MultiRegionDeployer:
    """
    Route users to the nearest healthy region, respecting data-residency rules.
    """

    REGIONS: dict[str, dict] = {
        "us-east": {
            "llm_provider": "openai",
            "fallback_provider": "anthropic",
            "vector_db_endpoint": "https://us-east.qdrant.example.com",
            "latency_to_provider_ms": 50,
            "eu_residency_compliant": False,
        },
        "eu-west": {
            "llm_provider": "openai",
            "fallback_provider": "anthropic",
            "vector_db_endpoint": "https://eu-west.qdrant.example.com",
            "latency_to_provider_ms": 80,
            "eu_residency_compliant": True,
        },
        "ap-southeast": {
            "llm_provider": "anthropic",
            "fallback_provider": "openai",
            "vector_db_endpoint": "https://ap-se.qdrant.example.com",
            "latency_to_provider_ms": 120,
            "eu_residency_compliant": False,
        },
    }

    # Coarse IP prefix → region mapping (illustrative; use MaxMind in production)
    _GEO_MAP: dict[str, str] = {
        "52.": "us-east",   # AWS US East
        "18.": "us-east",   # AWS US East
        "34.": "us-east",   # GCP US
        "35.": "eu-west",   # GCP EU
        "54.239.": "eu-west",
        "13.": "ap-southeast",
    }

    # EU CIDR prefixes (simplified; use MaxMind / ip2country in production)
    _EU_PREFIXES = {"195.", "212.", "217.", "82.", "185.", "37.", "31."}

    def __init__(self):
        self._circuit_breakers: dict[str, bool] = {r: False for r in self.REGIONS}

    def get_region(self, user_ip: str, user_preferences: dict | None = None) -> str:
        """Determine the best region for a user request."""
        if user_preferences and "region" in user_preferences:
            preferred = user_preferences["region"]
            if preferred in self.REGIONS and self._is_region_healthy(preferred):
                return preferred

        if self._is_eu_user(user_ip):
            region = "eu-west"
        else:
            region = self._geo_route(user_ip)

        if not self._is_region_healthy(region):
            return self._get_nearest_healthy_region(region)

        return region

    def _is_eu_user(self, ip: str) -> bool:
        """Coarse GDPR residency check by IP prefix."""
        return any(ip.startswith(prefix) for prefix in self._EU_PREFIXES)

    def _geo_route(self, ip: str) -> str:
        """Route to nearest geographic region by IP prefix."""
        for prefix, region in self._GEO_MAP.items():
            if ip.startswith(prefix):
                return region
        # Default to us-east for unknown IPs
        return "us-east"

    def _is_region_healthy(self, region: str) -> bool:
        """Return False if the circuit breaker for this region is open."""
        return not self._circuit_breakers.get(region, False)

    def _get_nearest_healthy_region(self, region: str) -> str:
        """Return the healthy region with the lowest LLM latency."""
        candidates = [
            (r, cfg["latency_to_provider_ms"])
            for r, cfg in self.REGIONS.items()
            if r != region and self._is_region_healthy(r)
        ]
        if not candidates:
            raise RuntimeError("No healthy regions available")
        return min(candidates, key=lambda x: x[1])[0]

    def get_region_config(self, region: str) -> dict:
        """Return the configuration dict for a region."""
        if region not in self.REGIONS:
            raise ValueError(f"Unknown region: {region}")
        return dict(self.REGIONS[region])

    def open_circuit_breaker(self, region: str) -> None:
        """Mark a region as unhealthy (circuit breaker open)."""
        logger.warning("Circuit breaker OPEN for region: %s", region)
        self._circuit_breakers[region] = True

    def close_circuit_breaker(self, region: str) -> None:
        """Mark a region as healthy again."""
        logger.info("Circuit breaker CLOSED for region: %s", region)
        self._circuit_breakers[region] = False


# ---------------------------------------------------------------------------
# 6. Rollback Manager
# ---------------------------------------------------------------------------

class RollbackManager:
    """
    Manage rollbacks for AI agent deployments.

    A full rollback may include: code, model, prompt, config, tools, documents.
    """

    def __init__(self, deployment_manager: DeploymentManager):
        self.dm = deployment_manager
        self._history: list[RollbackResult] = []
        self._rollback_items: list[RollbackItem] = []
        self._setup_items()

    def _setup_items(self) -> None:
        self._rollback_items = [
            RollbackItem(
                name="config",
                description="Revert harness configuration to previous version",
                method=self._rollback_config,
                time_seconds=10,
            ),
            RollbackItem(
                name="prompt",
                description="Revert system prompt to previous version",
                method=self._rollback_prompt,
                time_seconds=10,
            ),
            RollbackItem(
                name="model",
                description="Switch to previous model version",
                method=self._rollback_model,
                time_seconds=30,
            ),
            RollbackItem(
                name="tools",
                description="Revert tool definitions/implementations",
                method=self._rollback_tools,
                time_seconds=30,
            ),
            RollbackItem(
                name="code",
                description="Revert application code to previous git commit",
                method=self._rollback_code,
                time_seconds=60,
            ),
            RollbackItem(
                name="documents",
                description="Revert knowledge base and re-embed",
                method=self._rollback_documents,
                time_seconds=300,
            ),
        ]

    async def rollback(self, reason: str,
                       items: list[str] | None = None) -> RollbackResult:
        """
        Execute a rollback.

        Args:
            reason: Why are we rolling back?
            items:  Specific item names to rollback. None means rollback all.
        """
        logger.warning("ROLLBACK INITIATED: %s", reason)

        # First: halt rollout so no new users hit the bad canary
        self.dm.halt_rollout(reason)

        candidates = self._rollback_items
        if items is not None:
            names = set(items)
            candidates = [r for r in self._rollback_items if r.name in names]

        # Execute in ascending time order (fastest first)
        candidates_sorted = sorted(candidates, key=lambda r: r.time_seconds)
        results: list[RollbackItemResult] = []
        for item in candidates_sorted:
            logger.info("Rolling back: %s …", item.name)
            try:
                await item.method()
                results.append(RollbackItemResult(name=item.name, success=True))
                logger.info("Rollback complete: %s", item.name)
            except Exception as exc:  # noqa: BLE001
                results.append(RollbackItemResult(
                    name=item.name, success=False, error=str(exc)
                ))
                logger.error("Rollback failed: %s: %s", item.name, exc)

        total_time = sum(r.time_seconds for r in candidates_sorted)
        result = RollbackResult(
            reason=reason,
            items=results,
            total_time_seconds=total_time,
            success=all(r.success for r in results),
        )
        self._history.append(result)
        return result

    # -- Item implementations (stubs; wire to real systems) ------------------

    async def _rollback_code(self) -> None:
        """Simulate git revert and re-deploy."""
        logger.info("[stub] git revert HEAD && git push && kubectl rollout restart …")
        await asyncio.sleep(0.01)

    async def _rollback_model(self) -> None:
        """Switch model config back to previous version."""
        logger.info("[stub] Reverting model config to gpt-4o from gpt-4o-2024-11-20 …")
        await asyncio.sleep(0.01)

    async def _rollback_prompt(self) -> None:
        """Revert system prompt from the prompt library."""
        logger.info("[stub] Loading previous prompt version from library …")
        await asyncio.sleep(0.01)

    async def _rollback_config(self) -> None:
        """Revert harness configuration."""
        logger.info("[stub] Restoring harness config from previous snapshot …")
        await asyncio.sleep(0.01)

    async def _rollback_tools(self) -> None:
        """Revert tool definitions."""
        logger.info("[stub] Restoring tool definitions from previous release …")
        await asyncio.sleep(0.01)

    async def _rollback_documents(self) -> None:
        """Revert knowledge base and trigger re-embedding."""
        logger.info("[stub] Restoring previous document snapshot and re-embedding …")
        await asyncio.sleep(0.01)

    # -- History & dry-run ---------------------------------------------------

    def get_rollback_history(self) -> list[RollbackResult]:
        return list(self._history)

    async def test_rollback(self) -> bool:
        """
        Dry-run: verify each rollback method is callable without side effects.
        Returns True if all stubs complete without error.
        """
        logger.info("DRY-RUN: testing all rollback methods …")
        try:
            for item in self._rollback_items:
                await item.method()
            logger.info("DRY-RUN: all rollback methods succeeded.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("DRY-RUN: rollback test failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# 7. Deployment Health Checker
# ---------------------------------------------------------------------------

class DeploymentHealthChecker:
    """
    Aggregates health checks into a single HealthReport.

    overall == "healthy"   → all checks pass
    overall == "degraded"  → 1–2 non-critical checks fail
    overall == "unhealthy" → critical checks fail or 3+ checks fail
    """

    CRITICAL_CHECKS = {"llm_connectivity", "error_rate"}

    def __init__(self, cost_controller: ProductionCostController,
                 multi_region: MultiRegionDeployer):
        self.cost_controller = cost_controller
        self.multi_region = multi_region
        self._circuit_breaker_states: dict[str, str] = {}

    def check_all(self) -> HealthReport:
        """Run all health checks and return an aggregated HealthReport."""
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        checks["llm_connectivity"], details["llm_connectivity"] = (
            self._run("llm_connectivity", self.check_llm_connectivity)
        )
        checks["vector_db_connectivity"], details["vector_db_connectivity"] = (
            self._run("vector_db_connectivity", self.check_vector_db_connectivity)
        )
        tool_status = self.check_tool_connectivity()
        checks["tool_connectivity"] = all(tool_status.values())
        details["tool_connectivity"] = str(tool_status)

        checks["cost_within_budget"], details["cost_within_budget"] = (
            self._run("cost_within_budget", self.check_cost_within_budget)
        )
        checks["error_rate"], details["error_rate"] = (
            self._run("error_rate", self.check_error_rate)
        )
        checks["latency"], details["latency"] = (
            self._run("latency", self.check_latency)
        )
        cb_status = self.check_circuit_breakers()
        checks["circuit_breakers"] = all(
            s != "open" for s in cb_status.values()
        )
        details["circuit_breakers"] = str(cb_status)

        failing = [k for k, v in checks.items() if not v]
        critical_failing = [k for k in failing if k in self.CRITICAL_CHECKS]

        if critical_failing or len(failing) >= 3:
            overall = "unhealthy"
        elif failing:
            overall = "degraded"
        else:
            overall = "healthy"

        return HealthReport(overall=overall, checks=checks, details=details)

    def _run(self, name: str, fn: Callable[[], bool]) -> tuple[bool, str]:
        try:
            result = fn()
            return result, "ok" if result else "failed"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def check_llm_connectivity(self) -> bool:
        """Verify all configured LLM providers are reachable (stub)."""
        # In production: send a minimal probe request to each provider
        return True

    def check_vector_db_connectivity(self) -> bool:
        """Verify the vector database is queryable (stub)."""
        return True

    def check_tool_connectivity(self) -> dict[str, bool]:
        """Check each external tool (stub)."""
        return {"search_api": True, "weather_api": True, "order_api": True}

    def check_cost_within_budget(self) -> bool:
        """Fail if we have already exhausted the daily budget."""
        report = self.cost_controller.get_cost_report()
        return report["pct_used"] < 1.0

    def check_error_rate(self) -> bool:
        """Return False if the error rate exceeds 5% (stub)."""
        return True  # Replace with real error rate from metrics store

    def check_latency(self) -> bool:
        """Return False if P95 latency exceeds SLA (stub)."""
        return True  # Replace with real P95 from metrics store

    def check_circuit_breakers(self) -> dict[str, str]:
        """Return the state of all circuit breakers."""
        return {
            region: ("open" if open_ else "closed")
            for region, open_ in self.multi_region._circuit_breakers.items()
        }


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

async def run_demo() -> None:
    print("\n" + "=" * 60)
    print("  AI Agent Deployment Manager — Demo")
    print("=" * 60)

    # ── Setup ───────────────────────────────────────────────────────────────
    flags = FeatureFlagService()
    dm = DeploymentManager(flags)
    metrics = HarnessMetrics()
    stable_harness = ProductionHarness("v3.1.0")
    canary_harness = ProductionHarness("v3.2.1")
    deployer = CanaryDeployer(stable_harness, canary_harness, metrics, dm)
    cost_cfg = CostConfig()
    cost_ctrl = ProductionCostController(cost_cfg)
    multi_region = MultiRegionDeployer()
    rollback_mgr = RollbackManager(dm)
    health_checker = DeploymentHealthChecker(cost_ctrl, multi_region)
    streaming = StreamingDeployment()

    # ── Gradual rollout ──────────────────────────────────────────────────────
    print("\n--- GRADUAL ROLLOUT ---")

    # Stage 0: internal only
    print(f"\n[Stage 0] Status: {dm.get_rollout_status()}")
    print(f"  internal-001 → {dm.get_agent_version('internal-001')}")
    print(f"  external-user-42 → {dm.get_agent_version('external-user-42')}")

    # Stage 1: 1% canary
    dm.promote_rollout(0, 1)
    print(f"\n[Stage 1 — 1% canary] Status: {dm.get_rollout_status()}")

    # Simulate traffic
    for i in range(20):
        await deployer.process("Hello", user_id=f"user-{i:04d}")

    eval_result = deployer.evaluate_canary()
    print(f"  Canary eval: issues={eval_result.issues}, "
          f"recommendation='{eval_result.recommendation}'")

    # Stage 2: 5% canary
    dm.promote_rollout(1, 5)
    print(f"\n[Stage 2 — 5% canary] Status: {dm.get_rollout_status()}")

    # Simulate a spike in the canary error rate
    metrics.record("canary", error=True, latency=3.5, cost=0.04)
    metrics.record("canary", error=True, latency=3.8, cost=0.04)
    eval_result2 = deployer.evaluate_canary()
    print(f"  Canary eval: has_issues={eval_result2.has_issues}")
    print(f"  Issues: {eval_result2.issues}")
    print(f"  Recommendation: {eval_result2.recommendation}")

    if eval_result2.has_issues:
        print("\n  ⚠ Issues detected — halting rollout …")
        dm.halt_rollout("Error rate spike in canary at 5%")
        print(f"  Status after halt: {dm.get_rollout_status()}")

    # ── Rollback ────────────────────────────────────────────────────────────
    print("\n--- ROLLBACK ---")
    rb_result = await rollback_mgr.rollback(
        reason="Error rate exceeded 1.5× stable threshold",
        items=["config", "prompt"],
    )
    print(f"  Rollback success: {rb_result.success}")
    print(f"  Items processed: {[r.name for r in rb_result.items]}")
    print(f"  Estimated time: {rb_result.total_time_seconds}s")

    # ── Multi-region routing ─────────────────────────────────────────────────
    print("\n--- MULTI-REGION ROUTING ---")
    test_ips = [
        ("52.86.1.1",   "US user"),
        ("195.50.10.1", "EU user (GDPR)"),
        ("13.250.1.1",  "AP user"),
    ]
    for ip, label in test_ips:
        region = multi_region.get_region(ip)
        cfg = multi_region.get_region_config(region)
        print(f"  {label} ({ip}) → {region} "
              f"(provider: {cfg['llm_provider']}, "
              f"latency: {cfg['latency_to_provider_ms']}ms)")

    # Unhealthy region fallback
    multi_region.open_circuit_breaker("us-east")
    fallback = multi_region.get_region("52.86.1.1")
    print(f"  US user with us-east down → {fallback}")
    multi_region.close_circuit_breaker("us-east")

    # ── Cost controller ──────────────────────────────────────────────────────
    print("\n--- COST CONTROLLER ---")
    cost_ctrl.register_free_tier("free-user-1")

    for user, estimated in [
        ("premium-user-1", 0.03),
        ("free-user-1",    0.60),  # exceeds free tier limit of $0.50
        ("premium-user-1", 0.03),
    ]:
        check = cost_ctrl.check_budget(user, estimated)
        status = "✓ allowed" if check.allowed else f"✗ rejected: {check.reason}"
        print(f"  {user} (${estimated:.2f}) → {status}")
        if check.allowed:
            cost_ctrl.record_cost(user, estimated)

    print(f"\n  Cost report: {cost_ctrl.get_cost_report()}")

    # ── Streaming deployment ─────────────────────────────────────────────────
    print("\n--- STREAMING DEPLOYMENT ---")
    headers = streaming.configure_streaming_response(None)
    print(f"  SSE headers: {headers}")
    streaming.register_stream("conn-001")
    streaming.register_stream("conn-002")
    streaming.handle_disconnect("conn-001")
    print(f"  Streaming metrics: {streaming.get_streaming_metrics()}")

    # ── Health check ─────────────────────────────────────────────────────────
    print("\n--- DEPLOYMENT HEALTH CHECK ---")
    report = health_checker.check_all()
    print(f"  Overall: {report.overall}")
    for check, passed in report.checks.items():
        symbol = "✓" if passed else "✗"
        print(f"    {symbol} {check}: {report.details.get(check, '')}")

    # ── Rollout playbook ──────────────────────────────────────────────────────
    print("\n--- ROLLOUT PLAYBOOK ---")
    playbook = """
  Stage 0: Internal only      → internal team, run eval suite (Day 0)
  Stage 1: 1%  canary         → monitor 24h, check error/latency/cost/safety
  Stage 2: 5%  extended canary → monitor 48h, run A/B on task success rate
  Stage 3: 25% beta           → monitor 72h, collect user feedback
  Stage 4: 100% full rollout  → monitor 1 week, keep previous version warm

  ROLLBACK TRIGGERS:
    • Error rate > 2× baseline
    • Safety block rate > 3× baseline
    • Cost per request > 2× baseline
    • User complaints > 50% increase
    • Critical security vulnerability

  KEY RULE: Never deploy to 100% on Friday. Gradual rollout saves weekends.
"""
    print(playbook)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demo())
