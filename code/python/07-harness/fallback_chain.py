"""Multi-level LLM provider fallback chain with circuit breakers.

Implements Principle 3 of harness engineering: every LLM call needs a
fallback path. Providers are tried in priority order; failures trigger
automatic promotion to the next level.

Circuit breaker pattern:
    CLOSED  → OPEN (after threshold failures in window)
    OPEN    → HALF-OPEN (after cooldown expires)
    HALF-OPEN → CLOSED (on success) | OPEN (on failure)

See: docs/07-harness-engineering/01-the-harness-mindset.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types (mirror from harness_state_machine for standalone use)
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] | None
    tokens_used: int
    model: str
    provider_name: str
    fallback_level: int


class AllProvidersFailedError(Exception):
    """Raised when every provider in the chain has failed."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(
            f"All providers failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED    = "closed"      # Normal operation
    OPEN      = "open"        # Provider is skipped
    HALF_OPEN = "half_open"   # One test request allowed


class CircuitBreaker:
    """Per-provider circuit breaker.

    Opens after `threshold` failures within `window_seconds`.
    After `cooldown_seconds` in OPEN state, moves to HALF_OPEN.
    A success in HALF_OPEN closes the circuit; a failure reopens it.
    """

    def __init__(
        self,
        threshold: int = 5,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 120.0,
    ):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds

        self._state = CircuitState.CLOSED
        self._failure_timestamps: deque[float] = deque()
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0.0)
            if elapsed >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker → HALF_OPEN")
        return self._state

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        s = self.state
        return s in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_timestamps.clear()
            logger.info("Circuit breaker → CLOSED (success in HALF_OPEN)")

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failure_timestamps.append(now)

        # Prune old failures outside window
        cutoff = now - self.window_seconds
        while self._failure_timestamps and \
                self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = now
            logger.warning("Circuit breaker → OPEN (failure in HALF_OPEN)")
            return

        if len(self._failure_timestamps) >= self.threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now
            logger.warning(
                f"Circuit breaker → OPEN "
                f"({len(self._failure_timestamps)} failures in "
                f"{self.window_seconds}s window)"
            )

    def status(self) -> dict:
        return {
            "state":           self.state.value,
            "recent_failures": len(self._failure_timestamps),
            "threshold":       self.threshold,
            "opened_at":       self._opened_at,
        }


# ---------------------------------------------------------------------------
# Provider abstraction (mock + real interface)
# ---------------------------------------------------------------------------

class LLMProvider:
    """Abstract base — real providers implement chat_async."""

    name: str = "abstract"

    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Simulated provider for testing."""

    def __init__(
        self,
        name: str,
        *,
        fail_times: int = 0,
        timeout: bool = False,
        latency: float = 0.05,
        response: str | None = None,
    ):
        self.name = name
        self._fail_times = fail_times
        self._timeout = timeout
        self._latency = latency
        self._fixed_response = response
        self._call_count = 0

    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._call_count += 1

        if self._timeout:
            await asyncio.sleep(9999)

        if self._call_count <= self._fail_times:
            raise RuntimeError(f"{self.name}: simulated API failure "
                               f"(attempt {self._call_count})")

        await asyncio.sleep(self._latency)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        )
        content = (self._fixed_response or
                   f"[{self.name}] Response to: {last_user[:60]}")
        return LLMResponse(
            content=content,
            tool_calls=None,
            tokens_used=len(last_user.split()) + 30,
            model=self.name,
            provider_name=self.name,
            fallback_level=-1,  # Set by FallbackChain
        )


class StaticProvider(LLMProvider):
    """Last-resort provider — no LLM, just a static message."""

    name = "static-fallback"

    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                "Our AI service is temporarily unavailable. "
                "Please try again shortly or contact support."
            ),
            tool_calls=None,
            tokens_used=0,
            model="static",
            provider_name=self.name,
            fallback_level=-1,
        )


# ---------------------------------------------------------------------------
# Fallback level
# ---------------------------------------------------------------------------

@dataclass
class FallbackLevel:
    """A single link in the fallback chain."""

    provider: LLMProvider
    priority: int                          # Lower = tried first
    timeout_seconds: float = 30.0
    max_retries: int = 1
    circuit_breaker: CircuitBreaker = field(
        default_factory=CircuitBreaker
    )


# ---------------------------------------------------------------------------
# Fallback statistics
# ---------------------------------------------------------------------------

@dataclass
class _ProviderStats:
    successes: int = 0
    failures: int = 0
    failure_reasons: dict[str, int] = field(
        default_factory=lambda: defaultdict(int))


class FallbackStats:
    """Collect and expose fallback health data."""

    def __init__(self) -> None:
        self._data: dict[str, _ProviderStats] = defaultdict(_ProviderStats)
        self._level_successes: dict[int, int] = defaultdict(int)
        self._exhaustion_count: int = 0
        self._exhaustion_timestamps: deque[float] = deque(maxlen=100)

    def record_success(self, provider_name: str, level: int) -> None:
        self._data[provider_name].successes += 1
        self._level_successes[level] += 1

    def record_failure(self, provider_name: str, error: str) -> None:
        stats = self._data[provider_name]
        stats.failures += 1
        # Bucket the error
        if "timeout" in error.lower():
            bucket = "timeout"
        elif "rate" in error.lower() or "429" in error:
            bucket = "rate_limit"
        elif "connection" in error.lower() or "network" in error.lower():
            bucket = "connection"
        else:
            bucket = "api_error"
        stats.failure_reasons[bucket] += 1

    def record_exhaustion(self) -> None:
        self._exhaustion_count += 1
        self._exhaustion_timestamps.append(time.monotonic())

    def primary_success_rate(self) -> float:
        primary = next(iter(self._data.values()), None)
        if primary is None:
            return 1.0
        total = primary.successes + primary.failures
        return primary.successes / total if total else 1.0

    def fallback_activation_rate(self) -> float:
        total = sum(
            s.successes + s.failures for s in self._data.values()
        )
        if total == 0:
            return 0.0
        secondary_calls = sum(
            s.successes + s.failures
            for name, s in self._data.items()
            if name != next(iter(self._data), "")
        )
        return secondary_calls / total

    def exhaustions_last_minutes(self, minutes: int = 5) -> int:
        cutoff = time.monotonic() - minutes * 60
        return sum(1 for t in self._exhaustion_timestamps if t > cutoff)

    def get_health_report(self, levels: list[FallbackLevel]) -> dict:
        providers = {}
        for name, stats in self._data.items():
            total = stats.successes + stats.failures
            providers[name] = {
                "success_rate": (stats.successes / total if total else 1.0),
                "successes": stats.successes,
                "failures": stats.failures,
                "failure_reasons": dict(stats.failure_reasons),
            }

        circuit_breakers = {
            lvl.provider.name: lvl.circuit_breaker.status()
            for lvl in levels
        }

        return {
            "primary_success_rate": self.primary_success_rate(),
            "fallback_activation_rate": self.fallback_activation_rate(),
            "exhaustion_count": self._exhaustion_count,
            "exhaustions_last_5min": self.exhaustions_last_minutes(5),
            "providers": providers,
            "circuit_breakers": circuit_breakers,
            "level_success_distribution": dict(self._level_successes),
        }

    def should_alert(self) -> bool:
        if self.primary_success_rate() < 0.95:
            return True
        if self.exhaustions_last_minutes(5) > 0:
            return True
        return False


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class FallbackChain:
    """Chain of LLM providers with automatic failover and circuit breakers.

    Usage::

        chain = FallbackChain([
            FallbackLevel(MockProvider("gpt-4o"),          priority=0),
            FallbackLevel(MockProvider("claude-3-sonnet"), priority=1),
            FallbackLevel(MockProvider("gpt-4o-mini"),     priority=2),
            FallbackLevel(StaticProvider(),                priority=99),
        ])
        response = await chain.chat(messages)
    """

    def __init__(self, levels: list[FallbackLevel]) -> None:
        self.levels = sorted(levels, key=lambda l: l.priority)
        self.stats = FallbackStats()

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Try providers in priority order, falling back on any failure."""
        errors: list[str] = []
        prev_name: str | None = None

        for level_idx, level in enumerate(self.levels):
            provider = level.provider

            # Circuit breaker check
            if level.circuit_breaker.is_open():
                logger.warning(f"[FallbackChain] {provider.name}: "
                               f"circuit OPEN — skipping")
                errors.append(f"{provider.name}: circuit open (skipped)")
                continue

            if prev_name:
                logger.warning(
                    f"[FallbackChain] Falling back: "
                    f"{prev_name} → {provider.name}"
                )

            attempt = 0
            while attempt < max(level.max_retries, 1):
                attempt += 1
                try:
                    response = await asyncio.wait_for(
                        provider.chat_async(messages, tools, **kwargs),
                        timeout=level.timeout_seconds,
                    )
                    response.fallback_level = level_idx
                    level.circuit_breaker.record_success()
                    self.stats.record_success(provider.name, level_idx)
                    logger.info(
                        f"[FallbackChain] Success: {provider.name} "
                        f"(level {level_idx})"
                    )
                    return response

                except asyncio.TimeoutError:
                    err = f"{provider.name}: timeout after {level.timeout_seconds}s"
                    errors.append(err)
                    level.circuit_breaker.record_failure()
                    self.stats.record_failure(provider.name, "timeout")
                    logger.warning(f"[FallbackChain] {err}")
                    break  # Don't retry on timeout — try next provider

                except Exception as exc:  # noqa: BLE001
                    err = f"{provider.name}: {exc}"
                    errors.append(err)
                    level.circuit_breaker.record_failure()
                    self.stats.record_failure(provider.name, str(exc))
                    logger.warning(f"[FallbackChain] {err}")

                    # Exponential backoff before retry
                    if attempt < level.max_retries:
                        backoff = 2.0 ** (attempt - 1)
                        logger.info(
                            f"[FallbackChain] Retrying {provider.name} "
                            f"in {backoff:.1f}s (attempt {attempt+1}/"
                            f"{level.max_retries})"
                        )
                        await asyncio.sleep(backoff)

            prev_name = provider.name

        # All levels exhausted
        self.stats.record_exhaustion()
        raise AllProvidersFailedError(errors)

    def get_health_report(self) -> dict:
        return self.stats.get_health_report(self.levels)

    def circuit_breaker_status(self) -> dict[str, str]:
        return {
            lvl.provider.name: lvl.circuit_breaker.state.value
            for lvl in self.levels
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def _run_demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    print("=" * 65)
    print("FALLBACK CHAIN DEMO")
    print("=" * 65)

    # ── Scenario 1: Normal operation ──────────────────────────────────────
    print("\n--- Scenario 1: Normal operation ---")
    chain = FallbackChain([
        FallbackLevel(MockProvider("gpt-4o"),          priority=0, timeout_seconds=5),
        FallbackLevel(MockProvider("claude-3-sonnet"), priority=1, timeout_seconds=5),
        FallbackLevel(MockProvider("gpt-4o-mini"),     priority=2, timeout_seconds=5),
        FallbackLevel(StaticProvider(),                priority=99),
    ])
    messages = [{"role": "user", "content": "Hello, world!"}]
    response = await chain.chat(messages)
    print(f"  Response from: {response.provider_name} (level {response.fallback_level})")
    print(f"  Content: {response.content[:80]}")

    # ── Scenario 2: Primary fails, secondary succeeds ─────────────────────
    print("\n--- Scenario 2: Primary fails, secondary succeeds ---")
    chain2 = FallbackChain([
        FallbackLevel(MockProvider("gpt-4o", fail_times=99),  priority=0, timeout_seconds=5),
        FallbackLevel(MockProvider("claude-3-sonnet"),         priority=1, timeout_seconds=5),
        FallbackLevel(MockProvider("gpt-4o-mini"),             priority=2, timeout_seconds=5),
        FallbackLevel(StaticProvider(),                        priority=99),
    ])
    response2 = await chain2.chat(messages)
    print(f"  Response from: {response2.provider_name} (level {response2.fallback_level})")
    print(f"  Content: {response2.content[:80]}")

    # ── Scenario 3: All LLMs fail → static fallback ───────────────────────
    print("\n--- Scenario 3: All LLMs fail → static fallback ---")
    chain3 = FallbackChain([
        FallbackLevel(MockProvider("gpt-4o",          fail_times=99), priority=0, timeout_seconds=1),
        FallbackLevel(MockProvider("claude-3-sonnet", fail_times=99), priority=1, timeout_seconds=1),
        FallbackLevel(MockProvider("gpt-4o-mini",     fail_times=99), priority=2, timeout_seconds=1),
        FallbackLevel(StaticProvider(),                               priority=99),
    ])
    response3 = await chain3.chat(messages)
    print(f"  Response from: {response3.provider_name} (level {response3.fallback_level})")
    print(f"  Content: {response3.content}")

    # ── Scenario 4: Circuit breaker ───────────────────────────────────────
    print("\n--- Scenario 4: Circuit breaker opens after 3 failures ---")
    cb = CircuitBreaker(threshold=3, window_seconds=60, cooldown_seconds=1)
    failing_provider = MockProvider("gpt-4o", fail_times=99)
    level = FallbackLevel(failing_provider, priority=0, circuit_breaker=cb,
                          max_retries=1, timeout_seconds=1)
    backup = FallbackLevel(MockProvider("claude-3-sonnet"), priority=1,
                           timeout_seconds=5)
    chain4 = FallbackChain([level, backup])

    for i in range(4):
        try:
            r = await chain4.chat(messages)
            print(f"  Request {i+1}: success from {r.provider_name} "
                  f"(circuit: {cb.state.value})")
        except AllProvidersFailedError:
            print(f"  Request {i+1}: ALL FAILED")

    # ── Health report ─────────────────────────────────────────────────────
    print("\n--- Health Report (chain3) ---")
    report = chain3.get_health_report()
    print(json.dumps(report, indent=2))

    print("\n" + "=" * 65)
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(_run_demo())
