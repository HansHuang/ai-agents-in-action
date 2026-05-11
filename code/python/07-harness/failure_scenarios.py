"""
failure_scenarios.py — Simulates realistic failure scenarios for the resilience layer.

Demonstrates all 10 failure/recovery paths that production AI agents encounter:

  1.  Transient failure (auto-recovers via retry)
  2.  Rate limiting (respects server Retry-After header)
  3.  Persistent failure with successful fallback
  4.  Persistent failure exhausting all LLM fallbacks → static
  5.  Complete system failure (SystemUnavailableError)
  6.  Circuit breaker opening and recovery
  7.  Circuit breaker re-opening (service not yet recovered)
  8.  Graceful degradation path (full → reduced → static → error)
  9.  Concurrent requests during outage
  10. Recovery after extended outage

Run: python failure_scenarios.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from resilience_layer import (
    AllFallbacksExhausted,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    FallbackError,
    FallbackExecutor,
    FallbackLevel,
    MaxRetriesExceeded,
    RateLimitAwareRetry,
    RateLimitError,
    ResilienceLayer,
    ResilienceResult,
    RetryConfig,
    SystemUnavailableError,
    retry_with_backoff,
    retry_with_rate_limit_awareness,
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,          # Suppress INFO noise during demo
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

_DIVIDER = "─" * 68


def header(title: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {title}")
    print(f"{'═' * 68}")


def section(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def timeline(ts: float, message: str) -> None:
    """Print a timeline entry relative to *ts* (monotonic start time)."""
    elapsed = time.monotonic() - ts
    print(f"  [{elapsed:5.2f}s]  {message}")


# ---------------------------------------------------------------------------
# Shared mock provider factory
# ---------------------------------------------------------------------------

@dataclass
class MockResponse:
    """Simulated LLM response."""
    content: str
    model: str
    capability: str = "full"


def make_provider(
    name: str,
    fail_times: int = 0,
    fail_with: type[Exception] = ConnectionError,
    response_text: str = "mock response",
    latency: float = 0.0,
    rate_limit_retry_after: float | None = None,
) -> Any:
    """
    Return an async callable that simulates a provider.

    Args:
        name:                     Provider name for logging.
        fail_times:               How many calls fail before succeeding.
        fail_with:                Exception type to raise on failure.
        response_text:            Content returned on success.
        latency:                  Simulated response latency in seconds.
        rate_limit_retry_after:   If set, failing calls return a RateLimitError
                                  with this Retry-After value.
    """
    call_count = 0

    class FakeResponse:
        headers: dict[str, str] = {}

    async def provider(*args: Any, **kwargs: Any) -> MockResponse:
        nonlocal call_count
        call_count += 1
        if latency:
            await asyncio.sleep(latency)
        if call_count <= fail_times:
            if rate_limit_retry_after is not None:
                resp = FakeResponse()
                resp.headers = {"retry-after": str(rate_limit_retry_after)}
                raise RateLimitError(f"429 from {name}", response=resp)
            raise fail_with(f"Simulated {fail_with.__name__} from {name}")
        return MockResponse(content=response_text, model=name)

    return provider


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------

async def scenario_1_transient_failure() -> None:
    """
    Transient failure: operation fails once, succeeds on first retry.
    Mechanism: RetryConfig.  User impact: none (delay < 100ms).
    """
    section("Scenario 1 — Transient Failure (auto-recovers via retry)")

    ts = time.monotonic()
    provider = make_provider("gpt-4o", fail_times=1, fail_with=TimeoutError,
                             response_text="The weather in Tokyo is 22 °C.")

    try:
        result = await retry_with_backoff(
            provider,
            config=RetryConfig(max_retries=3, base_delay_seconds=0.05),
        )
        timeline(ts, f"✓ Succeeded — {result.content!r}")
        print("  Mechanism : RetryConfig (exponential backoff)")
        print("  User impact: none — retry was transparent")
    except MaxRetriesExceeded as exc:
        timeline(ts, f"✗ Exhausted: {exc}")


async def scenario_2_rate_limiting() -> None:
    """
    Rate limiting: provider returns 429 with Retry-After: 0.1s.
    RateLimitAwareRetry respects the header; standard backoff would differ.
    """
    section("Scenario 2 — Rate Limiting (respects Retry-After header)")

    ts = time.monotonic()
    provider = make_provider(
        "gpt-4o",
        fail_times=1,
        rate_limit_retry_after=0.1,
        response_text="Rate limit cleared.",
    )

    config = RateLimitAwareRetry(max_retries=3, base_delay_seconds=1.0)
    try:
        result = await retry_with_rate_limit_awareness(provider, config=config)
        elapsed = time.monotonic() - ts
        timeline(ts, f"✓ Succeeded after {elapsed:.2f}s — {result.content!r}")
        print("  Mechanism : RateLimitAwareRetry (used server Retry-After=0.1s)")
        print("  Note      : Standard backoff would have waited 1.0s (base_delay)")
    except MaxRetriesExceeded as exc:
        timeline(ts, f"✗ Exhausted: {exc}")


async def scenario_3_persistent_failure_fallback() -> None:
    """
    Primary fails 3 times (503) → secondary provider succeeds.
    """
    section("Scenario 3 — Persistent Failure → Successful Fallback")

    ts = time.monotonic()
    primary = make_provider("gpt-4o", fail_times=99, response_text="never reached")
    secondary = make_provider("claude-sonnet", fail_times=0,
                              response_text="Here is your answer (via claude-sonnet).")

    layer = ResilienceLayer(
        name="llm_call",
        circuit_breaker=CircuitBreaker("openai", failure_threshold=20),
        retry_config=RetryConfig(
            max_retries=2, base_delay_seconds=0.02,
            retryable_exceptions=(ConnectionError,),
        ),
        fallback_executor=FallbackExecutor([
            FallbackLevel("claude-sonnet", secondary, 30, "full", 1.2),
        ]),
    )

    try:
        result = await layer.execute(primary)
        timeline(ts, f"✓ {result.result.content!r}")
        print(f"  Path      : {result.path}")
        print("  Mechanism : FallbackExecutor (cross-provider)")
        print("  User impact: slight latency increase; full capability maintained")
    except SystemUnavailableError as exc:
        timeline(ts, f"✗ Unavailable: {exc}")


async def scenario_4_all_llm_fallbacks_exhausted() -> None:
    """
    All LLM providers fail → static response is returned.
    """
    section("Scenario 4 — All LLM Providers Fail → Static Response")

    ts = time.monotonic()

    async def static_provider(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse(
            content="Our AI assistant is temporarily unavailable. "
                    "Please try again in a few minutes.",
            model="static",
            capability="static",
        )

    layer = ResilienceLayer(
        name="llm_call",
        circuit_breaker=CircuitBreaker("openai", failure_threshold=20),
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        fallback_executor=FallbackExecutor([
            FallbackLevel("gpt-4o", make_provider("gpt-4o", fail_times=99), 5, "full"),
            FallbackLevel("claude", make_provider("claude", fail_times=99), 5, "full"),
            FallbackLevel("gpt-mini", make_provider("gpt-mini", fail_times=99), 5, "reduced"),
            FallbackLevel("static", static_provider, 2, "static", 0.0),
        ]),
    )

    try:
        result = await layer.execute(make_provider("primary", fail_times=99))
        timeline(ts, f"✓ Degraded response: {result.result.content!r}")
        print(f"  Path      : {result.path}")
        print("  Mechanism : FallbackExecutor — exhausted all LLM levels, used static")
        print("  User impact: degraded (static message), but no hard error")
    except SystemUnavailableError as exc:
        timeline(ts, f"✗ Unavailable: {exc}")


async def scenario_5_complete_system_failure() -> None:
    """
    Primary and all fallbacks fail → SystemUnavailableError.
    """
    section("Scenario 5 — Complete System Failure (all paths exhausted)")

    ts = time.monotonic()

    layer = ResilienceLayer(
        name="llm_call",
        circuit_breaker=CircuitBreaker("openai", failure_threshold=20),
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        fallback_executor=FallbackExecutor([
            FallbackLevel("fallback-a", make_provider("fa", fail_times=99), 2, "full"),
            FallbackLevel("fallback-b", make_provider("fb", fail_times=99), 2, "reduced"),
        ]),
    )

    try:
        await layer.execute(make_provider("primary", fail_times=99))
    except SystemUnavailableError as exc:
        timeline(ts, "✗ SystemUnavailableError raised (expected)")
        print(f"  Message   : {exc}")
        print("  Mechanism : All retries and fallbacks exhausted")
        print("  User impact: Hard error; on-call alerted; show maintenance page")


async def scenario_6_circuit_breaker_open_and_close() -> None:
    """
    Circuit opens after threshold, tests half-open, closes on success.
    """
    section("Scenario 6 — Circuit Breaker: CLOSED → OPEN → HALF_OPEN → CLOSED")

    ts = time.monotonic()

    cb = CircuitBreaker(
        "openai",
        failure_threshold=3,
        recovery_timeout_seconds=0.15,
        failure_window_seconds=10.0,
    )
    failing = make_provider("gpt-4o", fail_times=99)
    healthy = make_provider("gpt-4o", fail_times=0, response_text="All good now.")

    timeline(ts, f"Initial state: {cb.state.value}")

    # Trip the breaker
    for i in range(3):
        try:
            await cb.call(failing)
        except (ConnectionError, CircuitBreakerOpenError):
            timeline(ts, f"Failure {i + 1}/3 recorded  state={cb.state.value}")

    # Verify open
    try:
        await cb.call(healthy)
    except CircuitBreakerOpenError:
        timeline(ts, f"Request rejected (circuit OPEN)  state={cb.state.value}")

    # Wait for recovery timeout
    await asyncio.sleep(0.2)
    cb._maybe_transition()
    timeline(ts, f"After recovery timeout           state={cb.state.value}")

    # Probe request succeeds → close
    result = await cb.call(healthy)
    timeline(ts, f"Probe succeeded → circuit CLOSED  result={result.content!r}")

    stats = cb.get_stats()
    print(f"\n  Circuit stats: successes={stats['total_successes']}  "
          f"failures={stats['total_failures']}  rejected={stats['total_rejected']}  "
          f"times_opened={stats['times_opened']}")


async def scenario_7_circuit_breaker_stays_open() -> None:
    """
    Circuit opens, recovery test fails, circuit re-opens immediately.
    """
    section("Scenario 7 — Circuit Breaker: re-opens when probe fails")

    ts = time.monotonic()

    cb = CircuitBreaker(
        "openai",
        failure_threshold=3,
        recovery_timeout_seconds=0.1,
        failure_window_seconds=10.0,
    )
    still_failing = make_provider("gpt-4o", fail_times=99)

    # Trip the breaker
    for _ in range(3):
        try:
            await cb.call(still_failing)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    timeline(ts, f"Breaker opened: {cb.state.value}")

    # Wait for recovery
    await asyncio.sleep(0.15)
    cb._maybe_transition()
    timeline(ts, f"Transitioned to: {cb.state.value}")

    # Probe fails → back to OPEN
    try:
        await cb.call(still_failing)
    except (ConnectionError, CircuitBreakerOpenError):
        pass

    timeline(ts, f"Probe failed → re-opened: {cb.state.value}  "
              f"times_opened={cb.times_opened}")

    print("  Mechanism : HALF_OPEN probe failure re-triggers OPEN state")
    print("  User impact: Fast-fail continues; no wasted capacity on broken service")


async def scenario_8_graceful_degradation() -> None:
    """
    Degrade through full → reduced → static, showing capability at each step.
    """
    section("Scenario 8 — Graceful Degradation (full → reduced → static)")

    ts = time.monotonic()

    async def static_resp(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse(
            content="[Cached] Basic AI response (service degraded).",
            model="static",
            capability="static",
        )

    # Providers: first two fail, mini succeeds with "reduced" capability.
    executor = FallbackExecutor([
        FallbackLevel("gpt-4o",    make_provider("gpt-4o",    fail_times=99), 2, "full"),
        FallbackLevel("claude",    make_provider("claude",    fail_times=99), 2, "full"),
        FallbackLevel("gpt-mini",  make_provider("gpt-mini",  fail_times=0,
                                   response_text="Reduced-capability answer."),
                      5, "reduced"),
        FallbackLevel("static",    static_resp,                               2, "static"),
    ])

    primary = make_provider("primary", fail_times=99)
    try:
        fallback_result = await executor.execute(
            "llm_call",
            lambda level: level.provider(),
        )
        timeline(ts, f"✓ Served by level {fallback_result.level_used} "
                     f"({fallback_result.level_name}) "
                     f"capability={fallback_result.capability!r}")
        print(f"  Content   : {fallback_result.result.content!r}")
        print(f"  Errors    : {[e.level_name for e in fallback_result.errors]}")
        print("  Mechanism : FallbackExecutor — stopped at first working level")
    except AllFallbacksExhausted as exc:
        timeline(ts, f"✗ All exhausted: {exc}")


async def scenario_9_concurrent_requests_during_outage() -> None:
    """
    10 concurrent requests while primary is down.
    Circuit breaker limits how many reach the failing service.
    """
    section("Scenario 9 — Concurrent Requests During Outage")

    ts = time.monotonic()
    successful = 0
    rejected_by_cb = 0
    served_by_fallback = 0

    fallback_provider = make_provider(
        "fallback", fail_times=0, response_text="fallback answer"
    )

    cb = CircuitBreaker("openai", failure_threshold=3, failure_window_seconds=10.0)
    config = RetryConfig(max_retries=1, base_delay_seconds=0.01,
                        retryable_exceptions=(ConnectionError,))

    # Each coroutine represents one concurrent user request.
    async def one_request(i: int) -> str:
        nonlocal successful, rejected_by_cb, served_by_fallback
        failing_primary = make_provider("gpt-4o", fail_times=99)

        try:
            await cb.call(retry_with_backoff, failing_primary, config=config)
            successful += 1
            return f"request-{i}: primary"
        except CircuitBreakerOpenError:
            rejected_by_cb += 1
            # Fast-fail to fallback
            result = await fallback_provider()
            served_by_fallback += 1
            return f"request-{i}: circuit-rejected → fallback"
        except MaxRetriesExceeded:
            result = await fallback_provider()
            served_by_fallback += 1
            return f"request-{i}: exhausted → fallback"

    results = await asyncio.gather(*[one_request(i) for i in range(10)])

    for r in results:
        timeline(ts, r)

    print(f"\n  Summary: primary_success={successful}  "
          f"circuit_rejected={rejected_by_cb}  "
          f"fallback_served={served_by_fallback}")
    print(f"  Circuit state: {cb.state.value}  "
          f"times_opened={cb.times_opened}")
    print("  Mechanism : CircuitBreaker prevented thundering herd on failing service")


async def scenario_10_recovery_after_outage() -> None:
    """
    Primary was down, circuit opens, then service recovers.
    Traffic returns to primary after the probe succeeds.
    """
    section("Scenario 10 — Recovery After Outage (traffic returns to primary)")

    ts = time.monotonic()

    # Simulate 5-minute outage compressed to 200ms for the demo.
    recovery_at = time.monotonic() + 0.2
    call_count = 0

    async def primary_with_recovery(*args: Any, **kwargs: Any) -> MockResponse:
        nonlocal call_count
        call_count += 1
        if time.monotonic() < recovery_at:
            raise ConnectionError("Service unavailable (simulated outage)")
        return MockResponse(content="Service recovered!", model="gpt-4o")

    fallback_provider = make_provider(
        "claude", fail_times=0, response_text="Fallback during outage."
    )

    cb = CircuitBreaker(
        "openai",
        failure_threshold=3,
        recovery_timeout_seconds=0.15,
        failure_window_seconds=30.0,
    )
    config = RetryConfig(max_retries=0, retryable_exceptions=(ConnectionError,))

    # Phase 1: Trip the breaker
    timeline(ts, "Phase 1: Trip the circuit breaker")
    for _ in range(3):
        try:
            await cb.call(primary_with_recovery)
        except (ConnectionError, CircuitBreakerOpenError):
            pass
    timeline(ts, f"Breaker state: {cb.state.value}")

    # Phase 2: Requests go to fallback while circuit is open
    timeline(ts, "Phase 2: Requests fast-fail to fallback")
    for i in range(3):
        try:
            await cb.call(primary_with_recovery)
        except CircuitBreakerOpenError:
            result = await fallback_provider()
            timeline(ts, f"Request {i + 1} → fallback: {result.content!r}")

    # Phase 3: Wait for recovery timeout + outage to end
    await asyncio.sleep(0.25)
    cb._maybe_transition()
    timeline(ts, f"Phase 3: After recovery timeout  state={cb.state.value}")

    # Phase 4: Probe request — service has recovered
    result = await cb.call(primary_with_recovery)
    timeline(ts, f"Probe succeeded: {result.content!r}  state={cb.state.value}")

    # Phase 5: Traffic returns to primary
    timeline(ts, "Phase 4: Traffic returns to primary")
    for i in range(3):
        result = await cb.call(primary_with_recovery)
        timeline(ts, f"Request {i + 1} → primary: {result.content!r}")

    stats = cb.get_stats()
    print(f"\n  Final stats: successes={stats['total_successes']}  "
          f"failures={stats['total_failures']}  "
          f"rejected={stats['total_rejected']}")
    print("  Mechanism : CircuitBreaker half-open probe + automatic recovery")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    header("FAILURE SCENARIO SIMULATOR — AI Agent Resilience Layer")
    print("Each scenario shows: trigger → mechanism → user impact\n")

    await scenario_1_transient_failure()
    await scenario_2_rate_limiting()
    await scenario_3_persistent_failure_fallback()
    await scenario_4_all_llm_fallbacks_exhausted()
    await scenario_5_complete_system_failure()
    await scenario_6_circuit_breaker_open_and_close()
    await scenario_7_circuit_breaker_stays_open()
    await scenario_8_graceful_degradation()
    await scenario_9_concurrent_requests_during_outage()
    await scenario_10_recovery_after_outage()

    print(f"\n{'═' * 68}")
    print("  All 10 scenarios completed.")
    print(f"{'═' * 68}\n")


if __name__ == "__main__":
    asyncio.run(main())
