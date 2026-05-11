"""
test_resilience.py — pytest suite for the complete resilience system.

34 tests covering:
  - Retry: backoff math, jitter, non-retryable errors, rate-limit headers
  - Fallback: chain ordering, capability levels, stats, exhaustion
  - CircuitBreaker: state machine, rolling window, fast-fail
  - ResilienceLayer: primary / retry / fallback / circuit-open paths
  - Integration: end-to-end with simulated failures
  - Monitoring: alert generation

Run: pytest test_resilience.py -v
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from resilience_layer import (
    AllFallbacksExhausted,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    FallbackError,
    FallbackExecutor,
    FallbackLevel,
    FallbackResult,
    MaxRetriesExceeded,
    NonRetryableError,
    RateLimitAwareRetry,
    RateLimitError,
    ResilienceLayer,
    ResilienceMonitor,
    ResilienceResult,
    RetryConfig,
    SystemUnavailableError,
    calculate_delay,
    retry_with_backoff,
    retry_with_rate_limit_awareness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_operation(
    *,
    fail_times: int = 0,
    fail_with: type[Exception] = ConnectionError,
    return_value: Any = "ok",
) -> AsyncMock:
    """Return an async mock that fails *fail_times* times then succeeds."""
    call_count = 0

    async def _op(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_times:
            raise fail_with("simulated failure")
        return return_value

    mock = AsyncMock(side_effect=_op)
    return mock


def make_fallback_level(
    name: str = "level",
    fail: bool = False,
    capability: str = "full",
    cost: float = 1.0,
) -> FallbackLevel:
    async def provider(*args: Any, **kwargs: Any) -> str:
        if fail:
            raise ConnectionError(f"{name} failed")
        return f"response from {name}"

    return FallbackLevel(
        name=name,
        provider=provider,
        timeout_seconds=5.0,
        capability=capability,
        cost_multiplier=cost,
    )


def fast_retry(max_retries: int = 3) -> RetryConfig:
    """RetryConfig with near-zero delays for fast tests."""
    return RetryConfig(
        max_retries=max_retries,
        base_delay_seconds=0.001,
        max_delay_seconds=0.01,
        backoff_multiplier=2.0,
        jitter=False,
        retryable_exceptions=(TimeoutError, ConnectionError, RateLimitError),
        total_deadline_seconds=60.0,
    )


# ---------------------------------------------------------------------------
# RETRY TESTS (1–8)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt():
    """1. Operation succeeds immediately — no retries needed."""
    op = make_operation(return_value="hello")
    result = await retry_with_backoff(op, config=fast_retry())
    assert result == "hello"
    assert op.call_count == 1


@pytest.mark.asyncio
async def test_retry_after_one_failure():
    """2. Fails once, succeeds on the first retry."""
    op = make_operation(fail_times=1, fail_with=ConnectionError, return_value="recovered")
    result = await retry_with_backoff(op, config=fast_retry(max_retries=3))
    assert result == "recovered"
    assert op.call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted():
    """3. Fails all attempts → MaxRetriesExceeded."""
    op = make_operation(fail_times=99)
    with pytest.raises(MaxRetriesExceeded):
        await retry_with_backoff(op, config=fast_retry(max_retries=2))
    assert op.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_non_retryable_error_not_retried():
    """4. A 400-type (ValueError) error raises NonRetryableError immediately."""
    async def raises_value_error(*a: Any, **k: Any) -> None:
        raise ValueError("bad request")

    with pytest.raises(NonRetryableError):
        await retry_with_backoff(raises_value_error, config=fast_retry())


def test_exponential_backoff_increases():
    """5. Delay grows exponentially: ~1s, ~2s, ~4s, ~8s (without jitter)."""
    cfg = RetryConfig(
        base_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_delay_seconds=100.0,
        jitter=False,
    )
    delays = [calculate_delay(i, cfg) for i in range(4)]
    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(2.0)
    assert delays[2] == pytest.approx(4.0)
    assert delays[3] == pytest.approx(8.0)


def test_jitter_adds_randomness():
    """6. With jitter enabled, computed delays should vary."""
    cfg = RetryConfig(
        base_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_delay_seconds=100.0,
        jitter=True,
        jitter_factor=0.2,
    )
    delays = {calculate_delay(1, cfg) for _ in range(20)}
    assert len(delays) > 1, "All delays were identical — jitter is not working"


@pytest.mark.asyncio
async def test_total_deadline_exceeded():
    """7. Retry stops before max_retries when total deadline is exceeded."""
    cfg = RetryConfig(
        max_retries=10,
        base_delay_seconds=0.001,
        max_delay_seconds=1.0,
        jitter=False,
        retryable_exceptions=(ConnectionError,),
        total_deadline_seconds=0.001,  # Extremely tight deadline
    )

    async def always_fails(*a: Any, **k: Any) -> None:
        await asyncio.sleep(0.002)  # exceeds deadline
        raise ConnectionError("timeout simulation")

    with pytest.raises(MaxRetriesExceeded):
        await retry_with_backoff(always_fails, config=cfg)


@pytest.mark.asyncio
async def test_rate_limit_respects_retry_after():
    """8. RateLimitAwareRetry uses Retry-After=0.05s, not the base_delay of 1s."""

    class FakeResponse:
        headers = {"retry-after": "0.05"}

    call_count = 0

    async def rate_limited(*a: Any, **k: Any) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("429", response=FakeResponse())
        return "success"

    cfg = RateLimitAwareRetry(max_retries=2, base_delay_seconds=1.0)

    start = time.monotonic()
    result = await retry_with_rate_limit_awareness(rate_limited, config=cfg)
    elapsed = time.monotonic() - start

    assert result == "success"
    # Should wait ~0.05s not ~1.0s
    assert elapsed < 0.5, f"Waited {elapsed:.2f}s — should have used Retry-After header"
    assert call_count == 2


# ---------------------------------------------------------------------------
# FALLBACK TESTS (9–14)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_primary_succeeds():
    """9. Primary works → uses primary, secondary never called."""
    primary_provider = AsyncMock(return_value="primary response")
    secondary_mock = AsyncMock(return_value="secondary response")

    primary_level = FallbackLevel("primary", primary_provider, 5.0, "full", 1.0)
    secondary_level = FallbackLevel("secondary", secondary_mock, 5.0, "full", 1.0)

    executor = FallbackExecutor([primary_level, secondary_level])
    result = await executor.execute("test", lambda level: level.provider())

    assert result.result == "primary response"
    assert result.level_used == 0
    secondary_mock.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_secondary_used():
    """10. Primary fails, secondary works → uses secondary."""
    executor = FallbackExecutor([
        make_fallback_level("primary", fail=True),
        make_fallback_level("secondary", fail=False),
    ])
    result = await executor.execute("test", lambda level: level.provider())

    assert result.level_used == 1
    assert result.level_name == "secondary"
    assert "secondary" in result.result


@pytest.mark.asyncio
async def test_fallback_capability_decreases():
    """11. Level 0=full, Level 1=full, Level 2=reduced — first working wins."""
    executor = FallbackExecutor([
        make_fallback_level("l0", fail=True, capability="full"),
        make_fallback_level("l1", fail=True, capability="full"),
        make_fallback_level("l2", fail=False, capability="reduced"),
    ])
    result = await executor.execute("test", lambda level: level.provider())
    assert result.capability == "reduced"
    assert result.level_used == 2


@pytest.mark.asyncio
async def test_fallback_all_exhausted():
    """12. All levels fail → AllFallbacksExhausted."""
    executor = FallbackExecutor([
        make_fallback_level("a", fail=True),
        make_fallback_level("b", fail=True),
    ])
    with pytest.raises(AllFallbacksExhausted) as exc_info:
        await executor.execute("test", lambda level: level.provider())
    assert len(exc_info.value.errors) == 2


@pytest.mark.asyncio
async def test_fallback_stats_accurate():
    """13. Stats correctly track successes by level and exhaustion count."""
    executor = FallbackExecutor([
        make_fallback_level("primary", fail=True),
        make_fallback_level("secondary", fail=False),
    ])
    await executor.execute("op", lambda level: level.provider())

    assert executor.stats.success_by_level.get("secondary") == 1
    assert executor.stats.failure_by_level.get("primary") == 1
    assert executor.stats.exhaustion_count == 0


@pytest.mark.asyncio
async def test_fallback_result_includes_path():
    """14. FallbackResult.level_used and level_name reflect which level served it."""
    executor = FallbackExecutor([
        make_fallback_level("first", fail=True),
        make_fallback_level("second", fail=False),
    ])
    result = await executor.execute("test", lambda level: level.provider())
    assert result.level_used == 1
    assert result.level_name == "second"
    assert len(result.errors) == 1  # first level's failure recorded


# ---------------------------------------------------------------------------
# CIRCUIT BREAKER TESTS (15–23)
# ---------------------------------------------------------------------------

def test_circuit_closed_initially():
    """15. Newly created circuit breaker is CLOSED."""
    cb = CircuitBreaker("test")
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold():
    """16. 5 failures within the window → circuit is OPEN."""
    cb = CircuitBreaker("test", failure_threshold=5, failure_window_seconds=60.0)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    for _ in range(5):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_rejects_when_open():
    """17. OPEN circuit raises CircuitBreakerOpenError immediately."""
    cb = CircuitBreaker("test", failure_threshold=3, failure_window_seconds=60.0)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    for _ in range(3):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(fail)


@pytest.mark.asyncio
async def test_circuit_half_open_after_timeout():
    """18. After recovery timeout, OPEN transitions to HALF_OPEN."""
    cb = CircuitBreaker("test", failure_threshold=3,
                        recovery_timeout_seconds=0.05, failure_window_seconds=60.0)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    for _ in range(3):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    await asyncio.sleep(0.1)
    cb._maybe_transition()
    assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_circuit_test_request_succeeds():
    """19. HALF_OPEN: probe succeeds → circuit closes."""
    cb = CircuitBreaker("test", failure_threshold=3,
                        recovery_timeout_seconds=0.05, failure_window_seconds=60.0)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    async def succeed(*a: Any, **k: Any) -> str:
        return "ok"

    for _ in range(3):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    await asyncio.sleep(0.1)
    cb._maybe_transition()
    assert cb.state == CircuitState.HALF_OPEN

    await cb.call(succeed)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_test_request_fails():
    """20. HALF_OPEN: probe fails → circuit re-opens."""
    cb = CircuitBreaker("test", failure_threshold=3,
                        recovery_timeout_seconds=0.05, failure_window_seconds=60.0)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    for _ in range(3):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    await asyncio.sleep(0.1)
    cb._maybe_transition()

    try:
        await cb.call(fail)
    except (ConnectionError, CircuitBreakerOpenError):
        pass

    assert cb.state == CircuitState.OPEN
    assert cb.times_opened == 2


@pytest.mark.asyncio
async def test_circuit_counts_rejected():
    """21. Rejected requests are tracked separately from failures."""
    cb = CircuitBreaker("test", failure_threshold=3, failure_window_seconds=60.0)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    for _ in range(3):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    for _ in range(4):
        try:
            await cb.call(fail)
        except CircuitBreakerOpenError:
            pass

    stats = cb.get_stats()
    assert stats["total_rejected"] == 4
    assert stats["total_failures"] == 3  # Only the ones that actually ran


@pytest.mark.asyncio
async def test_circuit_rolling_window():
    """22. Failures outside the window don't count toward the threshold."""
    cb = CircuitBreaker("test", failure_threshold=5, failure_window_seconds=0.05)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    # Generate 4 failures — not enough to trip
    for _ in range(4):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    # Wait for them to expire
    await asyncio.sleep(0.1)

    # One more failure — but window is empty, so threshold not reached
    try:
        await cb.call(fail)
    except (ConnectionError, CircuitBreakerOpenError):
        pass

    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_fast_fail_when_open():
    """23. Open circuit rejects without calling the operation (fast-fail)."""
    cb = CircuitBreaker("test", failure_threshold=3, failure_window_seconds=60.0)

    async def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("fail")

    for _ in range(3):
        try:
            await cb.call(fail)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    assert cb.state == CircuitState.OPEN

    operation_called = False

    async def probe(*a: Any, **k: Any) -> None:
        nonlocal operation_called
        operation_called = True

    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(probe)

    assert not operation_called, "Operation should not have been called when circuit is OPEN"


# ---------------------------------------------------------------------------
# RESILIENCE LAYER TESTS (24–29)
# ---------------------------------------------------------------------------

def _make_layer(
    primary_fail_times: int = 0,
    fallback_levels: list[FallbackLevel] | None = None,
    max_retries: int = 2,
    cb_threshold: int = 10,
) -> tuple[ResilienceLayer, AsyncMock]:
    op = make_operation(fail_times=primary_fail_times, fail_with=ConnectionError)
    layer = ResilienceLayer(
        name="test",
        circuit_breaker=CircuitBreaker("test-cb", failure_threshold=cb_threshold,
                                       failure_window_seconds=60.0),
        retry_config=fast_retry(max_retries=max_retries),
        fallback_executor=FallbackExecutor(fallback_levels or []),
    )
    return layer, op


@pytest.mark.asyncio
async def test_resilience_primary_path():
    """24. All healthy → uses primary, path='primary'."""
    layer, op = _make_layer(primary_fail_times=0)
    result = await layer.execute(op)
    assert result.path == "primary"
    assert result.result == "ok"


@pytest.mark.asyncio
async def test_resilience_retry_path():
    """25. Transient failure → retry succeeds, still path='primary'."""
    layer, op = _make_layer(primary_fail_times=1)
    result = await layer.execute(op)
    assert result.path == "primary"
    assert op.call_count == 2


@pytest.mark.asyncio
async def test_resilience_fallback_path():
    """26. Retry exhausted → fallback used, path='fallback_level_0'."""
    layer, op = _make_layer(
        primary_fail_times=99,
        fallback_levels=[make_fallback_level("fallback", fail=False)],
        max_retries=1,
    )
    result = await layer.execute(op)
    assert result.path == "fallback_level_0"
    assert "fallback" in result.result


@pytest.mark.asyncio
async def test_resilience_circuit_open_path():
    """27. Circuit open → request goes directly to fallback chain."""
    layer, op = _make_layer(
        primary_fail_times=99,
        fallback_levels=[make_fallback_level("fallback", fail=False)],
        max_retries=2,
        cb_threshold=2,
    )

    # Trip the circuit
    for _ in range(2):
        try:
            await layer.execute(op)
        except SystemUnavailableError:
            pass

    # Reset op call count to measure next call
    op.reset_mock()
    # Force open state
    layer.circuit_breaker.state = CircuitState.OPEN

    result = await layer.execute(op)
    # Circuit was open, so operation should not have been called
    assert op.call_count == 0
    assert result.path.startswith("fallback_level_")


@pytest.mark.asyncio
async def test_resilience_all_exhausted():
    """28. Everything fails → SystemUnavailableError."""
    layer, op = _make_layer(
        primary_fail_times=99,
        fallback_levels=[make_fallback_level("fallback", fail=True)],
        max_retries=1,
    )
    with pytest.raises(SystemUnavailableError):
        await layer.execute(op)


@pytest.mark.asyncio
async def test_resilience_result_path_indicates_route():
    """29. ResilienceResult.path correctly reflects primary vs fallback_level_N."""
    # Primary
    layer1, op1 = _make_layer(primary_fail_times=0)
    r1 = await layer1.execute(op1)
    assert r1.path == "primary"

    # Fallback level 0
    layer2, op2 = _make_layer(
        primary_fail_times=99,
        fallback_levels=[make_fallback_level("fb0", fail=False)],
        max_retries=1,
    )
    r2 = await layer2.execute(op2)
    assert r2.path == "fallback_level_0"


# ---------------------------------------------------------------------------
# INTEGRATION TESTS (30–31)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_to_end_with_transient_failure():
    """30. Full harness: one transient failure → retry → user gets response."""
    call_count = 0

    async def real_llm_call(*a: Any, **k: Any) -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("transient timeout")
        return {"message": "Here is your answer."}

    layer = ResilienceLayer(
        name="llm",
        circuit_breaker=CircuitBreaker("openai", failure_threshold=5,
                                       failure_window_seconds=60.0),
        retry_config=RetryConfig(
            max_retries=3,
            base_delay_seconds=0.001,
            max_delay_seconds=0.01,
            retryable_exceptions=(TimeoutError, ConnectionError),
        ),
        fallback_executor=FallbackExecutor([]),
    )

    result = await layer.execute(real_llm_call)
    assert result.result["message"] == "Here is your answer."
    assert call_count == 2


@pytest.mark.asyncio
async def test_end_to_end_with_complete_outage():
    """31. Full harness: all providers down → SystemUnavailableError."""
    async def failing_primary(*a: Any, **k: Any) -> None:
        raise ConnectionError("provider down")

    async def failing_fallback(*a: Any, **k: Any) -> None:
        raise ConnectionError("fallback also down")

    layer = ResilienceLayer(
        name="llm",
        circuit_breaker=CircuitBreaker("openai", failure_threshold=10,
                                       failure_window_seconds=60.0),
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.001,
                                 retryable_exceptions=(ConnectionError,)),
        fallback_executor=FallbackExecutor([
            FallbackLevel("fallback", failing_fallback, 5.0, "static"),
        ]),
    )

    with pytest.raises(SystemUnavailableError) as exc_info:
        await layer.execute(failing_primary)

    assert exc_info.value.fallback_errors  # At least one fallback error recorded


# ---------------------------------------------------------------------------
# MONITORING TESTS (32–34)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_monitor_alerts_on_circuit_open():
    """32. Circuit open → CRITICAL alert generated."""
    layer, op = _make_layer(primary_fail_times=99, cb_threshold=2, max_retries=0)
    monitor = ResilienceMonitor(layer)

    # Force the circuit open
    layer.circuit_breaker.state = CircuitState.OPEN

    health = monitor.check_health()
    assert any("CRITICAL" in a and "OPEN" in a for a in health["alerts"]), \
        f"Expected CRITICAL alert, got: {health['alerts']}"


@pytest.mark.asyncio
async def test_monitor_alerts_on_low_primary_rate():
    """33. Primary success rate < 95% → WARNING alert generated."""
    layer, op = _make_layer(primary_fail_times=0)
    monitor = ResilienceMonitor(layer)

    # Simulate a low primary success rate by manipulating stats directly
    stats = layer.fallback_executor.stats
    # 3 primary successes, but total = 20 → 15% primary rate
    stats.success_by_level["primary"] = 3
    stats.success_by_level["fallback"] = 17
    stats.total_operations = 20

    health = monitor.check_health()
    assert any("WARNING" in a and "Primary success rate" in a for a in health["alerts"]), \
        f"Expected WARNING for low primary rate, got: {health['alerts']}"


def test_monitor_no_alerts_when_healthy():
    """34. All healthy → no alerts."""
    layer, _ = _make_layer(primary_fail_times=0)
    monitor = ResilienceMonitor(layer)

    # Healthy: circuit closed, no operations recorded yet
    health = monitor.check_health()
    assert health["alerts"] == [], f"Unexpected alerts: {health['alerts']}"
