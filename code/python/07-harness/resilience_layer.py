"""
resilience_layer.py — Retry + Fallback + Circuit Breaker for production AI agents.

Three complementary patterns that compose into a single resilience layer:

    Retry         → handles transient failures (packet loss, rate limits, 5xx)
    Fallback      → handles persistent failures (provider down, quota exceeded)
    CircuitBreaker → prevents cascading failure (stops calling broken services)

Usage:

    resilience = ResilienceLayer(
        name="llm_call",
        circuit_breaker=CircuitBreaker("openai"),
        retry_config=RetryConfig(max_retries=3),
        fallback_executor=FallbackExecutor([
            FallbackLevel("gpt-4o",    provider_a, 60, "full"),
            FallbackLevel("claude",    provider_b, 60, "full"),
            FallbackLevel("gpt-mini",  provider_c, 30, "reduced"),
        ]),
    )

    result = await resilience.execute(my_llm_call, messages=messages)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MaxRetriesExceeded(Exception):
    """All retry attempts exhausted without success."""


class NonRetryableError(Exception):
    """Error type that must not be retried (e.g. 400 Bad Request)."""


class RateLimitError(Exception):
    """
    HTTP 429 Too Many Requests.

    Attach the raw response as ``self.response`` so the retry logic can
    inspect Retry-After headers.
    """

    def __init__(self, message: str = "", response: Any = None) -> None:
        super().__init__(message)
        self.response = response


class CircuitBreakerOpenError(Exception):
    """Request rejected because the circuit breaker is in OPEN state."""


class AllFallbacksExhausted(Exception):
    """Every level in the fallback chain failed."""

    def __init__(
        self, message: str, errors: list[FallbackError], total_time_ms: float
    ) -> None:
        super().__init__(message)
        self.errors = errors
        self.total_time_ms = total_time_ms


class SystemUnavailableError(Exception):
    """Primary path and every fallback path failed."""

    def __init__(
        self,
        message: str,
        primary_error: str,
        fallback_errors: list[FallbackError],
    ) -> None:
        super().__init__(message)
        self.primary_error = primary_error
        self.fallback_errors = fallback_errors


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """
    Configuration for exponential-backoff retry behaviour.

    Attributes:
        max_retries:            Maximum number of retry attempts (initial attempt
                                not counted).
        base_delay_seconds:     Initial delay before the first retry.
        max_delay_seconds:      Upper cap on the computed delay.
        backoff_multiplier:     Factor by which delay grows each attempt.
        jitter:                 Whether to add ±jitter_factor randomness.
        jitter_factor:          Fraction of the computed delay to use as jitter
                                range (0.1 = ±10 %).
        retryable_exceptions:   Tuple of exception types that may be retried.
        total_deadline_seconds: Hard wall-clock deadline; give up even if
                                max_retries is not yet reached.
    """

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        RateLimitError,
    )
    total_deadline_seconds: float = 300.0


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Compute the sleep duration before retry number *attempt*.

    Formula (without jitter)::

        delay = base * multiplier ** attempt   (capped at max_delay)

    With jitter the result is randomised by ±(delay * jitter_factor).

    Examples for default config (base=1s, multiplier=2, max=60s, jitter=±10 %)::

        attempt 0 → ~1.00 s
        attempt 1 → ~2.00 s
        attempt 2 → ~4.00 s
        attempt 3 → ~8.00 s
        attempt 4 → ~16.0 s
        attempt 5 → ~32.0 s
        attempt 6 → ~60.0 s  (capped)
    """
    delay = config.base_delay_seconds * (config.backoff_multiplier ** attempt)
    delay = min(delay, config.max_delay_seconds)

    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0.0, delay)


async def retry_with_backoff(
    operation: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute *operation* with exponential-backoff retry.

    Args:
        operation: Async callable to execute.
        config:    Retry configuration; defaults to :class:`RetryConfig`.
        *args:     Forwarded to *operation*.
        **kwargs:  Forwarded to *operation*.

    Returns:
        The value returned by *operation* on the first successful attempt.

    Raises:
        NonRetryableError:  If the operation raises a non-retryable exception.
        MaxRetriesExceeded: If all attempts (initial + max_retries) fail, or
                            the total deadline is exceeded.
    """
    config = config or RetryConfig()
    start_time = time.monotonic()

    for attempt in range(config.max_retries + 1):  # +1 includes the initial attempt
        try:
            return await operation(*args, **kwargs)

        except Exception as exc:
            # Non-retryable errors bubble up immediately.
            if not isinstance(exc, config.retryable_exceptions):
                raise NonRetryableError(f"Non-retryable error: {exc}") from exc

            elapsed = time.monotonic() - start_time

            # Check total deadline before sleeping.
            if elapsed >= config.total_deadline_seconds:
                raise MaxRetriesExceeded(
                    f"Total deadline of {config.total_deadline_seconds}s exceeded "
                    f"after {attempt + 1} attempt(s). Last error: {exc}"
                ) from exc

            # No more retries available.
            if attempt >= config.max_retries:
                raise MaxRetriesExceeded(
                    f"All {config.max_retries + 1} attempt(s) failed. "
                    f"Last error: {exc}"
                ) from exc

            delay = calculate_delay(attempt, config)
            logger.warning(
                "Retry attempt %d/%d failed: %s. Retrying in %.2fs…",
                attempt + 1,
                config.max_retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    # Should be unreachable, but satisfies type checkers.
    raise MaxRetriesExceeded("Unexpected: all attempts exhausted")


# ---------------------------------------------------------------------------
# Rate-limit-aware retry
# ---------------------------------------------------------------------------


@dataclass
class RateLimitAwareRetry(RetryConfig):
    """
    RetryConfig subclass that respects ``Retry-After`` headers from 429 responses.

    When the server specifies how long to wait, that duration is used instead
    of the computed exponential-backoff value.
    """

    def get_delay_from_response(self, response: Any) -> float | None:
        """
        Extract the retry delay advertised by the server.

        Checks (in order):
        - ``retry-after`` header (seconds as a float)
        - ``x-ratelimit-reset-tokens`` header (OpenAI-specific)

        Returns:
            Delay in seconds, or ``None`` if the header is absent / unreadable.
        """
        if response is None:
            return None

        headers: Any = getattr(response, "headers", None) or {}

        for header in ("retry-after", "x-ratelimit-reset-tokens"):
            value = (
                headers.get(header)
                if hasattr(headers, "get")
                else None
            )
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass

        return None


async def retry_with_rate_limit_awareness(
    operation: Callable[..., Any],
    *args: Any,
    config: RateLimitAwareRetry | None = None,
    **kwargs: Any,
) -> Any:
    """
    Like :func:`retry_with_backoff` but uses server-supplied ``Retry-After``
    delays when a :class:`RateLimitError` is raised.

    For all other retryable errors the standard exponential backoff is used.
    """
    config = config or RateLimitAwareRetry()
    start_time = time.monotonic()

    for attempt in range(config.max_retries + 1):
        try:
            return await operation(*args, **kwargs)

        except RateLimitError as exc:
            elapsed = time.monotonic() - start_time
            if elapsed >= config.total_deadline_seconds or attempt >= config.max_retries:
                raise MaxRetriesExceeded(
                    f"Rate limit persisted after {attempt + 1} attempt(s)"
                ) from exc

            server_delay = config.get_delay_from_response(exc.response)
            delay = server_delay if server_delay is not None else calculate_delay(attempt, config)

            logger.warning(
                "Rate limited (attempt %d/%d). Waiting %.2fs (%s)…",
                attempt + 1,
                config.max_retries + 1,
                delay,
                "server-specified" if server_delay is not None else "exponential backoff",
            )
            await asyncio.sleep(delay)

        except Exception as exc:
            if not isinstance(exc, config.retryable_exceptions):
                raise NonRetryableError(f"Non-retryable error: {exc}") from exc

            elapsed = time.monotonic() - start_time
            if elapsed >= config.total_deadline_seconds or attempt >= config.max_retries:
                raise MaxRetriesExceeded(
                    f"All {config.max_retries + 1} attempt(s) failed. Last error: {exc}"
                ) from exc

            delay = calculate_delay(attempt, config)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.2fs…",
                attempt + 1,
                config.max_retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise MaxRetriesExceeded("Unexpected: all attempts exhausted")


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


@dataclass
class FallbackLevel:
    """
    A single level in a fallback chain.

    Attributes:
        name:            Human-readable identifier (used in logs and metrics).
        provider:        The callable / provider object for this level.
        timeout_seconds: Maximum time to wait for this level to respond.
        capability:      ``"full"`` | ``"reduced"`` | ``"static"``.
        cost_multiplier: Relative cost (1.0 = same as primary).  Levels with
                         lower cost are preferred when capability is equal.
    """

    name: str
    provider: Any
    timeout_seconds: float
    capability: str  # "full" | "reduced" | "static"
    cost_multiplier: float = 1.0


@dataclass
class FallbackError:
    """Record of a single fallback level failure."""

    level: int
    level_name: str
    error_type: str
    error_message: str


@dataclass
class FallbackResult:
    """Successful result from the fallback chain."""

    result: Any
    level_used: int
    level_name: str
    capability: str
    attempts: int
    total_time_ms: float
    errors: list[FallbackError] = field(default_factory=list)


class FallbackStats:
    """Aggregate statistics for a :class:`FallbackExecutor` instance."""

    def __init__(self) -> None:
        self.success_by_level: dict[str, int] = defaultdict(int)
        self.failure_by_level: dict[str, int] = defaultdict(int)
        self.failure_by_reason: dict[str, int] = defaultdict(int)
        self.exhaustion_count: int = 0
        self.total_operations: int = 0

    def record_success(self, level_name: str) -> None:
        self.total_operations += 1
        self.success_by_level[level_name] += 1

    def record_failure(self, level_name: str, reason: str) -> None:
        self.failure_by_level[level_name] += 1
        self.failure_by_reason[reason] += 1

    def record_exhaustion(self) -> None:
        self.exhaustion_count += 1
        self.total_operations += 1

    def summary(self) -> dict[str, Any]:
        """Return a dictionary suitable for logging or metrics export."""
        total = max(self.total_operations, 1)
        primary_successes = self.success_by_level.get(
            next(iter(self.success_by_level), ""), 0
        )
        # "primary" is defined as the first level that ever recorded a success.
        # For a more deterministic definition callers can pass level 0's name.
        non_primary_successes = sum(self.success_by_level.values()) - sum(
            [v for k, v in self.success_by_level.items()][:1]
        )
        return {
            "total_operations": self.total_operations,
            "primary_success_rate": primary_successes / total,
            "fallback_activation_rate": non_primary_successes / total,
            "exhaustion_rate": self.exhaustion_count / total,
            "by_level": dict(self.success_by_level),
            "top_failure_reasons": sorted(
                self.failure_by_reason.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


class FallbackExecutor:
    """
    Execute an operation through an ordered fallback chain.

    Each level is tried in order.  On failure the error is recorded and
    execution continues to the next level.  If every level fails,
    :class:`AllFallbacksExhausted` is raised.

    Args:
        levels: Ordered list of :class:`FallbackLevel` objects.  The list is
                *not* re-sorted; callers decide priority by ordering.
    """

    def __init__(self, levels: list[FallbackLevel]) -> None:
        self.levels = levels
        self.stats = FallbackStats()

    async def execute(
        self,
        operation_name: str,
        operation: Callable[[FallbackLevel], Any],
        context: dict[str, Any] | None = None,
    ) -> FallbackResult:
        """
        Execute *operation* through the fallback chain.

        Args:
            operation_name: Label used in logs and metrics.
            operation:      Async callable that accepts a :class:`FallbackLevel`
                            and returns the result.
            context:        Optional extra context forwarded to log records.

        Returns:
            :class:`FallbackResult` describing the successful level.

        Raises:
            AllFallbacksExhausted: If every level fails.
        """
        errors: list[FallbackError] = []
        start_time = time.monotonic()

        for i, level in enumerate(self.levels):
            try:
                logger.info(
                    "Fallback [%s]: Trying level %d (%s, capability=%s)",
                    operation_name,
                    i,
                    level.name,
                    level.capability,
                )

                result = await asyncio.wait_for(
                    operation(level),
                    timeout=level.timeout_seconds,
                )

                elapsed = time.monotonic() - start_time
                self.stats.record_success(level.name)

                logger.info(
                    "Fallback [%s]: Level %d (%s) succeeded in %.2fs",
                    operation_name,
                    i,
                    level.name,
                    elapsed,
                )

                return FallbackResult(
                    result=result,
                    level_used=i,
                    level_name=level.name,
                    capability=level.capability,
                    attempts=len(errors) + 1,
                    total_time_ms=elapsed * 1000,
                    errors=errors,
                )

            except Exception as exc:
                error_entry = FallbackError(
                    level=i,
                    level_name=level.name,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:200],
                )
                errors.append(error_entry)
                self.stats.record_failure(level.name, type(exc).__name__)

                logger.warning(
                    "Fallback [%s]: Level %d (%s) failed: %s: %s",
                    operation_name,
                    i,
                    level.name,
                    type(exc).__name__,
                    str(exc)[:100],
                )

        elapsed = time.monotonic() - start_time
        self.stats.record_exhaustion()

        raise AllFallbacksExhausted(
            f"All {len(self.levels)} fallback level(s) failed for '{operation_name}'",
            errors=errors,
            total_time_ms=elapsed * 1000,
        )


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """States of the circuit breaker state machine."""

    CLOSED = "closed"      # Normal operation — requests pass through.
    OPEN = "open"          # Failing — requests are rejected immediately.
    HALF_OPEN = "half_open"  # Testing recovery — one probe request allowed.


class CircuitBreaker:
    """
    Prevent calls to a service that is persistently failing.

    State machine::

        CLOSED ──(threshold failures in window)──▶ OPEN
        OPEN   ──(recovery timeout elapsed)     ──▶ HALF_OPEN
        HALF_OPEN ──(probe succeeds)            ──▶ CLOSED
        HALF_OPEN ──(probe fails)               ──▶ OPEN

    Args:
        name:                     Identifier used in logs and metrics.
        failure_threshold:        Number of failures within *failure_window_seconds*
                                  that trips the breaker.
        recovery_timeout_seconds: How long to stay OPEN before probing recovery.
        half_open_max_requests:   How many probe requests to allow in HALF_OPEN.
        failure_window_seconds:   Rolling window used to count recent failures.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 120.0,
        half_open_max_requests: int = 1,
        failure_window_seconds: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_max_requests = half_open_max_requests
        self.failure_window = failure_window_seconds

        self.state = CircuitState.CLOSED
        self._failure_timestamps: list[float] = []
        self._half_open_requests: int = 0
        self._last_state_change: float = time.monotonic()

        # Statistics
        self.total_successes: int = 0
        self.total_failures: int = 0
        self.total_rejected: int = 0
        self.times_opened: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute *operation* through the circuit breaker.

        Raises:
            CircuitBreakerOpenError: When the circuit is OPEN (or HALF_OPEN and
                                     the probe-request limit is reached).
        """
        self._maybe_transition()

        if self.state == CircuitState.OPEN:
            self.total_rejected += 1
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Recovery in {self._recovery_remaining():.0f}s."
            )

        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_requests >= self.half_open_max_requests:
                self.total_rejected += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is HALF_OPEN — "
                    f"probe-request limit ({self.half_open_max_requests}) reached."
                )
            self._half_open_requests += 1

        try:
            result = await operation(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of circuit-breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejected": self.total_rejected,
            "times_opened": self.times_opened,
            "recent_failures_in_window": len(self._current_window_failures()),
            "failure_rate": self.total_failures / max(
                self.total_successes + self.total_failures, 1
            ),
            "seconds_in_current_state": time.monotonic() - self._last_state_change,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_transition(self) -> None:
        """Check whether the OPEN → HALF_OPEN transition is due."""
        if (
            self.state == CircuitState.OPEN
            and time.monotonic() - self._last_state_change >= self.recovery_timeout
        ):
            self._transition_to(CircuitState.HALF_OPEN)
            self._half_open_requests = 0

    def _on_success(self) -> None:
        self.total_successes += 1
        if self.state == CircuitState.HALF_OPEN:
            logger.info(
                "Circuit breaker '%s': probe request succeeded — closing circuit.",
                self.name,
            )
            self._transition_to(CircuitState.CLOSED)

    def _on_failure(self) -> None:
        now = time.monotonic()
        self.total_failures += 1
        self._failure_timestamps.append(now)
        recent = self._current_window_failures()

        if self.state == CircuitState.CLOSED and len(recent) >= self.failure_threshold:
            logger.warning(
                "Circuit breaker '%s': %d failure(s) in %.0fs window — opening circuit.",
                self.name,
                len(recent),
                self.failure_window,
            )
            self._transition_to(CircuitState.OPEN)

        elif self.state == CircuitState.HALF_OPEN:
            logger.warning(
                "Circuit breaker '%s': probe request failed — re-opening circuit.",
                self.name,
            )
            self._transition_to(CircuitState.OPEN)

    def _current_window_failures(self) -> list[float]:
        """Return failure timestamps that fall within the rolling window."""
        cutoff = time.monotonic() - self.failure_window
        self._failure_timestamps = [t for t in self._failure_timestamps if t >= cutoff]
        return self._failure_timestamps

    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self.state
        self.state = new_state
        self._last_state_change = time.monotonic()
        if new_state == CircuitState.OPEN:
            self.times_opened += 1
        logger.info(
            "Circuit breaker '%s': %s → %s",
            self.name,
            old_state.value,
            new_state.value,
        )

    def _recovery_remaining(self) -> float:
        """Seconds until the recovery timeout expires."""
        elapsed = time.monotonic() - self._last_state_change
        return max(0.0, self.recovery_timeout - elapsed)


# ---------------------------------------------------------------------------
# ResilienceLayer — combines all three patterns
# ---------------------------------------------------------------------------


@dataclass
class ResilienceResult:
    """Outcome of a :class:`ResilienceLayer` execution."""

    result: Any
    path: str          # "primary" | "fallback_level_N"
    attempts: int
    total_time_ms: float
    fallback_errors: list[FallbackError] = field(default_factory=list)


class ResilienceLayer:
    """
    Unified resilience wrapper: circuit breaker → retry → fallback.

    For every external call the flow is:

    1. Ask the circuit breaker whether the service is healthy.
    2. If CLOSED / HALF_OPEN: attempt the operation with exponential-backoff
       retry.
    3. If retries are exhausted: try the fallback chain.
    4. If the circuit is OPEN: skip directly to the fallback chain.
    5. If every fallback fails: raise :class:`SystemUnavailableError`.

    Args:
        name:              Label for logs and metrics.
        circuit_breaker:   :class:`CircuitBreaker` guarding the primary path.
        retry_config:      :class:`RetryConfig` for the primary path.
        fallback_executor: :class:`FallbackExecutor` for the secondary paths.
    """

    def __init__(
        self,
        name: str,
        circuit_breaker: CircuitBreaker,
        retry_config: RetryConfig,
        fallback_executor: FallbackExecutor,
    ) -> None:
        self.name = name
        self.circuit_breaker = circuit_breaker
        self.retry_config = retry_config
        self.fallback_executor = fallback_executor

    async def execute(
        self,
        operation: Callable[..., Any],
        *args: Any,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ResilienceResult:
        """
        Execute *operation* with full resilience protection.

        Args:
            operation: Async callable to execute on the primary path.
            context:   Optional metadata forwarded to the fallback executor.
            *args:     Forwarded to *operation*.
            **kwargs:  Forwarded to *operation*.

        Returns:
            :class:`ResilienceResult` describing the path taken.

        Raises:
            SystemUnavailableError: When every path (primary + all fallbacks)
                                    has been exhausted.
        """
        start_time = time.monotonic()
        primary_error: Exception | None = None

        try:
            result = await self.circuit_breaker.call(
                retry_with_backoff,
                operation,
                *args,
                config=self.retry_config,
                **kwargs,
            )
            elapsed = time.monotonic() - start_time
            logger.info(
                "Resilience [%s]: primary path succeeded in %.2fs", self.name, elapsed
            )
            return ResilienceResult(
                result=result,
                path="primary",
                attempts=1,
                total_time_ms=elapsed * 1000,
            )

        except (CircuitBreakerOpenError, MaxRetriesExceeded, NonRetryableError) as exc:
            primary_error = exc
            logger.warning(
                "Resilience [%s]: primary path failed (%s: %s) — activating fallback chain.",
                self.name,
                type(exc).__name__,
                exc,
            )

        # Primary failed — try the fallback chain.
        try:
            fallback_result = await self.fallback_executor.execute(
                operation_name=self.name,
                operation=lambda level: level.provider(*args, **kwargs),
                context=context,
            )
            elapsed = time.monotonic() - start_time
            return ResilienceResult(
                result=fallback_result.result,
                path=f"fallback_level_{fallback_result.level_used}",
                attempts=fallback_result.attempts,
                total_time_ms=elapsed * 1000,
                fallback_errors=fallback_result.errors,
            )

        except AllFallbacksExhausted as exc:
            elapsed = time.monotonic() - start_time
            logger.error(
                "Resilience [%s]: all paths exhausted in %.2fs.", self.name, elapsed
            )
            raise SystemUnavailableError(
                f"'{self.name}' is currently unavailable — "
                "all primary and fallback paths exhausted.",
                primary_error=str(primary_error),
                fallback_errors=exc.errors,
            ) from exc


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


class ResilienceMonitor:
    """
    Inspect the health of a :class:`ResilienceLayer` and surface actionable
    alerts suitable for on-call routing.
    """

    # Alert thresholds (can be overridden per-instance)
    PRIMARY_RATE_WARNING_THRESHOLD: float = 0.95
    FALLBACK_RATE_WARNING_THRESHOLD: float = 0.10
    EXHAUSTION_RATE_CRITICAL_THRESHOLD: float = 0.01
    CIRCUIT_OPEN_REOPEN_WARNING: int = 3

    def __init__(self, resilience_layer: ResilienceLayer) -> None:
        self.layer = resilience_layer

    def check_health(self) -> dict[str, Any]:
        """
        Return a health snapshot including circuit breaker state, fallback
        statistics, and any active alerts.
        """
        circuit_stats = self.layer.circuit_breaker.get_stats()
        fallback_stats = self.layer.fallback_executor.stats.summary()

        return {
            "circuit_breaker": circuit_stats,
            "fallback": fallback_stats,
            "alerts": self._generate_alerts(circuit_stats, fallback_stats),
        }

    def _generate_alerts(
        self,
        circuit_stats: dict[str, Any],
        fallback_stats: dict[str, Any],
    ) -> list[str]:
        """Produce human-readable alert strings."""
        alerts: list[str] = []

        # No operations yet — nothing to alert on for fallback metrics.
        if fallback_stats.get("total_operations", 0) == 0:
            # Still surface circuit breaker alerts even with no traffic.
            if circuit_stats["state"] == CircuitState.OPEN.value:
                alerts.append(
                    f"CRITICAL: Circuit breaker '{circuit_stats['name']}' is OPEN"
                )
            return alerts

        # --- Circuit breaker alerts ---
        if circuit_stats["state"] == CircuitState.OPEN.value:
            alerts.append(
                f"CRITICAL: Circuit breaker '{circuit_stats['name']}' is OPEN"
            )

        if circuit_stats["times_opened"] > self.CIRCUIT_OPEN_REOPEN_WARNING:
            alerts.append(
                f"WARNING: Circuit breaker '{circuit_stats['name']}' has opened "
                f"{circuit_stats['times_opened']} time(s) — "
                "investigate underlying service health"
            )

        # --- Fallback alerts ---
        primary_rate = fallback_stats.get("primary_success_rate", 1.0)
        if primary_rate < self.PRIMARY_RATE_WARNING_THRESHOLD:
            alerts.append(
                f"WARNING: Primary success rate is {primary_rate:.1%} "
                f"(threshold: {self.PRIMARY_RATE_WARNING_THRESHOLD:.0%})"
            )

        exhaustion_rate = fallback_stats.get("exhaustion_rate", 0.0)
        if exhaustion_rate > self.EXHAUSTION_RATE_CRITICAL_THRESHOLD:
            alerts.append(
                f"CRITICAL: Fallback exhaustion rate is {exhaustion_rate:.1%} "
                "— system fully unavailable for some requests"
            )

        fallback_rate = fallback_stats.get("fallback_activation_rate", 0.0)
        if fallback_rate > self.FALLBACK_RATE_WARNING_THRESHOLD:
            alerts.append(
                f"WARNING: Fallback activated for {fallback_rate:.1%} of requests "
                f"(threshold: {self.FALLBACK_RATE_WARNING_THRESHOLD:.0%})"
            )

        return alerts


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    """Demonstrate the resilience layer against a variety of simulated failures."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("\n" + "=" * 60)
    print("RESILIENCE LAYER DEMO")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Scenario 1: Happy path
    # ------------------------------------------------------------------
    print("\n--- Scenario 1: Happy path ---")

    call_count = 0

    async def always_succeeds(*args: Any, **kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        return "Hello from primary!"

    layer = ResilienceLayer(
        name="demo",
        circuit_breaker=CircuitBreaker("demo-cb", failure_threshold=3),
        retry_config=RetryConfig(max_retries=2, base_delay_seconds=0.01),
        fallback_executor=FallbackExecutor([]),
    )

    try:
        res = await layer.execute(always_succeeds)
        print(f"Result: {res.result!r}  path={res.path}  attempts={res.attempts}")
    except SystemUnavailableError as exc:
        print(f"Unavailable: {exc}")

    # ------------------------------------------------------------------
    # Scenario 2: One transient failure, then success
    # ------------------------------------------------------------------
    print("\n--- Scenario 2: Transient failure → retry succeeds ---")

    transient_count = 0

    async def fails_once(*args: Any, **kwargs: Any) -> str:
        nonlocal transient_count
        transient_count += 1
        if transient_count == 1:
            raise TimeoutError("simulated timeout")
        return "Recovered!"

    layer2 = ResilienceLayer(
        name="demo2",
        circuit_breaker=CircuitBreaker("demo2-cb", failure_threshold=5),
        retry_config=RetryConfig(max_retries=3, base_delay_seconds=0.01),
        fallback_executor=FallbackExecutor([]),
    )

    try:
        res2 = await layer2.execute(fails_once)
        print(f"Result: {res2.result!r}  path={res2.path}")
    except SystemUnavailableError as exc:
        print(f"Unavailable: {exc}")

    # ------------------------------------------------------------------
    # Scenario 3: Primary exhausted, fallback succeeds
    # ------------------------------------------------------------------
    print("\n--- Scenario 3: Primary exhausted → fallback ---")

    async def always_fails(*args: Any, **kwargs: Any) -> str:
        raise ConnectionError("simulated connection error")

    async def fallback_provider(*args: Any, **kwargs: Any) -> str:
        return "Fallback response!"

    layer3 = ResilienceLayer(
        name="demo3",
        circuit_breaker=CircuitBreaker("demo3-cb", failure_threshold=10),
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        fallback_executor=FallbackExecutor([
            FallbackLevel("fallback", fallback_provider, 10.0, "reduced"),
        ]),
    )

    try:
        res3 = await layer3.execute(always_fails)
        print(f"Result: {res3.result!r}  path={res3.path}")
    except SystemUnavailableError as exc:
        print(f"Unavailable: {exc}")

    # ------------------------------------------------------------------
    # Scenario 4: Circuit breaker opens
    # ------------------------------------------------------------------
    print("\n--- Scenario 4: Circuit breaker opens ---")

    cb = CircuitBreaker("demo4-cb", failure_threshold=3, recovery_timeout_seconds=0.05)
    for _ in range(3):
        try:
            await cb.call(always_fails)
        except (ConnectionError, CircuitBreakerOpenError):
            pass

    print(f"Circuit state after 3 failures: {cb.state.value}")

    try:
        await cb.call(always_succeeds)
    except CircuitBreakerOpenError as exc:
        print(f"Rejected (expected): {exc}")

    await asyncio.sleep(0.1)  # Wait for recovery timeout
    cb._maybe_transition()
    print(f"Circuit state after timeout: {cb.state.value}")

    # Probe succeeds → close
    await cb.call(always_succeeds)
    print(f"Circuit state after probe success: {cb.state.value}")

    # ------------------------------------------------------------------
    # Scenario 5: All paths exhausted → SystemUnavailableError
    # ------------------------------------------------------------------
    print("\n--- Scenario 5: All paths exhausted ---")

    layer5 = ResilienceLayer(
        name="demo5",
        circuit_breaker=CircuitBreaker("demo5-cb", failure_threshold=10),
        retry_config=RetryConfig(max_retries=1, base_delay_seconds=0.01),
        fallback_executor=FallbackExecutor([
            FallbackLevel("fallback", always_fails, 5.0, "static"),
        ]),
    )

    try:
        await layer5.execute(always_fails)
    except SystemUnavailableError as exc:
        print(f"SystemUnavailableError (expected): {exc}")

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    print("\n--- Monitoring health check ---")
    monitor = ResilienceMonitor(layer3)
    health = monitor.check_health()
    print(f"Circuit state: {health['circuit_breaker']['state']}")
    print(f"Alerts: {health['alerts'] or ['(none)']}")

    print("\nDemo complete.\n")


if __name__ == "__main__":
    asyncio.run(_demo())
