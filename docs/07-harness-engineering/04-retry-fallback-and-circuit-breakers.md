# Retry, Fallback, and Circuit Breakers

## What You'll Learn
- Why "just call the API and hope" fails in production
- Retry strategies: when to retry, how many times, and how long to wait
- Exponential backoff with jitter: preventing the thundering herd
- Fallback chains: degrading gracefully when primary systems fail
- Circuit breakers: detecting failures and preventing cascading collapse
- Combining all three into a resilience layer that keeps your agent running

## Prerequisites
- [The Harness Mindset](01-the-harness-mindset.md) — every LLM call needs a timeout and fallback
- [Routing and Intent Classification](03-routing-and-intent-classification.md) — handlers are what you're protecting
- [Model Providers](../05-the-tool-ecosystem/01-model-providers.md) — multiple providers enable cross-provider fallback

---

## The Reality of External Dependencies

Your agent depends on external systems. Every single one will fail.

```
Your Agent
    │
    ├── LLM Provider (OpenAI)     → Will rate-limit you during traffic spikes
    ├── LLM Provider (Anthropic)  → Will have an outage during your launch
    ├── Vector Database (Qdrant)  → Will return 500 errors under load
    ├── Weather API               → Will timeout during a storm
    ├── Order Database            → Will be slow on Black Friday
    └── Email Service             → Will reject your API key at midnight
```

The question isn't *if* these will fail. It's *how your system handles it when they do.*

---

## The Resilience Triad

| Pattern | What It Does | Analogy |
|:---|:---|:---|
| **Retry** | Try the same operation again after a failure | "Let me try that one more time" |
| **Fallback** | Use an alternative when the primary fails | "Plan B" |
| **Circuit Breaker** | Stop trying when failures are persistent | "Let's give them a break and try later" |

They work together. Retry handles transient failures. Fallback handles persistent failures. Circuit breakers prevent you from wasting resources on known-broken systems.

---

## Retry: Handling Transient Failures

Not all failures should be retried. Some should. The distinction matters.

### When to Retry

| Error Type | Retry? | Why |
|:---|:---|:---|
| **Rate limit (429)** | Yes | Temporary. Server says "try again later" |
| **Server error (500-599)** | Yes | Temporary. Server might recover |
| **Network timeout** | Yes | Temporary. Packet loss, congestion |
| **Connection error** | Yes | Temporary. Network blip |
| **Service unavailable (503)** | Yes | Temporary. Server overloaded |
| **Connection reset** | Yes | Temporary. TCP connection dropped mid-flight |
| **Bad request (400)** | No | Your fault. Retrying won't fix it |
| **Authentication error (401/403)** | No | Your credentials are wrong. Retrying won't fix it |
| **Not found (404)** | No | The resource doesn't exist. Retrying won't fix it |
| **Validation error (422)** | No | Your input is invalid. Retrying won't fix it |
| **Quota exceeded** | No (usually) | You're out of credits. Retrying makes it worse |

### The Retry Decision Tree

```
Error occurred
    │
    ▼
┌─────────────────┐
│ Is it retryable? │─── NO ──→ Don't retry. Return error.
└────────┬────────┘
         │ YES
         ▼
┌─────────────────┐
│ Under max tries? │─── NO ──→ Don't retry. Trigger fallback.
└────────┬────────┘
         │ YES
         ▼
┌─────────────────┐
│ Has the total    │─── NO ──→ Don't retry. Deadline exceeded.
│ deadline passed? │
└────────┬────────┘
         │ YES
         ▼
    Retry with backoff
```

---

## Exponential Backoff with Jitter

The simplest retry strategy is "wait a bit and try again." But if 1,000 clients all retry at the same time, they create a **thundering herd** that overwhelms the recovering service.

The solution: **exponential backoff with jitter.**

### The Algorithm

```python
import random
import time
import asyncio
from typing import Callable, TypeVar

T = TypeVar('T')

class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0      # Initial delay
    max_delay_seconds: float = 60.0      # Cap the delay
    backoff_multiplier: float = 2.0      # Exponential factor
    jitter: bool = True                  # Add randomness
    jitter_factor: float = 0.1           # ±10% randomness
    retryable_exceptions: tuple = (TimeoutError, ConnectionError, RateLimitError)
    total_deadline_seconds: float = 300  # Give up entirely after 5 minutes

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for the next retry attempt.

    Without jitter (attempt index → delay):
    0 → 1s,  1 → 2s,  2 → 4s,  3 → 8s,  4 → 16s,  5 → 32s,  6 → 60s (capped)

    With jitter (±10%):
    0 → ~1.0s,  1 → ~2.0s,  2 → ~4.0s,  3 → ~8.0s,  4 → ~16s,  5 → ~32s
    """
    # Exponential backoff
    delay = config.base_delay_seconds * (config.backoff_multiplier ** attempt)
    
    # Cap at max delay
    delay = min(delay, config.max_delay_seconds)
    
    # Add jitter: randomize ±jitter_factor
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay = delay + random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)

async def retry_with_backoff(
    operation: Callable[..., T],
    *args,
    config: RetryConfig = None,
    **kwargs
) -> T:
    """
    Execute an operation with exponential backoff retry.
    
    Args:
        operation: Async function to call
        config: Retry configuration
        *args, **kwargs: Passed to the operation
    
    Returns:
        The operation's return value
    
    Raises:
        MaxRetriesExceeded: If all retries are exhausted
        NonRetryableError: If the error should not be retried
    """
    config = config or RetryConfig()
    last_error = None
    start_time = time.monotonic()
    
    for attempt in range(config.max_retries + 1):  # +1 for initial attempt
        try:
            return await operation(*args, **kwargs)
        
        except Exception as e:
            last_error = e
            
            # Check if this error is retryable
            if not isinstance(e, config.retryable_exceptions):
                raise NonRetryableError(f"Non-retryable error: {e}") from e
            
            # Check if we've exceeded the total deadline
            elapsed = time.monotonic() - start_time
            if elapsed > config.total_deadline_seconds:
                raise MaxRetriesExceeded(
                    f"Total deadline of {config.total_deadline_seconds}s exceeded "
                    f"after {attempt + 1} attempts"
                ) from e
            
            # Check if we're out of retries
            if attempt >= config.max_retries:
                raise MaxRetriesExceeded(
                    f"All {config.max_retries + 1} attempts failed. "
                    f"Last error: {e}"
                ) from e
            
            # Calculate delay and wait
            delay = calculate_delay(attempt, config)
            logger.warning(
                f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            
            await asyncio.sleep(delay)
    
    # Should never reach here, but just in case
    raise MaxRetriesExceeded(f"Unexpected: all attempts exhausted")

class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts have been exhausted."""
    pass

class NonRetryableError(Exception):
    """Raised when an error should not be retried."""
    pass
```

### Rate Limit Awareness

When you get a 429 (rate limit) response, the server often tells you how long to wait:

```python
class RateLimitAwareRetry(RetryConfig):
    """
    Retry configuration that respects server-specified retry delays.
    """
    
    def get_delay_from_response(self, response) -> float | None:
        """
        Extract retry delay from rate limit headers.
        
        OpenAI headers:
        - x-ratelimit-reset-tokens: When token capacity resets
        - retry-after: Seconds to wait
        
        Anthropic headers:
        - retry-after: Seconds to wait
        
        Generic:
        - retry-after: Seconds to wait
        """
        if hasattr(response, 'headers'):
            for header in ('retry-after', 'x-ratelimit-reset-requests', 'x-ratelimit-reset-tokens'):
                value = response.headers.get(header)
                if value:
                    try:
                        return float(value)
                    except ValueError:
                        pass

        return None

async def retry_with_rate_limit_awareness(
    operation: Callable,
    *args,
    config: RateLimitAwareRetry = None,
    **kwargs
) -> any:
    """
    Retry with awareness of rate limit headers.
    Uses server-specified delay when available, falls back to exponential backoff.
    """
    config = config or RateLimitAwareRetry()
    
    for attempt in range(config.max_retries + 1):
        try:
            return await operation(*args, **kwargs)
        
        except RateLimitError as e:
            if attempt >= config.max_retries:
                raise MaxRetriesExceeded(f"Rate limit persist after {attempt + 1} attempts")
            
            # Use server-specified delay if available
            server_delay = config.get_delay_from_response(e.response) if hasattr(e, 'response') else None
            delay = server_delay if server_delay else calculate_delay(attempt, config)
            
            logger.warning(
                f"Rate limited. Waiting {delay:.2f}s "
                f"({'server-specified' if server_delay else 'exponential backoff'})"
            )
            
            await asyncio.sleep(delay)
```

---

## Fallback: Degrading Gracefully

When retries are exhausted, you need a Plan B. The fallback chain tries alternatives in order.

### The Fallback Hierarchy

```
Level 0: Primary (best capability, highest cost)
    │
    ▼ (failure)
Level 1: Secondary (equivalent capability, different provider)
    │
    ▼ (failure)
Level 2: Tertiary (reduced capability, lower cost)
    │
    ▼ (failure)
Level 3: Static/Degraded (no LLM, pre-computed response)
    │
    ▼ (failure)
Level 4: Error (clear message to user)
```

### Implementation

```python
@dataclass
class FallbackLevel:
    """A single level in the fallback chain."""
    name: str
    provider: any  # The thing to call
    timeout_seconds: float
    capability: str  # "full", "reduced", "static"
    cost_multiplier: float = 1.0

class FallbackExecutor:
    """
    Execute an operation through a fallback chain.
    Tries each level in order until one succeeds.

    Levels are tried in the order provided — callers establish priority
    by ordering the list.  Do not sort by cost here; a cheap-but-reduced
    model should not be preferred over an expensive-but-full model unless
    the caller explicitly puts it first.
    """

    def __init__(self, levels: list[FallbackLevel]):
        self.levels = levels  # caller-defined priority order
        self.stats = FallbackStats()
    
    async def execute(
        self,
        operation_name: str,
        operation: Callable[[FallbackLevel], any],
        context: dict = None
    ) -> FallbackResult:
        """
        Execute an operation through the fallback chain.
        
        Args:
            operation_name: For logging/metrics
            operation: Function that takes a FallbackLevel and returns a result
            context: Additional context for logging
        
        Returns:
            FallbackResult with the response and which level served it
        """
        errors = []
        start_time = time.monotonic()
        
        for i, level in enumerate(self.levels):
            try:
                logger.info(
                    f"Fallback [{operation_name}]: Trying level {i} "
                    f"({level.name}, capability: {level.capability})"
                )
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    operation(level),
                    timeout=level.timeout_seconds
                )
                
                # Success
                elapsed = time.monotonic() - start_time
                self.stats.record_success(level.name, i, elapsed)
                
                logger.info(
                    f"Fallback [{operation_name}]: Level {i} ({level.name}) "
                    f"succeeded in {elapsed:.2f}s"
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
            
            except Exception as e:
                errors.append(FallbackError(
                    level=i,
                    level_name=level.name,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200],
                ))
                
                logger.warning(
                    f"Fallback [{operation_name}]: Level {i} ({level.name}) "
                    f"failed: {type(e).__name__}: {str(e)[:100]}"
                )
                
                # Log the failure
                self.stats.record_failure(level.name, type(e).__name__)
                
                continue
        
        # All levels exhausted
        elapsed = time.monotonic() - start_time
        self.stats.record_exhaustion(operation_name, elapsed)
        
        raise AllFallbacksExhausted(
            f"All {len(self.levels)} fallback levels failed for '{operation_name}'",
            errors=errors,
            total_time_ms=elapsed * 1000,
        )

@dataclass
class FallbackResult:
    result: any
    level_used: int
    level_name: str
    capability: str
    attempts: int
    total_time_ms: float
    errors: list

@dataclass
class FallbackError:
    level: int
    level_name: str
    error_type: str
    error_message: str

class AllFallbacksExhausted(Exception):
    def __init__(self, message: str, errors: list, total_time_ms: float):
        super().__init__(message)
        self.errors = errors
        self.total_time_ms = total_time_ms

class FallbackStats:
    """Track fallback performance for monitoring."""
    
    def __init__(self):
        self.success_by_level = defaultdict(int)
        self.failure_by_level = defaultdict(int)
        self.failure_by_reason = defaultdict(int)
        self.exhaustion_count = 0
        self.total_operations = 0
    
    def record_success(self, level_name: str, level_index: int, elapsed: float):
        self.total_operations += 1
        self.success_by_level[level_name] += 1
    
    def record_failure(self, level_name: str, reason: str):
        self.failure_by_level[level_name] += 1
        self.failure_by_reason[reason] += 1
    
    def record_exhaustion(self, operation: str, elapsed: float):
        self.exhaustion_count += 1
    
    def summary(self) -> dict:
        return {
            "total_operations": self.total_operations,
            "primary_success_rate": (
                self.success_by_level.get("primary", 0) / max(self.total_operations, 1)
            ),
            "fallback_activation_rate": (
                sum(self.success_by_level.values()) - self.success_by_level.get("primary", 0)
            ) / max(self.total_operations, 1),
            "exhaustion_rate": self.exhaustion_count / max(self.total_operations, 1),
            "by_level": dict(self.success_by_level),
            "top_failure_reasons": sorted(
                self.failure_by_reason.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }
```

### LLM Fallback Configuration

```python
# Configure the LLM fallback chain
llm_fallback = FallbackExecutor([
    FallbackLevel(
        name="gpt-4o",
        provider=openai_provider("gpt-4o"),
        timeout_seconds=60,
        capability="full",
        cost_multiplier=1.0,
    ),
    FallbackLevel(
        name="claude-sonnet",
        provider=anthropic_provider("claude-3-5-sonnet"),
        timeout_seconds=60,
        capability="full",
        cost_multiplier=1.2,  # Slightly more expensive
    ),
    FallbackLevel(
        name="gpt-4o-mini",
        provider=openai_provider("gpt-4o-mini"),
        timeout_seconds=30,
        capability="reduced",  # Lower quality but functional
        cost_multiplier=0.06,   # Much cheaper
    ),
    FallbackLevel(
        name="static-response",
        provider=static_response_provider(),
        timeout_seconds=5,
        capability="static",  # Pre-computed responses only
        cost_multiplier=0.0,    # Free
    ),
])

# Use it for every LLM call
result = await llm_fallback.execute(
    operation_name="generate_response",
    operation=lambda level: level.provider(messages, tools=tools),
)
```

---

## Circuit Breaker: Preventing Cascading Failure

When a service is clearly down, don't keep calling it. The circuit breaker detects persistent failures and stops sending requests for a cooling-off period.

### Circuit Breaker States

```
                    ┌──────────┐
                    │  CLOSED  │  ← Normal operation
                    │ (working)│     Requests pass through
                    └────┬─────┘
                         │
                    failure threshold reached
                    (5 failures in 60 seconds)
                         │
                         ▼
                    ┌──────────┐
                    │   OPEN   │  ← Service is down
                    │ (failing)│     Requests are rejected immediately
                    └────┬─────┘     (fast failure)
                         │
                    timeout expires
                    (120 seconds)
                         │
                         ▼
                    ┌──────────┐
                    │HALF-OPEN │  ← Testing if service recovered
                    │ (testing)│     One request allowed through
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │ success              │ failure
              ▼                      ▼
        ┌──────────┐          ┌──────────┐
        │  CLOSED  │          │   OPEN   │
        │ (working)│          │ (failing)│
        └──────────┘          └──────────┘
```

### Implementation

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """
    Prevent calls to a failing service.
    
    Configuration:
    - failure_threshold: Number of failures before opening (default: 5)
    - recovery_timeout: Seconds before trying half-open (default: 120)
    - half_open_max_requests: Requests allowed in half-open (default: 1)
    - failure_window: Rolling window for counting failures (default: 60s)
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 120.0,
        half_open_max_requests: int = 1,
        failure_window_seconds: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_max_requests = half_open_max_requests
        self.failure_window = failure_window_seconds
        
        self.state = CircuitState.CLOSED
        self.failure_timestamps: list[float] = []
        self.last_failure_time: float = None
        self.half_open_requests: int = 0
        self.last_state_change: float = time.monotonic()
        
        # Statistics
        self.total_successes = 0
        self.total_failures = 0
        self.total_rejected = 0
        self.times_opened = 0
    
    async def call(self, operation: Callable, *args, **kwargs) -> any:
        """
        Execute an operation through the circuit breaker.
        
        If the circuit is OPEN, raises CircuitBreakerOpenError immediately.
        If CLOSED or HALF_OPEN, executes the operation and tracks the result.
        """
        # Check state and maybe transition
        self._check_state()
        
        # If open, fast-fail
        if self.state == CircuitState.OPEN:
            self.total_rejected += 1
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Opened at {datetime.fromtimestamp(self.last_state_change)}. "
                f"Recovery in {self._recovery_remaining():.0f}s."
            )
        
        # If half-open, check request limit
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_requests >= self.half_open_max_requests:
                self.total_rejected += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is HALF_OPEN and "
                    f"request limit ({self.half_open_max_requests}) reached."
                )
            self.half_open_requests += 1
        
        try:
            result = await operation(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure()
            raise
    
    def _check_state(self):
        """Check and potentially transition state."""
        now = time.monotonic()
        
        # If OPEN and recovery timeout has passed → HALF_OPEN
        if self.state == CircuitState.OPEN:
            if now - self.last_state_change >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                self.half_open_requests = 0
    
    def _on_success(self):
        """Record a successful call."""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            # Successful test → close the circuit
            self._transition_to(CircuitState.CLOSED)
            logger.info(f"Circuit breaker '{self.name}': Test succeeded. Closing circuit.")
    
    def _on_failure(self):
        """Record a failed call."""
        now = time.monotonic()
        self.total_failures += 1
        self.failure_timestamps.append(now)
        
        # Clean old failures outside the window
        self.failure_timestamps = [
            t for t in self.failure_timestamps
            if now - t <= self.failure_window
        ]
        
        recent_failures = len(self.failure_timestamps)
        
        # If CLOSED and threshold exceeded → OPEN
        if self.state == CircuitState.CLOSED and recent_failures >= self.failure_threshold:
            self._transition_to(CircuitState.OPEN)
            logger.warning(
                f"Circuit breaker '{self.name}': {recent_failures} failures in "
                f"{self.failure_window}s. Opening circuit for {self.recovery_timeout}s."
            )
        
        # If HALF_OPEN and the test request fails → back to OPEN
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
            logger.warning(
                f"Circuit breaker '{self.name}': Test request failed. "
                f"Re-opening circuit."
            )
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.monotonic()
        
        if new_state == CircuitState.OPEN:
            self.times_opened += 1
        
        logger.info(
            f"Circuit breaker '{self.name}': {old_state.value} → {new_state.value}"
        )
    
    def _recovery_remaining(self) -> float:
        """Seconds until recovery timeout."""
        elapsed = time.monotonic() - self.last_state_change
        return max(0, self.recovery_timeout - elapsed)
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejected": self.total_rejected,
            "times_opened": self.times_opened,
            "recent_failures": len(self.failure_timestamps),
            "failure_rate": (
                self.total_failures / max(self.total_successes + self.total_failures, 1)
            ),
            "seconds_in_current_state": time.monotonic() - self.last_state_change,
        }

class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""
    pass
```

---

## The Complete Resilience Layer

Combine retry, fallback, and circuit breaker into a single resilience wrapper:

```python
class ResilienceLayer:
    """
    Complete resilience wrapper: circuit breaker → retry → fallback.
    
    For every external call:
    1. Check circuit breaker (is the service healthy?)
    2. Attempt with retry (transient failure?)
    3. On exhaustion, trigger fallback (persistent failure?)
    """
    
    def __init__(
        self,
        name: str,
        circuit_breaker: CircuitBreaker,
        retry_config: RetryConfig,
        fallback_executor: FallbackExecutor,
    ):
        self.name = name
        self.circuit_breaker = circuit_breaker
        self.retry_config = retry_config
        self.fallback_executor = fallback_executor
    
    async def execute(
        self,
        operation: Callable,
        *args,
        context: dict = None,
        **kwargs,
    ) -> ResilienceResult:
        """
        Execute an operation with full resilience.
        
        Flow:
        1. Circuit breaker check
        2. If closed: try operation with retry
        3. If retries exhausted: try fallback chain
        4. If circuit open: go directly to fallback
        """
        start_time = time.monotonic()
        
        try:
            # Try through circuit breaker
            result = await self.circuit_breaker.call(
                retry_with_backoff,
                operation,
                *args,
                config=self.retry_config,
                **kwargs,
            )
            
            elapsed = time.monotonic() - start_time
            
            logger.info(
                f"Resilience [{self.name}]: Primary succeeded "
                f"(attempts: 1, time: {elapsed:.2f}s)"
            )
            
            return ResilienceResult(
                result=result,
                path="primary",
                attempts=1,
                total_time_ms=elapsed * 1000,
            )
        
        except (CircuitBreakerOpenError, MaxRetriesExceeded) as e:
            logger.warning(
                f"Resilience [{self.name}]: Primary failed ({e}). "
                f"Activating fallback chain."
            )
            
            # Try fallback chain
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
            
            except AllFallbacksExhausted as fe:
                elapsed = time.monotonic() - start_time
                
                logger.error(
                    f"Resilience [{self.name}]: All paths exhausted. "
                    f"Total time: {elapsed:.2f}s"
                )
                
                raise SystemUnavailableError(
                    f"'{self.name}' is currently unavailable. "
                    f"All primary and fallback paths have been exhausted.",
                    primary_error=str(e),
                    fallback_errors=fe.errors,
                ) from fe

@dataclass
class ResilienceResult:
    result: any
    path: str  # "primary", "fallback_level_0", "fallback_level_1", etc.
    attempts: int
    total_time_ms: float
    fallback_errors: list = None

class SystemUnavailableError(Exception):
    def __init__(self, message: str, primary_error: str, fallback_errors: list):
        super().__init__(message)
        self.primary_error = primary_error
        self.fallback_errors = fallback_errors
```

---

## Wiring It into the Harness

```python
class ResilientHarness:
    """
    Complete harness with resilience at every level.
    """
    
    def __init__(self):
        # LLM resilience
        self.llm_resilience = ResilienceLayer(
            name="llm_call",
            circuit_breaker=CircuitBreaker(
                name="openai",
                failure_threshold=5,
                recovery_timeout_seconds=120,
            ),
            retry_config=RetryConfig(
                max_retries=3,
                base_delay_seconds=1.0,
                max_delay_seconds=30.0,
                retryable_exceptions=(TimeoutError, RateLimitError, ConnectionError),
            ),
            fallback_executor=FallbackExecutor([
                FallbackLevel("gpt-4o", openai_provider("gpt-4o"), 60, "full"),
                FallbackLevel("claude-sonnet", anthropic_provider("claude-sonnet"), 60, "full"),
                FallbackLevel("gpt-4o-mini", openai_provider("gpt-4o-mini"), 30, "reduced"),
            ]),
        )

        # Tool execution resilience
        self.tool_resilience = ResilienceLayer(
            name="tool_execution",
            circuit_breaker=CircuitBreaker(
                name="tools",
                failure_threshold=3,
                recovery_timeout_seconds=60,
            ),
            retry_config=RetryConfig(
                max_retries=2,
                base_delay_seconds=0.5,
            ),
            fallback_executor=FallbackExecutor([
                FallbackLevel("primary_tool", primary_tool, 30, "full"),
                FallbackLevel("cached_result", cached_result_provider, 5, "static"),
            ]),
        )
    
    async def process(self, user_input: str) -> str:
        """Process a user request with resilience at every step."""
        
        # Every LLM call goes through the resilience layer
        response = await self.llm_resilience.execute(
            operation=lambda: primary_llm_provider(messages, tools=tools),
        )

        # Every tool call goes through the resilience layer
        tool_result = await self.tool_resilience.execute(
            operation=lambda: primary_tool_provider("get_weather", {"city": "Tokyo"}),
        )
        
        return response.result
```

> **Code Reference:** [Python](../../code/python/07-harness/) · [Node.js](../../code/nodejs/07-harness/) · [Go](../../code/go/07-harness/)  
> The harness implementations include the complete resilience layer with retry, fallback, and circuit breaker.

---

## Monitoring the Resilience Layer

```python
class ResilienceMonitor:
    """
    Monitor resilience layer health and alert on degradation.
    """
    
    def __init__(self, resilience_layer: ResilienceLayer):
        self.layer = resilience_layer
    
    def check_health(self) -> dict:
        """Check the health of the resilience layer."""
        circuit_stats = self.layer.circuit_breaker.get_stats()
        fallback_stats = self.layer.fallback_executor.stats.summary()
        
        return {
            "circuit_breaker": circuit_stats,
            "fallback": fallback_stats,
            "alerts": self._generate_alerts(circuit_stats, fallback_stats),
        }
    
    def _generate_alerts(self, circuit_stats: dict, fallback_stats: dict) -> list[str]:
        """Generate alerts based on resilience metrics."""
        alerts = []
        
        # Circuit breaker alerts
        if circuit_stats["state"] == "open":
            alerts.append(f"CRITICAL: Circuit breaker '{circuit_stats['name']}' is OPEN")
        
        if circuit_stats["times_opened"] > 3:
            alerts.append(f"WARNING: Circuit breaker opened {circuit_stats['times_opened']} times")
        
        # Fallback alerts
        primary_rate = fallback_stats.get("primary_success_rate", 0)
        if primary_rate < 0.95:
            alerts.append(f"WARNING: Primary success rate is {primary_rate:.1%} (below 95%)")
        
        exhaustion_rate = fallback_stats.get("exhaustion_rate", 0)
        if exhaustion_rate > 0.01:
            alerts.append(f"CRITICAL: Fallback exhaustion rate is {exhaustion_rate:.1%}")
        
        fallback_rate = fallback_stats.get("fallback_activation_rate", 0)
        if fallback_rate > 0.10:
            alerts.append(f"WARNING: Fallback activated for {fallback_rate:.1%} of requests")
        
        return alerts
```

---

## Common Pitfalls

- **"I retry everything including 400 errors"**: Retrying a bad request just wastes resources and delays the error. Only retry transient errors (429, 5xx, timeouts).
- **"My retry delay is always the same"**: Without exponential backoff, you create a thundering herd. Without jitter, all your retries cluster at the same moment.
- **"I don't set a total deadline"**: Retry delays grow exponentially. A 3-second base delay with 5 retries and 2x multiplier = 93 seconds total. Set a deadline that makes sense for your user experience.
- **"My fallback is the same provider"**: If OpenAI is down, falling back to OpenAI won't help. Cross-provider fallback is essential. At minimum, have one provider on a different infrastructure.
- **"I don't close the circuit when the fallback is active"**: If primary is failing and all traffic is going to fallback, the fallback might also fail under the load. Circuit breakers protect both sides.
- **"I treat circuit breaker state as a binary"**: The half-open state is critical. It tests recovery without flooding the service. Don't skip it.
- **"I don't monitor resilience metrics"**: If your fallback activates 30% of the time, you have a problem even if users don't notice. Monitor primary success rate, fallback activation rate, and circuit breaker state.
- **"I use the same retry config for all calls"**: A background job can wait 60 s between retries; a user-facing API cannot. Configure `RetryConfig` per call-site with an appropriate `total_deadline_seconds` for the UX context. See the `ResilienceConfigBuilder` for named profiles.

## What's Next

Your external calls are now resilient. Next: validating what comes back — output guardrails that catch hallucinations, enforce schemas, and ensure quality before the user sees the response.
→ [Output Guardrails and Fact-Checking](05-output-guardrails-and-fact-checking.md)