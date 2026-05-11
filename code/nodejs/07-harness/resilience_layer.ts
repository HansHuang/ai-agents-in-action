/**
 * resilience_layer.ts — Retry + Fallback + Circuit Breaker for production AI agents.
 *
 * TypeScript port of the Python resilience_layer.py implementation.
 * Uses AbortController / Promise.race for timeouts and strict types throughout.
 *
 * Classes:
 *   RetryConfig             — exponential-backoff retry configuration
 *   RateLimitAwareRetry     — RetryConfig that respects Retry-After headers
 *   FallbackLevel           — single level in a fallback chain
 *   FallbackExecutor        — executes operation through ordered fallback chain
 *   CircuitBreaker          — three-state (CLOSED/OPEN/HALF_OPEN) circuit breaker
 *   ResilienceLayer         — combines all three patterns
 *   ResilienceMonitor       — health check and alerting
 *
 * See: docs/07-harness-engineering/04-retry-fallback-and-circuit-breakers.md
 */

// ---------------------------------------------------------------------------
// Custom error types
// ---------------------------------------------------------------------------

export class MaxRetriesExceeded extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MaxRetriesExceeded";
  }
}

export class NonRetryableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NonRetryableError";
  }
}

export class RateLimitError extends Error {
  readonly response?: Response | { headers: Record<string, string> };

  constructor(
    message: string,
    response?: Response | { headers: Record<string, string> },
  ) {
    super(message);
    this.name = "RateLimitError";
    this.response = response;
  }
}

export class CircuitBreakerOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CircuitBreakerOpenError";
  }
}

export class AllFallbacksExhausted extends Error {
  readonly errors: FallbackError[];
  readonly totalTimeMs: number;

  constructor(message: string, errors: FallbackError[], totalTimeMs: number) {
    super(message);
    this.name = "AllFallbacksExhausted";
    this.errors = errors;
    this.totalTimeMs = totalTimeMs;
  }
}

export class SystemUnavailableError extends Error {
  readonly primaryError: string;
  readonly fallbackErrors: FallbackError[];

  constructor(
    message: string,
    primaryError: string,
    fallbackErrors: FallbackError[],
  ) {
    super(message);
    this.name = "SystemUnavailableError";
    this.primaryError = primaryError;
    this.fallbackErrors = fallbackErrors;
  }
}

// ---------------------------------------------------------------------------
// Retry
// ---------------------------------------------------------------------------

/** Configuration for exponential-backoff retry behaviour. */
export interface RetryConfig {
  /** Maximum retry attempts (initial attempt not counted). */
  maxRetries: number;
  /** Initial delay before the first retry, in seconds. */
  baseDelaySec: number;
  /** Upper cap on the computed delay, in seconds. */
  maxDelaySec: number;
  /** Exponential growth factor. */
  backoffMultiplier: number;
  /** Whether to add ±jitterFactor randomness. */
  jitter: boolean;
  /** Fraction of the computed delay to use as jitter range (0.1 = ±10 %). */
  jitterFactor: number;
  /**
   * Predicate that returns true for errors that may be retried.
   * Errors that return false raise NonRetryableError immediately.
   */
  isRetryable: (err: unknown) => boolean;
  /** Hard wall-clock deadline in seconds; give up even if maxRetries not reached. */
  totalDeadlineSec: number;
}

/** Default retry configuration. */
export function defaultRetryConfig(overrides: Partial<RetryConfig> = {}): RetryConfig {
  return {
    maxRetries: 3,
    baseDelaySec: 1.0,
    maxDelaySec: 60.0,
    backoffMultiplier: 2.0,
    jitter: true,
    jitterFactor: 0.1,
    isRetryable: (err) =>
      err instanceof RateLimitError ||
      err instanceof TypeError ||         // network errors
      (err instanceof Error && err.message.toLowerCase().includes("timeout")),
    totalDeadlineSec: 300.0,
    ...overrides,
  };
}

/**
 * Compute the sleep duration (ms) before retry attempt *attempt*.
 *
 * Formula: `base * multiplier ** attempt` capped at `maxDelay`, with optional
 * ±`jitterFactor` randomness.
 */
export function calculateDelay(attempt: number, config: RetryConfig): number {
  let delay = config.baseDelaySec * Math.pow(config.backoffMultiplier, attempt);
  delay = Math.min(delay, config.maxDelaySec);

  if (config.jitter) {
    const jitterRange = delay * config.jitterFactor;
    delay += (Math.random() * 2 - 1) * jitterRange;
  }

  return Math.max(0, delay) * 1000; // return milliseconds
}

/** Sleep for *ms* milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Execute *operation* with exponential-backoff retry.
 *
 * @throws NonRetryableError  If a non-retryable error is raised.
 * @throws MaxRetriesExceeded If all attempts fail or the deadline is exceeded.
 */
export async function retryWithBackoff<T>(
  operation: () => Promise<T>,
  config: RetryConfig = defaultRetryConfig(),
): Promise<T> {
  const startTime = Date.now();
  let lastError: unknown;

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      return await operation();
    } catch (err) {
      lastError = err;

      if (!config.isRetryable(err)) {
        throw new NonRetryableError(
          `Non-retryable error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      const elapsedSec = (Date.now() - startTime) / 1000;
      if (elapsedSec >= config.totalDeadlineSec) {
        throw new MaxRetriesExceeded(
          `Total deadline of ${config.totalDeadlineSec}s exceeded after ${attempt + 1} attempt(s).`,
        );
      }

      if (attempt >= config.maxRetries) {
        throw new MaxRetriesExceeded(
          `All ${config.maxRetries + 1} attempt(s) failed. Last error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      const delayMs = calculateDelay(attempt, config);
      console.warn(
        `Retry attempt ${attempt + 1}/${config.maxRetries + 1} failed: ${err instanceof Error ? err.message : err}. Retrying in ${(delayMs / 1000).toFixed(2)}s…`,
      );
      await sleep(delayMs);
    }
  }

  throw new MaxRetriesExceeded("Unexpected: all attempts exhausted");
}

// ---------------------------------------------------------------------------
// Rate-limit-aware retry
// ---------------------------------------------------------------------------

/** RetryConfig that additionally reads Retry-After headers from 429 responses. */
export interface RateLimitAwareRetryConfig extends RetryConfig {
  /** Extract server-specified delay from a RateLimitError. */
  getDelayFromError: (err: RateLimitError) => number | null;
}

/** Default rate-limit-aware retry configuration. */
export function defaultRateLimitAwareRetryConfig(
  overrides: Partial<RateLimitAwareRetryConfig> = {},
): RateLimitAwareRetryConfig {
  return {
    ...defaultRetryConfig(),
    getDelayFromError: (err: RateLimitError): number | null => {
      const resp = err.response;
      if (!resp) return null;
      const headers =
        "headers" in resp && typeof (resp as any).headers?.get === "function"
          ? { get: (k: string) => (resp as Response).headers.get(k) }
          : { get: (k: string) => ((resp as any).headers?.[k] ?? null) };

      for (const key of ["retry-after", "x-ratelimit-reset-tokens"]) {
        const value = headers.get(key);
        if (value !== null && value !== undefined) {
          const parsed = parseFloat(value);
          if (!isNaN(parsed)) return parsed;
        }
      }
      return null;
    },
    ...overrides,
  };
}

/**
 * Like {@link retryWithBackoff} but uses server-supplied ``Retry-After`` delays
 * when a {@link RateLimitError} is raised.
 */
export async function retryWithRateLimitAwareness<T>(
  operation: () => Promise<T>,
  config: RateLimitAwareRetryConfig = defaultRateLimitAwareRetryConfig(),
): Promise<T> {
  const startTime = Date.now();

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      return await operation();
    } catch (err) {
      const elapsedSec = (Date.now() - startTime) / 1000;

      if (elapsedSec >= config.totalDeadlineSec || attempt >= config.maxRetries) {
        throw new MaxRetriesExceeded(
          `All ${config.maxRetries + 1} attempt(s) failed after ${elapsedSec.toFixed(2)}s.`,
        );
      }

      let delayMs: number;
      if (err instanceof RateLimitError) {
        const serverDelaySec = config.getDelayFromError(err);
        delayMs =
          serverDelaySec !== null
            ? serverDelaySec * 1000
            : calculateDelay(attempt, config);
        console.warn(
          `Rate limited (attempt ${attempt + 1}). Waiting ${(delayMs / 1000).toFixed(2)}s (${serverDelaySec !== null ? "server-specified" : "exponential backoff"})…`,
        );
      } else if (config.isRetryable(err)) {
        delayMs = calculateDelay(attempt, config);
        console.warn(
          `Attempt ${attempt + 1} failed. Retrying in ${(delayMs / 1000).toFixed(2)}s…`,
        );
      } else {
        throw new NonRetryableError(
          `Non-retryable error: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      await sleep(delayMs);
    }
  }

  throw new MaxRetriesExceeded("Unexpected: all attempts exhausted");
}

// ---------------------------------------------------------------------------
// Fallback
// ---------------------------------------------------------------------------

/** A single level in the fallback chain. */
export interface FallbackLevel<T = unknown> {
  /** Human-readable identifier. */
  name: string;
  /** Provider callable: receives no args (they are captured in the closure). */
  provider: () => Promise<T>;
  /** Maximum time to wait for this level, in seconds. */
  timeoutSec: number;
  /** "full" | "reduced" | "static" */
  capability: string;
  /** Relative cost (1.0 = same as primary). */
  costMultiplier: number;
}

/** Record of a single fallback level failure. */
export interface FallbackError {
  level: number;
  levelName: string;
  errorType: string;
  errorMessage: string;
}

/** Successful result from the fallback chain. */
export interface FallbackResult<T = unknown> {
  result: T;
  levelUsed: number;
  levelName: string;
  capability: string;
  attempts: number;
  totalTimeMs: number;
  errors: FallbackError[];
}

/** Aggregate statistics for a FallbackExecutor. */
export class FallbackStats {
  successByLevel: Map<string, number> = new Map();
  failureByLevel: Map<string, number> = new Map();
  failureByReason: Map<string, number> = new Map();
  exhaustionCount = 0;
  totalOperations = 0;

  recordSuccess(levelName: string): void {
    this.totalOperations++;
    this.successByLevel.set(levelName, (this.successByLevel.get(levelName) ?? 0) + 1);
  }

  recordFailure(levelName: string, reason: string): void {
    this.failureByLevel.set(levelName, (this.failureByLevel.get(levelName) ?? 0) + 1);
    this.failureByReason.set(reason, (this.failureByReason.get(reason) ?? 0) + 1);
  }

  recordExhaustion(): void {
    this.exhaustionCount++;
    this.totalOperations++;
  }

  summary(): Record<string, unknown> {
    const total = Math.max(this.totalOperations, 1);
    const firstLevelName = this.successByLevel.keys().next().value ?? "";
    const primarySuccesses = this.successByLevel.get(firstLevelName) ?? 0;
    const allSuccesses = [...this.successByLevel.values()].reduce((a, b) => a + b, 0);

    return {
      totalOperations: this.totalOperations,
      primarySuccessRate: primarySuccesses / total,
      fallbackActivationRate: (allSuccesses - primarySuccesses) / total,
      exhaustionRate: this.exhaustionCount / total,
      byLevel: Object.fromEntries(this.successByLevel),
      topFailureReasons: [...this.failureByReason.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5),
    };
  }
}

/**
 * Execute an operation through an ordered fallback chain.
 * Tries each level in order; on failure records the error and moves on.
 */
export class FallbackExecutor<T = unknown> {
  readonly levels: FallbackLevel<T>[];
  readonly stats: FallbackStats = new FallbackStats();

  constructor(levels: FallbackLevel<T>[]) {
    this.levels = levels;
  }

  /**
   * Execute *operationFactory* through the fallback chain.
   *
   * @param operationName  Label for logs and metrics.
   * @param operationFactory  Function that takes a FallbackLevel and returns
   *                          a Promise.  The level is provided so each level
   *                          can use its own provider.
   *
   * @throws AllFallbacksExhausted  If every level fails.
   */
  async execute(
    operationName: string,
    operationFactory: (level: FallbackLevel<T>) => Promise<T>,
  ): Promise<FallbackResult<T>> {
    const errors: FallbackError[] = [];
    const startTime = Date.now();

    for (let i = 0; i < this.levels.length; i++) {
      const level = this.levels[i];
      try {
        console.info(
          `Fallback [${operationName}]: Trying level ${i} (${level.name}, capability=${level.capability})`,
        );

        const result = await withTimeout(
          operationFactory(level),
          level.timeoutSec * 1000,
        );

        const totalTimeMs = Date.now() - startTime;
        this.stats.recordSuccess(level.name);

        console.info(
          `Fallback [${operationName}]: Level ${i} (${level.name}) succeeded in ${(totalTimeMs / 1000).toFixed(2)}s`,
        );

        return {
          result,
          levelUsed: i,
          levelName: level.name,
          capability: level.capability,
          attempts: errors.length + 1,
          totalTimeMs,
          errors,
        };
      } catch (err) {
        const errorEntry: FallbackError = {
          level: i,
          levelName: level.name,
          errorType: err instanceof Error ? err.constructor.name : "UnknownError",
          errorMessage: (err instanceof Error ? err.message : String(err)).slice(0, 200),
        };
        errors.push(errorEntry);
        this.stats.recordFailure(level.name, errorEntry.errorType);

        console.warn(
          `Fallback [${operationName}]: Level ${i} (${level.name}) failed: ${errorEntry.errorType}: ${errorEntry.errorMessage.slice(0, 100)}`,
        );
      }
    }

    const totalTimeMs = Date.now() - startTime;
    this.stats.recordExhaustion();

    throw new AllFallbacksExhausted(
      `All ${this.levels.length} fallback level(s) failed for '${operationName}'`,
      errors,
      totalTimeMs,
    );
  }
}

/** Race a promise against a timeout using AbortController. */
async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        controller.signal.addEventListener("abort", () =>
          reject(new Error(`Timed out after ${timeoutMs}ms`)),
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Circuit Breaker
// ---------------------------------------------------------------------------

export type CircuitState = "closed" | "open" | "half_open";

/**
 * Prevent calls to a service that is persistently failing.
 *
 * State machine:
 *   CLOSED  ──(threshold failures in window)──▶ OPEN
 *   OPEN    ──(recovery timeout elapsed)     ──▶ HALF_OPEN
 *   HALF_OPEN ──(probe succeeds)             ──▶ CLOSED
 *   HALF_OPEN ──(probe fails)               ──▶ OPEN
 */
export class CircuitBreaker {
  readonly name: string;
  readonly failureThreshold: number;
  readonly recoveryTimeoutSec: number;
  readonly halfOpenMaxRequests: number;
  readonly failureWindowSec: number;

  state: CircuitState = "closed";
  private failureTimestamps: number[] = [];
  private halfOpenRequests = 0;
  private lastStateChangeMs: number = Date.now();

  totalSuccesses = 0;
  totalFailures = 0;
  totalRejected = 0;
  timesOpened = 0;

  constructor(
    name: string,
    failureThreshold = 5,
    recoveryTimeoutSec = 120.0,
    halfOpenMaxRequests = 1,
    failureWindowSec = 60.0,
  ) {
    this.name = name;
    this.failureThreshold = failureThreshold;
    this.recoveryTimeoutSec = recoveryTimeoutSec;
    this.halfOpenMaxRequests = halfOpenMaxRequests;
    this.failureWindowSec = failureWindowSec;
  }

  /**
   * Execute *operation* through the circuit breaker.
   *
   * @throws CircuitBreakerOpenError  When the circuit is OPEN.
   */
  async call<T>(operation: () => Promise<T>): Promise<T> {
    this.maybeTransition();

    if (this.state === "open") {
      this.totalRejected++;
      throw new CircuitBreakerOpenError(
        `Circuit breaker '${this.name}' is OPEN. Recovery in ${this.recoveryRemaining().toFixed(0)}s.`,
      );
    }

    if (this.state === "half_open") {
      if (this.halfOpenRequests >= this.halfOpenMaxRequests) {
        this.totalRejected++;
        throw new CircuitBreakerOpenError(
          `Circuit breaker '${this.name}' is HALF_OPEN — probe-request limit (${this.halfOpenMaxRequests}) reached.`,
        );
      }
      this.halfOpenRequests++;
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (err) {
      this.onFailure();
      throw err;
    }
  }

  getStats(): Record<string, unknown> {
    return {
      name: this.name,
      state: this.state,
      totalSuccesses: this.totalSuccesses,
      totalFailures: this.totalFailures,
      totalRejected: this.totalRejected,
      timesOpened: this.timesOpened,
      recentFailuresInWindow: this.currentWindowFailures().length,
      failureRate:
        this.totalFailures / Math.max(this.totalSuccesses + this.totalFailures, 1),
      secondsInCurrentState: (Date.now() - this.lastStateChangeMs) / 1000,
    };
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  maybeTransition(): void {
    if (
      this.state === "open" &&
      (Date.now() - this.lastStateChangeMs) / 1000 >= this.recoveryTimeoutSec
    ) {
      this.transitionTo("half_open");
      this.halfOpenRequests = 0;
    }
  }

  private onSuccess(): void {
    this.totalSuccesses++;
    if (this.state === "half_open") {
      console.info(`Circuit breaker '${this.name}': probe succeeded — closing circuit.`);
      this.transitionTo("closed");
    }
  }

  private onFailure(): void {
    this.totalFailures++;
    this.failureTimestamps.push(Date.now());
    const recent = this.currentWindowFailures();

    if (this.state === "closed" && recent.length >= this.failureThreshold) {
      console.warn(
        `Circuit breaker '${this.name}': ${recent.length} failure(s) in ${this.failureWindowSec}s window — opening circuit.`,
      );
      this.transitionTo("open");
    } else if (this.state === "half_open") {
      console.warn(
        `Circuit breaker '${this.name}': probe failed — re-opening circuit.`,
      );
      this.transitionTo("open");
    }
  }

  private currentWindowFailures(): number[] {
    const cutoff = Date.now() - this.failureWindowSec * 1000;
    this.failureTimestamps = this.failureTimestamps.filter((t) => t >= cutoff);
    return this.failureTimestamps;
  }

  private transitionTo(newState: CircuitState): void {
    const oldState = this.state;
    this.state = newState;
    this.lastStateChangeMs = Date.now();
    if (newState === "open") this.timesOpened++;
    console.info(`Circuit breaker '${this.name}': ${oldState} → ${newState}`);
  }

  private recoveryRemaining(): number {
    const elapsed = (Date.now() - this.lastStateChangeMs) / 1000;
    return Math.max(0, this.recoveryTimeoutSec - elapsed);
  }
}

// ---------------------------------------------------------------------------
// ResilienceLayer
// ---------------------------------------------------------------------------

/** Outcome of a ResilienceLayer execution. */
export interface ResilienceResult<T = unknown> {
  result: T;
  /** "primary" | "fallback_level_N" */
  path: string;
  attempts: number;
  totalTimeMs: number;
  fallbackErrors: FallbackError[];
}

/**
 * Unified resilience wrapper: circuit breaker → retry → fallback.
 *
 * Flow:
 * 1. Check circuit breaker.
 * 2. If closed/half-open: attempt with retry.
 * 3. If retries exhausted: try fallback chain.
 * 4. If circuit open: go directly to fallback.
 * 5. If all fallbacks fail: throw SystemUnavailableError.
 */
export class ResilienceLayer<T = unknown> {
  readonly name: string;
  readonly circuitBreaker: CircuitBreaker;
  readonly retryConfig: RetryConfig;
  readonly fallbackExecutor: FallbackExecutor<T>;

  constructor(
    name: string,
    circuitBreaker: CircuitBreaker,
    retryConfig: RetryConfig,
    fallbackExecutor: FallbackExecutor<T>,
  ) {
    this.name = name;
    this.circuitBreaker = circuitBreaker;
    this.retryConfig = retryConfig;
    this.fallbackExecutor = fallbackExecutor;
  }

  /**
   * Execute *operation* with full resilience protection.
   *
   * @throws SystemUnavailableError  When every path has been exhausted.
   */
  async execute(operation: () => Promise<T>): Promise<ResilienceResult<T>> {
    const startTime = Date.now();
    let primaryError: string | undefined;

    try {
      const result = await this.circuitBreaker.call(() =>
        retryWithBackoff(operation, this.retryConfig),
      );

      const totalTimeMs = Date.now() - startTime;
      console.info(
        `Resilience [${this.name}]: primary path succeeded in ${(totalTimeMs / 1000).toFixed(2)}s`,
      );

      return {
        result,
        path: "primary",
        attempts: 1,
        totalTimeMs,
        fallbackErrors: [],
      };
    } catch (err) {
      primaryError =
        err instanceof Error ? `${err.constructor.name}: ${err.message}` : String(err);
      console.warn(
        `Resilience [${this.name}]: primary path failed (${primaryError}) — activating fallback chain.`,
      );
    }

    try {
      const fallbackResult = await this.fallbackExecutor.execute(
        this.name,
        (level) => level.provider(),
      );

      const totalTimeMs = Date.now() - startTime;
      return {
        result: fallbackResult.result,
        path: `fallback_level_${fallbackResult.levelUsed}`,
        attempts: fallbackResult.attempts,
        totalTimeMs,
        fallbackErrors: fallbackResult.errors,
      };
    } catch (err) {
      const totalTimeMs = Date.now() - startTime;
      console.error(
        `Resilience [${this.name}]: all paths exhausted in ${(totalTimeMs / 1000).toFixed(2)}s.`,
      );

      if (err instanceof AllFallbacksExhausted) {
        throw new SystemUnavailableError(
          `'${this.name}' is currently unavailable — all primary and fallback paths exhausted.`,
          primaryError ?? "unknown",
          err.errors,
        );
      }
      throw err;
    }
  }
}

// ---------------------------------------------------------------------------
// ResilienceMonitor
// ---------------------------------------------------------------------------

export interface HealthReport {
  circuitBreaker: Record<string, unknown>;
  fallback: Record<string, unknown>;
  alerts: string[];
}

/** Inspect a ResilienceLayer and surface actionable alerts. */
export class ResilienceMonitor<T = unknown> {
  static readonly PRIMARY_RATE_WARNING = 0.95;
  static readonly FALLBACK_RATE_WARNING = 0.1;
  static readonly EXHAUSTION_RATE_CRITICAL = 0.01;
  static readonly CIRCUIT_REOPEN_WARNING = 3;

  constructor(private readonly layer: ResilienceLayer<T>) {}

  checkHealth(): HealthReport {
    const circuitStats = this.layer.circuitBreaker.getStats();
    const fallbackStats = this.layer.fallbackExecutor.stats.summary();

    return {
      circuitBreaker: circuitStats,
      fallback: fallbackStats,
      alerts: this.generateAlerts(circuitStats, fallbackStats),
    };
  }

  private generateAlerts(
    circuitStats: Record<string, unknown>,
    fallbackStats: Record<string, unknown>,
  ): string[] {
    const alerts: string[] = [];

    if (circuitStats["state"] === "open") {
      alerts.push(`CRITICAL: Circuit breaker '${circuitStats["name"]}' is OPEN`);
    }
    if ((circuitStats["timesOpened"] as number) > ResilienceMonitor.CIRCUIT_REOPEN_WARNING) {
      alerts.push(
        `WARNING: Circuit breaker '${circuitStats["name"]}' has opened ${circuitStats["timesOpened"]} time(s)`,
      );
    }

    const primaryRate = fallbackStats["primarySuccessRate"] as number ?? 1;
    if (primaryRate < ResilienceMonitor.PRIMARY_RATE_WARNING) {
      alerts.push(
        `WARNING: Primary success rate is ${(primaryRate * 100).toFixed(1)}% (threshold: ${ResilienceMonitor.PRIMARY_RATE_WARNING * 100}%)`,
      );
    }

    const exhaustionRate = fallbackStats["exhaustionRate"] as number ?? 0;
    if (exhaustionRate > ResilienceMonitor.EXHAUSTION_RATE_CRITICAL) {
      alerts.push(
        `CRITICAL: Fallback exhaustion rate is ${(exhaustionRate * 100).toFixed(1)}%`,
      );
    }

    const fallbackRate = fallbackStats["fallbackActivationRate"] as number ?? 0;
    if (fallbackRate > ResilienceMonitor.FALLBACK_RATE_WARNING) {
      alerts.push(
        `WARNING: Fallback activated for ${(fallbackRate * 100).toFixed(1)}% of requests`,
      );
    }

    return alerts;
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function demo(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("RESILIENCE LAYER DEMO (TypeScript)");
  console.log("=".repeat(60));

  // Scenario 1: Happy path
  console.log("\n--- Scenario 1: Happy path ---");
  const layer1 = new ResilienceLayer<string>(
    "demo",
    new CircuitBreaker("demo-cb", 3),
    defaultRetryConfig({ maxRetries: 2, baseDelaySec: 0.01 }),
    new FallbackExecutor<string>([]),
  );

  try {
    const res = await layer1.execute(async () => "Hello from primary!");
    console.log(`Result: ${res.result}  path=${res.path}`);
  } catch (err) {
    console.error(`Unavailable: ${err}`);
  }

  // Scenario 2: Transient failure → retry succeeds
  console.log("\n--- Scenario 2: Transient failure → retry ---");
  let callCount2 = 0;
  const layer2 = new ResilienceLayer<string>(
    "demo2",
    new CircuitBreaker("demo2-cb", 5),
    defaultRetryConfig({ maxRetries: 3, baseDelaySec: 0.01 }),
    new FallbackExecutor<string>([]),
  );

  try {
    const res2 = await layer2.execute(async () => {
      callCount2++;
      if (callCount2 === 1) throw new Error("timeout");
      return "Recovered!";
    });
    console.log(`Result: ${res2.result}  path=${res2.path}`);
  } catch (err) {
    console.error(`Error: ${err}`);
  }

  // Scenario 3: All paths exhausted → SystemUnavailableError
  console.log("\n--- Scenario 3: SystemUnavailableError ---");
  const layer3 = new ResilienceLayer<string>(
    "demo3",
    new CircuitBreaker("demo3-cb", 10),
    defaultRetryConfig({ maxRetries: 1, baseDelaySec: 0.01 }),
    new FallbackExecutor<string>([
      {
        name: "fallback",
        provider: async () => {
          throw new Error("fallback also failed");
        },
        timeoutSec: 5,
        capability: "static",
        costMultiplier: 0,
      },
    ]),
  );

  try {
    await layer3.execute(async () => {
      throw new Error("primary failed");
    });
  } catch (err) {
    if (err instanceof SystemUnavailableError) {
      console.log(`SystemUnavailableError (expected): ${err.message}`);
    } else {
      console.error(`Unexpected error: ${err}`);
    }
  }

  console.log("\nDemo complete.\n");
}

if (require.main === module) {
  demo().catch(console.error);
}
