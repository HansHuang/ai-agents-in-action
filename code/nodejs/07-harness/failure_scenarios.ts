/**
 * Simulated failure scenarios for testing harness resilience.
 *
 * Provides controllable failure injection for:
 *   - Network timeouts and connection errors
 *   - Rate limiting (HTTP 429)
 *   - Partial responses and malformed JSON
 *   - LLM hallucination patterns
 *   - Cascading failures
 * See: docs/07-harness-engineering/01-the-harness-mindset.md
 */

// ---------------------------------------------------------------------------
// Failure types
// ---------------------------------------------------------------------------

export type FailureType =
  | "timeout"
  | "rate_limit"
  | "network_error"
  | "partial_response"
  | "malformed_json"
  | "hallucination"
  | "empty_response"
  | "server_error";

export interface FailureScenario {
  name: string;
  failureType: FailureType;
  description: string;
  probability: number;      // 0-1: how often this failure fires
  delayMs: number;          // Simulated delay before failure
  retryable: boolean;
}

// ---------------------------------------------------------------------------
// Built-in scenarios
// ---------------------------------------------------------------------------

export const SCENARIOS: Record<string, FailureScenario> = {
  intermittentTimeout: {
    name: "intermittent_timeout",
    failureType: "timeout",
    description: "LLM API times out ~20% of requests",
    probability: 0.2,
    delayMs: 5_000,
    retryable: true,
  },
  highRateLimit: {
    name: "high_rate_limit",
    failureType: "rate_limit",
    description: "Sustained 429 rate limiting",
    probability: 0.5,
    delayMs: 100,
    retryable: true,
  },
  networkFlap: {
    name: "network_flap",
    failureType: "network_error",
    description: "Intermittent DNS/TCP failures",
    probability: 0.15,
    delayMs: 50,
    retryable: true,
  },
  malformedJson: {
    name: "malformed_json",
    failureType: "malformed_json",
    description: "API returns truncated JSON",
    probability: 0.05,
    delayMs: 200,
    retryable: false,
  },
  emptyResponse: {
    name: "empty_response",
    failureType: "empty_response",
    description: "API returns empty content",
    probability: 0.03,
    delayMs: 100,
    retryable: false,
  },
  cascadingFailure: {
    name: "cascading_failure",
    failureType: "server_error",
    description: "All retries fail — tests circuit breaker",
    probability: 1.0,
    delayMs: 100,
    retryable: false,
  },
};

// ---------------------------------------------------------------------------
// FailureInjector
// ---------------------------------------------------------------------------

export interface InjectionResult {
  injected: boolean;
  scenario?: FailureScenario;
  error?: Error;
}

/**
 * Inject failures according to a scenario's probability.
 * Returns whether a failure was injected and what error to throw.
 */
export function injectFailure(scenario: FailureScenario): InjectionResult {
  if (Math.random() > scenario.probability) {
    return { injected: false };
  }

  let error: Error;
  switch (scenario.failureType) {
    case "timeout":
      error = new Error(`Request timeout after ${scenario.delayMs}ms`);
      error.name = "TimeoutError";
      break;
    case "rate_limit":
      error = new Error("HTTP 429: Too Many Requests. Retry-After: 30");
      error.name = "RateLimitError";
      break;
    case "network_error":
      error = new Error("ECONNREFUSED: Connection refused");
      error.name = "NetworkError";
      break;
    case "malformed_json":
      error = new SyntaxError('Unexpected end of JSON input: {"partial": tru');
      break;
    case "server_error":
      error = new Error("HTTP 503: Service Unavailable");
      error.name = "ServerError";
      break;
    default:
      error = new Error(`Simulated failure: ${scenario.failureType}`);
  }

  return { injected: true, scenario, error };
}

/**
 * Wrap an async function with failure injection for testing.
 */
export async function withFailureInjection<T>(
  scenario: FailureScenario,
  fn: () => Promise<T>
): Promise<T> {
  const { injected, error } = injectFailure(scenario);
  if (injected) {
    if (scenario.delayMs > 0) {
      await new Promise((r) => setTimeout(r, scenario.delayMs));
    }
    throw error;
  }
  return fn();
}

// ---------------------------------------------------------------------------
// Scenario runner for testing
// ---------------------------------------------------------------------------

export interface ScenarioResult {
  scenario: string;
  attempts: number;
  successes: number;
  failures: number;
  successRate: number;
  failureTypes: Record<string, number>;
}

/** Run a scenario N times and report success rate. */
export async function runScenario(
  scenario: FailureScenario,
  mockFn: () => Promise<string>,
  trials = 100
): Promise<ScenarioResult> {
  let successes = 0;
  let failures = 0;
  const failureTypes: Record<string, number> = {};

  for (let i = 0; i < trials; i++) {
    try {
      await withFailureInjection(scenario, mockFn);
      successes++;
    } catch (e) {
      failures++;
      const name = e instanceof Error ? e.name : "UnknownError";
      failureTypes[name] = (failureTypes[name] ?? 0) + 1;
    }
  }

  return {
    scenario: scenario.name,
    attempts: trials,
    successes,
    failures,
    successRate: successes / trials,
    failureTypes,
  };
}

/** Print scenario results. */
export function printScenarioResults(results: ScenarioResult[]): void {
  console.log("\n=== Failure Scenario Results ===");
  for (const r of results) {
    console.log(
      `  [${r.scenario}] ${(r.successRate * 100).toFixed(1)}% success ` +
      `(${r.successes}/${r.attempts})`
    );
  }
}
