/**
 * Multi-level LLM provider fallback chain with circuit breakers.
 *
 * Principle: every LLM call needs a fallback path. Providers are tried
 * in priority order; failures trigger automatic promotion to the next level.
 *
 * Circuit breaker states:
 *   CLOSED → OPEN (after threshold failures in window)
 *   OPEN → HALF_OPEN (after cooldown expires)
 *   HALF_OPEN → CLOSED (on success) | OPEN (on failure)
 *
 * See: docs/07-harness-engineering/01-the-harness-mindset.md
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LLMResponse {
  content: string;
  toolCalls?: unknown[];
  tokensUsed: number;
  model: string;
  providerName: string;
  fallbackLevel: number;
}

export type ProviderFn = (
  messages: Array<{ role: string; content: string }>,
  options: { model: string; maxTokens: number; temperature: number }
) => Promise<LLMResponse>;

export interface FallbackProvider {
  name: string;
  fn: ProviderFn;
  model: string;
  maxTokens: number;
  temperature: number;
  priority: number;     // Lower number = tried first
}

export class AllProvidersFailedError extends Error {
  constructor(public errors: string[]) {
    super(`All providers failed:\n${errors.map((e) => `  - ${e}`).join("\n")}`);
    this.name = "AllProvidersFailedError";
  }
}

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

type CircuitState = "closed" | "open" | "half_open";

export class CircuitBreaker {
  private state: CircuitState = "closed";
  private failureTimestamps: number[] = [];
  private openedAt?: number;

  constructor(
    private threshold = 5,
    private windowMs = 60_000,
    private cooldownMs = 120_000
  ) {}

  get currentState(): CircuitState {
    if (this.state === "open") {
      const elapsed = Date.now() - (this.openedAt ?? 0);
      if (elapsed >= this.cooldownMs) {
        this.state = "half_open";
      }
    }
    return this.state;
  }

  allowRequest(): boolean {
    return this.currentState !== "open";
  }

  recordSuccess(): void {
    this.state = "closed";
    this.failureTimestamps = [];
  }

  recordFailure(): void {
    const now = Date.now();
    this.failureTimestamps = this.failureTimestamps.filter((t) => now - t < this.windowMs);
    this.failureTimestamps.push(now);

    if (this.state === "half_open" || this.failureTimestamps.length >= this.threshold) {
      this.state = "open";
      this.openedAt = now;
    }
  }
}

// ---------------------------------------------------------------------------
// FallbackChain
// ---------------------------------------------------------------------------

export interface FallbackAttempt {
  provider: string;
  fallbackLevel: number;
  success: boolean;
  error?: string;
  durationMs: number;
}

export interface FallbackResult {
  response: LLMResponse;
  attempts: FallbackAttempt[];
  usedFallback: boolean;
  finalLevel: number;
}

export class FallbackChain {
  private circuits = new Map<string, CircuitBreaker>();

  constructor(
    private providers: FallbackProvider[],
    private circuitOptions: { threshold?: number; windowMs?: number; cooldownMs?: number } = {}
  ) {
    const sorted = [...providers].sort((a, b) => a.priority - b.priority);
    this.providers = sorted;
    for (const p of sorted) {
      this.circuits.set(p.name, new CircuitBreaker(
        circuitOptions.threshold,
        circuitOptions.windowMs,
        circuitOptions.cooldownMs
      ));
    }
  }

  async call(
    messages: Array<{ role: string; content: string }>
  ): Promise<FallbackResult> {
    const attempts: FallbackAttempt[] = [];
    const errors: string[] = [];

    for (let level = 0; level < this.providers.length; level++) {
      const provider = this.providers[level];
      const circuit = this.circuits.get(provider.name)!;

      if (!circuit.allowRequest()) {
        attempts.push({
          provider: provider.name,
          fallbackLevel: level,
          success: false,
          error: "Circuit open — skipped",
          durationMs: 0,
        });
        continue;
      }

      const start = Date.now();
      try {
        const response = await provider.fn(messages, {
          model: provider.model,
          maxTokens: provider.maxTokens,
          temperature: provider.temperature,
        });
        circuit.recordSuccess();
        attempts.push({
          provider: provider.name,
          fallbackLevel: level,
          success: true,
          durationMs: Date.now() - start,
        });
        return {
          response: { ...response, fallbackLevel: level },
          attempts,
          usedFallback: level > 0,
          finalLevel: level,
        };
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        circuit.recordFailure();
        errors.push(`${provider.name}: ${errorMsg}`);
        attempts.push({
          provider: provider.name,
          fallbackLevel: level,
          success: false,
          error: errorMsg,
          durationMs: Date.now() - start,
        });
      }
    }

    throw new AllProvidersFailedError(errors);
  }

  getCircuitStates(): Record<string, string> {
    const states: Record<string, string> = {};
    this.circuits.forEach((cb, name) => {
      states[name] = cb.currentState;
    });
    return states;
  }
}
