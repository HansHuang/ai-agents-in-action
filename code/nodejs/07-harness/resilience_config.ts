/**
 * Resilience configuration profiles for the ResilienceLayer.
 *
 * Provides ready-made profiles for common deployment contexts plus an
 * SLO-derived builder that works backwards from availability targets.
 * See: docs/07-harness-engineering/01-the-harness-mindset.md
 */

// ---------------------------------------------------------------------------
// ResilienceConfig
// ---------------------------------------------------------------------------

export interface ResilienceConfig {
  profileName: string;
  description: string;
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
  totalDeadlineMs: number;
  circuitFailureThreshold: number;
  circuitRecoveryMs: number;
  circuitFailureWindowMs: number;
  fallbackCapabilities: string[];
  notes: string[];
}

// ---------------------------------------------------------------------------
// Pre-built profiles
// ---------------------------------------------------------------------------

/** User-facing API: fast failures, moderate retries, smooth degradation. */
export function forUserFacingApi(): ResilienceConfig {
  return {
    profileName: "user_facing_api",
    description: "Interactive API where latency matters as much as correctness",
    maxRetries: 2,
    baseDelayMs: 500,
    maxDelayMs: 8_000,
    backoffMultiplier: 2.0,
    totalDeadlineMs: 15_000,
    circuitFailureThreshold: 5,
    circuitRecoveryMs: 30_000,
    circuitFailureWindowMs: 60_000,
    fallbackCapabilities: ["full", "reduced_quality", "cached_response"],
    notes: ["Fail fast to preserve UX", "Always have a cached fallback"],
  };
}

/** Background job: more patience, higher correctness requirement. */
export function forBackgroundJob(): ResilienceConfig {
  return {
    profileName: "background_job",
    description: "Async tasks where latency is less critical than correctness",
    maxRetries: 5,
    baseDelayMs: 2_000,
    maxDelayMs: 120_000,
    backoffMultiplier: 2.5,
    totalDeadlineMs: 600_000,
    circuitFailureThreshold: 10,
    circuitRecoveryMs: 120_000,
    circuitFailureWindowMs: 300_000,
    fallbackCapabilities: ["full", "partial", "skip_and_log"],
    notes: ["Can afford longer retries", "Circuit threshold is higher"],
  };
}

/** Critical path: highest resilience, never degrade silently. */
export function forCriticalPath(): ResilienceConfig {
  return {
    profileName: "critical_path",
    description: "Financial or safety-critical operations",
    maxRetries: 3,
    baseDelayMs: 1_000,
    maxDelayMs: 30_000,
    backoffMultiplier: 2.0,
    totalDeadlineMs: 60_000,
    circuitFailureThreshold: 3,
    circuitRecoveryMs: 60_000,
    circuitFailureWindowMs: 120_000,
    fallbackCapabilities: ["full", "conservative_estimate"],
    notes: ["Alert on every fallback", "Prefer rejection over wrong answer"],
  };
}

/** Cost-sensitive: minimal retries, cheap fallbacks. */
export function forCostSensitive(): ResilienceConfig {
  return {
    profileName: "cost_sensitive",
    description: "Minimize spend even at the cost of availability",
    maxRetries: 1,
    baseDelayMs: 1_000,
    maxDelayMs: 10_000,
    backoffMultiplier: 1.5,
    totalDeadlineMs: 20_000,
    circuitFailureThreshold: 3,
    circuitRecoveryMs: 60_000,
    circuitFailureWindowMs: 60_000,
    fallbackCapabilities: ["cheap_model", "cached_response"],
    notes: ["Use cheaper models as fallbacks", "Cache aggressively"],
  };
}

// ---------------------------------------------------------------------------
// SLO-derived builder
// ---------------------------------------------------------------------------

/**
 * Derive a resilience config from SLO targets.
 * @param targetAvailability  e.g. 0.999 for 99.9%
 * @param targetLatencyP99Ms  P99 latency budget in ms
 */
export function fromSlo(targetAvailability: number, targetLatencyP99Ms: number): ResilienceConfig {
  // Max attempts = ceil(log(1 - availability) / log(assumed_per_attempt_failure_rate))
  const assumedFailureRate = 0.1;
  const maxRetries = Math.min(
    5,
    Math.ceil(Math.log(1 - targetAvailability) / Math.log(assumedFailureRate))
  );

  // Distribute latency budget across retries with exponential backoff
  const baseDelayMs = Math.max(200, Math.floor(targetLatencyP99Ms / (maxRetries * 4)));
  const maxDelayMs = Math.min(targetLatencyP99Ms / 2, baseDelayMs * 8);

  return {
    profileName: "slo_derived",
    description: `Derived from ${(targetAvailability * 100).toFixed(2)}% availability, P99 ${targetLatencyP99Ms}ms`,
    maxRetries,
    baseDelayMs,
    maxDelayMs,
    backoffMultiplier: 2.0,
    totalDeadlineMs: targetLatencyP99Ms,
    circuitFailureThreshold: Math.max(3, maxRetries + 1),
    circuitRecoveryMs: 60_000,
    circuitFailureWindowMs: 120_000,
    fallbackCapabilities: ["full", "degraded"],
    notes: [`Auto-derived for ${(targetAvailability * 100).toFixed(3)}% availability`],
  };
}

/** Print a config summary. */
export function printResilienceConfig(cfg: ResilienceConfig): void {
  console.log(`\n[${cfg.profileName}] ${cfg.description}`);
  console.log(`  Retries: ${cfg.maxRetries}, delay ${cfg.baseDelayMs}–${cfg.maxDelayMs}ms (×${cfg.backoffMultiplier})`);
  console.log(`  Deadline: ${cfg.totalDeadlineMs}ms`);
  console.log(`  Circuit: opens after ${cfg.circuitFailureThreshold} failures, recovers in ${cfg.circuitRecoveryMs}ms`);
  console.log(`  Fallbacks: ${cfg.fallbackCapabilities.join(" → ")}`);
  cfg.notes.forEach((n) => console.log(`  Note: ${n}`));
}

// Demo
function main(): void {
  const profiles = [forUserFacingApi(), forBackgroundJob(), forCriticalPath(), forCostSensitive()];
  profiles.forEach(printResilienceConfig);
  console.log("\n--- SLO-derived ---");
  printResilienceConfig(fromSlo(0.999, 2_000));
}

main();
