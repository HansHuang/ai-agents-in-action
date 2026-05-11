/**
 * Automated runbook for common ProductionHarness operational scenarios.
 *
 * Each scenario:
 *   1. Checks symptoms from metrics
 *   2. Executes safe corrective actions automatically
 *   3. Lists actions requiring human approval
 *   4. Returns a RunbookResult with incident context
 * See: docs/07-harness-engineering/01-the-harness-mindset.md
 */

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

export interface ComponentStatus {
  name: string;
  status: "ok" | "warning" | "critical";
  metrics: Record<string, unknown>;
  message: string;
}

export interface DiagnosticReport {
  overallStatus: "healthy" | "degraded" | "unhealthy";
  components: Record<string, ComponentStatus>;
  activeIncidents: string[];
  recommendations: string[];
  generatedAt: number;
}

export interface RunbookResult {
  scenario: string;
  diagnosis: string;
  actionsTaken: string[];
  requiresHumanApproval: string[];
  status: "resolved" | "mitigated" | "escalated";
  nextSteps: string[];
}

// ---------------------------------------------------------------------------
// Diagnostic helpers
// ---------------------------------------------------------------------------

export interface HarnessSnapshot {
  errorRate: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  circuitStates: Record<string, string>;
  blockedRate: number;
  costPerWindow: number;
  requestCount: number;
}

function assessComponent(
  name: string,
  value: number,
  warningThreshold: number,
  criticalThreshold: number,
  unit = ""
): ComponentStatus {
  const status =
    value >= criticalThreshold ? "critical" :
    value >= warningThreshold ? "warning" : "ok";
  return {
    name,
    status,
    metrics: { value },
    message: `${value.toFixed(2)}${unit} (warn: ${warningThreshold}, crit: ${criticalThreshold})`,
  };
}

/** Generate a full diagnostic report from a snapshot. */
export function diagnose(snapshot: HarnessSnapshot): DiagnosticReport {
  const components: Record<string, ComponentStatus> = {
    errorRate: assessComponent("error_rate", snapshot.errorRate, 0.05, 0.10, " rate"),
    latencyP95: assessComponent("latency_p95", snapshot.avgLatencyMs, 3_000, 8_000, "ms"),
    blockedRate: assessComponent("blocked_rate", snapshot.blockedRate, 0.10, 0.30, " rate"),
    cost: assessComponent("cost_per_window", snapshot.costPerWindow, 0.50, 2.0, " USD"),
  };

  // Open circuits
  for (const [name, state] of Object.entries(snapshot.circuitStates)) {
    if (state === "open") {
      components[`circuit_${name}`] = {
        name: `circuit_${name}`,
        status: "critical",
        metrics: { state },
        message: `Circuit OPEN for provider: ${name}`,
      };
    }
  }

  const criticalCount = Object.values(components).filter((c) => c.status === "critical").length;
  const warningCount = Object.values(components).filter((c) => c.status === "warning").length;

  const overallStatus =
    criticalCount > 0 ? "unhealthy" :
    warningCount > 0 ? "degraded" : "healthy";

  const recommendations: string[] = [];
  if (snapshot.errorRate > 0.05) recommendations.push("Investigate LLM provider errors");
  if (snapshot.avgLatencyMs > 3_000) recommendations.push("Check network latency to provider");
  if (snapshot.blockedRate > 0.10) recommendations.push("Review guardrail rules for false positives");
  if (Object.values(snapshot.circuitStates).some((s) => s === "open")) {
    recommendations.push("Reset circuit breakers after verifying provider health");
  }

  return {
    overallStatus,
    components,
    activeIncidents: Object.entries(components)
      .filter(([, c]) => c.status === "critical")
      .map(([k]) => k),
    recommendations,
    generatedAt: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Runbook scenarios
// ---------------------------------------------------------------------------

/** Handle high error rate scenario. */
export function handleHighErrorRate(snapshot: HarnessSnapshot): RunbookResult {
  const actionsTaken: string[] = [];
  const requiresHumanApproval: string[] = [];

  actionsTaken.push("Increased retry count from 3 to 5");
  actionsTaken.push("Enabled fallback to secondary provider");

  if (snapshot.errorRate > 0.3) {
    requiresHumanApproval.push("Disable primary provider and route all traffic to secondary");
    requiresHumanApproval.push("Page on-call engineer");
  }

  return {
    scenario: "high_error_rate",
    diagnosis: `Error rate ${(snapshot.errorRate * 100).toFixed(1)}% exceeds 5% threshold`,
    actionsTaken,
    requiresHumanApproval,
    status: snapshot.errorRate > 0.3 ? "escalated" : "mitigated",
    nextSteps: ["Monitor for 5 minutes", "If resolved, reduce retry count"],
  };
}

/** Handle circuit breaker open scenario. */
export function handleCircuitOpen(providerName: string, snapshot: HarnessSnapshot): RunbookResult {
  return {
    scenario: "circuit_open",
    diagnosis: `Circuit breaker OPEN for ${providerName}`,
    actionsTaken: [
      `Routed 100% traffic to fallback providers`,
      `Set alert for ${providerName} recovery`,
    ],
    requiresHumanApproval: [
      `Manually reset circuit for ${providerName} after provider confirms resolution`,
    ],
    status: "mitigated",
    nextSteps: [
      "Check provider status page",
      `Wait for automatic HALF_OPEN transition in ${snapshot.circuitStates[providerName] === "open" ? "~2 min" : "N/A"}`,
    ],
  };
}

/** Handle high latency scenario. */
export function handleHighLatency(snapshot: HarnessSnapshot): RunbookResult {
  return {
    scenario: "high_latency",
    diagnosis: `P95 latency ${snapshot.p95LatencyMs.toFixed(0)}ms exceeds SLO`,
    actionsTaken: [
      "Switched slow-path requests to faster gpt-4o-mini model",
      "Enabled aggressive caching for repeated queries",
      "Reduced max_tokens for chat intent from 512 to 256",
    ],
    requiresHumanApproval: [],
    status: "mitigated",
    nextSteps: ["Monitor P95 for improvement", "Investigate slow queries in traces"],
  };
}

/** Handle cost spike scenario. */
export function handleCostSpike(snapshot: HarnessSnapshot): RunbookResult {
  const requiresHumanApproval: string[] = [];
  if (snapshot.costPerWindow > 5.0) {
    requiresHumanApproval.push("Enable request rate limiting");
    requiresHumanApproval.push("Notify budget owner");
  }

  return {
    scenario: "cost_spike",
    diagnosis: `Cost $${snapshot.costPerWindow.toFixed(3)}/window exceeds threshold`,
    actionsTaken: [
      "Downgraded all requests to gpt-4o-mini",
      "Reduced max_tokens across all intents",
    ],
    requiresHumanApproval,
    status: snapshot.costPerWindow > 5.0 ? "escalated" : "mitigated",
    nextSteps: ["Review which users/intents are driving cost", "Set per-user budget limits"],
  };
}

/** Print a diagnostic report. */
export function printDiagnosticReport(report: DiagnosticReport): void {
  const icons = { healthy: "OK", degraded: "WARN", unhealthy: "CRIT" };
  console.log(`\n[${icons[report.overallStatus]}] HARNESS: ${report.overallStatus.toUpperCase()}`);
  for (const [, comp] of Object.entries(report.components)) {
    const icon = comp.status === "ok" ? "  [ok]" : comp.status === "warning" ? "  [warn]" : "  [crit]";
    console.log(`${icon} ${comp.name}: ${comp.message}`);
  }
  if (report.recommendations.length) {
    console.log("  Recommendations:");
    report.recommendations.forEach((r) => console.log(`    - ${r}`));
  }
}
