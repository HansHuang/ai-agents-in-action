/**
 * Real-time harness monitoring and alerting.
 *
 * Collects metrics from every harness response, maintains sliding-window
 * aggregates, and fires alerts when health thresholds are breached.
 *
 * HarnessMetrics  — raw metric collection
 * HarnessMonitor  — computes dashboard data
 * HarnessAlerter  — checks thresholds and dispatches alerts
 * See: docs/07-harness-engineering/01-the-harness-mindset.md
 */

// ---------------------------------------------------------------------------
// Alert types
// ---------------------------------------------------------------------------

export type AlertSeverity = "info" | "warning" | "critical";

export interface Alert {
  severity: AlertSeverity;
  title: string;
  detail: string;
  metric: string;
  value: number;
  threshold: number;
  firedAt: number;
}

// ---------------------------------------------------------------------------
// HarnessMetrics
// ---------------------------------------------------------------------------

export interface MetricRecord {
  timestamp: number;
  durationMs: number;
  tokensUsed: number;
  cost: number;
  finalState: string;
  intent?: string;
  guardrailBlocked: boolean;
  approvalRequired: boolean;
}

export class HarnessMetrics {
  private windowMs: number;
  private records: MetricRecord[] = [];

  constructor(windowMs = 60_000) {
    this.windowMs = windowMs;
  }

  record(m: MetricRecord): void {
    this.records.push(m);
    this.evict();
  }

  private evict(): void {
    const cutoff = Date.now() - this.windowMs;
    this.records = this.records.filter((r) => r.timestamp >= cutoff);
  }

  get recent(): MetricRecord[] {
    this.evict();
    return this.records;
  }

  get count(): number { return this.recent.length; }
  get totalTokens(): number { return this.recent.reduce((s, r) => s + r.tokensUsed, 0); }
  get totalCost(): number { return this.recent.reduce((s, r) => s + r.cost, 0); }
  get avgLatencyMs(): number {
    const r = this.recent;
    return r.length ? r.reduce((s, rec) => s + rec.durationMs, 0) / r.length : 0;
  }
  get errorRate(): number {
    const r = this.recent;
    return r.length ? r.filter((rec) => rec.finalState === "error").length / r.length : 0;
  }
  get blockedRate(): number {
    const r = this.recent;
    return r.length ? r.filter((rec) => rec.guardrailBlocked).length / r.length : 0;
  }
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export interface DashboardData {
  timestamp: number;
  windowMs: number;
  requestCount: number;
  requestsPerMinute: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  errorRate: number;
  blockedRate: number;
  approvalRate: number;
  totalCostUsd: number;
  avgTokensPerRequest: number;
  intentDistribution: Record<string, number>;
}

export class HarnessMonitor {
  constructor(private metrics: HarnessMetrics) {}

  dashboard(): DashboardData {
    const r = this.metrics.recent;
    const windowSec = 60;

    const latencies = r.map((rec) => rec.durationMs).sort((a, b) => a - b);
    const p95Idx = Math.floor(latencies.length * 0.95);

    const intentDist: Record<string, number> = {};
    for (const rec of r) {
      if (rec.intent) intentDist[rec.intent] = (intentDist[rec.intent] ?? 0) + 1;
    }

    return {
      timestamp: Date.now(),
      windowMs: windowSec * 1000,
      requestCount: this.metrics.count,
      requestsPerMinute: this.metrics.count / (windowSec / 60),
      avgLatencyMs: this.metrics.avgLatencyMs,
      p95LatencyMs: latencies[p95Idx] ?? 0,
      errorRate: this.metrics.errorRate,
      blockedRate: this.metrics.blockedRate,
      approvalRate: r.length ? r.filter((rec) => rec.approvalRequired).length / r.length : 0,
      totalCostUsd: this.metrics.totalCost,
      avgTokensPerRequest: this.metrics.count ? this.metrics.totalTokens / this.metrics.count : 0,
      intentDistribution: intentDist,
    };
  }

  printDashboard(): void {
    const d = this.dashboard();
    console.log("\n=== Harness Monitor Dashboard ===");
    console.log(`  Requests: ${d.requestCount} (${d.requestsPerMinute.toFixed(1)}/min)`);
    console.log(`  Latency: avg ${d.avgLatencyMs.toFixed(0)}ms, P95 ${d.p95LatencyMs.toFixed(0)}ms`);
    console.log(`  Error rate: ${(d.errorRate * 100).toFixed(1)}%`);
    console.log(`  Blocked rate: ${(d.blockedRate * 100).toFixed(1)}%`);
    console.log(`  Approval rate: ${(d.approvalRate * 100).toFixed(1)}%`);
    console.log(`  Cost: $${d.totalCostUsd.toFixed(4)} (${d.avgTokensPerRequest.toFixed(0)} tokens/req avg)`);
    if (Object.keys(d.intentDistribution).length) {
      console.log("  Intents:", JSON.stringify(d.intentDistribution));
    }
  }
}

// ---------------------------------------------------------------------------
// Alerter
// ---------------------------------------------------------------------------

export interface AlertThresholds {
  maxErrorRate: number;
  maxLatencyMs: number;
  maxCostPerWindowUsd: number;
  maxBlockedRate: number;
}

const DEFAULT_THRESHOLDS: AlertThresholds = {
  maxErrorRate: 0.05,
  maxLatencyMs: 10_000,
  maxCostPerWindowUsd: 1.0,
  maxBlockedRate: 0.20,
};

export class HarnessAlerter {
  private fired: Alert[] = [];
  private handlers: Array<(a: Alert) => void> = [(a) => console.error(`[ALERT:${a.severity}] ${a.title}: ${a.detail}`)];

  constructor(private thresholds: AlertThresholds = DEFAULT_THRESHOLDS) {}

  addHandler(fn: (a: Alert) => void): void {
    this.handlers.push(fn);
  }

  check(data: DashboardData): Alert[] {
    const newAlerts: Alert[] = [];

    const checks: Array<{ metric: string; value: number; threshold: number; title: string; detail: string; severity: AlertSeverity }> = [
      { metric: "error_rate", value: data.errorRate, threshold: this.thresholds.maxErrorRate, title: "High Error Rate", detail: `${(data.errorRate * 100).toFixed(1)}% errors`, severity: "critical" },
      { metric: "latency_p95", value: data.p95LatencyMs, threshold: this.thresholds.maxLatencyMs, title: "High P95 Latency", detail: `${data.p95LatencyMs.toFixed(0)}ms`, severity: "warning" },
      { metric: "cost_per_window", value: data.totalCostUsd, threshold: this.thresholds.maxCostPerWindowUsd, title: "Cost Spike", detail: `$${data.totalCostUsd.toFixed(4)}`, severity: "warning" },
      { metric: "blocked_rate", value: data.blockedRate, threshold: this.thresholds.maxBlockedRate, title: "High Block Rate", detail: `${(data.blockedRate * 100).toFixed(1)}% blocked`, severity: "info" },
    ];

    for (const c of checks) {
      if (c.value > c.threshold) {
        const alert: Alert = { ...c, firedAt: Date.now() };
        newAlerts.push(alert);
        this.fired.push(alert);
        this.handlers.forEach((h) => h(alert));
      }
    }

    return newAlerts;
  }

  get allFired(): Alert[] { return this.fired; }
}
