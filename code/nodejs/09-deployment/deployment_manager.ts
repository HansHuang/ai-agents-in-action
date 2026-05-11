/**
 * Deployment Manager for AI Agents in Production (TypeScript)
 * ============================================================
 * Port of code/python/09-deployment/deployment_manager.py
 *
 * Includes:
 *   - DeploymentManager    — gradual rollout & feature flags
 *   - CanaryDeployer       — canary evaluation & routing
 *   - ProductionCostController — per-user & total daily budgets
 *   - MultiRegionDeployer  — geographic & GDPR-aware routing
 *   - RollbackManager      — multi-artifact rollback
 *
 * Reference: docs/09-from-dev-to-production/01-deployment-strategies.md
 */

import crypto from "crypto";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VersionMetrics {
  error_rate: number;
  p50_latency: number;
  p95_latency: number;
  p99_latency: number;
  avg_cost: number;
  safety_block_rate: number;
  task_success_rate: number;
  user_satisfaction: number;
}

export interface CanaryEvaluation {
  canary_pct: number;
  has_issues: boolean;
  issues: string[];
  stable_metrics: VersionMetrics;
  canary_metrics: VersionMetrics;
  recommendation: string;
}

export interface BudgetCheck {
  allowed: boolean;
  reason?: string;
  current_user_cost: number;
  user_budget: number;
}

export interface RollbackItemResult {
  name: string;
  success: boolean;
  error?: string;
}

export interface RollbackResult {
  reason: string;
  items: RollbackItemResult[];
  total_time_seconds: number;
  success: boolean;
  timestamp: number;
}

export interface RegionConfig {
  llm_provider: string;
  fallback_provider: string;
  vector_db_endpoint: string;
  latency_to_provider_ms: number;
  eu_residency_compliant: boolean;
}

// ---------------------------------------------------------------------------
// Stubs (replace with real implementations)
// ---------------------------------------------------------------------------

/** Minimal in-memory feature-flag service. */
class FeatureFlagService {
  private flags = new Map<string, string | number>();
  private internalUsers = new Set(["internal-001", "internal-002", "dev-team"]);

  isInternalUser(userId: string): boolean {
    return this.internalUsers.has(userId);
  }

  getString(key: string, defaultValue = ""): string {
    return String(this.flags.get(key) ?? defaultValue);
  }

  getInt(key: string, defaultValue = 0): number {
    return Number(this.flags.get(key) ?? defaultValue);
  }

  setInt(key: string, value: number): void {
    this.flags.set(key, value);
  }

  setString(key: string, value: string): void {
    this.flags.set(key, value);
  }
}

/** Minimal version-keyed metrics store. */
class HarnessMetrics {
  private data = new Map<string, {
    requests: number; errors: number; latencies: number[];
    costs: number[]; safetyBlocks: number; taskSuccesses: number;
  }>();

  record(version: string, opts: {
    error?: boolean; latency?: number; cost?: number;
    safetyBlocked?: boolean; taskSuccess?: boolean;
  } = {}): void {
    if (!this.data.has(version)) {
      this.data.set(version, {
        requests: 0, errors: 0, latencies: [],
        costs: [], safetyBlocks: 0, taskSuccesses: 0,
      });
    }
    const m = this.data.get(version)!;
    m.requests++;
    if (opts.error) m.errors++;
    m.latencies.push(opts.latency ?? 1.0);
    m.costs.push(opts.cost ?? 0.02);
    if (opts.safetyBlocked) m.safetyBlocks++;
    if (opts.taskSuccess !== false) m.taskSuccesses++;
  }

  getMetricsForVersion(version: string): VersionMetrics {
    const m = this.data.get(version);
    if (!m || m.requests === 0) {
      return {
        error_rate: 0, p50_latency: 1, p95_latency: 2, p99_latency: 3,
        avg_cost: 0.02, safety_block_rate: 0, task_success_rate: 1,
        user_satisfaction: 0.9,
      };
    }
    const sorted = [...m.latencies].sort((a, b) => a - b);
    const n = sorted.length;
    const pct = (p: number) => sorted[Math.min(Math.floor(n * p), n - 1)];
    return {
      error_rate: m.errors / m.requests,
      p50_latency: pct(0.50),
      p95_latency: pct(0.95),
      p99_latency: pct(0.99),
      avg_cost: m.costs.reduce((a, b) => a + b, 0) / m.costs.length,
      safety_block_rate: m.safetyBlocks / m.requests,
      task_success_rate: m.taskSuccesses / m.requests,
      user_satisfaction: 0.85,
    };
  }
}

// ---------------------------------------------------------------------------
// 1. Deployment Manager
// ---------------------------------------------------------------------------

/**
 * Manages gradual rollout of new agent versions.
 *
 * Rollout stages:
 *   Stage 0: Internal only (0% external)
 *   Stage 1: 1% canary
 *   Stage 2: 5% extended canary
 *   Stage 3: 25% beta
 *   Stage 4: 100% full rollout
 */
export class DeploymentManager {
  static readonly ROLLOUT_STAGES = [0, 1, 5, 25, 100] as const;

  private halted = false;
  private haltReason?: string;

  constructor(private readonly flags: FeatureFlagService) {
    flags.setInt("canary_rollout_pct", 0);
    flags.setString("stable_version", "v3.1.0");
    flags.setString("canary_version", "v3.2.1");
    flags.setString("internal_version", "v3.2.1");
  }

  /**
   * Determine which agent version a user should receive.
   *
   * Resolution order:
   *   1. Internal users always get the internal (latest) version.
   *   2. If rollout is halted, everyone gets stable.
   *   3. Otherwise, hash user_id to determine canary bucket.
   */
  getAgentVersion(userId: string): string {
    if (this.flags.isInternalUser(userId)) {
      return this.flags.getString("internal_version", "v3.2.1");
    }
    if (this.halted) {
      return this.flags.getString("stable_version", "v3.1.0");
    }
    const rolloutPct = this.flags.getInt("canary_rollout_pct", 0);
    return this._userInRolloutGroup(userId, rolloutPct)
      ? this.flags.getString("canary_version", "v3.2.1")
      : this.flags.getString("stable_version", "v3.1.0");
  }

  /**
   * Deterministic assignment: MD5(userId) mod 100 < percentage.
   */
  _userInRolloutGroup(userId: string, percentage: number): boolean {
    const hash = crypto.createHash("md5").update(userId).digest("hex");
    const value = parseInt(hash.slice(0, 8), 16);
    return value % 100 < percentage;
  }

  /** Increase canary rollout percentage. */
  promoteRollout(fromPct: number, toPct: number): void {
    const stages = DeploymentManager.ROLLOUT_STAGES as readonly number[];
    if (!stages.includes(toPct)) {
      throw new Error(`toPct must be one of ${stages.join(", ")}`);
    }
    const current = this.flags.getInt("canary_rollout_pct", 0);
    if (current !== fromPct) {
      throw new Error(`Current rollout is ${current}%, not ${fromPct}%`);
    }
    const canary = this.flags.getString("canary_version");
    console.log(`Promoting ${canary} rollout: ${fromPct}% → ${toPct}%`);
    this.flags.setInt("canary_rollout_pct", toPct);
    this.halted = false;
  }

  /** Immediately return all external users to stable. */
  haltRollout(reason: string): void {
    console.warn(`ROLLOUT HALTED: ${reason}`);
    this.halted = true;
    this.haltReason = reason;
    this.flags.setInt("canary_rollout_pct", 0);
  }

  /** Current rollout state. */
  getRolloutStatus(): Record<string, unknown> {
    return {
      stable_version: this.flags.getString("stable_version"),
      canary_version: this.flags.getString("canary_version"),
      internal_version: this.flags.getString("internal_version"),
      canary_rollout_pct: this.flags.getInt("canary_rollout_pct"),
      halted: this.halted,
      halt_reason: this.haltReason ?? null,
    };
  }
}

// ---------------------------------------------------------------------------
// 2. Canary Deployer
// ---------------------------------------------------------------------------

/** Stub agent harness. */
class AgentHarness {
  constructor(public readonly version: string) {}
  async process(input: string, userId: string): Promise<{ content: string; metadata: Record<string, string> }> {
    return { content: `[${this.version}] Response to: ${input}`, metadata: {} };
  }
}

/**
 * Routes requests to stable or canary harness and evaluates canary health.
 */
export class CanaryDeployer {
  constructor(
    private readonly stable: AgentHarness,
    private readonly canary: AgentHarness,
    private readonly metrics: HarnessMetrics,
    private readonly dm: DeploymentManager,
  ) {}

  async process(userInput: string, userId: string): Promise<ReturnType<AgentHarness["process"]>> {
    const version = this.dm.getAgentVersion(userId);
    const canaryVersion = (this.dm as any).flags.getString("canary_version");

    if (version === canaryVersion) {
      const r = await this.canary.process(userInput, userId);
      r.metadata["version"] = "canary";
      this.metrics.record("canary", { latency: 1.0, cost: 0.025 });
      return r;
    }
    const r = await this.stable.process(userInput, userId);
    r.metadata["version"] = "stable";
    this.metrics.record("stable", { latency: 0.9, cost: 0.020 });
    return r;
  }

  /** Compare canary vs stable across 5 dimensions. */
  evaluateCanary(): CanaryEvaluation {
    const stable = this.metrics.getMetricsForVersion("stable");
    const canary = this.metrics.getMetricsForVersion("canary");
    const pct = (this.dm as any).flags.getInt("canary_rollout_pct");

    const issues: string[] = [];
    if (!this._checkErrorRate(stable, canary)) {
      issues.push(`Error rate: stable=${(stable.error_rate * 100).toFixed(1)}%, canary=${(canary.error_rate * 100).toFixed(1)}%`);
    }
    if (!this._checkLatency(stable, canary)) {
      issues.push(`P95 latency: stable=${stable.p95_latency.toFixed(1)}s, canary=${canary.p95_latency.toFixed(1)}s`);
    }
    if (!this._checkCost(stable, canary)) {
      issues.push(`Avg cost: stable=$${stable.avg_cost.toFixed(3)}, canary=$${canary.avg_cost.toFixed(3)}`);
    }
    if (!this._checkSafety(stable, canary)) {
      issues.push(`Safety block rate: stable=${(stable.safety_block_rate * 100).toFixed(1)}%, canary=${(canary.safety_block_rate * 100).toFixed(1)}%`);
    }
    if (!this._checkTaskSuccess(stable, canary)) {
      issues.push(`Task success: stable=${(stable.task_success_rate * 100).toFixed(1)}%, canary=${(canary.task_success_rate * 100).toFixed(1)}%`);
    }

    return {
      canary_pct: pct,
      has_issues: issues.length > 0,
      issues,
      stable_metrics: stable,
      canary_metrics: canary,
      recommendation: this._generateRecommendation(issues),
    };
  }

  /** Canary error rate must be ≤ 1.5× stable. */
  _checkErrorRate(stable: VersionMetrics, canary: VersionMetrics): boolean {
    if (stable.error_rate === 0) return canary.error_rate === 0;
    return canary.error_rate <= stable.error_rate * 1.5;
  }

  /** Canary P95 latency must be ≤ 1.2× stable. */
  _checkLatency(stable: VersionMetrics, canary: VersionMetrics): boolean {
    return canary.p95_latency <= stable.p95_latency * 1.2;
  }

  /** Canary avg cost must be ≤ 1.2× stable. */
  _checkCost(stable: VersionMetrics, canary: VersionMetrics): boolean {
    return canary.avg_cost <= stable.avg_cost * 1.2;
  }

  /** Canary safety block rate must be ≤ 1.5× stable. */
  _checkSafety(stable: VersionMetrics, canary: VersionMetrics): boolean {
    if (stable.safety_block_rate === 0) return canary.safety_block_rate === 0;
    return canary.safety_block_rate <= stable.safety_block_rate * 1.5;
  }

  /** Canary task success rate must be ≥ 0.95× stable. */
  _checkTaskSuccess(stable: VersionMetrics, canary: VersionMetrics): boolean {
    return canary.task_success_rate >= stable.task_success_rate * 0.95;
  }

  _generateRecommendation(issues: string[]): string {
    if (issues.length === 0) return "Canary is healthy. Consider increasing rollout percentage.";
    if (issues.length === 1) return "Minor issues detected. Monitor for another hour before promoting.";
    return "Significant issues detected. Halt rollout and investigate.";
  }
}

// ---------------------------------------------------------------------------
// 3. Production Cost Controller
// ---------------------------------------------------------------------------

export interface CostConfig {
  user_daily_budget: number;
  total_daily_budget: number;
  max_cost_per_request: number;
  free_tier_daily_budget: number;
  enterprise_daily_budget: number;
}

export const DEFAULT_COST_CONFIG: CostConfig = {
  user_daily_budget: 10.0,
  total_daily_budget: 1000.0,
  max_cost_per_request: 1.0,
  free_tier_daily_budget: 0.50,
  enterprise_daily_budget: 50.0,
};

/**
 * Enforce per-user and total daily budget limits.
 *
 * Alert thresholds: warning=70%, critical=90%, shutdown=100%.
 */
export class ProductionCostController {
  private userDailyCosts = new Map<string, number>();
  private totalDailyCost = 0;
  private freeTierUsers = new Set<string>();
  private enterpriseUsers = new Set<string>();
  private lastAlertLevel?: string;

  private readonly alertThresholds = {
    warning: 0.70,
    critical: 0.90,
    shutdown: 1.00,
  } as const;

  constructor(private readonly config: CostConfig = DEFAULT_COST_CONFIG) {}

  registerFreeTier(userId: string): void { this.freeTierUsers.add(userId); }
  registerEnterprise(userId: string): void { this.enterpriseUsers.add(userId); }

  private userBudget(userId: string): number {
    if (this.enterpriseUsers.has(userId)) return this.config.enterprise_daily_budget;
    if (this.freeTierUsers.has(userId)) return this.config.free_tier_daily_budget;
    return this.config.user_daily_budget;
  }

  /** Pre-request budget check. */
  checkBudget(userId: string, estimatedCost: number): BudgetCheck {
    const currentUser = this.userDailyCosts.get(userId) ?? 0;
    const userBudget = this.userBudget(userId);

    if (!this._checkRequestLimit(estimatedCost)) {
      return {
        allowed: false,
        reason: `Request estimated cost $${estimatedCost.toFixed(3)} exceeds per-request limit $${this.config.max_cost_per_request.toFixed(2)}.`,
        current_user_cost: currentUser,
        user_budget: userBudget,
      };
    }
    if (!this._checkUserBudget(userId, estimatedCost)) {
      return {
        allowed: false,
        reason: `Daily budget of $${userBudget.toFixed(2)} exceeded. Current: $${currentUser.toFixed(2)}.`,
        current_user_cost: currentUser,
        user_budget: userBudget,
      };
    }
    if (!this._checkTotalBudget(estimatedCost)) {
      return {
        allowed: false,
        reason: "Service temporarily unavailable due to high demand. Please try again later.",
        current_user_cost: currentUser,
        user_budget: userBudget,
      };
    }
    return { allowed: true, current_user_cost: currentUser, user_budget: userBudget };
  }

  _checkUserBudget(userId: string, estimated: number): boolean {
    const current = this.userDailyCosts.get(userId) ?? 0;
    return current + estimated <= this.userBudget(userId);
  }

  _checkTotalBudget(estimated: number): boolean {
    return this.totalDailyCost + estimated <= this.config.total_daily_budget;
  }

  _checkRequestLimit(estimated: number): boolean {
    return estimated <= this.config.max_cost_per_request;
  }

  /** Record actual cost after request completion. */
  recordCost(userId: string, cost: number): void {
    this.userDailyCosts.set(userId, (this.userDailyCosts.get(userId) ?? 0) + cost);
    this.totalDailyCost += cost;
    this._checkAndFireAlerts();
  }

  private _checkAndFireAlerts(): void {
    const budget = this.config.total_daily_budget;
    if (budget <= 0) return;
    const pct = this.totalDailyCost / budget;
    if (pct >= this.alertThresholds.shutdown) {
      this._triggerAlert("critical", `Daily budget exhausted: $${this.totalDailyCost.toFixed(2)}`);
    } else if (pct >= this.alertThresholds.critical) {
      this._triggerAlert("warning", `Daily budget at ${(pct * 100).toFixed(0)}%: $${this.totalDailyCost.toFixed(2)}`);
    } else if (pct >= this.alertThresholds.warning) {
      this._triggerAlert("info", `Daily budget at ${(pct * 100).toFixed(0)}%: $${this.totalDailyCost.toFixed(2)}`);
    }
  }

  _triggerAlert(level: string, message: string): void {
    if (level !== this.lastAlertLevel) {
      console.warn(`[COST ALERT ${level.toUpperCase()}] ${message}`);
      this.lastAlertLevel = level;
    }
  }

  /** Daily cost summary. */
  getCostReport(): Record<string, unknown> {
    const budget = this.config.total_daily_budget || 1;
    const topUsers = [...this.userDailyCosts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
    return {
      total_daily_cost: this.totalDailyCost,
      daily_budget: this.config.total_daily_budget,
      budget_remaining: this.config.total_daily_budget - this.totalDailyCost,
      pct_used: this.totalDailyCost / budget,
      top_users: topUsers,
      user_count: this.userDailyCosts.size,
      avg_cost_per_user: this.totalDailyCost / Math.max(this.userDailyCosts.size, 1),
    };
  }

  /** Reset daily counters (call at midnight). */
  resetDailyCosts(): void {
    this.userDailyCosts.clear();
    this.totalDailyCost = 0;
    this.lastAlertLevel = undefined;
    console.log("Daily costs reset.");
  }
}

// ---------------------------------------------------------------------------
// 4. Multi-Region Deployer
// ---------------------------------------------------------------------------

const REGIONS: Record<string, RegionConfig> = {
  "us-east": {
    llm_provider: "openai",
    fallback_provider: "anthropic",
    vector_db_endpoint: "https://us-east.qdrant.example.com",
    latency_to_provider_ms: 50,
    eu_residency_compliant: false,
  },
  "eu-west": {
    llm_provider: "openai",
    fallback_provider: "anthropic",
    vector_db_endpoint: "https://eu-west.qdrant.example.com",
    latency_to_provider_ms: 80,
    eu_residency_compliant: true,
  },
  "ap-southeast": {
    llm_provider: "anthropic",
    fallback_provider: "openai",
    vector_db_endpoint: "https://ap-se.qdrant.example.com",
    latency_to_provider_ms: 120,
    eu_residency_compliant: false,
  },
};

const GEO_MAP: Array<[string, string]> = [
  ["52.", "us-east"],
  ["18.", "us-east"],
  ["34.", "us-east"],
  ["35.", "eu-west"],
  ["13.", "ap-southeast"],
];

const EU_PREFIXES = ["195.", "212.", "217.", "82.", "185.", "37.", "31."];

/**
 * Route users to the nearest healthy region, respecting data-residency rules.
 */
export class MultiRegionDeployer {
  private circuitBreakers = new Map<string, boolean>(
    Object.keys(REGIONS).map((r) => [r, false])
  );

  getRegion(userIp: string, userPreferences?: Record<string, string>): string {
    if (userPreferences?.["region"] && REGIONS[userPreferences["region"]] &&
        this._isRegionHealthy(userPreferences["region"])) {
      return userPreferences["region"];
    }
    const region = this._isEuUser(userIp) ? "eu-west" : this._geoRoute(userIp);
    return this._isRegionHealthy(region) ? region : this._getNearestHealthyRegion(region);
  }

  _isEuUser(ip: string): boolean {
    return EU_PREFIXES.some((prefix) => ip.startsWith(prefix));
  }

  _geoRoute(ip: string): string {
    for (const [prefix, region] of GEO_MAP) {
      if (ip.startsWith(prefix)) return region;
    }
    return "us-east";
  }

  _isRegionHealthy(region: string): boolean {
    return !(this.circuitBreakers.get(region) ?? false);
  }

  _getNearestHealthyRegion(exclude: string): string {
    const candidates = Object.entries(REGIONS)
      .filter(([r]) => r !== exclude && this._isRegionHealthy(r))
      .sort((a, b) => a[1].latency_to_provider_ms - b[1].latency_to_provider_ms);
    if (candidates.length === 0) throw new Error("No healthy regions available");
    return candidates[0][0];
  }

  getRegionConfig(region: string): RegionConfig {
    const config = REGIONS[region];
    if (!config) throw new Error(`Unknown region: ${region}`);
    return { ...config };
  }

  openCircuitBreaker(region: string): void {
    console.warn(`Circuit breaker OPEN for region: ${region}`);
    this.circuitBreakers.set(region, true);
  }

  closeCircuitBreaker(region: string): void {
    console.log(`Circuit breaker CLOSED for region: ${region}`);
    this.circuitBreakers.set(region, false);
  }
}

// ---------------------------------------------------------------------------
// 5. Rollback Manager
// ---------------------------------------------------------------------------

interface RollbackItem {
  name: string;
  description: string;
  method: () => Promise<void>;
  time_seconds: number;
}

/**
 * Manage rollbacks for AI agent deployments.
 */
export class RollbackManager {
  private readonly items: RollbackItem[];
  private history: RollbackResult[] = [];

  constructor(private readonly dm: DeploymentManager) {
    this.items = [
      {
        name: "config",
        description: "Revert harness configuration to previous version",
        method: this._rollbackConfig.bind(this),
        time_seconds: 10,
      },
      {
        name: "prompt",
        description: "Revert system prompt to previous version",
        method: this._rollbackPrompt.bind(this),
        time_seconds: 10,
      },
      {
        name: "model",
        description: "Switch to previous model version",
        method: this._rollbackModel.bind(this),
        time_seconds: 30,
      },
      {
        name: "tools",
        description: "Revert tool definitions/implementations",
        method: this._rollbackTools.bind(this),
        time_seconds: 30,
      },
      {
        name: "code",
        description: "Revert application code to previous git commit",
        method: this._rollbackCode.bind(this),
        time_seconds: 60,
      },
      {
        name: "documents",
        description: "Revert knowledge base and re-embed",
        method: this._rollbackDocuments.bind(this),
        time_seconds: 300,
      },
    ];
  }

  async rollback(reason: string, itemNames?: string[]): Promise<RollbackResult> {
    console.warn(`ROLLBACK INITIATED: ${reason}`);
    this.dm.haltRollout(reason);

    let candidates = this.items;
    if (itemNames) {
      const nameSet = new Set(itemNames);
      candidates = this.items.filter((i) => nameSet.has(i.name));
    }
    const sorted = [...candidates].sort((a, b) => a.time_seconds - b.time_seconds);

    const results: RollbackItemResult[] = [];
    for (const item of sorted) {
      console.log(`Rolling back: ${item.name} …`);
      try {
        await item.method();
        results.push({ name: item.name, success: true });
        console.log(`Rollback complete: ${item.name}`);
      } catch (err) {
        const error = err instanceof Error ? err.message : String(err);
        results.push({ name: item.name, success: false, error });
        console.error(`Rollback failed: ${item.name}: ${error}`);
      }
    }

    const totalTime = sorted.reduce((s, i) => s + i.time_seconds, 0);
    const result: RollbackResult = {
      reason,
      items: results,
      total_time_seconds: totalTime,
      success: results.every((r) => r.success),
      timestamp: Date.now() / 1000,
    };
    this.history.push(result);
    return result;
  }

  private async _rollbackCode(): Promise<void> {
    console.log("[stub] git revert HEAD && kubectl rollout restart …");
  }

  private async _rollbackModel(): Promise<void> {
    console.log("[stub] Reverting model config to previous version …");
  }

  private async _rollbackPrompt(): Promise<void> {
    console.log("[stub] Loading previous prompt version from library …");
  }

  private async _rollbackConfig(): Promise<void> {
    console.log("[stub] Restoring harness config from previous snapshot …");
  }

  private async _rollbackTools(): Promise<void> {
    console.log("[stub] Restoring tool definitions from previous release …");
  }

  private async _rollbackDocuments(): Promise<void> {
    console.log("[stub] Restoring previous document snapshot and re-embedding …");
  }

  getHistory(): RollbackResult[] {
    return [...this.history];
  }

  async testRollback(): Promise<boolean> {
    console.log("DRY-RUN: testing all rollback methods …");
    try {
      for (const item of this.items) await item.method();
      console.log("DRY-RUN: all rollback methods succeeded.");
      return true;
    } catch (err) {
      console.error(`DRY-RUN: rollback test failed: ${err}`);
      return false;
    }
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("  AI Agent Deployment Manager (TypeScript) — Demo");
  console.log("=".repeat(60));

  const flags = new FeatureFlagService();
  const dm = new DeploymentManager(flags);
  const metrics = new HarnessMetrics();
  const stable = new AgentHarness("v3.1.0");
  const canary = new AgentHarness("v3.2.1");
  const deployer = new CanaryDeployer(stable, canary, metrics, dm);
  const costCtrl = new ProductionCostController();
  const multiRegion = new MultiRegionDeployer();
  const rollbackMgr = new RollbackManager(dm);

  // Gradual rollout
  console.log("\n--- GRADUAL ROLLOUT ---");
  console.log("Stage 0:", dm.getRolloutStatus());
  console.log("  internal-001 →", dm.getAgentVersion("internal-001"));
  console.log("  external-42  →", dm.getAgentVersion("external-42"));

  dm.promoteRollout(0, 1);
  console.log("\nStage 1 (1%):", dm.getRolloutStatus());

  // Simulate traffic
  for (let i = 0; i < 20; i++) {
    await deployer.process("Hello", `user-${String(i).padStart(4, "0")}`);
  }

  let evaluation = deployer.evaluateCanary();
  console.log(`  Canary eval: has_issues=${evaluation.has_issues}, recommendation='${evaluation.recommendation}'`);

  // Simulate error spike
  dm.promoteRollout(1, 5);
  metrics.record("canary", { error: true, latency: 3.5, cost: 0.04 });
  metrics.record("canary", { error: true, latency: 3.8, cost: 0.04 });
  evaluation = deployer.evaluateCanary();
  console.log("\nStage 2 (5%) with error spike:");
  console.log(`  Issues: ${evaluation.issues}`);
  console.log(`  Recommendation: ${evaluation.recommendation}`);

  if (evaluation.has_issues) {
    dm.haltRollout("Error rate spike at 5%");
    console.log("  Halted:", dm.getRolloutStatus());
  }

  // Rollback
  console.log("\n--- ROLLBACK ---");
  const rbResult = await rollbackMgr.rollback("Error rate exceeded threshold", ["config", "prompt"]);
  console.log(`  Success: ${rbResult.success}, items: ${rbResult.items.map((i) => i.name)}`);

  // Multi-region routing
  console.log("\n--- MULTI-REGION ROUTING ---");
  for (const [ip, label] of [["52.86.1.1", "US"], ["195.50.10.1", "EU (GDPR)"], ["13.250.1.1", "AP"]]) {
    const region = multiRegion.getRegion(ip);
    const cfg = multiRegion.getRegionConfig(region);
    console.log(`  ${label} (${ip}) → ${region} (provider: ${cfg.llm_provider}, ${cfg.latency_to_provider_ms}ms)`);
  }

  // Cost controller
  console.log("\n--- COST CONTROLLER ---");
  costCtrl.registerFreeTier("free-user-1");
  for (const [userId, cost] of [["premium-1", 0.03], ["free-user-1", 0.60], ["premium-1", 0.03]] as const) {
    const check = costCtrl.checkBudget(userId, cost);
    const status = check.allowed ? "✓ allowed" : `✗ rejected: ${check.reason}`;
    console.log(`  ${userId} ($${cost}) → ${status}`);
    if (check.allowed) costCtrl.recordCost(userId, cost);
  }

  console.log("\n" + "=".repeat(60));
}

main().catch(console.error);
