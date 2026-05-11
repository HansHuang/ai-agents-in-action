/**
 * Approval Policy Optimizer.
 *
 * Analyzes historical approval decisions to identify policy improvements that
 * reduce unnecessary human review while maintaining safety guarantees.
 * See: docs/07-harness-engineering/06-human-in-the-loop.md
 */

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

export interface ApprovalRecord {
  timestamp: number;
  action: string;
  params: Record<string, unknown>;
  riskLevel: "low" | "medium" | "high" | "critical";
  estimatedCost: number;
  decision: "approved" | "rejected" | "approved_with_edits" | "auto_rejected";
  reviewerId: string;
  responseTimeSec: number;
  reviewerNotes?: string;
}

export interface Recommendation {
  title: string;
  description: string;
  impact: "high" | "medium" | "low";
  estimatedMonthlySaved: number;
  estimatedRiskChange: string;
  supportingData: Record<string, unknown>;
}

export interface OptimizationReport {
  periodStart: string;
  periodEnd: string;
  totalDecisions: number;
  recommendations: Recommendation[];
  summaryText: string;
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

const AUTO_APPROVAL_THRESHOLD = 0.95;  // >= 95% approved → auto-approve candidate
const IMPROVEMENT_THRESHOLD   = 0.30;  // >= 30% rejected → agent logic needs work
const EDIT_THRESHOLD          = 0.20;  // >= 20% edits    → bad parameter defaults

export class ApprovalPolicyOptimizer {
  constructor(private data: ApprovalRecord[]) {}

  analyze(): OptimizationReport {
    const recs: Recommendation[] = [
      ...this.findAutoApprovalCandidates(),
      ...this.findAgentImprovementAreas(),
      ...this.findThresholdAdjustments(),
      ...this.findWorkloadIssues(),
    ];

    // Sort: high impact first, then most reviews saved
    const impactOrder = { high: 0, medium: 1, low: 2 };
    recs.sort((a, b) => (impactOrder[a.impact] - impactOrder[b.impact]) || (b.estimatedMonthlySaved - a.estimatedMonthlySaved));

    const dates = this.data.map((r) => r.timestamp);
    const start = dates.length ? new Date(Math.min(...dates)).toISOString().slice(0, 10) : "N/A";
    const end = dates.length ? new Date(Math.max(...dates)).toISOString().slice(0, 10) : "N/A";

    return {
      periodStart: start,
      periodEnd: end,
      totalDecisions: this.data.length,
      recommendations: recs,
      summaryText: this.buildSummary(recs),
    };
  }

  private byAction(): Map<string, ApprovalRecord[]> {
    const m = new Map<string, ApprovalRecord[]>();
    for (const r of this.data) {
      if (!m.has(r.action)) m.set(r.action, []);
      m.get(r.action)!.push(r);
    }
    return m;
  }

  findAutoApprovalCandidates(): Recommendation[] {
    const recs: Recommendation[] = [];
    for (const [action, records] of this.byAction()) {
      const approved = records.filter((r) => r.decision === "approved").length;
      const rate = approved / records.length;
      if (records.length >= 10 && rate >= AUTO_APPROVAL_THRESHOLD) {
        recs.push({
          title: `Auto-approve "${action}"`,
          description: `${(rate * 100).toFixed(1)}% of "${action}" requests are approved without edits`,
          impact: "high",
          estimatedMonthlySaved: Math.round(records.length * 4),
          estimatedRiskChange: "Minimal — low-risk action with consistent approval",
          supportingData: { action, approvalRate: rate, sampleSize: records.length },
        });
      }
    }
    return recs;
  }

  findAgentImprovementAreas(): Recommendation[] {
    const recs: Recommendation[] = [];
    for (const [action, records] of this.byAction()) {
      const rejected = records.filter((r) => r.decision === "rejected").length;
      const rate = rejected / records.length;
      if (records.length >= 5 && rate >= IMPROVEMENT_THRESHOLD) {
        recs.push({
          title: `Improve agent logic for "${action}"`,
          description: `${(rate * 100).toFixed(1)}% of "${action}" requests are rejected — agent proposes bad actions`,
          impact: "medium",
          estimatedMonthlySaved: Math.round(records.length * rate * 0.8 * 4),
          estimatedRiskChange: "Positive — fewer bad actions proposed",
          supportingData: { action, rejectionRate: rate, sampleSize: records.length },
        });
      }
    }
    return recs;
  }

  findThresholdAdjustments(): Recommendation[] {
    const approved = this.data.filter((r) => r.decision === "approved" && r.estimatedCost > 0);
    if (approved.length === 0) return [];

    const costs = approved.map((r) => r.estimatedCost).sort((a, b) => a - b);
    const p10Idx = Math.floor(costs.length * 0.10);
    const p10Cost = costs[p10Idx] ?? 0;

    if (p10Cost > 0.01) {
      return [{
        title: "Raise low-cost auto-approval threshold",
        description: `10th percentile approved cost is $${p10Cost.toFixed(3)} — auto-approve requests below this`,
        impact: "medium",
        estimatedMonthlySaved: Math.round(approved.length * 0.1 * 4),
        estimatedRiskChange: "Low — only affects lowest-cost requests",
        supportingData: { p10Cost, sampleSize: approved.length },
      }];
    }
    return [];
  }

  findWorkloadIssues(): Recommendation[] {
    const responseTimes = this.data.map((r) => r.responseTimeSec);
    const avg = responseTimes.reduce((s, v) => s + v, 0) / (responseTimes.length || 1);
    const slow = this.data.filter((r) => r.responseTimeSec > 600).length;

    if (avg > 180) {
      return [{
        title: "High reviewer response times",
        description: `Average response time is ${(avg / 60).toFixed(1)}min — ${slow} requests exceeded 10min`,
        impact: avg > 300 ? "high" : "low",
        estimatedMonthlySaved: 0,
        estimatedRiskChange: "Operational — need more reviewers or async handling",
        supportingData: { avgResponseTimeSec: avg, slowRequests: slow },
      }];
    }
    return [];
  }

  private buildSummary(recs: Recommendation[]): string {
    const highImpact = recs.filter((r) => r.impact === "high").length;
    const totalSaved = recs.reduce((s, r) => s + r.estimatedMonthlySaved, 0);
    return `Analysis of ${this.data.length} decisions found ${recs.length} recommendations (${highImpact} high-impact). Estimated ${totalSaved} reviews/month could be eliminated.`;
  }
}

/** Generate synthetic approval history for demos. */
export function generateSyntheticHistory(n = 200): ApprovalRecord[] {
  const actions = ["send_email", "update_database", "make_purchase", "create_ticket", "delete_data"];
  const decisions: ApprovalRecord["decision"][] = ["approved", "approved", "approved", "rejected", "approved_with_edits"];
  const records: ApprovalRecord[] = [];
  for (let i = 0; i < n; i++) {
    records.push({
      timestamp: Date.now() - Math.random() * 30 * 86400 * 1000,
      action: actions[Math.floor(Math.random() * actions.length)],
      params: {},
      riskLevel: ["low", "medium", "high"][Math.floor(Math.random() * 3)] as ApprovalRecord["riskLevel"],
      estimatedCost: Math.random() * 10,
      decision: decisions[Math.floor(Math.random() * decisions.length)],
      reviewerId: `reviewer-${Math.floor(Math.random() * 5) + 1}`,
      responseTimeSec: Math.random() * 600,
    });
  }
  return records;
}

/** Print an optimization report. */
export function printOptimizationReport(report: OptimizationReport): void {
  console.log(`\n=== Approval Policy Optimization Report ===`);
  console.log(`  Period: ${report.periodStart} → ${report.periodEnd}`);
  console.log(`  Decisions analyzed: ${report.totalDecisions}`);
  console.log(`\n  Summary: ${report.summaryText}`);
  console.log("\n  Recommendations:");
  for (const rec of report.recommendations) {
    console.log(`\n  [${rec.impact.toUpperCase()}] ${rec.title}`);
    console.log(`    ${rec.description}`);
    console.log(`    Saves ~${rec.estimatedMonthlySaved} reviews/month`);
    console.log(`    Risk: ${rec.estimatedRiskChange}`);
  }
}

// Demo
function main(): void {
  const history = generateSyntheticHistory(150);
  const optimizer = new ApprovalPolicyOptimizer(history);
  const report = optimizer.analyze();
  printOptimizationReport(report);
}

main();
