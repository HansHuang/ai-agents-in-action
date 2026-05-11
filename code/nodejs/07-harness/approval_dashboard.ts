/**
 * Approval Dashboard — terminal-based reviewer interface.
 *
 * Tracks a human-in-the-loop approval queue, records decisions,
 * and reports metrics for review operations.
 * See: docs/07-harness-engineering/06-human-in-the-loop.md
 */

// ---------------------------------------------------------------------------
// Types (mirrors human_in_the_loop.ts concepts)
// ---------------------------------------------------------------------------

export type RiskLevel = "low" | "medium" | "high" | "critical";
export type ApprovalDecision = "approved" | "rejected" | "approved_with_edits" | "auto_rejected";

export interface ApprovalRequest {
  requestId: string;
  userId: string;
  proposedAction: string;
  params: Record<string, unknown>;
  riskLevel: RiskLevel;
  estimatedCost: number;
  reasoning: string;
  createdAt: number;
  deadlineMs?: number;
}

export interface ApprovalResponse {
  requestId: string;
  decision: ApprovalDecision;
  reviewerId: string;
  reviewerNotes?: string;
  editedParams?: Record<string, unknown>;
  decidedAt: number;
}

// ---------------------------------------------------------------------------
// Queue & metrics
// ---------------------------------------------------------------------------

export interface DashboardMetrics {
  totalReviewed: number;
  approvedCount: number;
  rejectedCount: number;
  approvedWithEditsCount: number;
  autoRejectedCount: number;
  approvalRate: number;
  avgResponseTimeSec: number;
  pendingCount: number;
}

export interface HistoryEntry {
  timestamp: number;
  action: string;
  requestId: string;
  decision: ApprovalDecision;
  reviewerId: string;
  cost: number;
}

export class ApprovalDashboard {
  private queue: ApprovalRequest[] = [];
  private history: HistoryEntry[] = [];
  private responseTimes: number[] = [];

  constructor(private reviewerId = "dashboard-reviewer") {}

  /** Add a request to the approval queue. */
  enqueue(request: ApprovalRequest): void {
    this.queue.push(request);
  }

  /** Get the queue sorted by risk (critical first), then wait time. */
  getSortedQueue(sortBy: "risk" | "wait" | "cost" = "risk"): ApprovalRequest[] {
    const riskOrder: Record<RiskLevel, number> = { critical: 0, high: 1, medium: 2, low: 3 };
    return [...this.queue].sort((a, b) => {
      if (sortBy === "risk") return (riskOrder[a.riskLevel] ?? 9) - (riskOrder[b.riskLevel] ?? 9);
      if (sortBy === "wait") return a.createdAt - b.createdAt;
      if (sortBy === "cost") return b.estimatedCost - a.estimatedCost;
      return 0;
    });
  }

  /** Record a decision for a request. */
  decide(requestId: string, decision: ApprovalDecision, notes?: string): ApprovalResponse | null {
    const idx = this.queue.findIndex((r) => r.requestId === requestId);
    if (idx === -1) return null;
    const request = this.queue.splice(idx, 1)[0];
    const now = Date.now();
    const elapsed = (now - request.createdAt) / 1000;
    this.responseTimes.push(elapsed);

    this.history.push({
      timestamp: now,
      action: request.proposedAction,
      requestId,
      decision,
      reviewerId: this.reviewerId,
      cost: request.estimatedCost,
    });

    return { requestId, decision, reviewerId: this.reviewerId, reviewerNotes: notes, decidedAt: now };
  }

  /** Check for expired requests and auto-reject them. */
  processExpired(): ApprovalResponse[] {
    const now = Date.now();
    const expired = this.queue.filter((r) => r.deadlineMs && now > r.deadlineMs);
    const responses: ApprovalResponse[] = [];
    for (const r of expired) {
      const resp = this.decide(r.requestId, "auto_rejected", "Deadline exceeded");
      if (resp) responses.push(resp);
    }
    return responses;
  }

  getMetrics(): DashboardMetrics {
    const counts = { approved: 0, rejected: 0, approved_with_edits: 0, auto_rejected: 0 };
    for (const h of this.history) {
      counts[h.decision] = (counts[h.decision] ?? 0) + 1;
    }
    const total = this.history.length;
    const avgResponseTime = this.responseTimes.length
      ? this.responseTimes.reduce((s, v) => s + v, 0) / this.responseTimes.length
      : 0;

    return {
      totalReviewed: total,
      approvedCount: counts.approved,
      rejectedCount: counts.rejected,
      approvedWithEditsCount: counts.approved_with_edits,
      autoRejectedCount: counts.auto_rejected,
      approvalRate: total > 0 ? counts.approved / total : 0,
      avgResponseTimeSec: avgResponseTime,
      pendingCount: this.queue.length,
    };
  }

  /** Print a summary dashboard. */
  printDashboard(): void {
    const metrics = this.getMetrics();
    const sorted = this.getSortedQueue();

    console.log("\n=== Approval Dashboard ===");
    console.log(`  Pending: ${metrics.pendingCount}  |  Reviewed: ${metrics.totalReviewed}`);
    console.log(`  Approval rate: ${(metrics.approvalRate * 100).toFixed(1)}%  |  Avg response: ${metrics.avgResponseTimeSec.toFixed(1)}s`);

    if (sorted.length > 0) {
      console.log("\n  Pending Queue (top 5):");
      for (const req of sorted.slice(0, 5)) {
        const wait = ((Date.now() - req.createdAt) / 1000).toFixed(0);
        console.log(`    [${req.riskLevel.toUpperCase()}] ${req.proposedAction} | $${req.estimatedCost.toFixed(2)} | wait: ${wait}s`);
      }
    }
  }
}

/** Create a sample approval request for demos. */
export function makeSampleRequest(overrides: Partial<ApprovalRequest> = {}): ApprovalRequest {
  return {
    requestId: `req-${Date.now()}`,
    userId: "user-123",
    proposedAction: "update_database",
    params: { table: "users", operation: "delete", filter: "id = 42" },
    riskLevel: "high",
    estimatedCost: 0.0,
    reasoning: "User requested deletion of their account data",
    createdAt: Date.now(),
    ...overrides,
  };
}

// Demo
function main(): void {
  const dashboard = new ApprovalDashboard("demo-reviewer");

  dashboard.enqueue(makeSampleRequest({ riskLevel: "high", proposedAction: "delete_data" }));
  dashboard.enqueue(makeSampleRequest({ riskLevel: "critical", proposedAction: "make_purchase", estimatedCost: 250.0 }));
  dashboard.enqueue(makeSampleRequest({ riskLevel: "medium", proposedAction: "send_email" }));

  dashboard.printDashboard();

  // Simulate decisions
  const queue = dashboard.getSortedQueue();
  if (queue[0]) dashboard.decide(queue[0].requestId, "approved", "Looks good");
  if (queue[1]) dashboard.decide(queue[1].requestId, "rejected", "Too expensive");

  console.log("\nAfter decisions:");
  dashboard.printDashboard();
}

main();
