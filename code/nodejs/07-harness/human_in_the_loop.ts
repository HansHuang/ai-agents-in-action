/**
 * Human-in-the-Loop Approval System — TypeScript
 * ================================================
 * Port of the Python implementation with the same workflow:
 *   propose → review → decide → execute
 *
 * Key adaptations:
 *   - Promise.race for approval timeouts
 *   - EventEmitter for reviewer response delivery
 *   - zod for runtime type validation
 *   - Strict TypeScript throughout
 *
 * Companion to: docs/07-harness-engineering/06-human-in-the-loop.md
 */

import { EventEmitter } from "events";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Zod schemas for runtime validation
// ---------------------------------------------------------------------------

export const ApprovalRequestSchema = z.object({
  requestId: z.string().uuid(),
  agentId: z.string(),
  sessionId: z.string(),
  proposedAction: z.string(),
  proposedParams: z.record(z.unknown()),
  reasoning: z.string(),
  conversationSummary: z.string(),
  evidence: z.array(z.record(z.unknown())),
  riskLevel: z.enum(["low", "medium", "high", "critical"]),
  estimatedCost: z.number().nonnegative(),
  affectedSystems: z.array(z.string()),
  createdAt: z.number(),
  deadline: z.number().nullable(),
});

export const ApprovalResponseSchema = z.object({
  requestId: z.string(),
  decision: z.enum(["approved", "rejected", "approved_with_edits"]),
  reviewerId: z.string(),
  reviewerNotes: z.string().nullable().optional(),
  editedParams: z.record(z.unknown()).nullable().optional(),
  automated: z.boolean().default(false),
  decidedAt: z.number(),
  reason: z.string().nullable().optional(),
});

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

export type ApprovalRequest = z.infer<typeof ApprovalRequestSchema>;
export type ApprovalResponse = z.infer<typeof ApprovalResponseSchema>;

export interface ExecutionResult {
  success: boolean;
  result?: unknown;
  error?: string;
}

export interface ApprovalDecision {
  requiresApproval: boolean;
  rule: string;
  riskLevel: string;
  timeoutSeconds: number;
}

export interface Reviewer {
  reviewerId: string;
  name: string;
  isSenior: boolean;
  expertise: string[];
  slackId?: string;
  email?: string;
}

export interface ReviewerGroup {
  type: "group";
  reviewers: Reviewer[];
  id: string;
}

export type ReviewerOrGroup = Reviewer | ReviewerGroup;

function isReviewerGroup(r: ReviewerOrGroup): r is ReviewerGroup {
  return (r as ReviewerGroup).type === "group";
}

// ---------------------------------------------------------------------------
// Approval Rule
// ---------------------------------------------------------------------------

export class ApprovalRule {
  readonly name: string;
  readonly description: string;
  readonly priority: number;
  readonly riskLevel: string;
  readonly timeoutSeconds: number;
  readonly actions?: string[];
  readonly minCost?: number;
  readonly affectedSystems?: string[];
  readonly userRoles?: string[];

  constructor(opts: {
    name: string;
    description: string;
    priority: number;
    riskLevel: string;
    timeoutSeconds?: number;
    actions?: string[];
    minCost?: number;
    affectedSystems?: string[];
    userRoles?: string[];
  }) {
    this.name = opts.name;
    this.description = opts.description;
    this.priority = opts.priority;
    this.riskLevel = opts.riskLevel;
    this.timeoutSeconds = opts.timeoutSeconds ?? 300;
    this.actions = opts.actions;
    this.minCost = opts.minCost;
    this.affectedSystems = opts.affectedSystems;
    this.userRoles = opts.userRoles;
  }

  /**
   * Return true if this rule applies to the given action and context.
   */
  matches(
    action: string,
    params: Record<string, unknown>,
    context: Record<string, unknown>
  ): boolean {
    if (this.actions && !this.actions.includes(action)) return false;

    if (this.minCost !== undefined) {
      const cost = this.estimateCost(action, params);
      if (cost < this.minCost) return false;
    }

    if (this.affectedSystems) {
      const actionSystems = this.getAffectedSystems(action);
      const overlap = actionSystems.some((s) => this.affectedSystems!.includes(s));
      if (!overlap) return false;
    }

    if (this.userRoles) {
      const role = context["role"] as string | undefined;
      if (!role || !this.userRoles.includes(role)) return false;
    }

    return true;
  }

  private estimateCost(action: string, params: Record<string, unknown>): number {
    if (action === "issue_refund") return Number(params["amount"] ?? 0);
    if (action === "send_email") return 0;
    if (["update_database", "delete_record"].includes(action)) return 50;
    return 10;
  }

  private getAffectedSystems(action: string): string[] {
    const map: Record<string, string[]> = {
      send_email: ["email_service"],
      issue_refund: ["payment_processor", "order_database"],
      update_database: ["database"],
      delete_record: ["database"],
      cancel_subscription: ["billing_system", "subscription_service"],
      export_user_data: ["data_warehouse", "gdpr_service"],
    };
    return map[action] ?? ["unknown"];
  }
}

// ---------------------------------------------------------------------------
// Approval Policy
// ---------------------------------------------------------------------------

export class ApprovalPolicy {
  private _rules: ApprovalRule[] = [];

  get rules(): readonly ApprovalRule[] {
    return this._rules;
  }

  /** Add a rule and re-sort by descending priority. */
  addRule(rule: ApprovalRule): void {
    this._rules.push(rule);
    this._rules.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Return whether an action requires human approval.
   * First matching rule wins; if none match, auto-approved.
   */
  requiresApproval(
    action: string,
    params: Record<string, unknown>,
    context: Record<string, unknown>
  ): ApprovalDecision {
    for (const rule of this._rules) {
      if (rule.matches(action, params, context)) {
        return {
          requiresApproval: true,
          rule: rule.name,
          riskLevel: rule.riskLevel,
          timeoutSeconds: rule.timeoutSeconds,
        };
      }
    }
    return {
      requiresApproval: false,
      rule: "default_allow",
      riskLevel: "none",
      timeoutSeconds: 0,
    };
  }

  /** Pre-configured production defaults. */
  static withDefaults(): ApprovalPolicy {
    const policy = new ApprovalPolicy();
    policy.addRule(new ApprovalRule({
      name: "critical_data_export",
      description: "Exporting user data requires approval",
      priority: 95,
      riskLevel: "high",
      actions: ["export_user_data"],
      timeoutSeconds: 600,
    }));
    policy.addRule(new ApprovalRule({
      name: "high_value_refund",
      description: "Refunds over $500 require human approval",
      priority: 100,
      riskLevel: "high",
      actions: ["issue_refund"],
      minCost: 500,
      timeoutSeconds: 600,
    }));
    policy.addRule(new ApprovalRule({
      name: "external_communication",
      description: "Sending email to customers requires approval",
      priority: 90,
      riskLevel: "medium",
      actions: ["send_email"],
      timeoutSeconds: 300,
    }));
    policy.addRule(new ApprovalRule({
      name: "subscription_cancellation",
      description: "Cancelling subscriptions requires approval",
      priority: 85,
      riskLevel: "high",
      actions: ["cancel_subscription"],
      timeoutSeconds: 600,
    }));
    policy.addRule(new ApprovalRule({
      name: "database_modification",
      description: "Any database write requires approval",
      priority: 80,
      riskLevel: "medium",
      actions: ["update_database", "delete_record"],
      timeoutSeconds: 300,
    }));
    return policy;
  }
}

// ---------------------------------------------------------------------------
// Human Reviewer Interface
// ---------------------------------------------------------------------------

export class HumanReviewerInterface {
  constructor(public readonly reviewerId: string) {}

  /** Approve the request as-is. */
  approve(requestId: string, notes?: string): ApprovalResponse {
    return ApprovalResponseSchema.parse({
      requestId,
      decision: "approved",
      reviewerId: this.reviewerId,
      reviewerNotes: notes ?? null,
      automated: false,
      decidedAt: Date.now() / 1000,
    });
  }

  /** Reject with a mandatory reason. */
  reject(requestId: string, reason: string): ApprovalResponse {
    return ApprovalResponseSchema.parse({
      requestId,
      decision: "rejected",
      reviewerId: this.reviewerId,
      reason,
      automated: false,
      decidedAt: Date.now() / 1000,
    });
  }

  /** Approve with modified parameters. */
  approveWithEdits(
    requestId: string,
    editedParams: Record<string, unknown>,
    notes?: string
  ): ApprovalResponse {
    return ApprovalResponseSchema.parse({
      requestId,
      decision: "approved_with_edits",
      reviewerId: this.reviewerId,
      editedParams,
      reviewerNotes: notes ?? null,
      automated: false,
      decidedAt: Date.now() / 1000,
    });
  }
}

// ---------------------------------------------------------------------------
// Approval Interface
// ---------------------------------------------------------------------------

export class ApprovalInterface extends EventEmitter {
  readonly pendingRequests = new Map<string, ApprovalRequest>();
  readonly reviewers = new Map<string, Reviewer>();
  private readonly channels: string[];

  constructor(channels: string[] = ["dashboard"]) {
    super();
    this.channels = channels;
  }

  /** Register a human reviewer. */
  registerReviewer(reviewer: Reviewer): void {
    this.reviewers.set(reviewer.reviewerId, reviewer);
  }

  /**
   * Send an approval request and wait up to `timeoutSeconds` for a response.
   * Auto-rejects on timeout (safe default).
   */
  async requestApproval(
    request: ApprovalRequest,
    timeoutSeconds = 300
  ): Promise<ApprovalResponse> {
    // Validate input
    ApprovalRequestSchema.parse(request);

    const reviewer = this.assignReviewer(request);
    await this.sendToReviewer(reviewer, request);
    this.pendingRequests.set(request.requestId, request);

    const waitPromise = this.waitForResponse(reviewer, request.requestId);
    const timeoutPromise = new Promise<ApprovalResponse>((resolve) =>
      setTimeout(() => {
        resolve({
          requestId: request.requestId,
          decision: "rejected",
          reviewerId: isReviewerGroup(reviewer) ? reviewer.id : reviewer.reviewerId,
          reason: "Approval timeout — automatically rejected for safety.",
          automated: true,
          decidedAt: Date.now() / 1000,
        });
      }, timeoutSeconds * 1000)
    );

    const response = await Promise.race([waitPromise, timeoutPromise]);
    this.pendingRequests.delete(request.requestId);
    return response;
  }

  /**
   * Submit a reviewer's decision.  Triggers the pending promise for that request.
   */
  submitResponse(response: ApprovalResponse): void {
    this.emit(`response:${response.requestId}`, response);
  }

  // -- private --------------------------------------------------------------

  private assignReviewer(request: ApprovalRequest): ReviewerOrGroup {
    const all = Array.from(this.reviewers.values());
    const seniors = all.filter((r) => r.isSenior);

    if (request.riskLevel === "critical") {
      const group = seniors.slice(0, 2);
      return group.length >= 2
        ? { type: "group", reviewers: group, id: group.map((r) => r.reviewerId).join(",") }
        : { type: "group", reviewers: all.slice(0, 2), id: all.slice(0, 2).map((r) => r.reviewerId).join(",") };
    }
    if (request.riskLevel === "high") return seniors[0] ?? all[0] ?? this.fallbackReviewer();
    if (request.riskLevel === "medium") {
      const expert = all.find((r) => r.expertise.includes(request.proposedAction));
      return expert ?? all[0] ?? this.fallbackReviewer();
    }
    return all[0] ?? this.fallbackReviewer();
  }

  private fallbackReviewer(): Reviewer {
    return { reviewerId: "unassigned", name: "Unassigned", isSenior: false, expertise: [] };
  }

  private async sendToReviewer(reviewer: ReviewerOrGroup, request: ApprovalRequest): Promise<void> {
    const revs = isReviewerGroup(reviewer) ? reviewer.reviewers : [reviewer];
    const message = this.formatRequest(request);

    for (const rev of revs) {
      if (this.channels.includes("slack") && rev.slackId) {
        console.log(`[SLACK → ${rev.slackId}] ${message.slice(0, 80)}…`);
      }
      if (this.channels.includes("email") && rev.email &&
          ["high", "critical"].includes(request.riskLevel)) {
        console.log(`[EMAIL → ${rev.email}] URGENT: ${request.proposedAction}`);
      }
    }
    if (this.channels.includes("dashboard")) {
      console.log(`[DASHBOARD] Pending: ${request.requestId}`);
    }
  }

  private waitForResponse(reviewer: ReviewerOrGroup, requestId: string): Promise<ApprovalResponse> {
    if (isReviewerGroup(reviewer)) {
      return this.waitForConsensus(reviewer, requestId);
    }
    return new Promise((resolve) => {
      this.once(`response:${requestId}`, resolve);
    });
  }

  private waitForConsensus(group: ReviewerGroup, requestId: string): Promise<ApprovalResponse> {
    return new Promise((resolve) => {
      const approvals: ApprovalResponse[] = [];
      const handler = (response: ApprovalResponse): void => {
        if (response.decision === "rejected") {
          this.removeAllListeners(`response:${requestId}`);
          resolve(response);
          return;
        }
        approvals.push(response);
        if (approvals.length >= group.reviewers.length) {
          this.removeAllListeners(`response:${requestId}`);
          resolve(approvals[approvals.length - 1]);
        }
      };
      this.on(`response:${requestId}`, handler);
    });
  }

  private formatRequest(request: ApprovalRequest): string {
    return (
      `\n${"═".repeat(45)}\n` +
      `APPROVAL REQUIRED\n` +
      `${"═".repeat(45)}\n` +
      `Action: ${request.proposedAction}\n` +
      `Risk:   ${request.riskLevel.toUpperCase()}\n` +
      `Cost:   $${request.estimatedCost.toFixed(2)}\n` +
      `Params: ${JSON.stringify(request.proposedParams)}\n` +
      `Why:    ${request.reasoning}\n` +
      `${"═".repeat(45)}\n`
    );
  }
}

// ---------------------------------------------------------------------------
// Approval Executor
// ---------------------------------------------------------------------------

export interface Agent {
  executeTool(toolName: string, params: Record<string, unknown>): Promise<unknown>;
  sendMessage(message: string): Promise<void>;
  sendMessageSync(message: string): void;
}

export class ApprovalExecutor {
  readonly approvedActions: Array<{ request: ApprovalRequest; result: unknown; timestamp: number }> = [];
  readonly rejectedActions: Array<{ request: ApprovalRequest; response: ApprovalResponse; timestamp: number }> = [];

  constructor(private readonly agent: Agent) {}

  /**
   * Execute (or record the rejection of) an approved action.
   */
  async execute(request: ApprovalRequest, response: ApprovalResponse): Promise<ExecutionResult> {
    switch (response.decision) {
      case "approved":
        return this.executeApproved(request);
      case "approved_with_edits":
        return this.executeEdited(request, response.editedParams ?? {});
      case "rejected":
        return this.handleRejection(request, response);
      default:
        throw new Error(`Unknown decision: ${response.decision}`);
    }
  }

  private async executeApproved(request: ApprovalRequest): Promise<ExecutionResult> {
    try {
      const result = await this.agent.executeTool(
        request.proposedAction,
        request.proposedParams as Record<string, unknown>
      );
      await this.agent.sendMessage(`Completed: ${request.proposedAction}`);
      this.approvedActions.push({ request, result, timestamp: Date.now() / 1000 });
      return { success: true, result };
    } catch (err) {
      return { success: false, error: String(err) };
    }
  }

  private async executeEdited(
    request: ApprovalRequest,
    editedParams: Record<string, unknown>
  ): Promise<ExecutionResult> {
    try {
      const result = await this.agent.executeTool(request.proposedAction, editedParams);
      await this.agent.sendMessage("Completed with the adjustments you specified.");
      this.approvedActions.push({ request, result, timestamp: Date.now() / 1000 });
      return { success: true, result };
    } catch (err) {
      return { success: false, error: String(err) };
    }
  }

  private handleRejection(request: ApprovalRequest, response: ApprovalResponse): ExecutionResult {
    const reason = response.reason ?? "This action requires additional review.";
    this.agent.sendMessageSync(
      `I wasn't able to complete '${request.proposedAction}'. ${reason}`
    );
    this.rejectedActions.push({ request, response, timestamp: Date.now() / 1000 });
    return { success: false, error: `Rejected by reviewer: ${reason}` };
  }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

interface RiskBucket {
  approved: number;
  rejected: number;
  approved_with_edits: number;
}

export class ApprovalMetrics {
  totalRequests = 0;
  approved = 0;
  rejected = 0;
  approvedWithEdits = 0;
  timedOut = 0;
  private totalResponseTime = 0;
  byRiskLevel: Record<string, RiskBucket> = {};

  /**
   * Record a single approval decision.
   */
  record(request: ApprovalRequest, response: ApprovalResponse, responseTime: number): void {
    this.totalRequests++;
    this.totalResponseTime += responseTime;

    if (response.automated) this.timedOut++;

    if (!this.byRiskLevel[request.riskLevel]) {
      this.byRiskLevel[request.riskLevel] = { approved: 0, rejected: 0, approved_with_edits: 0 };
    }
    const bucket = this.byRiskLevel[request.riskLevel];

    if (response.decision === "approved") {
      this.approved++;
      bucket.approved++;
    } else if (response.decision === "rejected") {
      this.rejected++;
      bucket.rejected++;
    } else if (response.decision === "approved_with_edits") {
      this.approvedWithEdits++;
      bucket.approved_with_edits++;
    }
  }

  /** Return a summary object with rates and breakdowns. */
  summary(): Record<string, unknown> {
    const total = Math.max(this.totalRequests, 1);
    const n = this.approved + this.rejected + this.approvedWithEdits;
    return {
      totalApprovalRequests: this.totalRequests,
      approved: this.approved,
      rejected: this.rejected,
      approvedWithEdits: this.approvedWithEdits,
      timedOut: this.timedOut,
      approvalRate: this.approved / total,
      rejectionRate: this.rejected / total,
      editRate: this.approvedWithEdits / total,
      timeoutRate: this.timedOut / total,
      avgResponseTimeSeconds: n > 0 ? this.totalResponseTime / n : 0,
      byRiskLevel: this.byRiskLevel,
    };
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function runDemo(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("  HUMAN-IN-THE-LOOP DEMO  (TypeScript)");
  console.log("=".repeat(60) + "\n");

  const { randomUUID } = await import("crypto");

  const policy = ApprovalPolicy.withDefaults();
  const iface = new ApprovalInterface(["dashboard"]);
  const metrics = new ApprovalMetrics();

  const alice: Reviewer = { reviewerId: "r-alice", name: "Alice", isSenior: true, expertise: ["send_email"] };
  const bob: Reviewer   = { reviewerId: "r-bob",   name: "Bob",   isSenior: true, expertise: ["issue_refund"] };
  iface.registerReviewer(alice);
  iface.registerReviewer(bob);

  const aliceReviewer = new HumanReviewerInterface("r-alice");
  const bobReviewer   = new HumanReviewerInterface("r-bob");

  const mockAgent: Agent = {
    async executeTool(toolName, params) {
      console.log(`  [Tool] ${toolName}(${JSON.stringify(params)})`);
      return { status: "ok", toolName, params };
    },
    async sendMessage(msg) { console.log(`  [Agent → User] ${msg}`); },
    sendMessageSync(msg)   { console.log(`  [Agent → User] ${msg}`); },
  };

  const executor = new ApprovalExecutor(mockAgent);
  const auditTrail: Array<Record<string, unknown>> = [];

  function makeRequest(
    action: string,
    params: Record<string, unknown>,
    riskLevel: "low" | "medium" | "high" | "critical",
    cost: number,
    systems: string[]
  ): ApprovalRequest {
    return ApprovalRequestSchema.parse({
      requestId: randomUUID(),
      agentId: "demo-agent",
      sessionId: "session-ts-01",
      proposedAction: action,
      proposedParams: params,
      reasoning: `Agent wants to perform ${action}.`,
      conversationSummary: "Customer: Please process my request.",
      evidence: [{ source: "order_system", data: params }],
      riskLevel,
      estimatedCost: cost,
      affectedSystems: systems,
      createdAt: Date.now() / 1000,
      deadline: null,
    });
  }

  async function scenario(
    label: string,
    request: ApprovalRequest,
    scheduleResponse?: (requestId: string) => void,
    timeout = 5
  ): Promise<void> {
    console.log(`\n${"─".repeat(60)}`);
    console.log(`Scenario: ${label}`);

    const decision = policy.requiresApproval(
      request.proposedAction,
      request.proposedParams as Record<string, unknown>,
      {}
    );

    if (!decision.requiresApproval) {
      console.log("  → Policy: AUTO-APPROVED");
      await mockAgent.executeTool(request.proposedAction, request.proposedParams as Record<string, unknown>);
      auditTrail.push({ label, decision: "auto_approved" });
      return;
    }

    if (scheduleResponse) {
      setTimeout(() => scheduleResponse(request.requestId), 50);
    }

    const start = Date.now();
    let response: ApprovalResponse;
    try {
      response = await iface.requestApproval(request, timeout);
    } catch (err) {
      console.log(`  → ERROR: ${err}`);
      auditTrail.push({ label, decision: "system_error" });
      return;
    }

    const elapsed = (Date.now() - start) / 1000;
    metrics.record(request, response, elapsed);
    console.log(
      `  → Decision: ${response.decision.toUpperCase()}` +
      (response.automated ? "  (automated)" : "")
    );
    if (response.reason) console.log(`  → Reason: ${response.reason}`);
    if (response.editedParams) console.log(`  → Edits: ${JSON.stringify(response.editedParams)}`);

    const result = await executor.execute(request, response);
    console.log(`  → Execution: ${result.success ? "OK" : "FAILED — " + result.error}`);
    auditTrail.push({ label, decision: response.decision, success: result.success });
  }

  // 1. Auto-approved
  await scenario("1. Low-risk (weather lookup) → auto-approved",
    makeRequest("get_weather", { city: "Tokyo" }, "low", 0, ["weather_api"])
  );

  // 2. Email approved
  await scenario("2. Medium-risk (send email) → approved",
    makeRequest("send_email", { to: "user@example.com" }, "medium", 0, ["email_service"]),
    (id) => iface.submitResponse(aliceReviewer.approve(id, "Looks good."))
  );

  // 3. Refund with edits
  await scenario("3. High-risk ($750 refund) → approved with edits ($500)",
    makeRequest("issue_refund", { orderId: "ORD-99", amount: 750 }, "high", 750, ["payment_processor"]),
    (id) => iface.submitResponse(bobReviewer.approveWithEdits(
      id, { orderId: "ORD-99", amount: 500 }, "Partial refund per policy."
    ))
  );

  // 4. Timeout
  await scenario("4. Approval timeout → auto-rejected",
    makeRequest("cancel_subscription", { subscriptionId: "SUB-7" }, "high", 99, ["billing_system"]),
    undefined, 0.1
  );

  // 5. Rejection
  await scenario("5. Human rejection",
    makeRequest("issue_refund", { orderId: "ORD-33", amount: 1200 }, "high", 1200, ["payment_processor"]),
    (id) => iface.submitResponse(bobReviewer.reject(id, "No evidence provided."))
  );

  // Audit trail
  console.log(`\n${"=".repeat(60)}\nAUDIT TRAIL\n${"=".repeat(60)}`);
  auditTrail.forEach((entry, i) =>
    console.log(`  ${(i + 1).toString().padStart(2)}. ${String(entry["label"]).padEnd(50)} decision=${entry["decision"]}`)
  );

  // Metrics
  console.log(`\n${"=".repeat(60)}\nMETRICS SUMMARY\n${"=".repeat(60)}`);
  const summary = metrics.summary();
  for (const [key, val] of Object.entries(summary)) {
    if (key !== "byRiskLevel") {
      const v = typeof val === "number" ? val.toFixed(3) : String(val);
      console.log(`  ${key.padEnd(35)}: ${v}`);
    }
  }
  console.log();
}

runDemo().catch(console.error);
