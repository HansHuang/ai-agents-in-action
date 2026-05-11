/**
 * Declarative policy engine for harness behaviour.
 *
 * Policies are code, not configuration. They can be version-controlled,
 * tested, and reviewed through pull requests.
 *
 * Policy lifecycle:
 *   PolicyRule[] → HarnessPolicy.evaluate(context) → PolicyDecision
 *
 * Actions: allow | block | redact | approval_required | log_only
 * See: docs/07-harness-engineering/01-the-harness-mindset.md
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PolicyAction = "allow" | "block" | "redact" | "approval_required" | "log_only";

export interface PolicyRule {
  name: string;
  description: string;
  /** Predicate function over PolicyContext. */
  condition: (ctx: PolicyContext) => boolean;
  action: PolicyAction;
  priority: number;         // Higher → evaluated first
  message: string;          // User-facing explanation
  metadata?: Record<string, unknown>;
}

export interface PolicyContext {
  userId: string;
  userInput: string;
  userRole: "admin" | "user" | "free" | "anonymous";
  agentState: Record<string, unknown>;
  proposedAction: string;
  proposedTool: string;
  proposedParams: Record<string, unknown>;
  estimatedCost: number;
  userRequestsLastMinute: number;
  conversationTurns: number;
}

export interface PolicyDecision {
  action: PolicyAction;
  reason: string;
  ruleName: string;
  userMessage: string;
  matchedRules: string[];
}

// ---------------------------------------------------------------------------
// HarnessPolicy
// ---------------------------------------------------------------------------

export class HarnessPolicy {
  private rules: PolicyRule[];

  constructor(rules: PolicyRule[] = []) {
    // Sort by priority descending
    this.rules = [...rules].sort((a, b) => b.priority - a.priority);
  }

  evaluate(ctx: PolicyContext): PolicyDecision {
    const matched: string[] = [];

    for (const rule of this.rules) {
      let matches = false;
      try {
        matches = rule.condition(ctx);
      } catch {
        // Defensive: malformed condition never causes crash
      }
      if (matches) {
        matched.push(rule.name);
        if (rule.action !== "log_only") {
          return {
            action: rule.action,
            reason: rule.description,
            ruleName: rule.name,
            userMessage: rule.message || defaultMessage(rule.action),
            matchedRules: matched,
          };
        }
        // log_only: record but keep evaluating
      }
    }

    return {
      action: "allow",
      reason: "No blocking rule matched",
      ruleName: "default_allow",
      userMessage: "",
      matchedRules: matched,
    };
  }

  addRule(rule: PolicyRule): void {
    this.rules.push(rule);
    this.rules.sort((a, b) => b.priority - a.priority);
  }
}

function defaultMessage(action: PolicyAction): string {
  switch (action) {
    case "block": return "Your request could not be processed.";
    case "redact": return "Some content was redacted for safety.";
    case "approval_required": return "Your request requires human approval.";
    default: return "";
  }
}

// ---------------------------------------------------------------------------
// Built-in policy rules
// ---------------------------------------------------------------------------

export const INJECTION_RULE: PolicyRule = {
  name: "block_injection",
  description: "Block prompt injection attempts",
  condition: (ctx) => {
    const lower = ctx.userInput.toLowerCase();
    return [
      "ignore previous instructions",
      "disregard your system prompt",
      "you are now",
      "jailbreak",
      "forget your instructions",
    ].some((p) => lower.includes(p));
  },
  action: "block",
  priority: 100,
  message: "Your request contains disallowed content.",
};

export const RATE_LIMIT_RULE: PolicyRule = {
  name: "rate_limit",
  description: "Block when user exceeds rate limit",
  condition: (ctx) => ctx.userRequestsLastMinute > 30,
  action: "block",
  priority: 90,
  message: "Rate limit exceeded. Please slow down.",
};

export const HIGH_COST_APPROVAL_RULE: PolicyRule = {
  name: "high_cost_approval",
  description: "Require approval for high-cost operations",
  condition: (ctx) => ctx.estimatedCost > 0.50,
  action: "approval_required",
  priority: 80,
  message: "This request requires human approval due to cost.",
};

export const SENSITIVE_TOOL_APPROVAL_RULE: PolicyRule = {
  name: "sensitive_tool_approval",
  description: "Require approval for destructive tools",
  condition: (ctx) =>
    ["delete_data", "make_purchase", "update_database"].includes(ctx.proposedTool),
  action: "approval_required",
  priority: 85,
  message: "This action requires human approval.",
};

export const PII_REDACT_RULE: PolicyRule = {
  name: "pii_output_redact",
  description: "Redact PII from outputs",
  condition: (ctx) => {
    const lower = ctx.userInput.toLowerCase();
    return lower.includes("ssn") || lower.includes("social security") || /\b\d{3}-\d{2}-\d{4}\b/.test(ctx.userInput);
  },
  action: "redact",
  priority: 70,
  message: "Sensitive information has been redacted.",
};

/** Create a default production policy with standard rules. */
export function defaultProductionPolicy(): HarnessPolicy {
  return new HarnessPolicy([
    INJECTION_RULE,
    RATE_LIMIT_RULE,
    HIGH_COST_APPROVAL_RULE,
    SENSITIVE_TOOL_APPROVAL_RULE,
    PII_REDACT_RULE,
  ]);
}
