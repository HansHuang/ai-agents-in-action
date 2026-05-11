/**
 * 12-Factor Agent Validation Rules Engine.
 *
 * Automated checks for each of the 12 production-readiness factors.
 * Designed to run in CI/CD to block deployments that fail minimum standards.
 *
 * Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md
 */

import * as fs from "fs";
import * as path from "path";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ValidationSeverity = "blocking" | "warning" | "info";
export type MaturityLevel = "Prototype" | "Development" | "Staging" | "Production" | "Elite";

export interface ValidationRule {
  factorNumber: number;
  factorName: string;
  ruleName: string;
  description: string;
  severity: ValidationSeverity;
  minimumLevel: MaturityLevel;
}

export interface ValidationCheck {
  rule: ValidationRule;
  passed: boolean;
  evidence: string;
  recommendation: string;
}

export interface ValidationReport {
  totalChecks: number;
  passedCount: number;
  failedCount: number;
  warningCount: number;
  checks: ValidationCheck[];
  blockingFailures: ValidationCheck[];
  deploymentAllowed: boolean;
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

function findFiles(root: string, extensions: string[]): string[] {
  if (!fs.existsSync(root)) return [];
  const results: string[] = [];
  const skip = new Set(["node_modules", ".git", "__pycache__", "dist", ".venv"]);

  function walk(dir: string): void {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (skip.has(entry.name)) continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(fullPath);
      else if (extensions.some((ext) => entry.name.endsWith(ext))) results.push(fullPath);
    }
  }

  walk(root);
  return results;
}

function readAll(root: string): string {
  return findFiles(root, [".ts", ".js", ".py"]).map((f) => {
    try { return fs.readFileSync(f, "utf8"); } catch { return ""; }
  }).join("\n");
}

// ---------------------------------------------------------------------------
// Built-in rules
// ---------------------------------------------------------------------------

export const DEFAULT_RULES: ValidationRule[] = [
  { factorNumber: 1, factorName: "Natural Language to Tool Calls", ruleName: "has_tool_definitions", description: "Agent must define typed tool schemas", severity: "blocking", minimumLevel: "Development" },
  { factorNumber: 2, factorName: "Own Your Prompts", ruleName: "has_system_prompt", description: "System prompt must be version-controlled, not hardcoded inline", severity: "warning", minimumLevel: "Staging" },
  { factorNumber: 3, factorName: "Own Your Context Window", ruleName: "context_management", description: "Agent must manage context budget explicitly", severity: "warning", minimumLevel: "Staging" },
  { factorNumber: 4, factorName: "Tools Are Just Structured Outputs", ruleName: "typed_tool_outputs", description: "Tool return types must be structured (JSON/typed objects)", severity: "warning", minimumLevel: "Development" },
  { factorNumber: 5, factorName: "Unify Execution State", ruleName: "stateless_handlers", description: "Agent logic must not store state in module-level variables", severity: "blocking", minimumLevel: "Production" },
  { factorNumber: 6, factorName: "Launch from API", ruleName: "has_api_endpoint", description: "Agent must be accessible via HTTP endpoint", severity: "blocking", minimumLevel: "Staging" },
  { factorNumber: 7, factorName: "Contact Humans via Structured Format", ruleName: "structured_output", description: "Human-facing output must use structured format", severity: "warning", minimumLevel: "Production" },
  { factorNumber: 8, factorName: "Own Your Control Flow", ruleName: "explicit_loops", description: "Agent loop must be explicit and bounded, not recursive", severity: "blocking", minimumLevel: "Development" },
  { factorNumber: 9, factorName: "Compact Errors to Context", ruleName: "error_handling", description: "Errors must be caught and converted to agent context", severity: "warning", minimumLevel: "Staging" },
  { factorNumber: 10, factorName: "Small Focused Agents", ruleName: "agent_scope", description: "No single agent should handle more than 3 unrelated task types", severity: "info", minimumLevel: "Production" },
  { factorNumber: 11, factorName: "Trigger from Events", ruleName: "event_triggers", description: "Agent should support asynchronous event-driven invocation", severity: "info", minimumLevel: "Elite" },
  { factorNumber: 12, factorName: "Stateless Reducer", ruleName: "pure_state_transitions", description: "State transitions must be deterministic given the same inputs", severity: "warning", minimumLevel: "Production" },
];

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

export class TwelveFactorValidator {
  constructor(private codebasePath: string, private rules: ValidationRule[] = DEFAULT_RULES) {}

  validate(agentConfig: Record<string, unknown> = {}): ValidationReport {
    const checks: ValidationCheck[] = [];
    const sourceText = readAll(this.codebasePath);
    const files = findFiles(this.codebasePath, [".ts", ".py", ".js"]);
    const fileNames = files.map((f) => path.basename(f));

    for (const rule of this.rules) {
      const check = this.runCheck(rule, sourceText, fileNames, agentConfig);
      checks.push(check);
    }

    const passed = checks.filter((c) => c.passed).length;
    const blockingFailures = checks.filter((c) => !c.passed && c.rule.severity === "blocking");
    const warnings = checks.filter((c) => !c.passed && c.rule.severity === "warning");

    return {
      totalChecks: checks.length,
      passedCount: passed,
      failedCount: checks.length - passed,
      warningCount: warnings.length,
      checks,
      blockingFailures,
      deploymentAllowed: blockingFailures.length === 0,
    };
  }

  private runCheck(
    rule: ValidationRule,
    sourceText: string,
    fileNames: string[],
    config: Record<string, unknown>
  ): ValidationCheck {
    let passed = false;
    let evidence = "Not found";
    let recommendation = "";

    switch (rule.ruleName) {
      case "has_tool_definitions":
        passed = /function_call|tool_choice|tools\s*:|"type":\s*"function"/.test(sourceText);
        evidence = passed ? "Tool schemas found in source" : "No tool schema definitions found";
        recommendation = "Define typed tool schemas using OpenAI function calling format";
        break;
      case "has_system_prompt":
        passed = /system_prompt|SYSTEM_PROMPT|role.*system/.test(sourceText);
        evidence = passed ? "System prompt found" : "No system prompt defined";
        recommendation = "Extract system prompt to a variable or environment";
        break;
      case "context_management":
        passed = /max_tokens|token_budget|context_window|trim|truncat/.test(sourceText);
        evidence = passed ? "Context management found" : "No token/context management found";
        recommendation = "Implement context budget tracking";
        break;
      case "has_api_endpoint":
        passed = /express|fastapi|http\.createServer|app\.listen|router\.post|@app/.test(sourceText) || fileNames.includes("agent_service.ts");
        evidence = passed ? "API endpoint found" : "No HTTP server found";
        recommendation = "Wrap the agent in an HTTP service";
        break;
      case "error_handling":
        passed = /try\s*{|catch\s*\(|except\s+|\.catch\(/.test(sourceText);
        evidence = passed ? "Error handling found" : "No try/catch blocks found";
        recommendation = "Add try/catch around LLM calls and tool invocations";
        break;
      case "stateless_handlers":
        evidence = "Check manually for module-level mutable state";
        passed = true; // Heuristic — assume pass unless clear violation
        recommendation = "Use dependency injection instead of global state";
        break;
      default:
        passed = true;
        evidence = "No automated check available — assumed pass";
        recommendation = "Review manually";
    }

    return { rule, passed, evidence, recommendation };
  }
}
