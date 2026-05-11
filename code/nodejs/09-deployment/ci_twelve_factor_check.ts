/**
 * CI/CD Integration Script for 12-Factor Agent Validation.
 *
 * Runs the 12-factor validator and gates deployments based on maturity level.
 * Exit codes:
 *   0 — All checks pass
 *   1 — Blocking failures detected
 *   2 — Warnings only, no blocking failures
 *
 * Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md
 */

import * as fs from "fs";
import * as path from "path";
import { TwelveFactorValidator, ValidationReport } from "./twelve_factor_validator.js";

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

function githubActionsOutput(report: ValidationReport): string {
  const lines: string[] = ["::group::12-Factor Agent Validation"];

  for (const check of report.checks.filter((c) => !c.passed && c.rule.severity === "warning")) {
    lines.push(`::warning title=Factor ${check.rule.factorNumber} - ${check.rule.factorName}::${check.evidence}. Fix: ${check.recommendation}`);
  }

  for (const f of report.blockingFailures) {
    lines.push(`::error title=Factor ${f.rule.factorNumber} - ${f.rule.factorName}::${f.evidence}. Fix: ${f.recommendation}`);
  }

  lines.push("::endgroup::");
  lines.push(`BLOCKING FAILURES: ${report.blockingFailures.length}`);
  lines.push(`WARNINGS: ${report.warningCount}`);
  lines.push(`DEPLOYMENT: ${report.deploymentAllowed ? "ALLOWED" : "BLOCKED"}`);

  return lines.join("\n");
}

function humanOutput(report: ValidationReport): string {
  const lines: string[] = ["\n=== 12-Factor Agent Validation ==="];
  lines.push(`  Checks: ${report.passedCount}/${report.totalChecks} passed`);
  lines.push(`  Blocking failures: ${report.blockingFailures.length}`);
  lines.push(`  Warnings: ${report.warningCount}`);
  lines.push(`  Deployment: ${report.deploymentAllowed ? "ALLOWED" : "BLOCKED"}`);

  if (report.blockingFailures.length) {
    lines.push("\n  Blocking failures:");
    for (const f of report.blockingFailures) {
      lines.push(`    [${f.rule.factorNumber}] ${f.rule.factorName}: ${f.evidence}`);
      lines.push(`        Fix: ${f.recommendation}`);
    }
  }

  const warnings = report.checks.filter((c) => !c.passed && c.rule.severity === "warning");
  if (warnings.length) {
    lines.push("\n  Warnings:");
    for (const w of warnings) {
      lines.push(`    [${w.rule.factorNumber}] ${w.rule.factorName}: ${w.evidence}`);
    }
  }

  return lines.join("\n");
}

function jsonOutput(report: ValidationReport): string {
  return JSON.stringify({
    deployment_allowed: report.deploymentAllowed,
    total_checks: report.totalChecks,
    passed: report.passedCount,
    blocking_failures: report.blockingFailures.length,
    warnings: report.warningCount,
    checks: report.checks.map((c) => ({
      factor: c.rule.factorNumber,
      name: c.rule.factorName,
      passed: c.passed,
      severity: c.rule.severity,
      evidence: c.evidence,
      recommendation: c.recommendation,
    })),
  }, null, 2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main(): void {
  const args = process.argv.slice(2);
  const outputFormat = args.includes("--json") ? "json" : args.includes("--github") ? "github" : "human";
  const codebasePath = args.find((a) => !a.startsWith("--")) ?? process.cwd();

  const validator = new TwelveFactorValidator(codebasePath);
  const report = validator.validate();

  switch (outputFormat) {
    case "github":
      console.log(githubActionsOutput(report));
      break;
    case "json":
      console.log(jsonOutput(report));
      break;
    default:
      console.log(humanOutput(report));
  }

  // Exit with appropriate code
  if (report.blockingFailures.length > 0) {
    process.exit(1);
  } else if (report.warningCount > 0) {
    process.exit(2);
  } else {
    process.exit(0);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { githubActionsOutput, humanOutput, jsonOutput };
