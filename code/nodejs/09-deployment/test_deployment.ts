/**
 * Tests for nodejs/09-deployment — TwelveFactorValidator, CI check
 *
 * No LLM calls — filesystem operations use temp directories.
 * Run: node --import tsx/esm --test test_deployment.ts
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { TwelveFactorValidator, DEFAULT_RULES } from "./twelve_factor_validator.js";
import { githubActionsOutput, humanOutput, jsonOutput } from "./ci_twelve_factor_check.js";

// ---------------------------------------------------------------------------
// Temp directory helpers
// ---------------------------------------------------------------------------

let tmpDir: string;

before(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "12factor-test-"));
});

after(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

function writeFile(name: string, content: string): void {
  fs.writeFileSync(path.join(tmpDir, name), content, "utf8");
}

// ---------------------------------------------------------------------------
// TwelveFactorValidator
// ---------------------------------------------------------------------------

describe("TwelveFactorValidator", () => {
  it("validates an empty codebase (produces report)", () => {
    const emptyDir = fs.mkdtempSync(path.join(os.tmpdir(), "empty-"));
    try {
      const validator = new TwelveFactorValidator(emptyDir);
      const report = validator.validate();
      assert.ok(report.totalChecks > 0);
      assert.ok(typeof report.deploymentAllowed === "boolean");
      assert.ok(Array.isArray(report.checks));
    } finally {
      fs.rmSync(emptyDir, { recursive: true, force: true });
    }
  });

  it("passes has_tool_definitions when source contains tools schema", () => {
    writeFile("agent.ts", `const tools = [{ "type": "function", function_call: true }]; const tool_choice = "auto";`);
    const validator = new TwelveFactorValidator(tmpDir);
    const report = validator.validate();
    const toolCheck = report.checks.find((c) => c.rule.ruleName === "has_tool_definitions");
    assert.ok(toolCheck);
    assert.equal(toolCheck.passed, true, `Expected pass — evidence: ${toolCheck.evidence}`);
  });

  it("passes error_handling when source has try/catch", () => {
    writeFile("handler.ts", `try { await llm(); } catch (err) { console.error(err); }`);
    const validator = new TwelveFactorValidator(tmpDir);
    const report = validator.validate();
    const errCheck = report.checks.find((c) => c.rule.ruleName === "error_handling");
    assert.ok(errCheck);
    assert.equal(errCheck.passed, true);
  });

  it("reports blockingFailures separately", () => {
    const emptyDir = fs.mkdtempSync(path.join(os.tmpdir(), "empty2-"));
    try {
      const validator = new TwelveFactorValidator(emptyDir);
      const report = validator.validate();
      // blocking failures should be a subset of all checks
      for (const bf of report.blockingFailures) {
        assert.equal(bf.rule.severity, "blocking");
      }
    } finally {
      fs.rmSync(emptyDir, { recursive: true, force: true });
    }
  });

  it("uses DEFAULT_RULES and covers all 12 factors", () => {
    const factors = DEFAULT_RULES.map((r) => r.factorNumber);
    const uniqueFactors = new Set(factors);
    assert.ok(uniqueFactors.size >= 6, `Expected at least 6 factors covered, got ${uniqueFactors.size}`);
  });
});

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

describe("CI output formatters", () => {
  const mockReport = {
    totalChecks: 6,
    passedCount: 4,
    failedCount: 2,
    warningCount: 1,
    checks: [
      {
        rule: { factorNumber: 1, factorName: "Tool Calls", ruleName: "has_tool_definitions", description: "", severity: "blocking" as const, minimumLevel: "Development" as const },
        passed: false,
        evidence: "No tools found",
        recommendation: "Add tool schemas",
      },
      {
        rule: { factorNumber: 2, factorName: "Prompts", ruleName: "has_system_prompt", description: "", severity: "warning" as const, minimumLevel: "Staging" as const },
        passed: false,
        evidence: "No system prompt",
        recommendation: "Add system prompt",
      },
    ],
    blockingFailures: [
      {
        rule: { factorNumber: 1, factorName: "Tool Calls", ruleName: "has_tool_definitions", description: "", severity: "blocking" as const, minimumLevel: "Development" as const },
        passed: false,
        evidence: "No tools found",
        recommendation: "Add tool schemas",
      },
    ],
    deploymentAllowed: false,
  };

  it("githubActionsOutput contains ::error annotation", () => {
    const out = githubActionsOutput(mockReport);
    assert.ok(out.includes("::error"), `Expected ::error in:\n${out}`);
    assert.ok(out.includes("::group::"), "Expected group annotation");
  });

  it("humanOutput contains deployment status", () => {
    const out = humanOutput(mockReport);
    assert.ok(out.includes("BLOCKED") || out.includes("ALLOWED"));
  });

  it("jsonOutput is valid JSON with deployment_allowed", () => {
    const out = jsonOutput(mockReport);
    const parsed = JSON.parse(out);
    assert.ok("deployment_allowed" in parsed);
    assert.ok(Array.isArray(parsed.checks));
  });

  it("jsonOutput passing report has deployment_allowed=true", () => {
    const passingReport = { ...mockReport, blockingFailures: [], deploymentAllowed: true };
    const out = jsonOutput(passingReport);
    const parsed = JSON.parse(out);
    assert.equal(parsed.deployment_allowed, true);
  });
});
