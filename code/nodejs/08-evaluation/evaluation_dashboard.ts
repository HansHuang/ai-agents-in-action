/**
 * Evaluation dashboard for agent evaluation reports.
 *
 * Renders color-coded summaries, per-query breakdowns, regression alerts,
 * and trend information from evaluation reports.
 * See: docs/08-evaluation-and-guardrails/01-evaluating-agents.md
 */

import type {
  RetrievalReport,
  GenerationReport,
  EndToEndReport,
  FullEvaluationReport,
} from "./agent_evaluator.js";

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

const THRESHOLDS = {
  RETRIEVAL_HIT_RATE_MIN: 0.70,
  RETRIEVAL_PRECISION_MIN: 0.60,
  GENERATION_FAITHFULNESS_MIN: 0.70,
  GENERATION_RELEVANCE_MIN: 0.70,
  E2E_SUCCESS_MIN: 0.80,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function bar(v: number, width = 20): string {
  const filled = Math.round(v * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

function status(v: number, threshold: number): string {
  return v >= threshold ? "PASS" : "FAIL";
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export class EvaluationDashboard {
  /** Print a full evaluation report summary. */
  printFullReport(report: FullEvaluationReport): void {
    console.log("\n╔══════════════════════════════════════════════════════╗");
    console.log("║            AGENT EVALUATION DASHBOARD                ║");
    console.log("╚══════════════════════════════════════════════════════╝");

    if (report.retrieval) this.printRetrievalSection(report.retrieval);
    if (report.generation) this.printGenerationSection(report.generation);
    if (report.endToEnd) this.printEndToEndSection(report.endToEnd);

    this.printOverallHealth(report);
  }

  printRetrievalSection(report: RetrievalReport): void {
    console.log("\n── Retrieval Metrics ──");
    const m = report.aggregate;
    const rows = [
      ["Hit Rate",     m.hitRate,     THRESHOLDS.RETRIEVAL_HIT_RATE_MIN],
      ["Precision@K",  m.precisionAtK, THRESHOLDS.RETRIEVAL_PRECISION_MIN],
      ["Recall@K",     m.recallAtK,   THRESHOLDS.RETRIEVAL_PRECISION_MIN],
      ["MRR",          m.mrr,         THRESHOLDS.RETRIEVAL_PRECISION_MIN],
    ] as Array<[string, number, number]>;

    for (const [name, value, threshold] of rows) {
      const st = status(value, threshold);
      console.log(`  ${name.padEnd(14)} ${bar(value)} ${pct(value).padStart(7)}  [${st}]`);
    }
  }

  printGenerationSection(report: GenerationReport): void {
    console.log("\n── Generation Metrics ──");
    const m = report.aggregate;
    const rows = [
      ["Faithfulness",  m.avgFaithfulness,  THRESHOLDS.GENERATION_FAITHFULNESS_MIN],
      ["Relevance",     m.avgRelevance,     THRESHOLDS.GENERATION_RELEVANCE_MIN],
      ["Completeness",  m.avgCompleteness,  THRESHOLDS.GENERATION_RELEVANCE_MIN],
      ["Rule Pass Rate",m.rulePassRate,     THRESHOLDS.GENERATION_FAITHFULNESS_MIN],
    ] as Array<[string, number, number]>;

    for (const [name, value, threshold] of rows) {
      const st = status(value, threshold);
      console.log(`  ${name.padEnd(14)} ${bar(value)} ${pct(value).padStart(7)}  [${st}]`);
    }
  }

  printEndToEndSection(report: EndToEndReport): void {
    console.log("\n── End-to-End Metrics ──");
    const m = report.aggregate;
    const st = status(m.successRate, THRESHOLDS.E2E_SUCCESS_MIN);
    console.log(`  ${"Success Rate".padEnd(14)} ${bar(m.successRate)} ${pct(m.successRate).padStart(7)}  [${st}]`);
    console.log(`  ${"Avg Turns".padEnd(14)} ${m.avgTurns.toFixed(1)}`);
  }

  printOverallHealth(report: FullEvaluationReport): void {
    console.log("\n── Overall Health ──");

    const checks: boolean[] = [];
    if (report.retrieval) {
      checks.push(report.retrieval.aggregate.hitRate >= THRESHOLDS.RETRIEVAL_HIT_RATE_MIN);
    }
    if (report.generation) {
      checks.push(report.generation.aggregate.avgFaithfulness >= THRESHOLDS.GENERATION_FAITHFULNESS_MIN);
    }
    if (report.endToEnd) {
      checks.push(report.endToEnd.aggregate.successRate >= THRESHOLDS.E2E_SUCCESS_MIN);
    }

    const passed = checks.filter(Boolean).length;
    const healthPct = checks.length ? passed / checks.length : 1;
    const verdict = healthPct === 1 ? "HEALTHY" : healthPct >= 0.6 ? "DEGRADED" : "UNHEALTHY";

    console.log(`  ${bar(healthPct)} ${pct(healthPct).padStart(7)}  [${verdict}]`);
    console.log("");
  }

  /** Check for regressions against a baseline. */
  checkRegressions(
    current: FullEvaluationReport,
    baseline: FullEvaluationReport,
    tolerancePct = 5
  ): string[] {
    const regressions: string[] = [];
    const tol = tolerancePct / 100;

    function check(name: string, cur: number, base: number): void {
      if (cur < base - tol) {
        regressions.push(`${name}: ${pct(cur)} vs baseline ${pct(base)} (delta ${pct(cur - base)})`);
      }
    }

    if (current.retrieval && baseline.retrieval) {
      check("Hit Rate", current.retrieval.aggregate.hitRate, baseline.retrieval.aggregate.hitRate);
    }
    if (current.generation && baseline.generation) {
      check("Faithfulness", current.generation.aggregate.avgFaithfulness, baseline.generation.aggregate.avgFaithfulness);
    }
    if (current.endToEnd && baseline.endToEnd) {
      check("Success Rate", current.endToEnd.aggregate.successRate, baseline.endToEnd.aggregate.successRate);
    }

    if (regressions.length) {
      console.log("\n[REGRESSION ALERT]");
      regressions.forEach((r) => console.log(`  - ${r}`));
    }

    return regressions;
  }
}
