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
    if (report.end_to_end) this.printEndToEndSection(report.end_to_end);

    this.printOverallHealth(report);
  }

  printRetrievalSection(report: RetrievalReport): void {
    console.log("\n── Retrieval Metrics ──");
    const rows = [
      ["Hit Rate",     report.hit_rate,       THRESHOLDS.RETRIEVAL_HIT_RATE_MIN],
      ["Precision@K",  report.precision_at_k, THRESHOLDS.RETRIEVAL_PRECISION_MIN],
      ["Recall@K",     report.recall_at_k,    THRESHOLDS.RETRIEVAL_PRECISION_MIN],
      ["MRR",          report.mrr,            THRESHOLDS.RETRIEVAL_PRECISION_MIN],
    ] as Array<[string, number, number]>;

    for (const [name, value, threshold] of rows) {
      const st = status(value, threshold);
      console.log(`  ${name.padEnd(14)} ${bar(value)} ${pct(value).padStart(7)}  [${st}]`);
    }
  }

  printGenerationSection(report: GenerationReport): void {
    console.log("\n── Generation Metrics ──");
    const rows = [
      ["Faithfulness",  report.faithfulness_pass_rate ?? 0,  THRESHOLDS.GENERATION_FAITHFULNESS_MIN],
      ["Relevance",     report.relevance_pass_rate ?? 0,     THRESHOLDS.GENERATION_RELEVANCE_MIN],
      ["Completeness",  report.completeness_pass_rate ?? 0,  THRESHOLDS.GENERATION_RELEVANCE_MIN],
      ["Rule Pass Rate",report.overall_pass_rate,            THRESHOLDS.GENERATION_FAITHFULNESS_MIN],
    ] as Array<[string, number, number]>;

    for (const [name, value, threshold] of rows) {
      const st = status(value, threshold);
      console.log(`  ${name.padEnd(14)} ${bar(value)} ${pct(value).padStart(7)}  [${st}]`);
    }
  }

  printEndToEndSection(report: EndToEndReport): void {
    console.log("\n── End-to-End Metrics ──");
    const st = status(report.task_success_rate, THRESHOLDS.E2E_SUCCESS_MIN);
    console.log(`  ${"Success Rate".padEnd(14)} ${bar(report.task_success_rate)} ${pct(report.task_success_rate).padStart(7)}  [${st}]`);
    console.log(`  ${"Avg Turns".padEnd(14)} ${report.avg_turns_to_resolution.toFixed(1)}`); 
  }

  printOverallHealth(report: FullEvaluationReport): void {
    console.log("\n── Overall Health ──");

    const checks: boolean[] = [];
    if (report.retrieval) {
      checks.push(report.retrieval.hit_rate >= THRESHOLDS.RETRIEVAL_HIT_RATE_MIN);
    }
    if (report.generation) {
      checks.push((report.generation.faithfulness_pass_rate ?? 0) >= THRESHOLDS.GENERATION_FAITHFULNESS_MIN);
    }
    if (report.end_to_end) {
      checks.push(report.end_to_end.task_success_rate >= THRESHOLDS.E2E_SUCCESS_MIN);
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
      check("Hit Rate", current.retrieval.hit_rate, baseline.retrieval.hit_rate);
    }
    if (current.generation && baseline.generation) {
      check("Faithfulness", current.generation.faithfulness_pass_rate ?? 0, baseline.generation.faithfulness_pass_rate ?? 0);
    }
    if (current.end_to_end && baseline.end_to_end) {
      check("Success Rate", current.end_to_end.task_success_rate, baseline.end_to_end.task_success_rate);
    }

    if (regressions.length) {
      console.log("\n[REGRESSION ALERT]");
      regressions.forEach((r) => console.log(`  - ${r}`));
    }

    return regressions;
  }
}
