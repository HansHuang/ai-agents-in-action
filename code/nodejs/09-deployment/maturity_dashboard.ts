/**
 * Agent Maturity Dashboard — visual terminal report for 12-Factor assessment.
 *
 * Renders color-coded factor scores, progress bars, and a summary report.
 * Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md
 */

import type { TwelveFactorReport, FactorAssessment } from "./twelve_factor_assessor.js";

// ---------------------------------------------------------------------------
// Visual helpers
// ---------------------------------------------------------------------------

function bar(score: number, maxScore = 5, width = 20): string {
  const filled = Math.round((score / maxScore) * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

function scoreLabel(score: number): string {
  if (score >= 5) return "EXCELLENT";
  if (score >= 4) return "GOOD     ";
  if (score >= 3) return "FAIR     ";
  return "POOR     ";
}

function maturityBar(overallScore: number, maxScore = 60, width = 40): string {
  const filled = Math.round((overallScore / maxScore) * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

// Level thresholds (score out of 60)
const LEVEL_THRESHOLDS: Array<{ min: number; level: string }> = [
  { min: 55, level: "Elite" },
  { min: 45, level: "Production" },
  { min: 35, level: "Staging" },
  { min: 25, level: "Development" },
  { min: 0, level: "Prototype" },
];

function getLevel(score: number): string {
  return LEVEL_THRESHOLDS.find((t) => score >= t.min)?.level ?? "Prototype";
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export class MaturityDashboard {
  /** Print a full 12-factor report. */
  printReport(report: TwelveFactorReport): void {
    console.log("\n╔══════════════════════════════════════════════════════════════╗");
    console.log("║           12-FACTOR AGENT MATURITY DASHBOARD                 ║");
    console.log("╚══════════════════════════════════════════════════════════════╝");

    const level = getLevel(report.overallScore);
    console.log(`\n  Maturity Level: ${level}`);
    console.log(`  Overall Score: ${report.overallScore}/60`);
    console.log(`  ${maturityBar(report.overallScore)}`);

    console.log("\n  Factor Breakdown:");
    console.log("  " + "─".repeat(62));
    console.log(`  ${"#".padEnd(3)} ${"Factor".padEnd(35)} ${"Score".padEnd(8)} Bar`);
    console.log("  " + "─".repeat(62));

    for (const factor of report.factors) {
      const label = scoreLabel(factor.score);
      console.log(
        `  ${factor.factorNumber.toString().padEnd(3)} ` +
        `${factor.factorName.slice(0, 33).padEnd(35)} ` +
        `${factor.score}/5  ${label}  ${bar(factor.score)}`
      );
    }

    if (report.criticalGaps.length) {
      console.log("\n  Critical Gaps:");
      report.criticalGaps.forEach((g) => console.log(`    ✗ ${g}`));
    }

    if (report.improvementPriorities.length) {
      console.log("\n  Improvement Priorities:");
      report.improvementPriorities.slice(0, 5).forEach((p, i) =>
        console.log(`    ${i + 1}. ${p}`)
      );
    }

    console.log(`\n  Assessed: ${report.assessedAt}`);
    console.log("");
  }

  /** Print a compact summary line. */
  printSummary(report: TwelveFactorReport): void {
    const level = getLevel(report.overallScore);
    console.log(`[12-Factor] ${level} | Score: ${report.overallScore}/60 | Gaps: ${report.criticalGaps.length}`);
  }

  /** Print improvement recommendations. */
  printRoadmap(report: TwelveFactorReport): void {
    console.log("\n  Improvement Roadmap:");
    const weak = report.factors
      .filter((f) => f.score <= 3)
      .sort((a, b) => a.score - b.score)
      .slice(0, 5);

    for (const f of weak) {
      console.log(`\n  Factor ${f.factorNumber}: ${f.factorName} (${f.score}/5)`);
      f.recommendations.slice(0, 2).forEach((r) => console.log(`    → ${r}`));
    }
  }
}

// Demo
function main(): void {
  const mockReport: TwelveFactorReport = {
    overallScore: 38,
    maturityLevel: "Staging",
    maturityLevelNumber: 3,
    factors: Array.from({ length: 12 }, (_, i) => ({
      factorNumber: i + 1,
      factorName: `Factor ${i + 1}`,
      score: Math.floor(Math.random() * 3) + 2,
      status: "needs_improvement" as const,
      evidence: [],
      gaps: [],
      recommendations: ["Review and improve this factor"],
    })),
    criticalGaps: ["No explicit context management", "Missing rate limiting"],
    improvementPriorities: ["Add context budget tracking", "Implement circuit breakers", "Add output validation"],
    assessedAt: new Date().toISOString(),
  };

  const dashboard = new MaturityDashboard();
  dashboard.printReport(mockReport);
  dashboard.printRoadmap(mockReport);
}

main();
