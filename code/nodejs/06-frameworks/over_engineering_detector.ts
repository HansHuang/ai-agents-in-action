/**
 * Over-engineering detector for multi-agent architectures.
 *
 * Analyses a project description and warns when multi-agent is overkill.
 * Stage 1: Rule-based pattern matching
 * Stage 2: Optional LLM-as-judge for nuanced cases
 * See: docs/06-frameworks-in-practice/03-crewai-autogen.md
 */

import OpenAI from "openai";

export interface OverEngineeringSignal {
  pattern: string;
  description: string;
  severity: "low" | "medium" | "high";
}

export interface OverEngineeringReport {
  projectDescription: string;
  signals: OverEngineeringSignal[];
  riskScore: number;      // 0-10
  verdict: "appropriate" | "likely_overkill" | "definitely_overkill";
  recommendation: string;
  llmAnalysis?: string;
}

// ---------------------------------------------------------------------------
// Rule-based signals
// ---------------------------------------------------------------------------

const OVERKILL_PATTERNS: Array<{ regex: RegExp; signal: OverEngineeringSignal }> = [
  {
    regex: /\b(simple|basic|small|tiny|single|one-page|demo|prototype|poc|mvp)\b/i,
    signal: { pattern: "simplicity_indicator", description: "Project scope suggests simplicity", severity: "high" },
  },
  {
    regex: /\b(crud|todo|notes?|blog|static site|landing page)\b/i,
    signal: { pattern: "crud_app", description: "CRUD app rarely needs multi-agent", severity: "high" },
  },
  {
    regex: /\bsingle (task|purpose|function|step)\b/i,
    signal: { pattern: "single_task", description: "Single-task workflow doesn't need multiple agents", severity: "medium" },
  },
  {
    regex: /\b(5|five|3|three|few|handful) (users?|requests?|queries)\b/i,
    signal: { pattern: "low_volume", description: "Very low usage volume doesn't justify multi-agent overhead", severity: "medium" },
  },
  {
    regex: /\bno (concurrency|parallel|async|background)\b/i,
    signal: { pattern: "sequential_only", description: "No parallelism needed — single agent is sufficient", severity: "medium" },
  },
];

const JUSTIFIED_PATTERNS: RegExp[] = [
  /\b(parallel|concurrent|simultaneous)\b/i,
  /\b(complex|multi-step|pipeline|workflow)\b/i,
  /\b(speciali[sz]ed|domain-specific|expert)\b/i,
  /\b(review|critique|feedback loop|self-correct)\b/i,
  /\b(research|analysis|synthesize|multiple sources)\b/i,
];

function ruleBasedAnalysis(description: string): { signals: OverEngineeringSignal[]; justifications: number } {
  const signals: OverEngineeringSignal[] = [];
  for (const { regex, signal } of OVERKILL_PATTERNS) {
    if (regex.test(description)) signals.push(signal);
  }
  const justifications = JUSTIFIED_PATTERNS.filter((r) => r.test(description)).length;
  return { signals, justifications };
}

/**
 * Detect over-engineering risks in a proposed multi-agent architecture.
 */
export async function detectOverEngineering(
  projectDescription: string,
  client?: OpenAI
): Promise<OverEngineeringReport> {
  const { signals, justifications } = ruleBasedAnalysis(projectDescription);

  const severityScore = signals.reduce(
    (s, sig) => s + (sig.severity === "high" ? 3 : sig.severity === "medium" ? 2 : 1),
    0
  );
  const riskScore = Math.min(10, Math.max(0, severityScore * 2 - justifications * 1.5));

  let verdict: OverEngineeringReport["verdict"];
  let recommendation: string;

  if (riskScore >= 7) {
    verdict = "definitely_overkill";
    recommendation = "Use a single-agent or simple function call. Multi-agent adds complexity without benefit here.";
  } else if (riskScore >= 4) {
    verdict = "likely_overkill";
    recommendation = "Start with a single agent. Add multi-agent only when you hit a concrete limitation.";
  } else {
    verdict = "appropriate";
    recommendation = "Multi-agent architecture appears justified. Proceed with clear role separation.";
  }

  let llmAnalysis: string | undefined;

  if (client && riskScore >= 3 && riskScore < 7) {
    try {
      const resp = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are an expert software architect. Evaluate whether multi-agent AI is appropriate for a project. " +
              "Be concise (2-3 sentences) and specific.",
          },
          {
            role: "user",
            content: `Project: ${projectDescription}\n\nIs multi-agent AI architecture appropriate here, or overkill?`,
          },
        ],
        temperature: 0,
        max_tokens: 120,
      });
      llmAnalysis = resp.choices[0].message.content?.trim();
    } catch { /* optional */ }
  }

  return { projectDescription, signals, riskScore, verdict, recommendation, llmAnalysis };
}

/** Print the over-engineering report. */
export function printOverEngineeringReport(report: OverEngineeringReport): void {
  console.log("\nOver-Engineering Analysis:");
  console.log(`  Project: ${report.projectDescription.slice(0, 80)}`);
  console.log(`  Risk Score: ${report.riskScore.toFixed(1)}/10`);
  console.log(`  Verdict: ${report.verdict.toUpperCase()}`);
  console.log(`  Recommendation: ${report.recommendation}`);
  if (report.signals.length > 0) {
    console.log("  Warning Signals:");
    report.signals.forEach((s) => console.log(`    [${s.severity}] ${s.description}`));
  }
  if (report.llmAnalysis) {
    console.log(`  LLM Analysis: ${report.llmAnalysis}`);
  }
}

// Demo
async function main(): Promise<void> {
  const projects = [
    "Build a simple TODO app with basic CRUD operations for 5 users",
    "Research assistant that queries multiple APIs, synthesizes findings from various sources, and creates a comprehensive report with fact-checking",
  ];

  for (const desc of projects) {
    const report = await detectOverEngineering(desc);
    printOverEngineeringReport(report);
  }
}

main().catch(console.error);
