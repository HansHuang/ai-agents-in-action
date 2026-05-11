/**
 * PII detection accuracy benchmark.
 *
 * Tests PII detection patterns against labelled examples and reports
 * per-type precision, recall, and F1.
 * See: docs/07-harness-engineering/02-input-guardrails-and-validation.md
 */

// ---------------------------------------------------------------------------
// PII detector (self-contained — mirrors input_guardrail_pipeline)
// ---------------------------------------------------------------------------

export type PIIType = "email" | "phone" | "ssn" | "credit_card" | "api_key";

export interface PIIMatch {
  type: PIIType;
  value: string;
  start: number;
  end: number;
}

const PII_PATTERNS: Record<PIIType, RegExp> = {
  email:       /\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b/g,
  phone:       /\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b/g,
  ssn:         /\b\d{3}-\d{2}-\d{4}\b/g,
  credit_card: /\b(?:\d[ -]?){13,16}\b/g,
  api_key:     /\b(?:sk-[a-zA-Z0-9]{20,}|AIza[a-zA-Z0-9_\-]{35}|AKIA[A-Z0-9]{16})\b/g,
};

export function detectPII(text: string): PIIMatch[] {
  const matches: PIIMatch[] = [];
  for (const [type, pattern] of Object.entries(PII_PATTERNS) as Array<[PIIType, RegExp]>) {
    const re = new RegExp(pattern.source, pattern.flags);
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      matches.push({ type, value: m[0], start: m.index, end: m.index + m[0].length });
    }
  }
  return matches;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

export interface BenchmarkCase {
  text: string;
  expectedPII: boolean;
  expectedTypes: PIIType[];
  description: string;
}

const TRUE_POSITIVES: BenchmarkCase[] = [
  { text: "Email me at alice@example.com", expectedPII: true, expectedTypes: ["email"], description: "simple email" },
  { text: "Call 555-867-5309", expectedPII: true, expectedTypes: ["phone"], description: "US phone" },
  { text: "SSN is 123-45-6789", expectedPII: true, expectedTypes: ["ssn"], description: "SSN" },
  { text: "API key: sk-abc123def456ghi789jkl012mno345pqr678stu", expectedPII: true, expectedTypes: ["api_key"], description: "OpenAI-style key" },
  { text: "My email alice@example.com and phone 415.555.2671", expectedPII: true, expectedTypes: ["email", "phone"], description: "multiple PII" },
];

const TRUE_NEGATIVES: BenchmarkCase[] = [
  { text: "The quick brown fox jumps over the lazy dog", expectedPII: false, expectedTypes: [], description: "no PII" },
  { text: "Version 1.2.3 released on 2024-01-01", expectedPII: false, expectedTypes: [], description: "version numbers" },
  { text: "Connect to localhost:8080", expectedPII: false, expectedTypes: [], description: "localhost URL" },
  { text: "HTTP 200 OK response", expectedPII: false, expectedTypes: [], description: "HTTP status" },
  { text: "Array index: arr[123]", expectedPII: false, expectedTypes: [], description: "array index" },
];

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

export interface TypeMetrics {
  truePositives: number;
  falsePositives: number;
  falseNegatives: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface BenchmarkReport {
  totalCases: number;
  overallAccuracy: number;
  typeMetrics: Record<PIIType, TypeMetrics>;
  falsePositives: Array<{ text: string; detectedTypes: PIIType[] }>;
  falseNegatives: Array<{ text: string; expectedTypes: PIIType[] }>;
}

export function runBenchmark(cases: BenchmarkCase[] = [...TRUE_POSITIVES, ...TRUE_NEGATIVES]): BenchmarkReport {
  let correct = 0;
  const tpCount: Record<PIIType, number> = { email: 0, phone: 0, ssn: 0, credit_card: 0, api_key: 0 };
  const fpCount: Record<PIIType, number> = { email: 0, phone: 0, ssn: 0, credit_card: 0, api_key: 0 };
  const fnCount: Record<PIIType, number> = { email: 0, phone: 0, ssn: 0, credit_card: 0, api_key: 0 };
  const falsePositives: Array<{ text: string; detectedTypes: PIIType[] }> = [];
  const falseNegatives: Array<{ text: string; expectedTypes: PIIType[] }> = [];

  for (const c of cases) {
    const matches = detectPII(c.text);
    const detectedTypes = [...new Set(matches.map((m) => m.type))];
    const detectedPII = detectedTypes.length > 0;

    if (detectedPII === c.expectedPII) correct++;

    // Per-type accounting
    for (const expected of c.expectedTypes) {
      if (detectedTypes.includes(expected)) tpCount[expected]++;
      else { fnCount[expected]++; falseNegatives.push({ text: c.text, expectedTypes: c.expectedTypes }); }
    }
    for (const detected of detectedTypes) {
      if (!c.expectedTypes.includes(detected)) {
        fpCount[detected]++;
        falsePositives.push({ text: c.text, detectedTypes });
      }
    }
  }

  const typeMetrics = {} as Record<PIIType, TypeMetrics>;
  for (const t of Object.keys(PII_PATTERNS) as PIIType[]) {
    const tp = tpCount[t], fp = fpCount[t], fn = fnCount[t];
    const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 1;
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    typeMetrics[t] = { truePositives: tp, falsePositives: fp, falseNegatives: fn, precision, recall, f1 };
  }

  return {
    totalCases: cases.length,
    overallAccuracy: correct / cases.length,
    typeMetrics,
    falsePositives: [...new Set(falsePositives.map((f) => JSON.stringify(f)))].map((s) => JSON.parse(s) as { text: string; detectedTypes: PIIType[] }),
    falseNegatives: [...new Set(falseNegatives.map((f) => JSON.stringify(f)))].map((s) => JSON.parse(s) as { text: string; expectedTypes: PIIType[] }),
  };
}

/** Print benchmark report. */
export function printBenchmarkReport(report: BenchmarkReport): void {
  console.log(`\n=== PII Detection Benchmark (${report.totalCases} cases) ===`);
  console.log(`  Overall Accuracy: ${(report.overallAccuracy * 100).toFixed(1)}%`);
  console.log("\n  Per-type metrics:");
  console.log("  Type".padEnd(14) + "Precision".padStart(10) + "Recall".padStart(10) + "F1".padStart(10));
  console.log("  " + "-".repeat(42));
  for (const [type, m] of Object.entries(report.typeMetrics)) {
    console.log(
      `  ${type.padEnd(12)}` +
      `${(m.precision * 100).toFixed(1)}%`.padStart(10) +
      `${(m.recall * 100).toFixed(1)}%`.padStart(10) +
      `${(m.f1 * 100).toFixed(1)}%`.padStart(10)
    );
  }
  if (report.falsePositives.length) {
    console.log(`\n  False positives: ${report.falsePositives.length}`);
  }
  if (report.falseNegatives.length) {
    console.log(`  False negatives: ${report.falseNegatives.length}`);
  }
}

// Demo
function main(): void {
  const report = runBenchmark();
  printBenchmarkReport(report);
}

main();
