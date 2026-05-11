// pii_benchmark.go — PII detection accuracy benchmark (Go port).
//
// See: docs/07-harness-engineering/02-input-guardrails-and-validation.md
package main

import (
	"fmt"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// PII patterns
// ---------------------------------------------------------------------------

// PIIPattern describes a regex-based PII detector.
type PIIPattern struct {
	Name    string
	Pattern *regexp.Regexp
}

// DefaultPIIPatterns returns common PII patterns.
func DefaultPIIPatterns() []PIIPattern {
	return []PIIPattern{
		{"email", regexp.MustCompile(`(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}`)},
		{"ssn", regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)},
		{"credit_card", regexp.MustCompile(`\b(?:\d[ -]?){13,16}\b`)},
		{"phone_us", regexp.MustCompile(`\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b`)},
		{"ip_address", regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`)},
	}
}

// DetectPII scans text for PII using the given patterns.
// Returns a map of pattern name → slice of matches.
func DetectPII(text string, patterns []PIIPattern) map[string][]string {
	results := make(map[string][]string)
	for _, p := range patterns {
		matches := p.Pattern.FindAllString(text, -1)
		if len(matches) > 0 {
			results[p.Name] = matches
		}
	}
	return results
}

// RedactPII replaces PII matches with [REDACTED-<type>].
func RedactPII(text string, patterns []PIIPattern) string {
	for _, p := range patterns {
		text = p.Pattern.ReplaceAllStringFunc(text, func(m string) string {
			return fmt.Sprintf("[REDACTED-%s]", strings.ToUpper(p.Name))
		})
	}
	return text
}

// ---------------------------------------------------------------------------
// Benchmark types
// ---------------------------------------------------------------------------

// PIIBenchmarkCase is a single test case for PII detection.
type PIIBenchmarkCase struct {
	ID          string
	Text        string
	ExpectedPII map[string]bool // pattern name → should be detected
}

// PIIBenchmarkResult holds the result for one case.
type PIIBenchmarkResult struct {
	Case       PIIBenchmarkCase
	Detected   map[string][]string
	Correct    bool
}

// PIIBenchmark runs PII detection over a set of test cases.
type PIIBenchmark struct {
	patterns []PIIPattern
	cases    []PIIBenchmarkCase
}

// NewPIIBenchmark creates a benchmark with the default patterns.
func NewPIIBenchmark() *PIIBenchmark {
	return &PIIBenchmark{patterns: DefaultPIIPatterns()}
}

// AddCase adds a test case.
func (b *PIIBenchmark) AddCase(tc PIIBenchmarkCase) {
	b.cases = append(b.cases, tc)
}

// Run executes all test cases and returns results.
func (b *PIIBenchmark) Run() []PIIBenchmarkResult {
	results := make([]PIIBenchmarkResult, 0, len(b.cases))
	for _, tc := range b.cases {
		detected := DetectPII(tc.Text, b.patterns)
		correct := true
		for piiType, shouldDetect := range tc.ExpectedPII {
			_, wasDetected := detected[piiType]
			if wasDetected != shouldDetect {
				correct = false
			}
		}
		results = append(results, PIIBenchmarkResult{
			Case:     tc,
			Detected: detected,
			Correct:  correct,
		})
	}
	return results
}

// PrintReport prints a summary of benchmark results.
func (b *PIIBenchmark) PrintReport(results []PIIBenchmarkResult) {
	passed := 0
	for _, r := range results {
		status := "✗"
		if r.Correct {
			status = "✓"
			passed++
		}
		fmt.Printf("  %s [%s] %s\n", status, r.Case.ID, truncStr(r.Case.Text, 50))
		for piiType, matches := range r.Detected {
			fmt.Printf("      Detected %s: %v\n", piiType, matches)
		}
	}
	fmt.Printf("\nResult: %d/%d passed\n", passed, len(results))
}

// RunPIIBenchmarkDemo demonstrates PII detection and redaction.
func RunPIIBenchmarkDemo() {
	bench := NewPIIBenchmark()
	bench.AddCase(PIIBenchmarkCase{
		ID:   "B001",
		Text: "Contact me at alice@example.com or call 415-555-1234",
		ExpectedPII: map[string]bool{"email": true, "phone_us": true},
	})
	bench.AddCase(PIIBenchmarkCase{
		ID:   "B002",
		Text: "My SSN is 123-45-6789",
		ExpectedPII: map[string]bool{"ssn": true},
	})
	bench.AddCase(PIIBenchmarkCase{
		ID:   "B003",
		Text: "The weather is sunny today",
		ExpectedPII: map[string]bool{},
	})

	results := bench.Run()
	bench.PrintReport(results)

	// Demo redaction
	text := "Please contact john@acme.com, SSN 987-65-4321"
	redacted := RedactPII(text, DefaultPIIPatterns())
	fmt.Printf("\nOriginal: %s\nRedacted: %s\n", text, redacted)
}
