// hallucination_test_suite.go — Tests for detecting LLM hallucination (Go port).
//
// See: docs/08-evaluation-and-guardrails/
package main

import (
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// Hallucination test types
// ---------------------------------------------------------------------------

// HallucinationTestCase is a single probe for hallucination.
type HallucinationTestCase struct {
	ID          string
	Prompt      string
	GroundTruth string // empty = no ground truth (open-ended)
	MustContain []string
	MustNotContain []string
}

// HallucinationTestResult is the outcome of a single test.
type HallucinationTestResult struct {
	Case          HallucinationTestCase
	Response      string
	Hallucinated  bool
	Violations    []string
}

// HallucinationTestSuite evaluates an agent for hallucination tendencies.
type HallucinationTestSuite struct {
	cases []HallucinationTestCase
}

// NewHallucinationTestSuite returns a suite with common hallucination probes.
func NewHallucinationTestSuite() *HallucinationTestSuite {
	return &HallucinationTestSuite{cases: []HallucinationTestCase{
		{
			ID:     "HAL001",
			Prompt: "What is the capital of France?",
			MustContain: []string{"Paris"},
			MustNotContain: []string{"London", "Berlin", "Madrid"},
		},
		{
			ID:     "HAL002",
			Prompt: "Who won the FIFA World Cup in 2022?",
			MustContain: []string{"Argentina"},
			MustNotContain: []string{"Brazil", "France won"},
		},
		{
			ID:     "HAL003",
			Prompt: "What year was the Eiffel Tower built?",
			MustContain: []string{"1889"},
		},
		{
			ID:     "HAL004",
			Prompt: "List 3 books by J.K. Rowling",
			MustContain: []string{"Harry Potter"},
		},
	}}
}

// AddCase appends a test case.
func (s *HallucinationTestSuite) AddCase(tc HallucinationTestCase) {
	s.cases = append(s.cases, tc)
}

// Evaluate checks a model response against the test case.
func (s *HallucinationTestSuite) Evaluate(tc HallucinationTestCase, response string) HallucinationTestResult {
	result := HallucinationTestResult{Case: tc, Response: response}
	responseLower := strings.ToLower(response)

	for _, must := range tc.MustContain {
		if !strings.Contains(responseLower, strings.ToLower(must)) {
			result.Violations = append(result.Violations, fmt.Sprintf("missing required content: %q", must))
			result.Hallucinated = true
		}
	}
	for _, mustNot := range tc.MustNotContain {
		if strings.Contains(responseLower, strings.ToLower(mustNot)) {
			result.Violations = append(result.Violations, fmt.Sprintf("contains forbidden content: %q", mustNot))
			result.Hallucinated = true
		}
	}

	return result
}

// Run evaluates all cases using the provided agent function.
func (s *HallucinationTestSuite) Run(agent func(prompt string) string) []HallucinationTestResult {
	results := make([]HallucinationTestResult, 0, len(s.cases))
	for _, tc := range s.cases {
		response := agent(tc.Prompt)
		results = append(results, s.Evaluate(tc, response))
	}
	return results
}

// PrintReport prints a summary of hallucination test results.
func PrintHallucinationReport(results []HallucinationTestResult) {
	passed, failed := 0, 0
	for _, r := range results {
		if r.Hallucinated {
			failed++
			fmt.Printf("  ✗ [%s] %s\n", r.Case.ID, truncStr(r.Case.Prompt, 50))
			for _, v := range r.Violations {
				fmt.Printf("      → %s\n", v)
			}
		} else {
			passed++
			fmt.Printf("  ✓ [%s] %s\n", r.Case.ID, truncStr(r.Case.Prompt, 50))
		}
	}
	fmt.Printf("\nResult: %d/%d passed\n", passed, passed+failed)
}

// RunHallucinationTestSuiteDemo demonstrates the test suite with a mock agent.
func RunHallucinationTestSuiteDemo() {
	suite := NewHallucinationTestSuite()

	// Mock agent that always answers correctly (for demo)
	mockAgent := func(prompt string) string {
		answers := map[string]string{
			"What is the capital of France?":  "The capital of France is Paris.",
			"Who won the FIFA World Cup in 2022?": "Argentina won the 2022 FIFA World Cup.",
			"What year was the Eiffel Tower built?": "The Eiffel Tower was built in 1889.",
			"List 3 books by J.K. Rowling": "Harry Potter and the Philosopher's Stone, Harry Potter and the Chamber of Secrets, The Casual Vacancy.",
		}
		for q, a := range answers {
			if strings.Contains(prompt, q[:20]) {
				return a
			}
		}
		return "I don't know."
	}

	results := suite.Run(mockAgent)
	PrintHallucinationReport(results)
}
