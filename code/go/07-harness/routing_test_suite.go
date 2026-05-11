// routing_test_suite.go — Tests for intent classification / routing (Go port).
//
// See: docs/07-harness-engineering/03-routing-and-intent-classification.md
package main

import (
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// Routing test types
// ---------------------------------------------------------------------------

// MyRoutingTestCase is a single routing probe.
type MyRoutingTestCase struct {
	ID             string
	Input          string
	ExpectedRoute  string
}

// MyRoutingTestResult holds the outcome of a routing test.
type MyRoutingTestResult struct {
	Case          MyRoutingTestCase
	ActualRoute   string
	Correct       bool
}

// MyRoutingTestSuite evaluates a router against labelled examples.
type MyRoutingTestSuite struct {
	cases []MyRoutingTestCase
}

// NewMyRoutingTestSuite returns a suite with standard routing probes.
func NewMyRoutingTestSuite() *MyRoutingTestSuite {
	return &MyRoutingTestSuite{cases: []MyRoutingTestCase{
		{ID: "RT001", Input: "What is the weather in Paris?", ExpectedRoute: "weather"},
		{ID: "RT002", Input: "Book me a flight to New York", ExpectedRoute: "travel"},
		{ID: "RT003", Input: "I need help with my tax return", ExpectedRoute: "finance"},
		{ID: "RT004", Input: "Tell me a joke", ExpectedRoute: "general"},
		{ID: "RT005", Input: "My account is locked, help!", ExpectedRoute: "support"},
		{ID: "RT006", Input: "What are the symptoms of diabetes?", ExpectedRoute: "health"},
	}}
}

// AddCase appends a test case.
func (s *MyRoutingTestSuite) AddCase(tc MyRoutingTestCase) {
	s.cases = append(s.cases, tc)
}

// Run evaluates all cases using the provided router function.
func (s *MyRoutingTestSuite) Run(router func(input string) string) []MyRoutingTestResult {
	results := make([]MyRoutingTestResult, 0, len(s.cases))
	for _, tc := range s.cases {
		actual := router(tc.Input)
		results = append(results, MyRoutingTestResult{
			Case:        tc,
			ActualRoute: actual,
			Correct:     actual == tc.ExpectedRoute,
		})
	}
	return results
}

// PrintRoutingReport prints a summary of routing results.
func PrintRoutingReport(results []MyRoutingTestResult) {
	passed, failed := 0, 0
	for _, r := range results {
		icon := "✓"
		if !r.Correct {
			icon = "✗"
			failed++
		} else {
			passed++
		}
		fmt.Printf("  %s [%s] %-40s expected=%-10s got=%s\n",
			icon, r.Case.ID, truncStr(r.Case.Input, 38), r.Case.ExpectedRoute, r.ActualRoute)
	}
	accuracy := 0.0
	if len(results) > 0 {
		accuracy = float64(passed) / float64(len(results)) * 100
	}
	fmt.Printf("\nAccuracy: %d/%d (%.1f%%)\n", passed, passed+failed, accuracy)
}

// keywordRouter is a simple keyword-based router for demo purposes.
func keywordRouter(input string) string {
	lower := strings.ToLower(input)
	rules := []struct {
		keywords []string
		route    string
	}{
		{[]string{"weather", "temperature", "forecast"}, "weather"},
		{[]string{"flight", "hotel", "travel", "trip"}, "travel"},
		{[]string{"tax", "finance", "investment", "money"}, "finance"},
		{[]string{"symptom", "health", "doctor", "medical"}, "health"},
		{[]string{"account", "locked", "help", "support"}, "support"},
	}
	for _, r := range rules {
		for _, kw := range r.keywords {
			if strings.Contains(lower, kw) {
				return r.route
			}
		}
	}
	return "general"
}

// RunRoutingTestSuiteDemo demonstrates the routing test suite.
func RunRoutingTestSuiteDemo() {
	suite := NewMyRoutingTestSuite()
	results := suite.Run(keywordRouter)
	fmt.Println("Routing Test Suite Results:")
	PrintRoutingReport(results)
}
