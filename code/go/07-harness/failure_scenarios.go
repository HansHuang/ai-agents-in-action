// failure_scenarios.go — Failure scenario definitions and simulation (Go port).
//
// See: docs/07-harness-engineering/01-the-harness-mindset.md
package main

import (
	"fmt"
	"math/rand"
)

// ---------------------------------------------------------------------------
// Failure scenarios
// ---------------------------------------------------------------------------

// FailureType classifies the kind of failure being simulated.
type FailureType string

const (
	FailureLLMTimeout      FailureType = "llm_timeout"
	FailureLLMRateLimit    FailureType = "llm_rate_limit"
	FailureLLMBadResponse  FailureType = "llm_bad_response"
	FailureToolTimeout     FailureType = "tool_timeout"
	FailureToolError       FailureType = "tool_error"
	FailureContextOverflow FailureType = "context_overflow"
	FailurePIILeak         FailureType = "pii_leak"
	FailurePromptInjection FailureType = "prompt_injection"
)

// FailureScenario describes a failure mode with its likelihood and impact.
type FailureScenario struct {
	Name        string
	Type        FailureType
	Description string
	Probability float64 // 0-1, used in chaos testing
	Impact      string  // "low" | "medium" | "high" | "critical"
	MitigationSteps []string
}

// FailureLibrary holds all known failure scenarios.
type FailureLibrary struct {
	scenarios []FailureScenario
}

// NewFailureLibrary returns a library with common LLM failure scenarios.
func NewFailureLibrary() *FailureLibrary {
	return &FailureLibrary{scenarios: []FailureScenario{
		{
			Name:        "LLM Provider Timeout",
			Type:        FailureLLMTimeout,
			Description: "The LLM provider does not respond within the configured timeout.",
			Probability: 0.05,
			Impact:      "high",
			MitigationSteps: []string{
				"Set a timeout on all LLM calls",
				"Retry with exponential backoff",
				"Fall back to a smaller/faster model",
			},
		},
		{
			Name:        "Rate Limit Exceeded",
			Type:        FailureLLMRateLimit,
			Description: "The LLM provider returns 429 Too Many Requests.",
			Probability: 0.10,
			Impact:      "medium",
			MitigationSteps: []string{
				"Implement token bucket rate limiting on the client side",
				"Spread requests across multiple API keys",
				"Queue requests and retry with jitter",
			},
		},
		{
			Name:        "Context Window Overflow",
			Type:        FailureContextOverflow,
			Description: "Input + history exceeds the model's maximum context size.",
			Probability: 0.08,
			Impact:      "medium",
			MitigationSteps: []string{
				"Truncate or summarise conversation history",
				"Use a model with a larger context window",
				"Implement dynamic context compression",
			},
		},
		{
			Name:        "Prompt Injection Attack",
			Type:        FailurePromptInjection,
			Description: "Malicious input attempts to override the system prompt.",
			Probability: 0.15,
			Impact:      "critical",
			MitigationSteps: []string{
				"Validate and sanitise all user inputs",
				"Use a strict system prompt boundary",
				"Run a secondary classifier to detect injection patterns",
			},
		},
		{
			Name:        "PII Leak in Output",
			Type:        FailurePIILeak,
			Description: "The model reveals personally identifiable information.",
			Probability: 0.03,
			Impact:      "critical",
			MitigationSteps: []string{
				"Scan outputs with a PII detector before returning to user",
				"Redact or mask detected PII",
				"Log and alert for every detected PII leak",
			},
		},
	}}
}

// All returns all scenarios.
func (l *FailureLibrary) All() []FailureScenario { return l.scenarios }

// ByImpact filters scenarios by impact level.
func (l *FailureLibrary) ByImpact(impact string) []FailureScenario {
	var out []FailureScenario
	for _, s := range l.scenarios {
		if s.Impact == impact {
			out = append(out, s)
		}
	}
	return out
}

// SimulateRandom simulates a random failure scenario based on probabilities.
// Returns the triggered scenario or nil if no failure occurred.
func (l *FailureLibrary) SimulateRandom() *FailureScenario {
	for i := range l.scenarios {
		if rand.Float64() < l.scenarios[i].Probability {
			return &l.scenarios[i]
		}
	}
	return nil
}

// RunFailureScenariosDemo demonstrates the failure library.
func RunFailureScenariosDemo() {
	lib := NewFailureLibrary()

	fmt.Println("All failure scenarios:")
	for _, s := range lib.All() {
		fmt.Printf("  [%s] %s (prob=%.0f%%, impact=%s)\n",
			s.Type, s.Name, s.Probability*100, s.Impact)
		for _, m := range s.MitigationSteps {
			fmt.Printf("    → %s\n", m)
		}
	}

	fmt.Println("\nCritical scenarios:")
	for _, s := range lib.ByImpact("critical") {
		fmt.Printf("  %s\n", s.Name)
	}
}
