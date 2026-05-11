// harness_config.go — Additional configuration presets and helpers (Go port).
//
// Builds on top of HarnessConfig defined in harness_state_machine.go.
// See: docs/07-harness-engineering/01-the-harness-mindset.md
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"time"
)

// ProductionPreset returns a high-security, high-reliability config preset.
func ProductionPreset() HarnessConfig {
	c := DefaultConfig()
	c.MaxAgentIterations = 10
	c.MaxRetriesPerState = 2
	c.LLMTimeout = 30 * time.Second
	return c
}

// DevelopmentPreset returns a permissive config suitable for development.
func DevelopmentPreset() HarnessConfig {
	c := DefaultConfig()
	c.BlockedPhrases = nil
	c.RequireApprovalFor = nil
	c.MaxAgentIterations = 20
	return c
}

// HarnessConfigFromEnv loads overrides from environment variables (HARNESS_ prefix).
func HarnessConfigFromEnv() HarnessConfig {
	c := DefaultConfig()
	if v := os.Getenv("HARNESS_MAX_INPUT_LENGTH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.MaxInputLength = n
		}
	}
	if v := os.Getenv("HARNESS_MAX_AGENT_ITERATIONS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.MaxAgentIterations = n
		}
	}
	return c
}

// HarnessConfigToJSON serialises a config to JSON.
func HarnessConfigToJSON(c HarnessConfig) ([]byte, error) {
	type jsonConfig struct {
		MaxInputLength     int      `json:"max_input_length"`
		MinInputLength     int      `json:"min_input_length"`
		MaxRetriesPerState int      `json:"max_retries_per_state"`
		MaxAgentIterations int      `json:"max_agent_iterations"`
		TokenBudgetPerReq  int      `json:"token_budget_per_req"`
		BlockedPhrases     []string `json:"blocked_phrases"`
		RequireApprovalFor []string `json:"require_approval_for"`
	}
	return json.Marshal(jsonConfig{
		MaxInputLength:     c.MaxInputLength,
		MinInputLength:     c.MinInputLength,
		MaxRetriesPerState: c.MaxRetriesPerState,
		MaxAgentIterations: c.MaxAgentIterations,
		TokenBudgetPerReq:  c.TokenBudgetPerReq,
		BlockedPhrases:     c.BlockedPhrases,
		RequireApprovalFor: c.RequireApprovalFor,
	})
}

// RunHarnessConfigDemo demonstrates configuration presets.
func RunHarnessConfigDemo() {
	prod := ProductionPreset()
	dev := DevelopmentPreset()

	prodJSON, _ := HarnessConfigToJSON(prod)
	devJSON, _ := HarnessConfigToJSON(dev)

	fmt.Printf("Production preset:\n%s\n\n", prodJSON)
	fmt.Printf("Development preset:\n%s\n", devJSON)
}
