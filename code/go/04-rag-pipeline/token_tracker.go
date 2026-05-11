// token_tracker.go — Additional TokenTracker methods and RunTokenTracker demo.
//
// TokenTracker and TokenUsage are defined in memory_manager.go.
//
// See: docs/03-memory-and-retrieval/01-short-term-memory.md
package ragpipeline

import (
	"encoding/json"
	"fmt"
	"math"
)

// TotalTokens returns total tokens (input + output) across all calls.
func (t *TokenTracker) TotalTokens() int {
	return t.TotalInputTokens() + t.TotalOutputTokens()
}

// EstimateRemaining returns remaining budget in USD, or nil if no cap set.
func (t *TokenTracker) EstimateRemaining() *float64 {
	if t.BudgetCap == nil {
		return nil
	}
	rem := math.Max(0.0, *t.BudgetCap-t.TotalCost())
	return &rem
}

// ToJSON exports all usage records as a JSON string.
func (t *TokenTracker) ToJSON() (string, error) {
	t.mu.Lock()
	records := make([]*TokenUsage, len(t.records))
	copy(records, t.records)
	t.mu.Unlock()

	totalIn, totalOut, totalCost := 0, 0, 0.0
	type serialRecord struct {
		Model        string `json:"model"`
		InputTokens  int    `json:"input_tokens"`
		OutputTokens int    `json:"output_tokens"`
		Timestamp    string `json:"timestamp"`
		Purpose      string `json:"purpose"`
	}
	recs := make([]serialRecord, len(records))
	for i, r := range records {
		totalIn += r.InputTokens
		totalOut += r.OutputTokens
		totalCost += r.CostUSD()
		recs[i] = serialRecord{
			Model:        r.Model,
			InputTokens:  r.InputTokens,
			OutputTokens: r.OutputTokens,
			Timestamp:    r.Timestamp.Format("2006-01-02T15:04:05Z"),
			Purpose:      r.Purpose,
		}
	}
	payload := map[string]interface{}{
		"total_calls":         len(records),
		"total_input_tokens":  totalIn,
		"total_output_tokens": totalOut,
		"total_cost_usd":      totalCost,
		"budget_cap":          t.BudgetCap,
		"records":             recs,
	}
	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Reset clears all records and resets the budget warning flag.
func (t *TokenTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.records = nil
	t.warned = false
}

// RunTokenTracker demonstrates the TokenTracker with a simulated session.
func RunTokenTracker() {
	cap := 0.01
	tracker := NewTokenTracker(&cap)
	tracker.RecordCall("gpt-4o", 1500, 350, "plan")
	tracker.RecordCall("gpt-4o-mini", 800, 200, "summarize")
	tracker.RecordCall("gpt-4o", 2000, 400, "execute")
	tracker.RecordCall("gpt-4o-mini", 600, 150, "summarize")
	tracker.RecordCall("gpt-4o", 1800, 380, "finalize")

	fmt.Println(tracker.GenerateReport())
	fmt.Printf("Budget exceeded: %v\n", tracker.IsBudgetExceeded())
	if rem := tracker.EstimateRemaining(); rem != nil {
		fmt.Printf("Remaining:       $%.6f\n", *rem)
	}
	if j, err := tracker.ToJSON(); err == nil {
		fmt.Printf("JSON export:\n%s\n", j)
	}
}
