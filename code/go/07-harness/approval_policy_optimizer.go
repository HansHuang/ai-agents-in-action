// approval_policy_optimizer.go — Learns and optimises approval thresholds (Go port).
//
// See: docs/07-harness-engineering/01-the-harness-mindset.md
package main

import (
	"fmt"
	"sort"
)

// ---------------------------------------------------------------------------
// Optimiser types
// ---------------------------------------------------------------------------

// ApprovalOutcome records whether a human-approval decision was correct.
type ApprovalOutcome struct {
	RequestID    string
	RiskScore    float64
	WasApproved  bool
	WasCorrect   bool // did the approved action turn out to be safe?
}

// ThresholdStats holds performance metrics at a given threshold.
type ThresholdStats struct {
	Threshold   float64
	TruePositives  int // correctly sent for approval
	FalsePositives int // safe requests unnecessarily sent for approval
	TrueNegatives  int // correctly auto-approved
	FalseNegatives int // dangerous requests auto-approved
	F1Score     float64
	Precision   float64
	Recall      float64
}

// ---------------------------------------------------------------------------
// ApprovalPolicyOptimizer
// ---------------------------------------------------------------------------

// ApprovalPolicyOptimizer analyses historical approval data to recommend
// the optimal risk-score threshold for sending requests to humans.
type ApprovalPolicyOptimizer struct {
	history []ApprovalOutcome
}

// NewApprovalPolicyOptimizer creates an optimiser.
func NewApprovalPolicyOptimizer() *ApprovalPolicyOptimizer {
	return &ApprovalPolicyOptimizer{}
}

// Record adds an approval outcome to the history.
func (o *ApprovalPolicyOptimizer) Record(outcome ApprovalOutcome) {
	o.history = append(o.history, outcome)
}

// Evaluate computes stats for a given threshold.
func (o *ApprovalPolicyOptimizer) Evaluate(threshold float64) ThresholdStats {
	stats := ThresholdStats{Threshold: threshold}
	for _, h := range o.history {
		needsApproval := !h.WasCorrect // outcome was bad ↔ approval was needed
		requestedApproval := h.RiskScore >= threshold

		switch {
		case requestedApproval && needsApproval:
			stats.TruePositives++
		case requestedApproval && !needsApproval:
			stats.FalsePositives++
		case !requestedApproval && !needsApproval:
			stats.TrueNegatives++
		case !requestedApproval && needsApproval:
			stats.FalseNegatives++
		}
	}

	denom := float64(stats.TruePositives + stats.FalsePositives)
	if denom > 0 {
		stats.Precision = float64(stats.TruePositives) / denom
	}
	denom = float64(stats.TruePositives + stats.FalseNegatives)
	if denom > 0 {
		stats.Recall = float64(stats.TruePositives) / denom
	}
	if stats.Precision+stats.Recall > 0 {
		stats.F1Score = 2 * stats.Precision * stats.Recall / (stats.Precision + stats.Recall)
	}
	return stats
}

// FindOptimalThreshold searches a grid of thresholds and returns the one
// with the highest F1 score.
func (o *ApprovalPolicyOptimizer) FindOptimalThreshold() ThresholdStats {
	if len(o.history) == 0 {
		return ThresholdStats{Threshold: 0.5}
	}

	// Candidate thresholds
	var candidates []float64
	for i := 0; i <= 20; i++ {
		candidates = append(candidates, float64(i)/20.0)
	}

	var best ThresholdStats
	for _, t := range candidates {
		stats := o.Evaluate(t)
		if stats.F1Score > best.F1Score {
			best = stats
		}
	}
	return best
}

// Report prints a comparison of all thresholds.
func (o *ApprovalPolicyOptimizer) Report() string {
	thresholds := []float64{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	stats := make([]ThresholdStats, 0, len(thresholds))
	for _, t := range thresholds {
		stats = append(stats, o.Evaluate(t))
	}
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].F1Score > stats[j].F1Score
	})

	out := fmt.Sprintf("%-12s %-10s %-10s %-10s %-6s %-6s\n",
		"Threshold", "Precision", "Recall", "F1", "TP", "FN")
	out += fmt.Sprintln(fmt.Sprintf("%s", "------------------------------------------------------"))
	for _, s := range stats {
		out += fmt.Sprintf("%-12.2f %-10.3f %-10.3f %-10.3f %-6d %-6d\n",
			s.Threshold, s.Precision, s.Recall, s.F1Score, s.TruePositives, s.FalseNegatives)
	}
	return out
}

// RunApprovalOptimizerDemo demonstrates the policy optimiser.
func RunApprovalOptimizerDemo() {
	opt := NewApprovalPolicyOptimizer()

	// Simulated history
	samples := []ApprovalOutcome{
		{"r1", 0.9, true, false},  // high risk, approved, was bad (FP with low threshold)
		{"r2", 0.2, false, true},  // low risk, auto-approved, was correct
		{"r3", 0.85, true, false}, // high risk, approved, bad
		{"r4", 0.3, false, true},  // low risk, correct
		{"r5", 0.75, true, true},  // medium risk, approved, turned out fine
		{"r6", 0.6, false, false}, // medium risk, auto-approved, bad (false negative)
	}
	for _, s := range samples {
		opt.Record(s)
	}

	best := opt.FindOptimalThreshold()
	fmt.Printf("Optimal threshold: %.2f (F1=%.3f)\n", best.Threshold, best.F1Score)
	fmt.Println(opt.Report())
}
