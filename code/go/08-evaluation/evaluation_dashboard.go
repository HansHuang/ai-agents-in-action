package main

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// EvaluationDashboard
// ---------------------------------------------------------------------------

// MetricRow stores evaluation results for one agent/run combination.
type MetricRow struct {
	RunID       string
	AgentName   string
	EvaluatedAt time.Time
	Scores      map[string]float64
	Tags        map[string]string
}

// EvaluationDashboard accumulates MetricRows and computes aggregate statistics.
type EvaluationDashboard struct {
	rows []MetricRow
}

// NewEvaluationDashboard creates an empty dashboard.
func NewEvaluationDashboard() *EvaluationDashboard {
	return &EvaluationDashboard{}
}

// Add appends a metric row.
func (d *EvaluationDashboard) Add(row MetricRow) {
	if row.EvaluatedAt.IsZero() {
		row.EvaluatedAt = time.Now()
	}
	d.rows = append(d.rows, row)
}

// AverageByAgent returns average scores per metric, grouped by agent name.
func (d *EvaluationDashboard) AverageByAgent() map[string]map[string]float64 {
	sums := make(map[string]map[string]float64)
	counts := make(map[string]int)
	for _, r := range d.rows {
		if sums[r.AgentName] == nil {
			sums[r.AgentName] = make(map[string]float64)
		}
		for k, v := range r.Scores {
			sums[r.AgentName][k] += v
		}
		counts[r.AgentName]++
	}
	avgs := make(map[string]map[string]float64, len(sums))
	for agent, m := range sums {
		avgs[agent] = make(map[string]float64, len(m))
		for k, v := range m {
			avgs[agent][k] = v / float64(counts[agent])
		}
	}
	return avgs
}

// TopN returns the N runs with the highest score for a given metric.
func (d *EvaluationDashboard) TopN(metric string, n int) []MetricRow {
	rows := make([]MetricRow, len(d.rows))
	copy(rows, d.rows)
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Scores[metric] > rows[j].Scores[metric]
	})
	if n > len(rows) {
		n = len(rows)
	}
	return rows[:n]
}

// PrintSummary prints a human-readable summary table.
func (d *EvaluationDashboard) PrintSummary() {
	avgs := d.AverageByAgent()
	if len(avgs) == 0 {
		fmt.Println("  (no evaluation data)")
		return
	}

	// Collect all metric names
	metricSet := make(map[string]bool)
	for _, metrics := range avgs {
		for k := range metrics {
			metricSet[k] = true
		}
	}
	var metricNames []string
	for k := range metricSet {
		metricNames = append(metricNames, k)
	}
	sort.Strings(metricNames)

	// Header
	header := fmt.Sprintf("%-20s", "Agent")
	for _, m := range metricNames {
		header += fmt.Sprintf("  %-12s", m)
	}
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)+4))

	// Rows
	var agentNames []string
	for a := range avgs {
		agentNames = append(agentNames, a)
	}
	sort.Strings(agentNames)
	for _, agent := range agentNames {
		row := fmt.Sprintf("%-20s", agent)
		for _, m := range metricNames {
			row += fmt.Sprintf("  %-12.3f", avgs[agent][m])
		}
		fmt.Println(row)
	}
}

// RunEvaluationDashboardDemo demonstrates the evaluation dashboard.
func RunEvaluationDashboardDemo() {
	dash := NewEvaluationDashboard()

	// Simulate two agents evaluated over three runs each
	agents := []string{"gpt-4o-agent", "gpt-4o-mini-agent"}
	runs := [][]float64{
		{0.91, 0.85, 0.72},
		{0.78, 0.82, 0.68},
	}
	for i, agent := range agents {
		for j, acc := range runs[i] {
			dash.Add(MetricRow{
				RunID:     fmt.Sprintf("run-%d-%d", i, j),
				AgentName: agent,
				Scores: map[string]float64{
					"accuracy":   acc,
					"latency_ms": float64(300 + i*100 + j*20),
					"cost_usd":   float64(j+1) * 0.002 * float64(i+1),
				},
			})
		}
	}

	fmt.Println("Evaluation Dashboard Summary:")
	dash.PrintSummary()

	top := dash.TopN("accuracy", 3)
	fmt.Println("\nTop 3 runs by accuracy:")
	for _, r := range top {
		fmt.Printf("  %s / %s → %.3f\n", r.AgentName, r.RunID, r.Scores["accuracy"])
	}
}
