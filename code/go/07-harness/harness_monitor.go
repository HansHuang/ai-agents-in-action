// harness_monitor.go — Real-time monitoring and alerting for the production harness (Go port).
//
// See: docs/07-harness-engineering/01-the-harness-mindset.md
package main

import (
	"fmt"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Metrics types
// ---------------------------------------------------------------------------

// RequestMetrics holds per-request statistics.
type RequestMetrics struct {
	RequestID    string
	Timestamp    time.Time
	DurationMs   float64
	TokensUsed   int
	Route        string
	Success      bool
	ErrorMessage string
}

// MonitorSnapshot is a point-in-time view of harness performance.
type MonitorSnapshot struct {
	TotalRequests     int
	SuccessRate       float64
	AvgDurationMs     float64
	TotalTokensUsed   int
	AlertsFired       int
	RequestsPerMinute float64
}

// AlertRule triggers a notification when a threshold is exceeded.
type AlertRule struct {
	Name      string
	Metric    string // "error_rate" | "avg_latency_ms" | "tokens_per_minute"
	Threshold float64
	Callback  func(rule AlertRule, value float64)
}

// ---------------------------------------------------------------------------
// HarnessMonitor
// ---------------------------------------------------------------------------

// HarnessMonitor collects and aggregates harness metrics.
type HarnessMonitor struct {
	mu          sync.Mutex
	requests    []RequestMetrics
	alertRules  []AlertRule
	alertsFired int
}

// NewHarnessMonitor creates a new monitor.
func NewHarnessMonitor() *HarnessMonitor {
	return &HarnessMonitor{}
}

// AddAlertRule registers an alert rule.
func (m *HarnessMonitor) AddAlertRule(rule AlertRule) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.alertRules = append(m.alertRules, rule)
}

// Record adds a request metric and checks alert rules.
func (m *HarnessMonitor) Record(req RequestMetrics) {
	m.mu.Lock()
	defer m.mu.Unlock()
	req.Timestamp = time.Now()
	m.requests = append(m.requests, req)
	m.checkAlerts()
}

// Snapshot returns the current aggregated metrics.
func (m *HarnessMonitor) Snapshot() MonitorSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()

	n := len(m.requests)
	if n == 0 {
		return MonitorSnapshot{}
	}

	var successes int
	var totalDuration float64
	var totalTokens int
	for _, r := range m.requests {
		if r.Success {
			successes++
		}
		totalDuration += r.DurationMs
		totalTokens += r.TokensUsed
	}

	// Requests per minute in the last 60s
	cutoff := time.Now().Add(-60 * time.Second)
	var recentCount float64
	for _, r := range m.requests {
		if r.Timestamp.After(cutoff) {
			recentCount++
		}
	}

	return MonitorSnapshot{
		TotalRequests:     n,
		SuccessRate:       float64(successes) / float64(n),
		AvgDurationMs:     totalDuration / float64(n),
		TotalTokensUsed:   totalTokens,
		AlertsFired:       m.alertsFired,
		RequestsPerMinute: recentCount,
	}
}

func (m *HarnessMonitor) checkAlerts() {
	snap := m.Snapshot()
	for _, rule := range m.alertRules {
		var value float64
		switch rule.Metric {
		case "error_rate":
			value = 1 - snap.SuccessRate
		case "avg_latency_ms":
			value = snap.AvgDurationMs
		}
		if value > rule.Threshold {
			m.alertsFired++
			if rule.Callback != nil {
				go rule.Callback(rule, value)
			}
		}
	}
}

// RunHarnessMonitorDemo demonstrates the monitor.
func RunHarnessMonitorDemo() {
	monitor := NewHarnessMonitor()
	monitor.AddAlertRule(AlertRule{
		Name:      "high_error_rate",
		Metric:    "error_rate",
		Threshold: 0.1,
		Callback: func(rule AlertRule, value float64) {
			fmt.Printf("[ALERT] %s: %.2f > %.2f\n", rule.Name, value, rule.Threshold)
		},
	})

	// Simulate requests
	for i := 0; i < 10; i++ {
		monitor.Record(RequestMetrics{
			RequestID:  fmt.Sprintf("req-%d", i),
			DurationMs: float64(100 + i*10),
			TokensUsed: 500,
			Route:      "agent",
			Success:    i < 8, // 80% success
		})
	}

	snap := monitor.Snapshot()
	fmt.Printf("Monitor snapshot: requests=%d success_rate=%.2f avg_latency=%.0fms\n",
		snap.TotalRequests, snap.SuccessRate, snap.AvgDurationMs)
}
