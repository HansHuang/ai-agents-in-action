// harness_runbook.go — Operational runbook and incident response (Go port).
//
// See: docs/07-harness-engineering/01-the-harness-mindset.md
package main

import (
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Runbook types
// ---------------------------------------------------------------------------

// Severity classifies incident severity.
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
)

// RunbookStep is a single action in the runbook.
type RunbookStep struct {
	Order       int
	Action      string
	Owner       string
	TimeoutMins int
}

// Runbook describes how to respond to an incident.
type Runbook struct {
	Name        string
	TriggerOn   []string // alert names or error codes
	Severity    Severity
	Steps       []RunbookStep
	Escalation  string
}

// IncidentRecord logs an incident and its resolution.
type IncidentRecord struct {
	ID          string
	Runbook     string
	StartedAt   time.Time
	ResolvedAt  *time.Time
	Steps       []string // steps completed
	Resolution  string
}

// ---------------------------------------------------------------------------
// RunbookEngine
// ---------------------------------------------------------------------------

// RunbookEngine maps alert names to runbooks and executes them.
type RunbookEngine struct {
	runbooks map[string]Runbook
}

// NewRunbookEngine creates an engine with built-in default runbooks.
func NewRunbookEngine() *RunbookEngine {
	e := &RunbookEngine{runbooks: make(map[string]Runbook)}

	e.Register(Runbook{
		Name:      "high_error_rate",
		TriggerOn: []string{"high_error_rate"},
		Severity:  SeverityCritical,
		Steps: []RunbookStep{
			{1, "Check LLM provider status page", "on-call", 5},
			{2, "Switch to fallback model if provider is down", "on-call", 10},
			{3, "Increase retry budget", "on-call", 5},
			{4, "Notify stakeholders", "lead", 15},
		},
		Escalation: "engineering-lead@company.com",
	})

	e.Register(Runbook{
		Name:      "pii_leak_detected",
		TriggerOn: []string{"pii_leak"},
		Severity:  SeverityCritical,
		Steps: []RunbookStep{
			{1, "Immediately disable affected endpoint", "security", 5},
			{2, "Identify affected requests from logs", "security", 30},
			{3, "Notify privacy team and legal", "security", 15},
			{4, "File incident report", "security", 60},
		},
		Escalation: "security@company.com",
	})

	return e
}

// Register adds or replaces a runbook.
func (e *RunbookEngine) Register(rb Runbook) {
	e.runbooks[rb.Name] = rb
}

// FindRunbook returns the runbook for the given alert, or nil.
func (e *RunbookEngine) FindRunbook(alertName string) *Runbook {
	for _, rb := range e.runbooks {
		for _, trigger := range rb.TriggerOn {
			if trigger == alertName {
				r := rb
				return &r
			}
		}
	}
	return nil
}

// ExecuteRunbook runs through a runbook's steps (dry run with logging).
func (e *RunbookEngine) ExecuteRunbook(rb *Runbook) *IncidentRecord {
	record := &IncidentRecord{
		ID:        fmt.Sprintf("INC-%d", time.Now().UnixMilli()),
		Runbook:   rb.Name,
		StartedAt: time.Now(),
	}

	fmt.Printf("[Runbook] Starting: %s (severity=%s)\n", rb.Name, rb.Severity)
	for _, step := range rb.Steps {
		fmt.Printf("[Runbook] Step %d/%d: %s (owner=%s, timeout=%dmin)\n",
			step.Order, len(rb.Steps), step.Action, step.Owner, step.TimeoutMins)
		record.Steps = append(record.Steps, step.Action)
	}

	if rb.Escalation != "" {
		fmt.Printf("[Runbook] Escalation contact: %s\n", rb.Escalation)
	}

	now := time.Now()
	record.ResolvedAt = &now
	record.Resolution = "runbook completed"
	return record
}

// RunbookSummary returns a formatted summary of all registered runbooks.
func (e *RunbookEngine) RunbookSummary() string {
	var sb strings.Builder
	for name, rb := range e.runbooks {
		fmt.Fprintf(&sb, "%-30s severity=%-10s steps=%d\n", name, rb.Severity, len(rb.Steps))
	}
	return sb.String()
}

// RunHarnessRunbookDemo demonstrates the runbook engine.
func RunHarnessRunbookDemo() {
	engine := NewRunbookEngine()
	fmt.Println("Registered runbooks:")
	fmt.Println(engine.RunbookSummary())

	rb := engine.FindRunbook("high_error_rate")
	if rb != nil {
		record := engine.ExecuteRunbook(rb)
		fmt.Printf("\nIncident %s started at %s\n", record.ID, record.StartedAt.Format(time.RFC3339))
	}
}
