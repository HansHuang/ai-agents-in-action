package main

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// MaturityDashboard
// ---------------------------------------------------------------------------

// MaturityLevel is a production-readiness maturity tier.
type MaturityLevel string

const (
	MaturityExperimental MaturityLevel = "experimental"
	MaturityBeta         MaturityLevel = "beta"
	MaturityStable       MaturityLevel = "stable"
	MaturityProduction   MaturityLevel = "production"
)

// MaturityDimension is one axis of the maturity assessment.
type MaturityDimension struct {
	Name        string
	Score       int // 0-100
	MaxScore    int // usually 100
	Description string
	Notes       string
}

// MaturityReport is the overall result of a maturity assessment.
type MaturityReport struct {
	AgentName       string
	Version         string
	AssessedAt      time.Time
	Dimensions      []MaturityDimension
	OverallScore    int
	Level           MaturityLevel
	Recommendations []string
}

// MaturityDashboard tracks maturity reports over time.
type MaturityDashboard struct {
	reports []MaturityReport
}

// NewMaturityDashboard creates an empty dashboard.
func NewMaturityDashboard() *MaturityDashboard { return &MaturityDashboard{} }

// AddReport records a maturity report.
func (d *MaturityDashboard) AddReport(r MaturityReport) {
	if r.AssessedAt.IsZero() {
		r.AssessedAt = time.Now()
	}
	r.OverallScore = calcOverallScore(r.Dimensions)
	r.Level = scoreToLevel(r.OverallScore)
	d.reports = append(d.reports, r)
}

// Latest returns the most recently added report.
func (d *MaturityDashboard) Latest() *MaturityReport {
	if len(d.reports) == 0 {
		return nil
	}
	return &d.reports[len(d.reports)-1]
}

// Trend returns overall scores in chronological order.
func (d *MaturityDashboard) Trend() []int {
	scores := make([]int, len(d.reports))
	for i, r := range d.reports {
		scores[i] = r.OverallScore
	}
	return scores
}

// calcOverallScore computes the weighted average across all dimensions.
func calcOverallScore(dims []MaturityDimension) int {
	if len(dims) == 0 {
		return 0
	}
	var total, maxTotal int
	for _, d := range dims {
		total += d.Score
		maxTotal += d.MaxScore
	}
	if maxTotal == 0 {
		return 0
	}
	return total * 100 / maxTotal
}

// scoreToLevel maps an overall score to a maturity level.
func scoreToLevel(score int) MaturityLevel {
	switch {
	case score >= 85:
		return MaturityProduction
	case score >= 70:
		return MaturityStable
	case score >= 50:
		return MaturityBeta
	default:
		return MaturityExperimental
	}
}

// PrintReport prints a human-readable maturity report.
func PrintMaturityReport(r MaturityReport) {
	fmt.Printf("Maturity Report: %s v%s\n", r.AgentName, r.Version)
	fmt.Printf("Assessed: %s\n", r.AssessedAt.Format(time.DateTime))
	fmt.Printf("Overall Score: %d/100 → %s\n", r.OverallScore, r.Level)
	fmt.Println()

	// Sort dimensions by score ascending so gaps are obvious
	dims := make([]MaturityDimension, len(r.Dimensions))
	copy(dims, r.Dimensions)
	sort.Slice(dims, func(i, j int) bool {
		return dims[i].Score < dims[j].Score
	})

	fmt.Printf("%-30s %-8s %s\n", "Dimension", "Score", "Notes")
	fmt.Println(strings.Repeat("-", 70))
	for _, d := range dims {
		fmt.Printf("%-30s %3d/%-3d  %s\n", d.Name, d.Score, d.MaxScore, d.Notes)
	}

	if len(r.Recommendations) > 0 {
		fmt.Println("\nRecommendations:")
		for _, rec := range r.Recommendations {
			fmt.Printf("  • %s\n", rec)
		}
	}
}

// DefaultMaturityAssessment creates a sample assessment for an agent.
func DefaultMaturityAssessment(agentName, version string) MaturityReport {
	return MaturityReport{
		AgentName:  agentName,
		Version:    version,
		AssessedAt: time.Now(),
		Dimensions: []MaturityDimension{
			{Name: "Observability", Score: 75, MaxScore: 100, Notes: "Traces and metrics present; distributed tracing missing"},
			{Name: "Testing", Score: 80, MaxScore: 100, Notes: "Unit and integration tests; load tests pending"},
			{Name: "Security", Score: 70, MaxScore: 100, Notes: "Input guardrails active; pentest not completed"},
			{Name: "Reliability", Score: 85, MaxScore: 100, Notes: "Circuit breaker and fallback chain in place"},
			{Name: "Scalability", Score: 65, MaxScore: 100, Notes: "Stateless design; autoscaling not configured"},
			{Name: "Documentation", Score: 90, MaxScore: 100, Notes: "API docs and runbooks up to date"},
			{Name: "CI/CD", Score: 80, MaxScore: 100, Notes: "Automated pipeline; canary deploys not implemented"},
			{Name: "Compliance", Score: 60, MaxScore: 100, Notes: "GDPR review in progress; SOC 2 audit pending"},
		},
		Recommendations: []string{
			"Implement distributed tracing with OpenTelemetry",
			"Add load testing to the CI pipeline",
			"Complete penetration testing",
			"Configure horizontal autoscaling",
			"Complete GDPR and SOC 2 compliance review",
		},
	}
}

// RunMaturityDashboardDemo demonstrates the maturity dashboard.
func RunMaturityDashboardDemo() {
	dash := NewMaturityDashboard()

	r := DefaultMaturityAssessment("my-ai-agent", "1.2.0")
	dash.AddReport(r)

	latest := dash.Latest()
	PrintMaturityReport(*latest)

	fmt.Printf("\nTrend: %v\n", dash.Trend())
}
