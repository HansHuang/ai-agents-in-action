// multi_agent_comparison.go — Compare three multi-agent approaches in Go.
//
// Three approaches are compared on the same research task:
//  1. From-scratch  : reuse Agent / Task / SequentialCrew from multi_agent_from_scratch.go
//  2. CrewAI-style  : reuse crewResearch() from crewai_research_crew.go
//  3. AutoGen-style : reuse dtRunDesignSession() from autogen_design_team.go
//
// Metrics collected:
//   - Code metrics  : conceptual lines (LangChain/CrewAI/AutoGen imports, custom lines)
//   - Exec metrics  : LLM calls, total tokens, estimated cost, execution time
//   - Quality metrics: word count, sources cited, has-critique, structure score
//   - Control metrics: traceability, ability to modify comms / add agents / reorder
//
// Run (all files in the package):
//
//	go run *.go
//
// Or selectively:
//
//	go run multi_agent_comparison.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go over_engineering_detector.go langsmith_tracer.go \
//	           autogen_design_team.go crewai_research_crew.go langgraph_multi_agent.go \
//	           langgraph_react_agent.go langchain_rag_pipeline.go
//
// See: docs/02-the-agent-loop/04-multi-agent-patterns.md
package main

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	maResearchTask    = "Research the impact of AI on software developer productivity."
	maGPT4oInputCost  = 0.0025 // USD per 1K input tokens
	maGPT4oOutputCost = 0.010  // USD per 1K output tokens
)

// ---------------------------------------------------------------------------
// Metric types
// ---------------------------------------------------------------------------

// MACodeMetrics records how many lines come from each tier of abstraction.
type MACodeMetrics struct {
	AgentDefinitionLines int
	OrchestrationLines   int
	TotalLines           int
	ImportCount          int
}

// MAExecMetrics records runtime performance.
type MAExecMetrics struct {
	ExecutionTimeS   float64
	LLMCalls         int
	TotalTokens      int
	EstimatedCostUSD float64
}

// MAQualityMetrics records output quality signals.
type MAQualityMetrics struct {
	ReportLengthWords   int
	SourcesCited        int
	HasCritiqueFeedback bool
	StructureScore      int // 0-4 (one point each: summary, findings, risks, recommendations)
}

// MAControlMetrics records developer-control capabilities.
type MAControlMetrics struct {
	TraceableDecisionPath   bool
	CanModifyAgentComms     bool
	CanAddAgentMidWorkflow  bool
	CanChangeExecutionOrder bool
	ControlScore            int // 0-4
}

// MACompResult holds the full comparison result for one approach.
type MACompResult struct {
	Name    string
	Code    MACodeMetrics
	Exec    MAExecMetrics
	Quality MAQualityMetrics
	Control MAControlMetrics
	Report  string
	Error   string
}

// ---------------------------------------------------------------------------
// Metrics helpers
// ---------------------------------------------------------------------------

// maCountSources counts source citations in the output text.
func maCountSources(text string) int {
	re := regexp.MustCompile(`(?i)\([A-Z][a-zA-Z\s]+,?\s*(19|20)\d{2}\)|source:|github\.|mckinsey|gartner|nber|stanford|mit|ieee|stack overflow`)
	return len(re.FindAllString(text, -1))
}

// maHasCritique returns true if the output contains critique or feedback language.
func maHasCritique(text string) bool {
	lower := strings.ToLower(text)
	keywords := []string{"critique", "feedback", "however", "limitation", "risk", "drawback", "concern", "caveat"}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// maHeuristicStructureScore scores the report structure out of 4.
func maHeuristicStructureScore(text string) int {
	lower := strings.ToLower(text)
	score := 0
	if strings.Contains(lower, "executive summary") || strings.Contains(lower, "overview") {
		score++
	}
	if strings.Contains(lower, "finding") || strings.Contains(lower, "key insight") {
		score++
	}
	if strings.Contains(lower, "risk") || strings.Contains(lower, "limitation") || strings.Contains(lower, "challenge") {
		score++
	}
	if strings.Contains(lower, "recommend") || strings.Contains(lower, "conclusion") || strings.Contains(lower, "next step") {
		score++
	}
	return score
}

// maEstimateCost estimates USD cost for a total token count (blended rate).
func maEstimateCost(totalTokens int) float64 {
	// Assume 60% input / 40% output split
	inputTokens := float64(totalTokens) * 0.6
	outputTokens := float64(totalTokens) * 0.4
	cost := (inputTokens/1000)*maGPT4oInputCost + (outputTokens/1000)*maGPT4oOutputCost
	return float64(int(cost*10000+0.5)) / 10000
}

// ---------------------------------------------------------------------------
// Runner: approach 1 — from-scratch (reuse SequentialCrew)
// ---------------------------------------------------------------------------

func maRunFromScratch(ctx context.Context, client *openai.Client) MACompResult {
	result := MACompResult{
		Name: "From Scratch",
		Code: MACodeMetrics{
			AgentDefinitionLines: 35,
			OrchestrationLines:   60,
			TotalLines:           95,
			ImportCount:          4,
		},
		Control: MAControlMetrics{
			TraceableDecisionPath:   true,
			CanModifyAgentComms:     true,
			CanAddAgentMidWorkflow:  true,
			CanChangeExecutionOrder: true,
			ControlScore:            4,
		},
	}

	researcher, _, writer := createResearchAgents()

	tasks := []Task{
		{
			Description:    "Research the impact of AI on software developer productivity. Provide key statistics, trends, and sources.",
			ExpectedOutput: "Research notes with statistics, trends, limitations, and 3+ citations.",
			Agent:          researcher,
		},
		{
			Description:    "Write a structured executive report based on the research findings.",
			ExpectedOutput: "A 250–350 word structured report with executive summary, findings, risks, and recommendations.",
			Agent:          writer,
		},
	}

	crew := &SequentialCrew{Tasks: tasks}

	t0 := time.Now()
	metrics, _, err := runSequentialCrew(ctx, client)
	elapsed := time.Since(t0).Seconds()
	_ = elapsed
	_ = crew
	_ = metrics
	if err != nil {
		result.Error = err.Error()
		return result
	}

	report, _, finalErr := crew.kickoff(ctx, client)
	if finalErr != nil {
		result.Error = finalErr.Error()
		return result
	}
	finalReport := report[writer.Name]
	result.Exec = MAExecMetrics{
		ExecutionTimeS:   elapsed,
		LLMCalls:         metrics.LLMCalls,
		TotalTokens:      metrics.TotalTokens,
		EstimatedCostUSD: maEstimateCost(metrics.TotalTokens),
	}
	result.Quality = MAQualityMetrics{
		ReportLengthWords:   countWords(finalReport),
		SourcesCited:        maCountSources(finalReport),
		HasCritiqueFeedback: maHasCritique(finalReport),
		StructureScore:      maHeuristicStructureScore(finalReport),
	}
	result.Report = finalReport
	return result
}

// ---------------------------------------------------------------------------
// Runner: approach 2 — CrewAI-style (reuse crewResearch)
// ---------------------------------------------------------------------------

func maRunCrewAIStyle(ctx context.Context, client *openai.Client) MACompResult {
	result := MACompResult{
		Name: "CrewAI-style",
		Code: MACodeMetrics{
			AgentDefinitionLines: 25,
			OrchestrationLines:   20,
			TotalLines:           45,
			ImportCount:          6,
		},
		Control: MAControlMetrics{
			TraceableDecisionPath:   true,
			CanModifyAgentComms:     false,
			CanAddAgentMidWorkflow:  false,
			CanChangeExecutionOrder: false,
			ControlScore:            1,
		},
	}

	t0 := time.Now()
	crew, err := crewResearch(ctx, client, maResearchTask)
	if err != nil {
		result.Error = err.Error()
		return result
	}

	report := crew.Report
	result.Report = report
	result.Exec = MAExecMetrics{
		ExecutionTimeS:   time.Since(t0).Seconds(),
		LLMCalls:         3, // 3 agents, 1 call each (minimum)
		TotalTokens:      crew.TokenUsage,
		EstimatedCostUSD: maEstimateCost(crew.TokenUsage),
	}
	result.Quality = MAQualityMetrics{
		ReportLengthWords:   countWords(report),
		SourcesCited:        len(crew.Sources),
		HasCritiqueFeedback: maHasCritique(report),
		StructureScore:      maHeuristicStructureScore(report),
	}
	return result
}

// ---------------------------------------------------------------------------
// Runner: approach 3 — AutoGen-style (reuse dtRunDesignSession)
// ---------------------------------------------------------------------------

func maRunAutoGenStyle(ctx context.Context, client *openai.Client) MACompResult {
	result := MACompResult{
		Name: "AutoGen-style",
		Code: MACodeMetrics{
			AgentDefinitionLines: 20,
			OrchestrationLines:   40,
			TotalLines:           60,
			ImportCount:          5,
		},
		Control: MAControlMetrics{
			TraceableDecisionPath:   true,
			CanModifyAgentComms:     true,
			CanAddAgentMidWorkflow:  false,
			CanChangeExecutionOrder: false,
			ControlScore:            2,
		},
	}

	// AutoGen-style is conversational — repurpose it to produce a research report
	req := "Produce a structured research report on: " + maResearchTask +
		" Include: executive summary, 3+ key findings with statistics, risks/limitations, and recommendations."

	t0 := time.Now()
	design, err := dtRunDesignSession(ctx, client, req)
	if err != nil {
		result.Error = err.Error()
		return result
	}

	report := design.FinalSpec
	result.Report = report
	result.Exec = MAExecMetrics{
		ExecutionTimeS:   time.Since(t0).Seconds(),
		LLMCalls:         design.RoundsTaken * 3, // 3 agents per round
		TotalTokens:      design.TokenUsage,
		EstimatedCostUSD: maEstimateCost(design.TokenUsage),
	}
	result.Quality = MAQualityMetrics{
		ReportLengthWords:   countWords(report),
		SourcesCited:        maCountSources(report),
		HasCritiqueFeedback: maHasCritique(report),
		StructureScore:      maHeuristicStructureScore(report),
	}
	return result
}

// ---------------------------------------------------------------------------
// Table printer
// ---------------------------------------------------------------------------

// maPrintTable prints a comparison table for all three approaches.
func maPrintTable(results []MACompResult) {
	col := 22
	sep := strings.Repeat("═", col*4+5)
	hdr := fmt.Sprintf("%-*s %-*s %-*s %-*s",
		col, "Metric",
		col, "From Scratch",
		col, "CrewAI-style",
		col, "AutoGen-style")

	fmt.Printf("\n%s\n  MULTI-AGENT FRAMEWORK COMPARISON\n  Task: %s\n%s\n", sep, maResearchTask, sep)
	fmt.Println(hdr)
	fmt.Printf("%s\n", strings.Repeat("─", col*4+5))

	get := func(name, field string) string {
		for _, r := range results {
			if r.Name != name {
				continue
			}
			switch field {
			case "Error":
				if r.Error != "" {
					return "ERROR: " + r.Error[:min(len(r.Error), 20)]
				}
				return "ok"
			case "AgentLines":
				return fmt.Sprintf("%d", r.Code.AgentDefinitionLines)
			case "OrchestrLines":
				return fmt.Sprintf("%d", r.Code.OrchestrationLines)
			case "TotalLines":
				return fmt.Sprintf("%d", r.Code.TotalLines)
			case "LLMCalls":
				return fmt.Sprintf("%d", r.Exec.LLMCalls)
			case "Tokens":
				return fmt.Sprintf("%d", r.Exec.TotalTokens)
			case "CostUSD":
				return fmt.Sprintf("$%.4f", r.Exec.EstimatedCostUSD)
			case "TimeS":
				return fmt.Sprintf("%.2fs", r.Exec.ExecutionTimeS)
			case "Words":
				return fmt.Sprintf("%d", r.Quality.ReportLengthWords)
			case "Sources":
				return fmt.Sprintf("%d", r.Quality.SourcesCited)
			case "Critique":
				if r.Quality.HasCritiqueFeedback {
					return "yes"
				}
				return "no"
			case "Structure":
				return fmt.Sprintf("%d/4", r.Quality.StructureScore)
			case "ControlScore":
				return fmt.Sprintf("%d/4", r.Control.ControlScore)
			case "TraceDecisions":
				if r.Control.TraceableDecisionPath {
					return "yes"
				}
				return "no"
			case "ModifyComms":
				if r.Control.CanModifyAgentComms {
					return "yes"
				}
				return "no"
			case "AddMidRun":
				if r.Control.CanAddAgentMidWorkflow {
					return "yes"
				}
				return "no"
			case "Reorder":
				if r.Control.CanChangeExecutionOrder {
					return "yes"
				}
				return "no"
			}
		}
		return "N/A"
	}

	names := []string{"From Scratch", "CrewAI-style", "AutoGen-style"}

	printRow := func(label, field string) {
		fmt.Printf("%-*s %-*s %-*s %-*s\n", col, label,
			col, get(names[0], field),
			col, get(names[1], field),
			col, get(names[2], field))
	}

	fmt.Printf("%-*s\n", col*4+5, "  CODE METRICS")
	printRow("  Agent def. lines", "AgentLines")
	printRow("  Orchestration lines", "OrchestrLines")
	printRow("  Total lines", "TotalLines")
	fmt.Println()

	fmt.Printf("%-*s\n", col*4+5, "  EXECUTION METRICS")
	printRow("  LLM calls", "LLMCalls")
	printRow("  Total tokens", "Tokens")
	printRow("  Est. cost (USD)", "CostUSD")
	printRow("  Execution time", "TimeS")
	fmt.Println()

	fmt.Printf("%-*s\n", col*4+5, "  OUTPUT QUALITY")
	printRow("  Report words", "Words")
	printRow("  Sources cited", "Sources")
	printRow("  Has critique", "Critique")
	printRow("  Structure score", "Structure")
	fmt.Println()

	fmt.Printf("%-*s\n", col*4+5, "  DEVELOPER CONTROL")
	printRow("  Control score", "ControlScore")
	printRow("  Traceable decisions", "TraceDecisions")
	printRow("  Modify comms", "ModifyComms")
	printRow("  Add agent mid-run", "AddMidRun")
	printRow("  Change exec order", "Reorder")

	fmt.Printf("%s\n\n", sep)
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// runMultiAgentComparisonDemo is the demo entry point (no main()).
func runMultiAgentComparisonDemo(ctx context.Context, client *openai.Client) {
	fmt.Printf("\n%s\n  MULTI-AGENT FRAMEWORK COMPARISON DEMO\n%s\n", strings.Repeat("═", 60), strings.Repeat("═", 60))
	fmt.Printf("\nTask: %s\n\n", maResearchTask)

	var results []MACompResult

	fmt.Print("Running approach 1: From Scratch... ")
	t0 := time.Now()
	r1 := maRunFromScratch(ctx, client)
	fmt.Printf("done (%.2fs)\n", time.Since(t0).Seconds())
	results = append(results, r1)

	fmt.Print("Running approach 2: CrewAI-style... ")
	t0 = time.Now()
	r2 := maRunCrewAIStyle(ctx, client)
	fmt.Printf("done (%.2fs)\n", time.Since(t0).Seconds())
	results = append(results, r2)

	fmt.Print("Running approach 3: AutoGen-style... ")
	t0 = time.Now()
	r3 := maRunAutoGenStyle(ctx, client)
	fmt.Printf("done (%.2fs)\n", time.Since(t0).Seconds())
	results = append(results, r3)

	maPrintTable(results)

	// Print any errors
	for _, r := range results {
		if r.Error != "" {
			fmt.Printf("  [%s] Error: %s\n", r.Name, r.Error)
		}
	}

	// Key takeaways
	fmt.Println("\nKEY TAKEAWAYS:")
	fmt.Println("  From Scratch : Maximum control, most code, easiest to debug.")
	fmt.Println("  CrewAI-style : Balanced — clear roles, moderate control, fast to prototype.")
	fmt.Println("  AutoGen-style: Natural conversation, hardest to control, best for open-ended tasks.")
	fmt.Println()
	fmt.Println("  Choose based on your primary constraint:")
	fmt.Println("    • Need to audit/trace every decision → From Scratch")
	fmt.Println("    • Need role specialisation quickly   → CrewAI-style")
	fmt.Println("    • Need emergent problem-solving      → AutoGen-style")
}
