// langgraph_multi_agent.go — LangGraph-equivalent multi-agent state machine in Go.
//
// LangGraph has no official Go port. This file implements the same CONCEPT —
// a directed state machine where specialised agents run as "nodes" and pass
// a shared state object to the next node — using Go-native code.
//
// Workflow (research → fact_check → writer → editor, with revision loop):
//
//	research_node → fact_check_node → writer_node → editor_node
//	                                       ↑_____________|  (if not approved)
//
// The editor decides whether to approve or request another revision.
// If maxRevisions is exceeded, the workflow terminates regardless.
//
// Run:
//
//	go run langgraph_multi_agent.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go over_engineering_detector.go langsmith_tracer.go \
//	           autogen_design_team.go crewai_research_crew.go
//
// See: docs/02-the-agent-loop/04-multi-agent-patterns.md
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// State types
// ---------------------------------------------------------------------------

// LGMAStep records one step in the workflow trace.
type LGMAStep struct {
	Step          string
	Agent         string
	Summary       string
	OutputPreview string
}

// LGMAState is the shared state passed between nodes.
type LGMAState struct {
	Topic          string
	ResearchNotes  string
	VerifiedFacts  string
	DraftReport    string
	EditorFeedback string
	Approved       bool
	RevisionCount  int
	Sources        []string
	WorkflowTrace  []LGMAStep
}

// LGMAResult is returned after the full workflow completes.
type LGMAResult struct {
	Report        string
	Sources       []string
	Revisions     int
	WorkflowTrace []LGMAStep
	ElapsedMs     float64
}

// ---------------------------------------------------------------------------
// Simulated search tools (same data as crewai_research_crew.go but independent)
// ---------------------------------------------------------------------------

var lgmaMockSearchData = map[string]string{
	"AI productivity": "Researchers at Stanford and MIT found AI coding assistants increase output by 20-55% (2023-2024 studies). GitHub Copilot study: 55% faster. NBER: 56% faster for experienced workers.",
	"developer tools": "Stack Overflow 2024: 77% adoption of AI tools. Most used: GitHub Copilot (34%), Cursor (18%), Codeium (12%).",
	"software output": "McKinsey Global Institute (2024): generative AI could automate 30% of software tasks within 3 years.",
	"impact":          "Key risks: over-reliance, security vulnerabilities in AI-generated code, hallucinations in complex logic.",
}

func lgmaSearch(query string) string {
	lower := strings.ToLower(query)
	for key, val := range lgmaMockSearchData {
		if strings.Contains(lower, key) {
			return val
		}
	}
	return fmt.Sprintf("Search: No specific data found for %q. General knowledge used.", query)
}

// ---------------------------------------------------------------------------
// Node functions
// ---------------------------------------------------------------------------

// lgmaResearchNode gathers raw research notes.
func lgmaResearchNode(ctx context.Context, client *openai.Client, state LGMAState) (LGMAState, error) {
	searchResults := lgmaSearch(state.Topic)
	prompt := fmt.Sprintf(
		"You are a research agent. Topic: %q\n\n"+
			"Available data:\n%s\n\n"+
			"Produce comprehensive research notes (200–300 words). Include:\n"+
			"1. Key statistics with sources.\n"+
			"2. Main trends.\n"+
			"3. At least one limitation or counterpoint.\n"+
			"List 3–5 sources at the end.",
		state.Topic, searchResults)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:     "gpt-4o-mini",
		Messages:  []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: prompt}},
		MaxTokens: 500,
	})
	if err != nil {
		return state, fmt.Errorf("research node: %w", err)
	}

	notes := resp.Choices[0].Message.Content
	state.ResearchNotes = notes

	// Extract sources mentioned in the notes
	for _, line := range strings.Split(notes, "\n") {
		l := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToLower(l), "source") || strings.HasPrefix(l, "-") {
			if len(l) > 5 {
				state.Sources = append(state.Sources, l)
			}
		}
	}

	preview := notes
	if len(preview) > 100 {
		preview = preview[:100] + "…"
	}
	state.WorkflowTrace = append(state.WorkflowTrace, LGMAStep{
		Step:          "research",
		Agent:         "ResearchAgent",
		Summary:       fmt.Sprintf("Gathered %d chars of research notes", len(notes)),
		OutputPreview: preview,
	})
	return state, nil
}

// lgmaFactCheckNode verifies claims in the research notes.
func lgmaFactCheckNode(ctx context.Context, client *openai.Client, state LGMAState) (LGMAState, error) {
	prompt := fmt.Sprintf(
		"You are a fact-checking agent. Review these research notes:\n\n%s\n\n"+
			"1. Mark each quantitative claim as VERIFIED, PLAUSIBLE, or UNVERIFIED.\n"+
			"2. Remove or flag any claim you cannot verify.\n"+
			"3. Produce a 'Verified Facts' document with only reliable information.\n"+
			"Be conservative — if in doubt, mark as PLAUSIBLE.",
		state.ResearchNotes)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:     "gpt-4o-mini",
		Messages:  []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: prompt}},
		MaxTokens: 500,
	})
	if err != nil {
		return state, fmt.Errorf("fact-check node: %w", err)
	}

	verified := resp.Choices[0].Message.Content
	state.VerifiedFacts = verified

	preview := verified
	if len(preview) > 100 {
		preview = preview[:100] + "…"
	}
	state.WorkflowTrace = append(state.WorkflowTrace, LGMAStep{
		Step:          "fact_check",
		Agent:         "FactCheckAgent",
		Summary:       "Verified claims in research notes",
		OutputPreview: preview,
	})
	return state, nil
}

// lgmaWriterNode drafts the report from verified facts.
func lgmaWriterNode(ctx context.Context, client *openai.Client, state LGMAState) (LGMAState, error) {
	prevFeedback := ""
	if state.EditorFeedback != "" {
		prevFeedback = fmt.Sprintf("\n\nEditor feedback from previous draft:\n%s\n\nPlease address all points above.", state.EditorFeedback)
	}

	prompt := fmt.Sprintf(
		"You are a technical writer. Write a 300–400-word report on: %q\n\n"+
			"Use only these verified facts:\n%s%s\n\n"+
			"Structure: executive summary → key findings → risks → recommendations.\n"+
			"Include inline citations (Author/Org, Year).",
		state.Topic, state.VerifiedFacts, prevFeedback)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:     "gpt-4o-mini",
		Messages:  []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: prompt}},
		MaxTokens: 700,
	})
	if err != nil {
		return state, fmt.Errorf("writer node: %w", err)
	}

	draft := resp.Choices[0].Message.Content
	state.DraftReport = draft

	preview := draft
	if len(preview) > 100 {
		preview = preview[:100] + "…"
	}
	label := "Draft"
	if state.RevisionCount > 0 {
		label = fmt.Sprintf("Revision %d", state.RevisionCount)
	}
	state.WorkflowTrace = append(state.WorkflowTrace, LGMAStep{
		Step:          "writer",
		Agent:         "WriterAgent",
		Summary:       fmt.Sprintf("%s written (%d chars)", label, len(draft)),
		OutputPreview: preview,
	})
	return state, nil
}

// lgmaEditorNode reviews the draft and either approves it or requests revisions.
func lgmaEditorNode(ctx context.Context, client *openai.Client, state LGMAState) (LGMAState, error) {
	prompt := fmt.Sprintf(
		"You are a senior editor. Review this draft report:\n\n%s\n\n"+
			"If the report is publication-ready (accurate, well-structured, ≥300 words, has citations), "+
			"respond with exactly: APPROVED\n\n"+
			"Otherwise, provide specific, actionable feedback on what must be improved. "+
			"Be strict — the report must cite sources and have clear structure.",
		state.DraftReport)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:     "gpt-4o-mini",
		Messages:  []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: prompt}},
		MaxTokens: 300,
	})
	if err != nil {
		return state, fmt.Errorf("editor node: %w", err)
	}

	feedback := resp.Choices[0].Message.Content
	state.EditorFeedback = feedback
	state.Approved = strings.Contains(strings.ToUpper(feedback), "APPROVED")
	state.RevisionCount++

	summary := "Requested revisions"
	if state.Approved {
		summary = "APPROVED"
	}
	preview := feedback
	if len(preview) > 100 {
		preview = preview[:100] + "…"
	}
	state.WorkflowTrace = append(state.WorkflowTrace, LGMAStep{
		Step:          "editor",
		Agent:         "EditorAgent",
		Summary:       summary,
		OutputPreview: preview,
	})
	return state, nil
}

// ---------------------------------------------------------------------------
// Workflow runner
// ---------------------------------------------------------------------------

// runLGMAWorkflow executes the full research → fact_check → writer ⇄ editor loop.
func runLGMAWorkflow(ctx context.Context, client *openai.Client, topic string, maxRevisions int) (LGMAResult, error) {
	t0 := time.Now()

	state := LGMAState{Topic: topic}

	// Phase 1: Research
	var err error
	state, err = lgmaResearchNode(ctx, client, state)
	if err != nil {
		return LGMAResult{}, err
	}

	// Phase 2: Fact-check
	state, err = lgmaFactCheckNode(ctx, client, state)
	if err != nil {
		return LGMAResult{}, err
	}

	// Phase 3+4: Write ↔ Edit (revision loop)
	for i := 0; i <= maxRevisions; i++ {
		state, err = lgmaWriterNode(ctx, client, state)
		if err != nil {
			return LGMAResult{}, err
		}
		state, err = lgmaEditorNode(ctx, client, state)
		if err != nil {
			return LGMAResult{}, err
		}
		if state.Approved {
			break
		}
	}

	return LGMAResult{
		Report:        state.DraftReport,
		Sources:       state.Sources,
		Revisions:     state.RevisionCount - 1, // first pass is not a revision
		WorkflowTrace: state.WorkflowTrace,
		ElapsedMs:     float64(time.Since(t0).Milliseconds()),
	}, nil
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

// lgmaPrintWorkflowTrace prints the workflow steps in a compact table.
func lgmaPrintWorkflowTrace(trace []LGMAStep) {
	fmt.Printf("\n%s\n  WORKFLOW TRACE\n%s\n", strings.Repeat("─", 70), strings.Repeat("─", 70))
	for i, step := range trace {
		fmt.Printf("  [%02d] %-12s  %-18s  %s\n", i+1, step.Step, step.Agent, step.Summary)
		if step.OutputPreview != "" {
			fmt.Printf("       → %s\n", step.OutputPreview)
		}
	}
	fmt.Printf("%s\n", strings.Repeat("─", 70))
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// runLangGraphMultiAgentDemo is the demo entry point (no main()).
func runLangGraphMultiAgentDemo(ctx context.Context, client *openai.Client) {
	topics := []string{
		"The impact of AI on software developer productivity",
	}

	for i, topic := range topics {
		fmt.Printf("\n%s\n  LANGGRAPH MULTI-AGENT WORKFLOW — TOPIC %d\n%s\n", strings.Repeat("═", 60), i+1, strings.Repeat("═", 60))
		fmt.Printf("\nTopic: %s\n", topic)
		fmt.Println("Workflow: research → fact_check → writer ⇄ editor")
		fmt.Printf("Max revisions: 2\n\n")

		result, err := runLGMAWorkflow(ctx, client, topic, 2)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		lgmaPrintWorkflowTrace(result.WorkflowTrace)

		fmt.Printf("\nRevisions requested: %d\n", result.Revisions)
		fmt.Printf("Total workflow time : %.0fms\n", result.ElapsedMs)
		fmt.Printf("Sources found       : %d\n", len(result.Sources))

		fmt.Printf("\nFinal Report (preview):\n")
		report := result.Report
		if len(report) > 500 {
			report = report[:500] + "\n…(truncated)"
		}
		fmt.Printf("%s\n", report)
	}
}
