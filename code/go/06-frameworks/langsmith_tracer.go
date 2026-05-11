// langsmith_tracer.go — Local in-memory trace store (LangSmith-equivalent).
//
// LangSmith is a LangChain observability platform with no official Go SDK.
// This file demonstrates the same CONCEPTS — trace IDs, LLM call spans,
// tool call spans, feedback scores, and trace comparison — using a simple
// in-memory store with no external dependencies.
//
// In production, emit structured JSON logs and ship to your preferred
// observability backend (Datadog, Honeycomb, OpenTelemetry, etc.).
//
// Run:
//
//	go run langsmith_tracer.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go over_engineering_detector.go
//
// See: docs/05-the-tool-ecosystem/03-agent-observability.md
package main

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Trace event types
// ---------------------------------------------------------------------------

// LSTraceEvent is a single observation in a trace.
type LSTraceEvent struct {
	TraceID   string
	EventType string // "start" | "llm_call" | "tool_call" | "end" | "feedback"
	Data      map[string]interface{}
	Timestamp string
}

// ---------------------------------------------------------------------------
// Local trace store
// ---------------------------------------------------------------------------

// LSLocalStore is a thread-safe in-memory trace store.
type LSLocalStore struct {
	mu     sync.RWMutex
	traces map[string][]LSTraceEvent
}

func newLSLocalStore() *LSLocalStore {
	return &LSLocalStore{traces: make(map[string][]LSTraceEvent)}
}

func (s *LSLocalStore) append(event LSTraceEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.traces[event.TraceID] = append(s.traces[event.TraceID], event)
}

func (s *LSLocalStore) get(traceID string) []LSTraceEvent {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.traces[traceID]
}

func (s *LSLocalStore) ids() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := make([]string, 0, len(s.traces))
	for id := range s.traces {
		ids = append(ids, id)
	}
	return ids
}

// ---------------------------------------------------------------------------
// LSTracer — the public API
// ---------------------------------------------------------------------------

// LSTracer provides LangSmith-equivalent tracing using local storage.
// In Go there is no LangSmith SDK, so all tracing is local and all
// concepts are implemented from scratch.
type LSTracer struct {
	ProjectName string
	mode        string // always "local" in Go
	store       *LSLocalStore
}

// NewLSTracer creates a new local tracer for the given project.
func NewLSTracer(projectName string) *LSTracer {
	return &LSTracer{
		ProjectName: projectName,
		mode:        "local",
		store:       newLSLocalStore(),
	}
}

// lsRandomID generates a short random trace ID.
func lsRandomID() string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 12)
	for i := range b {
		b[i] = chars[rand.Intn(len(chars))]
	}
	return string(b)
}

// now returns the current time in RFC3339 format.
func lsNow() string {
	return time.Now().UTC().Format(time.RFC3339)
}

// TraceAgentRun starts a new trace and returns its ID.
func (t *LSTracer) TraceAgentRun(agentName, userInput string) string {
	traceID := lsRandomID()
	t.store.append(LSTraceEvent{
		TraceID:   traceID,
		EventType: "start",
		Data: map[string]interface{}{
			"project":    t.ProjectName,
			"agent_name": agentName,
			"user_input": userInput,
			"mode":       t.mode,
		},
		Timestamp: lsNow(),
	})
	return traceID
}

// LogLLMCall records a single LLM call span inside a trace.
func (t *LSTracer) LogLLMCall(traceID, model string, msgCount, tokensUsed int, latencyMs float64) {
	t.store.append(LSTraceEvent{
		TraceID:   traceID,
		EventType: "llm_call",
		Data: map[string]interface{}{
			"model":       model,
			"msg_count":   msgCount,
			"tokens_used": tokensUsed,
			"latency_ms":  latencyMs,
		},
		Timestamp: lsNow(),
	})
}

// LogToolCall records a tool invocation span inside a trace.
func (t *LSTracer) LogToolCall(traceID, toolName string, inputParams map[string]string, outputResult string, latencyMs float64) {
	t.store.append(LSTraceEvent{
		TraceID:   traceID,
		EventType: "tool_call",
		Data: map[string]interface{}{
			"tool_name":     toolName,
			"input_params":  inputParams,
			"output_result": outputResult,
			"latency_ms":    latencyMs,
		},
		Timestamp: lsNow(),
	})
}

// EndTrace marks a trace as complete with the final answer.
func (t *LSTracer) EndTrace(traceID, answer string, traceErr string) {
	data := map[string]interface{}{
		"answer": answer,
	}
	if traceErr != "" {
		data["error"] = traceErr
	}
	t.store.append(LSTraceEvent{
		TraceID:   traceID,
		EventType: "end",
		Data:      data,
		Timestamp: lsNow(),
	})
}

// LogFeedback attaches a human (or automated) quality score to a trace.
func (t *LSTracer) LogFeedback(traceID string, score float64, comment string) error {
	events := t.store.get(traceID)
	if len(events) == 0 {
		return fmt.Errorf("trace %q not found", traceID)
	}
	if score < 0 || score > 1 {
		return fmt.Errorf("score must be in [0, 1], got %.2f", score)
	}
	t.store.append(LSTraceEvent{
		TraceID:   traceID,
		EventType: "feedback",
		Data: map[string]interface{}{
			"score":   score,
			"comment": comment,
		},
		Timestamp: lsNow(),
	})
	return nil
}

// GetTraceSummary returns a summary of the trace as a map.
func (t *LSTracer) GetTraceSummary(traceID string) map[string]interface{} {
	events := t.store.get(traceID)
	if len(events) == 0 {
		return map[string]interface{}{"error": "trace not found"}
	}

	summary := map[string]interface{}{
		"trace_id":    traceID,
		"project":     t.ProjectName,
		"event_count": len(events),
	}

	var llmCalls, toolCalls, totalTokens int
	var totalLLMLatency, totalToolLatency float64
	var feedbackScore float64
	hasFeedback := false

	for _, ev := range events {
		switch ev.EventType {
		case "start":
			summary["agent_name"] = ev.Data["agent_name"]
			summary["user_input"] = ev.Data["user_input"]
			summary["started_at"] = ev.Timestamp
		case "llm_call":
			llmCalls++
			if v, ok := ev.Data["tokens_used"].(int); ok {
				totalTokens += v
			}
			if v, ok := ev.Data["latency_ms"].(float64); ok {
				totalLLMLatency += v
			}
		case "tool_call":
			toolCalls++
			if v, ok := ev.Data["latency_ms"].(float64); ok {
				totalToolLatency += v
			}
		case "end":
			summary["answer"] = ev.Data["answer"]
			summary["ended_at"] = ev.Timestamp
			if errVal, ok := ev.Data["error"].(string); ok && errVal != "" {
				summary["error"] = errVal
			}
		case "feedback":
			if v, ok := ev.Data["score"].(float64); ok {
				feedbackScore = v
				hasFeedback = true
			}
		}
	}

	summary["llm_calls"] = llmCalls
	summary["tool_calls"] = toolCalls
	summary["total_tokens"] = totalTokens
	summary["total_llm_latency_ms"] = totalLLMLatency
	summary["total_tool_latency_ms"] = totalToolLatency
	if hasFeedback {
		summary["feedback_score"] = feedbackScore
	}
	return summary
}

// CompareTraces compares two traces and returns a human-readable summary.
func (t *LSTracer) CompareTraces(traceIDA, traceIDB string) string {
	a := t.GetTraceSummary(traceIDA)
	b := t.GetTraceSummary(traceIDB)

	col := 20
	sep := strings.Repeat("─", col*3+4)

	var sb strings.Builder
	sb.WriteString("TRACE COMPARISON\n")
	sb.WriteString(sep + "\n")
	sb.WriteString(fmt.Sprintf("%-*s %-*s %-*s\n", col, "Metric", col, "Trace A", col, "Trace B"))
	sb.WriteString(sep + "\n")

	formatVal := func(m map[string]interface{}, key string) string {
		v, ok := m[key]
		if !ok {
			return "N/A"
		}
		switch x := v.(type) {
		case float64:
			return fmt.Sprintf("%.0f", x)
		case int:
			return fmt.Sprintf("%d", x)
		default:
			return fmt.Sprintf("%v", v)
		}
	}

	metrics := []string{"llm_calls", "tool_calls", "total_tokens", "total_llm_latency_ms", "total_tool_latency_ms"}
	for _, m := range metrics {
		sb.WriteString(fmt.Sprintf("%-*s %-*s %-*s\n", col, m, col, formatVal(a, m), col, formatVal(b, m)))
	}
	sb.WriteString(sep + "\n")

	if fa, ok := a["feedback_score"]; ok {
		if fb, ok := b["feedback_score"]; ok {
			sb.WriteString(fmt.Sprintf("%-*s %-*s %-*s\n", col, "feedback_score", col, fmt.Sprintf("%.2f", fa), col, fmt.Sprintf("%.2f", fb)))
		}
	}
	return sb.String()
}

// PrintTrace prints a detailed trace timeline to stdout.
func (t *LSTracer) PrintTrace(traceID string) {
	events := t.store.get(traceID)
	if len(events) == 0 {
		fmt.Printf("Trace %q not found.\n", traceID)
		return
	}
	fmt.Printf("\nTRACE: %s  project=%s\n", traceID, t.ProjectName)
	fmt.Printf("%s\n", strings.Repeat("─", 60))
	for i, ev := range events {
		fmt.Printf("  [%02d] %-12s  %s\n", i+1, ev.EventType, ev.Timestamp)
		for k, v := range ev.Data {
			val := fmt.Sprintf("%v", v)
			if len(val) > 80 {
				val = val[:77] + "..."
			}
			fmt.Printf("       %-18s = %s\n", k, val)
		}
	}
	fmt.Printf("%s\n", strings.Repeat("─", 60))
}

// ---------------------------------------------------------------------------
// Demo: simulate a RAG agent run with tracing
// ---------------------------------------------------------------------------

// lsSimulateRAGAgentRun runs a single traced RAG query.
func lsSimulateRAGAgentRun(ctx context.Context, client *openai.Client, tracer *LSTracer, query string) (string, string, error) {
	traceID := tracer.TraceAgentRun("RAGAgent", query)

	t0 := time.Now()
	answer := "(no client)"

	if client != nil {
		resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model: "gpt-4o-mini",
			Messages: []openai.ChatCompletionMessage{
				{Role: openai.ChatMessageRoleSystem, Content: "Answer the question concisely using your knowledge about AI agents."},
				{Role: openai.ChatMessageRoleUser, Content: query},
			},
			MaxTokens: 150,
		})
		elapsed := float64(time.Since(t0).Milliseconds())
		if err != nil {
			tracer.EndTrace(traceID, "", err.Error())
			return traceID, "", err
		}
		tokens := resp.Usage.TotalTokens
		answer = resp.Choices[0].Message.Content
		tracer.LogLLMCall(traceID, "gpt-4o-mini", 2, tokens, elapsed)
	} else {
		// Simulated (no real client)
		time.Sleep(80 * time.Millisecond)
		tracer.LogLLMCall(traceID, "gpt-4o-mini (sim)", 2, 420, 80)
		answer = "(simulated answer) An agent loop iterates: perceive → think → act → observe."
	}

	// Simulate a tool call in the trace
	toolT0 := time.Now()
	_ = toolT0
	tracer.LogToolCall(traceID, "retrieve_docs", map[string]string{"query": query}, "3 docs retrieved", 22)

	tracer.EndTrace(traceID, answer, "")
	return traceID, answer, nil
}

// runLangSmithTracerDemo is the demo entry point (no main()).
func runLangSmithTracerDemo(ctx context.Context, client *openai.Client) {
	fmt.Println("\nLangSmith Tracer Demo")
	fmt.Println("(No LangSmith SDK in Go — uses local in-memory store with identical concepts)")
	fmt.Printf("%s\n", strings.Repeat("═", 60))

	tracer := NewLSTracer("ai-agents-in-action")

	// Run two traced queries
	queries := []string{
		"What is the agent loop?",
		"How does RAG retrieval work?",
	}

	var traceIDs []string
	for _, q := range queries {
		fmt.Printf("\nQuery: %q\n", q)
		traceID, answer, err := lsSimulateRAGAgentRun(ctx, client, tracer, q)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		traceIDs = append(traceIDs, traceID)
		preview := answer
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		fmt.Printf("Answer: %s\n", preview)

		// Attach synthetic feedback
		if err := tracer.LogFeedback(traceID, 0.9, "Correct and concise."); err != nil {
			fmt.Printf("Feedback error: %v\n", err)
		}
	}

	// Print individual traces
	for _, id := range traceIDs {
		tracer.PrintTrace(id)
	}

	// Compare the two traces
	if len(traceIDs) >= 2 {
		fmt.Println("\n" + tracer.CompareTraces(traceIDs[0], traceIDs[1]))
	}

	// Summarize all traces
	fmt.Printf("\nAll trace IDs stored: %v\n", tracer.store.ids())
	fmt.Println("\nKey concepts demonstrated:")
	fmt.Println("  • TraceAgentRun  → start a trace with a unique ID")
	fmt.Println("  • LogLLMCall     → record model, token count, latency")
	fmt.Println("  • LogToolCall    → record tool name, inputs, output, latency")
	fmt.Println("  • EndTrace       → close the trace with the final answer")
	fmt.Println("  • LogFeedback    → attach a human quality score (0–1)")
	fmt.Println("  • CompareTraces  → side-by-side metric diff between two runs")
}
