// langgraph_react_agent.go — LangGraph ReAct agent vs from-scratch comparison in Go.
//
// langgraph_alternative.go already implements the Go-native LangGraph-style
// state machine (GoNativeReActAgent, StateGraph, visualize, etc.).
//
// This file adds:
//   - A lightweight "from-scratch" ReAct agent for comparison
//   - A side-by-side comparison runner (lgracRunComparison)
//   - A formatted comparison report (lgracPrintComparison)
//
// The comparison measures: tool calls, iterations, answer quality, and latency.
//
// Run:
//
//	go run langgraph_react_agent.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go over_engineering_detector.go langsmith_tracer.go \
//	           autogen_design_team.go crewai_research_crew.go langgraph_multi_agent.go
//
// See: docs/02-the-agent-loop/01-anatomy-of-an-agent.md
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

// LGRACResult holds the side-by-side comparison of LangGraph vs from-scratch.
type LGRACResult struct {
	Query               string
	LangGraphAnswer     string
	ScratchAnswer       string
	LangGraphIterations int
	ScratchIterations   int
	LangGraphToolCalls  []string
	ScratchToolCalls    []string
	LangGraphElapsedMs  float64
	ScratchElapsedMs    float64
	TraceIdentical      bool
}

// ---------------------------------------------------------------------------
// From-scratch ReAct agent
// ---------------------------------------------------------------------------

// lgracScratchTools mirrors the tool definitions from langgraph_alternative.go
// (getWeather, getStockPrice, calculator) but uses local, self-contained dispatch.

// lgracDispatchTool dispatches a tool call to the matching function.
func lgracDispatchTool(toolName, argsJSON string) string {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "(invalid tool args)"
	}

	getString := func(key string) string {
		if v, ok := args[key]; ok {
			return fmt.Sprintf("%v", v)
		}
		return ""
	}

	switch toolName {
	case "get_weather":
		return getWeather(getString("city"))
	case "get_stock_price":
		return getStockPrice(getString("ticker"))
	case "calculator":
		return calculator(getString("expression"))
	default:
		return fmt.Sprintf("(unknown tool: %s)", toolName)
	}
}

// lgracRunScratch runs a minimal from-scratch ReAct agent.
// It shares the same tool functions as GoNativeReActAgent but uses a simpler
// loop without a StateGraph — demonstrating that both approaches are equivalent.
func lgracRunScratch(ctx context.Context, client *openai.Client, query string) (string, []string, int, float64, error) {
	t0 := time.Now()
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
		{Role: openai.ChatMessageRoleUser, Content: query},
	}

	var toolCallNames []string
	maxIter := reactMaxIter
	answer := ""

	for iter := 0; iter < maxIter; iter++ {
		resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    reactLLMModel,
			Messages: messages,
			Tools:    tools,
		})
		if err != nil {
			return "", toolCallNames, iter, float64(time.Since(t0).Milliseconds()), err
		}
		msg := resp.Choices[0].Message

		// Final answer — no tool calls
		if len(msg.ToolCalls) == 0 {
			answer = msg.Content
			break
		}

		messages = append(messages, msg)
		for _, tc := range msg.ToolCalls {
			toolCallNames = append(toolCallNames, tc.Function.Name)
			result := lgracDispatchTool(tc.Function.Name, tc.Function.Arguments)
			messages = append(messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleUser,
				Content:    result,
				Name:       tc.Function.Name,
				ToolCallID: tc.ID,
			})
		}
	}

	if answer == "" {
		answer = "(reached max iterations without a final answer)"
	}
	return answer, toolCallNames, len(toolCallNames), float64(time.Since(t0).Milliseconds()), nil
}

// ---------------------------------------------------------------------------
// Comparison runner
// ---------------------------------------------------------------------------

// lgracRunComparison runs both agents against the same query and returns a diff.
func lgracRunComparison(ctx context.Context, client *openai.Client, query string) (LGRACResult, error) {
	result := LGRACResult{Query: query}

	// --- LangGraph-style agent (GoNativeReActAgent from langgraph_alternative.go) ---
	lgAgent := NewGoNativeReActAgent(10)
	t0 := time.Now()
	lgResult, err := lgAgent.Run(ctx, query)
	if err != nil {
		return result, fmt.Errorf("LangGraph agent: %w", err)
	}
	result.LangGraphAnswer = lgResult.Answer
	result.LangGraphIterations = lgResult.Iterations
	result.LangGraphToolCalls = lgResult.ToolCallsMade
	result.LangGraphElapsedMs = float64(time.Since(t0).Milliseconds())

	// --- From-scratch agent ---
	answer, toolCalls, iters, elapsed, err := lgracRunScratch(ctx, client, query)
	if err != nil {
		return result, fmt.Errorf("scratch agent: %w", err)
	}
	result.ScratchAnswer = answer
	result.ScratchToolCalls = toolCalls
	result.ScratchIterations = iters
	result.ScratchElapsedMs = elapsed

	// Are the tool-call traces identical?
	result.TraceIdentical = strings.Join(result.LangGraphToolCalls, ",") == strings.Join(result.ScratchToolCalls, ",")

	return result, nil
}

// ---------------------------------------------------------------------------
// Report printer
// ---------------------------------------------------------------------------

// lgracPrintComparison prints the comparison table to stdout.
func lgracPrintComparison(r LGRACResult) {
	col := 26
	sep := strings.Repeat("═", col*2+5)
	fmt.Printf("\n%s\n", sep)
	fmt.Printf("  REACT AGENT COMPARISON\n")
	fmt.Printf("  Query: %q\n", r.Query)
	fmt.Printf("%s\n", sep)
	fmt.Printf("%-*s  %-*s  %-*s\n", col, "Metric", col, "LangGraph-style", col, "From Scratch")
	fmt.Printf("%s\n", strings.Repeat("─", col*2+5))

	formatList := func(items []string) string {
		if len(items) == 0 {
			return "(none)"
		}
		return strings.Join(items, ", ")
	}

	rows := []struct{ label, a, b string }{
		{"Iterations", fmt.Sprintf("%d", r.LangGraphIterations), fmt.Sprintf("%d", r.ScratchIterations)},
		{"Tool calls (count)", fmt.Sprintf("%d", len(r.LangGraphToolCalls)), fmt.Sprintf("%d", len(r.ScratchToolCalls))},
		{"Tools used", formatList(r.LangGraphToolCalls), formatList(r.ScratchToolCalls)},
		{"Elapsed (ms)", fmt.Sprintf("%.0f", r.LangGraphElapsedMs), fmt.Sprintf("%.0f", r.ScratchElapsedMs)},
	}
	for _, row := range rows {
		fmt.Printf("%-*s  %-*s  %-*s\n", col, row.label, col, row.a, col, row.b)
	}
	fmt.Printf("%s\n", strings.Repeat("─", col*2+5))

	traceStr := "yes"
	if !r.TraceIdentical {
		traceStr = "no (different tool sequences)"
	}
	fmt.Printf("%-*s  %s\n", col, "Tool traces identical?", traceStr)
	fmt.Printf("%s\n", sep)

	fmt.Printf("\n  LangGraph-style answer:\n    %s\n", truncate(r.LangGraphAnswer, 200))
	fmt.Printf("\n  From-scratch answer:\n    %s\n", truncate(r.ScratchAnswer, 200))

	fmt.Printf("\n%s\n", sep)
	fmt.Println("  KEY INSIGHT:")
	fmt.Println("  Both agents use the same tools and system prompt.")
	fmt.Println("  The state machine (LangGraph-style) adds explicit node boundaries,")
	fmt.Println("  making flow control visible and auditable — but costs a few lines of")
	fmt.Println("  boilerplate. For simple tools, a direct loop is equivalent.")
}

// truncate returns s[:n]+"…" if s is longer than n, else s.
func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// runLangGraphReActAgentDemo is the demo entry point (no main()).
func runLangGraphReActAgentDemo(ctx context.Context, client *openai.Client) {
	fmt.Printf("\n%s\n  LANGGRAPH REACT AGENT DEMO\n%s\n", strings.Repeat("═", 60), strings.Repeat("═", 60))

	// Visualise the state graph from langgraph_alternative.go
	fmt.Println("\nState graph (LangGraph-style):")
	visualize()

	queries := []string{
		"What's the weather in Tokyo and the price of NVDA stock?",
		"Calculate 15 * 37 and get the weather in London.",
	}

	for _, query := range queries {
		result, err := lgracRunComparison(ctx, client, query)
		if err != nil {
			fmt.Printf("\nQuery: %q\nError: %v\n", query, err)
			continue
		}
		lgracPrintComparison(result)
	}
}
