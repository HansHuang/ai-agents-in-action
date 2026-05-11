// strategy_comparison.go — Run the same task through all three planning strategies (Go port).
//
// Strategies compared:
//  1. ReAct — plan-while-executing
//  2. Plan-and-Execute — structured plan first, then execution
//  3. Reflection — generate + self-critique + revise
//
// See docs/02-the-agent-loop/03-planning-strategies.md
package main

import (
	"context"
	"fmt"
	"time"
)

// ---------------------------------------------------------------------------
// Call metrics
// ---------------------------------------------------------------------------

// CallMetrics accumulates metrics for a single strategy run.
type CallMetrics struct {
	LLMCalls         int
	PromptTokens     int
	CompletionTokens int
}

// TotalTokens returns the sum of prompt and completion tokens.
func (m *CallMetrics) TotalTokens() int { return m.PromptTokens + m.CompletionTokens }

// EstimatedCostUSD estimates the USD cost at gpt-4o pricing.
func (m *CallMetrics) EstimatedCostUSD() float64 {
	const promptPer1K = 0.005
	const completionPer1K = 0.015
	return float64(m.PromptTokens)/1000*promptPer1K + float64(m.CompletionTokens)/1000*completionPer1K
}

// ---------------------------------------------------------------------------
// Strategy result
// ---------------------------------------------------------------------------

// StrategyResult holds the output and metrics for one strategy.
type StrategyResult struct {
	Name       string
	Answer     string
	ElapsedSec float64
	Metrics    CallMetrics
	Err        error
}

// ---------------------------------------------------------------------------
// Comparison runner
// ---------------------------------------------------------------------------

// CompareStrategies runs the same question through all three strategies.
func CompareStrategies(ctx context.Context, question string) []StrategyResult {
	strategies := []struct {
		name string
		run  func(ctx context.Context, q string) (string, error)
	}{
		{
			name: "ReAct",
			run: func(ctx context.Context, q string) (string, error) {
				registry := ToolRegistry{
					"get_weather": func(args map[string]interface{}) (interface{}, error) {
						city, _ := args["city"].(string)
						return GetWeather(city), nil
					},
					"get_stock_price": func(args map[string]interface{}) (interface{}, error) {
						ticker, _ := args["ticker"].(string)
						return GetStockPrice(ticker), nil
					},
				}
				answer, _, err := RunAgent(ctx, q, nil, registry)
				return answer, err
			},
		},
		{
			name: "Plan-and-Execute",
			run: func(ctx context.Context, q string) (string, error) {
				agent := NewPlanAndExecuteAgent("gpt-4o", 20)
				output, err := agent.Run(ctx, q)
				if err != nil {
					return "", err
				}
				return output.Answer, nil
			},
		},
		{
			name: "Reflection",
			run: func(ctx context.Context, q string) (string, error) {
				agent := NewReflectionAgent()
				answer, _ := agent.Run(ctx, q)
				return answer, nil
			},
		},
	}

	results := make([]StrategyResult, 0, len(strategies))
	for _, s := range strategies {
		fmt.Printf("\n[Strategy] Running: %s\n", s.name)
		start := time.Now()
		answer, err := s.run(ctx, question)
		elapsed := time.Since(start).Seconds()

		results = append(results, StrategyResult{
			Name:       s.name,
			Answer:     answer,
			ElapsedSec: elapsed,
			Err:        err,
		})
	}
	return results
}

// PrintComparisonReport prints a formatted comparison of strategy results.
func PrintComparisonReport(question string, results []StrategyResult) {
	fmt.Printf("\n%s\n", repeatStr("=", 70))
	fmt.Printf("Strategy Comparison: %s\n", question)
	fmt.Println(repeatStr("=", 70))
	fmt.Printf("%-22s %-10s %-14s %s\n", "Strategy", "Elapsed(s)", "Answer(chars)", "Status")
	fmt.Println(repeatStr("-", 70))

	for _, r := range results {
		status := "OK"
		if r.Err != nil {
			status = "ERROR: " + r.Err.Error()
		}
		answerLen := len(r.Answer)
		fmt.Printf("%-22s %-10.2f %-14d %s\n", r.Name, r.ElapsedSec, answerLen, status)
	}

	fmt.Println("\n--- Answers ---")
	for _, r := range results {
		fmt.Printf("\n[%s]\n", r.Name)
		preview := r.Answer
		if len(preview) > 300 {
			preview = preview[:300] + "…"
		}
		fmt.Println(preview)
	}
}

// RunStrategyComparison demonstrates all three strategies on a sample question.
func RunStrategyComparison() {
	question := "What is the weather in Shanghai and should I invest in Tesla today?"
	fmt.Printf("Question: %s\n", question)

	results := CompareStrategies(context.Background(), question)
	PrintComparisonReport(question, results)
}
