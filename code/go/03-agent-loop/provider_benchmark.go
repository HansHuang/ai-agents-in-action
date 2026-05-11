// provider_benchmark.go — LLM provider benchmarking (Go port).
//
// Measures and compares LLM providers across latency, cost, and capability.
// Produces a formatted comparison report with a recommendation.
//
// See: docs/05-the-tool-ecosystem/01-model-providers.md
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// ---------------------------------------------------------------------------
// Pricing table
// ---------------------------------------------------------------------------

// ModelPricing holds per-1K-token USD pricing for a known model fragment.
type ModelPricing struct {
	Input  float64
	Output float64
}

// KnownPricing maps model name fragments to pricing.
var KnownPricing = map[string]ModelPricing{
	"gpt-4o":            {0.0025, 0.010},
	"gpt-4o-mini":       {0.00015, 0.0006},
	"gpt-3.5-turbo":     {0.0005, 0.0015},
	"claude-3-5-sonnet": {0.003, 0.015},
	"claude-3-haiku":    {0.00025, 0.00125},
}

// LookupPricing finds pricing for a model name (case-insensitive fragment match).
func LookupPricing(modelName string) ModelPricing {
	lower := strings.ToLower(modelName)
	for fragment, pricing := range KnownPricing {
		if strings.Contains(lower, fragment) {
			return pricing
		}
	}
	return ModelPricing{}
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

// ProviderBenchmarkResult holds all metrics for one provider-model run.
type ProviderBenchmarkResult struct {
	ModelName        string
	LatenciesMs      []float64
	PromptTokens     []int
	CompletionTokens []int
	Errors           int
}

// AvgLatencyMs returns the average latency in milliseconds.
func (r *ProviderBenchmarkResult) AvgLatencyMs() float64 {
	if len(r.LatenciesMs) == 0 {
		return 0
	}
	var sum float64
	for _, v := range r.LatenciesMs {
		sum += v
	}
	return sum / float64(len(r.LatenciesMs))
}

// P95LatencyMs returns the 95th-percentile latency.
func (r *ProviderBenchmarkResult) P95LatencyMs() float64 {
	if len(r.LatenciesMs) == 0 {
		return 0
	}
	sorted := make([]float64, len(r.LatenciesMs))
	copy(sorted, r.LatenciesMs)
	sort.Float64s(sorted)
	idx := int(math.Ceil(0.95*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	return sorted[idx]
}

// StdDevLatencyMs returns the standard deviation of latencies.
func (r *ProviderBenchmarkResult) StdDevLatencyMs() float64 {
	if len(r.LatenciesMs) < 2 {
		return 0
	}
	avg := r.AvgLatencyMs()
	var variance float64
	for _, v := range r.LatenciesMs {
		diff := v - avg
		variance += diff * diff
	}
	variance /= float64(len(r.LatenciesMs) - 1)
	return math.Sqrt(variance)
}

// TotalPromptTokens returns the sum of prompt tokens.
func (r *ProviderBenchmarkResult) TotalPromptTokens() int {
	var s int
	for _, v := range r.PromptTokens {
		s += v
	}
	return s
}

// TotalCompletionTokens returns the sum of completion tokens.
func (r *ProviderBenchmarkResult) TotalCompletionTokens() int {
	var s int
	for _, v := range r.CompletionTokens {
		s += v
	}
	return s
}

// EstimatedCostUSD returns the estimated total cost in USD.
func (r *ProviderBenchmarkResult) EstimatedCostUSD() float64 {
	pricing := LookupPricing(r.ModelName)
	return float64(r.TotalPromptTokens())/1000*pricing.Input +
		float64(r.TotalCompletionTokens())/1000*pricing.Output
}

// ---------------------------------------------------------------------------
// Benchmarker
// ---------------------------------------------------------------------------

// ProviderBenchmark runs a set of models on the same prompt and collects metrics.
type ProviderBenchmark struct {
	client *openai.Client
}

// NewProviderBenchmark creates a new ProviderBenchmark.
func NewProviderBenchmark() *ProviderBenchmark {
	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	return &ProviderBenchmark{client: &client}
}

// RunOne benchmarks a single model with the given messages and number of iterations.
func (b *ProviderBenchmark) RunOne(ctx context.Context, model string, messages []map[string]interface{}, iterations int) *ProviderBenchmarkResult {
	result := &ProviderBenchmarkResult{ModelName: model}

	// Convert generic messages to openai params
	var oaiMsgs []openai.ChatCompletionMessageParamUnion
	for _, m := range messages {
		role, _ := m["role"].(string)
		content, _ := m["content"].(string)
		switch role {
		case "system":
			oaiMsgs = append(oaiMsgs, openai.SystemMessage(content))
		case "user":
			oaiMsgs = append(oaiMsgs, openai.UserMessage(content))
		default:
			oaiMsgs = append(oaiMsgs, openai.UserMessage(content))
		}
	}

	for i := 0; i < iterations; i++ {
		start := time.Now()
		resp, err := b.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    openai.ChatModel(model),
			Messages: oaiMsgs,
		})
		elapsed := time.Since(start).Seconds() * 1000 // ms

		if err != nil {
			result.Errors++
			fmt.Fprintf(os.Stderr, "[Benchmark] %s iteration %d error: %v\n", model, i+1, err)
			continue
		}

		result.LatenciesMs = append(result.LatenciesMs, elapsed)
		result.PromptTokens = append(result.PromptTokens, int(resp.Usage.PromptTokens))
		result.CompletionTokens = append(result.CompletionTokens, int(resp.Usage.CompletionTokens))
	}

	return result
}

// GenerateReport formats a comparison report for multiple benchmark results.
func GenerateReport(results []*ProviderBenchmarkResult) string {
	var sb strings.Builder
	sb.WriteString("\n" + repeatStr("=", 70) + "\n")
	sb.WriteString("Provider Benchmark Report\n")
	sb.WriteString(repeatStr("=", 70) + "\n")
	fmt.Fprintf(&sb, "%-20s %12s %12s %12s %14s %8s\n",
		"Model", "Avg Lat(ms)", "P95 Lat(ms)", "StdDev(ms)", "Est Cost(USD)", "Errors")
	sb.WriteString(repeatStr("-", 70) + "\n")

	for _, r := range results {
		fmt.Fprintf(&sb, "%-20s %12.0f %12.0f %12.0f %14.5f %8d\n",
			r.ModelName, r.AvgLatencyMs(), r.P95LatencyMs(), r.StdDevLatencyMs(),
			r.EstimatedCostUSD(), r.Errors)
	}

	// Recommendation
	var bestCost, bestLatency *ProviderBenchmarkResult
	for _, r := range results {
		if bestCost == nil || r.EstimatedCostUSD() < bestCost.EstimatedCostUSD() {
			bestCost = r
		}
		if bestLatency == nil || r.AvgLatencyMs() < bestLatency.AvgLatencyMs() {
			bestLatency = r
		}
	}
	sb.WriteString("\nRecommendations:\n")
	if bestCost != nil {
		fmt.Fprintf(&sb, "  Most cost-efficient: %s ($%.5f)\n", bestCost.ModelName, bestCost.EstimatedCostUSD())
	}
	if bestLatency != nil {
		fmt.Fprintf(&sb, "  Lowest latency:      %s (%.0f ms avg)\n", bestLatency.ModelName, bestLatency.AvgLatencyMs())
	}

	return sb.String()
}

// RunProviderBenchmark demonstrates the benchmark tool.
func RunProviderBenchmark() {
	messages := []map[string]interface{}{
		{"role": "user", "content": "Explain the difference between machine learning and deep learning in 2 sentences."},
	}

	models := []string{"gpt-4o", "gpt-4o-mini"}
	bench := NewProviderBenchmark()
	var results []*ProviderBenchmarkResult

	for _, model := range models {
		fmt.Printf("[Benchmark] Testing %s...\n", model)
		result := bench.RunOne(context.Background(), model, messages, 3)
		results = append(results, result)
	}

	fmt.Println(GenerateReport(results))
}
