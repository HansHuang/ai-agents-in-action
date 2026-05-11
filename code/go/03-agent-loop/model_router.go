// model_router.go — Task-based model router (Go port).
//
// Routes LLM tasks to the most appropriate provider based on:
//   - Task type (chat, reasoning, classification, summarisation, code)
//   - Priority (cost / latency / quality)
//   - Context size requirements
//   - Tool-call / structured-output requirements
//
// See: docs/05-the-tool-ecosystem/01-model-providers.md
package main

import (
	"fmt"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

// TaskType classifies the LLM workload.
type TaskType string

const (
	TaskChat           TaskType = "chat"
	TaskReasoning      TaskType = "reasoning"
	TaskClassification TaskType = "classification"
	TaskSummarization  TaskType = "summarization"
	TaskCode           TaskType = "code"
)

// Priority selects the optimisation target.
type Priority string

const (
	PriorityCost    Priority = "cost"
	PriorityLatency Priority = "latency"
	PriorityQuality Priority = "quality"
)

// RoutingTask describes a task to be routed.
type RoutingTask struct {
	Messages                 []map[string]interface{}
	TaskType                 TaskType
	EstimatedInputTokens     int
	EstimatedOutputTokens    int
	Priority                 Priority
	RequiresTools            bool
	RequiresStructuredOutput bool
}

// ProviderCapabilities stores metadata for a registered provider.
type ProviderCapabilities struct {
	Name             string
	Capabilities     []string // e.g. ["fast", "smart", "cheap", "long_context"]
	CostPer1KInput   float64  // USD
	CostPer1KOutput  float64  // USD
	TypicalLatencyMs int
	MaxContextTokens int
}

// HasCapability returns true if the provider declares the given capability.
func (p *ProviderCapabilities) HasCapability(cap string) bool {
	for _, c := range p.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// RouterConfig holds tuneable knobs for the router's scoring algorithm.
type RouterConfig struct {
	MaxCostPer1KTokens *float64
	MaxLatencyMs       *int
}

// ModelRouter routes tasks to the most appropriate provider.
type ModelRouter struct {
	providers []*ProviderCapabilities
	config    RouterConfig
}

// NewModelRouter creates an empty router.
func NewModelRouter(config RouterConfig) *ModelRouter {
	return &ModelRouter{config: config}
}

// RegisterProvider adds a provider to the router.
func (r *ModelRouter) RegisterProvider(cap *ProviderCapabilities) {
	r.providers = append(r.providers, cap)
}

// Route selects the best provider for the given task.
// Returns the selected provider name and a score map for debugging.
func (r *ModelRouter) Route(task *RoutingTask) (*ProviderCapabilities, map[string]float64, error) {
	candidates := r.filterCandidates(task)
	if len(candidates) == 0 {
		return nil, nil, fmt.Errorf("no providers available for task %q with priority %q", task.TaskType, task.Priority)
	}

	scores := make(map[string]float64, len(candidates))
	for _, p := range candidates {
		scores[p.Name] = r.score(p, task)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return scores[candidates[i].Name] > scores[candidates[j].Name]
	})

	selected := candidates[0]
	fmt.Printf("[Router] Selected '%s' (score=%.2f) for task_type=%s priority=%s\n",
		selected.Name, scores[selected.Name], task.TaskType, task.Priority)
	return selected, scores, nil
}

func (r *ModelRouter) filterCandidates(task *RoutingTask) []*ProviderCapabilities {
	var out []*ProviderCapabilities
	for _, p := range r.providers {
		if task.RequiresTools && !p.HasCapability("function_calling") {
			continue
		}
		if task.RequiresStructuredOutput && !p.HasCapability("structured_output") {
			continue
		}
		totalTokens := task.EstimatedInputTokens + task.EstimatedOutputTokens
		if p.MaxContextTokens > 0 && totalTokens > p.MaxContextTokens {
			continue
		}
		if r.config.MaxLatencyMs != nil && p.TypicalLatencyMs > *r.config.MaxLatencyMs {
			continue
		}
		if r.config.MaxCostPer1KTokens != nil {
			costPer1K := (p.CostPer1KInput + p.CostPer1KOutput) / 2
			if costPer1K > *r.config.MaxCostPer1KTokens {
				continue
			}
		}
		out = append(out, p)
	}
	return out
}

// score assigns a numeric score to a provider for the given task.
// Higher is better.
func (r *ModelRouter) score(p *ProviderCapabilities, task *RoutingTask) float64 {
	var score float64

	switch task.Priority {
	case PriorityQuality:
		if p.HasCapability("smart") {
			score += 30
		}
		if p.HasCapability("function_calling") && task.RequiresTools {
			score += 20
		}
		if strings.Contains(string(task.TaskType), "reasoning") && p.HasCapability("smart") {
			score += 20
		}
	case PriorityCost:
		// Lower cost → higher score
		if p.CostPer1KInput < 0.001 {
			score += 30
		} else if p.CostPer1KInput < 0.003 {
			score += 20
		}
		if p.HasCapability("cheap") {
			score += 20
		}
	case PriorityLatency:
		// Lower latency → higher score
		if p.TypicalLatencyMs < 500 {
			score += 30
		} else if p.TypicalLatencyMs < 1500 {
			score += 15
		}
		if p.HasCapability("fast") {
			score += 20
		}
	}

	return score
}

// DefaultModelRouter returns a router pre-configured with common models.
func DefaultModelRouter() *ModelRouter {
	router := NewModelRouter(RouterConfig{})

	router.RegisterProvider(&ProviderCapabilities{
		Name:             "gpt-4o",
		Capabilities:     []string{"smart", "function_calling", "structured_output"},
		CostPer1KInput:   0.0025,
		CostPer1KOutput:  0.010,
		TypicalLatencyMs: 2000,
		MaxContextTokens: 128000,
	})
	router.RegisterProvider(&ProviderCapabilities{
		Name:             "gpt-4o-mini",
		Capabilities:     []string{"cheap", "fast", "function_calling"},
		CostPer1KInput:   0.00015,
		CostPer1KOutput:  0.0006,
		TypicalLatencyMs: 800,
		MaxContextTokens: 128000,
	})
	router.RegisterProvider(&ProviderCapabilities{
		Name:             "claude-3.5-sonnet",
		Capabilities:     []string{"smart", "long_context", "function_calling"},
		CostPer1KInput:   0.003,
		CostPer1KOutput:  0.015,
		TypicalLatencyMs: 2500,
		MaxContextTokens: 200000,
	})

	return router
}

// RunModelRouterDemo demonstrates the model router with sample tasks.
func RunModelRouterDemo() {
	router := DefaultModelRouter()

	tasks := []*RoutingTask{
		{TaskType: TaskClassification, Priority: PriorityCost, EstimatedInputTokens: 100},
		{TaskType: TaskReasoning, Priority: PriorityQuality, RequiresTools: true, EstimatedInputTokens: 2000},
		{TaskType: TaskSummarization, Priority: PriorityLatency, EstimatedInputTokens: 8000},
	}

	for _, task := range tasks {
		provider, _, err := router.Route(task)
		if err != nil {
			fmt.Printf("Error routing task: %v\n", err)
			continue
		}
		fmt.Printf("Task: type=%s priority=%s → Provider: %s\n",
			task.TaskType, task.Priority, provider.Name)
	}
}
