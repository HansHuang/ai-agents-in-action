// Token Cost Calculator — cost modeling and optimization for LLM context.
//
// Go port of code/python/05-context-assembly/token_cost_calculator.py
//
// Calculates and projects token costs across models and providers, identifies
// waste in system prompts and conversation history, and suggests concrete
// optimisations with projected savings.
//
// See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
package main

import (
	"fmt"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// Pricing table  (USD per 1 M tokens, May 2026)
// ---------------------------------------------------------------------------

// ModelPricing holds per-million-token costs for a model.
type ModelPricing struct {
	Input  float64
	Output float64 // 0 if not applicable (e.g. embedding models)
}

// ProviderPricing maps model names to their pricing.
type ProviderPricing map[string]ModelPricing

// LLMPricing is the full pricing table keyed by provider → model → prices.
var LLMPricing = map[string]ProviderPricing{
	"openai": {
		"gpt-4o":                 {Input: 2.50, Output: 10.00},
		"gpt-4o-mini":            {Input: 0.15, Output: 0.60},
		"text-embedding-3-small": {Input: 0.02},
		"text-embedding-3-large": {Input: 0.13},
	},
	"anthropic": {
		"claude-3.5-sonnet": {Input: 3.00, Output: 15.00},
		"claude-3-haiku":    {Input: 0.25, Output: 1.25},
	},
	"google": {
		"gemini-1.5-pro":   {Input: 3.50, Output: 10.50},
		"gemini-1.5-flash": {Input: 0.075, Output: 0.30},
	},
}

// flatPricing is a flat model → pricing lookup.
var flatPricing = func() map[string]ModelPricing {
	flat := map[string]ModelPricing{}
	for _, provider := range LLMPricing {
		for model, p := range provider {
			flat[model] = p
		}
	}
	return flat
}()

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

// CostEstimate is the cost estimate for a single API call.
type CostEstimate struct {
	Model        string
	InputTokens  int
	OutputTokens int
	CostUSD      float64
}

// TotalTokens returns the combined token count.
func (c CostEstimate) TotalTokens() int {
	return c.InputTokens + c.OutputTokens
}

// ProjectionResult holds cost projections over daily / monthly / annual horizons.
type ProjectionResult struct {
	Model           string
	CallsPerDay     int
	AvgInputTokens  int
	AvgOutputTokens int
	CostPerCall     float64
	Daily           float64
	Monthly         float64
	Annual          float64
}

// AuditResult is the audit of a message context for cost optimisation opportunities.
type AuditResult struct {
	TotalTokens            int
	ByRole                 map[string]int
	SystemPromptEfficiency float64
	WastedTokens           int
	Suggestions            []string
	ProjectedSavings       map[string]float64
}

// ---------------------------------------------------------------------------
// TokenCostCalculator
// ---------------------------------------------------------------------------

// TokenCostCalculator calculates and optimises token costs across models and providers.
type TokenCostCalculator struct {
	Model string
}

// NewTokenCostCalculator creates a new TokenCostCalculator.
func NewTokenCostCalculator(model string) *TokenCostCalculator {
	if model == "" {
		model = "gpt-4o"
	}
	return &TokenCostCalculator{Model: model}
}

// LLMCOSTCALC_PRICING exposes the pricing table for inspection and extension.
// Use LLMPricing directly for access.

// CalculateCallCost calculates the USD cost for a single API call.
func (tc *TokenCostCalculator) CalculateCallCost(model string, inputTokens, outputTokens int) (float64, error) {
	pricing, err := tc.getPricing(model)
	if err != nil {
		return 0, err
	}
	cost := float64(inputTokens) / 1_000_000 * pricing.Input
	if outputTokens > 0 && pricing.Output > 0 {
		cost += float64(outputTokens) / 1_000_000 * pricing.Output
	}
	return cost, nil
}

// CalculateDailyCost projects daily, monthly, and annual costs.
func (tc *TokenCostCalculator) CalculateDailyCost(model string, callsPerDay, avgInputTokens, avgOutputTokens int) (*ProjectionResult, error) {
	costPerCall, err := tc.CalculateCallCost(model, avgInputTokens, avgOutputTokens)
	if err != nil {
		return nil, err
	}
	daily := costPerCall * float64(callsPerDay)
	return &ProjectionResult{
		Model:           model,
		CallsPerDay:     callsPerDay,
		AvgInputTokens:  avgInputTokens,
		AvgOutputTokens: avgOutputTokens,
		CostPerCall:     costPerCall,
		Daily:           daily,
		Monthly:         daily * 30,
		Annual:          daily * 365,
	}, nil
}

// CompareModels returns a formatted comparison table of costs across all models.
func (tc *TokenCostCalculator) CompareModels(inputTokens, outputTokens, callsPerDay int) string {
	cModel := 28
	cDaily := 12
	cMonthly := 12
	cAnnual := 13

	header := fmt.Sprintf("%-*s | %*s | %*s | %*s",
		cModel, "Model",
		cDaily, "Daily",
		cMonthly, "Monthly",
		cAnnual, "Annual")
	sep := strings.Repeat("-", cModel+cDaily+cMonthly+cAnnual+9)
	rows := []string{header, sep}

	for model := range flatPricing {
		proj, err := tc.CalculateDailyCost(model, callsPerDay, inputTokens, outputTokens)
		if err != nil {
			continue
		}
		rows = append(rows, fmt.Sprintf("%-*s | $%*.2f | $%*.2f | $%*.2f",
			cModel, model,
			cDaily-1, proj.Daily,
			cMonthly-1, proj.Monthly,
			cAnnual-1, proj.Annual))
	}
	return strings.Join(rows, "\n")
}

// OptimizeSystemPrompt returns a condensed version of prompt using rule-based compression.
//
// Rules applied in order:
//  1. Collapse runs of blank lines to at most one.
//  2. Remove duplicate lines (case-insensitive).
//  3. Strip common verbose filler phrases.
//  4. Truncate to the target token count if still over budget.
func (tc *TokenCostCalculator) OptimizeSystemPrompt(prompt string, targetReductionPct float64) string {
	origTokens, _ := CountTokens(prompt, tc.Model)
	targetTokens := int(float64(origTokens) * (1.0 - targetReductionPct))

	// Rule 1: collapse blank lines
	lines := strings.Split(prompt, "\n")
	var collapsed []string
	prevBlank := false
	for _, line := range lines {
		blank := strings.TrimSpace(line) == ""
		if blank && prevBlank {
			continue
		}
		prevBlank = blank
		collapsed = append(collapsed, line)
	}

	// Rule 2: remove duplicate lines
	seen := map[string]bool{}
	var deduped []string
	for _, line := range collapsed {
		key := strings.ToLower(strings.TrimSpace(line))
		if key != "" && seen[key] {
			continue
		}
		seen[key] = true
		deduped = append(deduped, line)
	}

	// Rule 3: strip filler phrases
	fillerPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)\bplease note that\b`),
		regexp.MustCompile(`(?i)\bit is important to remember that\b`),
		regexp.MustCompile(`(?i)\bas an AI language model\b`),
		regexp.MustCompile(`(?i)\bfeel free to\b`),
		regexp.MustCompile(`(?i)\bdon't hesitate to\b`),
		regexp.MustCompile(`(?i)\bI hope this helps\b`),
		regexp.MustCompile(`(?i)\bof course\b`),
		regexp.MustCompile(`(?i)\bsure\b[,!]\s*`),
		regexp.MustCompile(`(?i)\babsolutely\b[,!]\s*`),
		regexp.MustCompile(`(?i)\bcertainly\b[,!]\s*`),
	}
	var resultLines []string
	for _, line := range deduped {
		for _, pat := range fillerPatterns {
			line = pat.ReplaceAllString(line, "")
		}
		resultLines = append(resultLines, strings.TrimRight(line, " \t"))
	}

	result := strings.TrimSpace(strings.Join(resultLines, "\n"))

	// Rule 4: truncate if still over target
	currentTokens, _ := CountTokens(result, tc.Model)
	if currentTokens > targetTokens {
		enc, err := getEncoding(tc.Model)
		if err == nil {
			tokens := enc.Encode(result, nil, nil)
			if len(tokens) > targetTokens {
				result = enc.Decode(tokens[:targetTokens])
			}
		}
	}

	return result
}

// AuditContext audits a message context for cost optimisation opportunities.
func (tc *TokenCostCalculator) AuditContext(messages []map[string]any, tools []map[string]any, callsPerDay int, model string) (*AuditResult, error) {
	if model == "" {
		model = tc.Model
	}

	byRole := map[string]int{}
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		if role == "" {
			role = "unknown"
		}
		content, _ := msg["content"].(string)
		n, err := CountTokens(content, model)
		if err != nil {
			return nil, err
		}
		byRole[role] += n
	}

	if len(tools) > 0 {
		toolTokens, err := CountTokens(convertToolsToAny(tools), model)
		if err != nil {
			return nil, err
		}
		if toolTokens > 0 {
			byRole["tools"] = toolTokens
		}
	}

	total := 0
	for _, n := range byRole {
		total += n
	}

	// System prompt efficiency
	sysText := ""
	for _, msg := range messages {
		if role, _ := msg["role"].(string); role == "system" {
			sysText, _ = msg["content"].(string)
			break
		}
	}
	sysTokens := byRole["system"]
	constraintLineWords := 0
	for _, ln := range strings.Split(sysText, "\n") {
		stripped := strings.TrimSpace(ln)
		wordCount := len(strings.Fields(stripped))
		if wordCount > 5 && len(stripped) < 120 && !strings.HasPrefix(stripped, "#") {
			constraintLineWords += wordCount
		}
	}
	estConstraintTokens := float64(constraintLineWords) * 1.3
	efficiency := 0.0
	if sysTokens > 0 {
		efficiency = estConstraintTokens / float64(sysTokens)
		if efficiency > 1.0 {
			efficiency = 1.0
		}
		if efficiency < 0.0 {
			efficiency = 0.0
		}
	}

	// Identify waste sources
	var suggestions []string
	wasted := 0

	// Examples in system prompt
	exampleRe := regexp.MustCompile(`(?i)(Example|e\.g\.|For example|For instance)`)
	exampleCount := len(exampleRe.FindAllString(sysText, -1))
	if exampleCount > 2 {
		est := exampleCount * 200
		wasted += est
		suggestions = append(suggestions, fmt.Sprintf(
			"System prompt: Move %d examples to dynamic context (save ~%d tokens per call).",
			exampleCount, est))
	}

	// Old assistant turns
	var assistantMsgs []map[string]any
	for _, msg := range messages {
		if role, _ := msg["role"].(string); role == "assistant" {
			assistantMsgs = append(assistantMsgs, msg)
		}
	}
	if len(assistantMsgs) > 6 {
		oldTurns := assistantMsgs[:len(assistantMsgs)-3]
		oldTokens := 0
		for _, msg := range oldTurns {
			content, _ := msg["content"].(string)
			n, _ := CountTokens(content, model)
			oldTokens += n
		}
		if oldTokens > 2_000 {
			est := int(float64(oldTokens) * 0.70)
			wasted += est
			suggestions = append(suggestions, fmt.Sprintf(
				"Messages: Summarise turns 1–%d (save ~%d tokens).",
				len(oldTurns), est))
		}
	}

	// Verbose tool descriptions
	for _, tool := range tools {
		fn, _ := tool["function"].(map[string]any)
		if fn == nil {
			continue
		}
		desc, _ := fn["description"].(string)
		name, _ := fn["name"].(string)
		wordCount := len(strings.Fields(desc))
		if wordCount > 25 {
			est := wordCount - 15
			suggestions = append(suggestions, fmt.Sprintf(
				"Tool '%s': Shorten description (%d words → 15; save ~%d tokens).",
				name, wordCount, est))
		}
	}

	if len(suggestions) == 0 {
		suggestions = append(suggestions, "No significant optimisation opportunities detected.")
	}

	// Projected savings
	wastedDaily := wasted * callsPerDay
	usdPerDay := 0.0
	if p, ok := flatPricing[strings.ToLower(model)]; ok {
		usdPerDay = float64(wastedDaily) / 1_000_000 * p.Input
	}

	projectedSavings := map[string]float64{
		"tokens_per_call": float64(wasted),
		"tokens_daily":    float64(wastedDaily),
		"tokens_monthly":  float64(wastedDaily) * 30,
		"tokens_annual":   float64(wastedDaily) * 365,
		"usd_daily":       usdPerDay,
		"usd_monthly":     usdPerDay * 30,
		"usd_annual":      usdPerDay * 365,
	}

	return &AuditResult{
		TotalTokens:            total,
		ByRole:                 byRole,
		SystemPromptEfficiency: efficiency,
		WastedTokens:           wasted,
		Suggestions:            suggestions,
		ProjectedSavings:       projectedSavings,
	}, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func (tc *TokenCostCalculator) getPricing(model string) (ModelPricing, error) {
	if p, ok := flatPricing[strings.ToLower(model)]; ok {
		return p, nil
	}
	return ModelPricing{}, fmt.Errorf("unknown model %q; available: see Pricing table", model)
}

// convertToolsToAny converts []map[string]any to []any for CountTokens.
func convertToolsToAny(tools []map[string]any) []any {
	result := make([]any, len(tools))
	for i, t := range tools {
		result[i] = t
	}
	return result
}
