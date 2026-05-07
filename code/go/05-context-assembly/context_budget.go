// Package main provides context budget management and dynamic assembly for LLM context windows.
// ContextBudget — zone-based token budget management for LLM context windows.
//
// Every LLM call passes through Enforce which ensures each zone stays within
// its allocated token quota.  Overflowing zones are compressed using
// zone-specific strategies, and every action is recorded in an audit trail.
//
// See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"

	tiktoken "github.com/pkoukk/tiktoken-go"
)

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

// getEncoding returns the tiktoken encoding for model, falling back to
// cl100k_base when the model is not found.
func getEncoding(model string) (*tiktoken.Tiktoken, error) {
	enc, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return tiktoken.GetEncoding("cl100k_base")
	}
	return enc, nil
}

// CountTokens returns the token count for content.
//
// Supported types:
//   - string — plain text token count
//   - []map[string]any — OpenAI message list (with per-message overhead)
//   - map[string]any — serialised as JSON then counted
//   - []any — each element counted individually
func CountTokens(content any, model string) (int, error) {
	if content == nil {
		return 0, nil
	}

	enc, err := getEncoding(model)
	if err != nil {
		return 0, fmt.Errorf("getEncoding: %w", err)
	}

	switch v := content.(type) {
	case string:
		return len(enc.Encode(v, nil, nil)), nil

	case []map[string]any:
		// OpenAI chat message list
		const tokensPerMessage = 3
		const tokensPerName = 1
		total := 0
		for _, msg := range v {
			total += tokensPerMessage
			for key, val := range msg {
				var s string
				switch sv := val.(type) {
				case string:
					s = sv
				default:
					b, _ := json.Marshal(val)
					s = string(b)
				}
				total += len(enc.Encode(s, nil, nil))
				if key == "name" {
					total += tokensPerName
				}
			}
		}
		total += 3 // reply primer
		return total, nil

	case map[string]any:
		b, err := json.Marshal(v)
		if err != nil {
			return 0, err
		}
		return len(enc.Encode(string(b), nil, nil)), nil

	case []any:
		total := 0
		for _, item := range v {
			n, err := CountTokens(item, model)
			if err != nil {
				return 0, err
			}
			total += n
		}
		return total, nil
	}

	return 0, nil
}

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

// ActionTaken describes what the enforcer did with a zone.
type ActionTaken string

const (
	ActionWithinBudget  ActionTaken = "within_budget"
	ActionTruncated     ActionTaken = "truncated"
	ActionSlidingWindow ActionTaken = "sliding_window"
	ActionFiltered      ActionTaken = "filtered"
	ActionReserved      ActionTaken = "reserved"
)

// ZoneAudit records what happened to one context zone during enforcement.
type ZoneAudit struct {
	Zone           string      `json:"zone"`
	OriginalTokens int         `json:"original_tokens"`
	BudgetTokens   int         `json:"budget_tokens"`
	FinalTokens    int         `json:"final_tokens"`
	ActionTaken    ActionTaken `json:"action_taken"`
}

// TokensSaved returns the number of tokens removed by compression.
func (a ZoneAudit) TokensSaved() int {
	if saved := a.OriginalTokens - a.FinalTokens; saved > 0 {
		return saved
	}
	return 0
}

// EnforceResult contains the compressed context and a full audit trail.
type EnforceResult struct {
	SystemPrompt    string               `json:"system_prompt"`
	Messages        []map[string]any     `json:"messages"`
	DynamicContext  string               `json:"dynamic_context"`
	ToolDefinitions []map[string]any     `json:"tool_definitions"`
	Audit           map[string]ZoneAudit `json:"audit"`
	Warnings        []string             `json:"warnings"`
}

// TotalTokensSaved returns the sum of tokens saved across all zones.
func (r *EnforceResult) TotalTokensSaved() int {
	total := 0
	for _, a := range r.Audit {
		total += a.TokensSaved()
	}
	return total
}

// TotalTokensUsed returns the sum of final tokens across all zones.
func (r *EnforceResult) TotalTokensUsed() int {
	total := 0
	for _, a := range r.Audit {
		total += a.FinalTokens
	}
	return total
}

// BudgetExceededError is returned when content cannot be compressed enough.
type BudgetExceededError struct {
	Zone    string
	Message string
}

func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf("budget exceeded in zone %q: %s", e.Zone, e.Message)
}

// ---------------------------------------------------------------------------
// ContextBudget
// ---------------------------------------------------------------------------

// Pricing holds per-1K-token costs for a model (USD).
type Pricing struct {
	Input  float64
	Output float64
}

var defaultPricing = map[string]Pricing{
	"gpt-4o":            {Input: 0.0025, Output: 0.010},
	"gpt-4o-mini":       {Input: 0.00015, Output: 0.0006},
	"claude-3.5-sonnet": {Input: 0.003, Output: 0.015},
	"claude-3-haiku":    {Input: 0.00025, Output: 0.00125},
	"gemini-1.5-pro":    {Input: 0.0035, Output: 0.0105},
	"gemini-1.5-flash":  {Input: 0.000075, Output: 0.0003},
}

// ContextBudget defines and enforces token allocation across context zones.
//
// Usage:
//
//	budget := NewContextBudget(128_000, "gpt-4o")
//	result, err := budget.Enforce(systemPrompt, messages, dynCtx, tools)
type ContextBudget struct {
	TotalTokens int
	Model       string
	Allocations map[string]float64
}

// NewContextBudget creates a ContextBudget with default zone allocations.
func NewContextBudget(totalTokens int, model string) *ContextBudget {
	return &ContextBudget{
		TotalTokens: totalTokens,
		Model:       model,
		Allocations: map[string]float64{
			"system_prompt":        0.02,
			"tool_definitions":     0.05,
			"dynamic_context":      0.45,
			"conversation_history": 0.33,
			"response_buffer":      0.15,
		},
	}
}

// SetAllocation sets the fraction for a zone.
//
// The sum of all allocations must remain <= 1.0.
func (b *ContextBudget) SetAllocation(zone string, percentage float64) error {
	if _, ok := b.Allocations[zone]; !ok {
		return fmt.Errorf("unknown zone %q", zone)
	}
	if percentage < 0 || percentage > 1 {
		return fmt.Errorf("percentage must be in [0, 1], got %f", percentage)
	}
	total := percentage
	for k, v := range b.Allocations {
		if k != zone {
			total += v
		}
	}
	if total > 1.0+1e-9 {
		return fmt.Errorf("allocation total %.3f > 1.0 after setting %q to %f", total, zone, percentage)
	}
	b.Allocations[zone] = percentage
	return nil
}

// GetTokenBudget returns the token budget for a zone.
func (b *ContextBudget) GetTokenBudget(zone string) (int, error) {
	frac, ok := b.Allocations[zone]
	if !ok {
		return 0, fmt.Errorf("unknown zone %q", zone)
	}
	return int(math.Floor(float64(b.TotalTokens) * frac)), nil
}

// GetAllBudgets returns all zone token budgets.
func (b *ContextBudget) GetAllBudgets() (map[string]int, error) {
	result := make(map[string]int, len(b.Allocations))
	for zone := range b.Allocations {
		tok, err := b.GetTokenBudget(zone)
		if err != nil {
			return nil, err
		}
		result[zone] = tok
	}
	return result, nil
}

// MeasureZone counts tokens for content.
func (b *ContextBudget) MeasureZone(_ string, content any) (int, error) {
	return CountTokens(content, b.Model)
}

// Enforce enforces the budget on all zones and returns compressed content.
//
// Compression strategies:
//   - system_prompt:        Truncate from end; preserve first instructions.
//   - tool_definitions:     Keep tools that fit; drop oversized ones.
//   - dynamic_context:      Truncate to budget.
//   - conversation_history: Sliding window — drop oldest messages.
//   - response_buffer:      Reservation only; nothing to compress.
func (b *ContextBudget) Enforce(
	systemPrompt string,
	messages []map[string]any,
	dynamicContext string,
	toolDefinitions []map[string]any,
) (*EnforceResult, error) {
	if toolDefinitions == nil {
		toolDefinitions = []map[string]any{}
	}

	result := &EnforceResult{
		SystemPrompt:    systemPrompt,
		Messages:        append([]map[string]any{}, messages...),
		DynamicContext:  dynamicContext,
		ToolDefinitions: append([]map[string]any{}, toolDefinitions...),
		Audit:           make(map[string]ZoneAudit),
		Warnings:        []string{},
	}

	budgets, err := b.GetAllBudgets()
	if err != nil {
		return nil, err
	}

	// 1. System prompt
	spTok, _ := CountTokens(systemPrompt, b.Model)
	spBudget := budgets["system_prompt"]
	if spTok <= spBudget {
		result.Audit["system_prompt"] = ZoneAudit{"system_prompt", spTok, spBudget, spTok, ActionWithinBudget}
	} else {
		compressed, _ := b.compressText(systemPrompt, spBudget)
		finalTok, _ := CountTokens(compressed, b.Model)
		result.SystemPrompt = compressed
		msg := fmt.Sprintf("system_prompt: %d tokens exceeded budget %d; truncated to %d tokens.", spTok, spBudget, finalTok)
		result.Warnings = append(result.Warnings, msg)
		result.Audit["system_prompt"] = ZoneAudit{"system_prompt", spTok, spBudget, finalTok, ActionTruncated}
		log.Println("WARN", msg)
	}

	// 2. Tool definitions
	tdAny := make([]any, len(toolDefinitions))
	for i, t := range toolDefinitions {
		tdAny[i] = t
	}
	tdTok, _ := CountTokens(tdAny, b.Model)
	tdBudget := budgets["tool_definitions"]
	if tdTok <= tdBudget {
		result.Audit["tool_definitions"] = ZoneAudit{"tool_definitions", tdTok, tdBudget, tdTok, ActionWithinBudget}
	} else {
		compressed := b.compressToolDefinitions(toolDefinitions, tdBudget)
		compAny := make([]any, len(compressed))
		for i, t := range compressed {
			compAny[i] = t
		}
		finalTok, _ := CountTokens(compAny, b.Model)
		result.ToolDefinitions = compressed
		msg := fmt.Sprintf("tool_definitions: %d tokens exceeded budget %d; trimmed to %d tokens (%d/%d tools kept).", tdTok, tdBudget, finalTok, len(compressed), len(toolDefinitions))
		result.Warnings = append(result.Warnings, msg)
		result.Audit["tool_definitions"] = ZoneAudit{"tool_definitions", tdTok, tdBudget, finalTok, ActionFiltered}
		log.Println("WARN", msg)
	}

	// 3. Dynamic context
	dcTok, _ := CountTokens(dynamicContext, b.Model)
	dcBudget := budgets["dynamic_context"]
	if dcTok <= dcBudget {
		result.Audit["dynamic_context"] = ZoneAudit{"dynamic_context", dcTok, dcBudget, dcTok, ActionWithinBudget}
	} else {
		compressed, _ := b.compressText(dynamicContext, dcBudget)
		finalTok, _ := CountTokens(compressed, b.Model)
		result.DynamicContext = compressed
		msg := fmt.Sprintf("dynamic_context: %d tokens exceeded budget %d; truncated to %d tokens.", dcTok, dcBudget, finalTok)
		result.Warnings = append(result.Warnings, msg)
		result.Audit["dynamic_context"] = ZoneAudit{"dynamic_context", dcTok, dcBudget, finalTok, ActionTruncated}
		log.Println("WARN", msg)
	}

	// 4. Conversation history
	var history []map[string]any
	for _, m := range result.Messages {
		if m["role"] != "system" {
			history = append(history, m)
		}
	}
	histAny := make([]any, len(history))
	for i, m := range history {
		histAny[i] = m
	}
	histTok, _ := CountTokens(histAny, b.Model)
	histBudget := budgets["conversation_history"]
	if histTok <= histBudget {
		result.Audit["conversation_history"] = ZoneAudit{"conversation_history", histTok, histBudget, histTok, ActionWithinBudget}
	} else {
		compressed := b.compressHistory(history, histBudget)
		compAny := make([]any, len(compressed))
		for i, m := range compressed {
			compAny[i] = m
		}
		finalTok, _ := CountTokens(compAny, b.Model)
		var systemMsgs []map[string]any
		for _, m := range result.Messages {
			if m["role"] == "system" {
				systemMsgs = append(systemMsgs, m)
			}
		}
		result.Messages = append(systemMsgs, compressed...)
		msg := fmt.Sprintf("conversation_history: %d tokens exceeded budget %d; sliding window applied, %d tokens kept.", histTok, histBudget, finalTok)
		result.Warnings = append(result.Warnings, msg)
		result.Audit["conversation_history"] = ZoneAudit{"conversation_history", histTok, histBudget, finalTok, ActionSlidingWindow}
		log.Println("WARN", msg)
	}

	// 5. Response buffer
	rbBudget := budgets["response_buffer"]
	result.Audit["response_buffer"] = ZoneAudit{"response_buffer", 0, rbBudget, 0, ActionReserved}

	return result, nil
}

// EstimateCost returns the estimated USD cost for the given token counts.
func (b *ContextBudget) EstimateCost(inputTokens, outputTokens int, model string) float64 {
	if model == "" {
		model = b.Model
	}
	p, ok := defaultPricing[strings.ToLower(model)]
	if !ok {
		return 0
	}
	return float64(inputTokens)/1000*p.Input + float64(outputTokens)/1000*p.Output
}

// ---------------------------------------------------------------------------
// Compression helpers
// ---------------------------------------------------------------------------

func (b *ContextBudget) compressText(text string, maxTokens int) (string, error) {
	enc, err := getEncoding(b.Model)
	if err != nil {
		// Rough character-based fallback
		if len(text) > maxTokens*4 {
			return text[:maxTokens*4], nil
		}
		return text, nil
	}
	tokens := enc.Encode(text, nil, nil)
	if len(tokens) <= maxTokens {
		return text, nil
	}
	decoded := enc.Decode(tokens[:maxTokens])
	return string(decoded), nil
}

func (b *ContextBudget) compressToolDefinitions(
	tools []map[string]any,
	maxTokens int,
) []map[string]any {
	kept := []map[string]any{}
	used := 0
	for _, tool := range tools {
		t, _ := CountTokens(tool, b.Model)
		if used+t <= maxTokens {
			kept = append(kept, tool)
			used += t
		}
		if maxTokens-used < 50 {
			break
		}
	}
	return kept
}

func (b *ContextBudget) compressHistory(
	messages []map[string]any,
	maxTokens int,
) []map[string]any {
	if len(messages) == 0 {
		return messages
	}

	kept := []map[string]any{}
	used := 0
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		msgSlice := []map[string]any{msg}
		msgAny := make([]any, len(msgSlice))
		for j, m := range msgSlice {
			msgAny[j] = m
		}
		t, _ := CountTokens(msgAny, b.Model)
		if used+t <= maxTokens {
			kept = append([]map[string]any{msg}, kept...)
			used += t
		} else {
			break
		}
	}

	if len(kept) == 0 && len(messages) > 0 {
		return []map[string]any{messages[len(messages)-1]}
	}
	return kept
}
