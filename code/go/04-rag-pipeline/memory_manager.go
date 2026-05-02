// Package ragpipeline provides token-aware conversation memory management.
//
// Three overflow strategies are supported:
//   - Truncate:      Drop oldest complete turns, keep system prompt + recent.
//   - Summarize:     LLM-compress old messages, keep recent verbatim.
//   - SlidingWindow: Rolling summary of old messages + verbatim recent (default).
//
// Token counting uses a character-based approximation (~4 chars/token for
// English prose). For production use, integrate tiktoken-go.
//
// See: docs/03-memory-and-retrieval/01-short-term-memory.md
package ragpipeline

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	outputReserve   = 4096
	msgOverhead     = 4
	primingOverhead = 2
)

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

// Message represents a single chat message.
type Message struct {
	Role       string        `json:"role"`
	Content    *string       `json:"content,omitempty"`
	ToolCalls  []interface{} `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

func strPtr(s string) *string { return &s }

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

// CountTokens returns an approximate token count for a message list.
// Uses ~4 chars/token for English. Replace with tiktoken-go for exactness.
func CountTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		total += msgOverhead
		total += approxTokens(msg.Role)
		if msg.Content != nil {
			total += approxTokens(*msg.Content)
		}
		if msg.ToolCallID != "" {
			total += approxTokens(msg.ToolCallID)
		}
		if len(msg.ToolCalls) > 0 {
			b, _ := json.Marshal(msg.ToolCalls)
			total += approxTokens(string(b))
		}
	}
	total += primingOverhead
	return total
}

func approxTokens(s string) int {
	return int(math.Ceil(float64(len(s)) / 4.0))
}

// ---------------------------------------------------------------------------
// Turn grouping
// ---------------------------------------------------------------------------

func groupIntoTurns(messages []Message) [][]Message {
	var turns [][]Message
	var current []Message

	for _, msg := range messages {
		current = append(current, msg)
		if msg.Role == "assistant" && msg.Content != nil && *msg.Content != "" && len(msg.ToolCalls) == 0 {
			turns = append(turns, current)
			current = nil
		}
	}
	if len(current) > 0 {
		turns = append(turns, current)
	}
	return turns
}

// ---------------------------------------------------------------------------
// Strategy type
// ---------------------------------------------------------------------------

// Strategy determines how context overflow is handled.
type Strategy string

const (
	StrategyNone          Strategy = "none"
	StrategyTruncate      Strategy = "truncate"
	StrategySummarize     Strategy = "summarize"
	StrategySlidingWindow Strategy = "sliding_window"
)

// ---------------------------------------------------------------------------
// Summarizer interface
// ---------------------------------------------------------------------------

// Summarizer compresses a message list into a summary string.
type Summarizer interface {
	Summarize(messages []Message) (string, error)
}

// ---------------------------------------------------------------------------
// MemoryManager
// ---------------------------------------------------------------------------

// MemoryManager holds the conversation history and applies overflow strategies.
type MemoryManager struct {
	Model     string
	MaxTokens int
	Messages  []Message

	summarizer      Summarizer
	summaryCache    string
	summaryInputLen int
	hasSummaryCache bool
}

// MemoryManagerOptions configures a new MemoryManager.
type MemoryManagerOptions struct {
	Model        string
	MaxTokens    int
	SystemPrompt string
	Summarizer   Summarizer
}

// NewMemoryManager creates a MemoryManager with the given options.
func NewMemoryManager(opts MemoryManagerOptions) *MemoryManager {
	if opts.Model == "" {
		opts.Model = "gpt-4o"
	}
	if opts.MaxTokens == 0 {
		opts.MaxTokens = 100_000
	}
	content := opts.SystemPrompt
	return &MemoryManager{
		Model:      opts.Model,
		MaxTokens:  opts.MaxTokens,
		Messages:   []Message{{Role: "system", Content: &content}},
		summarizer: opts.Summarizer,
	}
}

// AddMessage appends a raw message.
func (m *MemoryManager) AddMessage(msg Message) {
	m.Messages = append(m.Messages, msg)
	m.hasSummaryCache = false
}

// AddUserMessage appends a user message.
func (m *MemoryManager) AddUserMessage(content string) {
	m.AddMessage(Message{Role: "user", Content: &content})
}

// AddAssistantMessage appends an assistant message.
func (m *MemoryManager) AddAssistantMessage(content string, toolCalls []interface{}) {
	msg := Message{Role: "assistant", Content: &content}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	m.AddMessage(msg)
}

// AddToolResult appends a tool result message.
func (m *MemoryManager) AddToolResult(toolCallID, content string) {
	m.AddMessage(Message{Role: "tool", ToolCallID: toolCallID, Content: &content})
}

// TokenCount returns the current token count of the full message history.
func (m *MemoryManager) TokenCount() int {
	return CountTokens(m.Messages)
}

// Rollback truncates the message list to the given index (exclusive).
// Index 1 removes all messages after the system prompt.
func (m *MemoryManager) Rollback(toIndex int) error {
	if toIndex < 1 || toIndex > len(m.Messages) {
		return fmt.Errorf("toIndex must be in [1, %d]; got %d", len(m.Messages), toIndex)
	}
	m.Messages = m.Messages[:toIndex]
	m.hasSummaryCache = false
	return nil
}

// GetMessages returns the messages to send to the API using the chosen strategy.
func (m *MemoryManager) GetMessages(strategy Strategy, recentCount int) ([]Message, error) {
	if recentCount == 0 {
		recentCount = 10
	}

	currentTokens := m.TokenCount()
	if strategy == StrategyNone || currentTokens <= m.MaxTokens {
		return m.Messages, nil
	}

	switch strategy {
	case StrategyTruncate:
		return m.applyTruncation(), nil
	case StrategySummarize:
		return m.applySummarization(recentCount)
	case StrategySlidingWindow:
		return m.applySlidingWindow(recentCount)
	default:
		return nil, fmt.Errorf("unknown strategy %q; choose: none, truncate, summarize, sliding_window", strategy)
	}
}

func (m *MemoryManager) applyTruncation() []Message {
	systemMsg := m.Messages[0]
	turns := groupIntoTurns(m.Messages[1:])

	budget := m.MaxTokens - CountTokens([]Message{systemMsg})
	var kept [][]Message

	for i := len(turns) - 1; i >= 0; i-- {
		turnTokens := CountTokens(turns[i])
		if turnTokens <= budget {
			kept = append([][]Message{turns[i]}, kept...)
			budget -= turnTokens
		} else {
			break
		}
	}

	result := []Message{systemMsg}
	for _, turn := range kept {
		result = append(result, turn...)
	}
	return result
}

func (m *MemoryManager) applySummarization(recentCount int) ([]Message, error) {
	systemMsg := m.Messages[0]
	conversation := m.Messages[1:]

	if len(conversation) <= recentCount {
		return m.Messages, nil
	}

	toSummarize := conversation[:len(conversation)-recentCount]
	recent := conversation[len(conversation)-recentCount:]

	summary, err := m.getOrBuildSummary(toSummarize)
	if err != nil {
		return nil, err
	}

	summaryContent := "[Conversation summary: " + summary + "]"
	result := []Message{systemMsg, {Role: "user", Content: &summaryContent}}
	result = append(result, recent...)
	return result, nil
}

func (m *MemoryManager) applySlidingWindow(recentCount int) ([]Message, error) {
	systemMsg := m.Messages[0]
	conversation := m.Messages[1:]

	if len(conversation) <= recentCount {
		return m.Messages, nil
	}

	toSummarize := conversation[:len(conversation)-recentCount]
	recent := conversation[len(conversation)-recentCount:]

	summary, err := m.getOrBuildSummary(toSummarize)
	if err != nil {
		return nil, err
	}

	result := []Message{systemMsg}
	if summary != "" {
		sw := "[Conversation so far: " + summary + "]"
		result = append(result, Message{Role: "user", Content: &sw})
	}
	result = append(result, recent...)
	return result, nil
}

func (m *MemoryManager) getOrBuildSummary(messages []Message) (string, error) {
	if m.hasSummaryCache && m.summaryInputLen == len(messages) {
		return m.summaryCache, nil
	}
	if m.summarizer == nil {
		return "", fmt.Errorf("summarizer is required for summarize/sliding_window strategies")
	}
	summary, err := m.summarizer.Summarize(messages)
	if err != nil {
		return "", err
	}
	m.summaryCache = summary
	m.summaryInputLen = len(messages)
	m.hasSummaryCache = true
	return summary, nil
}

// ---------------------------------------------------------------------------
// Token Tracker
// ---------------------------------------------------------------------------

// Pricing holds per-1K token costs for a model.
type Pricing struct {
	Input  float64
	Output float64
}

// DefaultPricing is the built-in price table (USD per 1K tokens).
var DefaultPricing = map[string]Pricing{
	"gpt-4o":        {Input: 0.0025, Output: 0.01},
	"gpt-4o-mini":   {Input: 0.00015, Output: 0.0006},
	"gpt-3.5-turbo": {Input: 0.0005, Output: 0.0015},
}

// TokenUsage records a single LLM API call.
type TokenUsage struct {
	Model        string
	InputTokens  int
	OutputTokens int
	Timestamp    time.Time
	Purpose      string
}

// CostUSD returns the USD cost for this call.
func (t *TokenUsage) CostUSD() float64 {
	p, ok := DefaultPricing[t.Model]
	if !ok {
		return 0
	}
	return float64(t.InputTokens)/1000*p.Input + float64(t.OutputTokens)/1000*p.Output
}

// TokenTracker accumulates token usage across API calls.
type TokenTracker struct {
	BudgetCap *float64
	mu        sync.Mutex
	records   []*TokenUsage
	warned    bool
}

// NewTokenTracker creates a tracker with an optional budget cap (USD).
func NewTokenTracker(budgetCap *float64) *TokenTracker {
	return &TokenTracker{BudgetCap: budgetCap}
}

// RecordCall records a completed API call and checks budget.
func (t *TokenTracker) RecordCall(model string, inputTokens, outputTokens int, purpose string) *TokenUsage {
	record := &TokenUsage{
		Model:        model,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		Timestamp:    time.Now().UTC(),
		Purpose:      purpose,
	}

	t.mu.Lock()
	t.records = append(t.records, record)
	totalCost := t.totalCostLocked()
	t.mu.Unlock()

	if t.BudgetCap != nil {
		frac := totalCost / *t.BudgetCap
		if frac >= 0.8 && !t.warned {
			t.warned = true
			fmt.Printf("[TokenTracker] Budget warning: $%.4f / $%.2f (%.0f%% used)\n",
				totalCost, *t.BudgetCap, frac*100)
		}
	}
	return record
}

// TotalInputTokens returns the sum of all input tokens.
func (t *TokenTracker) TotalInputTokens() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	total := 0
	for _, r := range t.records {
		total += r.InputTokens
	}
	return total
}

// TotalOutputTokens returns the sum of all output tokens.
func (t *TokenTracker) TotalOutputTokens() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	total := 0
	for _, r := range t.records {
		total += r.OutputTokens
	}
	return total
}

// TotalCost returns the total spend in USD.
func (t *TokenTracker) TotalCost() float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.totalCostLocked()
}

func (t *TokenTracker) totalCostLocked() float64 {
	total := 0.0
	for _, r := range t.records {
		total += r.CostUSD()
	}
	return total
}

// IsBudgetExceeded returns true if spend >= budget cap.
func (t *TokenTracker) IsBudgetExceeded() bool {
	if t.BudgetCap == nil {
		return false
	}
	return t.TotalCost() >= *t.BudgetCap
}

// GenerateReport returns a human-readable report string.
func (t *TokenTracker) GenerateReport() string {
	t.mu.Lock()
	records := make([]*TokenUsage, len(t.records))
	copy(records, t.records)
	t.mu.Unlock()

	totalIn := 0
	totalOut := 0
	totalCost := 0.0
	for _, r := range records {
		totalIn += r.InputTokens
		totalOut += r.OutputTokens
		totalCost += r.CostUSD()
	}

	var sb strings.Builder
	sb.WriteString(strings.Repeat("=", 60) + "\n")
	sb.WriteString("TOKEN USAGE REPORT\n")
	sb.WriteString(strings.Repeat("=", 60) + "\n")
	sb.WriteString(fmt.Sprintf("  Total calls:         %d\n", len(records)))
	sb.WriteString(fmt.Sprintf("  Total input tokens:  %d\n", totalIn))
	sb.WriteString(fmt.Sprintf("  Total output tokens: %d\n", totalOut))
	sb.WriteString(fmt.Sprintf("  Total cost:          $%.6f\n", totalCost))
	if t.BudgetCap != nil {
		pct := totalCost / *t.BudgetCap * 100
		sb.WriteString(fmt.Sprintf("  Budget cap:          $%.2f\n", *t.BudgetCap))
		sb.WriteString(fmt.Sprintf("  Budget used:         %.1f%%\n", pct))
	}
	sb.WriteString(strings.Repeat("=", 60))
	return sb.String()
}
