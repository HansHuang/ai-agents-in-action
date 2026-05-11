// conversation_summarizer.go — Compresses message history for memory management.
//
// Uses a cheap model (gpt-4o-mini) to produce dense, information-preserving
// summaries of conversation history. Implements the Summarizer interface.
//
// See: docs/03-memory-and-retrieval/01-short-term-memory.md
package ragpipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const fullSummaryPrompt = `Summarize the following conversation between a user and an AI assistant.
Focus on information the assistant needs to continue helping the user.

INCLUDE:
- The user's original request and any changes to it
- Information gathered from tools (with specific values: numbers, dates, names)
- Decisions the assistant made and why
- Actions taken and their outcomes
- Pending tasks or unanswered questions
- User preferences or constraints mentioned

DO NOT INCLUDE:
- Greetings, small talk, pleasantries
- Exact wording of prompts unless critical
- Redundant or repeated information
- Tool call mechanics (just the results)

FORMAT:
Write as a dense paragraph in third person past tense. Be concise but complete.`

const incrementalSummaryPrompt = `You have an existing summary of a conversation and new messages that followed.
Update the summary to incorporate the new information.

Rules:
- Keep all important information from the existing summary.
- Add new facts, decisions, and outcomes from the new messages.
- Remove information that is no longer relevant.
- Keep the output as a single dense paragraph in third person past tense.`

const keyFactsPrompt = `Extract a list of key facts from this conversation.

A key fact is:
- A specific number, date, name, price, or measurement
- A decision or preference stated by the user
- A result or outcome from a tool call
- An unresolved question or pending task

Format as a JSON list of strings. Each item should be a single short sentence.
Example: ["The user's budget is $500.", "AAPL stock price is $192.35.",
          "The user needs the report by Friday 5pm EST."]

Return ONLY the JSON array, no other text.`

// ---------------------------------------------------------------------------
// Shared HTTP helper for simple (non-tool-calling) chat completions.
// ---------------------------------------------------------------------------

type basicChatReq struct {
	Model          string        `json:"model"`
	Messages       []chatMessage `json:"messages"`
	Temperature    float64       `json:"temperature,omitempty"`
	MaxTokens      int           `json:"max_tokens,omitempty"`
	ResponseFormat *jsonFmtType  `json:"response_format,omitempty"`
}

type jsonFmtType struct {
	Type string `json:"type"`
}

// callBasicChat calls the OpenAI chat completions endpoint and returns the
// first choice's content text and token count.
func callBasicChat(apiKey string, req basicChatReq) (string, int, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return "", 0, fmt.Errorf("marshal request: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", 0, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return "", 0, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBytes))
	}
	var cr chatResponse
	if err := json.Unmarshal(respBytes, &cr); err != nil {
		return "", 0, fmt.Errorf("decode response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return "", 0, fmt.Errorf("no choices in response")
	}
	return cr.Choices[0].Message.Content, cr.Usage.TotalTokens, nil
}

// ---------------------------------------------------------------------------
// formatForSummarizer
// ---------------------------------------------------------------------------

func formatForSummarizer(messages []Message) string {
	var lines []string
	for _, msg := range messages {
		role := msg.Role
		content := ""
		if msg.Content != nil {
			content = *msg.Content
		}
		if len(msg.ToolCalls) > 0 {
			tc, _ := json.Marshal(msg.ToolCalls)
			lines = append(lines, fmt.Sprintf("%s [tool_call]: %s", role, string(tc)))
		} else if msg.Role == "tool" {
			lines = append(lines, fmt.Sprintf("TOOL RESULT [%s]: %s", msg.ToolCallID, content))
		} else if content != "" {
			lines = append(lines, fmt.Sprintf("%s: %s", role, content))
		}
	}
	result := ""
	for i, l := range lines {
		if i > 0 {
			result += "\n"
		}
		result += l
	}
	return result
}

// ---------------------------------------------------------------------------
// ConversationSummarizer
// ---------------------------------------------------------------------------

// ConversationSummarizer summarizes conversation history into a concise format.
// It implements the Summarizer interface defined in memory_manager.go.
type ConversationSummarizer struct {
	Model  string
	apiKey string
}

// NewConversationSummarizer creates a ConversationSummarizer.
// model defaults to "gpt-4o-mini" if empty.
func NewConversationSummarizer(model string) *ConversationSummarizer {
	if model == "" {
		model = "gpt-4o-mini"
	}
	return &ConversationSummarizer{
		Model:  model,
		apiKey: os.Getenv("OPENAI_API_KEY"),
	}
}

// Summarize compresses messages into a single dense paragraph.
// Implements the Summarizer interface.
func (s *ConversationSummarizer) Summarize(messages []Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}
	formatted := formatForSummarizer(messages)
	content, _, err := callBasicChat(s.apiKey, basicChatReq{
		Model: s.Model,
		Messages: []chatMessage{
			{Role: "system", Content: fullSummaryPrompt},
			{Role: "user", Content: formatted},
		},
		MaxTokens: 512,
	})
	return content, err
}

// SummarizeIncremental updates an existing summary with new messages.
func (s *ConversationSummarizer) SummarizeIncremental(existingSummary string, newMessages []Message) (string, error) {
	if len(newMessages) == 0 {
		return existingSummary, nil
	}
	formattedNew := formatForSummarizer(newMessages)
	userContent := fmt.Sprintf("EXISTING SUMMARY:\n%s\n\nNEW MESSAGES:\n%s", existingSummary, formattedNew)
	content, _, err := callBasicChat(s.apiKey, basicChatReq{
		Model: s.Model,
		Messages: []chatMessage{
			{Role: "system", Content: incrementalSummaryPrompt},
			{Role: "user", Content: userContent},
		},
		MaxTokens: 512,
	})
	if err != nil {
		return existingSummary, err
	}
	return content, nil
}

// ExtractKeyFacts extracts a list of key facts from the conversation.
func (s *ConversationSummarizer) ExtractKeyFacts(messages []Message) ([]string, error) {
	if len(messages) == 0 {
		return nil, nil
	}
	formatted := formatForSummarizer(messages)
	raw, _, err := callBasicChat(s.apiKey, basicChatReq{
		Model: s.Model,
		Messages: []chatMessage{
			{Role: "system", Content: keyFactsPrompt},
			{Role: "user", Content: formatted},
		},
		MaxTokens: 256,
	})
	if err != nil {
		return nil, err
	}
	var facts []string
	if err := json.Unmarshal([]byte(raw), &facts); err != nil {
		return nil, nil
	}
	return facts, nil
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunConversationSummarizer demonstrates the ConversationSummarizer.
func RunConversationSummarizer() {
	conversation := []Message{
		{Role: "user", Content: strPtr("I need to research Apple stock. My budget for investing is $500.")},
		{Role: "assistant", Content: nil, ToolCalls: []interface{}{map[string]interface{}{
			"id": "c1", "function": map[string]interface{}{
				"name": "get_stock_price", "arguments": `{"ticker":"AAPL"}`,
			},
		}}},
		{Role: "tool", ToolCallID: "c1", Content: strPtr(`{"price": 192.35, "change_pct": 1.2}`)},
		{Role: "assistant", Content: strPtr("AAPL is currently $192.35, up 1.2% today.")},
		{Role: "user", Content: strPtr("What about Microsoft?")},
		{Role: "assistant", Content: strPtr("MSFT is at $415.10, up 0.8% today.")},
		{Role: "user", Content: strPtr("Which is a better buy for my $500 budget?")},
		{Role: "assistant", Content: strPtr("With $500 you can buy 2 shares of AAPL at $192.35 each ($384.70 total, $115.30 remaining), or 1 share of MSFT at $415.10 ($84.90 remaining).")},
	}

	orig := CountTokens(conversation)
	fmt.Printf("Original: %d messages, %d tokens\n", len(conversation), orig)

	s := NewConversationSummarizer("gpt-4o-mini")
	summary, err := s.Summarize(conversation)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	summaryTokens := CountTokens([]Message{{Role: "user", Content: &summary}})
	fmt.Printf("Summary:  1 message, %d tokens (%.0f%% of original)\n",
		summaryTokens, float64(summaryTokens)/float64(orig)*100)

	facts, _ := s.ExtractKeyFacts(conversation)
	fmt.Printf("Key facts: %d\n", len(facts))
	for _, f := range facts {
		fmt.Printf("  - %s\n", f)
	}
}
