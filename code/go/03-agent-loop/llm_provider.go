// LLM Provider abstraction — Go port.
//
// Implements a uniform interface over multiple LLM providers so agent code
// never depends on a specific SDK or wire format.
//
// Supported providers:
//
//	OpenAIProvider, AnthropicProvider, OllamaProvider, FallbackProvider
//
// Usage:
//
//	provider := NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o")
//	resp, err := provider.Chat(context.Background(), []Message{
//	    {Role: "user", Content: "Hello"},
//	}, ChatOptions{})
//
// See: docs/05-the-tool-ecosystem/01-model-providers.md
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Shared data types
// ---------------------------------------------------------------------------

// Message is a single chat message in the OpenAI format.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall is a function-calling entry from an assistant message.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the name and JSON-encoded arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolDefinition is the OpenAI-format tool definition.
type ToolDefinition struct {
	Type     string             `json:"type"` // "function"
	Function ToolFunctionSchema `json:"function"`
}

// ToolFunctionSchema describes a function tool.
type ToolFunctionSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// TokenUsage holds token consumption for a call.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// LLMResponse is the normalised response returned by every provider.
type LLMResponse struct {
	Content      string     // nil-equivalent: empty string
	ToolCalls    []ToolCall // nil when no tool calls
	TokenUsage   TokenUsage
	Model        string
	FinishReason string
	LatencyMs    int
}

// ChatOptions holds optional parameters for a Chat call.
type ChatOptions struct {
	Tools          []ToolDefinition
	Temperature    float64
	MaxTokens      int
	ResponseFormat map[string]any
}

func defaultChatOptions(o ChatOptions) ChatOptions {
	if o.Temperature == 0 {
		o.Temperature = 0.7
	}
	if o.MaxTokens == 0 {
		o.MaxTokens = 4096
	}
	return o
}

// EstimateTokens returns a rough token estimate (~4 chars per token).
func EstimateTokens(text string) int {
	n := len(text) / 4
	if n < 1 {
		return 1
	}
	return n
}

// ---------------------------------------------------------------------------
// LLMProvider interface
// ---------------------------------------------------------------------------

// LLMProvider is the common interface for all LLM providers.
type LLMProvider interface {
	Chat(ctx context.Context, messages []Message, opts ChatOptions) (LLMResponse, error)
	SupportsFunctionCalling() bool
	SupportsStructuredOutput() bool
	GetContextWindow() int
	GetModelName() string
	CountTokens(text string) int
}

// ---------------------------------------------------------------------------
// OpenAI provider  (also used as the engine for Ollama / Together)
// ---------------------------------------------------------------------------

// OpenAIProvider talks to the OpenAI chat completions endpoint
// (or any OpenAI-compatible endpoint via BaseURL).
type OpenAIProvider struct {
	APIKey  string
	Model   string
	BaseURL string
	client  *http.Client
}

var openaiContextWindows = map[string]int{
	"gpt-4o":        128_000,
	"gpt-4o-mini":   128_000,
	"gpt-3.5-turbo": 16_385,
}

// NewOpenAIProvider creates an OpenAIProvider for the given model.
func NewOpenAIProvider(apiKey, model string) *OpenAIProvider {
	return &OpenAIProvider{
		APIKey: apiKey, Model: model,
		BaseURL: "https://api.openai.com/v1",
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *OpenAIProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (LLMResponse, error) {
	opts = defaultChatOptions(opts)

	type oaiReq struct {
		Model          string           `json:"model"`
		Messages       []Message        `json:"messages"`
		Temperature    float64          `json:"temperature"`
		MaxTokens      int              `json:"max_tokens"`
		Tools          []ToolDefinition `json:"tools,omitempty"`
		ResponseFormat map[string]any   `json:"response_format,omitempty"`
	}
	type oaiChoice struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	}
	type oaiUsage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}
	type oaiResp struct {
		Choices []oaiChoice `json:"choices"`
		Usage   oaiUsage    `json:"usage"`
		Model   string      `json:"model"`
	}

	payload := oaiReq{
		Model:       p.Model,
		Messages:    messages,
		Temperature: opts.Temperature,
		MaxTokens:   opts.MaxTokens,
	}
	if len(opts.Tools) > 0 {
		payload.Tools = opts.Tools
	}
	if opts.ResponseFormat != nil {
		payload.ResponseFormat = opts.ResponseFormat
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return LLMResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return LLMResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" && p.APIKey != "ollama" {
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}

	t0 := time.Now()
	httpResp, err := p.client.Do(req)
	latencyMs := int(time.Since(t0).Milliseconds())
	if err != nil {
		return LLMResponse{}, fmt.Errorf("http request: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return LLMResponse{}, fmt.Errorf("read response: %w", err)
	}
	if httpResp.StatusCode >= 400 {
		return LLMResponse{}, fmt.Errorf("API error %d: %s", httpResp.StatusCode, respBody)
	}

	var parsed oaiResp
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return LLMResponse{}, fmt.Errorf("parse response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return LLMResponse{}, fmt.Errorf("empty choices in response")
	}

	msg := parsed.Choices[0].Message
	return LLMResponse{
		Content:      msg.Content,
		ToolCalls:    msg.ToolCalls,
		TokenUsage:   TokenUsage{PromptTokens: parsed.Usage.PromptTokens, CompletionTokens: parsed.Usage.CompletionTokens, TotalTokens: parsed.Usage.TotalTokens},
		Model:        p.Model,
		FinishReason: parsed.Choices[0].FinishReason,
		LatencyMs:    latencyMs,
	}, nil
}

func (p *OpenAIProvider) SupportsFunctionCalling() bool  { return true }
func (p *OpenAIProvider) SupportsStructuredOutput() bool { return true }
func (p *OpenAIProvider) GetContextWindow() int {
	if w, ok := openaiContextWindows[p.Model]; ok {
		return w
	}
	return 128_000
}
func (p *OpenAIProvider) GetModelName() string        { return p.Model }
func (p *OpenAIProvider) CountTokens(text string) int { return EstimateTokens(text) }

// ---------------------------------------------------------------------------
// Anthropic provider
// ---------------------------------------------------------------------------

// AnthropicProvider talks to the Anthropic Messages API.
type AnthropicProvider struct {
	APIKey string
	Model  string
	client *http.Client
}

// NewAnthropicProvider creates an AnthropicProvider.
func NewAnthropicProvider(apiKey, model string) *AnthropicProvider {
	return &AnthropicProvider{
		APIKey: apiKey, Model: model,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

// toAnthropicMessages converts OpenAI-format messages to Anthropic format.
func toAnthropicMessages(messages []Message) (system string, converted []map[string]any) {
	for _, m := range messages {
		switch m.Role {
		case "system":
			system = m.Content
		case "tool":
			converted = append(converted, map[string]any{
				"role": "user",
				"content": []map[string]any{{
					"type":        "tool_result",
					"tool_use_id": m.ToolCallID,
					"content":     m.Content,
				}},
			})
		default:
			converted = append(converted, map[string]any{
				"role":    m.Role,
				"content": m.Content,
			})
		}
	}
	return
}

// toAnthropicTools converts OpenAI tool definitions to Anthropic input_schema format.
func toAnthropicTools(tools []ToolDefinition) []map[string]any {
	out := make([]map[string]any, len(tools))
	for i, t := range tools {
		params := t.Function.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out[i] = map[string]any{
			"name":         t.Function.Name,
			"description":  t.Function.Description,
			"input_schema": params,
		}
	}
	return out
}

func (p *AnthropicProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (LLMResponse, error) {
	opts = defaultChatOptions(opts)

	system, converted := toAnthropicMessages(messages)

	payload := map[string]any{
		"model":       p.Model,
		"messages":    converted,
		"temperature": opts.Temperature,
		"max_tokens":  opts.MaxTokens,
	}
	if system != "" {
		payload["system"] = system
	}
	if len(opts.Tools) > 0 {
		payload["tools"] = toAnthropicTools(opts.Tools)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return LLMResponse{}, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		"https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return LLMResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	t0 := time.Now()
	httpResp, err := p.client.Do(req)
	latencyMs := int(time.Since(t0).Milliseconds())
	if err != nil {
		return LLMResponse{}, err
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return LLMResponse{}, err
	}
	if httpResp.StatusCode >= 400 {
		return LLMResponse{}, fmt.Errorf("Anthropic API error %d: %s", httpResp.StatusCode, respBody)
	}

	var parsed struct {
		Content []struct {
			Type  string         `json:"type"`
			Text  string         `json:"text,omitempty"`
			ID    string         `json:"id,omitempty"`
			Name  string         `json:"name,omitempty"`
			Input map[string]any `json:"input,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return LLMResponse{}, fmt.Errorf("parse Anthropic response: %w", err)
	}

	var contentParts []string
	var toolCalls []ToolCall
	for _, block := range parsed.Content {
		switch block.Type {
		case "text":
			contentParts = append(contentParts, block.Text)
		case "tool_use":
			argBytes, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:       block.ID,
				Type:     "function",
				Function: ToolCallFunction{Name: block.Name, Arguments: string(argBytes)},
			})
		}
	}

	return LLMResponse{
		Content:      strings.Join(contentParts, "\n"),
		ToolCalls:    toolCalls,
		TokenUsage:   TokenUsage{PromptTokens: parsed.Usage.InputTokens, CompletionTokens: parsed.Usage.OutputTokens, TotalTokens: parsed.Usage.InputTokens + parsed.Usage.OutputTokens},
		Model:        p.Model,
		FinishReason: parsed.StopReason,
		LatencyMs:    latencyMs,
	}, nil
}

func (p *AnthropicProvider) SupportsFunctionCalling() bool  { return true }
func (p *AnthropicProvider) SupportsStructuredOutput() bool { return false }
func (p *AnthropicProvider) GetContextWindow() int          { return 200_000 }
func (p *AnthropicProvider) GetModelName() string           { return p.Model }
func (p *AnthropicProvider) CountTokens(text string) int    { return EstimateTokens(text) }

// ---------------------------------------------------------------------------
// Ollama provider  (OpenAI-compatible local endpoint)
// ---------------------------------------------------------------------------

var ollamaToolModels = map[string]bool{
	"llama3.1": true, "llama3.1:8b": true, "llama3.1:70b": true,
	"llama3.2": true, "llama3.2:3b": true,
	"mistral": true, "mistral-nemo": true,
	"qwen2.5": true, "qwen2.5:7b": true,
}

// OllamaProvider uses the OpenAI-compatible endpoint that Ollama exposes.
type OllamaProvider struct {
	inner *OpenAIProvider
	model string
}

// NewOllamaProvider creates an OllamaProvider for the given model.
func NewOllamaProvider(model, baseURL string) *OllamaProvider {
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}
	inner := NewOpenAIProvider("ollama", model)
	inner.BaseURL = baseURL
	return &OllamaProvider{inner: inner, model: model}
}

func (p *OllamaProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (LLMResponse, error) {
	if !p.SupportsFunctionCalling() {
		opts.Tools = nil
	}
	return p.inner.Chat(ctx, messages, opts)
}

func (p *OllamaProvider) SupportsFunctionCalling() bool {
	base := strings.Split(p.model, ":")[0]
	return ollamaToolModels[base] || ollamaToolModels[p.model]
}
func (p *OllamaProvider) SupportsStructuredOutput() bool { return false }
func (p *OllamaProvider) GetContextWindow() int          { return 128_000 }
func (p *OllamaProvider) GetModelName() string           { return p.model }
func (p *OllamaProvider) CountTokens(text string) int    { return EstimateTokens(text) }

// ---------------------------------------------------------------------------
// FallbackProvider
// ---------------------------------------------------------------------------

// FallbackProvider tries providers in order, falling back on any error.
type FallbackProvider struct {
	Primary   LLMProvider
	Fallbacks []LLMProvider
}

func (p *FallbackProvider) Chat(ctx context.Context, messages []Message, opts ChatOptions) (LLMResponse, error) {
	candidates := append([]LLMProvider{p.Primary}, p.Fallbacks...)
	var lastErr error
	for _, provider := range candidates {
		resp, err := provider.Chat(ctx, messages, opts)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		log.Printf("WARN provider %s failed (%v) — trying next", provider.GetModelName(), err)
	}
	return LLMResponse{}, fmt.Errorf("all providers failed; last error: %w", lastErr)
}

func (p *FallbackProvider) SupportsFunctionCalling() bool { return p.Primary.SupportsFunctionCalling() }
func (p *FallbackProvider) SupportsStructuredOutput() bool {
	return p.Primary.SupportsStructuredOutput()
}
func (p *FallbackProvider) GetContextWindow() int { return p.Primary.GetContextWindow() }
func (p *FallbackProvider) GetModelName() string {
	names := []string{p.Primary.GetModelName()}
	for _, f := range p.Fallbacks {
		names = append(names, f.GetModelName())
	}
	return strings.Join(names, " → ")
}
func (p *FallbackProvider) CountTokens(text string) int { return p.Primary.CountTokens(text) }

// ---------------------------------------------------------------------------
// LLMFactory
// ---------------------------------------------------------------------------

// ProviderConfig is a simple map used by LLMFactory.CreateFromConfig.
type ProviderConfig map[string]string

// LLMFactory creates LLMProvider instances from configuration.
type LLMFactory struct{}

// Create builds a provider by name with the given options map.
//
//	opts["api_key"]  — API key (empty for Ollama)
//	opts["model"]    — model identifier
//	opts["base_url"] — override endpoint (optional)
func (LLMFactory) Create(provider string, opts ProviderConfig) (LLMProvider, error) {
	apiKey := opts["api_key"]
	model := opts["model"]

	switch provider {
	case "openai":
		if model == "" {
			model = "gpt-4o"
		}
		p := NewOpenAIProvider(apiKey, model)
		if bu := opts["base_url"]; bu != "" {
			p.BaseURL = bu
		}
		return p, nil
	case "anthropic":
		if model == "" {
			model = "claude-3-5-sonnet-20241022"
		}
		return NewAnthropicProvider(apiKey, model), nil
	case "ollama":
		if model == "" {
			model = "llama3.1:8b"
		}
		return NewOllamaProvider(model, opts["base_url"]), nil
	default:
		return nil, fmt.Errorf("unknown provider %q; available: openai, anthropic, ollama", provider)
	}
}

// CreateFromEnv builds a provider using the named environment variable as API key.
func (f LLMFactory) CreateFromEnv(provider, envKey, model string) (LLMProvider, error) {
	return f.Create(provider, ProviderConfig{
		"api_key": os.Getenv(envKey),
		"model":   model,
	})
}
