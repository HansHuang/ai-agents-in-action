// Package main implements a hybrid routing system for AI agent harnesses.
//
// Two-stage router:
//  1. DeterministicRouter — fast regex patterns, ~0 ms, no cost
//  2. LLMRouter           — gpt-4o-mini via OpenAI API, ~300 ms, tiny cost
//
// The HybridRouter combines both stages: deterministic first, LLM fallback.
// HandlerRegistry maps intents to handler functions + configs.
// EscalatingRouter adds automatic re-routing when a handler fails.
// RoutingEvaluator measures accuracy against labelled test cases.
//
// See: docs/07-harness-engineering/03-routing-and-intent-classification.md
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

// RouteResult is the classification decision from any router.
type RouteResult struct {
	Intent          string            `json:"intent"`
	Confidence      float64           `json:"confidence"`
	Method          string            `json:"method"` // "deterministic" | "llm" | "deterministic_fallback"
	Reasoning       string            `json:"reasoning,omitempty"`
	MatchedPattern  string            `json:"matched_pattern,omitempty"`
	ExtractedParams map[string]string `json:"extracted_params,omitempty"`
}

// HandlerConfig holds per-intent handler configuration.
type HandlerConfig struct {
	Model            string
	MaxTokens        int
	Temperature      float64
	TimeoutSeconds   int
	RequiresTools    bool
	RequiresRAG      bool
	RequiresApproval bool
	CostBudget       float64
}

// HandlerResponse is returned by every handler function.
type HandlerResponse struct {
	Content     string                 `json:"content"`
	HandlerUsed string                 `json:"handler_used"`
	TokensUsed  int                    `json:"tokens_used"`
	Cost        float64                `json:"cost"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// HandlerFunc is the signature for all route handlers.
type HandlerFunc func(ctx context.Context, userInput string, history []RouterMessage, cfg HandlerConfig) (*HandlerResponse, error)

// RouteHandler pairs a HandlerFunc with its configuration.
type RouteHandler struct {
	Handler HandlerFunc
	Config  HandlerConfig
}

// RouterMessage represents a single turn in a conversation.
type RouterMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// RoutingTestCase is a labelled test case for evaluating routing accuracy.
type RoutingTestCase struct {
	UserInput      string
	ExpectedIntent string
	Description    string
}

// EvaluationResult captures the outcome of a single test case.
type EvaluationResult struct {
	Input      string
	Expected   string
	Predicted  string
	Correct    bool
	Method     string
	Confidence float64
}

// RoutingReport aggregates evaluation outcomes.
type RoutingReport struct {
	OverallAccuracy        float64
	TotalCases             int
	ByIntent               map[string]float64
	TopMisclassifications  []MisclassEntry
	DeterministicRate      float64
	AvgConfidenceCorrect   float64
	AvgConfidenceIncorrect float64
}

// MisclassEntry describes a frequently occurring misclassification.
type MisclassEntry struct {
	Pattern string `json:"pattern"`
	Count   int    `json:"count"`
}

// ---------------------------------------------------------------------------
// DeterministicRouter
// ---------------------------------------------------------------------------

var defaultPatterns = map[string][]string{
	"greeting": {
		`(?i)^(hi|hello|hey|good morning|good evening|good afternoon|yo|sup)\b`,
		`(?i)^(how are you|how's it going|what's up|howdy)\b`,
		`(?i)^(nice to meet you|pleased to meet you)\b`,
	},
	"goodbye": {
		`(?i)\b(bye|goodbye|see you|talk later|farewell|ciao|later|ttyl)\b`,
		`(?i)\b(take care|have a good (day|night|one))\b`,
	},
	"thanks": {
		`(?i)\b(thanks|thank you|thx|ty|appreciate it|grateful|cheers)\b`,
		`(?i)\b(many thanks|much appreciated|that's helpful)\b`,
	},
	"reset": {
		`(?i)\b(start over|start fresh|reset|clear|new conversation|forget everything|fresh start)\b`,
		`(?i)\b(wipe (the slate|history)|begin again|restart)\b`,
	},
	"help": {
		`(?i)\b(what can you do|help me|capabilities|features|how do (I|you)|what do you (do|know))\b`,
		`(?i)^help$`,
	},
	"weather": {
		`(?i)\b(weather|temperature|forecast|humidity|raining?|sunny|cloudy|snowing?|wind)\b`,
		`(?i)\b(what's it like outside|will it (rain|snow))\b`,
	},
	"stock": {
		`(?i)\b(stock|market|ticker|nasdaq|dow jones|s&p|investment?|share price|equity)\b`,
		`(?i)\b(aapl|goog|msft|tsla|amzn)\b`,
	},
	"order_lookup": {
		`(?i)\b(order|tracking|shipment|delivery|where is my|status of)\b.{0,60}\b(order|package|item|number|parcel)\b`,
		`(?i)\border\s*#?\d+\b`,
		`(?i)\b(track|locate) my (package|order|shipment)\b`,
	},
	"return_request": {
		`(?i)\b(return|refund|exchange|money back|send back|cancel order|send it back)\b`,
		`(?i)\b(initiate a return|process a refund|want my money back)\b`,
	},
	"billing": {
		`(?i)\b(bill|invoice|charge|payment|subscription|receipt|pricing|cost|fee)\b`,
		`(?i)\b(overcharged|unauthorized charge|billing issue|payment failed)\b`,
	},
	"technical_support": {
		`(?i)\b(not working|broken|error|bug|crash|down|failed|issue|problem with)\b`,
		`(?i)\b(won't (load|open|start)|keeps (crashing|freezing)|can't (connect|access))\b`,
	},
	"account": {
		`(?i)\b(account|login|password|profile|settings|email change|sign in)\b`,
		`(?i)\b(forgot (my )?password|reset (my )?password|locked out|can't log in)\b`,
	},
}

// DeterministicRouter classifies requests using compiled regex patterns.
type DeterministicRouter struct {
	compiled map[string][]*regexp.Regexp
}

// NewDeterministicRouter returns a DeterministicRouter with the default pattern set.
// Pass a non-nil patterns map to override.
func NewDeterministicRouter(patterns map[string][]string) (*DeterministicRouter, error) {
	if patterns == nil {
		patterns = defaultPatterns
	}

	compiled := make(map[string][]*regexp.Regexp, len(patterns))
	for intent, pats := range patterns {
		regs := make([]*regexp.Regexp, 0, len(pats))
		for _, p := range pats {
			re, err := regexp.Compile(p)
			if err != nil {
				return nil, fmt.Errorf("invalid pattern for %q: %w", intent, err)
			}
			regs = append(regs, re)
		}
		compiled[intent] = regs
	}

	return &DeterministicRouter{compiled: compiled}, nil
}

// Classify tries to match userInput against known patterns.
// Returns nil when no pattern matches (defer to LLM).
func (r *DeterministicRouter) Classify(userInput string) *RouteResult {
	type match struct {
		intent  string
		pattern *regexp.Regexp
	}

	var matches []match

	for intent, patterns := range r.compiled {
		for _, re := range patterns {
			if re.MatchString(userInput) {
				matches = append(matches, match{intent, re})
			}
		}
	}

	if len(matches) == 0 {
		return nil
	}

	// Longest pattern source = most specific
	best := matches[0]
	for _, m := range matches[1:] {
		if len(m.pattern.String()) > len(best.pattern.String()) {
			best = m
		}
	}

	confidence := 0.85
	if len(matches) > 1 {
		confidence = 0.65
	}

	slog.Info("deterministic_classify",
		"intent", best.intent,
		"confidence", confidence,
		"total_matches", len(matches),
	)

	return &RouteResult{
		Intent:         best.intent,
		Confidence:     confidence,
		Method:         "deterministic",
		MatchedPattern: best.pattern.String(),
	}
}

// ---------------------------------------------------------------------------
// LLMRouter
// ---------------------------------------------------------------------------

const routingPrompt = `You are a request classifier for a customer-facing AI assistant.
Analyze the user's message and determine its primary intent.

Available routes:
- simple_chat: Casual conversation, greetings, general questions not requiring tools
- knowledge_question: Questions answerable from a knowledge base (policies, docs, FAQs)
- agent_task: Requests requiring tool use (lookups, calculations, multi-step tasks)
- human_escalation: User explicitly asks for a human, or the request is too complex/sensitive
- support_request: Customer support issues, complaints, problems with products or services
- out_of_scope: Requests that cannot or should not be handled

Output ONLY a JSON object:
{
    "intent": "<intent_name>",
    "confidence": 0.0,
    "reasoning": "<one sentence>",
    "extracted_params": {"order_number": null, "product_name": null, "issue_type": null}
}`

// llmResponse is the shape returned by the OpenAI API.
type llmResponse struct {
	Choices []struct {
		RouterMessage struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// llmRouteJSON is the shape we expect inside the LLM's response content.
type llmRouteJSON struct {
	Intent          string            `json:"intent"`
	Confidence      float64           `json:"confidence"`
	Reasoning       string            `json:"reasoning"`
	ExtractedParams map[string]string `json:"extracted_params"`
}

// LLMRouter classifies ambiguous requests using a cheap LLM.
type LLMRouter struct {
	Model   string
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewLLMRouter returns an LLMRouter configured to use the OpenAI API.
func NewLLMRouter(model string) *LLMRouter {
	if model == "" {
		model = "gpt-4o-mini"
	}
	return &LLMRouter{
		Model:   model,
		apiKey:  os.Getenv("OPENAI_API_KEY"),
		baseURL: "https://api.openai.com/v1/chat/completions",
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// Classify sends userInput to the LLM and returns a RouteResult.
func (r *LLMRouter) Classify(ctx context.Context, userInput string, history []RouterMessage) (*RouteResult, error) {
	userContent := userInput
	if len(history) > 0 {
		recent := history
		if len(recent) > 4 {
			recent = recent[len(recent)-4:]
		}
		var sb strings.Builder
		for _, m := range recent {
			role := "User"
			if m.Role != "user" {
				role = "Assistant"
			}
			content := m.Content
			if len(content) > 200 {
				content = content[:200]
			}
			fmt.Fprintf(&sb, "%s: %s\n", role, content)
		}
		userContent = fmt.Sprintf("Recent conversation:\n%s\nClassify this message: %s", sb.String(), userInput)
	}

	reqBody := map[string]interface{}{
		"model": r.Model,
		"messages": []map[string]string{
			{"role": "system", "content": routingPrompt},
			{"role": "user", "content": userContent},
		},
		"response_format": map[string]string{"type": "json_object"},
		"temperature":     0.1,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, r.baseURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return r.fallback(err), nil
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+r.apiKey)

	resp, err := r.client.Do(httpReq)
	if err != nil {
		return r.fallback(err), nil
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return r.fallback(err), nil
	}

	var llmResp llmResponse
	if err := json.Unmarshal(raw, &llmResp); err != nil || len(llmResp.Choices) == 0 {
		return r.fallback(fmt.Errorf("parse error")), nil
	}

	var route llmRouteJSON
	if err := json.Unmarshal([]byte(llmResp.Choices[0].RouterMessage.Content), &route); err != nil {
		return r.fallback(err), nil
	}

	slog.Info("llm_classify", "intent", route.Intent, "confidence", route.Confidence)

	return &RouteResult{
		Intent:          route.Intent,
		Confidence:      route.Confidence,
		Method:          "llm",
		Reasoning:       route.Reasoning,
		ExtractedParams: route.ExtractedParams,
	}, nil
}

func (r *LLMRouter) fallback(err error) *RouteResult {
	slog.Warn("llm_classify_error", "error", err.Error())
	return &RouteResult{
		Intent:     "simple_chat",
		Confidence: 0.3,
		Method:     "llm",
		Reasoning:  fmt.Sprintf("Classification failed (%v); defaulting to simple_chat", err),
	}
}

// ---------------------------------------------------------------------------
// RouterMetrics
// ---------------------------------------------------------------------------

// RouterMetrics collects routing statistics in a concurrency-safe way.
type RouterMetrics struct {
	mu          sync.RWMutex
	total       int
	byMethod    map[string]int
	byIntent    map[string]int
	latenciesMs []float64
	llmCosts    []float64
}

// NewRouterMetrics returns an initialised RouterMetrics.
func NewRouterMetrics() *RouterMetrics {
	return &RouterMetrics{
		byMethod: make(map[string]int),
		byIntent: make(map[string]int),
	}
}

// Record adds a routing event to the metrics store.
func (m *RouterMetrics) Record(method, intent string, latencyMs, cost float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.total++
	m.byMethod[method]++
	m.byIntent[intent]++
	m.latenciesMs = append(m.latenciesMs, latencyMs)
	m.llmCosts = append(m.llmCosts, cost)
}

// Summary returns a snapshot of current metrics.
func (m *RouterMetrics) Summary() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	n := math.Max(float64(m.total), 1)
	avg, p50, p95 := latencyStats(m.latenciesMs)
	totalCost := 0.0
	for _, c := range m.llmCosts {
		totalCost += c
	}

	return map[string]interface{}{
		"total_routed":        m.total,
		"deterministic_rate":  float64(m.byMethod["deterministic"]) / n,
		"llm_fallback_rate":   float64(m.byMethod["llm"]) / n,
		"intent_distribution": m.byIntent,
		"avg_latency_ms":      avg,
		"p50_latency_ms":      p50,
		"p95_latency_ms":      p95,
		"total_llm_cost_usd":  totalCost,
	}
}

func latencyStats(lats []float64) (avg, p50, p95 float64) {
	if len(lats) == 0 {
		return 0, 0, 0
	}
	sorted := make([]float64, len(lats))
	copy(sorted, lats)
	// simple insertion sort for small slices
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	avg = sum / float64(len(sorted))
	p50 = sorted[len(sorted)/2]
	p95 = sorted[int(float64(len(sorted))*0.95)]
	return
}

// ---------------------------------------------------------------------------
// HybridRouter
// ---------------------------------------------------------------------------

// HybridRouter combines deterministic and LLM-based classification.
type HybridRouter struct {
	Deterministic *DeterministicRouter
	LLM           *LLMRouter
	Metrics       *RouterMetrics
}

// NewHybridRouter constructs a HybridRouter with default components.
func NewHybridRouter() (*HybridRouter, error) {
	det, err := NewDeterministicRouter(nil)
	if err != nil {
		return nil, fmt.Errorf("deterministic router: %w", err)
	}
	return &HybridRouter{
		Deterministic: det,
		LLM:           NewLLMRouter(""),
		Metrics:       NewRouterMetrics(),
	}, nil
}

// Route classifies userInput. Deterministic first, LLM fallback.
func (r *HybridRouter) Route(ctx context.Context, userInput string, history []RouterMessage) (*RouteResult, error) {
	t0 := time.Now()

	detResult := r.Deterministic.Classify(userInput)

	if detResult != nil && detResult.Confidence > 0.8 {
		latency := float64(time.Since(t0).Milliseconds())
		r.Metrics.Record("deterministic", detResult.Intent, latency, 0)
		slog.Info("route_decision", "method", "deterministic", "intent", detResult.Intent)
		return detResult, nil
	}

	llmResult, err := r.LLM.Classify(ctx, userInput, history)
	if err != nil {
		return nil, err
	}
	latency := float64(time.Since(t0).Milliseconds())

	if detResult != nil && detResult.Confidence > llmResult.Confidence {
		r.Metrics.Record("deterministic_fallback", detResult.Intent, latency, 0)
		result := *detResult
		result.Method = "deterministic_fallback"
		return &result, nil
	}

	r.Metrics.Record("llm", llmResult.Intent, latency, 0)
	slog.Info("route_decision", "method", "llm", "intent", llmResult.Intent)
	return llmResult, nil
}

// GetMetrics returns a summary of routing statistics.
func (r *HybridRouter) GetMetrics() map[string]interface{} {
	return r.Metrics.Summary()
}

// ---------------------------------------------------------------------------
// HandlerRegistry
// ---------------------------------------------------------------------------

// HandlerRegistry maps intents to handler functions and their configurations.
type HandlerRegistry struct {
	mu            sync.RWMutex
	handlers      map[string]*RouteHandler
	defaultIntent string
}

// NewHandlerRegistry returns an empty HandlerRegistry.
func NewHandlerRegistry() *HandlerRegistry {
	return &HandlerRegistry{
		handlers:      make(map[string]*RouteHandler),
		defaultIntent: "simple_chat",
	}
}

// Register adds a handler for the given intent.
func (reg *HandlerRegistry) Register(intent string, fn HandlerFunc, cfg HandlerConfig) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.handlers[intent] = &RouteHandler{Handler: fn, Config: cfg}
}

// GetHandler returns the handler for intent, falling back to the default.
func (reg *HandlerRegistry) GetHandler(intent string) *RouteHandler {
	reg.mu.RLock()
	defer reg.mu.RUnlock()
	if h, ok := reg.handlers[intent]; ok {
		return h
	}
	slog.Warn("handler_fallback", "unknown_intent", intent, "using", reg.defaultIntent)
	return reg.handlers[reg.defaultIntent]
}

// SetDefaultIntent changes the fallback intent for unknown routes.
func (reg *HandlerRegistry) SetDefaultIntent(intent string) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.defaultIntent = intent
}

// ---------------------------------------------------------------------------
// EscalatingRouter
// ---------------------------------------------------------------------------

var escalationPaths = map[string][]string{
	"simple_chat":        {"knowledge_question"},
	"knowledge_question": {"agent_task"},
	"agent_task":         {"human_escalation"},
	"support_request":    {"human_escalation"},
	"human_escalation":   {},
}

var uncertaintyPhrases = []string{
	"i'm not sure",
	"i don't know",
	"i cannot",
	"i'm unable",
	"i don't have enough information",
	"i have no information",
	"i couldn't find",
}

// EscalatingRouter wraps HybridRouter and HandlerRegistry with automatic
// re-routing when a handler fails or returns a low-quality response.
type EscalatingRouter struct {
	Router   *HybridRouter
	Registry *HandlerRegistry
}

// NewEscalatingRouter constructs an EscalatingRouter.
func NewEscalatingRouter(router *HybridRouter, registry *HandlerRegistry) *EscalatingRouter {
	return &EscalatingRouter{Router: router, Registry: registry}
}

// Handle routes and handles userInput, escalating automatically on failure.
func (e *EscalatingRouter) Handle(ctx context.Context, userInput string, history []RouterMessage) (*HandlerResponse, error) {
	intent, err := e.Router.Route(ctx, userInput, history)
	if err != nil {
		return nil, fmt.Errorf("routing failed: %w", err)
	}

	rh := e.Registry.GetHandler(intent.Intent)
	resp, err := rh.Handler(ctx, userInput, history, rh.Config)
	if err != nil {
		return e.escalate(ctx, intent.Intent, userInput, history, nil, err.Error())
	}

	if e.shouldEscalate(resp) {
		return e.escalate(ctx, intent.Intent, userInput, history, resp, "")
	}

	return resp, nil
}

func (e *EscalatingRouter) shouldEscalate(resp *HandlerResponse) bool {
	if resp.Metadata != nil {
		if n, ok := resp.Metadata["documents_found"].(int); ok && n == 0 {
			return true
		}
		if iter, ok := resp.Metadata["iterations"].(int); ok && iter >= 10 {
			return true
		}
	}
	lower := strings.ToLower(resp.Content)
	for _, phrase := range uncertaintyPhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

func (e *EscalatingRouter) escalate(
	ctx context.Context,
	originalIntent, userInput string,
	history []RouterMessage,
	prev *HandlerResponse,
	errMsg string,
) (*HandlerResponse, error) {
	path := escalationPaths[originalIntent]
	if path == nil {
		path = []string{"human_escalation"}
	}

	for _, nextIntent := range path {
		rh := e.Registry.GetHandler(nextIntent)

		augmented := userInput
		if prev != nil {
			short := prev.Content
			if len(short) > 200 {
				short = short[:200]
			}
			augmented = fmt.Sprintf(
				"[Previous attempt via '%s' was insufficient. Response: '%s...']\n\nOriginal request: %s",
				originalIntent, short, userInput,
			)
		}

		resp, err := rh.Handler(ctx, augmented, history, rh.Config)
		if err != nil {
			slog.Warn("escalation_handler_error", "intent", nextIntent, "error", err.Error())
			continue
		}

		if resp.Metadata == nil {
			resp.Metadata = make(map[string]interface{})
		}
		resp.Metadata["escalated_from"] = originalIntent
		reason := errMsg
		if reason == "" {
			reason = "low_confidence"
		}
		resp.Metadata["escalation_reason"] = reason
		return resp, nil
	}

	return &HandlerResponse{
		Content:     "I apologize — I'm having trouble processing your request. A human team member will follow up shortly.",
		HandlerUsed: "escalation_fallback",
		Metadata:    map[string]interface{}{"escalation_chain_exhausted": true},
	}, nil
}

// ---------------------------------------------------------------------------
// RoutingEvaluator
// ---------------------------------------------------------------------------

// RoutingEvaluator measures routing accuracy against labelled test cases.
type RoutingEvaluator struct {
	Router *HybridRouter
}

// Evaluate runs all test cases through the router and returns a RoutingReport.
func (ev *RoutingEvaluator) Evaluate(ctx context.Context, cases []RoutingTestCase) (*RoutingReport, error) {
	results := make([]EvaluationResult, 0, len(cases))

	for _, tc := range cases {
		route, err := ev.Router.Route(ctx, tc.UserInput, nil)
		if err != nil {
			return nil, err
		}
		correct := route.Intent == tc.ExpectedIntent
		results = append(results, EvaluationResult{
			Input:      tc.UserInput,
			Expected:   tc.ExpectedIntent,
			Predicted:  route.Intent,
			Correct:    correct,
			Method:     route.Method,
			Confidence: route.Confidence,
		})
	}

	return generateReport(results), nil
}

func generateReport(results []EvaluationResult) *RoutingReport {
	total := len(results)
	correctCount := 0
	byIntentRaw := make(map[string][2]int) // [correct, total]
	misclass := make(map[string]int)

	for _, r := range results {
		if r.Correct {
			correctCount++
		}
		prev := byIntentRaw[r.Expected]
		prev[1]++
		if r.Correct {
			prev[0]++
		}
		byIntentRaw[r.Expected] = prev

		if !r.Correct {
			key := r.Expected + " → " + r.Predicted
			misclass[key]++
		}
	}

	byIntent := make(map[string]float64, len(byIntentRaw))
	for intent, counts := range byIntentRaw {
		byIntent[intent] = float64(counts[0]) / math.Max(float64(counts[1]), 1)
	}

	topMisclass := make([]MisclassEntry, 0, len(misclass))
	for pattern, count := range misclass {
		topMisclass = append(topMisclass, MisclassEntry{Pattern: pattern, Count: count})
	}
	// Sort descending by count (simple selection sort for small slices)
	for i := 0; i < len(topMisclass); i++ {
		for j := i + 1; j < len(topMisclass); j++ {
			if topMisclass[j].Count > topMisclass[i].Count {
				topMisclass[i], topMisclass[j] = topMisclass[j], topMisclass[i]
			}
		}
	}
	if len(topMisclass) > 10 {
		topMisclass = topMisclass[:10]
	}

	detCount := 0
	var correctConfs, incorrectConfs []float64
	for _, r := range results {
		if r.Method == "deterministic" {
			detCount++
		}
		if r.Correct {
			correctConfs = append(correctConfs, r.Confidence)
		} else {
			incorrectConfs = append(incorrectConfs, r.Confidence)
		}
	}

	avgConf := func(confs []float64) float64 {
		if len(confs) == 0 {
			return 0
		}
		sum := 0.0
		for _, c := range confs {
			sum += c
		}
		return sum / float64(len(confs))
	}

	return &RoutingReport{
		OverallAccuracy:        float64(correctCount) / math.Max(float64(total), 1),
		TotalCases:             total,
		ByIntent:               byIntent,
		TopMisclassifications:  topMisclass,
		DeterministicRate:      float64(detCount) / math.Max(float64(total), 1),
		AvgConfidenceCorrect:   avgConf(correctConfs),
		AvgConfidenceIncorrect: avgConf(incorrectConfs),
	}
}
