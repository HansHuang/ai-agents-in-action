// Agent Observability System — Go port.
//
// Implements the three pillars of agent observability:
//   - Tracing:  the full decision tree of a single request
//   - Logging:  structured JSON events (using log/slog, stdlib Go 1.21+)
//   - Metrics:  rolling aggregates behind sync.RWMutex for safe concurrency
//
// Additional components:
//   - TokenAccountant  – per-user/session/model cost tracking
//   - DecisionTracer   – reasoning capture for debugging
//
// See: docs/05-the-tool-ecosystem/03-agent-observability.md
package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ---------------------------------------------------------------------------
// Pricing defaults (USD per 1 000 tokens, input / output)
// ---------------------------------------------------------------------------

// ModelPricing holds per-1000-token pricing for one model.
type ModelPricing struct {
	Input  float64
	Output float64
}

// DefaultPricing maps model names to their USD pricing.
var DefaultPricing = map[string]ModelPricing{
	"gpt-4o":            {Input: 0.0025, Output: 0.01},
	"gpt-4o-mini":       {Input: 0.00015, Output: 0.0006},
	"claude-3-5-sonnet": {Input: 0.003, Output: 0.015},
	"claude-3-haiku":    {Input: 0.00025, Output: 0.00125},
	"gemini-1.5-pro":    {Input: 0.00125, Output: 0.005},
	"gemini-1.5-flash":  {Input: 0.000075, Output: 0.0003},
	"unknown":           {Input: 0.001, Output: 0.003},
}

func computeCost(model string, inputTokens, outputTokens int,
	pricing map[string]ModelPricing) float64 {
	p, ok := pricing[model]
	if !ok {
		p = pricing["unknown"]
	}
	return (float64(inputTokens)*p.Input + float64(outputTokens)*p.Output) / 1000
}

// ===========================================================================
// Core data model: Span and Trace
// ===========================================================================

// Span represents one operation within a trace.
type Span struct {
	mu sync.Mutex

	SpanID       string                 `json:"span_id"`
	ParentSpanID string                 `json:"parent_span_id,omitempty"`
	Type         string                 `json:"type"` // llm_call | tool_call | planning | execution
	Name         string                 `json:"name"`
	StartTime    time.Time              `json:"start_time"`
	EndTime      *time.Time             `json:"end_time,omitempty"`
	InputData    map[string]interface{} `json:"input_data,omitempty"`
	OutputData   map[string]interface{} `json:"output_data,omitempty"`
	InputTokens  int                    `json:"input_tokens"`
	OutputTokens int                    `json:"output_tokens"`
	TokensUsed   int                    `json:"tokens_used"`
	Cost         float64                `json:"cost"`
	Model        string                 `json:"model,omitempty"`
	Status       string                 `json:"status"` // running | success | error
	ErrorMessage string                 `json:"error_message,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	DurationMS   float64                `json:"duration_ms"`
}

// NewSpan creates a running span.
func NewSpan(spanType, name, parentID string) *Span {
	return &Span{
		SpanID:       uuid.New().String(),
		ParentSpanID: parentID,
		Type:         spanType,
		Name:         name,
		StartTime:    time.Now(),
		Status:       "running",
		Metadata:     map[string]interface{}{},
	}
}

// Finish marks the span as complete and records duration.
func (s *Span) Finish(outputData map[string]interface{}, status, errMsg string) *Span {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	s.EndTime = &now
	s.OutputData = outputData
	if status == "" {
		status = "success"
	}
	s.Status = status
	s.ErrorMessage = errMsg
	s.DurationMS = float64(now.Sub(s.StartTime).Milliseconds())
	return s
}

// Duration returns span latency in milliseconds.
func (s *Span) Duration() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.EndTime != nil {
		return float64(s.EndTime.Sub(s.StartTime).Milliseconds())
	}
	return float64(time.Since(s.StartTime).Milliseconds())
}

// Trace represents the complete record of one agent request.
type Trace struct {
	mu sync.Mutex

	TraceID   string                 `json:"trace_id"`
	UserQuery string                 `json:"user_query"`
	UserID    string                 `json:"user_id,omitempty"`
	SessionID string                 `json:"session_id,omitempty"`
	Spans     []*Span                `json:"spans"`
	StartTime time.Time              `json:"start_time"`
	EndTime   *time.Time             `json:"end_time,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// NewTrace creates and returns a new Trace.
func NewTrace(userQuery, userID, sessionID string) *Trace {
	return &Trace{
		TraceID:   uuid.New().String(),
		UserQuery: userQuery,
		UserID:    userID,
		SessionID: sessionID,
		Spans:     []*Span{},
		StartTime: time.Now(),
		Metadata:  map[string]interface{}{},
	}
}

// NewSpan appends and returns a new Span on this Trace.
func (t *Trace) NewSpan(spanType, name string, parent *Span) *Span {
	parentID := ""
	if parent != nil {
		parentID = parent.SpanID
	}
	span := NewSpan(spanType, name, parentID)
	t.mu.Lock()
	t.Spans = append(t.Spans, span)
	t.mu.Unlock()
	return span
}

// Finish records end time on the trace.
func (t *Trace) Finish() *Trace {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now()
	t.EndTime = &now
	return t
}

func (t *Trace) durationMs() float64 {
	if t.EndTime != nil {
		return float64(t.EndTime.Sub(t.StartTime).Milliseconds())
	}
	return float64(time.Since(t.StartTime).Milliseconds())
}

func (t *Trace) totalTokens() int {
	n := 0
	for _, s := range t.Spans {
		n += s.TokensUsed
	}
	return n
}

func (t *Trace) totalCost() float64 {
	c := 0.0
	for _, s := range t.Spans {
		c += s.Cost
	}
	return c
}

func (t *Trace) llmCallCount() int {
	n := 0
	for _, s := range t.Spans {
		if s.Type == "llm_call" {
			n++
		}
	}
	return n
}

func (t *Trace) toolCallCount() int {
	n := 0
	for _, s := range t.Spans {
		if s.Type == "tool_call" {
			n++
		}
	}
	return n
}

func (t *Trace) hasError() bool {
	for _, s := range t.Spans {
		if s.Status == "error" {
			return true
		}
	}
	return false
}

func (t *Trace) status() string {
	if t.hasError() {
		return "error"
	}
	return "success"
}

// ToMap returns a map representation suitable for JSON marshalling.
func (t *Trace) ToMap() map[string]interface{} {
	spans := make([]interface{}, len(t.Spans))
	for i, s := range t.Spans {
		spans[i] = s
	}
	return map[string]interface{}{
		"trace_id":     t.TraceID,
		"user_query":   t.UserQuery,
		"user_id":      t.UserID,
		"session_id":   t.SessionID,
		"status":       t.status(),
		"duration_ms":  math.Round(t.durationMs()),
		"llm_calls":    t.llmCallCount(),
		"tool_calls":   t.toolCallCount(),
		"total_tokens": t.totalTokens(),
		"total_cost":   math.Round(t.totalCost()*1e6) / 1e6,
		"spans":        spans,
		"metadata":     t.Metadata,
	}
}

// ===========================================================================
// TraceExporter interface
// ===========================================================================

// TraceExporter is the interface implemented by all exporters.
type TraceExporter interface {
	Export(trace *Trace)
}

// ConsoleExporter writes a formatted tree view to stdout.
type ConsoleExporter struct{}

func (e *ConsoleExporter) Export(trace *Trace) {
	icon := "✓"
	if trace.hasError() {
		icon = "✗"
	}
	border := strings.Repeat("═", 56)
	fmt.Printf("\n%s\n", border)
	query := trace.UserQuery
	if len(query) > 55 {
		query = query[:55]
	}
	fmt.Printf("%s Trace %s  query='%s'\n", icon, trace.TraceID[:8], query)
	fmt.Printf("  duration=%.0fms  tokens=%d  cost=$%.4f\n",
		trace.durationMs(), trace.totalTokens(), trace.totalCost())
	fmt.Printf("  llm_calls=%d  tool_calls=%d  status=%s\n\n",
		trace.llmCallCount(), trace.toolCallCount(), trace.status())

	for _, span := range trace.Spans {
		indent := "  "
		if span.ParentSpanID != "" {
			indent = "    "
		}
		sIcon := "·"
		if span.Status == "error" {
			sIcon = "✗"
		}
		errStr := ""
		if span.ErrorMessage != "" {
			errStr = "  ERROR: " + span.ErrorMessage
		}
		fmt.Printf("%s%s [%-10s] %-22s %6.0fms  tokens=%d%s\n",
			indent, sIcon, span.Type, span.Name,
			span.Duration(), span.TokensUsed, errStr)
	}
	fmt.Printf("%s\n\n", border)
}

// ===========================================================================
// TraceCollector
// ===========================================================================

// TraceCollector manages the lifecycle of traces.
type TraceCollector struct {
	mu       sync.RWMutex
	exporter TraceExporter
	traces   map[string]*Trace
}

// NewTraceCollector returns a TraceCollector with the given exporter.
func NewTraceCollector(exporter TraceExporter) *TraceCollector {
	if exporter == nil {
		exporter = &ConsoleExporter{}
	}
	return &TraceCollector{
		exporter: exporter,
		traces:   map[string]*Trace{},
	}
}

// StartTrace creates and stores a new Trace.
func (c *TraceCollector) StartTrace(userQuery, userID, sessionID string) *Trace {
	t := NewTrace(userQuery, userID, sessionID)
	c.mu.Lock()
	c.traces[t.TraceID] = t
	c.mu.Unlock()
	return t
}

// EndTrace finalises and exports a Trace.
func (c *TraceCollector) EndTrace(trace *Trace) {
	if trace.EndTime == nil {
		trace.Finish()
	}
	c.exporter.Export(trace)
}

// GetTrace retrieves a Trace by ID. Returns nil if not found.
func (c *TraceCollector) GetTrace(traceID string) *Trace {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.traces[traceID]
}

// QueryTraces filters stored traces.
func (c *TraceCollector) QueryTraces(userID, status string,
	since time.Time, limit int) []*Trace {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var results []*Trace
	for _, t := range c.traces {
		if userID != "" && t.UserID != userID {
			continue
		}
		if status != "" && t.status() != status {
			continue
		}
		if !since.IsZero() && t.StartTime.Before(since) {
			continue
		}
		results = append(results, t)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].StartTime.After(results[j].StartTime)
	})
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results
}

// ===========================================================================
// AgentMetrics
// ===========================================================================

// MetricsSummary is a snapshot of rolling statistics.
type MetricsSummary struct {
	Requests     int     `json:"requests"`
	AvgLatencyMs float64 `json:"avg_latency_ms"`
	P95LatencyMs float64 `json:"p95_latency_ms"`
	P99LatencyMs float64 `json:"p99_latency_ms"`
	AvgTokens    float64 `json:"avg_tokens"`
	AvgCost      float64 `json:"avg_cost"`
	TotalCost    float64 `json:"total_cost"`
	ErrorRatePct float64 `json:"error_rate_pct"`
	AvgLLMCalls  float64 `json:"avg_llm_calls"`
	AvgToolCalls float64 `json:"avg_tool_calls"`
}

// Alert signals an anomaly detected by AgentMetrics.
type Alert struct {
	Kind      string  `json:"kind"`
	Message   string  `json:"message"`
	Value     float64 `json:"value"`
	Threshold float64 `json:"threshold"`
	Severity  string  `json:"severity"`
}

func (a Alert) String() string {
	return fmt.Sprintf("[%s] %s", strings.ToUpper(a.Severity), a.Message)
}

// ringBuffer is a fixed-capacity FIFO of float64 values.
type ringBuffer struct {
	mu   sync.Mutex
	data []float64
	cap  int
}

func newRingBuffer(cap int) *ringBuffer { return &ringBuffer{cap: cap} }

func (rb *ringBuffer) push(v float64) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	rb.data = append(rb.data, v)
	if len(rb.data) > rb.cap {
		rb.data = rb.data[len(rb.data)-rb.cap:]
	}
}

func (rb *ringBuffer) snapshot() []float64 {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	cp := make([]float64, len(rb.data))
	copy(cp, rb.data)
	return cp
}

func average(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func pctile(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	idx := int(float64(len(sorted)) * p / 100)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// AgentMetrics collects rolling performance metrics.
type AgentMetrics struct {
	totalLatency *ringBuffer
	llmLatency   *ringBuffer
	toolLatency  *ringBuffer
	tokens       *ringBuffer
	costs        *ringBuffer
	llmCalls     *ringBuffer
	toolCalls    *ringBuffer
	errors       *ringBuffer
}

// NewAgentMetrics returns an AgentMetrics with the given window size.
func NewAgentMetrics(windowSize int) *AgentMetrics {
	return &AgentMetrics{
		totalLatency: newRingBuffer(windowSize),
		llmLatency:   newRingBuffer(windowSize),
		toolLatency:  newRingBuffer(windowSize),
		tokens:       newRingBuffer(windowSize),
		costs:        newRingBuffer(windowSize),
		llmCalls:     newRingBuffer(windowSize),
		toolCalls:    newRingBuffer(windowSize),
		errors:       newRingBuffer(windowSize),
	}
}

// Record ingests all metrics from a completed Trace.
func (m *AgentMetrics) Record(trace *Trace) {
	m.totalLatency.push(trace.durationMs())
	m.tokens.push(float64(trace.totalTokens()))
	m.costs.push(trace.totalCost())
	m.llmCalls.push(float64(trace.llmCallCount()))
	m.toolCalls.push(float64(trace.toolCallCount()))
	errVal := 0.0
	if trace.hasError() {
		errVal = 1.0
	}
	m.errors.push(errVal)

	for _, s := range trace.Spans {
		if s.Type == "llm_call" && s.EndTime != nil {
			m.llmLatency.push(s.Duration())
		}
		if s.Type == "tool_call" && s.EndTime != nil {
			m.toolLatency.push(s.Duration())
		}
	}
}

// GetSummary returns a MetricsSummary over all recorded data.
func (m *AgentMetrics) GetSummary() MetricsSummary {
	lat := m.totalLatency.snapshot()
	tok := m.tokens.snapshot()
	cost := m.costs.snapshot()
	err := m.errors.snapshot()
	llm := m.llmCalls.snapshot()
	tool := m.toolCalls.snapshot()

	totalCost := 0.0
	for _, v := range cost {
		totalCost += v
	}
	return MetricsSummary{
		Requests:     len(lat),
		AvgLatencyMs: math.Round(average(lat)*10) / 10,
		P95LatencyMs: math.Round(pctile(lat, 95)*10) / 10,
		P99LatencyMs: math.Round(pctile(lat, 99)*10) / 10,
		AvgTokens:    math.Round(average(tok)*10) / 10,
		AvgCost:      math.Round(average(cost)*1e4) / 1e4,
		TotalCost:    math.Round(totalCost*1e4) / 1e4,
		ErrorRatePct: math.Round(average(err)*1e4) / 100,
		AvgLLMCalls:  math.Round(average(llm)*100) / 100,
		AvgToolCalls: math.Round(average(tool)*100) / 100,
	}
}

// DetectAnomalies returns a list of triggered alerts.
func (m *AgentMetrics) DetectAnomalies() []Alert {
	s := m.GetSummary()
	var alerts []Alert

	if s.ErrorRatePct > 5 {
		sev := "warning"
		if s.ErrorRatePct > 20 {
			sev = "critical"
		}
		alerts = append(alerts, Alert{
			Kind:      "error_rate",
			Message:   fmt.Sprintf("High error rate: %.1f%%", s.ErrorRatePct),
			Value:     s.ErrorRatePct,
			Threshold: 5,
			Severity:  sev,
		})
	}

	if s.Requests >= 10 && s.AvgLatencyMs > 0 &&
		s.P95LatencyMs > 2*s.AvgLatencyMs {
		alerts = append(alerts, Alert{
			Kind:      "latency",
			Message:   fmt.Sprintf("P95 %.0fms > 2× avg %.0fms", s.P95LatencyMs, s.AvgLatencyMs),
			Value:     s.P95LatencyMs,
			Threshold: 2 * s.AvgLatencyMs,
			Severity:  "warning",
		})
	}

	if s.AvgCost > 0.50 {
		sev := "warning"
		if s.AvgCost > 2.0 {
			sev = "critical"
		}
		alerts = append(alerts, Alert{
			Kind:      "cost",
			Message:   fmt.Sprintf("High cost per request: $%.2f", s.AvgCost),
			Value:     s.AvgCost,
			Threshold: 0.50,
			Severity:  sev,
		})
	}

	return alerts
}

// ExportPrometheus returns metrics in Prometheus text format.
func (m *AgentMetrics) ExportPrometheus() string {
	s := m.GetSummary()
	return fmt.Sprintf(
		"agent_requests_total %d\n"+
			"agent_latency_avg_ms %.2f\n"+
			"agent_latency_p95_ms %.2f\n"+
			"agent_error_rate_pct %.2f\n"+
			"agent_avg_tokens_per_request %.1f\n"+
			"agent_avg_cost_per_request %.6f\n"+
			"agent_total_cost_usd %.6f\n",
		s.Requests, s.AvgLatencyMs, s.P95LatencyMs,
		s.ErrorRatePct, s.AvgTokens, s.AvgCost, s.TotalCost,
	)
}

// ===========================================================================
// AgentLogger — structured JSON via slog
// ===========================================================================

var redactedKeys = map[string]bool{
	"api_key": true, "apikey": true, "secret": true, "password": true,
	"token": true, "authorization": true, "credential": true, "private_key": true,
}

// AgentLogger emits structured JSON logs via slog.
type AgentLogger struct {
	logger *slog.Logger
}

// NewAgentLogger creates a JSON-formatted slog logger.
func NewAgentLogger(level slog.Level) *AgentLogger {
	opts := &slog.HandlerOptions{Level: level}
	handler := slog.NewJSONHandler(os.Stdout, opts)
	return &AgentLogger{logger: slog.New(handler)}
}

func (l *AgentLogger) LogLLMCall(traceID, spanID, model string,
	messagesCount, estimatedTokens int) {
	l.logger.Info("llm_call_start",
		"trace_id", traceID,
		"span_id", spanID,
		"model", model,
		"messages_count", messagesCount,
		"estimated_tokens", estimatedTokens,
	)
}

func (l *AgentLogger) LogLLMResponse(traceID, spanID, model string,
	inputTokens, outputTokens int, latencyMs float64, hasToolCalls bool) {
	l.logger.Info("llm_call_complete",
		"trace_id", traceID,
		"span_id", spanID,
		"model", model,
		"input_tokens", inputTokens,
		"output_tokens", outputTokens,
		"total_tokens", inputTokens+outputTokens,
		"latency_ms", latencyMs,
		"has_tool_calls", hasToolCalls,
	)
}

func (l *AgentLogger) LogToolExecution(traceID, spanID, toolName, paramsSummary string) {
	l.logger.Info("tool_call_start",
		"trace_id", traceID,
		"span_id", spanID,
		"tool_name", toolName,
		"params_summary", paramsSummary,
	)
}

func (l *AgentLogger) LogToolResult(traceID, spanID, toolName string,
	success bool, resultSummary string, latencyMs float64) {
	if len(resultSummary) > 200 {
		resultSummary = resultSummary[:200]
	}
	level := slog.LevelInfo
	if !success {
		level = slog.LevelWarn
	}
	l.logger.Log(nil, level, "tool_call_complete",
		"trace_id", traceID,
		"span_id", spanID,
		"tool_name", toolName,
		"success", success,
		"result_summary", resultSummary,
		"latency_ms", latencyMs,
	)
}

func (l *AgentLogger) LogContextManagement(traceID, action string,
	originalTokens, resultTokens int) {
	l.logger.Warn("context_management",
		"trace_id", traceID,
		"action", action,
		"original_tokens", originalTokens,
		"result_tokens", resultTokens,
		"tokens_removed", originalTokens-resultTokens,
	)
}

func (l *AgentLogger) LogError(traceID string, err error, context map[string]interface{}) {
	l.logger.Error("agent_error",
		"trace_id", traceID,
		"error_type", fmt.Sprintf("%T", err),
		"error_message", err.Error(),
	)
}

// ===========================================================================
// TokenAccountant
// ===========================================================================

// TokenRecord stores cost data for one LLM call.
type TokenRecord struct {
	TraceID      string
	UserID       string
	SessionID    string
	Model        string
	InputTokens  int
	OutputTokens int
	Cost         float64
	Timestamp    time.Time
}

// TokenAccountant tracks token usage and costs.
type TokenAccountant struct {
	mu           sync.RWMutex
	pricing      map[string]ModelPricing
	records      []TokenRecord
	budgetAlerts map[string]float64
}

// NewTokenAccountant creates a TokenAccountant with the given pricing table.
func NewTokenAccountant(pricing map[string]ModelPricing) *TokenAccountant {
	if pricing == nil {
		pricing = DefaultPricing
	}
	return &TokenAccountant{
		pricing:      pricing,
		budgetAlerts: map[string]float64{},
	}
}

// Record extracts LLM spans from trace and stores cost records.
func (a *TokenAccountant) Record(trace *Trace, userID, sessionID string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, s := range trace.Spans {
		if s.Type != "llm_call" {
			continue
		}
		model := s.Model
		if model == "" {
			model = "unknown"
		}
		r := TokenRecord{
			TraceID:      trace.TraceID,
			UserID:       userID,
			SessionID:    sessionID,
			Model:        model,
			InputTokens:  s.InputTokens,
			OutputTokens: s.OutputTokens,
			Cost:         s.Cost,
			Timestamp:    time.Now(),
		}
		a.records = append(a.records, r)

		if max, ok := a.budgetAlerts[userID]; ok {
			total := a.getUserCostLocked(userID, 30)
			if total > max {
				fmt.Printf("[BUDGET ALERT] User '%s' exceeded $%.2f: current $%.4f\n",
					userID, max, total)
			}
		}
	}
}

func (a *TokenAccountant) getUserCostLocked(userID string, days int) float64 {
	cutoff := time.Now().AddDate(0, 0, -days)
	sum := 0.0
	for _, r := range a.records {
		if r.UserID == userID && r.Timestamp.After(cutoff) {
			sum += r.Cost
		}
	}
	return sum
}

// GetUserCost returns total cost for a user over the last N days.
func (a *TokenAccountant) GetUserCost(userID string, days int) float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.getUserCostLocked(userID, days)
}

// GetSessionCost returns total cost for a session.
func (a *TokenAccountant) GetSessionCost(sessionID string) float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	sum := 0.0
	for _, r := range a.records {
		if r.SessionID == sessionID {
			sum += r.Cost
		}
	}
	return sum
}

// GetModelUsage returns usage statistics for a model over the last N days.
func (a *TokenAccountant) GetModelUsage(model string, days int) map[string]interface{} {
	a.mu.RLock()
	defer a.mu.RUnlock()
	cutoff := time.Now().AddDate(0, 0, -days)
	var recs []TokenRecord
	for _, r := range a.records {
		if r.Model == model && r.Timestamp.After(cutoff) {
			recs = append(recs, r)
		}
	}
	inputTok, outputTok := 0, 0
	totalCost := 0.0
	for _, r := range recs {
		inputTok += r.InputTokens
		outputTok += r.OutputTokens
		totalCost += r.Cost
	}
	return map[string]interface{}{
		"model":         model,
		"calls":         len(recs),
		"input_tokens":  inputTok,
		"output_tokens": outputTok,
		"total_cost":    totalCost,
	}
}

// GetDailyCostReport returns a 24-hour cost breakdown.
func (a *TokenAccountant) GetDailyCostReport() map[string]interface{} {
	a.mu.RLock()
	defer a.mu.RUnlock()
	cutoff := time.Now().Add(-24 * time.Hour)
	byModel := map[string]map[string]interface{}{}
	users := map[string]bool{}
	total, inTok, outTok := 0.0, 0, 0
	for _, r := range a.records {
		if r.Timestamp.Before(cutoff) {
			continue
		}
		total += r.Cost
		inTok += r.InputTokens
		outTok += r.OutputTokens
		users[r.UserID] = true
		if _, ok := byModel[r.Model]; !ok {
			byModel[r.Model] = map[string]interface{}{
				"calls": 0, "input_tokens": 0,
				"output_tokens": 0, "cost": 0.0,
			}
		}
		m := byModel[r.Model]
		m["calls"] = m["calls"].(int) + 1
		m["input_tokens"] = m["input_tokens"].(int) + r.InputTokens
		m["output_tokens"] = m["output_tokens"].(int) + r.OutputTokens
		m["cost"] = m["cost"].(float64) + r.Cost
	}
	return map[string]interface{}{
		"date":                time.Now().UTC().Format("2006-01-02"),
		"total_requests":      len(a.records),
		"total_input_tokens":  inTok,
		"total_output_tokens": outTok,
		"total_cost":          total,
		"unique_users":        len(users),
		"by_model":            byModel,
	}
}

// SetBudgetAlert registers a per-user cost cap.
func (a *TokenAccountant) SetBudgetAlert(userID string, maxCost float64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.budgetAlerts[userID] = maxCost
}

// ===========================================================================
// DecisionTracer
// ===========================================================================

// Decision holds one captured agent decision.
type Decision struct {
	DecisionID string
	Timestamp  time.Time
	Step       string
	Context    map[string]interface{}
	Options    []string
	Chosen     string
	Reasoning  string
}

// DecisionTracer records agent decisions for debugging.
type DecisionTracer struct {
	mu        sync.Mutex
	Decisions []Decision
}

// Capture records a decision and returns it.
func (d *DecisionTracer) Capture(step string, ctx map[string]interface{},
	options []string, chosen, reasoning string) Decision {
	dec := Decision{
		DecisionID: uuid.New().String(),
		Timestamp:  time.Now(),
		Step:       step,
		Context:    ctx,
		Options:    options,
		Chosen:     chosen,
		Reasoning:  reasoning,
	}
	d.mu.Lock()
	d.Decisions = append(d.Decisions, dec)
	d.mu.Unlock()
	return dec
}

// Replay returns a human-readable decision trail.
func (d *DecisionTracer) Replay(traceID string) string {
	d.mu.Lock()
	decisions := make([]Decision, len(d.Decisions))
	copy(decisions, d.Decisions)
	d.mu.Unlock()

	if len(decisions) == 0 {
		return "(no decisions recorded)"
	}
	var sb strings.Builder
	if traceID != "" {
		fmt.Fprintf(&sb, "# Agent Decision Trail — trace %s\n\n", traceID)
	} else {
		sb.WriteString("# Agent Decision Trail\n\n")
	}
	for i, dec := range decisions {
		ctxBytes, _ := json.Marshal(dec.Context)
		fmt.Fprintf(&sb, "## Step %d: %s\n", i+1, dec.Step)
		fmt.Fprintf(&sb, "  Context:   %s\n", ctxBytes)
		fmt.Fprintf(&sb, "  Options:   %s\n", strings.Join(dec.Options, ", "))
		fmt.Fprintf(&sb, "  Chosen:    %s\n", dec.Chosen)
		fmt.Fprintf(&sb, "  Reasoning: %s\n\n", dec.Reasoning)
	}
	return sb.String()
}

// FindDivergence locates the step where the agent diverged from expectedChoice.
func (d *DecisionTracer) FindDivergence(expectedChoice string) string {
	d.mu.Lock()
	decisions := make([]Decision, len(d.Decisions))
	copy(decisions, d.Decisions)
	d.mu.Unlock()

	for i, dec := range decisions {
		for _, opt := range dec.Options {
			if opt == expectedChoice && dec.Chosen != expectedChoice {
				ctxBytes, _ := json.Marshal(dec.Context)
				return fmt.Sprintf(
					"Divergence at step %d (%s):\n  Expected:  '%s'\n  Chosen:    '%s'\n  Reasoning: %s\n  Context:   %s",
					i+1, dec.Step, expectedChoice, dec.Chosen, dec.Reasoning, ctxBytes,
				)
			}
		}
	}
	return fmt.Sprintf("No divergence found — '%s' was either chosen or not in options.", expectedChoice)
}

// ===========================================================================
// Simulated LLM + ObservableAgent (demo)
// ===========================================================================

// SimulatedLLM is a deterministic fake LLM for demos.
type SimulatedLLM struct {
	FailOn string
}

type llmResponse struct {
	Model        string
	InputTokens  int
	OutputTokens int
	Content      string
}

func (l *SimulatedLLM) Chat(query string) (llmResponse, error) {
	if l.FailOn != "" && strings.Contains(query, l.FailOn) {
		return llmResponse{}, fmt.Errorf("simulated LLM failure for query: %s", query)
	}
	time.Sleep(50 * time.Millisecond)
	short := query
	if len(short) > 40 {
		short = short[:40]
	}
	return llmResponse{
		Model:        "gpt-4o-mini",
		InputTokens:  512,
		OutputTokens: 128,
		Content:      "Simulated answer for: " + short,
	}, nil
}

// ObservableAgent is a demo agent wired with all observability components.
type ObservableAgent struct {
	collector  *TraceCollector
	metrics    *AgentMetrics
	logger     *AgentLogger
	accountant *TokenAccountant
	dt         *DecisionTracer
	llm        *SimulatedLLM
	pricing    map[string]ModelPricing
}

// NewObservableAgent wires up and returns an ObservableAgent.
func NewObservableAgent(collector *TraceCollector, metrics *AgentMetrics,
	logger *AgentLogger, accountant *TokenAccountant, dt *DecisionTracer,
	failOn string) *ObservableAgent {
	return &ObservableAgent{
		collector:  collector,
		metrics:    metrics,
		logger:     logger,
		accountant: accountant,
		dt:         dt,
		llm:        &SimulatedLLM{FailOn: failOn},
		pricing:    DefaultPricing,
	}
}

func (a *ObservableAgent) llmSpan(trace *Trace, parent *Span,
	callName, query string) (*Span, error) {
	span := trace.NewSpan("llm_call", callName, parent)
	a.logger.LogLLMCall(trace.TraceID, span.SpanID, "gpt-4o-mini", 1, 512)

	resp, err := a.llm.Chat(query)
	if err != nil {
		span.Finish(nil, "error", err.Error())
		return span, err
	}
	span.Model = resp.Model
	span.InputTokens = resp.InputTokens
	span.OutputTokens = resp.OutputTokens
	span.TokensUsed = resp.InputTokens + resp.OutputTokens
	span.Cost = computeCost(resp.Model, resp.InputTokens, resp.OutputTokens, a.pricing)
	span.Finish(map[string]interface{}{"content": resp.Content}, "success", "")

	a.logger.LogLLMResponse(
		trace.TraceID, span.SpanID, resp.Model,
		resp.InputTokens, resp.OutputTokens, span.Duration(), false,
	)
	return span, nil
}

// Run executes one agent request with full observability.
func (a *ObservableAgent) Run(query, userID, sessionID string) (map[string]interface{}, error) {
	trace := a.collector.StartTrace(query, userID, sessionID)

	defer func() {
		trace.Finish()
		a.collector.EndTrace(trace)
		a.metrics.Record(trace)
		a.accountant.Record(trace, userID, sessionID)
	}()

	// Planning
	planSpan := trace.NewSpan("planning", "generate_plan", nil)
	a.dt.Capture("plan", map[string]interface{}{"query": query},
		[]string{"direct_answer", "tool_use"}, "tool_use",
		"Query requires external data lookup")
	planSpan.Finish(map[string]interface{}{"steps": 2}, "success", "")

	// Execution
	execSpan := trace.NewSpan("execution", "execute_plan", planSpan)

	// Tool call
	toolSpan := trace.NewSpan("tool_call", "get_data", execSpan)
	a.logger.LogToolExecution(trace.TraceID, toolSpan.SpanID, "get_data",
		fmt.Sprintf("query=%s", query[:min(30, len(query))]))
	time.Sleep(20 * time.Millisecond)
	toolSpan.Finish(map[string]interface{}{"data": "mock_result"}, "success", "")
	a.logger.LogToolResult(trace.TraceID, toolSpan.SpanID, "get_data",
		true, `{"data":"mock_result"}`, toolSpan.Duration())

	// Synthesis
	synthSpan, err := a.llmSpan(trace, execSpan, "synthesise", query)
	if err != nil {
		execSpan.Finish(nil, "error", err.Error())
		a.logger.LogError(trace.TraceID, err,
			map[string]interface{}{"query": query, "user_id": userID})
		return nil, err
	}
	execSpan.Finish(map[string]interface{}{"tool_calls": 1, "llm_calls": 1}, "success", "")

	// Final generation
	if _, err := a.llmSpan(trace, execSpan, "generate_answer", query); err != nil {
		a.logger.LogError(trace.TraceID, err, nil)
		return nil, err
	}

	content, _ := synthSpan.OutputData["content"].(string)
	answer := fmt.Sprintf("Answer to '%s': %s", query[:min(40, len(query))], content)
	return map[string]interface{}{"answer": answer, "trace_id": trace.TraceID}, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ===========================================================================
// Demo
// ===========================================================================

func main() {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("AGENT OBSERVABILITY DEMO (Go)")
	fmt.Println(strings.Repeat("=", 60))

	collector := NewTraceCollector(&ConsoleExporter{})
	metrics := NewAgentMetrics(1000)
	logger := NewAgentLogger(slog.LevelWarn) // quiet for demo clarity
	accountant := NewTokenAccountant(nil)
	dt := &DecisionTracer{}

	accountant.SetBudgetAlert("alice", 0.01)

	agent := NewObservableAgent(collector, metrics, logger, accountant, dt, "CRASH")

	type query struct {
		text, userID, sessionID string
	}
	queries := []query{
		{"Compare AAPL vs MSFT stock performance", "alice", "s1"},
		{"Summarise the latest earnings report", "bob", "s2"},
		{"CRASH: trigger a deliberate failure", "alice", "s1"},
	}

	var failingTraceID string

	for _, q := range queries {
		fmt.Printf("\n>>> Running: %q\n", q.text)
		result, err := agent.Run(q.text, q.userID, q.sessionID)
		if err != nil {
			failed := collector.QueryTraces(q.userID, "error", time.Time{}, 1)
			if len(failed) > 0 {
				failingTraceID = failed[0].TraceID
			}
			fmt.Printf("    ✗ Error: %v\n", err)
		} else {
			answer, _ := result["answer"].(string)
			if len(answer) > 80 {
				answer = answer[:80]
			}
			fmt.Printf("    ✓ %s\n", answer)
		}
	}

	// Full trace for failing query
	if failingTraceID != "" {
		fmt.Println("\n" + strings.Repeat("─", 60))
		fmt.Println("FULL TRACE FOR FAILING QUERY")
		fmt.Println(strings.Repeat("─", 60))
		t := collector.GetTrace(failingTraceID)
		if t != nil {
			b, _ := json.MarshalIndent(t.ToMap(), "", "  ")
			fmt.Println(string(b))
		}
	}

	// Decision trail
	fmt.Println("\n" + strings.Repeat("─", 60))
	fmt.Println("DECISION TRAIL")
	fmt.Println(strings.Repeat("─", 60))
	fmt.Println(dt.Replay(""))
	fmt.Println("Divergence check:", dt.FindDivergence("direct_answer"))

	// Metrics summary
	fmt.Println("\n" + strings.Repeat("─", 60))
	fmt.Println("METRICS SUMMARY")
	fmt.Println(strings.Repeat("─", 60))
	b, _ := json.MarshalIndent(metrics.GetSummary(), "", "  ")
	fmt.Println(string(b))

	// Error spike demo for anomaly detection
	fmt.Println("\n--- Simulating error spike ---")
	for i := 0; i < 20; i++ {
		t := NewTrace("bad", "test", "s")
		span := t.NewSpan("llm_call", "fail", nil)
		span.Finish(nil, "error", "simulated")
		t.Finish()
		metrics.Record(t)
		_ = rand.Int() // satisfy import
	}
	alerts := metrics.DetectAnomalies()
	fmt.Printf("\nAlerts triggered: %d\n", len(alerts))
	for _, a := range alerts {
		fmt.Printf("  %s\n", a)
	}

	// Daily cost report
	fmt.Println("\n" + strings.Repeat("─", 60))
	fmt.Println("DAILY COST REPORT")
	fmt.Println(strings.Repeat("─", 60))
	b, _ = json.MarshalIndent(accountant.GetDailyCostReport(), "", "  ")
	fmt.Println(string(b))

	// Prometheus metrics
	fmt.Println("\n" + strings.Repeat("─", 60))
	fmt.Println("PROMETHEUS METRICS")
	fmt.Println(strings.Repeat("─", 60))
	fmt.Println(metrics.ExportPrometheus())
}
