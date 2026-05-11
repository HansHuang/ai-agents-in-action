// Package harness provides a human-in-the-loop approval system for AI agents.
//
// The approval workflow follows four stages:
//
//	propose → review → decide → execute
//
// Human reviewers are assigned by risk level, responses are delivered via channels,
// and all decisions are logged to an immutable audit trail.
//
// Companion to: docs/07-harness-engineering/06-human-in-the-loop.md
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Risk levels
// ---------------------------------------------------------------------------

// RiskLevel classifies the potential impact of a proposed action.
type RiskLevel string

const (
	RiskLow      RiskLevel = "low"
	RiskMedium   RiskLevel = "medium"
	RiskHigh     RiskLevel = "high"
	RiskCritical RiskLevel = "critical"
)

// Decision represents the outcome of a human review.
type Decision string

const (
	DecisionApproved          Decision = "approved"
	DecisionRejected          Decision = "rejected"
	DecisionApprovedWithEdits Decision = "approved_with_edits"
)

// ---------------------------------------------------------------------------
// ApprovalRequest
// ---------------------------------------------------------------------------

// ApprovalRequest is an action proposed by the agent that requires human approval
// before it can be executed.
type ApprovalRequest struct {
	RequestID           string           `json:"request_id"`
	AgentID             string           `json:"agent_id"`
	SessionID           string           `json:"session_id"`
	ProposedAction      string           `json:"proposed_action"`
	ProposedParams      map[string]any   `json:"proposed_params"`
	Reasoning           string           `json:"reasoning"`
	ConversationSummary string           `json:"conversation_summary"`
	Evidence            []map[string]any `json:"evidence"`
	RiskLevel           RiskLevel        `json:"risk_level"`
	EstimatedCost       float64          `json:"estimated_cost"`
	AffectedSystems     []string         `json:"affected_systems"`
	CreatedAt           time.Time        `json:"created_at"`
	Deadline            *time.Time       `json:"deadline,omitempty"`
}

// ToHumanReadable formats the request for display to a human reviewer.
func (r *ApprovalRequest) ToHumanReadable() string {
	params, _ := json.MarshalIndent(r.ProposedParams, "  ", "  ")

	deadlineStr := "No deadline"
	if r.Deadline != nil {
		remaining := time.Until(*r.Deadline)
		deadlineStr = fmt.Sprintf("Deadline in %s", remaining.Round(time.Second))
	}

	evidence := ""
	for i, e := range r.Evidence {
		if i >= 5 {
			break
		}
		b, _ := json.Marshal(e)
		evidence += fmt.Sprintf("  - %s\n", b)
	}
	if evidence == "" {
		evidence = "  (none)\n"
	}

	return fmt.Sprintf(`
%s
APPROVAL REQUIRED
%s

Request ID  : %s
Action      : %s
Risk Level  : %s
Estimated $ : $%.2f
Systems     : %v
%s

Parameters:
  %s

Reasoning:
%s

Evidence:
%s
%s
Approve? [Y/N/Edit]
%s
`,
		"═══════════════════════════════════════════════",
		"═══════════════════════════════════════════════",
		r.RequestID, r.ProposedAction, r.RiskLevel,
		r.EstimatedCost, r.AffectedSystems, deadlineStr,
		string(params),
		r.Reasoning,
		evidence,
		"═══════════════════════════════════════════════",
		"═══════════════════════════════════════════════",
	)
}

// ---------------------------------------------------------------------------
// ApprovalResponse
// ---------------------------------------------------------------------------

// ApprovalResponse is a human reviewer's decision on an ApprovalRequest.
type ApprovalResponse struct {
	RequestID     string         `json:"request_id"`
	Decision      Decision       `json:"decision"`
	ReviewerID    string         `json:"reviewer_id"`
	ReviewerNotes string         `json:"reviewer_notes,omitempty"`
	EditedParams  map[string]any `json:"edited_params,omitempty"`
	Automated     bool           `json:"automated"`
	DecidedAt     time.Time      `json:"decided_at"`
	Reason        string         `json:"reason,omitempty"`
}

// ---------------------------------------------------------------------------
// ExecutionResult
// ---------------------------------------------------------------------------

// ExecutionResult is the outcome of executing (or declining) an approved action.
type ExecutionResult struct {
	Success bool
	Result  any
	Err     error
}

// ---------------------------------------------------------------------------
// ApprovalDecision
// ---------------------------------------------------------------------------

// ApprovalDecision records whether a specific action requires human review.
type ApprovalDecision struct {
	RequiresApproval bool
	Rule             string
	RiskLevel        RiskLevel
	TimeoutSeconds   float64
}

// ---------------------------------------------------------------------------
// Reviewer
// ---------------------------------------------------------------------------

// Reviewer represents a human reviewer who can approve or reject requests.
type Reviewer struct {
	ReviewerID string
	Name       string
	IsSenior   bool
	Expertise  []string
	SlackID    string
	Email      string
}

// ---------------------------------------------------------------------------
// ApprovalRule
// ---------------------------------------------------------------------------

// ApprovalRule defines when a proposed action requires human approval.
type ApprovalRule struct {
	Name            string
	Description     string
	Priority        int
	RiskLevel       RiskLevel
	TimeoutSeconds  float64
	Actions         []string // if non-nil, action must be in this list
	MinCost         *float64 // if non-nil, cost must be >= this value
	AffectedSystems []string // if non-nil, at least one system must match
	UserRoles       []string // if non-nil, context role must be in this list
}

// Matches returns true if this rule applies to the given action and context.
func (r *ApprovalRule) Matches(action string, params map[string]any, context map[string]any) bool {
	// Action filter
	if len(r.Actions) > 0 {
		found := false
		for _, a := range r.Actions {
			if a == action {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Minimum cost filter
	if r.MinCost != nil {
		cost := r.estimateCost(action, params)
		if cost < *r.MinCost {
			return false
		}
	}

	// Affected systems filter
	if len(r.AffectedSystems) > 0 {
		actionSystems := r.getAffectedSystems(action)
		overlap := false
		for _, s := range actionSystems {
			for _, rs := range r.AffectedSystems {
				if s == rs {
					overlap = true
					break
				}
			}
		}
		if !overlap {
			return false
		}
	}

	// User role filter
	if len(r.UserRoles) > 0 {
		role, _ := context["role"].(string)
		found := false
		for _, ur := range r.UserRoles {
			if ur == role {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

func (r *ApprovalRule) estimateCost(action string, params map[string]any) float64 {
	switch action {
	case "issue_refund":
		if v, ok := params["amount"]; ok {
			switch n := v.(type) {
			case float64:
				return n
			case int:
				return float64(n)
			}
		}
		return 0
	case "send_email":
		return 0
	case "update_database", "delete_record":
		return 50
	default:
		return 10
	}
}

func (r *ApprovalRule) getAffectedSystems(action string) []string {
	m := map[string][]string{
		"send_email":          {"email_service"},
		"issue_refund":        {"payment_processor", "order_database"},
		"update_database":     {"database"},
		"delete_record":       {"database"},
		"cancel_subscription": {"billing_system", "subscription_service"},
		"export_user_data":    {"data_warehouse", "gdpr_service"},
	}
	if systems, ok := m[action]; ok {
		return systems
	}
	return []string{"unknown"}
}

// ---------------------------------------------------------------------------
// ApprovalPolicy
// ---------------------------------------------------------------------------

// ApprovalPolicy holds an ordered set of ApprovalRules.
// Rules are evaluated in descending priority order; the first match wins.
// If no rule matches, the action is auto-approved.
type ApprovalPolicy struct {
	mu    sync.RWMutex
	rules []*ApprovalRule
}

// AddRule inserts a rule and re-sorts by descending priority.
func (p *ApprovalPolicy) AddRule(rule *ApprovalRule) {
	if rule.TimeoutSeconds == 0 {
		rule.TimeoutSeconds = 300
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.rules = append(p.rules, rule)
	// insertion sort (list is small)
	for i := len(p.rules) - 1; i > 0; i-- {
		if p.rules[i].Priority > p.rules[i-1].Priority {
			p.rules[i], p.rules[i-1] = p.rules[i-1], p.rules[i]
		} else {
			break
		}
	}
}

// RequiresApproval checks whether an action needs human review.
func (p *ApprovalPolicy) RequiresApproval(
	action string,
	params map[string]any,
	ctx map[string]any,
) ApprovalDecision {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for _, rule := range p.rules {
		if rule.Matches(action, params, ctx) {
			return ApprovalDecision{
				RequiresApproval: true,
				Rule:             rule.Name,
				RiskLevel:        rule.RiskLevel,
				TimeoutSeconds:   rule.TimeoutSeconds,
			}
		}
	}
	return ApprovalDecision{
		RequiresApproval: false,
		Rule:             "default_allow",
		RiskLevel:        "none",
	}
}

// WithDefaults returns a policy pre-loaded with standard production rules.
func WithDefaults() *ApprovalPolicy {
	minRefund := 500.0
	policy := &ApprovalPolicy{}
	policy.AddRule(&ApprovalRule{
		Name: "high_value_refund", Priority: 100, RiskLevel: RiskHigh,
		Actions: []string{"issue_refund"}, MinCost: &minRefund, TimeoutSeconds: 600,
		Description: "Refunds over $500 require human approval",
	})
	policy.AddRule(&ApprovalRule{
		Name: "critical_data_export", Priority: 95, RiskLevel: RiskHigh,
		Actions: []string{"export_user_data"}, TimeoutSeconds: 600,
		Description: "Exporting user data requires approval",
	})
	policy.AddRule(&ApprovalRule{
		Name: "external_communication", Priority: 90, RiskLevel: RiskMedium,
		Actions: []string{"send_email"}, TimeoutSeconds: 300,
		Description: "Sending email requires approval",
	})
	policy.AddRule(&ApprovalRule{
		Name: "subscription_cancellation", Priority: 85, RiskLevel: RiskHigh,
		Actions: []string{"cancel_subscription"}, TimeoutSeconds: 600,
		Description: "Cancelling subscriptions requires approval",
	})
	policy.AddRule(&ApprovalRule{
		Name: "database_modification", Priority: 80, RiskLevel: RiskMedium,
		Actions: []string{"update_database", "delete_record"}, TimeoutSeconds: 300,
		Description: "Database writes require approval",
	})
	return policy
}

// ---------------------------------------------------------------------------
// HumanReviewerInterface
// ---------------------------------------------------------------------------

// HumanReviewerInterface provides methods for a human reviewer to respond.
type HumanReviewerInterface struct {
	ReviewerID string
}

// Approve approves the request as-is.
func (h *HumanReviewerInterface) Approve(requestID string, notes string) ApprovalResponse {
	return ApprovalResponse{
		RequestID:     requestID,
		Decision:      DecisionApproved,
		ReviewerID:    h.ReviewerID,
		ReviewerNotes: notes,
		DecidedAt:     time.Now(),
	}
}

// Reject rejects the request with a mandatory reason.
func (h *HumanReviewerInterface) Reject(requestID string, reason string) ApprovalResponse {
	return ApprovalResponse{
		RequestID:  requestID,
		Decision:   DecisionRejected,
		ReviewerID: h.ReviewerID,
		Reason:     reason,
		DecidedAt:  time.Now(),
	}
}

// ApproveWithEdits approves the action with modified parameters.
func (h *HumanReviewerInterface) ApproveWithEdits(
	requestID string,
	editedParams map[string]any,
	notes string,
) ApprovalResponse {
	return ApprovalResponse{
		RequestID:     requestID,
		Decision:      DecisionApprovedWithEdits,
		ReviewerID:    h.ReviewerID,
		EditedParams:  editedParams,
		ReviewerNotes: notes,
		DecidedAt:     time.Now(),
	}
}

// ---------------------------------------------------------------------------
// ApprovalInterface
// ---------------------------------------------------------------------------

// pendingEntry holds a request and its response channel.
type pendingEntry struct {
	request *ApprovalRequest
	respCh  chan ApprovalResponse
}

// ApprovalInterface routes approval requests to human reviewers and waits for responses.
// It is safe for concurrent use.
type ApprovalInterface struct {
	mu              sync.Mutex
	pendingRequests map[string]*pendingEntry
	reviewers       map[string]*Reviewer
	channels        []string
}

// NewApprovalInterface creates an interface using the specified notification channels.
// Valid channel names: "dashboard", "slack", "email".
func NewApprovalInterface(channels []string) *ApprovalInterface {
	if len(channels) == 0 {
		channels = []string{"dashboard"}
	}
	return &ApprovalInterface{
		pendingRequests: make(map[string]*pendingEntry),
		reviewers:       make(map[string]*Reviewer),
		channels:        channels,
	}
}

// RegisterReviewer adds a reviewer to the pool.
func (a *ApprovalInterface) RegisterReviewer(r *Reviewer) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.reviewers[r.ReviewerID] = r
}

// RequestApproval sends a request and blocks until a reviewer responds or the
// context deadline is reached.  On timeout the request is auto-rejected.
func (a *ApprovalInterface) RequestApproval(
	ctx context.Context,
	request *ApprovalRequest,
) (ApprovalResponse, error) {
	reviewer := a.assignReviewer(request)
	a.sendToReviewer(reviewer, request)

	respCh := make(chan ApprovalResponse, 1)
	a.mu.Lock()
	a.pendingRequests[request.RequestID] = &pendingEntry{request: request, respCh: respCh}
	a.mu.Unlock()

	select {
	case resp := <-respCh:
		a.mu.Lock()
		delete(a.pendingRequests, request.RequestID)
		a.mu.Unlock()
		return resp, nil

	case <-ctx.Done():
		a.mu.Lock()
		delete(a.pendingRequests, request.RequestID)
		a.mu.Unlock()

		reviewerID := "system"
		if reviewer != nil {
			reviewerID = reviewer.ReviewerID
		}
		slog.Warn("approval timeout — auto-rejecting",
			"request_id", request.RequestID,
			"action", request.ProposedAction)
		return ApprovalResponse{
			RequestID:  request.RequestID,
			Decision:   DecisionRejected,
			ReviewerID: reviewerID,
			Reason:     "Approval timeout — automatically rejected for safety.",
			Automated:  true,
			DecidedAt:  time.Now(),
		}, nil
	}
}

// SubmitResponse delivers a reviewer's decision, unblocking a pending RequestApproval call.
func (a *ApprovalInterface) SubmitResponse(resp ApprovalResponse) error {
	a.mu.Lock()
	entry, ok := a.pendingRequests[resp.RequestID]
	a.mu.Unlock()
	if !ok {
		return fmt.Errorf("no pending request with id %q", resp.RequestID)
	}
	entry.respCh <- resp
	return nil
}

// PendingCount returns the number of requests awaiting review.
func (a *ApprovalInterface) PendingCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.pendingRequests)
}

func (a *ApprovalInterface) assignReviewer(req *ApprovalRequest) *Reviewer {
	a.mu.Lock()
	defer a.mu.Unlock()

	var all, seniors []*Reviewer
	for _, r := range a.reviewers {
		all = append(all, r)
		if r.IsSenior {
			seniors = append(seniors, r)
		}
	}

	switch req.RiskLevel {
	case RiskCritical, RiskHigh:
		if len(seniors) > 0 {
			return seniors[0]
		}
	case RiskMedium:
		for _, r := range all {
			for _, exp := range r.Expertise {
				if exp == req.ProposedAction {
					return r
				}
			}
		}
	}
	if len(all) > 0 {
		return all[0]
	}
	return nil
}

func (a *ApprovalInterface) sendToReviewer(reviewer *Reviewer, req *ApprovalRequest) {
	msg := req.ToHumanReadable()
	for _, ch := range a.channels {
		switch ch {
		case "dashboard":
			slog.Info("dashboard: new pending approval",
				"request_id", req.RequestID,
				"action", req.ProposedAction,
				"risk_level", req.RiskLevel)
		case "slack":
			if reviewer != nil && reviewer.SlackID != "" {
				slog.Info("slack notification sent",
					"to", reviewer.SlackID,
					"preview", msg[:min(80, len(msg))])
			}
		case "email":
			if reviewer != nil && reviewer.Email != "" &&
				(req.RiskLevel == RiskHigh || req.RiskLevel == RiskCritical) {
				slog.Info("email notification sent",
					"to", reviewer.Email,
					"subject", fmt.Sprintf("URGENT: Approval Required — %s", req.ProposedAction))
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ---------------------------------------------------------------------------
// Agent interface
// ---------------------------------------------------------------------------

// Agent is implemented by anything that can execute tools and send messages.
type Agent interface {
	ExecuteTool(ctx context.Context, toolName string, params map[string]any) (any, error)
	SendMessage(ctx context.Context, message string) error
}

// ---------------------------------------------------------------------------
// ApprovalExecutor
// ---------------------------------------------------------------------------

// ApprovalExecutor carries out (or records the rejection of) approved actions.
type ApprovalExecutor struct {
	agent           Agent
	mu              sync.Mutex
	ApprovedActions []AuditEntry
	RejectedActions []AuditEntry
}

// AuditEntry is a single record in the executor audit trail.
type AuditEntry struct {
	Request   *ApprovalRequest
	Response  ApprovalResponse
	Result    any
	Error     error
	Timestamp time.Time
}

// NewApprovalExecutor creates an executor backed by the given agent.
func NewApprovalExecutor(agent Agent) *ApprovalExecutor {
	return &ApprovalExecutor{agent: agent}
}

// Execute carries out the action based on the human's decision.
func (e *ApprovalExecutor) Execute(
	ctx context.Context,
	request *ApprovalRequest,
	response ApprovalResponse,
) ExecutionResult {
	switch response.Decision {
	case DecisionApproved:
		return e.executeApproved(ctx, request, response)
	case DecisionApprovedWithEdits:
		return e.executeEdited(ctx, request, response)
	case DecisionRejected:
		return e.handleRejection(ctx, request, response)
	default:
		return ExecutionResult{Err: fmt.Errorf("unknown decision: %q", response.Decision)}
	}
}

func (e *ApprovalExecutor) executeApproved(
	ctx context.Context,
	req *ApprovalRequest,
	resp ApprovalResponse,
) ExecutionResult {
	slog.Info("executing approved action", "action", req.ProposedAction)
	result, err := e.agent.ExecuteTool(ctx, req.ProposedAction, req.ProposedParams)
	if err != nil {
		slog.Error("approved action failed", "error", err)
		return ExecutionResult{Err: fmt.Errorf("execute %s: %w", req.ProposedAction, err)}
	}
	_ = e.agent.SendMessage(ctx, fmt.Sprintf("Completed: %s", req.ProposedAction))
	e.appendApproved(req, resp, result)
	return ExecutionResult{Success: true, Result: result}
}

func (e *ApprovalExecutor) executeEdited(
	ctx context.Context,
	req *ApprovalRequest,
	resp ApprovalResponse,
) ExecutionResult {
	slog.Info("executing edited action", "action", req.ProposedAction, "params", resp.EditedParams)
	result, err := e.agent.ExecuteTool(ctx, req.ProposedAction, resp.EditedParams)
	if err != nil {
		slog.Error("edited action failed", "error", err)
		return ExecutionResult{Err: fmt.Errorf("execute edited %s: %w", req.ProposedAction, err)}
	}
	_ = e.agent.SendMessage(ctx, "Completed with the adjustments you specified.")
	e.appendApproved(req, resp, result)
	return ExecutionResult{Success: true, Result: result}
}

func (e *ApprovalExecutor) handleRejection(
	ctx context.Context,
	req *ApprovalRequest,
	resp ApprovalResponse,
) ExecutionResult {
	reason := resp.Reason
	if reason == "" {
		reason = "This action requires additional review."
	}
	slog.Info("action rejected", "action", req.ProposedAction, "reason", reason)
	_ = e.agent.SendMessage(ctx,
		fmt.Sprintf("I wasn't able to complete '%s'. %s", req.ProposedAction, reason))
	e.mu.Lock()
	e.RejectedActions = append(e.RejectedActions, AuditEntry{
		Request: req, Response: resp, Timestamp: time.Now(),
	})
	e.mu.Unlock()
	return ExecutionResult{
		Err: fmt.Errorf("rejected by reviewer: %s", reason),
	}
}

func (e *ApprovalExecutor) appendApproved(req *ApprovalRequest, resp ApprovalResponse, result any) {
	e.mu.Lock()
	e.ApprovedActions = append(e.ApprovedActions, AuditEntry{
		Request: req, Response: resp, Result: result, Timestamp: time.Now(),
	})
	e.mu.Unlock()
}

// ---------------------------------------------------------------------------
// ApprovalMetrics
// ---------------------------------------------------------------------------

// ApprovalMetrics tracks human-in-the-loop statistics.
// All methods are safe for concurrent use.
type ApprovalMetrics struct {
	mu                sync.RWMutex
	TotalRequests     int
	Approved          int
	Rejected          int
	ApprovedWithEdits int
	TimedOut          int
	totalResponseTime float64
	ByRiskLevel       map[RiskLevel]map[Decision]int
}

// NewApprovalMetrics initialises a metrics collector.
func NewApprovalMetrics() *ApprovalMetrics {
	return &ApprovalMetrics{
		ByRiskLevel: make(map[RiskLevel]map[Decision]int),
	}
}

// Record adds a single decision to the metrics.
func (m *ApprovalMetrics) Record(
	request *ApprovalRequest,
	response ApprovalResponse,
	responseTime time.Duration,
) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalRequests++
	m.totalResponseTime += responseTime.Seconds()

	if response.Automated {
		m.TimedOut++
	}

	switch response.Decision {
	case DecisionApproved:
		m.Approved++
	case DecisionRejected:
		m.Rejected++
	case DecisionApprovedWithEdits:
		m.ApprovedWithEdits++
	}

	if _, ok := m.ByRiskLevel[request.RiskLevel]; !ok {
		m.ByRiskLevel[request.RiskLevel] = make(map[Decision]int)
	}
	m.ByRiskLevel[request.RiskLevel][response.Decision]++
}

// AvgResponseTime returns the mean response time across all non-automated decisions.
func (m *ApprovalMetrics) AvgResponseTime() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	n := m.Approved + m.Rejected + m.ApprovedWithEdits
	if n == 0 {
		return 0
	}
	return m.totalResponseTime / float64(n)
}

// Summary returns a map suitable for structured logging or JSON serialisation.
func (m *ApprovalMetrics) Summary() map[string]any {
	m.mu.RLock()
	defer m.mu.RUnlock()

	total := math.Max(float64(m.TotalRequests), 1)
	return map[string]any{
		"total_requests":         m.TotalRequests,
		"approved":               m.Approved,
		"rejected":               m.Rejected,
		"approved_with_edits":    m.ApprovedWithEdits,
		"timed_out":              m.TimedOut,
		"approval_rate":          float64(m.Approved) / total,
		"rejection_rate":         float64(m.Rejected) / total,
		"edit_rate":              float64(m.ApprovedWithEdits) / total,
		"timeout_rate":           float64(m.TimedOut) / total,
		"avg_response_time_secs": m.AvgResponseTime(),
		"by_risk_level":          m.ByRiskLevel,
	}
}
