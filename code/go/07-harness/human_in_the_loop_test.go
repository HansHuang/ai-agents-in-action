package main
package main

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Mock Agent
// ---------------------------------------------------------------------------

type mockAgent struct {
	toolCalls []mockCall
	messages  []string
	failTool  bool
}

type mockCall struct {
	toolName string
	params   map[string]any
}

func (m *mockAgent) ExecuteTool(_ context.Context, toolName string, params map[string]any) (any, error) {
	if m.failTool {
		return nil, fmt.Errorf("tool %q unavailable", toolName)
	}
	m.toolCalls = append(m.toolCalls, mockCall{toolName, params})
	return map[string]any{"status": "ok"}, nil
}

func (m *mockAgent) SendMessage(_ context.Context, message string) error {
	m.messages = append(m.messages, message)
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func minCost(v float64) *float64 { return &v }

func defaultPolicy() *ApprovalPolicy {
	return WithDefaults()
}

func makeRequest(action string, params map[string]any, risk RiskLevel, cost float64) *ApprovalRequest {
	return &ApprovalRequest{
		RequestID:           "req-test-001",
		AgentID:             "test-agent",
		SessionID:           "session-test",
		ProposedAction:      action,
		ProposedParams:      params,
		Reasoning:           "test reasoning",
		ConversationSummary: "test context",
		Evidence:            []map[string]any{},
		RiskLevel:           risk,
		EstimatedCost:       cost,
		AffectedSystems:     []string{"test_system"},
		CreatedAt:           time.Now(),
	}
}

// ---------------------------------------------------------------------------
// ApprovalPolicy Tests
// ---------------------------------------------------------------------------

func TestLowRiskActionAutoApproved(t *testing.T) {
	policy := defaultPolicy()
	decision := policy.RequiresApproval("get_weather", map[string]any{"city": "Tokyo"}, nil)
	if decision.RequiresApproval {
		t.Errorf("expected no approval required for get_weather, got rule=%q", decision.Rule)
	}
	if decision.Rule != "default_allow" {
		t.Errorf("expected rule=default_allow, got %q", decision.Rule)
	}
}

func TestHighValueRefundRequiresApproval(t *testing.T) {
	policy := defaultPolicy()
	decision := policy.RequiresApproval("issue_refund", map[string]any{"amount": 750.0}, nil)
	if !decision.RequiresApproval {
		t.Error("expected approval required for $750 refund")
	}
	if decision.Rule != "high_value_refund" {
		t.Errorf("expected rule=high_value_refund, got %q", decision.Rule)
	}
}

func TestLowValueRefundAutoApproved(t *testing.T) {
	policy := defaultPolicy()
	decision := policy.RequiresApproval("issue_refund", map[string]any{"amount": 250.0}, nil)
	if decision.RequiresApproval {
		t.Errorf("expected no approval for $250 refund, got rule=%q", decision.Rule)
	}
}

func TestExternalCommunicationRequiresApproval(t *testing.T) {
	policy := defaultPolicy()
	decision := policy.RequiresApproval("send_email", map[string]any{"to": "user@example.com"}, nil)
	if !decision.RequiresApproval {
		t.Error("expected approval required for send_email")
	}
	if decision.Rule != "external_communication" {
		t.Errorf("expected rule=external_communication, got %q", decision.Rule)
	}
}

func TestPolicyEvaluatesRulesInPriorityOrder(t *testing.T) {
	policy := &ApprovalPolicy{}
	policy.AddRule(&ApprovalRule{
		Name: "high_priority", Priority: 100, RiskLevel: RiskHigh,
		Actions: []string{"send_email"}, TimeoutSeconds: 600,
	})
	policy.AddRule(&ApprovalRule{
		Name: "low_priority", Priority: 10, RiskLevel: RiskLow,
		Actions: []string{"send_email"}, TimeoutSeconds: 60,
	})

	decision := policy.RequiresApproval("send_email", nil, nil)
	if !decision.RequiresApproval {
		t.Fatal("expected approval required")
	}
	if decision.Rule != "high_priority" {
		t.Errorf("expected high_priority rule to win, got %q", decision.Rule)
	}
	if decision.TimeoutSeconds != 600 {
		t.Errorf("expected timeout=600, got %.0f", decision.TimeoutSeconds)
	}
}

func TestDefaultAllowWhenNoRulesMatch(t *testing.T) {
	policy := &ApprovalPolicy{}
	decision := policy.RequiresApproval("unknown_action", nil, nil)
	if decision.RequiresApproval {
		t.Error("expected no approval for unknown action")
	}
	if string(decision.RiskLevel) != "none" {
		t.Errorf("expected risk_level=none, got %q", decision.RiskLevel)
	}
}

func TestUserRoleTriggersApproval(t *testing.T) {
	policy := &ApprovalPolicy{}
	policy.AddRule(&ApprovalRule{
		Name: "free_tier_approval", Priority: 50, RiskLevel: RiskMedium,
		Actions: []string{"export_data"}, UserRoles: []string{"free"},
	})

	// Should require approval for "free" role
	decision := policy.RequiresApproval("export_data", nil, map[string]any{"role": "free"})
	if !decision.RequiresApproval {
		t.Error("expected approval for free role")
	}

	// Should NOT require approval for "pro" role
	decision = policy.RequiresApproval("export_data", nil, map[string]any{"role": "pro"})
	if decision.RequiresApproval {
		t.Error("expected no approval for pro role")
	}
}

func TestAffectedSystemsTriggersApproval(t *testing.T) {
	policy := &ApprovalPolicy{}
	policy.AddRule(&ApprovalRule{
		Name: "payment_gate", Priority: 50, RiskLevel: RiskHigh,
		AffectedSystems: []string{"payment_processor"},
	})
	// issue_refund affects payment_processor
	decision := policy.RequiresApproval("issue_refund", map[string]any{"amount": 1.0}, nil)
	if !decision.RequiresApproval {
		t.Error("expected approval for payment_processor action")
	}
}

// ---------------------------------------------------------------------------
// ApprovalInterface Tests
// ---------------------------------------------------------------------------

func newTestInterface() *ApprovalInterface {
	iface := NewApprovalInterface([]string{"dashboard"})
	iface.RegisterReviewer(&Reviewer{
		ReviewerID: "r-alice", Name: "Alice",
		IsSenior: true, Expertise: []string{"send_email"},
	})
	iface.RegisterReviewer(&Reviewer{
		ReviewerID: "r-bob", Name: "Bob",
		IsSenior: false, Expertise: []string{},
	})
	return iface
}

func TestApprovalRequestApproved(t *testing.T) {
	iface := newTestInterface()
	reviewer := &HumanReviewerInterface{ReviewerID: "r-alice"}
	req := makeRequest("send_email", map[string]any{"to": "u@example.com"}, RiskMedium, 0)
	req.RequestID = "test-approve-001"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		time.Sleep(50 * time.Millisecond)
		_ = iface.SubmitResponse(reviewer.Approve(req.RequestID, "Looks fine."))
	}()

	resp, err := iface.RequestApproval(ctx, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Decision != DecisionApproved {
		t.Errorf("expected approved, got %q", resp.Decision)
	}
	if resp.Automated {
		t.Error("expected non-automated response")
	}
}

func TestApprovalRequestRejected(t *testing.T) {
	iface := newTestInterface()
	reviewer := &HumanReviewerInterface{ReviewerID: "r-alice"}
	req := makeRequest("send_email", nil, RiskMedium, 0)
	req.RequestID = "test-reject-001"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		time.Sleep(50 * time.Millisecond)
		_ = iface.SubmitResponse(reviewer.Reject(req.RequestID, "Not approved."))
	}()

	resp, _ := iface.RequestApproval(ctx, req)
	if resp.Decision != DecisionRejected {
		t.Errorf("expected rejected, got %q", resp.Decision)
	}
}

func TestApprovalRequestEdited(t *testing.T) {
	iface := newTestInterface()
	reviewer := &HumanReviewerInterface{ReviewerID: "r-bob"}
	req := makeRequest("issue_refund", map[string]any{"amount": 750.0}, RiskHigh, 750)
	req.RequestID = "test-edit-001"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		time.Sleep(50 * time.Millisecond)
		_ = iface.SubmitResponse(reviewer.ApproveWithEdits(
			req.RequestID,
			map[string]any{"amount": 500.0},
			"Partial refund per policy.",
		))
	}()

	resp, _ := iface.RequestApproval(ctx, req)
	if resp.Decision != DecisionApprovedWithEdits {
		t.Errorf("expected approved_with_edits, got %q", resp.Decision)
	}
	if resp.EditedParams["amount"] != 500.0 {
		t.Errorf("expected edited amount=500, got %v", resp.EditedParams["amount"])
	}
}

func TestApprovalTimeoutAutoRejects(t *testing.T) {
	iface := newTestInterface()
	req := makeRequest("send_email", nil, RiskMedium, 0)
	req.RequestID = "test-timeout-001"

	// 100ms timeout — no one responds
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	resp, _ := iface.RequestApproval(ctx, req)
	if resp.Decision != DecisionRejected {
		t.Errorf("expected rejected on timeout, got %q", resp.Decision)
	}
	if !resp.Automated {
		t.Error("expected automated=true on timeout")
	}
}

func TestReviewerAssignedByRiskLevel(t *testing.T) {
	iface := newTestInterface()

	// High risk → senior reviewer (alice)
	req := makeRequest("issue_refund", map[string]any{"amount": 750.0}, RiskHigh, 750)
	assigned := iface.assignReviewer(req)
	if assigned == nil || !assigned.IsSenior {
		t.Error("expected senior reviewer for high-risk request")
	}

	// Low risk → any reviewer
	req2 := makeRequest("get_weather", nil, RiskLow, 0)
	assigned2 := iface.assignReviewer(req2)
	if assigned2 == nil {
		t.Error("expected some reviewer for low-risk request")
	}
}

// ---------------------------------------------------------------------------
// ApprovalExecutor Tests
// ---------------------------------------------------------------------------

func TestApprovedActionExecuted(t *testing.T) {
	agent := &mockAgent{}
	executor := NewApprovalExecutor(agent)
	req := makeRequest("send_email", map[string]any{"to": "user@example.com"}, RiskMedium, 0)
	resp := ApprovalResponse{RequestID: req.RequestID, Decision: DecisionApproved, ReviewerID: "r-test"}

	result := executor.Execute(context.Background(), req, resp)
	if !result.Success {
		t.Errorf("expected success, got error: %v", result.Err)
	}
	if len(agent.toolCalls) != 1 || agent.toolCalls[0].toolName != "send_email" {
		t.Error("expected send_email to be called with original params")
	}
}

func TestRejectedActionNotExecuted(t *testing.T) {
	agent := &mockAgent{}
	executor := NewApprovalExecutor(agent)
	req := makeRequest("send_email", nil, RiskMedium, 0)
	resp := ApprovalResponse{
		RequestID: req.RequestID, Decision: DecisionRejected,
		ReviewerID: "r-test", Reason: "Not approved.",
	}

	result := executor.Execute(context.Background(), req, resp)
	if result.Success {
		t.Error("expected failure for rejected action")
	}
	if len(agent.toolCalls) > 0 {
		t.Error("expected no tool calls for rejected action")
	}
	if len(agent.messages) == 0 {
		t.Error("expected user notification message")
	}
}

func TestEditedActionExecutedWithModifiedParams(t *testing.T) {
	agent := &mockAgent{}
	executor := NewApprovalExecutor(agent)
	req := makeRequest("issue_refund", map[string]any{"amount": 750.0}, RiskHigh, 750)
	editedParams := map[string]any{"amount": 500.0}
	resp := ApprovalResponse{
		RequestID: req.RequestID, Decision: DecisionApprovedWithEdits,
		ReviewerID: "r-test", EditedParams: editedParams,
	}

	result := executor.Execute(context.Background(), req, resp)
	if !result.Success {
		t.Fatalf("expected success: %v", result.Err)
	}
	if len(agent.toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(agent.toolCalls))
	}
	if agent.toolCalls[0].params["amount"] != 500.0 {
		t.Errorf("expected edited amount=500, got %v", agent.toolCalls[0].params["amount"])
	}
}

func TestExecutionFailureHandled(t *testing.T) {
	agent := &mockAgent{failTool: true}
	executor := NewApprovalExecutor(agent)
	req := makeRequest("send_email", nil, RiskMedium, 0)
	resp := ApprovalResponse{RequestID: req.RequestID, Decision: DecisionApproved, ReviewerID: "r-test"}

	result := executor.Execute(context.Background(), req, resp)
	if result.Success {
		t.Error("expected failure when tool errors")
	}
	if result.Err == nil {
		t.Error("expected non-nil error")
	}
}

// ---------------------------------------------------------------------------
// ApprovalMetrics Tests
// ---------------------------------------------------------------------------

func TestMetricsTracksAllDecisions(t *testing.T) {
	m := NewApprovalMetrics()
	req := makeRequest("send_email", nil, RiskMedium, 0)

	for i := 0; i < 10; i++ {
		m.Record(req, ApprovalResponse{Decision: DecisionApproved}, 30*time.Second)
	}
	for i := 0; i < 3; i++ {
		m.Record(req, ApprovalResponse{Decision: DecisionRejected}, 20*time.Second)
	}
	for i := 0; i < 2; i++ {
		m.Record(req, ApprovalResponse{Decision: DecisionApprovedWithEdits}, 25*time.Second)
	}
	m.Record(req, ApprovalResponse{Decision: DecisionRejected, Automated: true}, 300*time.Second)

	if m.TotalRequests != 16 {
		t.Errorf("expected 16 total, got %d", m.TotalRequests)
	}
	if m.Approved != 10 {
		t.Errorf("expected 10 approved, got %d", m.Approved)
	}
	if m.Rejected != 4 {
		t.Errorf("expected 4 rejected (3 + 1 timeout), got %d", m.Rejected)
	}
	if m.ApprovedWithEdits != 2 {
		t.Errorf("expected 2 edits, got %d", m.ApprovedWithEdits)
	}
	if m.TimedOut != 1 {
		t.Errorf("expected 1 timeout, got %d", m.TimedOut)
	}
}

func TestMetricsByRiskLevel(t *testing.T) {
	m := NewApprovalMetrics()
	highReq := makeRequest("issue_refund", nil, RiskHigh, 750)
	lowReq  := makeRequest("get_weather", nil, RiskLow, 0)

	m.Record(highReq, ApprovalResponse{Decision: DecisionApproved}, 10*time.Second)
	m.Record(highReq, ApprovalResponse{Decision: DecisionRejected}, 10*time.Second)
	m.Record(lowReq, ApprovalResponse{Decision: DecisionApproved}, 5*time.Second)

	if m.ByRiskLevel[RiskHigh][DecisionApproved] != 1 {
		t.Errorf("expected 1 approved high-risk, got %d", m.ByRiskLevel[RiskHigh][DecisionApproved])
	}
	if m.ByRiskLevel[RiskHigh][DecisionRejected] != 1 {
		t.Errorf("expected 1 rejected high-risk, got %d", m.ByRiskLevel[RiskHigh][DecisionRejected])
	}
	if m.ByRiskLevel[RiskLow][DecisionApproved] != 1 {
		t.Errorf("expected 1 approved low-risk, got %d", m.ByRiskLevel[RiskLow][DecisionApproved])
	}
}

func TestApprovalRateCalculation(t *testing.T) {
	m := NewApprovalMetrics()
	req := makeRequest("send_email", nil, RiskMedium, 0)

	for i := 0; i < 8; i++ {
		m.Record(req, ApprovalResponse{Decision: DecisionApproved}, 10*time.Second)
	}
	for i := 0; i < 2; i++ {
		m.Record(req, ApprovalResponse{Decision: DecisionRejected}, 10*time.Second)
	}

	summary := m.Summary()
	rate := summary["approval_rate"].(float64)
	if rate != 0.8 {
		t.Errorf("expected approval_rate=0.8, got %.3f", rate)
	}
}

func TestAvgResponseTimeCalculation(t *testing.T) {
	m := NewApprovalMetrics()
	req := makeRequest("send_email", nil, RiskMedium, 0)

	m.Record(req, ApprovalResponse{Decision: DecisionApproved}, 10*time.Second)
	m.Record(req, ApprovalResponse{Decision: DecisionApproved}, 20*time.Second)
	m.Record(req, ApprovalResponse{Decision: DecisionApproved}, 30*time.Second)

	avg := m.AvgResponseTime()
	if avg != 20.0 {
		t.Errorf("expected avg=20.0s, got %.1f", avg)
	}
}

// ---------------------------------------------------------------------------
// Integration Tests
// ---------------------------------------------------------------------------

func TestFullApprovalLifecycle(t *testing.T) {
	policy := defaultPolicy()
	iface := newTestInterface()
	agent := &mockAgent{}
	executor := NewApprovalExecutor(agent)
	reviewer := &HumanReviewerInterface{ReviewerID: "r-alice"}

	req := makeRequest("send_email", map[string]any{"to": "u@example.com"}, RiskMedium, 0)
	req.RequestID = "integration-approve-001"

	// 1. Policy requires approval
	decision := policy.RequiresApproval(req.ProposedAction, req.ProposedParams, nil)
	if !decision.RequiresApproval {
		t.Fatal("expected approval required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// 2. Schedule reviewer response
	go func() {
		time.Sleep(50 * time.Millisecond)
		_ = iface.SubmitResponse(reviewer.Approve(req.RequestID, "OK"))
	}()

	// 3. Request approval
	resp, err := iface.RequestApproval(ctx, req)
	if err != nil || resp.Decision != DecisionApproved {
		t.Fatalf("expected approved response, got err=%v decision=%q", err, resp.Decision)
	}

	// 4. Execute
	result := executor.Execute(ctx, req, resp)
	if !result.Success {
		t.Errorf("expected successful execution: %v", result.Err)
	}
	if len(agent.messages) == 0 {
		t.Error("expected user notification after execution")
	}
}

func TestFullRejectionLifecycle(t *testing.T) {
	iface := newTestInterface()
	agent := &mockAgent{}
	executor := NewApprovalExecutor(agent)
	reviewer := &HumanReviewerInterface{ReviewerID: "r-alice"}

	req := makeRequest("send_email", nil, RiskMedium, 0)
	req.RequestID = "integration-reject-001"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		time.Sleep(50 * time.Millisecond)
		_ = iface.SubmitResponse(reviewer.Reject(req.RequestID, "Content needs review."))
	}()

	resp, _ := iface.RequestApproval(ctx, req)
	result := executor.Execute(ctx, req, resp)

	if result.Success {
		t.Error("expected failure after rejection")
	}
	if len(agent.toolCalls) > 0 {
		t.Error("no tool calls expected after rejection")
	}
	if len(agent.messages) == 0 {
		t.Error("expected rejection notification to user")
	}
}

func TestFullTimeoutLifecycle(t *testing.T) {
	iface := newTestInterface()
	agent := &mockAgent{}
	executor := NewApprovalExecutor(agent)

	req := makeRequest("send_email", nil, RiskMedium, 0)
	req.RequestID = "integration-timeout-001"

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	resp, _ := iface.RequestApproval(ctx, req)
	if !resp.Automated {
		t.Error("expected automated=true after timeout")
	}

	result := executor.Execute(context.Background(), req, resp)
	if result.Success {
		t.Error("expected failure for auto-rejected request")
	}
}

func TestConcurrentApprovalRequests(t *testing.T) {
	iface := newTestInterface()
	metrics := NewApprovalMetrics()
	reviewer := &HumanReviewerInterface{ReviewerID: "r-alice"}

	const n = 5
	done := make(chan struct{}, n)

	for i := 0; i < n; i++ {
		reqID := fmt.Sprintf("concurrent-%d", i)
		req := makeRequest("send_email", nil, RiskMedium, 0)
		req.RequestID = reqID

		go func(r *ApprovalRequest) {
			defer func() { done <- struct{}{} }()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			go func() {
				time.Sleep(20 * time.Millisecond)
				_ = iface.SubmitResponse(reviewer.Approve(r.RequestID, ""))
			}()

			start := time.Now()
			resp, _ := iface.RequestApproval(ctx, r)
			metrics.Record(r, resp, time.Since(start))
		}(req)
	}

	for i := 0; i < n; i++ {
		<-done
	}

	if metrics.TotalRequests != n {
		t.Errorf("expected %d total requests, got %d", n, metrics.TotalRequests)
	}
	if metrics.Approved != n {
		t.Errorf("expected %d approved, got %d", n, metrics.Approved)
	}
}

func TestAuditTrailComplete(t *testing.T) {
	iface := newTestInterface()
	agent := &mockAgent{}
	executor := NewApprovalExecutor(agent)
	reviewer := &HumanReviewerInterface{ReviewerID: "r-alice"}

	req := makeRequest("send_email", nil, RiskMedium, 0)
	req.RequestID = "audit-trail-001"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		time.Sleep(50 * time.Millisecond)
		_ = iface.SubmitResponse(reviewer.Approve(req.RequestID, "audit test"))
	}()

	resp, _ := iface.RequestApproval(ctx, req)
	executor.Execute(ctx, req, resp)

	if len(executor.ApprovedActions) != 1 {
		t.Errorf("expected 1 audit entry, got %d", len(executor.ApprovedActions))
	}
	entry := executor.ApprovedActions[0]
	if entry.Request.RequestID != req.RequestID {
		t.Error("audit entry has wrong request ID")
	}
	if entry.Timestamp.IsZero() {
		t.Error("audit entry timestamp is zero")
	}
}
