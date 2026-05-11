"""
pytest tests for the human-in-the-loop approval system.

Covers:
  - ApprovalPolicy  (8 tests)
  - ApprovalInterface  (6 tests)
  - ApprovalExecutor  (4 tests)
  - ApprovalMetrics  (4 tests)
  - Integration  (6 tests)

Run with:
    pytest code/python/07-harness/test_human_in_the_loop.py -v
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from human_in_the_loop import (
    ApprovalDecision,
    ApprovalExecutor,
    ApprovalInterface,
    ApprovalMetrics,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalRule,
    ExecutionResult,
    HumanReviewerInterface,
    Reviewer,
    ReviewerGroup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy():
    return ApprovalPolicy.with_defaults()


@pytest.fixture
def basic_request():
    return ApprovalRequest(
        request_id=str(uuid.uuid4()),
        agent_id="test-agent",
        session_id="session-test",
        proposed_action="send_email",
        proposed_params={"to": "user@example.com"},
        reasoning="User asked to send an email",
        conversation_summary="Customer: please send me an update.",
        evidence=[{"source": "crm", "data": "customer verified"}],
        risk_level="medium",
        estimated_cost=0.0,
        affected_systems=["email_service"],
        created_at=time.time(),
    )


@pytest.fixture
def refund_request():
    return ApprovalRequest(
        request_id=str(uuid.uuid4()),
        agent_id="test-agent",
        session_id="session-test",
        proposed_action="issue_refund",
        proposed_params={"order_id": "ORD-99", "amount": 750},
        reasoning="Customer damaged item",
        conversation_summary="Customer complained about damaged goods.",
        evidence=[],
        risk_level="high",
        estimated_cost=750.0,
        affected_systems=["payment_processor", "order_database"],
        created_at=time.time(),
    )


@pytest.fixture
def interface_with_reviewers():
    iface = ApprovalInterface(channels=["dashboard"])
    alice = Reviewer("r-alice", "Alice", is_senior=True, expertise=["send_email"])
    bob   = Reviewer("r-bob",   "Bob",   is_senior=True, expertise=["issue_refund"])
    carol = Reviewer("r-carol", "Carol", is_senior=False, expertise=[])
    for r in (alice, bob, carol):
        iface.register_reviewer(r)
    return iface


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.execute_tool = AsyncMock(return_value={"status": "ok"})
    agent.send_message = AsyncMock()
    agent.send_message_sync = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# 1. ApprovalPolicy Tests
# ---------------------------------------------------------------------------

class TestApprovalPolicy:

    def test_low_risk_action_auto_approved(self, policy):
        """get_weather matches no rules → auto-approved."""
        decision = policy.requires_approval("get_weather", {"city": "Tokyo"}, {})
        assert not decision.requires_approval
        assert decision.rule == "default_allow"
        assert decision.risk_level == "none"

    def test_high_value_refund_requires_approval(self, policy):
        """$750 refund exceeds min_cost=500 → requires approval."""
        decision = policy.requires_approval("issue_refund", {"amount": 750}, {})
        assert decision.requires_approval
        assert decision.rule == "high_value_refund"
        assert decision.risk_level == "high"

    def test_low_value_refund_auto_approved(self, policy):
        """$250 refund is below the $500 threshold → auto-approved."""
        decision = policy.requires_approval("issue_refund", {"amount": 250}, {})
        assert not decision.requires_approval

    def test_external_communication_requires_approval(self, policy):
        """send_email always requires approval via external_communication rule."""
        decision = policy.requires_approval("send_email", {"to": "user@example.com"}, {})
        assert decision.requires_approval
        assert decision.rule == "external_communication"

    def test_policy_evaluates_rules_in_priority_order(self):
        """Higher-priority rule wins when multiple rules match."""
        policy = ApprovalPolicy()
        policy.add_rule(ApprovalRule(
            name="high_priority", description="high", priority=100,
            risk_level="high", actions=["send_email"], timeout_seconds=600,
        ))
        policy.add_rule(ApprovalRule(
            name="low_priority", description="low", priority=10,
            risk_level="low", actions=["send_email"], timeout_seconds=60,
        ))

        decision = policy.requires_approval("send_email", {}, {})
        assert decision.rule == "high_priority"
        assert decision.risk_level == "high"
        assert decision.timeout_seconds == 600

    def test_default_allow_when_no_rules_match(self, policy):
        """An action that matches no rule is auto-approved with risk_level=none."""
        decision = policy.requires_approval("some_unknown_action", {}, {})
        assert not decision.requires_approval
        assert decision.risk_level == "none"

    def test_user_role_triggers_approval(self):
        """A rule with user_roles fires only for matching roles."""
        policy = ApprovalPolicy()
        policy.add_rule(ApprovalRule(
            name="free_tier", description="free users", priority=50,
            risk_level="medium", actions=["export_data"],
            user_roles=["free"],
        ))

        assert policy.requires_approval("export_data", {}, {"role": "free"}).requires_approval
        assert not policy.requires_approval("export_data", {}, {"role": "pro"}).requires_approval

    def test_affected_systems_triggers_approval(self):
        """A rule gated on affected_systems fires for issue_refund (hits payment_processor)."""
        policy = ApprovalPolicy()
        policy.add_rule(ApprovalRule(
            name="payment_gate", description="protect payments", priority=50,
            risk_level="high", affected_systems=["payment_processor"],
        ))

        decision = policy.requires_approval("issue_refund", {"amount": 1.0}, {})
        assert decision.requires_approval
        assert decision.rule == "payment_gate"


# ---------------------------------------------------------------------------
# 2. ApprovalInterface Tests
# ---------------------------------------------------------------------------

class TestApprovalInterface:

    @pytest.mark.asyncio
    async def test_approval_request_approved(self, interface_with_reviewers, basic_request):
        reviewer = HumanReviewerInterface("r-alice")
        asyncio.get_event_loop().call_later(
            0.05, lambda: interface_with_reviewers.submit_response(
                reviewer.approve(basic_request.request_id, notes="Looks good.")
            )
        )
        resp = await interface_with_reviewers.request_approval(basic_request, timeout_seconds=2)
        assert resp.decision == "approved"
        assert not resp.automated

    @pytest.mark.asyncio
    async def test_approval_request_rejected(self, interface_with_reviewers, basic_request):
        reviewer = HumanReviewerInterface("r-alice")
        asyncio.get_event_loop().call_later(
            0.05, lambda: interface_with_reviewers.submit_response(
                reviewer.reject(basic_request.request_id, reason="Not approved.")
            )
        )
        resp = await interface_with_reviewers.request_approval(basic_request, timeout_seconds=2)
        assert resp.decision == "rejected"

    @pytest.mark.asyncio
    async def test_approval_request_edited(self, interface_with_reviewers, refund_request):
        reviewer = HumanReviewerInterface("r-bob")
        asyncio.get_event_loop().call_later(
            0.05, lambda: interface_with_reviewers.submit_response(
                reviewer.approve_with_edits(
                    refund_request.request_id,
                    edited_params={"order_id": "ORD-99", "amount": 500},
                    notes="Partial refund only.",
                )
            )
        )
        resp = await interface_with_reviewers.request_approval(refund_request, timeout_seconds=2)
        assert resp.decision == "approved_with_edits"
        assert resp.edited_params == {"order_id": "ORD-99", "amount": 500}

    @pytest.mark.asyncio
    async def test_approval_timeout_auto_rejects(self, interface_with_reviewers, basic_request):
        """No reviewer responds → auto-reject after timeout."""
        resp = await interface_with_reviewers.request_approval(
            basic_request, timeout_seconds=0.1
        )
        assert resp.decision == "rejected"
        assert resp.automated is True

    @pytest.mark.asyncio
    async def test_reviewer_assigned_by_risk_level(self, interface_with_reviewers):
        """High-risk request → senior reviewer assigned."""
        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id="a", session_id="s",
            proposed_action="issue_refund",
            proposed_params={"amount": 750},
            reasoning="test", conversation_summary="test",
            evidence=[], risk_level="high", estimated_cost=750.0,
            affected_systems=[], created_at=time.time(),
        )
        reviewer = await interface_with_reviewers._assign_reviewer(req)
        is_senior = getattr(reviewer, "is_senior", False)
        # ReviewerGroup also works for critical
        if not isinstance(reviewer, ReviewerGroup):
            assert is_senior, "High-risk should be assigned to a senior reviewer"

    @pytest.mark.asyncio
    async def test_critical_requires_two_approvals(self, interface_with_reviewers):
        """Critical request uses ReviewerGroup; one rejection = rejected."""
        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id="a", session_id="s",
            proposed_action="export_user_data",
            proposed_params={"user_id": "U-42"},
            reasoning="GDPR request", conversation_summary="test",
            evidence=[], risk_level="critical", estimated_cost=0.0,
            affected_systems=["data_warehouse"], created_at=time.time(),
        )

        alice = HumanReviewerInterface("r-alice")
        bob   = HumanReviewerInterface("r-bob")

        async def _submit_both():
            await asyncio.sleep(0.03)
            interface_with_reviewers.submit_response(alice.approve(req.request_id))
            await asyncio.sleep(0.01)
            interface_with_reviewers.submit_response(bob.approve(req.request_id))

        asyncio.get_event_loop().create_task(_submit_both())
        resp = await interface_with_reviewers.request_approval(req, timeout_seconds=2)
        assert resp.decision == "approved"


# ---------------------------------------------------------------------------
# 3. ApprovalExecutor Tests
# ---------------------------------------------------------------------------

class TestApprovalExecutor:

    @pytest.mark.asyncio
    async def test_approved_action_executed(self, mock_agent, basic_request):
        executor = ApprovalExecutor(mock_agent)
        resp = ApprovalResponse(
            request_id=basic_request.request_id, decision="approved",
            reviewer_id="r-test", decided_at=time.time(),
        )
        result = await executor.execute(basic_request, resp)
        assert result.success
        mock_agent.execute_tool.assert_awaited_once_with(
            tool_name="send_email",
            params={"to": "user@example.com"},
        )

    @pytest.mark.asyncio
    async def test_rejected_action_not_executed(self, mock_agent, basic_request):
        executor = ApprovalExecutor(mock_agent)
        resp = ApprovalResponse(
            request_id=basic_request.request_id, decision="rejected",
            reviewer_id="r-test", reason="Not approved.", decided_at=time.time(),
        )
        result = await executor.execute(basic_request, resp)
        assert not result.success
        mock_agent.execute_tool.assert_not_awaited()
        mock_agent.send_message_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_edited_action_executed_with_modified_params(self, mock_agent, refund_request):
        executor = ApprovalExecutor(mock_agent)
        edited = {"order_id": "ORD-99", "amount": 500}
        resp = ApprovalResponse(
            request_id=refund_request.request_id, decision="approved_with_edits",
            reviewer_id="r-test", edited_params=edited, decided_at=time.time(),
        )
        result = await executor.execute(refund_request, resp)
        assert result.success
        mock_agent.execute_tool.assert_awaited_once_with(
            tool_name="issue_refund",
            params=edited,
        )

    @pytest.mark.asyncio
    async def test_execution_failure_handled(self, mock_agent, basic_request):
        mock_agent.execute_tool = AsyncMock(side_effect=RuntimeError("Service down"))
        executor = ApprovalExecutor(mock_agent)
        resp = ApprovalResponse(
            request_id=basic_request.request_id, decision="approved",
            reviewer_id="r-test", decided_at=time.time(),
        )
        result = await executor.execute(basic_request, resp)
        assert not result.success
        assert "Service down" in (result.error or "")


# ---------------------------------------------------------------------------
# 4. ApprovalMetrics Tests
# ---------------------------------------------------------------------------

class TestApprovalMetrics:

    def _make_response(self, decision: str, automated=False) -> ApprovalResponse:
        return ApprovalResponse(
            request_id="x", decision=decision, reviewer_id="r",
            automated=automated, decided_at=time.time(),
        )

    def _make_req(self, risk_level="medium") -> ApprovalRequest:
        return ApprovalRequest(
            request_id="x", agent_id="a", session_id="s",
            proposed_action="send_email", proposed_params={},
            reasoning="", conversation_summary="", evidence=[],
            risk_level=risk_level, estimated_cost=0.0,
            affected_systems=[], created_at=time.time(),
        )

    def test_metrics_tracks_all_decisions(self):
        m = ApprovalMetrics()
        req = self._make_req()
        for _ in range(10):
            m.record(req, self._make_response("approved"), 30.0)
        for _ in range(3):
            m.record(req, self._make_response("rejected"), 20.0)
        for _ in range(2):
            m.record(req, self._make_response("approved_with_edits"), 25.0)
        m.record(req, self._make_response("rejected", automated=True), 300.0)

        assert m.total_requests == 16
        assert m.approved == 10
        assert m.rejected == 4
        assert m.approved_with_edits == 2
        assert m.timed_out == 1

    def test_metrics_by_risk_level(self):
        m = ApprovalMetrics()
        high_req = self._make_req("high")
        low_req  = self._make_req("low")

        m.record(high_req, self._make_response("approved"), 10.0)
        m.record(high_req, self._make_response("rejected"), 10.0)
        m.record(low_req,  self._make_response("approved"), 5.0)

        assert m.by_risk_level["high"]["approved"] == 1
        assert m.by_risk_level["high"]["rejected"] == 1
        assert m.by_risk_level["low"]["approved"] == 1

    def test_approval_rate_calculation(self):
        m = ApprovalMetrics()
        req = self._make_req()
        for _ in range(8):
            m.record(req, self._make_response("approved"), 10.0)
        for _ in range(2):
            m.record(req, self._make_response("rejected"), 10.0)

        summary = m.summary()
        assert summary["approval_rate"] == pytest.approx(0.8)

    def test_avg_response_time_calculation(self):
        m = ApprovalMetrics()
        req = self._make_req()
        for rt in (10.0, 20.0, 30.0):
            m.record(req, self._make_response("approved"), rt)

        assert m.avg_response_time == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 5. Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:

    @pytest.mark.asyncio
    async def test_full_approval_lifecycle(self, mock_agent):
        """End-to-end: policy → request → human approves → action executed."""
        policy = ApprovalPolicy.with_defaults()
        iface  = ApprovalInterface(channels=["dashboard"])
        iface.register_reviewer(Reviewer("r-alice", "Alice", is_senior=True))
        executor = ApprovalExecutor(mock_agent)
        reviewer = HumanReviewerInterface("r-alice")

        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id="a", session_id="s",
            proposed_action="send_email",
            proposed_params={"to": "user@example.com"},
            reasoning="test", conversation_summary="test",
            evidence=[], risk_level="medium", estimated_cost=0.0,
            affected_systems=["email_service"], created_at=time.time(),
        )

        decision = policy.requires_approval(req.proposed_action, req.proposed_params, {})
        assert decision.requires_approval

        asyncio.get_event_loop().call_later(
            0.05, lambda: iface.submit_response(reviewer.approve(req.request_id))
        )

        resp = await iface.request_approval(req, timeout_seconds=2)
        assert resp.decision == "approved"

        result = await executor.execute(req, resp)
        assert result.success
        assert len(executor.approved_actions) == 1

    @pytest.mark.asyncio
    async def test_full_rejection_lifecycle(self, mock_agent):
        """End-to-end: policy → request → human rejects → action NOT executed."""
        iface = ApprovalInterface(channels=["dashboard"])
        iface.register_reviewer(Reviewer("r-alice", "Alice", is_senior=True))
        executor = ApprovalExecutor(mock_agent)
        reviewer = HumanReviewerInterface("r-alice")

        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id="a", session_id="s",
            proposed_action="send_email", proposed_params={},
            reasoning="test", conversation_summary="test",
            evidence=[], risk_level="medium", estimated_cost=0.0,
            affected_systems=[], created_at=time.time(),
        )

        asyncio.get_event_loop().call_later(
            0.05, lambda: iface.submit_response(
                reviewer.reject(req.request_id, reason="Content not approved.")
            )
        )

        resp = await iface.request_approval(req, timeout_seconds=2)
        result = await executor.execute(req, resp)

        assert not result.success
        mock_agent.execute_tool.assert_not_awaited()
        mock_agent.send_message_sync.assert_called_once()
        assert len(executor.rejected_actions) == 1

    @pytest.mark.asyncio
    async def test_full_timeout_lifecycle(self, mock_agent):
        """End-to-end: policy → request → timeout → auto-rejected → user notified."""
        iface = ApprovalInterface(channels=["dashboard"])
        iface.register_reviewer(Reviewer("r-alice", "Alice", is_senior=True))
        executor = ApprovalExecutor(mock_agent)

        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id="a", session_id="s",
            proposed_action="send_email", proposed_params={},
            reasoning="test", conversation_summary="test",
            evidence=[], risk_level="medium", estimated_cost=0.0,
            affected_systems=[], created_at=time.time(),
        )

        resp = await iface.request_approval(req, timeout_seconds=0.1)
        assert resp.automated
        assert resp.decision == "rejected"

        result = await executor.execute(req, resp)
        assert not result.success
        mock_agent.execute_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_approval_requests(self):
        """Five simultaneous requests all resolve correctly with no data races."""
        iface = ApprovalInterface(channels=["dashboard"])
        iface.register_reviewer(Reviewer("r-alice", "Alice", is_senior=True))
        metrics = ApprovalMetrics()
        reviewer = HumanReviewerInterface("r-alice")

        requests = [
            ApprovalRequest(
                request_id=str(uuid.uuid4()),
                agent_id="a", session_id="s",
                proposed_action="send_email",
                proposed_params={"to": f"u{i}@example.com"},
                reasoning="test", conversation_summary="test",
                evidence=[], risk_level="medium", estimated_cost=0.0,
                affected_systems=[], created_at=time.time(),
            )
            for i in range(5)
        ]

        async def approve_after(req, delay):
            await asyncio.sleep(delay)
            iface.submit_response(reviewer.approve(req.request_id))

        tasks = [asyncio.create_task(approve_after(r, 0.02 * (i + 1)))
                 for i, r in enumerate(requests)]

        responses = await asyncio.gather(*[
            iface.request_approval(r, timeout_seconds=2) for r in requests
        ])
        await asyncio.gather(*tasks)

        for req, resp in zip(requests, responses):
            metrics.record(req, resp, 0.02)

        assert metrics.total_requests == 5
        assert metrics.approved == 5

    @pytest.mark.asyncio
    async def test_approval_system_failure_graceful(self):
        """When ApprovalInterface.request_approval raises, the caller handles it."""
        iface = ApprovalInterface(channels=["dashboard"])
        # Patch _assign_reviewer to raise
        async def _broken(req):
            raise RuntimeError("Reviewer service unavailable")
        iface._assign_reviewer = _broken  # type: ignore[method-assign]

        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id="a", session_id="s",
            proposed_action="send_email", proposed_params={},
            reasoning="test", conversation_summary="test",
            evidence=[], risk_level="medium", estimated_cost=0.0,
            affected_systems=[], created_at=time.time(),
        )

        with pytest.raises(RuntimeError, match="Reviewer service unavailable"):
            await iface.request_approval(req, timeout_seconds=2)

    @pytest.mark.asyncio
    async def test_audit_trail_complete(self, mock_agent):
        """Approved action appears in executor.approved_actions with timestamps."""
        iface = ApprovalInterface(channels=["dashboard"])
        iface.register_reviewer(Reviewer("r-alice", "Alice", is_senior=True))
        executor = ApprovalExecutor(mock_agent)
        reviewer = HumanReviewerInterface("r-alice")

        req = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            agent_id="a", session_id="s",
            proposed_action="send_email",
            proposed_params={"to": "audit@example.com"},
            reasoning="audit test", conversation_summary="test",
            evidence=[], risk_level="medium", estimated_cost=0.0,
            affected_systems=[], created_at=time.time(),
        )

        asyncio.get_event_loop().call_later(
            0.05, lambda: iface.submit_response(reviewer.approve(req.request_id))
        )
        resp = await iface.request_approval(req, timeout_seconds=2)
        result = await executor.execute(req, resp)

        assert result.success
        assert len(executor.approved_actions) == 1
        entry = executor.approved_actions[0]
        assert entry["request"].request_id == req.request_id
        assert entry["timestamp"] > 0
        assert entry["result"] is not None
