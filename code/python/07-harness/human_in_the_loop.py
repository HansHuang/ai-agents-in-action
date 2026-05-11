"""
Human-in-the-Loop Approval System
==================================
Complete implementation of the propose → review → decide → execute workflow
with conditional approval policies, multi-channel reviewer interface,
timeout handling, and full audit trail.

Companion to: docs/07-harness-engineering/06-human-in-the-loop.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """An action proposed by the agent that requires human approval."""

    request_id: str
    agent_id: str
    session_id: str
    proposed_action: str
    proposed_params: dict
    reasoning: str
    conversation_summary: str
    evidence: list[dict]
    risk_level: str           # "low" | "medium" | "high" | "critical"
    estimated_cost: float
    affected_systems: list[str]
    created_at: float
    deadline: float | None = None   # epoch seconds; None = no hard deadline

    def to_human_readable(self) -> str:
        """Format the request for display to a human reviewer."""
        deadline_str = (
            f"Deadline: {time.strftime('%H:%M:%S', time.localtime(self.deadline))}"
            if self.deadline else "No deadline"
        )
        evidence_lines = "\n".join(
            f"  - {json.dumps(e)}" for e in self.evidence[:5]
        )
        return (
            f"\n{'═' * 45}\n"
            f"APPROVAL REQUIRED\n"
            f"{'═' * 45}\n\n"
            f"Request ID  : {self.request_id}\n"
            f"Action      : {self.proposed_action}\n"
            f"Risk Level  : {self.risk_level.upper()}\n"
            f"Estimated $  : ${self.estimated_cost:.2f}\n"
            f"Systems     : {', '.join(self.affected_systems)}\n"
            f"{deadline_str}\n\n"
            f"Parameters:\n{json.dumps(self.proposed_params, indent=2)}\n\n"
            f"Reasoning:\n{self.reasoning}\n\n"
            f"Conversation context:\n{self.conversation_summary[:500]}\n\n"
            f"Evidence:\n{evidence_lines or '  (none)'}\n\n"
            f"{'═' * 45}\n"
            f"Approve? [Y/N/Edit]  Timeout: {self._timeout_str()}\n"
            f"{'═' * 45}\n"
        )

    def _timeout_str(self) -> str:
        if self.deadline is None:
            return "none"
        remaining = max(0, self.deadline - time.time())
        m, s = divmod(int(remaining), 60)
        return f"{m}:{s:02d}"


@dataclass
class ApprovalResponse:
    """A human reviewer's decision on an approval request."""

    request_id: str
    decision: str           # "approved" | "rejected" | "approved_with_edits"
    reviewer_id: str
    reviewer_notes: str | None = None
    edited_params: dict | None = None
    automated: bool = False
    decided_at: float = field(default_factory=time.time)
    reason: str | None = None


@dataclass
class ExecutionResult:
    """The outcome of executing (or not executing) an approved action."""

    success: bool
    result: Any = None
    error: str | None = None


@dataclass
class ApprovalDecision:
    """Whether an action requires human approval and at what risk level."""

    requires_approval: bool
    rule: str
    risk_level: str
    timeout_seconds: float = 300.0


@dataclass
class ApprovalRule:
    """A single conditional rule that determines when approval is required."""

    name: str
    description: str
    priority: int
    risk_level: str
    timeout_seconds: float = 300.0
    actions: list[str] | None = None
    min_cost: float | None = None
    affected_systems: list[str] | None = None
    user_roles: list[str] | None = None

    def matches(self, action: str, params: dict, context: dict) -> bool:
        """Return True when this rule applies to the given action/context."""
        if self.actions and action not in self.actions:
            return False

        if self.min_cost is not None:
            cost = self._estimate_cost(action, params)
            if cost < self.min_cost:
                return False

        if self.affected_systems:
            action_systems = self._get_affected_systems(action, params)
            if not any(s in self.affected_systems for s in action_systems):
                return False

        if self.user_roles:
            if context.get("role") not in self.user_roles:
                return False

        return True

    # -- helpers -----------------------------------------------------------

    def _estimate_cost(self, action: str, params: dict) -> float:
        if action == "issue_refund":
            return float(params.get("amount", 0))
        if action == "send_email":
            return 0.0
        if action in ("update_database", "delete_record"):
            return 50.0
        if action == "cancel_subscription":
            return float(params.get("monthly_value", 0))
        if action == "export_user_data":
            return 0.0
        return 10.0

    def _get_affected_systems(self, action: str, _params: dict) -> list[str]:
        system_map: dict[str, list[str]] = {
            "send_email":          ["email_service"],
            "issue_refund":        ["payment_processor", "order_database"],
            "update_database":     ["database"],
            "delete_record":       ["database"],
            "create_ticket":       ["support_system"],
            "cancel_subscription": ["billing_system", "subscription_service"],
            "export_user_data":    ["data_warehouse", "gdpr_service"],
        }
        return system_map.get(action, ["unknown"])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class ApprovalPolicy:
    """
    Ordered set of rules that determine when human approval is required.
    Rules are evaluated highest-priority-first; first match wins.
    If no rule matches, the action is auto-approved (default_allow).
    """

    def __init__(self) -> None:
        self.rules: list[ApprovalRule] = []

    def add_rule(self, rule: ApprovalRule) -> None:
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def requires_approval(
        self, action: str, params: dict, context: dict
    ) -> ApprovalDecision:
        for rule in self.rules:
            if rule.matches(action, params, context):
                return ApprovalDecision(
                    requires_approval=True,
                    rule=rule.name,
                    risk_level=rule.risk_level,
                    timeout_seconds=rule.timeout_seconds,
                )
        return ApprovalDecision(
            requires_approval=False,
            rule="default_allow",
            risk_level="none",
        )

    @classmethod
    def with_defaults(cls) -> "ApprovalPolicy":
        """Build the standard set of approval rules for a customer-service agent."""
        policy = cls()

        policy.add_rule(ApprovalRule(
            name="high_value_refund",
            description="Refunds over $500 require human approval",
            priority=100,
            risk_level="high",
            actions=["issue_refund"],
            min_cost=500.0,
            timeout_seconds=600,
        ))
        policy.add_rule(ApprovalRule(
            name="external_communication",
            description="Sending email to customers requires approval",
            priority=90,
            risk_level="medium",
            actions=["send_email"],
            timeout_seconds=300,
        ))
        policy.add_rule(ApprovalRule(
            name="subscription_cancellation",
            description="Cancelling subscriptions requires approval",
            priority=85,
            risk_level="high",
            actions=["cancel_subscription"],
            timeout_seconds=600,
        ))
        policy.add_rule(ApprovalRule(
            name="database_modification",
            description="Any database write requires approval",
            priority=80,
            risk_level="medium",
            actions=["update_database", "delete_record"],
            timeout_seconds=300,
        ))
        policy.add_rule(ApprovalRule(
            name="critical_data_export",
            description="Exporting user data requires approval",
            priority=95,
            risk_level="high",
            actions=["export_user_data"],
            timeout_seconds=600,
        ))

        return policy


# ---------------------------------------------------------------------------
# Reviewer abstractions
# ---------------------------------------------------------------------------

@dataclass
class Reviewer:
    reviewer_id: str
    name: str
    is_senior: bool = False
    expertise: list[str] = field(default_factory=list)
    slack_id: str | None = None
    email: str | None = None


class ReviewerGroup:
    """Wraps multiple reviewers for critical-risk consensus approval."""

    def __init__(self, reviewers: list[Reviewer]) -> None:
        self.reviewers = reviewers

    @property
    def id(self) -> str:
        return ",".join(r.reviewer_id for r in self.reviewers)


# ---------------------------------------------------------------------------
# Human Reviewer Interface
# ---------------------------------------------------------------------------

class HumanReviewerInterface:
    """Interface for a human reviewer to respond to pending approval requests."""

    def __init__(self, reviewer_id: str) -> None:
        self.reviewer_id = reviewer_id

    def approve(self, request_id: str, notes: str | None = None) -> ApprovalResponse:
        """Approve the request as-is."""
        return ApprovalResponse(
            request_id=request_id,
            decision="approved",
            reviewer_id=self.reviewer_id,
            reviewer_notes=notes,
        )

    def reject(self, request_id: str, reason: str) -> ApprovalResponse:
        """Reject the request with a mandatory reason."""
        return ApprovalResponse(
            request_id=request_id,
            decision="rejected",
            reviewer_id=self.reviewer_id,
            reason=reason,
        )

    def approve_with_edits(
        self,
        request_id: str,
        edited_params: dict,
        notes: str | None = None,
    ) -> ApprovalResponse:
        """Approve the action but with modified parameters."""
        return ApprovalResponse(
            request_id=request_id,
            decision="approved_with_edits",
            reviewer_id=self.reviewer_id,
            edited_params=edited_params,
            reviewer_notes=notes,
        )


# ---------------------------------------------------------------------------
# Approval Interface  (async, multi-channel)
# ---------------------------------------------------------------------------

class ApprovalInterface:
    """
    Routes approval requests to human reviewers via the configured channels
    (dashboard, slack, email).  Auto-rejects on timeout (safe default).
    """

    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels: list[str] = channels or ["dashboard"]
        self.pending_requests: dict[str, ApprovalRequest] = {}
        self.reviewers: dict[str, Reviewer] = {}
        self._response_events: dict[str, asyncio.Event] = {}
        self._responses: dict[str, ApprovalResponse] = {}

    # -- public API --------------------------------------------------------

    def register_reviewer(self, reviewer: Reviewer) -> None:
        self.reviewers[reviewer.reviewer_id] = reviewer

    async def request_approval(
        self,
        request: ApprovalRequest,
        timeout_seconds: float = 300.0,
    ) -> ApprovalResponse:
        """
        Send *request* to the appropriate human reviewer and wait up to
        *timeout_seconds* for a response.  Auto-rejects on timeout.
        """
        reviewer = await self._assign_reviewer(request)
        await self._send_to_reviewer(reviewer, request)
        self.pending_requests[request.request_id] = request

        try:
            response = await asyncio.wait_for(
                self._wait_for_reviewer_response(reviewer, request.request_id),
                timeout=timeout_seconds,
            )
            return response

        except asyncio.TimeoutError:
            logger.warning(
                "Approval %s timed out after %.0fs — auto-rejecting.",
                request.request_id,
                timeout_seconds,
            )
            self.pending_requests.pop(request.request_id, None)
            return ApprovalResponse(
                request_id=request.request_id,
                decision="rejected",
                reason="Approval timeout — automatically rejected for safety.",
                reviewer_id=getattr(reviewer, "id", "system"),
                automated=True,
            )

    def submit_response(self, response: ApprovalResponse) -> None:
        """
        Called by a reviewer (or a test) to deliver their decision.
        Unblocks the corresponding _wait_for_reviewer_response coroutine.
        """
        self._responses[response.request_id] = response
        event = self._response_events.get(response.request_id)
        if event:
            event.set()

    # -- reviewer routing --------------------------------------------------

    async def _assign_reviewer(
        self, request: ApprovalRequest
    ) -> Reviewer | ReviewerGroup:
        if request.risk_level == "critical":
            seniors = [r for r in self.reviewers.values() if r.is_senior]
            if len(seniors) >= 2:
                return ReviewerGroup(seniors[:2])
            # fallback: use all available
            return ReviewerGroup(list(self.reviewers.values())[:2] or list(self.reviewers.values()))
        if request.risk_level == "high":
            seniors = [r for r in self.reviewers.values() if r.is_senior]
            return seniors[0] if seniors else self._any_reviewer()
        if request.risk_level == "medium":
            experts = [
                r for r in self.reviewers.values()
                if request.proposed_action in r.expertise
            ]
            return experts[0] if experts else self._any_reviewer()
        return self._any_reviewer()

    def _any_reviewer(self) -> Reviewer:
        if not self.reviewers:
            return Reviewer(reviewer_id="unassigned", name="Unassigned")
        return next(iter(self.reviewers.values()))

    # -- channel dispatch --------------------------------------------------

    async def _send_to_reviewer(
        self, reviewer: Reviewer | ReviewerGroup, request: ApprovalRequest
    ) -> None:
        message = request.to_human_readable()
        reviewers = reviewer.reviewers if isinstance(reviewer, ReviewerGroup) else [reviewer]

        for rev in reviewers:
            if "slack" in self.channels and rev.slack_id:
                await self._send_slack_message(rev.slack_id, message)
            if "email" in self.channels and rev.email and request.risk_level in ("high", "critical"):
                await self._send_email(
                    rev.email,
                    f"URGENT: Approval Required — {request.proposed_action}",
                    message,
                )
        if "dashboard" in self.channels:
            logger.info("[DASHBOARD] New pending request: %s", request.request_id)

    async def _send_slack_message(self, slack_id: str, message: str) -> None:
        logger.info("[SLACK → %s] %s", slack_id, message[:80] + "…")

    async def _send_email(self, email: str, subject: str, body: str) -> None:
        logger.info("[EMAIL → %s] %s", email, subject)

    # -- response wait -----------------------------------------------------

    async def _wait_for_reviewer_response(
        self,
        reviewer: Reviewer | ReviewerGroup,
        request_id: str,
    ) -> ApprovalResponse:
        if isinstance(reviewer, ReviewerGroup):
            return await self._wait_for_consensus(reviewer, request_id)

        event = asyncio.Event()
        self._response_events[request_id] = event
        await event.wait()
        self._response_events.pop(request_id, None)
        self.pending_requests.pop(request_id, None)
        return self._responses.pop(request_id)

    async def _wait_for_consensus(
        self, group: ReviewerGroup, request_id: str
    ) -> ApprovalResponse:
        """Both reviewers in the group must approve; one rejection = rejected."""
        event = asyncio.Event()
        self._response_events[request_id] = event
        approvals: list[ApprovalResponse] = []

        # collect both responses
        for _ in range(len(group.reviewers)):
            await event.wait()
            event.clear()
            latest = self._responses.pop(request_id, None)
            if latest is None:
                continue
            if latest.decision == "rejected":
                self._response_events.pop(request_id, None)
                self.pending_requests.pop(request_id, None)
                return latest
            approvals.append(latest)

        self._response_events.pop(request_id, None)
        self.pending_requests.pop(request_id, None)
        # all approved — return last response
        return approvals[-1] if approvals else ApprovalResponse(
            request_id=request_id,
            decision="rejected",
            reviewer_id="system",
            reason="No approvals received.",
            automated=True,
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ApprovalExecutor:
    """Carries out (or records the rejection of) an approved action."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.approved_actions: list[dict] = []
        self.rejected_actions: list[dict] = []

    async def execute(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> ExecutionResult:
        if response.decision == "approved":
            return await self._execute_approved(request)
        if response.decision == "approved_with_edits":
            assert response.edited_params is not None
            return await self._execute_edited(request, response.edited_params)
        if response.decision == "rejected":
            return self._handle_rejection(request, response)
        raise ValueError(f"Unknown decision: {response.decision!r}")

    async def _execute_approved(self, request: ApprovalRequest) -> ExecutionResult:
        logger.info("Executing approved action: %s", request.proposed_action)
        try:
            result = await self.agent.execute_tool(
                tool_name=request.proposed_action,
                params=request.proposed_params,
            )
            await self.agent.send_message(
                f"Completed: {request.proposed_action}"
            )
            self.approved_actions.append(
                {"request": request, "result": result, "timestamp": time.time()}
            )
            return ExecutionResult(success=True, result=result)
        except Exception as exc:
            logger.error("Approved action failed: %s", exc)
            return ExecutionResult(success=False, error=str(exc))

    async def _execute_edited(
        self, request: ApprovalRequest, edited_params: dict
    ) -> ExecutionResult:
        logger.info(
            "Executing edited action: %s with %s",
            request.proposed_action, edited_params,
        )
        try:
            result = await self.agent.execute_tool(
                tool_name=request.proposed_action,
                params=edited_params,
            )
            await self.agent.send_message(
                "Completed with the adjustments you specified."
            )
            self.approved_actions.append(
                {"request": request, "result": result,
                 "edited_params": edited_params, "timestamp": time.time()}
            )
            return ExecutionResult(success=True, result=result)
        except Exception as exc:
            logger.error("Edited action failed: %s", exc)
            return ExecutionResult(success=False, error=str(exc))

    def _handle_rejection(
        self, request: ApprovalRequest, response: ApprovalResponse
    ) -> ExecutionResult:
        reason = response.reason or "This action requires additional review."
        logger.info(
            "Action rejected: %s. Reason: %s", request.proposed_action, reason
        )
        self.agent.send_message_sync(
            f"I wasn't able to complete '{request.proposed_action}'. {reason}"
        )
        self.rejected_actions.append(
            {"request": request, "response": response, "timestamp": time.time()}
        )
        return ExecutionResult(
            success=False,
            error=f"Rejected by reviewer: {reason}",
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ApprovalMetrics:
    """Rolling metrics for the human-in-the-loop system."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.approved: int = 0
        self.rejected: int = 0
        self.approved_with_edits: int = 0
        self.timed_out: int = 0
        self._total_response_time: float = 0.0
        self.by_risk_level: dict[str, dict[str, int]] = defaultdict(
            lambda: {"approved": 0, "rejected": 0, "approved_with_edits": 0}
        )

    @property
    def avg_response_time(self) -> float:
        n = self.approved + self.rejected + self.approved_with_edits
        return self._total_response_time / n if n > 0 else 0.0

    def record(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
        response_time: float,
    ) -> None:
        self.total_requests += 1
        self._total_response_time += response_time

        if response.automated:
            self.timed_out += 1
        
        bucket = self.by_risk_level[request.risk_level]

        if response.decision == "approved":
            self.approved += 1
            bucket["approved"] += 1
        elif response.decision == "rejected":
            self.rejected += 1
            bucket["rejected"] += 1
        elif response.decision == "approved_with_edits":
            self.approved_with_edits += 1
            bucket["approved_with_edits"] = bucket.get("approved_with_edits", 0) + 1

    def summary(self) -> dict:
        total = max(self.total_requests, 1)
        return {
            "total_approval_requests": self.total_requests,
            "approved": self.approved,
            "rejected": self.rejected,
            "approved_with_edits": self.approved_with_edits,
            "timed_out": self.timed_out,
            "approval_rate": self.approved / total,
            "rejection_rate": self.rejected / total,
            "edit_rate": self.approved_with_edits / total,
            "timeout_rate": self.timed_out / total,
            "avg_response_time_seconds": self.avg_response_time,
            "by_risk_level": dict(self.by_risk_level),
        }


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------

class MockAgent:
    """Minimal agent stand-in for the demo."""

    def __init__(self, name: str = "demo-agent") -> None:
        self.name = name
        self.messages: list[str] = []
        self.tool_calls: list[dict] = []

    async def execute_tool(self, tool_name: str, params: dict) -> dict:
        self.tool_calls.append({"tool": tool_name, "params": params})
        return {"status": "ok", "tool": tool_name, "params": params}

    async def send_message(self, msg: str) -> None:
        print(f"  [Agent → User] {msg}")
        self.messages.append(msg)

    def send_message_sync(self, msg: str) -> None:
        print(f"  [Agent → User] {msg}")
        self.messages.append(msg)


def _make_request(
    action: str,
    params: dict,
    risk_level: str,
    cost: float,
    systems: list[str],
    reasoning: str,
    deadline_in: float | None = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=str(uuid.uuid4()),
        agent_id="demo-agent",
        session_id="session-001",
        proposed_action=action,
        proposed_params=params,
        reasoning=reasoning,
        conversation_summary="Customer wrote: 'Please process my request.'",
        evidence=[{"source": "order_system", "data": params}],
        risk_level=risk_level,
        estimated_cost=cost,
        affected_systems=systems,
        created_at=time.time(),
        deadline=time.time() + deadline_in if deadline_in else None,
    )


async def run_demo() -> None:
    print("\n" + "=" * 60)
    print("  HUMAN-IN-THE-LOOP DEMO")
    print("=" * 60)

    # Setup
    policy   = ApprovalPolicy.with_defaults()
    interface = ApprovalInterface(channels=["dashboard"])
    agent    = MockAgent()
    executor = ApprovalExecutor(agent)
    metrics  = ApprovalMetrics()

    # Register reviewers
    alice = Reviewer("r-alice", "Alice", is_senior=True, expertise=["send_email"])
    bob   = Reviewer("r-bob",   "Bob",   is_senior=True, expertise=["issue_refund"])
    carol = Reviewer("r-carol", "Carol", is_senior=False, expertise=["send_email"])
    for r in (alice, bob, carol):
        interface.register_reviewer(r)

    alice_reviewer = HumanReviewerInterface("r-alice")
    bob_reviewer   = HumanReviewerInterface("r-bob")
    carol_reviewer = HumanReviewerInterface("r-carol")

    audit_trail: list[dict] = []

    async def process(
        label: str,
        request: ApprovalRequest,
        reviewer_fn,          # coroutine or None
        timeout: float = 10.0,
    ) -> None:
        print(f"\n{'─' * 60}")
        print(f"Scenario: {label}")
        print(f"Action  : {request.proposed_action}  |  Risk: {request.risk_level.upper()}")

        decision = policy.requires_approval(
            request.proposed_action, request.proposed_params, {}
        )

        if not decision.requires_approval:
            print("  → Policy: AUTO-APPROVED (no human needed)")
            result = await agent.execute_tool(
                tool_name=request.proposed_action,
                params=request.proposed_params,
            )
            audit_trail.append({
                "label": label,
                "action": request.proposed_action,
                "decision": "auto_approved",
                "result": result,
            })
            return

        start = time.time()
        # Schedule the reviewer's response (if any)
        if reviewer_fn is not None:
            asyncio.get_event_loop().call_later(
                0.05,
                lambda: interface.submit_response(reviewer_fn(request.request_id)),
            )

        try:
            response = await interface.request_approval(request, timeout_seconds=timeout)
        except Exception as exc:
            print(f"  → Approval system ERROR: {exc}")
            audit_trail.append({
                "label": label,
                "action": request.proposed_action,
                "decision": "system_error",
                "error": str(exc),
            })
            return

        elapsed = time.time() - start
        metrics.record(request, response, elapsed)

        print(f"  → Decision: {response.decision.upper()}"
              f"{'  (automated)' if response.automated else ''}")
        if response.reason:
            print(f"  → Reason  : {response.reason}")
        if response.reviewer_notes:
            print(f"  → Notes   : {response.reviewer_notes}")
        if response.edited_params:
            print(f"  → Edits   : {response.edited_params}")

        result = await executor.execute(request, response)
        print(f"  → Execution: {'OK' if result.success else 'FAILED — ' + str(result.error)}")

        audit_trail.append({
            "label":     label,
            "action":    request.proposed_action,
            "decision":  response.decision,
            "automated": response.automated,
            "success":   result.success,
            "elapsed_s": round(elapsed, 3),
        })

    # ── Scenario 1: Low-risk (auto-approved) ────────────────────────────
    await process(
        "1. Low-risk action (weather lookup)",
        _make_request("get_weather", {"city": "Tokyo"}, "low", 0.0, ["weather_api"],
                      "User asked for Tokyo weather."),
        reviewer_fn=None,
    )

    # ── Scenario 2: Medium-risk → human approves ────────────────────────
    await process(
        "2. Medium-risk (send email) → approved",
        _make_request("send_email", {"to": "user@example.com", "subject": "Update"},
                      "medium", 0.0, ["email_service"], "User asked to send update."),
        reviewer_fn=lambda rid: alice_reviewer.approve(rid, notes="Looks good."),
    )

    # ── Scenario 3: High-risk ($750 refund) → approved with edits ───────
    await process(
        "3. High-risk ($750 refund) → approved with edits ($500)",
        _make_request("issue_refund", {"order_id": "ORD-99", "amount": 750},
                      "high", 750.0, ["payment_processor", "order_database"],
                      "Customer claims item was damaged."),
        reviewer_fn=lambda rid: bob_reviewer.approve_with_edits(
            rid,
            edited_params={"order_id": "ORD-99", "amount": 500},
            notes="Standard partial refund for this category is $500.",
        ),
    )

    # ── Scenario 4: Critical (data export) → two reviewers both approve ─
    req4 = _make_request(
        "export_user_data", {"user_id": "U-42", "format": "csv"},
        "critical", 0.0, ["data_warehouse", "gdpr_service"],
        "User requested full data export (GDPR Article 20).",
    )

    async def _submit_two_approvals(request_id: str) -> None:
        await asyncio.sleep(0.03)
        interface.submit_response(alice_reviewer.approve(request_id, notes="GDPR request verified."))
        await asyncio.sleep(0.01)
        interface.submit_response(bob_reviewer.approve(request_id, notes="Second sign-off."))

    asyncio.get_event_loop().create_task(_submit_two_approvals(req4.request_id))

    print(f"\n{'─' * 60}")
    print("Scenario: 4. Critical (data export) → two reviewers both approve")
    print(f"Action  : {req4.proposed_action}  |  Risk: CRITICAL")
    start4 = time.time()
    response4 = await interface.request_approval(req4, timeout_seconds=10.0)
    metrics.record(req4, response4, time.time() - start4)
    result4 = await executor.execute(req4, response4)
    print(f"  → Decision: {response4.decision.upper()}")
    print(f"  → Execution: {'OK' if result4.success else 'FAILED'}")
    audit_trail.append({"label": "4. Critical data export", "decision": response4.decision,
                        "success": result4.success})

    # ── Scenario 5: Timeout → auto-rejected ─────────────────────────────
    await process(
        "5. Approval timeout → auto-rejected",
        _make_request("cancel_subscription", {"subscription_id": "SUB-7"},
                      "high", 99.0, ["billing_system"], "User requested cancellation."),
        reviewer_fn=None,   # nobody responds
        timeout=0.1,        # very short timeout for demo
    )

    # ── Scenario 6: Human rejection ─────────────────────────────────────
    await process(
        "6. Human rejection → user notified",
        _make_request("issue_refund", {"order_id": "ORD-33", "amount": 1200},
                      "high", 1200.0, ["payment_processor"], "Refund requested without evidence."),
        reviewer_fn=lambda rid: bob_reviewer.reject(
            rid, reason="No supporting evidence. Request more documentation."
        ),
    )

    # ── Scenario 7: Approval system failure ─────────────────────────────
    print(f"\n{'─' * 60}")
    print("Scenario: 7. Approval system failure → graceful handling")
    broken_interface = ApprovalInterface(channels=["dashboard"])
    # Intentionally break it: no reviewers registered, will get a minimal reviewer
    req7 = _make_request("update_database", {"table": "users", "set": {"flag": True}},
                         "medium", 50.0, ["database"], "Batch flag update.")
    # Simulate failure by monkey-patching
    async def _failing_assign(_req):
        raise RuntimeError("Reviewer service unavailable")
    broken_interface._assign_reviewer = _failing_assign  # type: ignore[method-assign]

    try:
        await broken_interface.request_approval(req7, timeout_seconds=2.0)
        print("  → (unexpected success)")
    except Exception as exc:
        print(f"  → Caught error: {exc}")
        print("  → Graceful: user would be notified to retry later.")
        audit_trail.append({"label": "7. System failure", "decision": "system_error"})

    # ── Scenario 8: Concurrent approvals ────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Scenario: 8. Concurrent approvals → all handled correctly")

    reqs = [
        _make_request("send_email", {"to": f"u{i}@example.com"}, "medium", 0.0,
                      ["email_service"], f"Concurrent request {i}.")
        for i in range(3)
    ]

    async def _approve_after(req: ApprovalRequest, delay: float) -> None:
        await asyncio.sleep(delay)
        interface.submit_response(carol_reviewer.approve(req.request_id))

    tasks = [
        asyncio.create_task(_approve_after(r, 0.05 * (i + 1)))
        for i, r in enumerate(reqs)
    ]
    responses = await asyncio.gather(*[
        interface.request_approval(r, timeout_seconds=5.0) for r in reqs
    ])
    await asyncio.gather(*tasks)

    for i, (req, resp) in enumerate(zip(reqs, responses)):
        metrics.record(req, resp, 0.05 * (i + 1))
        print(f"  Request {i+1}: {resp.decision}")
    audit_trail.append({"label": "8. Concurrent approvals",
                        "decisions": [r.decision for r in responses]})

    # ── Audit trail ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FULL AUDIT TRAIL")
    print("=" * 60)
    for i, entry in enumerate(audit_trail, 1):
        print(f"  {i:2d}. {entry.get('label', '?'):50s}  "
              f"decision={entry.get('decision', '?')}")

    # ── Metrics ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("APPROVAL METRICS SUMMARY")
    print("=" * 60)
    summary = metrics.summary()
    for key, val in summary.items():
        if key != "by_risk_level":
            print(f"  {key:35s}: {val}")
    print("  by_risk_level:")
    for level, counts in summary["by_risk_level"].items():
        print(f"    {level:10s}: {counts}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_demo())
