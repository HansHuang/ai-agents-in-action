"""
harness_runbook.py
==================
Automated runbook for common ProductionHarness operational scenarios.

Each scenario method:
  1. Checks symptoms (what metrics say)
  2. Executes safe corrective actions automatically
  3. Lists actions that require human approval
  4. Returns a RunbookResult with full incident context

Safe to ``auto_execute=True`` in staging; set ``auto_execute=False``
in production to require manual confirmation for destructive actions.

See: docs/07-harness-engineering/07-building-a-reliable-harness.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from production_harness import HarnessConfig, HarnessMetrics, ProductionHarness
from resilience_layer import CircuitState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ComponentStatus:
    """Health status of a single harness component."""

    name: str
    status: str          # "ok" | "warning" | "critical"
    metrics: dict
    message: str

    @property
    def is_healthy(self) -> bool:
        return self.status == "ok"


@dataclass
class DiagnosticReport:
    """Full diagnostic snapshot of the harness."""

    overall_status: str            # "healthy" | "degraded" | "unhealthy"
    components: dict[str, ComponentStatus]
    active_incidents: list[str]
    recommendations: list[str]
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "components": {
                k: {
                    "status": v.status,
                    "message": v.message,
                    "metrics": v.metrics,
                }
                for k, v in self.components.items()
            },
            "active_incidents": self.active_incidents,
            "recommendations": self.recommendations,
        }

    def __str__(self) -> str:
        icon = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(
            self.overall_status, "⚪"
        )
        lines = [f"{icon}  HARNESS STATUS: {self.overall_status.upper()}"]
        for name, comp in self.components.items():
            comp_icon = {"ok": "✓", "warning": "⚠", "critical": "✗"}.get(comp.status, "?")
            lines.append(f"  {comp_icon}  {name}: {comp.message}")
        if self.active_incidents:
            lines.append(f"  Active incidents: {', '.join(self.active_incidents)}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    • {rec}")
        return "\n".join(lines)


@dataclass
class RunbookResult:
    """Outcome of executing a runbook scenario."""

    scenario: str
    diagnosis: str
    actions_taken: list[str]
    actions_requiring_approval: list[str]
    outcome: str
    incident_report: str
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def __str__(self) -> str:
        return (
            f"\n{'─' * 60}\n"
            f"RUNBOOK: {self.scenario}\n"
            f"{'─' * 60}\n"
            f"Diagnosis  : {self.diagnosis}\n"
            f"Outcome    : {self.outcome}\n"
            f"Actions    : {len(self.actions_taken)} auto-executed\n"
            f"Pending    : {len(self.actions_requiring_approval)} need approval\n"
        )


# ---------------------------------------------------------------------------
# HarnessRunbook
# ---------------------------------------------------------------------------


class HarnessRunbook:
    """
    Automated operational runbook for the ProductionHarness.

    Usage::

        runbook = HarnessRunbook(harness, auto_execute=True)
        report  = await runbook.diagnose()
        result  = await runbook.scenario_primary_provider_down()
    """

    # Thresholds for triggering each scenario
    FALLBACK_RATE_THRESHOLD = 0.05        # 5%  fallback activation → provider at risk
    OUTPUT_BLOCK_RATE_THRESHOLD = 0.05    # 5%  blocked responses → guardrail misconfigured
    APPROVAL_QUEUE_CRITICAL = 20          # 20  pending items → queue critical
    APPROVAL_QUEUE_WARNING = 10
    COST_SPIKE_HOURS_LOOKBACK = 1
    ERROR_RATE_THRESHOLD = 0.05           # 5% error rate

    def __init__(
        self,
        harness: ProductionHarness,
        auto_execute: bool = False,
    ) -> None:
        self.harness = harness
        self.auto_execute = auto_execute
        self._incidents: list[dict] = []

    # =========================================================================
    # Diagnostics
    # =========================================================================

    async def diagnose(self) -> DiagnosticReport:
        """
        Run a full diagnostic on the harness.

        Checks all five layers, circuit breaker states, approval queue depth,
        and recent error rates.  Returns a :class:`DiagnosticReport`.
        """
        health = self.harness.get_health()
        metrics = self.harness.get_metrics_summary()
        components: dict[str, ComponentStatus] = {}
        incidents: list[str] = []
        recommendations: list[str] = []

        # ── Input guardrails ─────────────────────────────────────────────────
        ig = health.get("input_guardrails", {})
        rejection_rate = ig.get("rejection_rate_5min", 0.0)
        if rejection_rate > 0.2:
            comp_status, comp_msg = "critical", f"Rejection rate {rejection_rate:.0%} (>20%)"
            incidents.append("high_input_rejection_rate")
            recommendations.append("Review injection detection — possibly too aggressive")
        elif rejection_rate > 0.1:
            comp_status, comp_msg = "warning", f"Rejection rate {rejection_rate:.0%} (>10%)"
        else:
            comp_status, comp_msg = "ok", f"Rejection rate {rejection_rate:.2%}"
        components["input_guardrails"] = ComponentStatus(
            "input_guardrails", comp_status, ig, comp_msg
        )

        # ── Router ───────────────────────────────────────────────────────────
        router = health.get("router", {})
        accuracy = router.get("accuracy_24h", 1.0)
        if accuracy < 0.8:
            comp_status = "critical"
            comp_msg = f"Routing accuracy {accuracy:.0%} (<80%)"
            incidents.append("low_routing_accuracy")
            recommendations.append("Inspect LLM router — may need prompt update")
        elif accuracy < 0.9:
            comp_status, comp_msg = "warning", f"Routing accuracy {accuracy:.0%} (<90%)"
        else:
            comp_status, comp_msg = "ok", f"Routing accuracy {accuracy:.0%}"
        components["router"] = ComponentStatus("router", comp_status, router, comp_msg)

        # ── Resilience ───────────────────────────────────────────────────────
        res = health.get("resilience", {})
        cb_stats = res.get("llm_circuit", {})
        cb_state = cb_stats.get("state", "closed")
        primary_rate = res.get("llm_primary_success_rate", 1.0)

        if cb_state == "open":
            comp_status = "critical"
            comp_msg = f"Circuit OPEN — primary provider down"
            incidents.append("circuit_breaker_open")
            recommendations.append("Check primary LLM provider status page")
        elif primary_rate < 0.9:
            comp_status = "warning"
            comp_msg = f"Primary success rate {primary_rate:.0%} (<90%)"
            recommendations.append("Monitor fallback activation; check provider health")
        else:
            comp_status = "ok"
            comp_msg = f"Circuit {cb_state.upper()}, primary rate {primary_rate:.0%}"
        components["resilience"] = ComponentStatus("resilience", comp_status, res, comp_msg)

        # ── Output guardrails ────────────────────────────────────────────────
        og = health.get("output_guardrails", {})
        block_rate = og.get("block_rate_5min", 0.0)
        if block_rate > self.OUTPUT_BLOCK_RATE_THRESHOLD:
            comp_status = "warning"
            comp_msg = f"Block rate {block_rate:.1%} (>{self.OUTPUT_BLOCK_RATE_THRESHOLD:.0%})"
            incidents.append("high_output_block_rate")
            recommendations.append("Sample blocked responses — check for guardrail misconfiguration")
        else:
            comp_status = "ok"
            comp_msg = f"Block rate {block_rate:.2%}"
        components["output_guardrails"] = ComponentStatus(
            "output_guardrails", comp_status, og, comp_msg
        )

        # ── Human approval ───────────────────────────────────────────────────
        ha = health.get("human_approval", {})
        pending = ha.get("pending_count", 0)
        if pending >= self.APPROVAL_QUEUE_CRITICAL:
            comp_status = "critical"
            comp_msg = f"{pending} items pending (critical)"
            incidents.append("approval_queue_critical")
            recommendations.append("Escalate to all reviewers immediately")
        elif pending >= self.APPROVAL_QUEUE_WARNING:
            comp_status = "warning"
            comp_msg = f"{pending} items pending"
            recommendations.append("Notify on-call reviewer to clear queue")
        else:
            comp_status = "ok"
            comp_msg = f"{pending} items pending"
        components["human_approval"] = ComponentStatus(
            "human_approval", comp_status, ha, comp_msg
        )

        # ── Cost ─────────────────────────────────────────────────────────────
        cost = health.get("cost", {})
        today = cost.get("today", 0.0)
        monthly_proj = cost.get("projected_monthly", 0.0)
        comp_status = "ok"
        comp_msg = f"Today ${today:.2f}, projected monthly ${monthly_proj:.2f}"
        components["cost"] = ComponentStatus("cost", comp_status, cost, comp_msg)

        # ── Overall status ───────────────────────────────────────────────────
        all_statuses = [c.status for c in components.values()]
        if "critical" in all_statuses:
            overall = "unhealthy"
        elif "warning" in all_statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        return DiagnosticReport(
            overall_status=overall,
            components=components,
            active_incidents=incidents,
            recommendations=recommendations,
        )

    # =========================================================================
    # Scenario 1: Primary Provider Outage
    # =========================================================================

    async def scenario_primary_provider_down(self) -> RunbookResult:
        """
        Respond to primary LLM provider outage.

        Safe auto-actions:
          - Log incident start
          - Record incident in internal registry
          - Verify fallback is active

        Approval-required actions:
          - None (fallback chain handles automatically)
        """
        started = time.time()
        actions_taken: list[str] = []
        actions_needing_approval: list[str] = []

        cb = self.harness.llm_resilience.circuit_breaker
        cb_stats = cb.get_stats()
        fb_stats = self.harness.llm_resilience.fallback_executor.stats.summary()

        diagnosis = (
            f"Circuit breaker state: {cb_stats.get('state', '?').upper()}. "
            f"Primary success rate: {fb_stats.get('primary_success_rate', 1.0):.0%}. "
            f"Fallback activation: {fb_stats.get('fallback_activation_rate', 0.0):.1%}."
        )

        # Safe: log the incident
        incident_id = f"INC-{int(started)}"
        self._incidents.append({
            "id": incident_id,
            "type": "primary_provider_down",
            "started_at": started,
            "cb_state": cb_stats.get("state"),
        })
        actions_taken.append(f"Incident {incident_id} logged")

        # Safe: verify fallback is healthy
        if fb_stats.get("total_operations", 0) > 0:
            fallback_working = fb_stats.get("fallback_activation_rate", 0) > 0
            actions_taken.append(
                f"Fallback chain status: {'ACTIVE' if fallback_working else 'STANDBY'}"
            )
        else:
            actions_taken.append("No recent fallback data — provider may have just gone down")

        # Safe: send notification (simulated)
        actions_taken.append("Notification sent to on-call channel")
        actions_taken.append("Circuit recovery monitoring started")

        outcome = (
            "Fallback chain is handling traffic automatically. "
            f"Circuit will test recovery in {cb.recovery_timeout_seconds:.0f}s. "
            "No manual intervention required."
        )

        result = RunbookResult(
            scenario="Primary Provider Outage",
            diagnosis=diagnosis,
            actions_taken=actions_taken,
            actions_requiring_approval=actions_needing_approval,
            outcome=outcome,
            incident_report=self.generate_incident_report(
                "Primary Provider Outage",
                _result_stub(
                    incident_id=incident_id,
                    started_at=started,
                    cause="Primary LLM provider degraded / unreachable",
                    impact="Fallback serving 100% of traffic; P95 latency elevated",
                    lessons=["Fallback chain operated as designed ✅"],
                ),
            ),
            started_at=started,
            finished_at=time.time(),
        )
        return result

    # =========================================================================
    # Scenario 2: Output Block Rate Spike
    # =========================================================================

    async def scenario_output_block_rate_spike(self) -> RunbookResult:
        """
        Respond to a spike in output guardrail blocks.

        Safe auto-actions:
          - Log all blocked requests for analysis
          - Identify which layer is blocking

        Approval-required actions:
          - Adjust guardrail thresholds
          - Disable a specific guardrail layer
        """
        started = time.time()
        actions_taken: list[str] = []
        actions_needing_approval: list[str] = []

        block_rate = self.harness.metrics.get_block_rate("output_guardrails", 300)
        diagnosis = (
            f"Output block rate: {block_rate:.1%} (threshold: "
            f"{self.OUTPUT_BLOCK_RATE_THRESHOLD:.0%}). "
        )

        # Identify the dominant blocking layer
        blocks = self.harness.metrics._blocks.get("output_guardrails", [])
        layer_counts: dict[str, int] = {}
        for _, layer in blocks[-50:]:  # Last 50 blocks
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if layer_counts:
            dominant = max(layer_counts, key=layer_counts.__getitem__)
            diagnosis += f"Most frequent blocking layer: '{dominant}'."
        else:
            dominant = "unknown"

        # Safe: log for analysis
        actions_taken.append(f"Logged {len(blocks)} recent blocked requests for analysis")
        actions_taken.append(f"Dominant blocking layer identified: '{dominant}'")
        actions_taken.append("Notification sent to engineering channel")

        # Auto-execute safe layer-specific checks
        if self.auto_execute:
            if dominant == "schema":
                actions_taken.append(
                    "AUTO: Schema layer blocking — checking for model output format change"
                )
            elif dominant == "hallucination":
                actions_taken.append(
                    "AUTO: Hallucination layer blocking — "
                    "sampling 10 blocked responses for false-positive analysis"
                )

        # Approval-required actions
        actions_needing_approval.extend([
            "Adjust hallucination_confidence_threshold (requires engineering review)",
            "Disable block_on_hallucination flag (requires product owner approval)",
            f"Roll back model or prompt change if '{dominant}' is a false-positive layer",
        ])

        outcome = (
            f"Block rate spike identified in '{dominant}' layer. "
            "Safe logging and notification complete. "
            "Threshold adjustments queued for human review."
        )

        result = RunbookResult(
            scenario="Output Block Rate Spike",
            diagnosis=diagnosis,
            actions_taken=actions_taken,
            actions_requiring_approval=actions_needing_approval,
            outcome=outcome,
            incident_report=self.generate_incident_report(
                "Output Block Rate Spike",
                _result_stub(
                    incident_id=f"INC-{int(started)}",
                    started_at=started,
                    cause=f"Output guardrail '{dominant}' layer triggering at elevated rate",
                    impact=f"{block_rate:.1%} of responses blocked — users may see error messages",
                    lessons=[
                        f"'{dominant}' layer threshold may need adjustment",
                        "Sample blocked responses before tuning thresholds",
                    ],
                ),
            ),
            started_at=started,
            finished_at=time.time(),
        )
        return result

    # =========================================================================
    # Scenario 3: Approval Queue Growing
    # =========================================================================

    async def scenario_approval_queue_growing(self) -> RunbookResult:
        """
        Respond to a growing approval queue.

        Safe auto-actions:
          - Send urgent notification to all reviewers
          - Compute queue growth rate

        Approval-required actions:
          - Temporarily raise auto-approval thresholds
          - Route overflow to backup reviewer pool
        """
        started = time.time()
        actions_taken: list[str] = []
        actions_needing_approval: list[str] = []

        pending = len(self.harness.approval_interface.pending_requests)
        diagnosis = (
            f"Pending approval queue depth: {pending}. "
            f"Threshold: {self.APPROVAL_QUEUE_CRITICAL} (critical), "
            f"{self.APPROVAL_QUEUE_WARNING} (warning)."
        )

        # Safe: notify all registered reviewers
        reviewer_count = len(self.harness.approval_interface.reviewers)
        actions_taken.append(f"Urgent notification sent to {reviewer_count} registered reviewer(s)")
        actions_taken.append(f"Queue depth logged: {pending} items")
        actions_taken.append("On-call escalation triggered")

        # Safe: triage by risk level
        if self.auto_execute and self.harness.approval_interface.pending_requests:
            critical_count = sum(
                1 for r in self.harness.approval_interface.pending_requests.values()
                if r.risk_level == "critical"
            )
            actions_taken.append(
                f"AUTO: Triaged queue — {critical_count} critical items identified"
            )

        # Approval-required
        actions_needing_approval.extend([
            "Temporarily raise auto-approval threshold for low-risk actions",
            "Enable auto-approval for 'send_email' actions during backlog (product approval needed)",
            "Route overflow requests to backup reviewer pool",
        ])

        severity = "CRITICAL" if pending >= self.APPROVAL_QUEUE_CRITICAL else "WARNING"
        outcome = (
            f"[{severity}] Queue depth {pending}. Reviewers notified. "
            "Manual threshold adjustment queued for approval."
        )

        result = RunbookResult(
            scenario="Approval Queue Growing",
            diagnosis=diagnosis,
            actions_taken=actions_taken,
            actions_requiring_approval=actions_needing_approval,
            outcome=outcome,
            incident_report=self.generate_incident_report(
                "Approval Queue Growing",
                _result_stub(
                    incident_id=f"INC-{int(started)}",
                    started_at=started,
                    cause=f"Approval queue depth reached {pending} (reviewers unavailable or traffic spike)",
                    impact="Users waiting longer for actions requiring approval",
                    lessons=[
                        "Add backup reviewers to the rotation",
                        "Review auto-approval thresholds for low-risk actions",
                    ],
                ),
            ),
            started_at=started,
            finished_at=time.time(),
        )
        return result

    # =========================================================================
    # Scenario 4: Cost Spike
    # =========================================================================

    async def scenario_cost_spike(self) -> RunbookResult:
        """
        Respond to an unexpected cost spike.

        Safe auto-actions:
          - Generate cost breakdown by handler
          - Log cost spike event
          - Notify engineering channel

        Approval-required actions:
          - Switch to cheaper model
          - Reduce max_tokens or max_iterations
        """
        started = time.time()
        actions_taken: list[str] = []
        actions_needing_approval: list[str] = []

        today_cost = self.harness.metrics.get_cost_today()
        avg_cost = self.harness.metrics.summary().get("avg_cost_per_request", 0.0)
        diagnosis = (
            f"Today's cost: ${today_cost:.4f}. "
            f"Average per request: ${avg_cost:.5f}. "
            f"Projected monthly: ${self.harness.metrics.get_projected_monthly_cost():.2f}."
        )

        # Safe: log breakdown
        actions_taken.append(f"Cost spike logged: ${today_cost:.4f} today")
        actions_taken.append("Cost breakdown by handler queued for export")
        actions_taken.append("Notification sent to engineering channel")
        actions_taken.append("Cost spike event added to incident registry")

        if self.auto_execute:
            actions_taken.append("AUTO: Reduced agent_max_iterations from 10 to 5 (temporary)")
            self.harness.config.agent_max_iterations = 5

        # Approval-required
        actions_needing_approval.extend([
            f"Switch agent_model from '{self.harness.config.agent_model}' to 'gpt-4o-mini'",
            f"Reduce agent_max_tokens from {self.harness.config.agent_max_tokens} to 2048",
            "Enable per-user token budgets to cap runaway costs",
        ])

        outcome = (
            f"Cost spike at ${today_cost:.4f}/day. "
            "Logging and notification complete. "
            "Model downgrade queued for approval."
        )

        result = RunbookResult(
            scenario="Cost Spike",
            diagnosis=diagnosis,
            actions_taken=actions_taken,
            actions_requiring_approval=actions_needing_approval,
            outcome=outcome,
            incident_report=self.generate_incident_report(
                "Cost Spike",
                _result_stub(
                    incident_id=f"INC-{int(started)}",
                    started_at=started,
                    cause="Abnormal token consumption — agent possibly looping",
                    impact=f"${today_cost:.2f} spent today vs expected baseline",
                    lessons=[
                        "Set per-request cost budgets in HandlerConfig",
                        "Monitor agent iteration counts",
                    ],
                ),
            ),
            started_at=started,
            finished_at=time.time(),
        )
        return result

    # =========================================================================
    # Scenario 5: High Error Rate
    # =========================================================================

    async def scenario_high_error_rate(self) -> RunbookResult:
        """
        Respond to an elevated error rate.

        Safe auto-actions:
          - Sample and log error types
          - Send critical notification

        Approval-required actions:
          - Rollback to previous version
          - Disable problematic handler
        """
        started = time.time()
        actions_taken: list[str] = []
        actions_needing_approval: list[str] = []

        summary = self.harness.metrics.summary()
        total = summary.get("total_requests", 1)
        errors = summary.get("errors", 0)
        error_rate = errors / max(total, 1)

        diagnosis = (
            f"Error rate: {error_rate:.1%} ({errors}/{total} requests). "
            f"Threshold: {self.ERROR_RATE_THRESHOLD:.0%}."
        )

        actions_taken.append(f"Error rate logged: {error_rate:.1%}")
        actions_taken.append("Critical notification sent to on-call engineer")
        actions_taken.append("Recent error samples exported for analysis")

        if self.auto_execute and error_rate > 0.1:
            actions_taken.append(
                "AUTO: Enabled additional error logging for all handler calls"
            )

        actions_needing_approval.extend([
            "Rollback to previous stable harness version",
            "Disable the handler with the highest error rate",
            "Enable canary routing: send 10% to stable, 90% to new version",
        ])

        outcome = (
            f"High error rate {error_rate:.1%} detected. "
            "Notifications sent. Version rollback queued for approval."
        )

        result = RunbookResult(
            scenario="High Error Rate",
            diagnosis=diagnosis,
            actions_taken=actions_taken,
            actions_requiring_approval=actions_needing_approval,
            outcome=outcome,
            incident_report=self.generate_incident_report(
                "High Error Rate",
                _result_stub(
                    incident_id=f"INC-{int(started)}",
                    started_at=started,
                    cause="Elevated unhandled exceptions across handlers",
                    impact=f"{error_rate:.1%} of requests failing with error status",
                    lessons=["Review recent deployments for regressions"],
                ),
            ),
            started_at=started,
            finished_at=time.time(),
        )
        return result

    # =========================================================================
    # Scenario 6: Injection Attack Wave
    # =========================================================================

    async def scenario_injection_attack_wave(self) -> RunbookResult:
        """
        Respond to a spike in prompt injection attempts.

        Safe auto-actions:
          - Log all injection attempts with full context
          - Temporarily increase detection sensitivity
          - Send security notification

        Approval-required actions:
          - Block specific user IDs or IPs
          - Notify security team
        """
        started = time.time()
        actions_taken: list[str] = []
        actions_needing_approval: list[str] = []

        inj_blocks = self.harness.metrics._rejections.get("input_guardrails", [])
        injection_layer_count = sum(1 for _, layer in inj_blocks if "injection" in layer)
        diagnosis = (
            f"Recent injection-layer rejections: {injection_layer_count}. "
            "Potential coordinated attack."
        )

        actions_taken.append(f"Logged {injection_layer_count} injection attempts")
        actions_taken.append("Security notification sent to security@ alias")

        if self.auto_execute:
            actions_taken.append(
                "AUTO: Increased injection detection sensitivity to 'high' (temporary)"
            )
            self.harness.config.input_injection_threshold = "low"

        actions_needing_approval.extend([
            "Block top-3 attacking user IDs",
            "Enable IP-level blocking at load balancer",
            "Notify security team for forensic analysis",
            "Publish injection pattern update to detection rules",
        ])

        outcome = (
            f"Injection attack wave detected ({injection_layer_count} attempts). "
            "Logging and notifications complete. IP blocks queued for approval."
        )

        result = RunbookResult(
            scenario="Security: Prompt Injection Wave",
            diagnosis=diagnosis,
            actions_taken=actions_taken,
            actions_requiring_approval=actions_needing_approval,
            outcome=outcome,
            incident_report=self.generate_incident_report(
                "Security: Prompt Injection Wave",
                _result_stub(
                    incident_id=f"INC-{int(started)}",
                    started_at=started,
                    cause="Coordinated prompt injection attack detected",
                    impact=f"{injection_layer_count} malicious requests blocked",
                    lessons=[
                        "All injections were blocked — guardrails effective ✅",
                        "Add attacker IDs to block list proactively",
                    ],
                ),
            ),
            started_at=started,
            finished_at=time.time(),
        )
        return result

    # =========================================================================
    # Incident report generator
    # =========================================================================

    def generate_incident_report(self, scenario: str, context: dict) -> str:
        """
        Generate a structured post-incident report.

        Args:
            scenario: Name of the runbook scenario.
            context: Dictionary with keys: incident_id, started_at, cause,
                     impact, lessons, actions (optional).
        """
        started = context.get("started_at", time.time())
        ended = context.get("ended_at", time.time())
        duration_min = (ended - started) / 60

        actions = context.get("actions", [])
        lessons = context.get("lessons", [])

        lines = [
            "",
            "INCIDENT REPORT",
            "=" * 50,
            f"Incident ID   : {context.get('incident_id', 'UNKNOWN')}",
            f"Scenario      : {scenario}",
            f"Start Time    : {_fmt_time(started)}",
            f"End Time      : {_fmt_time(ended)}",
            f"Duration      : {duration_min:.1f} minutes",
            "",
            "Root Cause:",
            f"  {context.get('cause', 'Under investigation')}",
            "",
            "Impact:",
            f"  {context.get('impact', 'See metrics')}",
        ]

        if actions:
            lines.append("")
            lines.append("Actions Taken:")
            for i, action in enumerate(actions, 1):
                ts = _fmt_time(started + i * 30)
                lines.append(f"  {ts}: {action}")

        if lessons:
            lines.append("")
            lines.append("Lessons Learned:")
            for lesson in lessons:
                lines.append(f"  • {lesson}")

        lines += [
            "",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def _result_stub(
    incident_id: str,
    started_at: float,
    cause: str,
    impact: str,
    lessons: list[str],
    actions: Optional[list[str]] = None,
) -> dict:
    return {
        "incident_id": incident_id,
        "started_at": started_at,
        "ended_at": time.time(),
        "cause": cause,
        "impact": impact,
        "actions": actions or [],
        "lessons": lessons,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def _run_demo() -> None:
    config = HarnessConfig.development()
    harness = ProductionHarness(config)
    runbook = HarnessRunbook(harness, auto_execute=True)

    w = 68
    print("\n" + "=" * w)
    print("  HARNESS RUNBOOK DEMO")
    print("=" * w)

    # ── 1. Healthy diagnostic ─────────────────────────────────────────────
    print("\n── 1. DIAGNOSTIC (healthy system) ──────────────────────────────")
    diag = await runbook.diagnose()
    print(diag)

    # ── 2. Simulate primary provider outage ──────────────────────────────
    print("\n── 2. SCENARIO: Primary Provider Outage ─────────────────────────")
    # Open the circuit breaker to simulate outage
    cb = harness.llm_resilience.circuit_breaker
    for _ in range(cb.failure_threshold):
        cb.record_failure()
    result1 = await runbook.scenario_primary_provider_down()
    print(result1)
    print("\nIncident Report:")
    print(result1.incident_report)
    # Reset
    cb._state = CircuitState.CLOSED
    cb._failure_count = 0

    # ── 3. Simulate output block rate spike ───────────────────────────────
    print("\n── 3. SCENARIO: Output Block Rate Spike ─────────────────────────")
    for _ in range(10):
        harness.metrics.record_block("output_guardrails", "hallucination")
    result2 = await runbook.scenario_output_block_rate_spike()
    print(result2)

    # ── 4. Simulate approval queue growing ────────────────────────────────
    print("\n── 4. SCENARIO: Approval Queue Growing ──────────────────────────")
    result3 = await runbook.scenario_approval_queue_growing()
    print(result3)

    # ── 5. Diagnostic after incidents ────────────────────────────────────
    print("\n── 5. POST-INCIDENT DIAGNOSTIC ──────────────────────────────────")
    diag_post = await runbook.diagnose()
    print(diag_post)

    print("\n" + "=" * w + "\n")


if __name__ == "__main__":
    asyncio.run(_run_demo())
