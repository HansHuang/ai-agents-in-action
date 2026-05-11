"""
Approval Dashboard — CLI with Rich
====================================
A terminal-based reviewer dashboard for managing the human-in-the-loop
approval queue.  Uses Rich for colour-coded tables, live refresh, and
formatted panels.

Usage:
    python approval_dashboard.py           # run demo
    python approval_dashboard.py --live    # simulated live queue (auto-plays)

Companion to: docs/07-harness-engineering/06-human-in-the-loop.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
except ImportError:
    print("ERROR: 'rich' is required.  Run:  pip install rich")
    sys.exit(1)

# Re-use core data models
from human_in_the_loop import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalMetrics,
    HumanReviewerInterface,
)

console = Console()
REVIEWER = HumanReviewerInterface("reviewer-dashboard")


# ---------------------------------------------------------------------------
# Colour / style helpers
# ---------------------------------------------------------------------------

RISK_STYLE: dict[str, str] = {
    "low":      "bold green",
    "medium":   "bold yellow",
    "high":     "bold red",
    "critical": "bold red blink",
}

DECISION_STYLE: dict[str, str] = {
    "approved":            "bold green",
    "rejected":            "bold red",
    "approved_with_edits": "bold yellow",
    "auto_rejected":       "dim red",
}


def risk_text(level: str) -> Text:
    return Text(level.upper(), style=RISK_STYLE.get(level, "white"))


def decision_text(decision: str) -> Text:
    return Text(decision.replace("_", " ").title(), style=DECISION_STYLE.get(decision, "white"))


def fmt_wait(created_at: float) -> str:
    secs = int(time.time() - created_at)
    m, s = divmod(secs, 60)
    return f"{m}m {s:02d}s"


def fmt_deadline(deadline: float | None) -> str:
    if deadline is None:
        return "—"
    remaining = deadline - time.time()
    if remaining <= 0:
        return "[bold red]EXPIRED[/bold red]"
    m, s = divmod(int(remaining), 60)
    style = "bold red" if remaining < 60 else "yellow"
    return f"[{style}]{m}:{s:02d}[/{style}]"


# ---------------------------------------------------------------------------
# History record
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    timestamp: float
    action: str
    request_id: str
    decision: str
    reviewer_id: str
    notes: str | None
    cost: float


# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------

@dataclass
class DashboardState:
    queue: list[ApprovalRequest] = field(default_factory=list)
    history: deque[HistoryEntry] = field(default_factory=lambda: deque(maxlen=50))
    metrics: ApprovalMetrics = field(default_factory=ApprovalMetrics)
    sort_by: str = "risk"           # "risk" | "wait" | "cost"
    show_history: bool = False
    current_index: int = 0

    # today's counters
    approved_today: int = 0
    rejected_today: int = 0

    def sorted_queue(self) -> list[ApprovalRequest]:
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        if self.sort_by == "risk":
            return sorted(self.queue, key=lambda r: risk_order.get(r.risk_level, 9))
        if self.sort_by == "wait":
            return sorted(self.queue, key=lambda r: r.created_at)
        if self.sort_by == "cost":
            return sorted(self.queue, key=lambda r: -r.estimated_cost)
        return self.queue

    def record_decision(
        self, request: ApprovalRequest, response: ApprovalResponse, elapsed: float
    ) -> None:
        self.metrics.record(request, response, elapsed)
        self.history.append(HistoryEntry(
            timestamp=time.time(),
            action=request.proposed_action,
            request_id=request.request_id,
            decision=response.decision,
            reviewer_id=response.reviewer_id,
            notes=response.reviewer_notes,
            cost=request.estimated_cost,
        ))
        if response.decision in ("approved", "approved_with_edits"):
            self.approved_today += 1
        else:
            self.rejected_today += 1


# ---------------------------------------------------------------------------
# View renderers
# ---------------------------------------------------------------------------

def render_queue_table(state: DashboardState) -> Table:
    table = Table(
        title="[bold]Pending Approval Requests[/bold]",
        box=box.ROUNDED,
        expand=True,
        show_lines=True,
    )
    table.add_column("#",       width=3,  justify="right")
    table.add_column("ID",      width=10, style="dim")
    table.add_column("Action",  width=24)
    table.add_column("Risk",    width=10, justify="center")
    table.add_column("Cost",    width=10, justify="right")
    table.add_column("Wait",    width=8,  justify="right")
    table.add_column("Deadline",width=10, justify="right")

    queue = state.sorted_queue()
    if not queue:
        table.add_row("", "[dim]Queue is empty[/dim]", "", "", "", "", "")
        return table

    for idx, req in enumerate(queue):
        marker = "▶" if idx == state.current_index else " "
        table.add_row(
            f"{marker}{idx + 1}",
            req.request_id[:8],
            req.proposed_action,
            risk_text(req.risk_level),
            f"${req.estimated_cost:,.2f}",
            fmt_wait(req.created_at),
            fmt_deadline(req.deadline),
        )

    return table


def render_detail_panel(req: ApprovalRequest) -> Panel:
    params_json = json.dumps(req.proposed_params, indent=2)
    syntax = Syntax(params_json, "json", theme="monokai", line_numbers=False)

    evidence_lines = "\n".join(
        f"  • {json.dumps(e, separators=(',', ':'))}" for e in req.evidence[:5]
    ) or "  (none)"

    content = (
        f"[bold]Action:[/bold] {req.proposed_action}\n"
        f"[bold]Risk Level:[/bold] {risk_text(req.risk_level)}\n"
        f"[bold]Estimated Cost:[/bold] ${req.estimated_cost:,.2f}\n"
        f"[bold]Affected Systems:[/bold] {', '.join(req.affected_systems)}\n\n"
        f"[bold]Parameters:[/bold]\n"
    )
    text = Text.from_markup(content)
    # Rich can't mix Syntax into Text directly; we'll use a flat string fallback
    flat = (
        content
        + params_json
        + f"\n\n[bold]Reasoning:[/bold]\n{req.reasoning}\n\n"
        f"[bold]Conversation Context:[/bold]\n{req.conversation_summary[:500]}\n\n"
        f"[bold]Evidence:[/bold]\n{evidence_lines}\n\n"
        "[dim]─────────────────────────────────────[/dim]\n"
        "[bold]Shortcuts:[/bold]  "
        "[green]Y[/green]=Approve  "
        "[red]N[/red]=Reject  "
        "[yellow]E[/yellow]=Edit+Approve  "
        "[dim]S[/dim]=Skip  "
        "[dim]Q[/dim]=Quit  "
        "[dim]H[/dim]=Help  "
        "[dim]Tab[/dim]=Toggle history"
    )
    return Panel(
        Text.from_markup(flat),
        title=f"[bold]Request Detail — {req.request_id[:8]}[/bold]",
        border_style="cyan",
        expand=True,
    )


def render_history_panel(state: DashboardState) -> Panel:
    table = Table(box=box.SIMPLE, expand=True, show_header=True)
    table.add_column("Time",     width=8)
    table.add_column("Action",   width=24)
    table.add_column("Decision", width=20)
    table.add_column("Reviewer", width=12)
    table.add_column("Notes",    width=30)

    for entry in reversed(state.history):
        ts = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        table.add_row(
            ts,
            entry.action,
            decision_text(entry.decision),
            entry.reviewer_id[:10],
            (entry.notes or "")[:28],
        )

    return Panel(table, title="[bold]Decision History (last 50)[/bold]",
                 border_style="magenta")


def render_stats_bar(state: DashboardState) -> Panel:
    s = state.metrics.summary()
    avg_str = f"{s['avg_response_time_seconds']:.1f}s"
    pending = len(state.queue)
    warning = " [bold red]⚠ QUEUE OVERLOAD[/bold red]" if pending > 10 else ""

    content = (
        f"[bold]Pending:[/bold] {pending}{warning}  │  "
        f"[bold green]Approved today:[/bold green] {state.approved_today}  │  "
        f"[bold red]Rejected today:[/bold red] {state.rejected_today}  │  "
        f"[bold]Avg response:[/bold] {avg_str}  │  "
        f"[bold]Total requests:[/bold] {s['total_approval_requests']}"
    )
    return Panel(Text.from_markup(content), box=box.HORIZONTALS, style="dim")


# ---------------------------------------------------------------------------
# Interactive decision loop
# ---------------------------------------------------------------------------

def prompt_approve(req: ApprovalRequest, state: DashboardState) -> ApprovalResponse | None:
    notes = Prompt.ask("  Notes (optional, Enter to skip)", default="", show_default=False)
    return REVIEWER.approve(req.request_id, notes=notes or None)


def prompt_reject(req: ApprovalRequest, state: DashboardState) -> ApprovalResponse | None:
    reason = Prompt.ask("  Reason for rejection (required)")
    if not reason.strip():
        console.print("[red]  Rejection requires a reason.[/red]")
        return None
    return REVIEWER.reject(req.request_id, reason=reason)


def prompt_edit(req: ApprovalRequest, state: DashboardState) -> ApprovalResponse | None:
    console.print("\n[bold yellow]  Parameter Editor[/bold yellow]")
    console.print("  Current parameters:")
    console.print(Syntax(json.dumps(req.proposed_params, indent=2), "json",
                         theme="monokai"))
    console.print("  Enter modified JSON (Ctrl+C to cancel):")
    try:
        raw = Prompt.ask("  Edited params")
        edited = json.loads(raw)
    except (json.JSONDecodeError, KeyboardInterrupt):
        console.print("[red]  Invalid JSON or cancelled.[/red]")
        return None
    notes = Prompt.ask("  Notes (optional)", default="", show_default=False)
    return REVIEWER.approve_with_edits(req.request_id, edited, notes=notes or None)


def handle_keypress(
    key: str,
    req: ApprovalRequest,
    state: DashboardState,
) -> ApprovalResponse | None:
    """Return an ApprovalResponse if a decision was made, else None."""
    k = key.strip().upper()

    if k == "Y":
        return prompt_approve(req, state)
    if k == "N":
        return prompt_reject(req, state)
    if k == "E":
        return prompt_edit(req, state)
    if k == "S":
        console.print("  [dim]Skipped.[/dim]")
        return None
    if k == "H":
        console.print(Panel(
            "[green]Y[/green] — Approve\n"
            "[red]N[/red] — Reject\n"
            "[yellow]E[/yellow] — Edit and approve\n"
            "[dim]S[/dim] — Skip to next request\n"
            "[dim]Q[/dim] — Quit dashboard\n"
            "[dim]H[/dim] — Show this help\n"
            "[dim]Tab[/dim] — Toggle history panel\n"
            "[dim]1/2/3[/dim] — Sort by risk / wait / cost",
            title="[bold]Keyboard Shortcuts[/bold]",
        ))
        return None
    if k in ("1", "2", "3"):
        state.sort_by = ["risk", "wait", "cost"][int(k) - 1]
        console.print(f"  [dim]Sorted by: {state.sort_by}[/dim]")
        return None
    console.print(f"  [dim]Unknown key: {key!r}  (H for help)[/dim]")
    return None


# ---------------------------------------------------------------------------
# Demo scenario runner (non-interactive)
# ---------------------------------------------------------------------------

def build_demo_queue() -> list[ApprovalRequest]:
    import uuid as _uuid

    def req(action, params, risk, cost, systems, reasoning, deadline_in=None):
        now = time.time()
        return ApprovalRequest(
            request_id=str(_uuid.uuid4()),
            agent_id="demo-agent",
            session_id="session-dash-01",
            proposed_action=action,
            proposed_params=params,
            reasoning=reasoning,
            conversation_summary="Customer: Please process my request as soon as possible.",
            evidence=[{"source": "order_system", "data": params}],
            risk_level=risk,
            estimated_cost=cost,
            affected_systems=systems,
            created_at=now - (300 - i * 30),   # stagger creation times
            deadline=now + deadline_in if deadline_in else None,
        )

    items = [
        ("get_recommendations", {"user_id": "U-1", "category": "books"},
         "low", 0.0, ["recommendation_engine"], "User asked for book recommendations.", None),
        ("send_promo_email", {"to": "customer@example.com", "template": "spring_sale"},
         "low", 0.0, ["email_service"], "Scheduled promotional email.", None),
        ("send_email", {"to": "vip@corp.com", "subject": "Contract renewal"},
         "medium", 0.0, ["email_service"], "Outbound customer success email.", 300),
        ("update_database", {"table": "subscriptions", "set": {"status": "paused"}},
         "medium", 50.0, ["database"], "Admin batch pause request.", None),
        ("issue_refund", {"order_id": "ORD-77", "amount": 750},
         "high", 750.0, ["payment_processor", "order_database"],
         "Customer reports item damaged on arrival.", 300),
        ("cancel_subscription", {"subscription_id": "SUB-19", "reason": "price"},
         "high", 120.0, ["billing_system", "subscription_service"],
         "Customer wants to cancel due to pricing.", None),
        ("export_user_data", {"user_id": "U-99", "format": "json"},
         "critical", 0.0, ["data_warehouse", "gdpr_service"],
         "GDPR Article 20 data portability request.", 300),
        ("bulk_delete_records", {"table": "audit_logs", "older_than_days": 30},
         "critical", 0.0, ["database", "audit_system"],
         "Automated retention policy enforcement.", 180),
    ]

    result = []
    for i, (action, params, risk, cost, systems, reasoning, dl) in enumerate(items):
        now = time.time()
        r = ApprovalRequest(
            request_id=f"req-{i+1:04d}",
            agent_id="demo-agent",
            session_id="session-dash-01",
            proposed_action=action,
            proposed_params=params,
            reasoning=reasoning,
            conversation_summary="Customer: Please process my request as soon as possible.",
            evidence=[{"source": "order_system", "data": {"amount": cost}}],
            risk_level=risk,
            estimated_cost=cost,
            affected_systems=systems,
            created_at=now - (10 + i * 30),
            deadline=now + dl if dl else None,
        )
        result.append(r)
    return result


def run_demo_session(queue: list[ApprovalRequest]) -> None:
    """
    Walk through a scripted demo session that mirrors the described interactions:
      1. Sort queue by risk, review
      2. Open critical request, read context
      3. Approve with edits (refund $750 → $500)
      4. Reject a medium-risk request
      5. Approve a low-risk request
      6. Timeout on a high-risk request (auto-rejected)
    """
    state = DashboardState(queue=list(queue))

    console.rule("[bold]APPROVAL DASHBOARD DEMO[/bold]")
    console.print()

    # ── Step 1: Show queue sorted by risk ──────────────────────────────
    console.print("[bold underline]Step 1 — Queue (sorted by risk level)[/bold underline]")
    state.sort_by = "risk"
    console.print(render_queue_table(state))
    console.print()

    sorted_q = state.sorted_queue()

    # ── Step 2: Open critical request, read context ────────────────────
    critical = next((r for r in sorted_q if r.risk_level == "critical"), sorted_q[0])
    console.print(f"[bold underline]Step 2 — Detail view for critical request: {critical.request_id}[/bold underline]")
    console.print(render_detail_panel(critical))
    console.print()

    # ── Step 3: Approve with edits ($750 → $500) ───────────────────────
    refund_req = next((r for r in sorted_q if r.proposed_action == "issue_refund"), None)
    if refund_req:
        console.print("[bold underline]Step 3 — Approve refund with edits ($750 → $500)[/bold underline]")
        start = time.time()
        resp = REVIEWER.approve_with_edits(
            refund_req.request_id,
            edited_params={"order_id": refund_req.proposed_params.get("order_id"), "amount": 500},
            notes="Standard partial refund for damaged item category is $500.",
        )
        state.queue.remove(refund_req)
        state.record_decision(refund_req, resp, time.time() - start)
        console.print(f"  → [bold yellow]APPROVED WITH EDITS[/bold yellow]  "
                      f"amount: $750 → $500  |  notes: {resp.reviewer_notes}")
        console.print()

    # ── Step 4: Reject a medium-risk request ──────────────────────────
    medium_req = next((r for r in state.sorted_queue() if r.risk_level == "medium"), None)
    if medium_req:
        console.print(f"[bold underline]Step 4 — Reject medium-risk: {medium_req.proposed_action}[/bold underline]")
        start = time.time()
        resp = REVIEWER.reject(
            medium_req.request_id,
            reason="Email content needs compliance review before sending.",
        )
        state.queue.remove(medium_req)
        state.record_decision(medium_req, resp, time.time() - start)
        console.print(f"  → [bold red]REJECTED[/bold red]  "
                      f"reason: {resp.reason}")
        console.print()

    # ── Step 5: Approve a low-risk request ────────────────────────────
    low_req = next((r for r in state.sorted_queue() if r.risk_level == "low"), None)
    if low_req:
        console.print(f"[bold underline]Step 5 — Approve low-risk: {low_req.proposed_action}[/bold underline]")
        start = time.time()
        resp = REVIEWER.approve(low_req.request_id, notes="Looks fine.")
        state.queue.remove(low_req)
        state.record_decision(low_req, resp, time.time() - start)
        console.print(f"  → [bold green]APPROVED[/bold green]")
        console.print()

    # ── Step 6: Timeout on a high-risk request (simulate) ─────────────
    high_req = next((r for r in state.sorted_queue() if r.risk_level == "high"), None)
    if high_req:
        console.print(f"[bold underline]Step 6 — Timeout on high-risk: {high_req.proposed_action}[/bold underline]")
        start = time.time()
        timeout_resp = ApprovalResponse(
            request_id=high_req.request_id,
            decision="rejected",
            reviewer_id="system",
            reason="Approval timeout — automatically rejected for safety.",
            automated=True,
        )
        state.queue.remove(high_req)
        state.record_decision(high_req, timeout_resp, time.time() - start)
        console.print(f"  → [bold red]AUTO-REJECTED[/bold red] (timeout)  "
                      f"reason: {timeout_resp.reason}")
        console.print()

    # ── Final statistics ───────────────────────────────────────────────
    console.rule("[bold]FINAL STATISTICS[/bold]")
    console.print(render_stats_bar(state))
    console.print()

    summary = state.metrics.summary()
    stats_table = Table(box=box.SIMPLE, show_header=False)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value")

    for k, v in summary.items():
        if k == "by_risk_level":
            continue
        if isinstance(v, float):
            stats_table.add_row(k, f"{v:.3f}")
        else:
            stats_table.add_row(k, str(v))

    console.print(stats_table)
    console.print()

    # History panel
    console.print("[bold underline]Decision History[/bold underline]")
    console.print(render_history_panel(state))


# ---------------------------------------------------------------------------
# Interactive mode (reads stdin)
# ---------------------------------------------------------------------------

def run_interactive(queue: list[ApprovalRequest]) -> None:
    state = DashboardState(queue=list(queue))

    while state.queue:
        sorted_q = state.sorted_queue()
        if state.current_index >= len(sorted_q):
            state.current_index = 0

        # Render
        console.clear()
        console.print(render_queue_table(state))

        if state.show_history:
            console.print(render_history_panel(state))

        console.print(render_stats_bar(state))

        req = sorted_q[state.current_index]
        console.print(render_detail_panel(req))

        # Alert for critical requests
        if req.risk_level == "critical":
            console.bell()
            console.print("[bold red blink]⚠  CRITICAL REQUEST — immediate attention required[/bold red blink]")

        # Input
        try:
            key = Prompt.ask("[bold]Action[/bold]", default="S")
        except KeyboardInterrupt:
            break

        if key.strip().upper() == "Q":
            break
        if key.strip() == "\t" or key.strip().upper() == "TAB":
            state.show_history = not state.show_history
            continue

        start = time.time()
        response = handle_keypress(key, req, state)

        if response is not None:
            elapsed = time.time() - start
            state.queue.remove(req)
            state.record_decision(req, response, elapsed)
            if state.current_index >= len(state.queue):
                state.current_index = 0
        else:
            if key.strip().upper() == "S":
                state.current_index = (state.current_index + 1) % max(len(sorted_q), 1)

    console.rule("[bold]Session Complete[/bold]")
    console.print(render_stats_bar(state))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approval Review Dashboard")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (reads keyboard input)",
    )
    args = parser.parse_args()

    demo_queue = build_demo_queue()

    if args.interactive:
        run_interactive(demo_queue)
    else:
        run_demo_session(demo_queue)
