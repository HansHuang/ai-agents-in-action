"""
Approval Policy Optimizer
==========================
Analyzes historical approval decisions to identify policy improvements that
reduce unnecessary human review while maintaining safety guarantees.

Companion to: docs/07-harness-engineering/06-human-in-the-loop.md
"""

from __future__ import annotations

import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRecord:
    """A single historical approval decision."""
    timestamp: float
    action: str
    params: dict
    risk_level: str
    estimated_cost: float
    decision: str           # "approved" | "rejected" | "approved_with_edits"
    reviewer_id: str
    response_time: float    # seconds from request to decision
    reviewer_notes: str | None = None


@dataclass
class Recommendation:
    """A single optimization suggestion with supporting evidence."""
    title: str
    description: str
    impact: str             # "high" | "medium" | "low"
    estimated_monthly_reviews_saved: int
    estimated_risk_change: str
    supporting_data: dict


@dataclass
class OptimizationReport:
    """Complete output of a policy analysis run."""
    period_start: str
    period_end: str
    total_decisions: int
    recommendations: list[Recommendation]
    summary_text: str


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class ApprovalPolicyOptimizer:
    """
    Analyze historical approval decisions and recommend policy improvements.

    Analysis dimensions:
      1. High-approval-rate actions   → candidates for auto-approval
      2. High-rejection-rate actions  → agent logic needs improvement
      3. Frequent parameter edits     → agent proposes wrong defaults
      4. Slow / timed-out responses   → reviewer workload issues
      5. Cost threshold adjustment    → fine-tune min_cost thresholds
    """

    AUTO_APPROVAL_THRESHOLD = 0.95   # >= 95% approval → candidate for removal
    IMPROVEMENT_THRESHOLD   = 0.30   # >= 30% rejection → agent problem
    EDIT_THRESHOLD          = 0.20   # >= 20% edits     → bad parameter defaults

    def __init__(self, historical_data: list[ApprovalRecord]) -> None:
        self.data = historical_data

    # -- public API --------------------------------------------------------

    def analyze(self) -> OptimizationReport:
        """Run all analyses and return a consolidated optimization report."""
        recs: list[Recommendation] = []
        recs.extend(self.find_auto_approval_candidates())
        recs.extend(self.find_agent_improvement_areas())
        recs.extend(self.find_threshold_adjustments())
        recs.extend(self.find_workload_issues())

        # sort by impact priority, then by reviews saved
        impact_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda r: (impact_order.get(r.impact, 9),
                                 -r.estimated_monthly_reviews_saved))

        period_start, period_end = self._period_range()
        report = OptimizationReport(
            period_start=period_start,
            period_end=period_end,
            total_decisions=len(self.data),
            recommendations=recs,
            summary_text=self.generate_report(recs, period_start, period_end),
        )
        return report

    def find_auto_approval_candidates(
        self, threshold: float = 0.95
    ) -> list[Recommendation]:
        """
        Find actions whose approval rate exceeds *threshold*.
        These are candidates for removing the human-review requirement entirely.
        """
        by_action = self._group_by_action()
        recs = []

        for action, records in by_action.items():
            total = len(records)
            if total < 10:          # not enough data
                continue
            approved = sum(
                1 for r in records if r.decision in ("approved", "approved_with_edits")
            )
            rate = approved / total
            if rate < threshold:
                continue

            avg_cost = statistics.mean(r.estimated_cost for r in records)
            monthly  = self._estimate_monthly(records)

            recs.append(Recommendation(
                title=f"Auto-approve '{action}' actions",
                description=(
                    f"'{action}' has a {rate:.1%} approval rate ({approved}/{total} approved). "
                    f"Average cost: ${avg_cost:.2f}. "
                    f"Removing the approval requirement could save ~{monthly} "
                    f"reviewer interruptions per month with very low additional risk."
                ),
                impact="high" if monthly > 50 else "medium",
                estimated_monthly_reviews_saved=monthly,
                estimated_risk_change=(
                    "Very low" if avg_cost < 10 else f"Low (avg ${avg_cost:.0f} per action)"
                ),
                supporting_data={
                    "action": action,
                    "total_decisions": total,
                    "approved": approved,
                    "approval_rate": rate,
                    "avg_cost": avg_cost,
                    "monthly_volume_estimate": monthly,
                },
            ))

        return recs

    def find_agent_improvement_areas(
        self, threshold: float = 0.30
    ) -> list[Recommendation]:
        """
        Find actions with a rejection rate above *threshold*.
        High rejection rates indicate the agent is proposing bad actions.
        """
        by_action = self._group_by_action()
        recs = []

        for action, records in by_action.items():
            total = len(records)
            if total < 5:
                continue
            rejected = sum(1 for r in records if r.decision == "rejected")
            rate = rejected / total
            if rate < threshold:
                continue

            # Extract common rejection themes from notes
            notes = [
                r.reviewer_notes.lower()
                for r in records
                if r.decision == "rejected" and r.reviewer_notes
            ]
            themes = self._extract_themes(notes)
            monthly = self._estimate_monthly(records)

            recs.append(Recommendation(
                title=f"Improve agent logic for '{action}'",
                description=(
                    f"'{action}' has a {rate:.1%} rejection rate ({rejected}/{total} rejected). "
                    f"Common rejection reasons: {themes}. "
                    f"Fixing the agent's proposal logic could eliminate ~{int(monthly * rate)} "
                    f"incorrect proposals per month."
                ),
                impact="high" if rate > 0.40 else "medium",
                estimated_monthly_reviews_saved=int(monthly * rate),
                estimated_risk_change="Neutral (reduces bad proposals)",
                supporting_data={
                    "action": action,
                    "total_decisions": total,
                    "rejected": rejected,
                    "rejection_rate": rate,
                    "common_rejection_themes": themes,
                },
            ))

        return recs

    def find_threshold_adjustments(self) -> list[Recommendation]:
        """
        Identify cost thresholds where the approval rate is high enough to
        justify raising the auto-approval ceiling.
        """
        refund_records = [r for r in self.data if r.action == "issue_refund"]
        if len(refund_records) < 10:
            return []

        recs = []

        # Define candidate brackets
        brackets = [(0, 500), (500, 750), (750, 1000), (1000, float("inf"))]
        for lo, hi in brackets:
            segment = [
                r for r in refund_records
                if lo <= r.estimated_cost < hi
            ]
            if len(segment) < 5:
                continue
            approved = sum(
                1 for r in segment if r.decision in ("approved", "approved_with_edits")
            )
            rate = approved / len(segment)

            if rate < 0.88:
                continue                    # not safe to auto-approve this range
            if lo == 0:
                continue                    # already auto-approved below threshold

            monthly_segment = self._estimate_monthly(segment)
            exposure = sum(r.estimated_cost for r in segment) / len(segment) * monthly_segment

            recs.append(Recommendation(
                title=f"Raise refund auto-approval threshold to ${hi:.0f}",
                description=(
                    f"Refunds ${lo:.0f}–${hi:.0f}: {rate:.1%} approval rate "
                    f"({approved}/{len(segment)} approved). "
                    f"Raising the threshold from ${lo:.0f} to ${hi:.0f} could save "
                    f"~{monthly_segment} reviews/month. "
                    f"Additional monthly risk exposure: ~${exposure:,.0f}."
                ),
                impact="high" if monthly_segment > 30 else "medium",
                estimated_monthly_reviews_saved=monthly_segment,
                estimated_risk_change=f"+${exposure:,.0f}/month additional exposure",
                supporting_data={
                    "bracket": f"${lo}–${hi}",
                    "total": len(segment),
                    "approved": approved,
                    "approval_rate": rate,
                    "estimated_monthly_exposure": exposure,
                },
            ))

        return recs

    def find_workload_issues(
        self, max_response_time: float = 300.0
    ) -> list[Recommendation]:
        """
        Identify patterns in slow responses and high timeout rates that
        suggest reviewer overload or under-staffing at peak hours.
        """
        if not self.data:
            return []

        recs = []

        # Overall response time analysis
        response_times = [r.response_time for r in self.data if r.response_time > 0]
        if response_times:
            avg_rt = statistics.mean(response_times)
            slow = [rt for rt in response_times if rt > max_response_time]
            slow_rate = len(slow) / len(response_times)

            if avg_rt > max_response_time * 0.6:
                suggested_timeout = int(avg_rt * 2 / 60) * 60     # round up to minutes
                recs.append(Recommendation(
                    title="Reviewers are overloaded — reduce queue or add reviewers",
                    description=(
                        f"Average response time is {avg_rt:.0f}s (target: {max_response_time:.0f}s). "
                        f"{slow_rate:.1%} of requests exceed the target. "
                        f"Consider adding reviewers during peak hours or raising the "
                        f"timeout from {max_response_time:.0f}s to {suggested_timeout}s."
                    ),
                    impact="medium" if slow_rate < 0.15 else "high",
                    estimated_monthly_reviews_saved=0,
                    estimated_risk_change="Neutral (fewer auto-rejections due to timeout)",
                    supporting_data={
                        "avg_response_time_s": avg_rt,
                        "slow_rate": slow_rate,
                        "suggested_timeout_s": suggested_timeout,
                    },
                ))

        # Peak-hour analysis (bucket by hour-of-day)
        hour_counts: dict[int, int] = defaultdict(int)
        for rec in self.data:
            hour = datetime.fromtimestamp(rec.timestamp, tz=timezone.utc).hour
            hour_counts[hour] += 1

        if hour_counts:
            sorted_hours = sorted(hour_counts.items(), key=lambda x: -x[1])
            peak_hour, peak_count = sorted_hours[0]
            overall_avg = sum(hour_counts.values()) / len(hour_counts)

            if peak_count > overall_avg * 2:
                recs.append(Recommendation(
                    title=f"Add reviewer coverage at peak hour {peak_hour:02d}:00",
                    description=(
                        f"Hour {peak_hour:02d}:00 has {peak_count} requests "
                        f"vs {overall_avg:.0f} average. "
                        f"Peak-hour queue depth may cause long customer wait times. "
                        f"Consider scheduling an additional reviewer during this window."
                    ),
                    impact="medium",
                    estimated_monthly_reviews_saved=0,
                    estimated_risk_change="Neutral (faster decisions, same guardrails)",
                    supporting_data={
                        "peak_hour": peak_hour,
                        "peak_count": peak_count,
                        "average_hourly_count": overall_avg,
                    },
                ))

        # Timeout / auto-reject rate by risk level
        for risk in ("low", "medium", "high", "critical"):
            subset = [r for r in self.data if r.risk_level == risk]
            if len(subset) < 5:
                continue
            # Heuristic: response_time == -1 means timeout in our synthetic data
            timeouts = sum(1 for r in subset if r.response_time < 0)
            timeout_rate = timeouts / len(subset)
            if timeout_rate > 0.10:
                recs.append(Recommendation(
                    title=f"High timeout rate for {risk}-risk requests ({timeout_rate:.0%})",
                    description=(
                        f"{timeout_rate:.1%} of {risk}-risk requests time out "
                        f"before a reviewer responds ({timeouts}/{len(subset)}). "
                        f"These are auto-rejected for safety but create a poor user "
                        f"experience.  Consider increasing the timeout or adding "
                        f"dedicated reviewers for {risk}-risk actions."
                    ),
                    impact="low" if timeout_rate < 0.15 else "medium",
                    estimated_monthly_reviews_saved=0,
                    estimated_risk_change=f"Slight increase (fewer auto-rejections)",
                    supporting_data={
                        "risk_level": risk,
                        "total": len(subset),
                        "timeouts": timeouts,
                        "timeout_rate": timeout_rate,
                    },
                ))

        return recs

    def generate_report(
        self,
        recs: list[Recommendation] | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> str:
        if recs is None:
            report = self.analyze()
            return report.summary_text

        ps = period_start or self._period_range()[0]
        pe = period_end   or self._period_range()[1]
        total = len(self.data)

        total_saved = sum(r.estimated_monthly_reviews_saved for r in recs)

        lines = [
            "",
            "APPROVAL POLICY OPTIMIZATION REPORT",
            "=" * 44,
            f"Period : {ps}  →  {pe}",
            f"Total decisions analyzed: {total:,}",
            f"Recommendations generated: {len(recs)}",
            f"Estimated monthly reviews saved (if all applied): {total_saved:,}",
            "",
            "TOP RECOMMENDATIONS (sorted by impact):",
            "",
        ]

        impact_emoji = {"high": "[HIGH IMPACT]", "medium": "[MEDIUM IMPACT]", "low": "[LOW IMPACT]"}
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {impact_emoji.get(rec.impact, '')} {rec.title}")
            # wrap description
            for para in rec.description.split(". "):
                if para:
                    lines.append(f"   {para.strip()}.")
            lines.append(f"   → Monthly reviews saved : {rec.estimated_monthly_reviews_saved}")
            lines.append(f"   → Risk change           : {rec.estimated_risk_change}")
            # Show 2-3 key data points
            for k, v in list(rec.supporting_data.items())[:3]:
                if isinstance(v, float):
                    lines.append(f"   → {k:30s}: {v:.3f}")
                else:
                    lines.append(f"   → {k:30s}: {v}")
            lines.append("")

        return "\n".join(lines)

    # -- private helpers ---------------------------------------------------

    def _group_by_action(self) -> dict[str, list[ApprovalRecord]]:
        groups: dict[str, list[ApprovalRecord]] = defaultdict(list)
        for rec in self.data:
            groups[rec.action].append(rec)
        return dict(groups)

    def _estimate_monthly(self, records: list[ApprovalRecord]) -> int:
        """Estimate monthly volume from the sample period."""
        if len(records) < 2:
            return len(records)
        span_days = (
            max(r.timestamp for r in records)
            - min(r.timestamp for r in records)
        ) / 86400
        if span_days < 1:
            span_days = 1
        daily_rate = len(records) / span_days
        return max(1, int(daily_rate * 30))

    def _period_range(self) -> tuple[str, str]:
        if not self.data:
            return "N/A", "N/A"
        timestamps = [r.timestamp for r in self.data]
        fmt = "%Y-%m-%d"
        start = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).strftime(fmt)
        end   = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).strftime(fmt)
        return start, end

    def _extract_themes(self, notes: list[str]) -> str:
        """Very simple keyword counter to summarise common rejection reasons."""
        keywords = [
            "amount", "evidence", "documentation", "policy", "wrong", "duplicate",
            "partial", "limit", "expired", "fraud", "high", "low", "format",
        ]
        counts: dict[str, int] = defaultdict(int)
        for note in notes:
            for kw in keywords:
                if kw in note:
                    counts[kw] += 1

        if not counts:
            return "(insufficient notes data)"

        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        total_notes = max(len(notes), 1)
        return ", ".join(f"'{kw}' ({cnt/total_notes:.0%})" for kw, cnt in top)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_records(n: int = 500, seed: int = 42) -> list[ApprovalRecord]:
    """
    Generate *n* realistic approval records spanning a 30-day period.

    Approval rates per action type:
      - send_email           : 98% approved
      - issue_refund (< $500): 95% approved
      - issue_refund ($500-$1000): 85% approved
      - issue_refund (> $1000): 55% approved
      - update_database      : 90% approved
      - cancel_subscription  : 70% approved
      - export_user_data     : 60% approved
    """
    rng = random.Random(seed)
    now = time.time()
    start_ts = now - 30 * 86400

    reviewers = ["r-alice", "r-bob", "r-carol", "r-dave"]

    # (action, weight, risk_level, cost_fn, approval_rate, notes_rejected, notes_edited)
    action_specs = [
        ("send_email",          25, "medium", lambda r: 0.0,                    0.98,
         ["content not approved", "wrong template", "missing compliance review"],
         ["subject line needs edit", "wrong recipient"]),

        ("issue_refund_low",    20, "medium", lambda r: rng.uniform(10, 499),   0.95,
         ["insufficient evidence", "duplicate request"],
         ["amount too high", "partial refund only"]),

        ("issue_refund_mid",    15, "high",   lambda r: rng.uniform(500, 999),  0.85,
         ["amount too high", "missing evidence", "policy limit exceeded"],
         ["partial refund", "reduced amount"]),

        ("issue_refund_high",   10, "high",   lambda r: rng.uniform(1000, 2500), 0.55,
         ["amount too high", "no evidence", "fraud pattern", "policy limit"],
         ["partial refund only", "max allowed is $1000"]),

        ("update_database",     12, "medium", lambda r: 50.0,                   0.90,
         ["wrong table", "missing approval chain"],
         ["update subset only"]),

        ("cancel_subscription", 10, "high",   lambda r: rng.uniform(20, 200),   0.70,
         ["retention not attempted", "customer on contract", "wrong account"],
         ["apply pause instead"]),

        ("export_user_data",     8, "high",   lambda r: 0.0,                    0.60,
         ["gdpr basis missing", "wrong user id", "format not supported"],
         ["limit fields exported"]),
    ]

    # normalize weights
    weights = [s[1] for s in action_specs]
    total_w = sum(weights)
    cum_weights = []
    c = 0
    for w in weights:
        c += w / total_w
        cum_weights.append(c)

    records = []
    for _ in range(n):
        # pick action
        r = rng.random()
        spec_idx = next(i for i, cw in enumerate(cum_weights) if r <= cw)
        action_name, _, risk, cost_fn, approval_rate, rej_notes, edit_notes = action_specs[spec_idx]

        # Normalise action name
        action = action_name.replace("_low", "").replace("_mid", "").replace("_high", "")

        cost = cost_fn(rng)
        reviewer = rng.choice(reviewers)

        # Determine decision
        roll = rng.random()
        if roll < approval_rate * 0.85:
            decision = "approved"
            notes = None
        elif roll < approval_rate:
            decision = "approved_with_edits"
            notes = rng.choice(edit_notes) if edit_notes else None
        else:
            decision = "rejected"
            notes = rng.choice(rej_notes) if rej_notes else None

        # Response time: normal-ish around 90s, some outliers/timeouts
        rt_roll = rng.random()
        if rt_roll > 0.92:
            response_time = -1.0        # simulated timeout
        elif rt_roll > 0.75:
            response_time = rng.uniform(200, 400)
        else:
            response_time = rng.uniform(15, 180)

        timestamp = start_ts + rng.uniform(0, 30 * 86400)

        records.append(ApprovalRecord(
            timestamp=timestamp,
            action=action,
            params={"cost": cost},
            risk_level=risk,
            estimated_cost=cost,
            decision=decision,
            reviewer_id=reviewer,
            response_time=response_time,
            reviewer_notes=notes,
        ))

    # sort by timestamp for realism
    records.sort(key=lambda r: r.timestamp)
    return records


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def run_demo() -> None:
    print_section("APPROVAL POLICY OPTIMIZER DEMO")

    # Generate synthetic data
    print(f"\nGenerating 500 synthetic approval records…")
    records = generate_synthetic_records(n=500)

    # Action summary
    action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        action_counts[rec.action][rec.decision] += 1

    print("\nSynthetic dataset summary:")
    print(f"  {'Action':<25}  {'Total':>6}  {'Approved%':>10}  {'Rejected%':>10}  {'Edited%':>10}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    for action, counts in sorted(action_counts.items()):
        total = sum(counts.values())
        app   = (counts.get("approved", 0) + counts.get("approved_with_edits", 0)) / total
        rej   = counts.get("rejected", 0) / total
        edit  = counts.get("approved_with_edits", 0) / total
        print(f"  {action:<25}  {total:>6}  {app:>9.1%}  {rej:>9.1%}  {edit:>9.1%}")

    # Run optimizer
    print("\nRunning optimizer…")
    optimizer = ApprovalPolicyOptimizer(records)
    report = optimizer.analyze()

    # Print report
    print_section("OPTIMIZATION REPORT")
    print(report.summary_text)

    # Monthly impact summary
    total_saved = sum(r.estimated_monthly_reviews_saved for r in report.recommendations)
    high_impact = [r for r in report.recommendations if r.impact == "high"]

    print_section("MONTHLY IMPACT SUMMARY")
    print(f"  Total recommendations      : {len(report.recommendations)}")
    print(f"  High-impact recommendations: {len(high_impact)}")
    print(f"  Potential reviews saved/mo : {total_saved:,}")
    print()

    for rec in report.recommendations:
        bar_len = min(40, rec.estimated_monthly_reviews_saved)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  [{rec.impact.upper():6s}] {rec.estimated_monthly_reviews_saved:4d} reviews  "
              f"{bar}  {rec.title[:40]}")

    print()


if __name__ == "__main__":
    run_demo()
