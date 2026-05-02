"""Token usage tracker — audit trail and cost estimator for LLM calls.

Records every API call's token usage, computes running costs, enforces
budget caps, and generates human-readable reports.

Designed to be shared across all components in the agent pipeline so you
get a single source of truth for total spend.

See: docs/03-memory-and-retrieval/01-short-term-memory.md
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table (USD per 1 000 tokens)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

# Fraction of budget that triggers a warning log
_BUDGET_WARNING_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Record of a single LLM API call."""

    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    purpose: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Compute cost based on the PRICING table; returns 0.0 for unknown models."""
        prices = PRICING.get(self.model, {})
        if not prices:
            return 0.0
        return (
            self.input_tokens / 1000 * prices["input"]
            + self.output_tokens / 1000 * prices["output"]
        )


# ---------------------------------------------------------------------------
# Token Tracker
# ---------------------------------------------------------------------------


class TokenTracker:
    """Thread-safe accumulator for LLM token usage and cost.

    Args:
        budget_cap: Optional total spend cap in USD.  When 80 % is consumed,
                    a WARNING is logged.  When exceeded, ``is_budget_exceeded``
                    returns True.
    """

    def __init__(self, budget_cap: Optional[float] = None) -> None:
        self.budget_cap = budget_cap
        self._records: list[TokenUsage] = []
        self._lock = threading.RLock()  # Reentrant: record_call holds lock and calls total_cost()
        self._budget_warned = False  # track whether we've already warned

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str = "",
    ) -> TokenUsage:
        """Record a completed API call.

        Args:
            model:         Model name (e.g. "gpt-4o").
            input_tokens:  Prompt tokens consumed.
            output_tokens: Completion tokens produced.
            purpose:       Optional label for this call (e.g. "summarize").

        Returns:
            The ``TokenUsage`` record that was stored.
        """
        record = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            purpose=purpose,
        )
        with self._lock:
            self._records.append(record)
            total_cost = self.total_cost()

        if self.budget_cap is not None:
            fraction = total_cost / self.budget_cap
            if fraction >= 1.0:
                logger.error(
                    "Budget exceeded: $%.4f / $%.4f (%.0f%%)",
                    total_cost,
                    self.budget_cap,
                    fraction * 100,
                )
            elif fraction >= _BUDGET_WARNING_THRESHOLD and not self._budget_warned:
                self._budget_warned = True
                logger.warning(
                    "Budget warning: $%.4f / $%.4f (%.0f%% used)",
                    total_cost,
                    self.budget_cap,
                    fraction * 100,
                )

        logger.debug(
            "Recorded call: model=%s in=%d out=%d cost=$%.6f purpose=%s",
            model,
            input_tokens,
            output_tokens,
            record.cost_usd,
            purpose or "—",
        )
        return record

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def total_input_tokens(self) -> int:
        """Total input tokens across all recorded calls."""
        with self._lock:
            return sum(r.input_tokens for r in self._records)

    def total_output_tokens(self) -> int:
        """Total output tokens across all recorded calls."""
        with self._lock:
            return sum(r.output_tokens for r in self._records)

    def total_tokens(self) -> int:
        """Total tokens (input + output) across all calls."""
        return self.total_input_tokens() + self.total_output_tokens()

    def total_cost(self) -> float:
        """Total spend in USD across all recorded calls."""
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    def estimate_remaining(self) -> Optional[float]:
        """Remaining budget in USD, or None if no budget cap is set."""
        if self.budget_cap is None:
            return None
        return max(0.0, self.budget_cap - self.total_cost())

    def is_budget_exceeded(self) -> bool:
        """Return True if total cost has exceeded the budget cap."""
        if self.budget_cap is None:
            return False
        return self.total_cost() >= self.budget_cap

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Return a human-readable report as a multi-line string."""
        with self._lock:
            records = list(self._records)

        if not records:
            return "Token Tracker: no calls recorded."

        total_in = sum(r.input_tokens for r in records)
        total_out = sum(r.output_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)

        # Per-model breakdown
        by_model: dict[str, dict] = {}
        for r in records:
            m = by_model.setdefault(
                r.model,
                {"calls": 0, "input": 0, "output": 0, "cost": 0.0},
            )
            m["calls"] += 1
            m["input"] += r.input_tokens
            m["output"] += r.output_tokens
            m["cost"] += r.cost_usd

        lines = [
            "=" * 60,
            "TOKEN USAGE REPORT",
            "=" * 60,
            f"  Total calls:         {len(records)}",
            f"  Total input tokens:  {total_in:,}",
            f"  Total output tokens: {total_out:,}",
            f"  Total tokens:        {total_in + total_out:,}",
            f"  Total cost:          ${total_cost:.6f}",
        ]
        if self.budget_cap is not None:
            pct = total_cost / self.budget_cap * 100
            remaining = max(0.0, self.budget_cap - total_cost)
            lines += [
                f"  Budget cap:          ${self.budget_cap:.2f}",
                f"  Budget used:         {pct:.1f}%",
                f"  Budget remaining:    ${remaining:.6f}",
            ]
        lines.append("")
        lines.append("  BY MODEL:")
        for model_name, stats in sorted(by_model.items()):
            lines.append(
                f"    {model_name:<20s} {stats['calls']:>4d} calls, "
                f"{stats['input']:>8,} in, {stats['output']:>8,} out, "
                f"${stats['cost']:.6f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export all usage records as a JSON string."""
        with self._lock:
            records = [asdict(r) for r in self._records]
        return json.dumps(
            {
                "total_calls": len(records),
                "total_input_tokens": sum(r["input_tokens"] for r in records),
                "total_output_tokens": sum(r["output_tokens"] for r in records),
                "total_cost_usd": sum(
                    (PRICING.get(r["model"], {}).get("input", 0) * r["input_tokens"] / 1000)
                    + (PRICING.get(r["model"], {}).get("output", 0) * r["output_tokens"] / 1000)
                    for r in records
                ),
                "budget_cap": self.budget_cap,
                "records": records,
            },
            indent=2,
        )

    def reset(self) -> None:
        """Clear all records and reset the budget warning flag."""
        with self._lock:
            self._records.clear()
            self._budget_warned = False


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tracker = TokenTracker(budget_cap=0.01)  # Low cap so warning triggers

    # Simulate a series of API calls
    tracker.record_call("gpt-4o", 1500, 350, purpose="plan")
    tracker.record_call("gpt-4o-mini", 800, 200, purpose="summarize")
    tracker.record_call("gpt-4o", 2000, 400, purpose="execute")
    tracker.record_call("gpt-4o-mini", 600, 150, purpose="summarize")
    tracker.record_call("gpt-4o", 1800, 380, purpose="finalize")

    print(tracker.generate_report())
    print(f"Budget exceeded: {tracker.is_budget_exceeded()}")
    print(f"Remaining:       ${tracker.estimate_remaining():.6f}")
    print(f"JSON export:\n{tracker.to_json()}")


if __name__ == "__main__":
    main()
