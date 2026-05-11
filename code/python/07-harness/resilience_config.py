"""
resilience_config.py — Configuration builder for the resilience layer.

Provides ready-made resilience profiles for the four most common deployment
contexts, plus a mathematically-derived builder that works backwards from SLO
targets.

Usage:

    builder = ResilienceConfigBuilder()

    # Named profiles
    cfg = builder.for_user_facing_api()
    cfg = builder.for_background_job()
    cfg = builder.for_critical_path()
    cfg = builder.for_cost_sensitive()

    # SLO-derived
    cfg = builder.from_slo(target_availability=0.999, target_latency_p99=2.0)

    # Build real objects
    layer = cfg.build("llm_call", providers=[primary, secondary])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from resilience_layer import (
    CircuitBreaker,
    FallbackExecutor,
    FallbackLevel,
    ResilienceLayer,
    RetryConfig,
)


# ---------------------------------------------------------------------------
# ResilienceConfig data class
# ---------------------------------------------------------------------------

@dataclass
class ResilienceConfig:
    """
    Aggregated configuration for the full resilience layer.

    Attributes:
        profile_name:              Human-readable label.
        description:               Why this profile exists.
        max_retries:               Maximum retry attempts on the primary path.
        base_delay_seconds:        Initial retry delay.
        max_delay_seconds:         Retry delay cap.
        backoff_multiplier:        Exponential factor.
        total_deadline_seconds:    Hard wall-clock deadline for the primary path.
        circuit_failure_threshold: Failures before the circuit opens.
        circuit_recovery_seconds:  Seconds before the circuit moves to HALF_OPEN.
        circuit_failure_window:    Rolling window for counting failures.
        fallback_capabilities:     Ordered list of capability labels for each
                                   fallback level (e.g. ["full", "reduced", "static"]).
        notes:                     Design rationale notes.
    """

    profile_name: str
    description: str
    max_retries: int
    base_delay_seconds: float
    max_delay_seconds: float
    backoff_multiplier: float
    total_deadline_seconds: float
    circuit_failure_threshold: int
    circuit_recovery_seconds: float
    circuit_failure_window: float
    fallback_capabilities: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Builder helpers
    # -----------------------------------------------------------------------

    def retry_config(
        self,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> RetryConfig:
        """Return a :class:`RetryConfig` built from this profile."""
        from resilience_layer import RateLimitError

        base_exceptions: tuple[type[Exception], ...] = (
            TimeoutError,
            ConnectionError,
            RateLimitError,
        )
        return RetryConfig(
            max_retries=self.max_retries,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            backoff_multiplier=self.backoff_multiplier,
            total_deadline_seconds=self.total_deadline_seconds,
            retryable_exceptions=retryable_exceptions or base_exceptions,
        )

    def circuit_breaker(self, name: str) -> CircuitBreaker:
        """Return a :class:`CircuitBreaker` built from this profile."""
        return CircuitBreaker(
            name=name,
            failure_threshold=self.circuit_failure_threshold,
            recovery_timeout_seconds=self.circuit_recovery_seconds,
            failure_window_seconds=self.circuit_failure_window,
        )

    def build(
        self,
        name: str,
        providers: list[tuple[str, Any, str, float]] | None = None,
    ) -> ResilienceLayer:
        """
        Construct a :class:`ResilienceLayer` from this config.

        Args:
            name:      Layer name used in logs and metrics.
            providers: Optional list of ``(level_name, callable, capability, timeout)``
                       tuples.  If omitted, an empty :class:`FallbackExecutor` is used.

        Returns:
            Fully configured :class:`ResilienceLayer`.
        """
        levels: list[FallbackLevel] = []
        if providers:
            for i, (lvl_name, provider, capability, timeout) in enumerate(providers):
                levels.append(
                    FallbackLevel(
                        name=lvl_name,
                        provider=provider,
                        timeout_seconds=timeout,
                        capability=capability,
                        cost_multiplier=float(i + 1),
                    )
                )

        return ResilienceLayer(
            name=name,
            circuit_breaker=self.circuit_breaker(name),
            retry_config=self.retry_config(),
            fallback_executor=FallbackExecutor(levels),
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (for tables, JSON, etc.)."""
        return {
            "profile": self.profile_name,
            "max_retries": self.max_retries,
            "base_delay_s": self.base_delay_seconds,
            "max_delay_s": self.max_delay_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "total_deadline_s": self.total_deadline_seconds,
            "cb_failure_threshold": self.circuit_failure_threshold,
            "cb_recovery_s": self.circuit_recovery_seconds,
            "cb_failure_window_s": self.circuit_failure_window,
            "fallback_levels": " → ".join(self.fallback_capabilities) or "(none)",
        }


# ---------------------------------------------------------------------------
# ResilienceConfigBuilder
# ---------------------------------------------------------------------------

class ResilienceConfigBuilder:
    """
    Factory for :class:`ResilienceConfig` objects tuned to common deployment
    contexts.

    Each named profile documents its design rationale so that the resulting
    configuration is self-explanatory.
    """

    # -----------------------------------------------------------------------
    # Named profiles
    # -----------------------------------------------------------------------

    def for_user_facing_api(self) -> ResilienceConfig:
        """
        User-facing API — optimise for responsiveness.

        Design priorities:
        - Fail fast: users expect < 5 s response times.
        - Two retries maximum (third failure is probably not transient).
        - Circuit breaker trips after only 3 failures to protect the API
          pool from slow-to-fail upstream calls.
        - Cross-provider fallback to maintain user experience.
        """
        return ResilienceConfig(
            profile_name="user_facing_api",
            description="Optimise for responsiveness. Fail fast, degrade gracefully.",
            max_retries=2,
            base_delay_seconds=0.5,
            max_delay_seconds=5.0,
            backoff_multiplier=2.0,
            total_deadline_seconds=10.0,
            circuit_failure_threshold=3,
            circuit_recovery_seconds=30.0,
            circuit_failure_window=60.0,
            fallback_capabilities=["full (cross-provider)", "reduced", "static"],
            notes=[
                "base_delay=0.5s: visible to user — keep small",
                "total_deadline=10s: p99 budget for external calls",
                "cb_threshold=3: trip quickly to prevent pool exhaustion",
                "cb_recovery=30s: short recovery for user-visible service",
            ],
        )

    def for_background_job(self) -> ResilienceConfig:
        """
        Background job — optimise for completion.

        Design priorities:
        - Retry aggressively; the job may run for minutes.
        - Long total deadline: 5 minutes is acceptable for batch processing.
        - Circuit breaker threshold is high (10 failures) to avoid
          premature failure of long-running jobs with noisy dependencies.
        """
        return ResilienceConfig(
            profile_name="background_job",
            description="Optimise for completion. Retry aggressively, wait patiently.",
            max_retries=5,
            base_delay_seconds=2.0,
            max_delay_seconds=120.0,
            backoff_multiplier=2.0,
            total_deadline_seconds=300.0,
            circuit_failure_threshold=10,
            circuit_recovery_seconds=300.0,
            circuit_failure_window=120.0,
            fallback_capabilities=["full", "full (cross-provider)", "reduced"],
            notes=[
                "max_retries=5: total wait up to 2+4+8+16+32=62s before giving up",
                "total_deadline=300s: generous for nightly batch jobs",
                "cb_threshold=10: don't open the circuit on a flaky dependency",
                "cb_recovery=300s: long recovery mirrors job cadence",
            ],
        )

    def for_critical_path(self) -> ResilienceConfig:
        """
        Critical path — optimise for reliability.

        Design priorities:
        - Balance latency and reliability: 30 s deadline is tight enough for
          real-time interactions but allows 3 retries.
        - Multiple full-capability fallback levels before degrading.
        - Circuit breaker trips at 5 failures (standard production threshold).
        """
        return ResilienceConfig(
            profile_name="critical_path",
            description="Optimise for reliability. Balance latency and redundancy.",
            max_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            backoff_multiplier=2.0,
            total_deadline_seconds=30.0,
            circuit_failure_threshold=5,
            circuit_recovery_seconds=60.0,
            circuit_failure_window=60.0,
            fallback_capabilities=[
                "full (provider A)",
                "full (provider B)",
                "reduced",
                "static",
            ],
            notes=[
                "3 retries: 1+2+4=7s of backoff before fallback",
                "two full-capability fallbacks before degrading",
                "cb_threshold=5: industry-standard trip threshold",
                "cb_recovery=60s: long enough to avoid thrashing",
            ],
        )

    def for_cost_sensitive(self) -> ResilienceConfig:
        """
        Cost-sensitive — minimise expensive retries.

        Design priorities:
        - Only one retry: each call costs money; don't retry excessively.
        - Fall back to cheapest option (small model / cached result) quickly.
        - Short total deadline prevents accumulating costs on hanging calls.
        """
        return ResilienceConfig(
            profile_name="cost_sensitive",
            description="Minimise cost. One retry, then cheapest fallback.",
            max_retries=1,
            base_delay_seconds=1.0,
            max_delay_seconds=10.0,
            backoff_multiplier=2.0,
            total_deadline_seconds=15.0,
            circuit_failure_threshold=3,
            circuit_recovery_seconds=120.0,
            circuit_failure_window=60.0,
            fallback_capabilities=["reduced (cheap model)", "static (free)"],
            notes=[
                "max_retries=1: avoid accumulating token costs on failures",
                "fallback to cheap model: maintain service at fraction of cost",
                "static fallback: zero-cost last resort",
                "cb_recovery=120s: avoid hammering expensive API during outage",
            ],
        )

    # -----------------------------------------------------------------------
    # SLO-derived builder
    # -----------------------------------------------------------------------

    def from_slo(
        self,
        target_availability: float,
        target_latency_p99: float,
    ) -> ResilienceConfig:
        """
        Derive a resilience configuration from SLO targets.

        Mathematical derivation
        -----------------------

        **Availability → retry budget**

        The probability that *k* independent identical calls all fail is
        ``p_fail ** k``, where ``p_fail`` is the estimated single-call failure
        rate.  We set ``p_fail`` pessimistically to::

            p_fail = 1 - sqrt(target_availability)

        …which means that achieving ``target_availability`` requires that
        *two* successive attempts have a combined availability of at least
        ``target_availability``.

        The minimum number of attempts ``k`` satisfies::

            p_fail ** k  ≤  1 - target_availability
            k ≥ log(1 - target_availability) / log(p_fail)

        We round up and clamp to [1, 5].

        **Latency → deadline and base delay**

        ``total_deadline_seconds`` is set to ``0.8 × target_latency_p99``
        (leaving 20 % headroom for the caller stack).

        ``base_delay_seconds`` is sized so that ``max_retries`` retries fit
        inside the deadline with room to spare::

            budget = total_deadline × 0.6   # 60 % of deadline for backoff
            base   = budget / (2 ** max_retries - 1)  # geometric series

        **Availability → circuit breaker thresholds**

        Higher availability targets tolerate fewer failures before tripping::

            ≥ 0.999  → threshold 3,  recovery 30 s
            ≥ 0.99   → threshold 5,  recovery 60 s
            ≥ 0.95   → threshold 10, recovery 120 s

        Args:
            target_availability:  Desired fraction of requests served successfully
                                  (e.g. 0.999 = three nines).
            target_latency_p99:   Desired p99 latency budget in seconds.

        Returns:
            :class:`ResilienceConfig` derived from the SLO.
        """
        if not (0.5 < target_availability < 1.0):
            raise ValueError("target_availability must be between 0.5 and 1.0 (exclusive)")
        if target_latency_p99 <= 0:
            raise ValueError("target_latency_p99 must be positive")

        # --- Retry count from availability ---
        p_fail = 1.0 - math.sqrt(target_availability)
        p_fail = max(p_fail, 1e-6)  # avoid log(0)
        tolerance = 1.0 - target_availability

        if tolerance <= 0:
            raw_retries = 5
        else:
            raw_retries = math.ceil(
                math.log(tolerance) / math.log(p_fail)
            ) - 1  # -1 because one attempt is not a "retry"

        max_retries = max(1, min(5, raw_retries))

        # --- Deadline and base delay from latency ---
        total_deadline = target_latency_p99 * 0.8

        # Budget for all retry sleeps = 60 % of the deadline
        sleep_budget = total_deadline * 0.6
        # Geometric series sum: base * (2^k - 1) ≈ sleep_budget
        denominator = (2 ** max_retries) - 1
        base_delay = sleep_budget / max(denominator, 1)
        base_delay = max(0.1, round(base_delay, 2))
        max_delay = min(total_deadline * 0.4, 60.0)

        # --- Circuit breaker thresholds from availability ---
        if target_availability >= 0.999:
            cb_threshold, cb_recovery = 3, 30.0
        elif target_availability >= 0.99:
            cb_threshold, cb_recovery = 5, 60.0
        else:
            cb_threshold, cb_recovery = 10, 120.0

        # --- Fallback levels from availability ---
        if target_availability >= 0.999:
            fallback_caps = ["full (cross-provider)", "reduced", "static"]
        elif target_availability >= 0.99:
            fallback_caps = ["full (cross-provider)", "static"]
        else:
            fallback_caps = ["reduced"]

        return ResilienceConfig(
            profile_name=f"slo_{target_availability:.3f}_p99_{target_latency_p99:.1f}s",
            description=(
                f"SLO-derived: availability={target_availability:.3f}, "
                f"p99_latency={target_latency_p99:.1f}s"
            ),
            max_retries=max_retries,
            base_delay_seconds=base_delay,
            max_delay_seconds=round(max_delay, 2),
            backoff_multiplier=2.0,
            total_deadline_seconds=round(total_deadline, 2),
            circuit_failure_threshold=cb_threshold,
            circuit_recovery_seconds=cb_recovery,
            circuit_failure_window=60.0,
            fallback_capabilities=fallback_caps,
            notes=[
                f"p_fail estimate: {p_fail:.4f}",
                f"raw retry count: {raw_retries} (clamped to {max_retries})",
                f"sleep budget: {sleep_budget:.2f}s ({int(60)}% of deadline)",
                f"base_delay derived from geometric series over {max_retries} retries",
            ],
        )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_config(cfg: ResilienceConfig, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}Profile     : {cfg.profile_name}")
    print(f"{pad}Description : {cfg.description}")
    print(f"{pad}Retry       : max={cfg.max_retries}  base={cfg.base_delay_seconds}s"
          f"  max={cfg.max_delay_seconds}s  multiplier={cfg.backoff_multiplier}"
          f"  deadline={cfg.total_deadline_seconds}s")
    print(f"{pad}Circuit     : threshold={cfg.circuit_failure_threshold}"
          f"  recovery={cfg.circuit_recovery_seconds}s"
          f"  window={cfg.circuit_failure_window}s")
    if cfg.fallback_capabilities:
        print(f"{pad}Fallbacks   : {' → '.join(cfg.fallback_capabilities)}")
    if cfg.notes:
        for note in cfg.notes:
            print(f"{pad}  • {note}")


def _print_comparison_table(configs: list[ResilienceConfig]) -> None:
    """Print all configs side by side in a markdown-style table."""
    keys = [
        ("Profile",        lambda c: c.profile_name),
        ("Max retries",    lambda c: str(c.max_retries)),
        ("Base delay (s)", lambda c: f"{c.base_delay_seconds:.2f}"),
        ("Max delay (s)",  lambda c: f"{c.max_delay_seconds:.1f}"),
        ("Deadline (s)",   lambda c: f"{c.total_deadline_seconds:.1f}"),
        ("CB threshold",   lambda c: str(c.circuit_failure_threshold)),
        ("CB recovery (s)",lambda c: f"{c.circuit_recovery_seconds:.0f}"),
        ("Fallback levels",lambda c: str(len(c.fallback_capabilities))),
    ]

    # Compute column widths
    col_widths = [max(len(k[0]), max(len(k[1](c)) for c in configs)) for k in keys]

    def row(values: list[str]) -> str:
        return "  " + " | ".join(v.ljust(w) for v, w in zip(values, col_widths))

    header_row = row([k[0] for k in keys])
    sep_row = "  " + "-+-".join("-" * w for w in col_widths)

    print(header_row)
    print(sep_row)
    for cfg in configs:
        print(row([k[1](cfg) for k in keys]))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    builder = ResilienceConfigBuilder()

    print("\n" + "=" * 70)
    print("  RESILIENCE CONFIGURATION BUILDER DEMO")
    print("=" * 70)

    # ---- Named profiles ----
    profiles = [
        ("User-Facing API",  builder.for_user_facing_api()),
        ("Background Job",   builder.for_background_job()),
        ("Critical Path",    builder.for_critical_path()),
        ("Cost-Sensitive",   builder.for_cost_sensitive()),
    ]

    for title, cfg in profiles:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")
        _print_config(cfg)

    # ---- SLO-derived profiles ----
    print("\n\n" + "=" * 70)
    print("  SLO-DERIVED CONFIGURATIONS")
    print("=" * 70)

    slo_targets = [
        (0.999, 5.0,  "Three nines, 5 s p99"),
        (0.999, 2.0,  "Three nines, 2 s p99"),
        (0.99,  10.0, "Two nines, 10 s p99"),
        (0.95,  30.0, "95 %, 30 s p99"),
    ]

    slo_configs: list[ResilienceConfig] = []
    for availability, latency, label in slo_targets:
        cfg = builder.from_slo(availability, latency)
        slo_configs.append(cfg)
        print(f"\n  {label}")
        print(f"  {'─' * 60}")
        _print_config(cfg)

    # ---- Comparison table ----
    print("\n\n" + "=" * 70)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 70 + "\n")

    all_configs = [c for _, c in profiles] + slo_configs
    _print_comparison_table(all_configs)
    print()


if __name__ == "__main__":
    main()
