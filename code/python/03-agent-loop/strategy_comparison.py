"""Run the same task through all three planning strategies and compare.

Strategies compared:
  1. ReAct (react loop)          — plan-while-executing
  2. Plan-and-Execute            — structured plan first, then parallel execution
  3. Reflection                  — generate + self-critique + revise

Metrics collected:
  - Wall-clock time (seconds)
  - Estimated LLM calls (tracked via a counting wrapper)
  - Token usage (prompt + completion where available)
  - Estimated cost (using gpt-4o public pricing at time of writing)
  - Answer character length

Usage:
    export OPENAI_API_KEY=sk-...
    python strategy_comparison.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from plan_execute_agent import PlanAndExecuteAgent
from reflection_agent import ReflectionAgent

# ---------------------------------------------------------------------------
# Cost constants (gpt-4o, 2024 pricing — update as needed)
# ---------------------------------------------------------------------------

_MODEL = "gpt-4o"
_COST_PER_1K_PROMPT_TOKENS = 0.005   # USD per 1 000 input tokens
_COST_PER_1K_COMPLETION_TOKENS = 0.015  # USD per 1 000 output tokens


# ---------------------------------------------------------------------------
# Counting OpenAI client wrapper
# ---------------------------------------------------------------------------


@dataclass
class CallMetrics:
    """Accumulated metrics for a single strategy run."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1000 * _COST_PER_1K_PROMPT_TOKENS
            + self.completion_tokens / 1000 * _COST_PER_1K_COMPLETION_TOKENS
        )


class CountingClient:
    """Thin wrapper around OpenAI that counts calls and aggregates token usage."""

    def __init__(self, metrics: CallMetrics, real_client: OpenAI) -> None:
        self._metrics = metrics
        self._real = real_client
        # Expose chat.completions.create at the right path
        self.chat = _ChatWrapper(metrics, real_client)


class _CompletionsWrapper:
    def __init__(self, metrics: CallMetrics, real_client: OpenAI) -> None:
        self._metrics = metrics
        self._real = real_client

    def create(self, **kwargs: Any) -> Any:
        self._metrics.llm_calls += 1
        response = self._real.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage:
            self._metrics.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._metrics.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        return response


class _ChatWrapper:
    def __init__(self, metrics: CallMetrics, real_client: OpenAI) -> None:
        self.completions = _CompletionsWrapper(metrics, real_client)


# ---------------------------------------------------------------------------
# ReAct strategy (simple direct-call version without the full tool loop)
# We import agent.run_agent but mock the client to count tokens.
# ---------------------------------------------------------------------------


def _run_react(task: str, real_client: OpenAI) -> tuple[str, CallMetrics]:
    """Run a simplified ReAct-style agent and return (answer, metrics)."""
    from agent import run_agent  # noqa: PLC0415

    metrics = CallMetrics()

    # Patch the module-level openai client inside agent.py
    import agent as agent_mod  # noqa: PLC0415

    original_client = getattr(agent_mod, "_client", None)
    counting = CountingClient(metrics, real_client)

    # agent.py creates its client inline; we patch via monkey-patch
    _original_OpenAI = None
    try:
        import openai as _openai_mod  # noqa: PLC0415

        _original_OpenAI = _openai_mod.OpenAI

        class _PatchedOpenAI:
            def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                return counting

        _openai_mod.OpenAI = _PatchedOpenAI  # type: ignore[assignment]

        from tools import TOOLS  # noqa: PLC0415

        answer = run_agent(task, tools=TOOLS)
    finally:
        if _original_OpenAI is not None:
            import openai as _openai_mod  # noqa: PLC0415

            _openai_mod.OpenAI = _original_OpenAI  # type: ignore[assignment]

    return answer, metrics


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


@dataclass
class StrategyResult:
    name: str
    answer: str
    elapsed_sec: float
    metrics: CallMetrics
    error: str | None = None


def run_comparison(task: str) -> list[StrategyResult]:
    """Run all three strategies sequentially and collect metrics."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    real_client = OpenAI(api_key=api_key)
    results: list[StrategyResult] = []

    # --- 1. ReAct ---
    print("\n[1/3] Running ReAct agent…")
    try:
        t0 = time.monotonic()
        react_metrics = CallMetrics()
        counting = CountingClient(react_metrics, real_client)
        import openai as _openai_mod

        _orig = _openai_mod.OpenAI

        class _PatchedOpenAI:
            def __new__(cls, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
                return counting

        _openai_mod.OpenAI = _PatchedOpenAI  # type: ignore[assignment]
        try:
            from agent import run_agent
            from tools import TOOLS

            react_answer = run_agent(task, tools=TOOLS)
        finally:
            _openai_mod.OpenAI = _orig  # type: ignore[assignment]
        elapsed = time.monotonic() - t0
        results.append(StrategyResult("ReAct", react_answer, elapsed, react_metrics))
        print(f"    Done ({elapsed:.1f}s, {react_metrics.llm_calls} LLM calls)")
    except Exception as exc:  # noqa: BLE001
        results.append(StrategyResult("ReAct", "", 0.0, CallMetrics(), error=str(exc)))
        print(f"    ERROR: {exc}")

    # --- 2. Plan-and-Execute ---
    print("[2/3] Running Plan-and-Execute agent…")
    try:
        pe_metrics = CallMetrics()
        pe_client = CountingClient(pe_metrics, real_client)
        pe_agent = PlanAndExecuteAgent(model=_MODEL)
        pe_agent._client = pe_client  # type: ignore[assignment]

        t0 = time.monotonic()
        pe_output = pe_agent.run(task)
        elapsed = time.monotonic() - t0
        pe_answer = pe_output["answer"]
        results.append(StrategyResult("Plan-and-Execute", pe_answer, elapsed, pe_metrics))
        print(f"    Done ({elapsed:.1f}s, {pe_metrics.llm_calls} LLM calls)")
    except Exception as exc:  # noqa: BLE001
        results.append(StrategyResult("Plan-and-Execute", "", 0.0, CallMetrics(), error=str(exc)))
        print(f"    ERROR: {exc}")

    # --- 3. Reflection ---
    print("[3/3] Running Reflection agent…")
    try:
        ref_metrics = CallMetrics()
        ref_client = CountingClient(ref_metrics, real_client)
        ref_agent = ReflectionAgent(model=_MODEL, max_reflections=2, quality_threshold=8)
        ref_agent._client = ref_client  # type: ignore[assignment]

        t0 = time.monotonic()
        ref_output = ref_agent.run(task)
        elapsed = time.monotonic() - t0
        ref_answer = ref_output["final_answer"]
        ref_metrics.extra["reflections_used"] = ref_output["reflections_used"]
        results.append(StrategyResult("Reflection", ref_answer, elapsed, ref_metrics))
        print(f"    Done ({elapsed:.1f}s, {ref_metrics.llm_calls} LLM calls)")
    except Exception as exc:  # noqa: BLE001
        results.append(StrategyResult("Reflection", "", 0.0, CallMetrics(), error=str(exc)))
        print(f"    ERROR: {exc}")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(task: str, results: list[StrategyResult]) -> None:
    col_w = [20, 8, 12, 8, 8, 12, 10]
    headers = ["Strategy", "Time(s)", "LLM Calls", "Prompt", "Compl.", "Est.Cost($)", "Ans.Len"]

    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(f"\n{'=' * 60}")
    print(f"Task: {task[:80]}")
    print("=" * 60)
    print(sep)
    print(fmt.format(*headers))
    print(sep)

    for r in results:
        if r.error:
            row = [r.name, "ERROR", r.error[:30], "-", "-", "-", "-"]
        else:
            row = [
                r.name,
                f"{r.elapsed_sec:.1f}",
                str(r.metrics.llm_calls),
                str(r.metrics.prompt_tokens),
                str(r.metrics.completion_tokens),
                f"{r.metrics.estimated_cost_usd:.4f}",
                str(len(r.answer)),
            ]
        print(fmt.format(*row))
    print(sep)

    # Recommendation
    valid = [r for r in results if not r.error]
    if valid:
        fastest = min(valid, key=lambda r: r.elapsed_sec)
        cheapest = min(valid, key=lambda r: r.metrics.estimated_cost_usd)
        most_detailed = max(valid, key=lambda r: len(r.answer))
        print("\nRecommendations:")
        print(f"  Fastest:      {fastest.name} ({fastest.elapsed_sec:.1f}s)")
        print(f"  Cheapest:     {cheapest.name} (${cheapest.metrics.estimated_cost_usd:.4f})")
        print(f"  Most detailed:{most_detailed.name} ({len(most_detailed.answer)} chars)")
        print()
        print("  Strategy selection guide:")
        print("  - Simple factual queries     → ReAct (low latency, low cost)")
        print("  - Multi-source research      → Plan-and-Execute (parallel efficiency)")
        print("  - High-stakes writing/analysis → Reflection (best quality)")

    print()

    # Show first 300 chars of each answer
    print("Answer previews (first 300 chars):")
    for r in results:
        print(f"\n  [{r.name}]")
        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            preview = r.answer[:300].replace("\n", " ")
            print(f"  {preview}…" if len(r.answer) > 300 else f"  {preview}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    task = (
        "Research the top 3 cloud providers (AWS, Azure, GCP). "
        "For each: summarise their market share, key strengths, and a weakness. "
        "Conclude with a recommendation for a startup choosing a cloud provider."
    )

    results = run_comparison(task)
    print_report(task, results)


if __name__ == "__main__":
    main()
