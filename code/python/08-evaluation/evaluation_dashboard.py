"""
evaluation_dashboard.py
=======================
Rich terminal dashboard and HTML export for evaluation reports.

Renders color-coded tables, per-query breakdowns, regression alerts,
ASCII sparkline trend charts, and a standalone HTML export.

Requires: pip install rich

See: docs/08-evaluation-and-guardrails/01-evaluating-agents.md
"""

from __future__ import annotations

import asyncio
import html
import math
import os
from dataclasses import dataclass
from typing import Optional

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from agent_evaluator import (
    ContinuousEvaluationPipeline,
    EndToEndEvaluator,
    EndToEndReport,
    EndToEndTestCase,
    FullEvaluationReport,
    GenerationEvaluator,
    GenerationReport,
    GenerationTestCase,
    LLMJudge,
    RegressionCheck,
    RetrievalEvaluator,
    RetrievalReport,
    RetrievalTestCase,
    _DemoAgent,
    _DemoRetriever,
    _build_generation_tests,
    _build_retrieval_corpus,
    _build_retrieval_tests,
    _build_e2e_tests,
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_RETRIEVAL_TARGETS = {
    "hit_rate": 0.90,
    "precision_at_k": 0.70,
    "recall_at_k": 0.80,
    "mrr": 0.60,
    "ndcg_at_k": 0.70,
}

_GENERATION_TARGETS = {
    "overall_pass_rate": 0.85,
    "contains_required_rate": 0.90,
    "avoids_forbidden_rate": 0.95,
    "faithfulness_pass_rate": 0.90,
    "relevance_pass_rate": 0.90,
    "completeness_pass_rate": 0.85,
}

_E2E_TARGETS = {
    "task_success_rate": 0.85,
}


def _status_style(value: float, target: float) -> str:
    """Return a Rich style string based on proximity to the target."""
    if value >= target:
        return "green"
    if value >= target - 0.10:
        return "yellow"
    return "red"


def _status_icon(value: float, target: float) -> str:
    if value >= target:
        return "✅ PASS"
    if value >= target - 0.10:
        return "⚠️  NEAR"
    return "❌ FAIL"


# ---------------------------------------------------------------------------
# EvaluationDashboard
# ---------------------------------------------------------------------------


class EvaluationDashboard:
    """
    Generate formatted evaluation reports for terminal display and HTML export.

    Args:
        evaluator: Optional evaluator instance (not directly used during
                   render calls; reserved for future integrations).
    """

    def __init__(self, evaluator: object = None) -> None:
        self.evaluator = evaluator
        if _RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def render_retrieval_report(self, report: RetrievalReport) -> None:
        """Render a color-coded retrieval evaluation report."""
        if not _RICH_AVAILABLE:
            print(report.to_string())
            return

        self.console.print(
            Panel("[bold cyan]RETRIEVAL EVALUATION REPORT[/bold cyan]", expand=False)
        )

        # Overall metrics table
        metrics_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column("Target", justify="right")
        metrics_table.add_column("Status")

        rows = [
            ("Hit Rate", report.hit_rate, _RETRIEVAL_TARGETS["hit_rate"]),
            ("Precision@5", report.precision_at_k, _RETRIEVAL_TARGETS["precision_at_k"]),
            ("Recall@5", report.recall_at_k, _RETRIEVAL_TARGETS["recall_at_k"]),
            ("MRR", report.mrr, _RETRIEVAL_TARGETS["mrr"]),
            ("NDCG@5", report.ndcg_at_k, _RETRIEVAL_TARGETS["ndcg_at_k"]),
        ]

        for label, value, target in rows:
            style = _status_style(value, target)
            metrics_table.add_row(
                label,
                Text(f"{value:.2%}", style=style),
                f"> {target:.0%}",
                Text(_status_icon(value, target), style=style),
            )

        self.console.print(metrics_table)
        self.console.print(
            f"\nTotal queries: {report.total_queries}  |  "
            f"Queries with zero relevant results: {report.queries_with_zero_results}\n"
        )

        # Per-query breakdown
        pq_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        pq_table.add_column("#", style="dim", width=4)
        pq_table.add_column("Query", max_width=45)
        pq_table.add_column("Hit", justify="center")
        pq_table.add_column("Prec@5", justify="right")
        pq_table.add_column("Recall@5", justify="right")
        pq_table.add_column("MRR", justify="right")
        pq_table.add_column("NDCG@5", justify="right")

        for i, r in enumerate(report.per_query, start=1):
            hit_icon = "✅" if r["hit"] else "❌"
            pq_table.add_row(
                str(i),
                r["query"][:44],
                hit_icon,
                f"{r['precision_at_k']:.2f}",
                f"{r['recall_at_k']:.2f}",
                f"{r['reciprocal_rank']:.2f}",
                f"{r['ndcg_at_k']:.2f}",
            )

        self.console.print(pq_table)

        if report.queries_with_zero_results:
            self.console.print(
                f"[yellow]⚠️  {report.queries_with_zero_results} query/queries returned "
                "zero relevant results.[/yellow]\n"
            )

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    def render_generation_report(self, report: GenerationReport) -> None:
        """Render generation evaluation with rule-based and LLM judge results."""
        if not _RICH_AVAILABLE:
            print(report.to_string())
            return

        self.console.print(
            Panel("[bold cyan]GENERATION EVALUATION REPORT[/bold cyan]", expand=False)
        )

        metrics_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column("Target", justify="right")
        metrics_table.add_column("Status")

        overall_rows = [
            ("Overall Pass Rate", report.overall_pass_rate, _GENERATION_TARGETS["overall_pass_rate"]),
            ("Contains Required", report.contains_required_rate, _GENERATION_TARGETS["contains_required_rate"]),
            ("Avoids Forbidden", report.avoids_forbidden_rate, _GENERATION_TARGETS["avoids_forbidden_rate"]),
        ]
        judge_rows = [
            ("Faithfulness (judge)", report.faithfulness_pass_rate, _GENERATION_TARGETS["faithfulness_pass_rate"]),
            ("Relevance (judge)", report.relevance_pass_rate, _GENERATION_TARGETS["relevance_pass_rate"]),
            ("Completeness (judge)", report.completeness_pass_rate, _GENERATION_TARGETS["completeness_pass_rate"]),
        ]

        for label, value, target in overall_rows:
            style = _status_style(value, target)
            metrics_table.add_row(
                label,
                Text(f"{value:.2%}", style=style),
                f"> {target:.0%}",
                Text(_status_icon(value, target), style=style),
            )

        for label, value, target in judge_rows:
            if value is None:
                continue
            style = _status_style(value, target)
            metrics_table.add_row(
                label,
                Text(f"{value:.2%}", style=style),
                f"> {target:.0%}",
                Text(_status_icon(value, target), style=style),
            )

        self.console.print(metrics_table)
        self.console.print(f"\nTotal queries: {report.total_queries}\n")

        # Per-query detail
        detail_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        detail_table.add_column("#", style="dim", width=4)
        detail_table.add_column("Query", max_width=40)
        detail_table.add_column("Pass", justify="center")
        detail_table.add_column("Contains", justify="center")
        detail_table.add_column("Forbidden", justify="center")
        detail_table.add_column("Length", justify="center")

        for i, r in enumerate(report.per_query, start=1):
            rc = r["rule_checks"]
            pass_icon = "✅" if r["overall_pass"] else "❌"
            contains_icon = "✅" if rc.get("contains_required", True) else "❌"
            forbidden_icon = "✅" if rc.get("avoids_forbidden", True) else "❌"
            length_icon = "✅" if rc.get("length_ok", True) else "❌"
            detail_table.add_row(
                str(i),
                r["query"][:39],
                pass_icon,
                contains_icon,
                forbidden_icon,
                length_icon,
            )

        self.console.print(detail_table)

        # Show judge explanations for failures
        failures = [r for r in report.per_query if not r["overall_pass"]]
        if failures:
            self.console.print("[bold red]Failed test details:[/bold red]")
            for r in failures:
                self.console.print(f"  Query: {r['query'][:60]}")
                rc = r["rule_checks"]
                if rc.get("missing_required"):
                    self.console.print(f"    Missing required: {rc['missing_required']}")
                if rc.get("found_forbidden"):
                    self.console.print(f"    Found forbidden: {rc['found_forbidden']}")
                for dim, judge_result in r.get("judge_checks", {}).items():
                    if not judge_result.passed:
                        self.console.print(
                            f"    [{dim}] {judge_result.explanation}"
                        )
            self.console.print()

    # -----------------------------------------------------------------------
    # End-to-End
    # -----------------------------------------------------------------------

    def render_end_to_end_report(self, report: EndToEndReport) -> None:
        """Render end-to-end evaluation scenario outcomes."""
        if not _RICH_AVAILABLE:
            print(report.to_string())
            return

        self.console.print(
            Panel("[bold cyan]END-TO-END EVALUATION REPORT[/bold cyan]", expand=False)
        )

        target = _E2E_TARGETS["task_success_rate"]
        style = _status_style(report.task_success_rate, target)
        self.console.print(
            f"Task Success Rate: [{style}]{report.task_success_rate:.2%}[/{style}]  "
            f"(target: > {target:.0%})\n"
            f"Avg Turns to Resolution: {report.avg_turns_to_resolution:.1f}\n"
        )

        scenario_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        scenario_table.add_column("#", style="dim", width=4)
        scenario_table.add_column("Scenario", max_width=45)
        scenario_table.add_column("Outcome", justify="center")
        scenario_table.add_column("Expected", justify="center")
        scenario_table.add_column("Turns", justify="right")
        scenario_table.add_column("Result", justify="center")

        for i, r in enumerate(report.per_scenario, start=1):
            success = r["success"]
            result_style = "green" if success else "red"
            outcome_style = (
                "green" if r["outcome"] == "resolved" else
                "yellow" if r["outcome"] == "escalated" else "red"
            )
            scenario_table.add_row(
                str(i),
                r["scenario"][:44],
                Text(r["outcome"], style=outcome_style),
                r["expected_outcome"],
                str(r["turns_taken"]),
                Text("✅ PASS" if success else "❌ FAIL", style=result_style),
            )

        self.console.print(scenario_table)
        self.console.print()

    # -----------------------------------------------------------------------
    # Full report
    # -----------------------------------------------------------------------

    def render_full_report(self, report: FullEvaluationReport) -> None:
        """Render a comprehensive report with all three evaluation levels."""
        if not _RICH_AVAILABLE:
            print(report.to_string())
            return

        self.console.print(
            Panel(
                "[bold white]FULL EVALUATION REPORT — EXECUTIVE SUMMARY[/bold white]",
                style="bold blue",
                expand=True,
            )
        )

        # Executive summary table
        summary_table = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold")
        summary_table.add_column("Level", style="bold")
        summary_table.add_column("Key Metric", justify="right")
        summary_table.add_column("Target", justify="right")
        summary_table.add_column("Status")

        r = report.retrieval
        g = report.generation
        e = report.end_to_end

        summary_rows = [
            ("Retrieval", "Hit Rate", r.hit_rate, 0.90),
            ("Retrieval", "MRR", r.mrr, 0.60),
            ("Generation", "Overall Pass", g.overall_pass_rate, 0.85),
            ("End-to-End", "Task Success", e.task_success_rate, 0.85),
        ]

        for level, metric, value, target in summary_rows:
            style = _status_style(value, target)
            summary_table.add_row(
                level,
                f"{metric}: {value:.2%}",
                f"> {target:.0%}",
                Text(_status_icon(value, target), style=style),
            )

        self.console.print(summary_table)
        self.console.print()

        self.render_retrieval_report(report.retrieval)
        self.render_generation_report(report.generation)
        self.render_end_to_end_report(report.end_to_end)

    # -----------------------------------------------------------------------
    # Regression check
    # -----------------------------------------------------------------------

    def render_regression_check(self, check: RegressionCheck) -> None:
        """Render regression check results."""
        if not _RICH_AVAILABLE:
            print(check.to_string())
            return

        if not check.has_regressions:
            self.console.print(
                Panel(
                    "[green]✅  No regressions detected. All metrics within acceptable range.[/green]",
                    expand=False,
                )
            )
            return

        self.console.print(
            Panel("[bold red]❌  REGRESSIONS DETECTED[/bold red]", expand=False)
        )

        reg_table = Table(box=box.ROUNDED, show_header=True, header_style="bold red")
        reg_table.add_column("Regression", min_width=60)

        for regression in check.regressions:
            reg_table.add_row(Text(f"• {regression}", style="red"))

        self.console.print(reg_table)
        self.console.print()

    # -----------------------------------------------------------------------
    # Trend chart (ASCII sparklines)
    # -----------------------------------------------------------------------

    def render_trend_chart(self, metric_history: list[dict]) -> None:
        """
        Render ASCII sparkline charts for the last up to 10 evaluation runs.

        *metric_history* is a list of dicts with keys:
        ``hit_rate``, ``mrr``, ``generation_pass_rate``, ``task_success_rate``.
        """
        if not _RICH_AVAILABLE:
            self._plain_trend_chart(metric_history)
            return

        runs = metric_history[-10:]
        metrics = [
            ("Hit Rate", [r.get("hit_rate", 0) for r in runs]),
            ("MRR", [r.get("mrr", 0) for r in runs]),
            ("Generation Pass", [r.get("generation_pass_rate", 0) for r in runs]),
            ("Task Success", [r.get("task_success_rate", 0) for r in runs]),
        ]

        self.console.print(
            Panel("[bold cyan]METRIC TREND (last 10 runs)[/bold cyan]", expand=False)
        )
        for label, values in metrics:
            sparkline = _build_sparkline(values)
            latest = values[-1] if values else 0
            self.console.print(f"  {label:<22} {sparkline}  {latest:.2%}")
        self.console.print()

    def _plain_trend_chart(self, metric_history: list[dict]) -> None:
        runs = metric_history[-10:]
        print("\nMETRIC TREND (last 10 runs)")
        print("=" * 40)
        for label, key in [
            ("Hit Rate", "hit_rate"),
            ("MRR", "mrr"),
            ("Generation Pass", "generation_pass_rate"),
            ("Task Success", "task_success_rate"),
        ]:
            values = [r.get(key, 0) for r in runs]
            sparkline = _build_sparkline(values)
            latest = values[-1] if values else 0
            print(f"  {label:<22} {sparkline}  {latest:.2%}")
        print()

    # -----------------------------------------------------------------------
    # HTML export
    # -----------------------------------------------------------------------

    def export_html(self, report: FullEvaluationReport, filepath: str) -> None:
        """Export the full evaluation report as a standalone HTML file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        def _badge(value: float, target: float) -> str:
            if value >= target:
                color = "#22c55e"  # green-500
            elif value >= target - 0.10:
                color = "#f59e0b"  # amber-500
            else:
                color = "#ef4444"  # red-500
            return (
                f'<span style="background:{color};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-weight:bold">{value:.2%}</span>'
            )

        r = report.retrieval
        g = report.generation
        e = report.end_to_end

        retrieval_rows = "".join(
            f"<tr>"
            f"<td>{html.escape(pq['query'][:80])}</td>"
            f"<td>{'✅' if pq['hit'] else '❌'}</td>"
            f"<td>{pq['precision_at_k']:.2f}</td>"
            f"<td>{pq['recall_at_k']:.2f}</td>"
            f"<td>{pq['reciprocal_rank']:.2f}</td>"
            f"<td>{pq['ndcg_at_k']:.2f}</td>"
            f"</tr>"
            for pq in r.per_query
        )

        generation_rows = "".join(
            f"<tr>"
            f"<td>{html.escape(pq['query'][:80])}</td>"
            f"<td>{'✅' if pq['overall_pass'] else '❌'}</td>"
            f"<td>{'✅' if pq['rule_checks'].get('contains_required', True) else '❌'}</td>"
            f"<td>{'✅' if pq['rule_checks'].get('avoids_forbidden', True) else '❌'}</td>"
            f"<td>{'✅' if pq['rule_checks'].get('length_ok', True) else '❌'}</td>"
            f"</tr>"
            for pq in g.per_query
        )

        e2e_rows = "".join(
            f"<tr>"
            f"<td>{html.escape(sc['scenario'][:80])}</td>"
            f"<td>{html.escape(sc['outcome'])}</td>"
            f"<td>{html.escape(sc['expected_outcome'])}</td>"
            f"<td>{sc['turns_taken']}</td>"
            f"<td>{'✅' if sc['success'] else '❌'}</td>"
            f"</tr>"
            for sc in e.per_scenario
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agent Evaluation Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f9fafb; color: #111; }}
    h1 {{ color: #1e40af; }}
    h2 {{ color: #1d4ed8; border-bottom: 2px solid #dbeafe; padding-bottom: .4rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: #fff;
             box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }}
    th {{ background: #1e40af; color: #fff; padding: .6rem 1rem; text-align: left; }}
    td {{ padding: .5rem 1rem; border-bottom: 1px solid #e5e7eb; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover {{ background: #eff6ff; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 1rem; margin-bottom: 2rem; }}
    .summary-card {{ background: #fff; border-radius: 8px; padding: 1rem;
                     box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
    .summary-card .value {{ font-size: 2rem; font-weight: bold; }}
    details {{ margin-bottom: 1rem; }}
    summary {{ cursor: pointer; font-weight: bold; padding: .4rem; }}
  </style>
</head>
<body>
  <h1>Agent Evaluation Report</h1>

  <div class="summary-grid">
    <div class="summary-card">
      <div>Hit Rate</div>
      <div class="value">{_badge(r.hit_rate, 0.90)}</div>
    </div>
    <div class="summary-card">
      <div>MRR</div>
      <div class="value">{_badge(r.mrr, 0.60)}</div>
    </div>
    <div class="summary-card">
      <div>Generation Pass</div>
      <div class="value">{_badge(g.overall_pass_rate, 0.85)}</div>
    </div>
    <div class="summary-card">
      <div>Task Success</div>
      <div class="value">{_badge(e.task_success_rate, 0.85)}</div>
    </div>
  </div>

  <h2>Retrieval</h2>
  <p>Precision@5: {_badge(r.precision_at_k, 0.70)} &nbsp;
     Recall@5: {_badge(r.recall_at_k, 0.80)} &nbsp;
     NDCG@5: {_badge(r.ndcg_at_k, 0.70)}</p>
  <details open><summary>Per-query breakdown</summary>
  <table>
    <tr><th>Query</th><th>Hit</th><th>Prec@5</th><th>Recall@5</th><th>MRR</th><th>NDCG@5</th></tr>
    {retrieval_rows}
  </table>
  </details>

  <h2>Generation</h2>
  <p>Contains Required: {_badge(g.contains_required_rate, 0.90)} &nbsp;
     Avoids Forbidden: {_badge(g.avoids_forbidden_rate, 0.95)}</p>
  <details><summary>Per-query breakdown</summary>
  <table>
    <tr><th>Query</th><th>Overall</th><th>Contains</th><th>Forbidden</th><th>Length</th></tr>
    {generation_rows}
  </table>
  </details>

  <h2>End-to-End</h2>
  <p>Avg turns to resolution: {e.avg_turns_to_resolution:.1f}</p>
  <details><summary>Scenario outcomes</summary>
  <table>
    <tr><th>Scenario</th><th>Outcome</th><th>Expected</th><th>Turns</th><th>Result</th></tr>
    {e2e_rows}
  </table>
  </details>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(html_content)

        print(f"HTML report exported to: {filepath}")


# ---------------------------------------------------------------------------
# Sparkline helpers
# ---------------------------------------------------------------------------

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _build_sparkline(values: list[float]) -> str:
    """Build an 8-block Unicode sparkline from a list of floats in [0, 1]."""
    if not values:
        return " " * 10
    min_v, max_v = min(values), max(values)
    span = max_v - min_v or 1.0
    bars = [_SPARK_CHARS[min(8, int((v - min_v) / span * 8))] for v in values]
    return "".join(bars)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------


async def main() -> None:
    import tempfile

    print("=" * 60)
    print("EVALUATION DASHBOARD — DEMO")
    print("=" * 60)

    # Build evaluators with demo stubs
    retrieval_tests = _build_retrieval_tests()
    generation_tests = _build_generation_tests()
    e2e_tests = _build_e2e_tests()

    retriever = _DemoRetriever(_build_retrieval_corpus())
    agent = _DemoAgent()

    retrieval_eval = RetrievalEvaluator(retriever, retrieval_tests)
    generation_eval = GenerationEvaluator(agent, generation_tests)
    e2e_eval = EndToEndEvaluator(agent, e2e_tests)

    pipeline = ContinuousEvaluationPipeline(
        harness=None,
        retrieval_evaluator=retrieval_eval,
        generation_evaluator=generation_eval,
        end_to_end_evaluator=e2e_eval,
    )

    dashboard = EvaluationDashboard()

    # Baseline
    print("\n[ Step 1 ] Rendering individual reports…\n")
    full_report = await pipeline.run_all()
    await pipeline.set_baseline()

    dashboard.render_retrieval_report(full_report.retrieval)
    dashboard.render_generation_report(full_report.generation)
    dashboard.render_end_to_end_report(full_report.end_to_end)

    # Full combined report
    print("\n[ Step 2 ] Rendering full combined report…\n")
    dashboard.render_full_report(full_report)

    # Regression check (degraded retriever)
    print("\n[ Step 3 ] Simulating regression and rendering check…\n")
    from agent_evaluator import _build_retrieval_corpus as _corpus
    degraded_retriever = _DemoRetriever(_corpus(degraded=True))
    pipeline.retrieval_evaluator = RetrievalEvaluator(degraded_retriever, retrieval_tests)
    regression = await pipeline.check_regression()
    dashboard.render_regression_check(regression)

    # Trend chart (simulated history)
    print("\n[ Step 4 ] Rendering metric trend chart…\n")
    import random

    random.seed(42)
    history = [
        {
            "hit_rate": 0.85 + random.uniform(-0.05, 0.08),
            "mrr": 0.62 + random.uniform(-0.05, 0.06),
            "generation_pass_rate": 0.80 + random.uniform(-0.05, 0.10),
            "task_success_rate": 0.82 + random.uniform(-0.05, 0.08),
        }
        for _ in range(10)
    ]
    dashboard.render_trend_chart(history)

    # HTML export
    print("\n[ Step 5 ] Exporting HTML report…")
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w"
    ) as f:
        html_path = f.name

    dashboard.export_html(full_report, html_path)
    print(f"  Written to: {html_path}\n")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
