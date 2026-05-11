"""
Agent Maturity Dashboard
=========================
Visual terminal dashboard showing 12-Factor Agent maturity using the rich library.
Also exports a standalone HTML report.

Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md

Dependencies: pip install rich
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

from twelve_factor_assessor import (
    TwelveFactorAssessor,
    TwelveFactorReport,
    FactorAssessment,
    ComparisonReport,
    ImprovementTracker,
    FACTOR_NAMES,
    _ROMAN,
)


# ---------------------------------------------------------------------------
# Color / style helpers
# ---------------------------------------------------------------------------

def _score_color(score: int) -> str:
    """Rich color for a factor score."""
    if score >= 4:
        return "bold green"
    if score == 3:
        return "bold yellow"
    return "bold red"


def _score_icon(score: int) -> str:
    if score >= 4:
        return "✅"
    if score == 3:
        return "⚠️ "
    return "❌"


def _bar(score: int, width: int = 20) -> str:
    filled = round((score / 5) * width)
    return "█" * filled + "░" * (width - filled)


def _sparkline(score: int, max_score: int = 60, width: int = 10) -> str:
    filled = round((score / max_score) * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class MaturityDashboard:
    """
    Interactive terminal dashboard for 12-Factor Agent maturity reporting.

    Usage::

        assessor = TwelveFactorAssessor()
        report = assessor.assess(config)
        dash = MaturityDashboard(assessor)
        dash.render_full(report)
    """

    def __init__(self, assessor: TwelveFactorAssessor) -> None:
        self.assessor = assessor
        self.console = Console() if _RICH_AVAILABLE else None

    # ------------------------------------------------------------------
    # Overview panel
    # ------------------------------------------------------------------

    def render_overview(self, report: TwelveFactorReport) -> None:
        """Render the maturity scorecard overview."""
        pct = round((report.overall_score / 60) * 100)
        bar_width = 40
        filled = round((report.overall_score / 60) * bar_width)
        progress_bar = "█" * filled + "░" * (bar_width - filled)

        if _RICH_AVAILABLE and self.console:
            color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
            content = Text()
            content.append(f"\n  Overall Score: ", style="bold")
            content.append(f"{report.overall_score}/60\n", style=f"bold {color}")
            content.append(f"  Maturity Level: ", style="bold")
            content.append(
                f"{report.maturity_level.upper()} (Level {report.maturity_level_number})\n",
                style=f"bold {color}",
            )
            content.append(f"  Assessed:       {report.assessed_at}\n\n")
            content.append(f"  Progress: ", style="dim")
            content.append(progress_bar, style=color)
            content.append(f"  {pct}%\n")
            self.console.print(Panel(content, title="[bold]12-FACTOR AGENT MATURITY[/bold]", border_style=color))
        else:
            print("\n╔══════════════════════════════════════════════════════╗")
            print(  "║           12-FACTOR AGENT MATURITY                   ║")
            print(  "╠══════════════════════════════════════════════════════╣")
            print(f"║  Overall Score: {report.overall_score}/60{' ' * (37 - len(str(report.overall_score)))}║")
            print(f"║  Maturity Level: {report.maturity_level.upper()} (Level {report.maturity_level_number}){' ' * max(0, 33 - len(report.maturity_level) - 9)}║")
            print(f"║  Progress: {progress_bar}  {pct}%   ║")
            print("╚══════════════════════════════════════════════════════╝\n")

    # ------------------------------------------------------------------
    # Factor scores bar chart
    # ------------------------------------------------------------------

    def render_factor_scores(self, report: TwelveFactorReport) -> None:
        """Render individual factor scores as a coloured bar chart."""
        if _RICH_AVAILABLE and self.console:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim", width=5)
            table.add_column("Factor", min_width=26)
            table.add_column("Score", justify="center", width=8)
            table.add_column("Bar", width=22)
            table.add_column("", width=3)

            for f in report.factors:
                roman = _ROMAN[f.factor_number]
                color = _score_color(f.score)
                bar_text = Text(_bar(f.score), style=color)
                table.add_row(
                    roman,
                    f.factor_name,
                    Text(f"{f.score}/5", style=color),
                    bar_text,
                    _score_icon(f.score),
                )
            self.console.print("\n[bold cyan]Factor Scores[/bold cyan]")
            self.console.print(table)
        else:
            print("\nFactor Scores")
            print("-" * 62)
            for f in report.factors:
                roman = f"{_ROMAN[f.factor_number]:<5}"
                name = f"{f.factor_name:<26}"
                bar = _bar(f.score)
                icon = _score_icon(f.score)
                print(f"  {roman} {name} {bar} {f.score}/5 {icon}")
            print()

    # ------------------------------------------------------------------
    # Critical gaps
    # ------------------------------------------------------------------

    def render_gaps(self, report: TwelveFactorReport) -> None:
        """Render critical gaps with actionable recommendations."""
        critical = [f for f in report.factors if f.score <= 2]
        needs_work = [f for f in report.factors if f.score == 3]

        if _RICH_AVAILABLE and self.console:
            if critical:
                self.console.print("\n[bold red]❌ CRITICAL GAPS (Must Fix)[/bold red]")
                for i, f in enumerate(critical, 1):
                    panel_content = Text()
                    panel_content.append(f"Score: {f.score}/5\n", style="bold red")
                    for g in f.gaps:
                        panel_content.append(f"  • Missing: {g}\n", style="red")
                    for r in f.recommendations:
                        panel_content.append(f"  → {r}\n", style="green")
                    self.console.print(Panel(
                        panel_content,
                        title=f"[bold red]{i}. Factor {_ROMAN[f.factor_number]} — {f.factor_name}[/bold red]",
                        border_style="red",
                    ))

            if needs_work:
                self.console.print("\n[bold yellow]⚠️  NEEDS IMPROVEMENT[/bold yellow]")
                for f in needs_work:
                    self.console.print(
                        f"  [yellow]Factor {_ROMAN[f.factor_number]} ({f.factor_name}):[/yellow] "
                        + (f.recommendations[0] if f.recommendations else "See assessment")
                    )
        else:
            if critical:
                print("\n❌ CRITICAL GAPS (Must Fix)")
                print("-" * 50)
                for i, f in enumerate(critical, 1):
                    print(f"\n{i}. {f.factor_name} (Factor {_ROMAN[f.factor_number]}) — Score: {f.score}/5")
                    for g in f.gaps:
                        print(f"   Missing: {g}")
                    for r in f.recommendations:
                        print(f"   Fix: {r}")

            if needs_work:
                print("\n⚠️  NEEDS IMPROVEMENT")
                print("-" * 50)
                for f in needs_work:
                    print(f"  Factor {_ROMAN[f.factor_number]} ({f.factor_name}): {f.recommendations[0] if f.recommendations else ''}")

    # ------------------------------------------------------------------
    # Improvement path
    # ------------------------------------------------------------------

    def render_improvement_path(
        self,
        current: TwelveFactorReport,
        target_level: str,
    ) -> None:
        """Render the steps from current level to target level."""
        tracker = ImprovementTracker()
        tracker.save_baseline(current)
        steps = tracker.generate_roadmap(current, target_level)

        level_scores = {
            "prototype": 24, "development": 36, "staging": 48, "production": 60, "elite": 60,
        }
        target_score = level_scores.get(target_level.lower(), 60)
        gap = max(0, target_score - current.overall_score)

        if _RICH_AVAILABLE and self.console:
            self.console.print(
                f"\n[bold cyan]PATH TO {target_level.upper()} "
                f"(Current: {current.maturity_level} {current.overall_score}/60 | "
                f"Gap: {gap} points)[/bold cyan]"
            )
            for i, step in enumerate(steps, 1):
                style = "green" if "Level" not in step else "yellow"
                self.console.print(f"  [dim]{i}.[/dim] [{style}]{step}[/{style}]")
        else:
            print(f"\nPATH TO {target_level.upper()}")
            print(f"Current: {current.maturity_level} ({current.overall_score}/60) | Gap: {gap} points")
            print("-" * 55)
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")

    # ------------------------------------------------------------------
    # Historical trend
    # ------------------------------------------------------------------

    def render_trend(self, history: list[TwelveFactorReport]) -> None:
        """Render score trends over time as an ASCII sparkline table."""
        if not history:
            print("No assessment history to display.")
            return

        if _RICH_AVAILABLE and self.console:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            table.add_column("Date", width=12)
            table.add_column("Score", justify="right", width=7)
            table.add_column("Level", width=14)
            table.add_column("Trend", width=12)

            for r in history:
                color = "green" if r.overall_score >= 49 else "yellow" if r.overall_score >= 37 else "red"
                table.add_row(
                    r.assessed_at,
                    Text(f"{r.overall_score}/60", style=f"bold {color}"),
                    Text(r.maturity_level, style=color),
                    Text(_sparkline(r.overall_score), style=color),
                )
            self.console.print("\n[bold cyan]Assessment History[/bold cyan]")
            self.console.print(table)
        else:
            print("\nAssessment History")
            print(f"{'Date':<12} {'Score':>6}  {'Level':<16}  Trend")
            print("-" * 55)
            for r in history:
                bar = _sparkline(r.overall_score)
                print(f"{r.assessed_at:<12} {r.overall_score:>4}/60  {r.maturity_level:<16}  {bar}")

    # ------------------------------------------------------------------
    # Before/after comparison
    # ------------------------------------------------------------------

    def render_comparison(
        self,
        current: TwelveFactorReport,
        baseline: TwelveFactorReport,
    ) -> None:
        """Render a before/after comparison of factor scores."""
        comparison = self.assessor.compare_to_baseline(current, baseline)

        if _RICH_AVAILABLE and self.console:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            table.add_column("Factor", min_width=28)
            table.add_column("Before", justify="center", width=8)
            table.add_column("After", justify="center", width=8)
            table.add_column("Change", justify="center", width=10)

            all_factors = {f.factor_number: f for f in current.factors}
            base_scores = {f.factor_number: f.score for f in baseline.factors}

            for num in range(1, 13):
                f = all_factors[num]
                old = base_scores.get(num, 1)
                new = f.score
                delta = new - old
                if delta > 0:
                    change_text = Text(f"+{delta} " + "⬆" * delta, style="bold green")
                elif delta < 0:
                    change_text = Text(f"{delta} " + "⬇" * abs(delta), style="bold red")
                else:
                    change_text = Text("-", style="dim")
                table.add_row(
                    f"{_ROMAN[num]}  {f.factor_name}",
                    Text(str(old), style=_score_color(old)),
                    Text(str(new), style=_score_color(new)),
                    change_text,
                )
            table.add_section()
            delta_total = comparison.score_delta
            delta_style = "bold green" if delta_total > 0 else "bold red" if delta_total < 0 else "dim"
            table.add_row(
                "[bold]TOTAL[/bold]",
                str(comparison.baseline_score),
                str(comparison.current_score),
                Text(f"{delta_total:+d}", style=delta_style),
            )
            self.console.print("\n[bold cyan]Comparison to Baseline[/bold cyan]")
            self.console.print(table)
        else:
            print("\nComparison to Baseline")
            print(f"{'Factor':<30} {'Before':>6}  {'After':>5}  Change")
            print("-" * 55)
            base_scores = {f.factor_number: f.score for f in baseline.factors}
            for f in current.factors:
                old = base_scores.get(f.factor_number, 1)
                new = f.score
                delta = new - old
                symbol = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "-"
                name = f"{_ROMAN[f.factor_number]:<5} {f.factor_name}"
                print(f"  {name:<30} {old:>4}    {new:>4}   {symbol}")
            print("-" * 55)
            print(
                f"  {'TOTAL':<30} {comparison.baseline_score:>4}    {comparison.current_score:>4}   "
                f"{comparison.score_delta:+d}"
            )

    # ------------------------------------------------------------------
    # Full dashboard
    # ------------------------------------------------------------------

    def render_full(
        self,
        report: TwelveFactorReport,
        target_level: str = "production",
        history: list[TwelveFactorReport] | None = None,
        baseline: TwelveFactorReport | None = None,
    ) -> None:
        """Render the complete dashboard."""
        self.render_overview(report)
        self.render_factor_scores(report)
        self.render_gaps(report)
        self.render_improvement_path(report, target_level)
        if history:
            self.render_trend(history)
        if baseline:
            self.render_comparison(report, baseline)

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------

    def export_html(self, report: TwelveFactorReport, filepath: str) -> None:
        """Export the dashboard as a standalone HTML file."""
        pct = round((report.overall_score / 60) * 100)
        score_colors = {5: "#22c55e", 4: "#86efac", 3: "#eab308", 2: "#ef4444", 1: "#b91c1c"}

        def _html_bar(score: int) -> str:
            color = score_colors.get(score, "#888")
            width = score * 20
            return (
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="background:#e2e8f0;border-radius:4px;width:100px;height:14px">'
                f'<div style="background:{color};width:{width}px;height:14px;border-radius:4px"></div></div>'
                f'<span style="color:{color};font-weight:700">{score}/5</span></div>'
            )

        rows = "".join(
            f'<tr>'
            f'<td style="font-weight:600;color:#64748b">{_ROMAN[f.factor_number]}</td>'
            f'<td>{f.factor_name}</td>'
            f'<td>{_html_bar(f.score)}</td>'
            f'<td>{_score_icon(f.score)}</td>'
            f'</tr>'
            for f in report.factors
        )

        gaps_html = "".join(
            f'<li><strong>Factor {_ROMAN[f.factor_number]} — {f.factor_name}</strong>'
            f'<ul>{"".join(f"<li>{g}</li>" for g in f.gaps)}</ul>'
            f'<em>Fix: {"<br>".join(f.recommendations)}</em></li>'
            for f in report.factors if f.score <= 2
        ) or "<li>No critical gaps!</li>"

        prio_html = "".join(f"<li>{p}</li>" for p in report.improvement_priorities)

        level_color = "#22c55e" if pct >= 80 else "#eab308" if pct >= 50 else "#ef4444"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>12-Factor Agent Maturity Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ color: #0f172a; margin-bottom: 0.5rem; }}
  .scorecard {{ background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,.12); margin-bottom: 2rem; display: flex; align-items: center; gap: 2rem; }}
  .score-circle {{ width: 120px; height: 120px; border-radius: 50%; background: conic-gradient({level_color} {pct}%, #e2e8f0 {pct}%); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
  .score-inner {{ background: white; width: 96px; height: 96px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
  .score-num {{ font-size: 1.75rem; font-weight: 700; color: {level_color}; line-height: 1; }}
  .score-denom {{ font-size: 0.75rem; color: #94a3b8; }}
  .score-info h2 {{ font-size: 1.5rem; color: {level_color}; }}
  .score-info p {{ color: #64748b; margin-top: 0.25rem; }}
  .progress-bar {{ background: #e2e8f0; border-radius: 99px; height: 8px; width: 300px; margin-top: 0.75rem; }}
  .progress-fill {{ background: {level_color}; border-radius: 99px; height: 8px; width: {pct}%; }}
  section {{ background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.12); margin-bottom: 1.5rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: #0f172a; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #f1f5f9; text-align: left; }}
  th {{ font-weight: 600; color: #64748b; font-size: 0.85rem; background: #f8fafc; }}
  ul.gaps li {{ color: #dc2626; margin: 0.5rem 0 0.5rem 1.5rem; }}
  ul.gaps li ul li {{ color: #64748b; }}
  ol.priorities li {{ margin: 0.4rem 0 0.4rem 1.5rem; color: #1e293b; }}
  em {{ color: #16a34a; font-style: normal; font-size: 0.9rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>12-Factor Agent Maturity Dashboard</h1>
  <p style="color:#64748b;margin-bottom:1.5rem">Assessed on {report.assessed_at}</p>

  <div class="scorecard">
    <div class="score-circle">
      <div class="score-inner">
        <span class="score-num">{report.overall_score}</span>
        <span class="score-denom">/60</span>
      </div>
    </div>
    <div class="score-info">
      <h2>{report.maturity_level} (Level {report.maturity_level_number})</h2>
      <p>{pct}% toward Elite</p>
      <div class="progress-bar"><div class="progress-fill"></div></div>
    </div>
  </div>

  <section>
    <h2>Factor Scores</h2>
    <table>
      <thead><tr><th>#</th><th>Factor</th><th>Score</th><th>Status</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

  <section>
    <h2>❌ Critical Gaps</h2>
    <ul class="gaps">{gaps_html}</ul>
  </section>

  <section>
    <h2>🎯 Improvement Priorities</h2>
    <ol class="priorities">{prio_html or "<li>All factors at maximum.</li>"}</ol>
  </section>
</div>
</body>
</html>"""

        Path(filepath).write_text(html, encoding="utf-8")
        print(f"HTML dashboard exported to {filepath}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    from twelve_factor_assessor import TwelveFactorAssessor

    assessor = TwelveFactorAssessor()
    dash = MaturityDashboard(assessor)

    # Simulate 6 assessments over 3 months
    configs = [
        # March 1 — prototype
        {
            "prompts_in_version_control": True,
            "prompts_semantically_versioned": False,
            "prompts_code_reviewed": False,
            "prompts_have_change_log": False,
            "prompts_independently_rollbackable": False,
            "has_conversation_state_class": False,
            "state_survives_truncation": False,
            "state_is_serializable": False,
            "state_persisted_across_sessions": False,
            "state_transitions_logged": False,
            "has_llm_provider_interface": False,
            "num_providers_supported": 1,
            "provider_configurable_via_env": False,
            "provider_specific_features_abstracted": False,
            "has_provider_fallback_chain": False,
            "has_per_request_token_budget": False,
            "has_per_request_cost_budget": False,
            "token_usage_tracked_and_logged": False,
            "cost_alerts_configured": False,
            "budget_enforcement_blocks_excess": False,
            "all_llm_outputs_schema_validated": True,
            "schema_definitions_in_version_control": True,
            "has_parse_validate_retry_pattern": False,
            "schemas_consistent_across_interactions": True,
            "schema_violations_logged_and_alerted": False,
            "has_explicit_token_allocation_per_zone": False,
            "context_consumption_measured_per_request": False,
            "has_automatic_compression_when_over_budget": False,
            "has_sliding_window_for_history": False,
            "system_prompts_audited_for_efficiency": False,
            "input_guardrail_layer_count": 0,
            "output_guardrail_layer_count": 0,
            "safety_filters_on_both_input_and_output": False,
            "has_prompt_injection_detection": False,
            "has_pii_detection_and_redaction": False,
            "has_fallback_llm_provider": False,
            "has_fallback_for_vector_db": False,
            "has_static_response_for_complete_failure": False,
            "degradation_events_logged": False,
            "degradation_regularly_chaos_tested": False,
            "has_request_tracing_with_trace_ids": False,
            "traces_exported_to_centralized_system": False,
            "key_metrics_tracked_latency_tokens_cost_errors": False,
            "has_dashboards_for_metrics": False,
            "has_alerts_for_metric_degradation": False,
            "has_approval_policy_defined": False,
            "high_stakes_actions_flagged_for_approval": False,
            "has_reviewer_interface": False,
            "has_timeout_handling_for_approvals": False,
            "approval_decisions_logged_for_audit": False,
            "test_set_has_50_plus_queries": False,
            "evaluations_run_on_every_deployment": False,
            "has_regression_detection_baseline_comparison": False,
            "safety_red_team_run_regularly": False,
            "evaluation_results_block_deployment_on_regression": False,
            "same_guardrail_config_in_dev_and_prod": False,
            "production_model_tested_before_deployment": False,
            "has_staging_environment_matching_production": False,
            "knowledge_base_structures_consistent_across_envs": False,
            "environment_differences_documented_and_intentional": False,
        },
    ]

    dates = ["2026-03-01", "2026-03-15", "2026-04-01", "2026-04-15", "2026-05-01", "2026-05-10"]
    # Progressive improvements for subsequent assessments
    improvements = [
        {},
        {"has_conversation_state_class": True, "state_is_serializable": True,
         "has_request_tracing_with_trace_ids": True, "key_metrics_tracked_latency_tokens_cost_errors": True},
        {"prompts_semantically_versioned": True, "prompts_code_reviewed": True,
         "has_llm_provider_interface": True, "num_providers_supported": 2,
         "has_per_request_token_budget": True, "state_survives_truncation": True,
         "has_explicit_token_allocation_per_zone": True},
        {"has_per_request_cost_budget": True, "token_usage_tracked_and_logged": True,
         "input_guardrail_layer_count": 3, "output_guardrail_layer_count": 3,
         "safety_filters_on_both_input_and_output": True, "has_fallback_llm_provider": True,
         "has_static_response_for_complete_failure": True, "provider_configurable_via_env": True},
        {"cost_alerts_configured": True, "budget_enforcement_blocks_excess": True,
         "traces_exported_to_centralized_system": True, "has_dashboards_for_metrics": True,
         "test_set_has_50_plus_queries": True, "has_staging_environment_matching_production": True},
        {"has_approval_policy_defined": True, "high_stakes_actions_flagged_for_approval": True,
         "evaluations_run_on_every_deployment": True, "has_regression_detection_baseline_comparison": True,
         "same_guardrail_config_in_dev_and_prod": True, "has_alerts_for_metric_degradation": True},
    ]

    history: list = []
    running_config: dict = dict(configs[0])
    for i, (dt, impr) in enumerate(zip(dates, improvements)):
        running_config.update(impr)
        r = assessor.assess(dict(running_config))
        r.assessed_at = dt
        history.append(r)

    baseline = history[0]
    latest = history[-1]

    print("\n" + "=" * 70)
    print("FULL 12-FACTOR AGENT MATURITY DASHBOARD")
    print("=" * 70)

    dash.render_full(
        latest,
        target_level="elite",
        history=history,
        baseline=baseline,
    )

    # Export HTML
    dash.export_html(latest, "/tmp/twelve_factor_dashboard.html")


if __name__ == "__main__":
    _demo()
