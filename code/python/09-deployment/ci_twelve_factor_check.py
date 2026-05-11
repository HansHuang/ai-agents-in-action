"""
CI/CD Integration Script for 12-Factor Agent Validation
=========================================================
Integrates the TwelveFactorValidator into a CI/CD pipeline to gate deployments
based on the agent's 12-Factor maturity level.

Usage::

    python ci_twelve_factor_check.py \\
        --minimum-level staging \\
        --codebase-path . \\
        --output-format github-actions \\
        --baseline-file maturity_baseline.json

Exit codes:
    0  All checks pass. Deployment can proceed.
    1  Blocking failures detected. Deployment blocked.
    2  Warnings only, no blocking failures. Proceed with caution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from twelve_factor_assessor import TwelveFactorAssessor, TwelveFactorReport
from twelve_factor_validator import TwelveFactorValidator, ValidationReport, ValidationCheck


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _github_actions_output(
    validation: ValidationReport,
    assessment: TwelveFactorReport,
    comparison: dict | None = None,
) -> str:
    """Format output for GitHub Actions annotations."""
    lines: list[str] = []
    lines.append("::group::12-Factor Agent Assessment")
    lines.append(
        f"::notice title=Maturity Level::{assessment.maturity_level} "
        f"(Level {assessment.maturity_level_number}) — Score: {assessment.overall_score}/60"
    )

    warnings = [c for c in validation.checks if not c.passed and c.rule.severity == "warning"]
    for w in warnings:
        lines.append(
            f"::warning title=Factor {w.rule.factor_number} - {w.rule.factor_name}::"
            f"{w.rule.description}: {w.evidence}"
        )

    for f in validation.blocking_failures:
        lines.append(
            f"::error title=Factor {f.rule.factor_number} - {f.rule.factor_name}::"
            f"{f.rule.description}: {f.evidence}. Fix: {f.recommendation}"
        )

    lines.append("::endgroup::")
    lines.append("")
    lines.append(f"BLOCKING FAILURES: {validation.failed_count}")
    lines.append(f"WARNINGS: {validation.warning_count}")
    lines.append(f"DEPLOYMENT: {'ALLOWED' if validation.deployment_allowed else 'BLOCKED'}")

    if not validation.deployment_allowed:
        lines.append("")
        lines.append("Fix the blocking failures before deploying.")
    elif validation.warning_count:
        lines.append("")
        lines.append("Deployment allowed. Review warnings when possible.")

    if comparison:
        delta = comparison.get("score_delta", 0)
        if delta > 0:
            lines.append(f"\n📈 Score improved by +{delta} since last assessment.")
        elif delta < 0:
            lines.append(f"\n📉 Score regressed by {delta} since last assessment.")

    return "\n".join(lines)


def _gitlab_ci_output(
    validation: ValidationReport,
    assessment: TwelveFactorReport,
    comparison: dict | None = None,
) -> str:
    """Format output for GitLab CI."""
    lines: list[str] = [
        f"section_start:{int(datetime.now(timezone.utc).timestamp())}:twelve_factor[collapsed=true]",
        "\r\033[0K12-Factor Agent Assessment",
        f"Maturity Level: {assessment.maturity_level} (Level {assessment.maturity_level_number})",
        f"Score: {assessment.overall_score}/60",
        "",
    ]
    for c in validation.checks:
        if not c.passed:
            icon = "ERROR" if c.rule.severity == "blocking" else "WARNING"
            lines.append(f"[{icon}] Factor {c.rule.factor_number}: {c.evidence}")

    lines += [
        "",
        f"Blocking failures: {validation.failed_count}",
        f"Warnings: {validation.warning_count}",
        f"Deployment: {'ALLOWED' if validation.deployment_allowed else 'BLOCKED'}",
        f"section_end:{int(datetime.now(timezone.utc).timestamp())}:twelve_factor",
    ]
    return "\n".join(lines)


def _jenkins_output(
    validation: ValidationReport,
    assessment: TwelveFactorReport,
    comparison: dict | None = None,
) -> str:
    """Format output for Jenkins (plain text with clear status markers)."""
    lines: list[str] = [
        "[12-FACTOR AGENT ASSESSMENT]",
        f"  Maturity Level: {assessment.maturity_level} (Level {assessment.maturity_level_number})",
        f"  Overall Score:  {assessment.overall_score}/60",
        "",
    ]
    if validation.blocking_failures:
        lines.append("[BLOCKING FAILURES]")
        for f in validation.blocking_failures:
            lines.append(f"  FAIL: Factor {f.rule.factor_number} — {f.rule.description}")
            lines.append(f"        {f.evidence}")
            lines.append(f"        Fix: {f.recommendation}")
        lines.append("")
    warnings = [c for c in validation.checks if not c.passed and c.rule.severity == "warning"]
    if warnings:
        lines.append("[WARNINGS]")
        for w in warnings:
            lines.append(f"  WARN: Factor {w.rule.factor_number} — {w.rule.description}")
            lines.append(f"        {w.evidence}")
        lines.append("")
    lines.append(f"[RESULT] {'DEPLOYMENT BLOCKED' if not validation.deployment_allowed else 'DEPLOYMENT ALLOWED'}")
    return "\n".join(lines)


def _plain_output(
    validation: ValidationReport,
    assessment: TwelveFactorReport,
    comparison: dict | None = None,
) -> str:
    """Format output as plain text."""
    lines: list[str] = [
        "12-Factor Agent Assessment",
        "=" * 50,
        f"Score:  {assessment.overall_score}/60",
        f"Level:  {assessment.maturity_level} (Level {assessment.maturity_level_number})",
        f"Date:   {assessment.assessed_at}",
        "",
    ]
    if validation.blocking_failures:
        lines.append("BLOCKING FAILURES:")
        for i, f in enumerate(validation.blocking_failures, 1):
            lines.append(f"  {i}. Factor {f.rule.factor_number} ({f.rule.factor_name})")
            lines.append(f"     {f.evidence}")
            lines.append(f"     Fix: {f.recommendation}")
        lines.append("")
    warnings = [c for c in validation.checks if not c.passed and c.rule.severity == "warning"]
    if warnings:
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(f"  • Factor {w.rule.factor_number}: {w.evidence}")
        lines.append("")
    lines.append(f"DEPLOYMENT: {'BLOCKED' if not validation.deployment_allowed else 'ALLOWED'}")
    lines.append(
        f"({validation.passed_count} passed, {validation.failed_count} blocking, "
        f"{validation.warning_count} warnings)"
    )
    return "\n".join(lines)


_FORMATTERS = {
    "github-actions": _github_actions_output,
    "github":         _github_actions_output,
    "gitlab":         _gitlab_ci_output,
    "gitlab-ci":      _gitlab_ci_output,
    "jenkins":        _jenkins_output,
    "plain":          _plain_output,
}


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

def _report_to_dict(report: TwelveFactorReport) -> dict[str, Any]:
    return {
        "assessed_at": report.assessed_at,
        "overall_score": report.overall_score,
        "maturity_level": report.maturity_level,
        "maturity_level_number": report.maturity_level_number,
        "factors": [
            {
                "factor_number": f.factor_number,
                "factor_name": f.factor_name,
                "score": f.score,
                "status": f.status,
            }
            for f in report.factors
        ],
        "critical_gaps": report.critical_gaps,
        "improvement_priorities": report.improvement_priorities,
    }


def _load_baseline(baseline_file: str) -> dict[str, Any] | None:
    path = Path(baseline_file)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_baseline(baseline_file: str, report: TwelveFactorReport) -> None:
    path = Path(baseline_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_report_to_dict(report), indent=2))


def _compare_to_baseline(
    current: TwelveFactorReport,
    baseline_data: dict[str, Any],
) -> dict[str, Any]:
    base_scores = {f["factor_number"]: f["score"] for f in baseline_data.get("factors", [])}
    curr_scores = {f.factor_number: f.score for f in current.factors}
    improved = [(n, curr_scores[n] - base_scores[n]) for n in curr_scores if curr_scores[n] > base_scores.get(n, 1)]
    regressed = [(n, curr_scores[n] - base_scores[n]) for n in curr_scores if curr_scores[n] < base_scores.get(n, 5)]
    return {
        "baseline_score": baseline_data.get("overall_score", 0),
        "current_score": current.overall_score,
        "score_delta": current.overall_score - baseline_data.get("overall_score", 0),
        "baseline_level": baseline_data.get("maturity_level", "Unknown"),
        "current_level": current.maturity_level,
        "improved_factors": improved,
        "regressed_factors": regressed,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _generate_ci_markdown(
    validation: ValidationReport,
    assessment: TwelveFactorReport,
    comparison: dict | None,
    minimum_level: str,
) -> str:
    from twelve_factor_assessor import generate_markdown_report, _ROMAN, _STATUS_EMOJI
    base_md = generate_markdown_report(assessment)
    sections: list[str] = [
        f"<!-- Generated by ci_twelve_factor_check.py -->",
        base_md,
        "",
        "## CI/CD Validation Summary",
        "",
        f"- **Minimum level required:** {minimum_level.capitalize()}",
        f"- **Total checks run:** {validation.total_checks}",
        f"- **Passed:** {validation.passed_count}",
        f"- **Blocking failures:** {validation.failed_count}",
        f"- **Warnings:** {validation.warning_count}",
        f"- **Deployment:** {'✅ ALLOWED' if validation.deployment_allowed else '❌ BLOCKED'}",
    ]
    if comparison:
        delta = comparison["score_delta"]
        sections += [
            "",
            "## Comparison to Baseline",
            "",
            f"- Previous score: {comparison['baseline_score']}/60 ({comparison['baseline_level']})",
            f"- Current score:  {comparison['current_score']}/60 ({comparison['current_level']})",
            f"- Delta: {delta:+d}",
        ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="12-Factor Agent CI/CD validation check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--minimum-level",
        choices=["development", "staging", "production", "elite"],
        default="staging",
        help="Minimum maturity level required for deployment (default: staging)",
    )
    parser.add_argument(
        "--codebase-path",
        default=".",
        help="Path to the codebase to validate (default: current directory)",
    )
    parser.add_argument(
        "--output-format",
        choices=list(_FORMATTERS.keys()),
        default="plain",
        help="Output format for CI annotation messages (default: plain)",
    )
    parser.add_argument(
        "--baseline-file",
        default="maturity_baseline.json",
        help="JSON file to store/compare assessment baseline (default: maturity_baseline.json)",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Write a full markdown report to this file (optional)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save the current assessment as the new baseline after a successful run",
    )
    parser.add_argument(
        "--ci-mode",
        choices=["github", "gitlab", "jenkins", "plain"],
        default=None,
        help="Alias for --output-format (overrides if set)",
    )
    args = parser.parse_args(argv)

    output_format = args.ci_mode or args.output_format
    codebase_path = Path(args.codebase_path).resolve()

    if not codebase_path.exists():
        print(f"ERROR: codebase-path does not exist: {codebase_path}", file=sys.stderr)
        return 1

    # --- Run validation ---
    validator = TwelveFactorValidator(minimum_level=args.minimum_level)
    validation = validator.validate({}, codebase_path=codebase_path)

    # --- Run assessment ---
    assessor = TwelveFactorAssessor()
    # Build a config dict from validation results for the assessor
    cfg: dict[str, Any] = {}
    for check in validation.checks:
        # Map rule names to assessor config keys (best-effort)
        rule_to_config = {
            "prompts_dir_exists": "prompts_in_version_control",
            "prompt_versions": "prompts_semantically_versioned",
            "state_class_exists": "has_conversation_state_class",
            "state_serializable": "state_is_serializable",
            "state_injected": "state_survives_truncation",
            "provider_interface": "has_llm_provider_interface",
            "token_counting": "has_per_request_token_budget",
            "budget_check_before_llm": "budget_enforcement_blocks_excess",
            "token_usage_logged": "token_usage_tracked_and_logged",
            "pydantic_schemas": "all_llm_outputs_schema_validated",
            "schema_validation_after_llm": "schemas_consistent_across_interactions",
            "parse_validate_retry": "has_parse_validate_retry_pattern",
            "context_budget_code": "has_explicit_token_allocation_per_zone",
            "context_compression": "has_automatic_compression_when_over_budget",
            "three_input_guardrails": None,   # handled separately
            "three_output_guardrails": None,
            "fallback_provider": "has_fallback_llm_provider",
            "static_fallback": "has_static_response_for_complete_failure",
            "trace_id": "has_request_tracing_with_trace_ids",
            "metrics_collection": "key_metrics_tracked_latency_tokens_cost_errors",
            "approval_policy": "has_approval_policy_defined",
            "high_stakes_flagged": "high_stakes_actions_flagged_for_approval",
            "test_set_50": "test_set_has_50_plus_queries",
            "eval_in_ci": "evaluations_run_on_every_deployment",
            "regression_detection": "has_regression_detection_baseline_comparison",
            "guardrail_parity": "same_guardrail_config_in_dev_and_prod",
            "staging_exists": "has_staging_environment_matching_production",
        }
        mapped = rule_to_config.get(check.rule.rule_name)
        if mapped:
            cfg[mapped] = check.passed
    # Guardrail counts
    input_check = next((c for c in validation.checks if c.rule.rule_name == "three_input_guardrails"), None)
    output_check = next((c for c in validation.checks if c.rule.rule_name == "three_output_guardrails"), None)
    cfg["input_guardrail_layer_count"] = 3 if (input_check and input_check.passed) else 1
    cfg["output_guardrail_layer_count"] = 3 if (output_check and output_check.passed) else 1

    assessment = assessor.assess(cfg)

    # --- Compare to baseline ---
    baseline_data = _load_baseline(args.baseline_file)
    comparison = _compare_to_baseline(assessment, baseline_data) if baseline_data else None

    # --- Print CI output ---
    formatter = _FORMATTERS.get(output_format, _plain_output)
    print(formatter(validation, assessment, comparison))

    # --- Write markdown report ---
    if args.report_file:
        md = _generate_ci_markdown(validation, assessment, comparison, args.minimum_level)
        Path(args.report_file).write_text(md)
        print(f"\nMarkdown report written to {args.report_file}")

    # --- Save baseline ---
    if args.save_baseline:
        _save_baseline(args.baseline_file, assessment)
        print(f"Baseline saved to {args.baseline_file}")

    # --- Return exit code ---
    if not validation.deployment_allowed:
        return 1
    if validation.warning_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
