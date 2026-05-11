"""
12-Factor Agent Self-Assessment Tool
=====================================
Evaluates an agent configuration against all 12 production-readiness factors
and generates actionable reports, roadmaps, and improvement tracking.

Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FactorAssessment:
    factor_number: int
    factor_name: str
    score: int  # 1-5
    status: str  # "critical" | "needs_improvement" | "good" | "excellent"
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 1 <= self.score <= 5:
            raise ValueError(f"Score must be 1-5, got {self.score}")
        if self.status not in {"critical", "needs_improvement", "good", "excellent"}:
            raise ValueError(f"Invalid status: {self.status}")


@dataclass
class TwelveFactorReport:
    overall_score: int  # 12-60
    maturity_level: str  # "Prototype" | "Development" | "Staging" | "Production" | "Elite"
    maturity_level_number: int  # 1-5
    factors: list[FactorAssessment] = field(default_factory=list)
    critical_gaps: list[str] = field(default_factory=list)
    improvement_priorities: list[str] = field(default_factory=list)
    assessed_at: str = field(default_factory=lambda: date.today().isoformat())


@dataclass
class ComparisonReport:
    baseline_score: int
    current_score: int
    score_delta: int
    baseline_level: str
    current_level: str
    improved_factors: list[tuple[int, str, int, int]] = field(default_factory=list)   # (num, name, old, new)
    regressed_factors: list[tuple[int, str, int, int]] = field(default_factory=list)  # (num, name, old, new)
    unchanged_factors: list[tuple[int, str, int]] = field(default_factory=list)       # (num, name, score)


@dataclass
class TrendReport:
    assessments: list[dict]  # list of {date, score, level}
    first_score: int
    latest_score: int
    total_improvement: int
    average_improvement_per_assessment: float


# ---------------------------------------------------------------------------
# Factor names
# ---------------------------------------------------------------------------

FACTOR_NAMES: dict[int, str] = {
    1:  "Prompt as Code",
    2:  "Explicit State",
    3:  "Provider Agnostic",
    4:  "Token Budgeting",
    5:  "Structured Everything",
    6:  "Context Is a Resource",
    7:  "Defense in Depth",
    8:  "Graceful Degradation",
    9:  "Observability First",
    10: "Human in the Loop",
    11: "Continuous Evaluation",
    12: "Dev-Prod Parity",
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_to_status(score: int) -> str:
    if score <= 2:
        return "critical"
    if score == 3:
        return "needs_improvement"
    if score == 4:
        return "good"
    return "excellent"


def _checks_to_score(passed: int, total: int = 5) -> int:
    """Map number of passed checks (0-5) to a 1-5 score."""
    if passed == 0:
        return 1
    return min(5, max(1, passed))


# ---------------------------------------------------------------------------
# Individual factor assessors
# ---------------------------------------------------------------------------

def _assess_factor_1(cfg: dict) -> FactorAssessment:
    """Factor I — Prompt as Code"""
    checks = {
        "prompts_in_vcs": cfg.get("prompts_in_version_control", False),
        "semantic_versioning": cfg.get("prompts_semantically_versioned", False),
        "code_reviewed": cfg.get("prompts_code_reviewed", False),
        "change_log": cfg.get("prompts_have_change_log", False),
        "independent_rollback": cfg.get("prompts_independently_rollbackable", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["prompts_in_vcs"]:
        evidence.append("Prompts stored in version control (Git)")
    else:
        gaps.append("Prompts not tracked in version control")
        recs.append("Move prompts to a prompts/ directory under version control")

    if checks["semantic_versioning"]:
        evidence.append("Prompts use semantic versioning (e.g. v2.3.1)")
    else:
        gaps.append("Prompts lack semantic versioning")
        recs.append("Add a version field to each prompt file (YAML/Markdown frontmatter)")

    if checks["code_reviewed"]:
        evidence.append("Prompt changes go through code review")
    else:
        gaps.append("Prompt changes bypass code review")
        recs.append("Add prompts/ to required CODEOWNERS for PR review")

    if checks["change_log"]:
        evidence.append("Prompts include a change_log field")
    else:
        gaps.append("No prompt change log")
        recs.append("Add change_log field to every prompt YAML")

    if checks["independent_rollback"]:
        evidence.append("Prompts can be rolled back independently of application code")
    else:
        gaps.append("Prompt rollback requires full application rollback")
        recs.append("Decouple prompt deployment from application deployment")

    return FactorAssessment(
        factor_number=1,
        factor_name=FACTOR_NAMES[1],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_2(cfg: dict) -> FactorAssessment:
    """Factor II — Explicit State"""
    checks = {
        "state_class_exists": cfg.get("has_conversation_state_class", False),
        "survives_truncation": cfg.get("state_survives_truncation", False),
        "serializable": cfg.get("state_is_serializable", False),
        "persisted_across_sessions": cfg.get("state_persisted_across_sessions", False),
        "transitions_logged": cfg.get("state_transitions_logged", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["state_class_exists"]:
        evidence.append("ConversationState class/dataclass exists")
    else:
        gaps.append("No explicit ConversationState class found")
        recs.append("Create a ConversationState dataclass with all session fields")

    if checks["survives_truncation"]:
        evidence.append("State is injected into every LLM call, surviving message truncation")
    else:
        gaps.append("State may be lost when message list is truncated")
        recs.append("Inject a state summary into every LLM call prompt")

    if checks["serializable"]:
        evidence.append("State is JSON-serializable (to_dict/model_dump exists)")
    else:
        gaps.append("State cannot be serialized for debugging or persistence")
        recs.append("Implement to_dict() / model_dump() on ConversationState")

    if checks["persisted_across_sessions"]:
        evidence.append("State is persisted to a durable store (DB/Redis)")
    else:
        gaps.append("State is lost when session ends")
        recs.append("Persist ConversationState to Redis or a database by session ID")

    if checks["transitions_logged"]:
        evidence.append("State transitions are logged for observability")
    else:
        gaps.append("State changes are not logged")
        recs.append("Log every state field change with timestamp and reason")

    return FactorAssessment(
        factor_number=2,
        factor_name=FACTOR_NAMES[2],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_3(cfg: dict) -> FactorAssessment:
    """Factor III — Provider Agnostic"""
    checks = {
        "provider_abstraction": cfg.get("has_llm_provider_interface", False),
        "two_providers": cfg.get("num_providers_supported", 0) >= 2,
        "config_driven": cfg.get("provider_configurable_via_env", False),
        "features_abstracted": cfg.get("provider_specific_features_abstracted", False),
        "fallback_chain": cfg.get("has_provider_fallback_chain", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["provider_abstraction"]:
        evidence.append("LLM provider interface/abstract class exists")
    else:
        gaps.append("LLM calls are tightly coupled to a specific provider")
        recs.append("Create an LLMProvider abstract class with a chat() method")

    num = cfg.get("num_providers_supported", 0)
    if checks["two_providers"]:
        evidence.append(f"{num} provider(s) supported")
    else:
        gaps.append(f"Only {num} provider(s) supported — need at least 2")
        recs.append("Implement at least one fallback provider (e.g., OpenAI + Anthropic)")

    if checks["config_driven"]:
        evidence.append("Provider and model configured via environment variables")
    else:
        gaps.append("Provider is hardcoded in source code")
        recs.append("Read LLM_PROVIDER and LLM_MODEL from environment variables")

    if checks["features_abstracted"]:
        evidence.append("Provider-specific features abstracted behind interface")
    else:
        gaps.append("Provider-specific API details leak into application code")
        recs.append("Wrap provider-specific features behind a common interface method")

    if checks["fallback_chain"]:
        evidence.append("Provider fallback chain configured")
    else:
        gaps.append("No fallback chain — single provider failure causes outage")
        recs.append("Implement provider fallback chain with circuit breaker")

    return FactorAssessment(
        factor_number=3,
        factor_name=FACTOR_NAMES[3],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_4(cfg: dict) -> FactorAssessment:
    """Factor IV — Token Budgeting"""
    checks = {
        "per_request_token_budget": cfg.get("has_per_request_token_budget", False),
        "per_request_cost_budget": cfg.get("has_per_request_cost_budget", False),
        "usage_tracked": cfg.get("token_usage_tracked_and_logged", False),
        "cost_alerts": cfg.get("cost_alerts_configured", False),
        "budget_enforced": cfg.get("budget_enforcement_blocks_excess", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["per_request_token_budget"]:
        evidence.append("Per-request token budget is defined")
    else:
        gaps.append("No per-request token budget")
        recs.append("Define a TokenBudget with max_tokens per request")

    if checks["per_request_cost_budget"]:
        evidence.append("Per-request cost budget is defined")
    else:
        gaps.append("No per-request cost budget")
        recs.append("Add max_cost (USD) to TokenBudget alongside max_tokens")

    if checks["usage_tracked"]:
        evidence.append("Token counts are tracked and logged per request")
    else:
        gaps.append("Token usage not tracked — no visibility into costs")
        recs.append("Log prompt_tokens, completion_tokens, and total_cost per request")

    if checks["cost_alerts"]:
        evidence.append("Cost alerts configured for budget overruns")
    else:
        gaps.append("No cost alerts — cost spikes go undetected")
        recs.append("Alert when request cost exceeds 2x baseline average")

    if checks["budget_enforced"]:
        evidence.append("Requests exceeding budget are blocked before LLM call")
    else:
        gaps.append("Budget is tracked but not enforced")
        recs.append("Return early with a user-friendly message when budget would be exceeded")

    return FactorAssessment(
        factor_number=4,
        factor_name=FACTOR_NAMES[4],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_5(cfg: dict) -> FactorAssessment:
    """Factor V — Structured Everything"""
    checks = {
        "all_outputs_schema_validated": cfg.get("all_llm_outputs_schema_validated", False),
        "schemas_in_vcs": cfg.get("schema_definitions_in_version_control", False),
        "parse_validate_retry": cfg.get("has_parse_validate_retry_pattern", False),
        "consistent_schemas": cfg.get("schemas_consistent_across_interactions", False),
        "violations_logged": cfg.get("schema_violations_logged_and_alerted", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["all_outputs_schema_validated"]:
        evidence.append("All LLM outputs are validated against a schema")
    else:
        gaps.append("Some or all LLM outputs are parsed as raw text")
        recs.append("Use Pydantic models with model_validate_json() for all LLM calls")

    if checks["schemas_in_vcs"]:
        evidence.append("Schema definitions are version-controlled")
    else:
        gaps.append("Schemas are defined inline or not tracked")
        recs.append("Store Pydantic schemas in a schemas/ module under version control")

    if checks["parse_validate_retry"]:
        evidence.append("Parse-validate-retry pattern implemented for schema failures")
    else:
        gaps.append("Schema failures raise exceptions without retry")
        recs.append("Add retry logic: on ValidationError, re-prompt LLM with schema error context")

    if checks["consistent_schemas"]:
        evidence.append("Schemas are consistent across all LLM interactions")
    else:
        gaps.append("Schema definitions are inconsistent or duplicated")
        recs.append("Centralise schema definitions and import from a single module")

    if checks["violations_logged"]:
        evidence.append("Schema violations are logged with full context")
    else:
        gaps.append("Schema violations are silently swallowed")
        recs.append("Log every schema validation failure with model output and request context")

    return FactorAssessment(
        factor_number=5,
        factor_name=FACTOR_NAMES[5],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_6(cfg: dict) -> FactorAssessment:
    """Factor VI — Context Is a Resource"""
    checks = {
        "explicit_allocation": cfg.get("has_explicit_token_allocation_per_zone", False),
        "consumption_measured": cfg.get("context_consumption_measured_per_request", False),
        "auto_compression": cfg.get("has_automatic_compression_when_over_budget", False),
        "sliding_window": cfg.get("has_sliding_window_for_history", False),
        "prompt_audited": cfg.get("system_prompts_audited_for_efficiency", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["explicit_allocation"]:
        evidence.append("Token allocation defined per context zone (system, history, tools, RAG, buffer)")
    else:
        gaps.append("No explicit context zone allocation — context fills up unpredictably")
        recs.append("Define percentage allocations: system 2%, tools 5%, history 33%, dynamic 45%, buffer 15%")

    if checks["consumption_measured"]:
        evidence.append("Token consumption is measured and logged per request per zone")
    else:
        gaps.append("Context consumption not measured")
        recs.append("Count tokens per zone on each request and emit as metrics")

    if checks["auto_compression"]:
        evidence.append("Automatic context compression triggers when over budget")
    else:
        gaps.append("No compression — context window overflows silently")
        recs.append("Summarise conversation history when it exceeds its allocation")

    if checks["sliding_window"]:
        evidence.append("Sliding window keeps conversation history within budget")
    else:
        gaps.append("All conversation history included, consuming unbounded tokens")
        recs.append("Implement a sliding window: summarise old turns, keep last N verbatim")

    if checks["prompt_audited"]:
        evidence.append("System prompts have been audited for token efficiency")
    else:
        gaps.append("System prompts have never been audited for wasted tokens")
        recs.append("Run a quarterly token efficiency audit on system prompts")

    return FactorAssessment(
        factor_number=6,
        factor_name=FACTOR_NAMES[6],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_7(cfg: dict) -> FactorAssessment:
    """Factor VII — Defense in Depth"""
    input_layers = cfg.get("input_guardrail_layer_count", 0)
    output_layers = cfg.get("output_guardrail_layer_count", 0)
    checks = {
        "input_guardrails": input_layers >= 3,
        "output_guardrails": output_layers >= 3,
        "safety_filters_both": cfg.get("safety_filters_on_both_input_and_output", False),
        "injection_detection": cfg.get("has_prompt_injection_detection", False),
        "pii_detection": cfg.get("has_pii_detection_and_redaction", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["input_guardrails"]:
        evidence.append(f"{input_layers} independent input guardrail layers")
    else:
        gaps.append(f"Only {input_layers} input guardrail layer(s) — need at least 3")
        recs.append("Add: rate limiter → structural validator → PII detector → content filter → injection detector")

    if checks["output_guardrails"]:
        evidence.append(f"{output_layers} independent output guardrail layers")
    else:
        gaps.append(f"Only {output_layers} output guardrail layer(s) — need at least 3")
        recs.append("Add: schema validator → PII detector → safety filter → leakage detector → hallucination check")

    if checks["safety_filters_both"]:
        evidence.append("Safety filters applied to both input and output")
    else:
        gaps.append("Safety filter not applied symmetrically to input and output")
        recs.append("Ensure every safety check applies to both the user's message and the model's response")

    if checks["injection_detection"]:
        evidence.append("Prompt injection detection in place")
    else:
        gaps.append("No prompt injection detection")
        recs.append("Add an injection detector that scans for role-override attempts in user input")

    if checks["pii_detection"]:
        evidence.append("PII detection and redaction applied")
    else:
        gaps.append("PII may flow into LLM context or be exposed in responses")
        recs.append("Add PII detection (names, emails, phone, card numbers) on input and output")

    return FactorAssessment(
        factor_number=7,
        factor_name=FACTOR_NAMES[7],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_8(cfg: dict) -> FactorAssessment:
    """Factor VIII — Graceful Degradation"""
    checks = {
        "llm_fallback_provider": cfg.get("has_fallback_llm_provider", False),
        "vector_db_fallback": cfg.get("has_fallback_for_vector_db", False),
        "static_response_fallback": cfg.get("has_static_response_for_complete_failure", False),
        "degradation_logged": cfg.get("degradation_events_logged", False),
        "chaos_tested": cfg.get("degradation_regularly_chaos_tested", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["llm_fallback_provider"]:
        evidence.append("Fallback LLM provider configured for primary provider failures")
    else:
        gaps.append("No LLM fallback — primary provider outage causes complete failure")
        recs.append("Add a fallback provider chain: primary → secondary → cheaper model → static")

    if checks["vector_db_fallback"]:
        evidence.append("Vector database fallback implemented")
    else:
        gaps.append("No fallback when vector database is unavailable")
        recs.append("Continue without RAG when vector DB is unavailable, note reduced confidence")

    if checks["static_response_fallback"]:
        evidence.append("Static fallback response for complete system failure")
    else:
        gaps.append("Complete system failure exposes raw errors to users")
        recs.append("Add a final catch-all that returns a polite error message with support contact")

    if checks["degradation_logged"]:
        evidence.append("Degradation events are logged for postmortem analysis")
    else:
        gaps.append("Degradation events are silent — hard to diagnose production issues")
        recs.append("Log every degradation event with level, reason, and fallback taken")

    if checks["chaos_tested"]:
        evidence.append("Degradation paths are regularly tested via chaos engineering")
    else:
        gaps.append("Fallback paths untested — may not work when needed")
        recs.append("Run monthly chaos tests: kill primary LLM, kill vector DB, kill all external deps")

    return FactorAssessment(
        factor_number=8,
        factor_name=FACTOR_NAMES[8],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_9(cfg: dict) -> FactorAssessment:
    """Factor IX — Observability First"""
    checks = {
        "request_tracing": cfg.get("has_request_tracing_with_trace_ids", False),
        "centralized_export": cfg.get("traces_exported_to_centralized_system", False),
        "key_metrics_tracked": cfg.get("key_metrics_tracked_latency_tokens_cost_errors", False),
        "dashboards": cfg.get("has_dashboards_for_metrics", False),
        "metric_alerts": cfg.get("has_alerts_for_metric_degradation", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["request_tracing"]:
        evidence.append("Unique trace IDs generated for every request")
    else:
        gaps.append("No request tracing — impossible to reconstruct what happened for a given request")
        recs.append("Generate a UUID trace_id for every request and attach to all log lines")

    if checks["centralized_export"]:
        evidence.append("Traces exported to a centralized system (LangSmith, Arize, OTEL)")
    else:
        gaps.append("Traces only in local logs — no centralized visibility")
        recs.append("Export traces via OpenTelemetry to a centralized backend (e.g., LangSmith)")

    if checks["key_metrics_tracked"]:
        evidence.append("Latency, token usage, cost, and error rate tracked per request")
    else:
        gaps.append("Key metrics (latency, tokens, cost, errors) not consistently tracked")
        recs.append("Emit structured metrics for every request: p50/p95 latency, tokens, cost, error_rate")

    if checks["dashboards"]:
        evidence.append("Operational dashboards show real-time metrics")
    else:
        gaps.append("No dashboards — metrics not visible to the team")
        recs.append("Create a Grafana or DataDog dashboard with the 5 key agent metrics")

    if checks["metric_alerts"]:
        evidence.append("Alerts fire when metrics degrade (e.g., error rate >1%, latency >5s)")
    else:
        gaps.append("No metric alerts — degradation discovered by users, not ops")
        recs.append("Set up alerts: error rate >1%, p95 latency >5s, cost >2x baseline")

    return FactorAssessment(
        factor_number=9,
        factor_name=FACTOR_NAMES[9],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_10(cfg: dict) -> FactorAssessment:
    """Factor X — Human in the Loop"""
    checks = {
        "approval_policy": cfg.get("has_approval_policy_defined", False),
        "high_stakes_flagged": cfg.get("high_stakes_actions_flagged_for_approval", False),
        "reviewer_interface": cfg.get("has_reviewer_interface", False),
        "timeout_handling": cfg.get("has_timeout_handling_for_approvals", False),
        "decisions_logged": cfg.get("approval_decisions_logged_for_audit", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["approval_policy"]:
        evidence.append("Explicit approval policy defines which actions require human review")
    else:
        gaps.append("No approval policy — every action or no action gets reviewed")
        recs.append("Define an ApprovalPolicy with action types, thresholds, and risk levels")

    if checks["high_stakes_flagged"]:
        evidence.append("High-stakes actions (financial, irreversible, external) flagged for approval")
    else:
        gaps.append("High-stakes actions execute without human oversight")
        recs.append("Categorise actions by risk: auto-execute LOW, review MEDIUM/HIGH/CRITICAL")

    if checks["reviewer_interface"]:
        evidence.append("Reviewer interface exists for approving/rejecting/editing proposed actions")
    else:
        gaps.append("No reviewer interface — approval is not actionable")
        recs.append("Build a Slack/email/web UI for reviewers with approve/reject/edit options")

    if checks["timeout_handling"]:
        evidence.append("Approval requests time out safely (default to rejection)")
    else:
        gaps.append("Approval requests can hang indefinitely")
        recs.append("Add timeout: default to rejection after N seconds, notify user of delay")

    if checks["decisions_logged"]:
        evidence.append("All approval decisions logged with reviewer, reason, and timestamp")
    else:
        gaps.append("No audit trail for human approval decisions")
        recs.append("Log every approval decision to a tamper-evident audit log")

    return FactorAssessment(
        factor_number=10,
        factor_name=FACTOR_NAMES[10],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_11(cfg: dict) -> FactorAssessment:
    """Factor XI — Continuous Evaluation"""
    checks = {
        "test_set_50_plus": cfg.get("test_set_has_50_plus_queries", False),
        "eval_on_every_deploy": cfg.get("evaluations_run_on_every_deployment", False),
        "regression_detection": cfg.get("has_regression_detection_baseline_comparison", False),
        "safety_red_team": cfg.get("safety_red_team_run_regularly", False),
        "blocks_on_regression": cfg.get("evaluation_results_block_deployment_on_regression", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["test_set_50_plus"]:
        evidence.append("Test set of 50+ queries covering all intent categories")
    else:
        gaps.append("No evaluation test set (or fewer than 50 queries)")
        recs.append("Build a test set of 50+ queries: mix of happy path, edge cases, and adversarial")

    if checks["eval_on_every_deploy"]:
        evidence.append("Full evaluation runs on every deployment in CI/CD")
    else:
        gaps.append("Evaluations are manual and infrequent")
        recs.append("Add evaluation step to CI/CD pipeline — block merge if eval fails to run")

    if checks["regression_detection"]:
        evidence.append("Regression detection compares each run to a stored baseline")
    else:
        gaps.append("No regression detection — improvements in one area can silently break another")
        recs.append("Store evaluation baseline and alert when any metric drops >5% from baseline")

    if checks["safety_red_team"]:
        evidence.append("Safety red-team evaluations run on every significant change")
    else:
        gaps.append("No safety red-team evaluations")
        recs.append("Run adversarial safety prompts on every significant change or weekly")

    if checks["blocks_on_regression"]:
        evidence.append("Deployment is blocked when evaluation detects a significant regression")
    else:
        gaps.append("Regressions are reported but don't block deployment")
        recs.append("Make evaluation results a deployment gate: fail CI on critical regressions")

    return FactorAssessment(
        factor_number=11,
        factor_name=FACTOR_NAMES[11],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


def _assess_factor_12(cfg: dict) -> FactorAssessment:
    """Factor XII — Dev-Prod Parity"""
    checks = {
        "same_guardrails": cfg.get("same_guardrail_config_in_dev_and_prod", False),
        "prod_model_tested": cfg.get("production_model_tested_before_deployment", False),
        "staging_mirrors_prod": cfg.get("has_staging_environment_matching_production", False),
        "kb_structures_consistent": cfg.get("knowledge_base_structures_consistent_across_envs", False),
        "differences_documented": cfg.get("environment_differences_documented_and_intentional", False),
    }
    passed = sum(checks.values())
    score = _checks_to_score(passed)

    evidence, gaps, recs = [], [], []
    if checks["same_guardrails"]:
        evidence.append("Guardrail configuration is identical in dev and production")
    else:
        gaps.append("Guardrails differ between dev and prod — safety issues only discovered in production")
        recs.append("Use a single guardrail config file loaded in all environments")

    if checks["prod_model_tested"]:
        evidence.append("Production model is tested before every deployment")
    else:
        gaps.append("Dev uses a cheaper model — production model behaviour not verified until deploy")
        recs.append("Add a pre-deployment test step that runs the evaluation suite with the prod model")

    if checks["staging_mirrors_prod"]:
        evidence.append("Staging environment mirrors production configuration")
    else:
        gaps.append("No staging environment — changes go straight from dev to prod")
        recs.append("Maintain a staging environment with the same config, secrets rotation, and data shape as prod")

    if checks["kb_structures_consistent"]:
        evidence.append("Knowledge base schemas are consistent across environments")
    else:
        gaps.append("Knowledge base structure differs across environments — RAG behaviour unpredictable in prod")
        recs.append("Keep knowledge base schemas (index config, metadata fields) identical across all environments")

    if checks["differences_documented"]:
        evidence.append("Intentional environment differences are documented")
    else:
        gaps.append("Environment differences are undocumented and may be accidental")
        recs.append("Document every deliberate difference (e.g., model tier, KB size) in an environment parity file")

    return FactorAssessment(
        factor_number=12,
        factor_name=FACTOR_NAMES[12],
        score=score,
        status=_score_to_status(score),
        evidence=evidence,
        gaps=gaps,
        recommendations=recs,
    )


_FACTOR_ASSESSORS = {
    1:  _assess_factor_1,
    2:  _assess_factor_2,
    3:  _assess_factor_3,
    4:  _assess_factor_4,
    5:  _assess_factor_5,
    6:  _assess_factor_6,
    7:  _assess_factor_7,
    8:  _assess_factor_8,
    9:  _assess_factor_9,
    10: _assess_factor_10,
    11: _assess_factor_11,
    12: _assess_factor_12,
}


# ---------------------------------------------------------------------------
# Maturity level calculation
# ---------------------------------------------------------------------------

def _compute_maturity(factors: list[FactorAssessment], total_score: int) -> tuple[str, int]:
    """Return (maturity_name, level_number) based on score and factor requirements."""
    scores = {f.factor_number: f.score for f in factors}

    def _all_at_least(factor_nums: list[int], minimum: int) -> bool:
        return all(scores.get(n, 1) >= minimum for n in factor_nums)

    # Level 5: Elite — all 12 at ≥4 AND total == 60
    if total_score == 60 and _all_at_least(list(range(1, 13)), 4):
        return "Elite", 5

    # Level 4: Production — score 49-60, factors 1-11 all ≥3
    if total_score >= 49 and _all_at_least(list(range(1, 12)), 3):
        return "Production", 4

    # Level 3: Staging — score 37-48, factors 1-6,8,9 all ≥3
    if total_score >= 37 and _all_at_least([1, 2, 3, 4, 5, 6, 8, 9], 3):
        return "Staging", 3

    # Level 2: Development — score 25-36, factors 1,2,5,9 all ≥3
    if total_score >= 25 and _all_at_least([1, 2, 5, 9], 3):
        return "Development", 2

    # Level 1: Prototype
    return "Prototype", 1


# ---------------------------------------------------------------------------
# Main assessor class
# ---------------------------------------------------------------------------

class TwelveFactorAssessor:
    """
    Evaluate an agent configuration against all 12 production-readiness factors.

    Usage::

        assessor = TwelveFactorAssessor()
        report = assessor.assess(agent_config)
        print(assessor.export_report(report))
    """

    def assess(self, agent_or_config: dict[str, Any]) -> TwelveFactorReport:
        """Evaluate an agent against all 12 factors and return a full report."""
        factors = [
            _FACTOR_ASSESSORS[n](agent_or_config)
            for n in range(1, 13)
        ]
        total = sum(f.score for f in factors)
        level_name, level_num = _compute_maturity(factors, total)

        critical_gaps = [
            f"Factor {f.factor_number} ({f.factor_name}) scored {f.score}/5 — {f.gaps[0] if f.gaps else 'see report'}"
            for f in factors if f.score <= 2
        ]

        # Priorities: factors with lowest scores, ordered by potential gain
        sorted_by_score = sorted(factors, key=lambda f: (f.score, f.factor_number))
        priorities = [
            f"Improve Factor {f.factor_number} ({f.factor_name}): {f.recommendations[0]}"
            for f in sorted_by_score
            if f.recommendations and f.score < 5
        ][:7]  # top 7 priorities

        return TwelveFactorReport(
            overall_score=total,
            maturity_level=level_name,
            maturity_level_number=level_num,
            factors=factors,
            critical_gaps=critical_gaps,
            improvement_priorities=priorities,
        )

    def assess_factor(self, factor_number: int, agent_or_config: dict[str, Any]) -> FactorAssessment:
        """Evaluate a single factor."""
        if factor_number not in _FACTOR_ASSESSORS:
            raise ValueError(f"Factor number must be 1-12, got {factor_number}")
        return _FACTOR_ASSESSORS[factor_number](agent_or_config)

    def generate_recommendations(self, report: TwelveFactorReport) -> list[str]:
        """Return a flat, ordered list of actionable recommendations."""
        recs: list[str] = []
        # Critical first
        for f in report.factors:
            if f.score <= 2:
                for r in f.recommendations:
                    recs.append(f"[CRITICAL] Factor {f.factor_number} — {r}")
        # Then needs-improvement
        for f in report.factors:
            if f.score == 3:
                for r in f.recommendations:
                    recs.append(f"[IMPROVE] Factor {f.factor_number} — {r}")
        # Then good (minor polish)
        for f in report.factors:
            if f.score == 4:
                for r in f.recommendations:
                    recs.append(f"[POLISH] Factor {f.factor_number} — {r}")
        return recs

    def compare_to_baseline(
        self,
        current: TwelveFactorReport,
        baseline: TwelveFactorReport,
    ) -> ComparisonReport:
        """Compare a current report to a previous baseline."""
        base_scores = {f.factor_number: f.score for f in baseline.factors}
        curr_scores = {f.factor_number: f.score for f in current.factors}

        improved, regressed, unchanged = [], [], []
        for num in range(1, 13):
            name = FACTOR_NAMES[num]
            old = base_scores.get(num, 1)
            new = curr_scores.get(num, 1)
            if new > old:
                improved.append((num, name, old, new))
            elif new < old:
                regressed.append((num, name, old, new))
            else:
                unchanged.append((num, name, old))

        return ComparisonReport(
            baseline_score=baseline.overall_score,
            current_score=current.overall_score,
            score_delta=current.overall_score - baseline.overall_score,
            baseline_level=baseline.maturity_level,
            current_level=current.maturity_level,
            improved_factors=improved,
            regressed_factors=regressed,
            unchanged_factors=unchanged,
        )

    def export_report(self, report: TwelveFactorReport, fmt: str = "markdown") -> str:
        """Export report in 'markdown' or 'html' format."""
        if fmt == "markdown":
            return generate_markdown_report(report)
        if fmt == "html":
            return generate_html_report(report)
        raise ValueError(f"Unknown format: {fmt!r}. Use 'markdown' or 'html'.")


# ---------------------------------------------------------------------------
# Improvement tracker
# ---------------------------------------------------------------------------

class ImprovementTracker:
    """Track maturity assessments over time and generate improvement roadmaps."""

    def __init__(self) -> None:
        self._history: list[TwelveFactorReport] = []
        self._assessor = TwelveFactorAssessor()

    def save_baseline(self, report: TwelveFactorReport) -> None:
        """Store a report as the current baseline (appended to history)."""
        self._history.append(report)

    def compare_to_baseline(self, current: TwelveFactorReport) -> ComparisonReport | None:
        """Compare to the most recent saved baseline, or None if no baseline exists."""
        if not self._history:
            return None
        return self._assessor.compare_to_baseline(current, self._history[-1])

    def track_over_time(self) -> TrendReport:
        """Summarise score trends across all saved assessments."""
        if not self._history:
            raise ValueError("No assessments recorded yet")
        data = [
            {"date": r.assessed_at, "score": r.overall_score, "level": r.maturity_level}
            for r in self._history
        ]
        first = self._history[0].overall_score
        latest = self._history[-1].overall_score
        delta = latest - first
        avg = delta / max(len(self._history) - 1, 1)
        return TrendReport(
            assessments=data,
            first_score=first,
            latest_score=latest,
            total_improvement=delta,
            average_improvement_per_assessment=round(avg, 2),
        )

    def generate_roadmap(
        self,
        current: TwelveFactorReport,
        target_level: str,
    ) -> list[str]:
        """Return an ordered list of steps to reach the target maturity level."""
        level_map = {
            "prototype": 1, "development": 2, "staging": 3,
            "production": 4, "elite": 5,
        }
        target_num = level_map.get(target_level.lower())
        if target_num is None:
            raise ValueError(f"Unknown target level: {target_level!r}")

        if current.maturity_level_number >= target_num:
            return [f"Already at or above {target_level} level. Focus on continuous improvement."]

        # Required factor minimums by level
        required: dict[int, list[int]] = {
            2: [1, 2, 5, 9],
            3: [1, 2, 3, 4, 5, 6, 8, 9],
            4: list(range(1, 12)),
            5: list(range(1, 13)),
        }
        score_by_factor = {f.factor_number: f.score for f in current.factors}
        steps: list[str] = []
        for level in range(current.maturity_level_number + 1, target_num + 1):
            reqs = required.get(level, [])
            for fn in reqs:
                if score_by_factor.get(fn, 1) < 3:
                    name = FACTOR_NAMES[fn]
                    fa = next(f for f in current.factors if f.factor_number == fn)
                    rec = fa.recommendations[0] if fa.recommendations else "See assessment gaps"
                    steps.append(
                        f"[Level {level}] Bring Factor {fn} ({name}) to ≥3: {rec}"
                    )
        # Also suggest improvements for factors at exactly 3 when targeting elite
        if target_num == 5:
            for f in current.factors:
                if f.score == 3 and f.recommendations:
                    steps.append(
                        f"[Elite] Raise Factor {f.factor_number} ({f.factor_name}) from 3→4: {f.recommendations[0]}"
                    )
        return steps or ["No additional steps required — you are on track."]


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

_STATUS_EMOJI = {
    "excellent":        "✅ Excellent",
    "good":             "✅ Good",
    "needs_improvement": "⚠️ Needs Improvement",
    "critical":         "❌ Critical",
}

_ROMAN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
    7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII",
}


def generate_markdown_report(report: TwelveFactorReport) -> str:
    lines: list[str] = [
        "# 12-Factor Agent Assessment Report",
        "",
        f"**Date:** {report.assessed_at}",
        f"**Overall Score:** {report.overall_score}/60",
        f"**Maturity Level:** {report.maturity_level} (Level {report.maturity_level_number})",
        "",
        "## Factor Scores",
        "",
        "| # | Factor | Score | Status |",
        "|---|--------|-------|--------|",
    ]
    for f in report.factors:
        roman = _ROMAN[f.factor_number]
        status = _STATUS_EMOJI.get(f.status, f.status)
        lines.append(f"| {roman} | {f.factor_name} | {f.score}/5 | {status} |")

    if report.critical_gaps:
        lines += ["", "## Critical Gaps (Must Fix)", ""]
        for i, gap in enumerate(report.critical_gaps, 1):
            f_num = int(gap.split("Factor")[1].split()[0].rstrip("("))
            fa = next(x for x in report.factors if x.factor_number == f_num)
            lines.append(f"{i}. **Factor {_ROMAN[f_num]} — {fa.factor_name} (Score: {fa.score}/5)**")
            for g in fa.gaps:
                lines.append(f"   - Missing: {g}")
            for r in fa.recommendations:
                lines.append(f"   - Recommendation: {r}")
            lines.append("")

    if report.improvement_priorities:
        lines += ["## Improvement Priorities", ""]
        for i, p in enumerate(report.improvement_priorities, 1):
            lines.append(f"{i}. {p}")

    # Next steps to next level
    next_level_num = report.maturity_level_number + 1
    level_names = {1: "Prototype", 2: "Development", 3: "Staging", 4: "Production", 5: "Elite"}
    if next_level_num <= 5:
        next_name = level_names[next_level_num]
        lines += [
            "",
            f"## Next Steps",
            "",
            f"To reach **Level {next_level_num} ({next_name})**:",
            f"- Fix all critical gaps (factors scored ≤ 2)",
            f"- Bring all required factors to score ≥ 3",
            f"- Target: {'49' if next_level_num == 4 else '37' if next_level_num == 3 else '25'}+ total score",
        ]

    return "\n".join(lines)


def generate_html_report(report: TwelveFactorReport) -> str:
    score_color = {
        5: "#22c55e", 4: "#86efac", 3: "#eab308", 2: "#ef4444", 1: "#b91c1c",
    }
    pct = round((report.overall_score / 60) * 100)
    factor_rows = ""
    for f in report.factors:
        color = score_color.get(f.score, "#888")
        bar_width = f.score * 20
        roman = _ROMAN[f.factor_number]
        status = _STATUS_EMOJI.get(f.status, f.status)
        gaps_html = "".join(f"<li>{g}</li>" for g in f.gaps)
        recs_html = "".join(f"<li>{r}</li>" for r in f.recommendations)
        detail_html = (
            f'<details><summary>Details</summary>'
            f'<strong>Gaps:</strong><ul>{gaps_html}</ul>'
            f'<strong>Recommendations:</strong><ul>{recs_html}</ul>'
            f'</details>'
        ) if f.gaps or f.recommendations else ""
        factor_rows += (
            f'<tr>'
            f'<td>{roman}</td>'
            f'<td>{f.factor_name}</td>'
            f'<td><div style="background:{color};width:{bar_width}px;height:16px;border-radius:4px;display:inline-block"></div>'
            f' <strong style="color:{color}">{f.score}/5</strong></td>'
            f'<td>{status}</td>'
            f'<td>{detail_html}</td>'
            f'</tr>'
        )

    gaps_html = "".join(
        f'<li><strong>{g}</strong></li>' for g in report.critical_gaps
    )
    prio_html = "".join(
        f'<li>{p}</li>' for p in report.improvement_priorities
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>12-Factor Agent Assessment</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1e293b; }}
  h1 {{ color: #0f172a; }}
  .scorecard {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }}
  .score-big {{ font-size: 2.5rem; font-weight: 700; color: #0f172a; }}
  .level {{ font-size: 1.25rem; color: #64748b; }}
  .progress-bar {{ background: #e2e8f0; border-radius: 99px; height: 12px; margin: 0.5rem 0; }}
  .progress-fill {{ background: #22c55e; border-radius: 99px; height: 12px; width: {pct}%; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  details summary {{ cursor: pointer; color: #3b82f6; }}
  .critical {{ background: #fef2f2; }}
  ul.gaps li {{ color: #dc2626; }}
  ul.recs li {{ color: #16a34a; }}
</style>
</head>
<body>
<h1>12-Factor Agent Assessment Report</h1>
<div class="scorecard">
  <div class="score-big">{report.overall_score}/60</div>
  <div class="level">{report.maturity_level} (Level {report.maturity_level_number}) — {report.assessed_at}</div>
  <div class="progress-bar"><div class="progress-fill"></div></div>
  <small>{pct}% toward Elite</small>
</div>

<h2>Factor Scores</h2>
<table>
  <thead><tr><th>#</th><th>Factor</th><th>Score</th><th>Status</th><th></th></tr></thead>
  <tbody>{factor_rows}</tbody>
</table>

<h2>Critical Gaps</h2>
<ul class="gaps">{gaps_html or "<li>None — no critical gaps!</li>"}</ul>

<h2>Improvement Priorities</h2>
<ol class="recs">{prio_html or "<li>All factors are at maximum score.</li>"}</ol>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    assessor = TwelveFactorAssessor()
    tracker = ImprovementTracker()

    # --- Baseline agent config ---
    baseline_config: dict[str, Any] = {
        # Factor I: prompts in Git, versioned, but no review or rollback
        "prompts_in_version_control": True,
        "prompts_semantically_versioned": True,
        "prompts_code_reviewed": False,
        "prompts_have_change_log": False,
        "prompts_independently_rollbackable": False,
        # Factor II: has state class, serializable, but not persisted
        "has_conversation_state_class": True,
        "state_survives_truncation": True,
        "state_is_serializable": True,
        "state_persisted_across_sessions": False,
        "state_transitions_logged": False,
        # Factor III: no provider abstraction
        "has_llm_provider_interface": False,
        "num_providers_supported": 1,
        "provider_configurable_via_env": False,
        "provider_specific_features_abstracted": False,
        "has_provider_fallback_chain": False,
        # Factor IV: no token budgeting at all
        "has_per_request_token_budget": False,
        "has_per_request_cost_budget": False,
        "token_usage_tracked_and_logged": False,
        "cost_alerts_configured": False,
        "budget_enforcement_blocks_excess": False,
        # Factor V: strong — full structured output
        "all_llm_outputs_schema_validated": True,
        "schema_definitions_in_version_control": True,
        "has_parse_validate_retry_pattern": True,
        "schemas_consistent_across_interactions": True,
        "schema_violations_logged_and_alerted": True,
        # Factor VI: basic context management
        "has_explicit_token_allocation_per_zone": True,
        "context_consumption_measured_per_request": True,
        "has_automatic_compression_when_over_budget": False,
        "has_sliding_window_for_history": False,
        "system_prompts_audited_for_efficiency": False,
        # Factor VII: minimal guardrails
        "input_guardrail_layer_count": 2,
        "output_guardrail_layer_count": 2,
        "safety_filters_on_both_input_and_output": True,
        "has_prompt_injection_detection": False,
        "has_pii_detection_and_redaction": False,
        # Factor VIII: partial degradation support
        "has_fallback_llm_provider": False,
        "has_fallback_for_vector_db": False,
        "has_static_response_for_complete_failure": True,
        "degradation_events_logged": True,
        "degradation_regularly_chaos_tested": False,
        # Factor IX: basic observability
        "has_request_tracing_with_trace_ids": True,
        "traces_exported_to_centralized_system": False,
        "key_metrics_tracked_latency_tokens_cost_errors": True,
        "has_dashboards_for_metrics": False,
        "has_alerts_for_metric_degradation": False,
        # Factor X: no human-in-the-loop
        "has_approval_policy_defined": False,
        "high_stakes_actions_flagged_for_approval": False,
        "has_reviewer_interface": False,
        "has_timeout_handling_for_approvals": False,
        "approval_decisions_logged_for_audit": False,
        # Factor XI: limited evaluation
        "test_set_has_50_plus_queries": True,
        "evaluations_run_on_every_deployment": False,
        "has_regression_detection_baseline_comparison": False,
        "safety_red_team_run_regularly": False,
        "evaluation_results_block_deployment_on_regression": False,
        # Factor XII: weak dev-prod parity
        "same_guardrail_config_in_dev_and_prod": False,
        "production_model_tested_before_deployment": False,
        "has_staging_environment_matching_production": True,
        "knowledge_base_structures_consistent_across_envs": True,
        "environment_differences_documented_and_intentional": False,
    }

    print("=" * 60)
    print("BASELINE ASSESSMENT")
    print("=" * 60)
    baseline_report = assessor.assess(baseline_config)
    print(assessor.export_report(baseline_report))
    print(f"\nMaturity level: {baseline_report.maturity_level} (Level {baseline_report.maturity_level_number})")

    tracker.save_baseline(baseline_report)

    # --- Improved config: add token budgeting + human approval + provider abstraction ---
    improved_config = dict(baseline_config)
    improved_config.update({
        # Factor III: add provider abstraction
        "has_llm_provider_interface": True,
        "num_providers_supported": 2,
        "provider_configurable_via_env": True,
        "has_provider_fallback_chain": True,
        # Factor IV: full token budgeting
        "has_per_request_token_budget": True,
        "has_per_request_cost_budget": True,
        "token_usage_tracked_and_logged": True,
        "cost_alerts_configured": True,
        "budget_enforcement_blocks_excess": True,
        # Factor IX: enhanced observability
        "traces_exported_to_centralized_system": True,
        "has_dashboards_for_metrics": True,
        "has_alerts_for_metric_degradation": True,
        # Factor X: human in the loop
        "has_approval_policy_defined": True,
        "high_stakes_actions_flagged_for_approval": True,
        "has_reviewer_interface": True,
        "has_timeout_handling_for_approvals": True,
        "approval_decisions_logged_for_audit": True,
        # Factor XI: CI evaluation
        "evaluations_run_on_every_deployment": True,
        "has_regression_detection_baseline_comparison": True,
    })

    print("\n" + "=" * 60)
    print("IMPROVED ASSESSMENT")
    print("=" * 60)
    improved_report = assessor.assess(improved_config)
    improved_report.assessed_at = "2026-05-24"
    print(assessor.export_report(improved_report))

    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINE")
    print("=" * 60)
    comparison = assessor.compare_to_baseline(improved_report, baseline_report)
    print(f"Score: {comparison.baseline_score} → {comparison.current_score} ({comparison.score_delta:+d})")
    print(f"Level: {comparison.baseline_level} → {comparison.current_level}")
    print("\nImproved factors:")
    for num, name, old, new in comparison.improved_factors:
        print(f"  Factor {num} ({name}): {old} → {new} (+{new - old})")
    if comparison.regressed_factors:
        print("\nRegressed factors:")
        for num, name, old, new in comparison.regressed_factors:
            print(f"  Factor {num} ({name}): {old} → {new} ({new - old:+d})")

    tracker.save_baseline(improved_report)
    trend = tracker.track_over_time()
    print("\n" + "=" * 60)
    print("TREND REPORT")
    print("=" * 60)
    for entry in trend.assessments:
        bar = "█" * (entry["score"] // 6) + "░" * (10 - entry["score"] // 6)
        print(f"  {entry['date']}  {entry['score']:>3}/60  {entry['level']:<14}  {bar}")
    print(f"\nTotal improvement: +{trend.total_improvement} points")

    print("\n" + "=" * 60)
    print("ROADMAP TO ELITE")
    print("=" * 60)
    roadmap = ImprovementTracker()
    roadmap.save_baseline(improved_report)
    steps = roadmap.generate_roadmap(improved_report, "elite")
    for step in steps:
        print(f"  {step}")


if __name__ == "__main__":
    _demo()
