"""
12-Factor Agent Validation Rules Engine
=========================================
Automated static-analysis checks for each of the 12 production-readiness factors.
Designed to run in CI/CD to block deployments that do not meet minimum standards.

Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationRule:
    factor_number: int
    factor_name: str
    rule_name: str
    description: str
    check_function: Callable[[dict, Path], tuple[bool, str]]
    severity: str            # "blocking" | "warning" | "info"
    minimum_level: str       # The maturity level at which this rule is enforced


@dataclass
class ValidationCheck:
    rule: ValidationRule
    passed: bool
    evidence: str        # What was found (or not found)
    recommendation: str  # How to fix if failed


@dataclass
class ValidationReport:
    total_checks: int
    passed_count: int
    failed_count: int
    warning_count: int
    checks: list[ValidationCheck] = field(default_factory=list)
    blocking_failures: list[ValidationCheck] = field(default_factory=list)
    deployment_allowed: bool = True  # True if no blocking failures


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _find_files(root: Path, *patterns: str) -> list[Path]:
    """Return all files under root matching any of the given glob patterns."""
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    return matches


def _source_files(root: Path) -> list[Path]:
    """Return all Python source files under root (excluding hidden dirs / venvs)."""
    skip = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"}
    result: list[Path] = []
    for p in root.rglob("*.py"):
        if not any(part in skip for part in p.parts):
            result.append(p)
    return result


def _all_source_text(root: Path) -> str:
    """Concatenate all Python source files for broad text searches."""
    parts: list[str] = []
    for p in _source_files(root):
        try:
            parts.append(p.read_text(errors="replace"))
        except OSError:
            pass
    return "\n".join(parts)


def _file_contains(path: Path, pattern: str, regex: bool = False) -> bool:
    try:
        text = path.read_text(errors="replace")
        if regex:
            return bool(re.search(pattern, text))
        return pattern in text
    except OSError:
        return False


def _any_file_contains(root: Path, pattern: str, regex: bool = False) -> bool:
    for p in _source_files(root):
        if _file_contains(p, pattern, regex=regex):
            return True
    return False


def _count_pattern_occurrences(root: Path, pattern: str, regex: bool = False) -> int:
    count = 0
    for p in _source_files(root):
        try:
            text = p.read_text(errors="replace")
            if regex:
                count += len(re.findall(pattern, text))
            elif pattern in text:
                count += 1
        except OSError:
            pass
    return count


# ---------------------------------------------------------------------------
# Rule check functions  (each returns (passed: bool, evidence: str))
# ---------------------------------------------------------------------------

# --- Factor I: Prompt as Code ---

def _check_prompts_dir(cfg: dict, root: Path) -> tuple[bool, str]:
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        return False, "No prompts/ directory found at project root"
    files = list(prompts_dir.rglob("*.yaml")) + list(prompts_dir.rglob("*.yml")) + list(prompts_dir.rglob("*.md"))
    if not files:
        return False, "prompts/ directory exists but contains no .yaml / .md files"
    return True, f"Found {len(files)} prompt file(s) in prompts/"


def _check_prompt_versions(cfg: dict, root: Path) -> tuple[bool, str]:
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        return False, "No prompts/ directory — cannot check versioning"
    versioned = 0
    total = 0
    for f in prompts_dir.rglob("*.yaml"):
        total += 1
        try:
            text = f.read_text()
            if re.search(r"version\s*:", text):
                versioned += 1
        except OSError:
            pass
    if total == 0:
        return False, "No YAML prompt files found to check for version field"
    if versioned < total:
        return False, f"Only {versioned}/{total} prompt YAML files have a version field"
    return True, f"All {total} YAML prompt files include a version field"


def _check_prompts_not_hardcoded(cfg: dict, root: Path) -> tuple[bool, str]:
    """Warning if long string literals that look like system prompts appear in .py files."""
    suspicious = 0
    for p in _source_files(root):
        if "prompts" in str(p):
            continue  # allow prompts/ directory Python files
        try:
            text = p.read_text(errors="replace")
            # Look for multi-line strings > 200 chars containing "You are"
            if re.search(r'"""[^"]{200,}"""', text) and re.search(r'You are\b', text):
                suspicious += 1
        except OSError:
            pass
    if suspicious > 0:
        return False, f"Found {suspicious} file(s) with inline system prompts (should be in prompts/)"
    return True, "No inline system prompts detected in Python source files"


# --- Factor II: Explicit State ---

def _check_state_class_exists(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"class\s+ConversationState", regex=True):
        return True, "ConversationState class/dataclass found"
    if _any_file_contains(root, r"class\s+\w*State\w*\s*[:\(]", regex=True):
        return True, "A State class (matching *State*) was found"
    return False, "No ConversationState or *State* class found in codebase"


def _check_state_serializable(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"to_dict|model_dump|asdict|json\.dumps", regex=True):
        return True, "State serialization method (to_dict/model_dump/asdict) found"
    return False, "No state serialization method found — state cannot be persisted or debugged"


def _check_state_injected(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"inject_state|state_to_prompt|state\.to_dict\(\)|state_summary", regex=True):
        return True, "State injection into LLM prompt detected"
    return False, "No state injection code found — state may not survive message truncation"


def _check_state_not_only_in_messages(cfg: dict, root: Path) -> tuple[bool, str]:
    """Warning if state appears to live only in the message list."""
    has_state_class = _any_file_contains(root, r"class\s+\w*State", regex=True)
    only_messages = _any_file_contains(root, r"messages\.append.*state", regex=True)
    if not has_state_class and only_messages:
        return False, "State appears to only exist in the message list (no separate State class)"
    return True, "State management is not solely in the message list"


# --- Factor III: Provider Agnostic ---

def _check_provider_interface(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"class\s+\w*Provider\w*|LLMProvider|BaseLLM|AbstractLLM", regex=True):
        return True, "LLM provider interface/abstract class found"
    return False, "No LLM provider interface found — provider is likely tightly coupled"


def _check_two_providers(cfg: dict, root: Path) -> tuple[bool, str]:
    providers_found: list[str] = []
    source = _all_source_text(root)
    if "openai" in source.lower():
        providers_found.append("OpenAI")
    if "anthropic" in source.lower():
        providers_found.append("Anthropic")
    if "google" in source.lower() or "gemini" in source.lower():
        providers_found.append("Google")
    if "mistral" in source.lower():
        providers_found.append("Mistral")
    if len(providers_found) >= 2:
        return True, f"Multiple providers found: {', '.join(providers_found)}"
    return False, f"Only {len(providers_found)} provider(s) found ({', '.join(providers_found) or 'none'}) — need ≥2"


def _check_provider_env_config(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"LLM_PROVIDER|LLM_MODEL|PROVIDER|os\.environ|os\.getenv", regex=True):
        return True, "Provider/model configurable via environment variables"
    return False, "No environment-variable-based provider configuration found"


def _check_only_openai_warn(cfg: dict, root: Path) -> tuple[bool, str]:
    source = _all_source_text(root)
    has_openai = "openai" in source.lower()
    has_other = any(p in source.lower() for p in ["anthropic", "gemini", "mistral", "google"])
    if has_openai and not has_other:
        return False, "OpenAI is the only provider — add at least one fallback provider"
    return True, "Multiple providers configured (not OpenAI-only)"


# --- Factor IV: Token Budgeting ---

def _check_token_counting(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"tiktoken|count_tokens|token_count|num_tokens|TokenBudget", regex=True):
        return True, "Token counting function/class found"
    return False, "No token counting code found — token usage is blind"


def _check_budget_check_before_llm(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"budget\.check|check_budget|remaining_tokens|max_tokens.*budget", regex=True):
        return True, "Budget check before LLM call detected"
    return False, "No budget check before LLM calls — requests may exceed limits silently"


def _check_token_usage_logged(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"tokens_used|prompt_tokens|completion_tokens|total_tokens", regex=True):
        return True, "Token usage logging found"
    return False, "Token usage not logged — no post-hoc cost analysis possible"


def _check_cost_estimation(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"cost_per_token|price|usd|cost_incurred|token_cost", regex=True):
        return True, "Cost estimation code found"
    return False, "No cost estimation code — spending is invisible"


# --- Factor V: Structured Everything ---

def _check_pydantic_schemas(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"from pydantic|BaseModel|model_validate|TypedDict", regex=True):
        return True, "Pydantic/TypedDict schema definitions found"
    return False, "No Pydantic or TypedDict schemas found — LLM output parsing is likely fragile"


def _check_schema_validation_after_llm(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"model_validate_json|parse_raw|parse_obj|\.validate\(|ValidationError", regex=True):
        return True, "Schema validation after LLM call detected"
    return False, "No schema validation call found after LLM responses"


def _check_parse_validate_retry(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"ValidationError|retry.*parse|parse.*retry|except.*Validation", regex=True):
        return True, "Parse-validate-retry pattern found"
    return False, "No parse-validate-retry pattern — schema failures will crash the agent"


def _check_no_string_parsing(cfg: dict, root: Path) -> tuple[bool, str]:
    """Warning if regex or string search is used on LLM output."""
    if _any_file_contains(root, r're\.search.*response|re\.match.*response|response\.split|response\.strip\(\)\.split', regex=True):
        return False, "String/regex parsing of LLM output detected — use structured output instead"
    return True, "No raw string parsing of LLM output detected"


# --- Factor VI: Context Is a Resource ---

def _check_context_budget_code(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"ContextBudget|context_budget|token_allocation|context_window", regex=True):
        return True, "Context budget/allocation code found"
    return False, "No context budget code — context window fills up unpredictably"


def _check_context_token_counting(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"count_tokens.*context|context.*tokens|context_tokens", regex=True):
        return True, "Context token counting found"
    return False, "Context tokens not measured — cannot enforce budget"


def _check_context_compression(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"summarize|compress|truncate|sliding_window", regex=True):
        return True, "Context compression/summarization code found"
    return False, "No context compression — long conversations will overflow the context window"


# --- Factor VII: Defense in Depth ---

def _check_three_input_guardrails(cfg: dict, root: Path) -> tuple[bool, str]:
    guardrail_patterns = [
        r"rate.?limit",
        r"structural.?valid|input.?valid",
        r"pii.?detect|redact",
        r"content.?filter|input.?guard",
        r"injection.?detect|prompt.?inject",
    ]
    count = sum(
        1 for p in guardrail_patterns
        if _any_file_contains(root, p, regex=True)
    )
    if count >= 3:
        return True, f"{count} distinct input guardrail pattern(s) detected"
    return False, f"Only {count} input guardrail pattern(s) found — need ≥3 for defense in depth"


def _check_three_output_guardrails(cfg: dict, root: Path) -> tuple[bool, str]:
    output_patterns = [
        r"output.?guard|output.?valid|schema.?valid",
        r"output.*pii|pii.*output|leakage",
        r"safety.?filter|harmful|toxicity",
        r"hallucin|fact.?check",
    ]
    count = sum(
        1 for p in output_patterns
        if _any_file_contains(root, p, regex=True)
    )
    if count >= 3:
        return True, f"{count} distinct output guardrail pattern(s) detected"
    return False, f"Only {count} output guardrail pattern(s) found — need ≥3"


def _check_guardrail_not_single_flag(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"guardrails_enabled\s*=\s*False|disable_guardrails\s*=\s*True", regex=True):
        return False, "Guardrails can be disabled via a single flag — no single point of failure allowed"
    return True, "No single-flag guardrail disable found"


def _check_injection_detection(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"injection|prompt.?inject|role.?override|system.?override", regex=True):
        return True, "Prompt injection detection code found"
    return False, "No prompt injection detection — agent vulnerable to role-override attacks"


# --- Factor VIII: Graceful Degradation ---

def _check_fallback_provider(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"fallback.?provider|provider.?fallback|circuit.?break|switch_provider", regex=True):
        return True, "Fallback provider / circuit breaker found"
    return False, "No fallback provider — primary LLM outage causes complete failure"


def _check_static_fallback(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"static.?response|fallback.?response|apologize.*technical|sorry.*unavailable", regex=True):
        return True, "Static/error fallback response found"
    return False, "No static fallback response — raw exceptions may reach users"


def _check_circuit_breaker(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"CircuitBreak|circuit_break|OPEN|HALF_OPEN|CLOSED", regex=True):
        return True, "Circuit breaker implementation found"
    return False, "No circuit breaker — failing external services will be retried indefinitely"


def _check_no_fallback_warn(cfg: dict, root: Path) -> tuple[bool, str]:
    if not _any_file_contains(root, r"except.*Provider|fallback|circuit", regex=True):
        return False, "No fallback handling for any external dependency"
    return True, "Fallback handling for external dependencies found"


# --- Factor IX: Observability First ---

def _check_trace_id(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"trace_id|TraceID|uuid4|request_id", regex=True):
        return True, "Trace ID generation found"
    return False, "No trace ID generation — requests cannot be tracked end-to-end"


def _check_structured_logging(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"structlog|logging\.getLogger|logger\.|json_log|structured_log", regex=True):
        return True, "Structured logging implementation found"
    return False, "No structured logging — log lines are hard to query"


def _check_metrics_collection(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"latency|duration_ms|error_rate|total_cost|tokens_used", regex=True):
        return True, "Metrics collection (latency/tokens/cost/errors) found"
    return False, "No metrics collection — impossible to track agent performance over time"


def _check_metrics_export_warn(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"langsmith|arize|opentelemetry|otel|prometheus|datadog|grafana", regex=True):
        return True, "Metrics export to centralized system detected"
    return False, "Metrics not exported to a centralized system — limited cross-request visibility"


# --- Factor X: Human in the Loop ---

def _check_approval_policy(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"ApprovalPolicy|approval_policy|requires_approval|HumanApproval", regex=True):
        return True, "Approval policy class/function found"
    return False, "No approval policy — all actions execute without human oversight"


def _check_high_stakes_flagged(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"risk_level|high.?stakes|requires_review|action.*approval", regex=True):
        return True, "High-stakes action identification code found"
    return False, "No high-stakes action flagging — agent may execute dangerous actions silently"


def _check_approval_timeout(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"timeout.*approv|approv.*timeout|asyncio\.wait_for.*approv", regex=True):
        return True, "Approval timeout handling found"
    return False, "No approval timeout — approval requests may hang indefinitely"


def _check_no_approval_warn(cfg: dict, root: Path) -> tuple[bool, str]:
    if not _any_file_contains(root, r"approv|human.?review|human.?in.?loop", regex=True):
        return False, "No human approval code found — entire agent runs autonomously"
    return True, "Human approval/review code present"


# --- Factor XI: Continuous Evaluation ---

def _check_test_set_50(cfg: dict, root: Path) -> tuple[bool, str]:
    """Heuristic: look for evaluation JSON/YAML files with 50+ items."""
    for p in list(root.rglob("*.json")) + list(root.rglob("*.jsonl")) + list(root.rglob("*.yaml")):
        if any(kw in p.name.lower() for kw in ("eval", "test", "benchmark", "queries")):
            try:
                text = p.read_text(errors="replace")
                # Count array entries
                entries = len(re.findall(r'^\s*[\{\[]', text, re.MULTILINE))
                if entries >= 50:
                    return True, f"Evaluation file {p.name} contains ~{entries} test entries"
                # Try counting lines for .jsonl
                lines = [l for l in text.splitlines() if l.strip().startswith("{")]
                if len(lines) >= 50:
                    return True, f"Evaluation file {p.name} contains {len(lines)} JSONL entries"
            except OSError:
                pass
    return False, "No evaluation test set with ≥50 queries found"


def _check_eval_in_ci(cfg: dict, root: Path) -> tuple[bool, str]:
    ci_files = (
        list(root.rglob(".github/workflows/*.yml"))
        + list(root.rglob(".gitlab-ci.yml"))
        + list(root.rglob("Jenkinsfile"))
        + list(root.rglob(".circleci/config.yml"))
    )
    for f in ci_files:
        if _file_contains(f, r"eval|evaluate|assessment", regex=True):
            return True, f"Evaluation step found in CI config: {f.name}"
    return False, "No evaluation step found in CI/CD configuration"


def _check_regression_detection(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"baseline|regression|detect_regression|compare.*baseline", regex=True):
        return True, "Regression detection / baseline comparison code found"
    return False, "No regression detection — quality regressions go unnoticed"


def _check_eval_not_only_manual_warn(cfg: dict, root: Path) -> tuple[bool, str]:
    has_eval_code = _any_file_contains(root, r"evaluate|evaluator|eval_suite", regex=True)
    has_ci = any(
        _file_contains(p, r"eval|evaluate", regex=True)
        for p in (
            list(root.rglob(".github/workflows/*.yml"))
            + list(root.rglob(".gitlab-ci.yml"))
        )
    )
    if has_eval_code and not has_ci:
        return False, "Evaluation code exists but is not wired into CI/CD"
    return True, "Evaluation is automated in CI/CD (or no evaluation code found)"


# --- Factor XII: Dev-Prod Parity ---

def _check_guardrail_parity(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"guardrails_enabled.*True|same.*guardrail|guardrail.*config", regex=True):
        return True, "Guardrail configuration appears consistent across environments"
    return False, "Cannot confirm guardrail parity across environments"


def _check_env_configs_documented(cfg: dict, root: Path) -> tuple[bool, str]:
    doc_files = (
        list(root.rglob("ENVIRONMENTS.md"))
        + list(root.rglob("environments.yaml"))
        + list(root.rglob("config/*.yaml"))
        + list(root.rglob("config/*.yml"))
    )
    if doc_files:
        return True, f"Environment configuration files found: {[f.name for f in doc_files[:3]]}"
    return False, "No environment configuration documentation found"


def _check_staging_exists(cfg: dict, root: Path) -> tuple[bool, str]:
    for pattern in ("staging", "stage", "preprod"):
        if (root / pattern).exists():
            return True, f"Staging directory ({pattern}/) found"
        if list(root.rglob(f"*{pattern}*")):
            return True, f"Staging references found in codebase"
    return False, "No staging environment configuration found"


def _check_dev_prod_diff_warn(cfg: dict, root: Path) -> tuple[bool, str]:
    if _any_file_contains(root, r"dev.*guardrail.*False|guardrails.*dev.*False|disable.*dev", regex=True):
        return False, "Guardrails appear to be disabled in dev — this is a parity violation"
    return True, "No obvious dev-only guardrail disabling detected"


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

def _build_rules() -> list[ValidationRule]:
    return [
        # Factor I
        ValidationRule(1, "Prompt as Code", "prompts_dir_exists",
                       "prompts/ directory with versioned YAML/MD files exists",
                       _check_prompts_dir, "blocking", "development"),
        ValidationRule(1, "Prompt as Code", "prompt_versions",
                       "Each prompt file has a version field",
                       _check_prompt_versions, "warning", "development"),
        ValidationRule(1, "Prompt as Code", "prompts_not_hardcoded",
                       "Prompts are not hardcoded in Python source files",
                       _check_prompts_not_hardcoded, "warning", "staging"),

        # Factor II
        ValidationRule(2, "Explicit State", "state_class_exists",
                       "ConversationState class or dataclass exists",
                       _check_state_class_exists, "blocking", "development"),
        ValidationRule(2, "Explicit State", "state_serializable",
                       "State has to_dict() or model_dump() serialization method",
                       _check_state_serializable, "warning", "staging"),
        ValidationRule(2, "Explicit State", "state_injected",
                       "State is injected into LLM calls",
                       _check_state_injected, "warning", "staging"),
        ValidationRule(2, "Explicit State", "state_not_only_in_messages",
                       "State is not only stored in the message list",
                       _check_state_not_only_in_messages, "warning", "development"),

        # Factor III
        ValidationRule(3, "Provider Agnostic", "provider_interface",
                       "LLM provider interface / abstract class exists",
                       _check_provider_interface, "blocking", "staging"),
        ValidationRule(3, "Provider Agnostic", "two_providers",
                       "At least 2 LLM providers are configured",
                       _check_two_providers, "blocking", "staging"),
        ValidationRule(3, "Provider Agnostic", "provider_env_config",
                       "Provider is configurable via environment variable",
                       _check_provider_env_config, "warning", "development"),
        ValidationRule(3, "Provider Agnostic", "only_openai_warn",
                       "Not exclusively using OpenAI",
                       _check_only_openai_warn, "warning", "staging"),

        # Factor IV
        ValidationRule(4, "Token Budgeting", "token_counting",
                       "Token counting function/class exists",
                       _check_token_counting, "blocking", "staging"),
        ValidationRule(4, "Token Budgeting", "budget_check_before_llm",
                       "Budget check occurs before LLM calls",
                       _check_budget_check_before_llm, "blocking", "staging"),
        ValidationRule(4, "Token Budgeting", "token_usage_logged",
                       "Token usage is logged per request",
                       _check_token_usage_logged, "warning", "development"),
        ValidationRule(4, "Token Budgeting", "cost_estimation",
                       "Cost estimation code exists",
                       _check_cost_estimation, "warning", "staging"),

        # Factor V
        ValidationRule(5, "Structured Everything", "pydantic_schemas",
                       "Pydantic or TypedDict schema definitions exist",
                       _check_pydantic_schemas, "blocking", "development"),
        ValidationRule(5, "Structured Everything", "schema_validation_after_llm",
                       "Schema validation occurs after every LLM call",
                       _check_schema_validation_after_llm, "blocking", "development"),
        ValidationRule(5, "Structured Everything", "parse_validate_retry",
                       "Parse-validate-retry pattern is implemented",
                       _check_parse_validate_retry, "warning", "staging"),
        ValidationRule(5, "Structured Everything", "no_string_parsing",
                       "LLM output is not parsed with regex or string splitting",
                       _check_no_string_parsing, "warning", "development"),

        # Factor VI
        ValidationRule(6, "Context Is a Resource", "context_budget_code",
                       "Context budget/allocation code exists",
                       _check_context_budget_code, "blocking", "staging"),
        ValidationRule(6, "Context Is a Resource", "context_token_counting",
                       "Tokens are counted per context zone",
                       _check_context_token_counting, "warning", "staging"),
        ValidationRule(6, "Context Is a Resource", "context_compression",
                       "Context compression or sliding window implemented",
                       _check_context_compression, "warning", "staging"),

        # Factor VII
        ValidationRule(7, "Defense in Depth", "three_input_guardrails",
                       "At least 3 independent input guardrail layers present",
                       _check_three_input_guardrails, "blocking", "staging"),
        ValidationRule(7, "Defense in Depth", "three_output_guardrails",
                       "At least 3 independent output guardrail layers present",
                       _check_three_output_guardrails, "blocking", "staging"),
        ValidationRule(7, "Defense in Depth", "guardrail_not_single_flag",
                       "Guardrails cannot be disabled via a single config flag",
                       _check_guardrail_not_single_flag, "warning", "staging"),
        ValidationRule(7, "Defense in Depth", "injection_detection",
                       "Prompt injection detection is implemented",
                       _check_injection_detection, "blocking", "production"),

        # Factor VIII
        ValidationRule(8, "Graceful Degradation", "fallback_provider",
                       "Fallback LLM provider or circuit breaker configured",
                       _check_fallback_provider, "blocking", "staging"),
        ValidationRule(8, "Graceful Degradation", "static_fallback",
                       "Static/error fallback response exists for complete failures",
                       _check_static_fallback, "warning", "staging"),
        ValidationRule(8, "Graceful Degradation", "circuit_breaker",
                       "Circuit breaker implementation exists",
                       _check_circuit_breaker, "blocking", "production"),
        ValidationRule(8, "Graceful Degradation", "no_fallback_warn",
                       "Fallback handling for external dependencies exists",
                       _check_no_fallback_warn, "warning", "development"),

        # Factor IX
        ValidationRule(9, "Observability First", "trace_id",
                       "Unique trace ID generated for every request",
                       _check_trace_id, "blocking", "development"),
        ValidationRule(9, "Observability First", "structured_logging",
                       "Structured logging is implemented",
                       _check_structured_logging, "warning", "development"),
        ValidationRule(9, "Observability First", "metrics_collection",
                       "Key metrics (latency, tokens, cost, errors) are collected",
                       _check_metrics_collection, "blocking", "staging"),
        ValidationRule(9, "Observability First", "metrics_export",
                       "Metrics exported to a centralized observability system",
                       _check_metrics_export_warn, "warning", "staging"),

        # Factor X
        ValidationRule(10, "Human in the Loop", "approval_policy",
                       "Approval policy class/function defined",
                       _check_approval_policy, "blocking", "production"),
        ValidationRule(10, "Human in the Loop", "high_stakes_flagged",
                       "High-stakes actions are flagged for human review",
                       _check_high_stakes_flagged, "blocking", "production"),
        ValidationRule(10, "Human in the Loop", "approval_timeout",
                       "Approval timeout handling is implemented",
                       _check_approval_timeout, "warning", "production"),
        ValidationRule(10, "Human in the Loop", "no_approval_warn",
                       "Human approval code present",
                       _check_no_approval_warn, "warning", "staging"),

        # Factor XI
        ValidationRule(11, "Continuous Evaluation", "test_set_50",
                       "Evaluation test set of ≥50 queries exists",
                       _check_test_set_50, "blocking", "staging"),
        ValidationRule(11, "Continuous Evaluation", "eval_in_ci",
                       "Evaluation step present in CI/CD pipeline",
                       _check_eval_in_ci, "blocking", "production"),
        ValidationRule(11, "Continuous Evaluation", "regression_detection",
                       "Regression detection / baseline comparison implemented",
                       _check_regression_detection, "blocking", "production"),
        ValidationRule(11, "Continuous Evaluation", "eval_automated",
                       "Evaluation is automated, not only run manually",
                       _check_eval_not_only_manual_warn, "warning", "staging"),

        # Factor XII
        ValidationRule(12, "Dev-Prod Parity", "guardrail_parity",
                       "Guardrail configuration consistent across environments",
                       _check_guardrail_parity, "blocking", "staging"),
        ValidationRule(12, "Dev-Prod Parity", "env_configs_documented",
                       "Environment configuration differences are documented",
                       _check_env_configs_documented, "warning", "staging"),
        ValidationRule(12, "Dev-Prod Parity", "staging_exists",
                       "Staging environment configuration exists",
                       _check_staging_exists, "blocking", "staging"),
        ValidationRule(12, "Dev-Prod Parity", "dev_prod_diff_warn",
                       "Dev environment does not disable guardrails relative to prod",
                       _check_dev_prod_diff_warn, "warning", "development"),
    ]


# ---------------------------------------------------------------------------
# Main validator class
# ---------------------------------------------------------------------------

_LEVEL_ORDER = {"development": 1, "staging": 2, "production": 3, "elite": 4}


class TwelveFactorValidator:
    """
    Automated validation of the 12 factors against a codebase.

    Usage::

        validator = TwelveFactorValidator(minimum_level="staging")
        report = validator.validate({}, codebase_path=Path("."))
        print(validator.format_fix_list(report))
    """

    def __init__(self, minimum_level: str = "staging") -> None:
        if minimum_level.lower() not in _LEVEL_ORDER:
            raise ValueError(f"minimum_level must be one of {list(_LEVEL_ORDER)}")
        self.minimum_level = minimum_level.lower()
        self.rules = _build_rules()

    def validate(
        self,
        agent_config: dict,
        codebase_path: str | Path = ".",
    ) -> ValidationReport:
        """Run all automated validation checks against the codebase."""
        root = Path(codebase_path).resolve()
        checks: list[ValidationCheck] = []
        blocking_failures: list[ValidationCheck] = []
        passed = 0
        failed = 0
        warnings = 0

        for rule in self.rules:
            # Only enforce rules at or below the minimum level
            if _LEVEL_ORDER.get(rule.minimum_level, 99) > _LEVEL_ORDER[self.minimum_level]:
                continue

            try:
                ok, evidence = rule.check_function(agent_config, root)
            except Exception as exc:
                ok, evidence = False, f"Check raised an error: {exc}"

            rec = self._get_recommendation(rule.rule_name)
            check = ValidationCheck(rule=rule, passed=ok, evidence=evidence, recommendation=rec)
            checks.append(check)

            if ok:
                passed += 1
            elif rule.severity == "blocking":
                failed += 1
                blocking_failures.append(check)
            else:
                warnings += 1

        return ValidationReport(
            total_checks=len(checks),
            passed_count=passed,
            failed_count=failed,
            warning_count=warnings,
            checks=checks,
            blocking_failures=blocking_failures,
            deployment_allowed=len(blocking_failures) == 0,
        )

    def check_factor(
        self,
        factor_number: int,
        agent_config: dict,
        codebase_path: str | Path = ".",
    ) -> list[ValidationCheck]:
        """Run all rules for a specific factor number."""
        root = Path(codebase_path).resolve()
        results: list[ValidationCheck] = []
        for rule in self.rules:
            if rule.factor_number != factor_number:
                continue
            try:
                ok, evidence = rule.check_function(agent_config, root)
            except Exception as exc:
                ok, evidence = False, f"Check error: {exc}"
            rec = self._get_recommendation(rule.rule_name)
            results.append(ValidationCheck(rule=rule, passed=ok, evidence=evidence, recommendation=rec))
        return results

    def format_fix_list(self, report: ValidationReport) -> str:
        """Return a human-readable, priority-ordered fix list."""
        lines: list[str] = ["=== 12-Factor Validation Fix List ===", ""]
        if report.deployment_allowed:
            lines.append("✅ No blocking failures — deployment allowed.")
        else:
            lines.append("❌ DEPLOYMENT BLOCKED — fix the following issues:")

        if report.blocking_failures:
            lines += ["", "BLOCKING FAILURES (must fix):"]
            for i, c in enumerate(report.blocking_failures, 1):
                lines.append(f"  {i}. [Factor {c.rule.factor_number}] {c.rule.description}")
                lines.append(f"     Found:  {c.evidence}")
                lines.append(f"     Fix:    {c.recommendation}")
                lines.append("")

        warnings = [c for c in report.checks if not c.passed and c.rule.severity == "warning"]
        if warnings:
            lines += ["WARNINGS (should fix):"]
            for i, c in enumerate(warnings, 1):
                lines.append(f"  {i}. [Factor {c.rule.factor_number}] {c.rule.description}")
                lines.append(f"     Found:  {c.evidence}")
                lines.append(f"     Fix:    {c.recommendation}")
                lines.append("")

        lines.append(
            f"Summary: {report.passed_count} passed, "
            f"{report.failed_count} blocking, {report.warning_count} warnings "
            f"({report.total_checks} total checks)"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _RECOMMENDATIONS: dict[str, str] = {
        "prompts_dir_exists": "Create a prompts/ directory at the project root with versioned YAML files",
        "prompt_versions": "Add a 'version: x.y.z' field to every prompt YAML file",
        "prompts_not_hardcoded": "Move system prompt strings from Python files into prompts/*.yaml",
        "state_class_exists": "Create a ConversationState dataclass that holds all session context",
        "state_serializable": "Add a to_dict() or model_dump() method to ConversationState",
        "state_injected": "Call inject_state_into_prompt(messages, state) before every LLM call",
        "state_not_only_in_messages": "Create a separate ConversationState object; do not rely solely on the message list",
        "provider_interface": "Define an abstract LLMProvider class with a chat() method",
        "two_providers": "Implement a second provider (e.g., AnthropicProvider) alongside the primary",
        "provider_env_config": "Read provider and model from LLM_PROVIDER and LLM_MODEL env vars",
        "only_openai_warn": "Add at least one non-OpenAI provider as a fallback",
        "token_counting": "Integrate tiktoken or a similar library to count tokens before LLM calls",
        "budget_check_before_llm": "Add a budget.check(estimated_tokens) guard before every LLM call",
        "token_usage_logged": "Log prompt_tokens, completion_tokens, and total_cost for each response",
        "cost_estimation": "Add cost estimation: tokens × cost_per_token for the configured model",
        "pydantic_schemas": "Define Pydantic models for every LLM response shape",
        "schema_validation_after_llm": "Call Model.model_validate_json(response.content) after every LLM call",
        "parse_validate_retry": "Catch ValidationError and retry the LLM call with the error message as context",
        "no_string_parsing": "Replace regex/split parsing with a structured output schema",
        "context_budget_code": "Create a ContextBudget class that allocates tokens per zone",
        "context_token_counting": "Count tokens for each context zone on every request",
        "context_compression": "Add a summarize_history() function that triggers when history exceeds budget",
        "three_input_guardrails": "Add: rate limiter, structural validator, PII detector, content filter, injection detector",
        "three_output_guardrails": "Add: schema validator, PII check, safety filter, leakage detector, hallucination check",
        "guardrail_not_single_flag": "Remove any single flag that disables all guardrails at once",
        "injection_detection": "Add a prompt injection scanner before passing user input to the agent",
        "fallback_provider": "Implement a provider fallback chain with circuit breaker",
        "static_fallback": "Add a final except: block that returns a user-friendly error message",
        "circuit_breaker": "Implement a CircuitBreaker class with CLOSED/OPEN/HALF_OPEN states",
        "no_fallback_warn": "Add try/except with fallback logic around every external dependency call",
        "trace_id": "Generate a UUID4 trace_id at request entry and attach it to all log lines",
        "structured_logging": "Switch to structured JSON logging (structlog or logging with JsonFormatter)",
        "metrics_collection": "Emit latency_ms, tokens_used, cost_usd, and error_rate per request",
        "metrics_export": "Export metrics via OpenTelemetry to LangSmith, Arize, or Prometheus",
        "approval_policy": "Implement an ApprovalPolicy class with add_rule() for action types and risk levels",
        "high_stakes_flagged": "Tag actions with a risk_level field; route HIGH/CRITICAL through approval",
        "approval_timeout": "Wrap approval_interface.request_approval() in asyncio.wait_for() with a timeout",
        "no_approval_warn": "Add an ApprovalPolicy for at least the highest-risk agent actions",
        "test_set_50": "Create evals/test_queries.jsonl with ≥50 annotated query/expected-response pairs",
        "eval_in_ci": "Add an evaluation step to .github/workflows/ci.yml that runs your eval suite",
        "regression_detection": "Store eval baseline in a JSON file; compare each run and alert on >5% regression",
        "eval_automated": "Ensure your eval script is invoked in CI, not only run manually",
        "guardrail_parity": "Load guardrail config from a shared YAML that all environments read",
        "env_configs_documented": "Create ENVIRONMENTS.md documenting every intentional dev/staging/prod difference",
        "staging_exists": "Create a staging/ config directory or staging deployment manifest",
        "dev_prod_diff_warn": "Remove any code that disables guardrails in development",
    }

    def _get_recommendation(self, rule_name: str) -> str:
        return self._RECOMMENDATIONS.get(rule_name, "Review the 12-Factor Agent documentation for guidance")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import tempfile, json

    validator = TwelveFactorValidator(minimum_level="staging")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Scaffold a minimal codebase that passes some but not all checks
        (root / "prompts").mkdir()
        (root / "prompts" / "support_v1.yaml").write_text("version: '1.0.0'\nsystem_prompt: |\n  You are a support agent.\n")

        (root / "agent.py").write_text(
            "from pydantic import BaseModel\n"
            "import uuid, logging\n"
            "logger = logging.getLogger(__name__)\n\n"
            "class ConversationState(BaseModel):\n"
            "    user_id: str\n"
            "    def model_dump(self): return {}\n\n"
            "class SupportResponse(BaseModel):\n"
            "    answer: str\n\n"
            "def handle(state, user_input):\n"
            "    trace_id = str(uuid.uuid4())\n"
            "    # call LLM\n"
            "    tokens_used = 150\n"
            "    latency_ms = 900\n"
            "    return SupportResponse(answer='ok')\n"
        )

        (root / "llm.py").write_text(
            "import os\n"
            "import anthropic, openai\n"
            "LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')\n\n"
            "class LLMProvider:\n"
            "    def chat(self, messages): ...\n\n"
            "class OpenAIProvider(LLMProvider):\n"
            "    def chat(self, messages):\n"
            "        response = openai.chat.completions.create(model='gpt-4o', messages=messages)\n"
            "        return response\n\n"
            "class AnthropicProvider(LLMProvider):\n"
            "    def chat(self, messages):\n"
            "        return anthropic.Anthropic().messages.create(model='claude-3-5-sonnet', messages=messages)\n"
        )

        # No circuit breaker, no guardrails, no evaluation files
        report = validator.validate({}, codebase_path=root)

    print(validator.format_fix_list(report))
    print(f"\nDeployment allowed: {report.deployment_allowed}")


if __name__ == "__main__":
    _demo()
