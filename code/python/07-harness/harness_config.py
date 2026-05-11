"""
harness_config.py
=================
Complete configuration system for the ProductionHarness.

Provides:
  - HarnessConfig dataclass with every tuneable parameter
  - Loading from environment variables, YAML, JSON, or dict
  - Export to YAML and .env formats
  - Validation with human-readable warnings
  - Named presets: development, production, cost_optimized, high_security
  - Configuration diff between two instances

See: docs/07-harness-engineering/07-building-a-reliable-harness.md
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HarnessConfig
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """
    Complete harness configuration.

    Every parameter has a documented default.  Load from environment
    variables (``from_env()``), a YAML file (``from_yaml()``), a JSON file
    (``from_json()``), or a plain dictionary (``from_dict()``).

    Export with ``to_yaml()``, ``to_env_file()``, or ``to_dict()``.
    Validate with ``validate()``.  Compare with ``diff()``.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    agent_id: str = "production-agent-v1"
    system_prompt: str = ""
    tool_definitions: Optional[list[dict]] = None

    # ── Input guardrails ────────────────────────────────────────────────────
    max_input_length: int = 100_000
    min_input_length: int = 2
    rate_limit_rpm: int = 30
    rate_limit_rph: int = 500
    rate_limit_rpd: int = 5_000
    check_input_pii: bool = True
    check_input_content: bool = True
    check_input_injection: bool = True
    input_injection_threshold: str = "medium"  # "low" | "medium" | "high"

    # ── Routing ─────────────────────────────────────────────────────────────
    chat_model: str = "gpt-4o-mini"
    chat_max_tokens: int = 512
    chat_temperature: float = 0.7
    chat_timeout: int = 15
    rag_model: str = "gpt-4o"
    rag_max_tokens: int = 2_048
    rag_temperature: float = 0.3
    rag_timeout: int = 45
    agent_model: str = "gpt-4o"
    agent_max_tokens: int = 4_096
    agent_temperature: float = 0.2
    agent_timeout: int = 120
    agent_max_iterations: int = 10

    # ── Resilience ──────────────────────────────────────────────────────────
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 120.0
    circuit_breaker_window: float = 60.0
    llm_max_retries: int = 3
    llm_base_delay: float = 1.0
    llm_max_delay: float = 60.0
    llm_total_deadline: float = 300.0
    tool_max_retries: int = 2
    tool_base_delay: float = 0.5
    tool_timeout: int = 30

    # ── Output guardrails ───────────────────────────────────────────────────
    validate_output_schema: bool = True
    check_output_pii: bool = True
    check_output_safety: bool = True
    check_output_leakage: bool = True
    check_output_hallucination: bool = True
    check_output_facts: bool = False
    block_on_hallucination: bool = False
    hallucination_confidence_threshold: float = 0.7
    output_max_length: int = 50_000

    # ── Human-in-the-loop ───────────────────────────────────────────────────
    approval_channels: Optional[list[str]] = None        # ["dashboard", "slack", "email"]
    approval_high_value_refund_threshold: float = 500.0
    approval_external_communication: bool = True
    approval_database_modification: bool = True
    approval_default_timeout: float = 300.0
    approval_critical_timeout: float = 600.0

    # ── Observability ───────────────────────────────────────────────────────
    log_level: str = "INFO"
    trace_export_enabled: bool = True
    metrics_export_interval: int = 60
    alert_on_circuit_open: bool = True
    alert_on_fallback_exhaustion: bool = True
    alert_on_cost_spike: bool = True
    cost_spike_threshold_multiplier: float = 2.0

    # =========================================================================
    # Loading
    # =========================================================================

    @classmethod
    def from_env(cls) -> "HarnessConfig":
        """
        Load configuration from environment variables.

        Environment variable names follow the pattern UPPER_SNAKE_CASE.
        Missing variables fall back to the dataclass defaults.  A DEBUG
        log entry is emitted for each value sourced from the environment.
        """
        sourced: list[str] = []

        def _str(key: str, default: str) -> str:
            val = os.getenv(key)
            if val is not None:
                sourced.append(key)
                return val
            return default

        def _int(key: str, default: int) -> int:
            val = os.getenv(key)
            if val is not None:
                sourced.append(key)
                try:
                    return int(val)
                except ValueError:
                    logger.warning("Invalid int for %s=%r; using default %d", key, val, default)
            return default

        def _float(key: str, default: float) -> float:
            val = os.getenv(key)
            if val is not None:
                sourced.append(key)
                try:
                    return float(val)
                except ValueError:
                    logger.warning("Invalid float for %s=%r; using default %f", key, val, default)
            return default

        def _bool(key: str, default: bool) -> bool:
            val = os.getenv(key)
            if val is not None:
                sourced.append(key)
                return val.lower() in ("true", "1", "yes", "on")
            return default

        def _list(key: str, default: Optional[list]) -> Optional[list]:
            val = os.getenv(key)
            if val is not None:
                sourced.append(key)
                items = [v.strip() for v in val.split(",") if v.strip()]
                return items or None
            return default

        cfg = cls(
            # Identity
            agent_id=_str("AGENT_ID", "production-agent-v1"),
            system_prompt=_str("SYSTEM_PROMPT", ""),
            # Input
            max_input_length=_int("MAX_INPUT_LENGTH", 100_000),
            min_input_length=_int("MIN_INPUT_LENGTH", 2),
            rate_limit_rpm=_int("RATE_LIMIT_RPM", 30),
            rate_limit_rph=_int("RATE_LIMIT_RPH", 500),
            rate_limit_rpd=_int("RATE_LIMIT_RPD", 5_000),
            check_input_pii=_bool("CHECK_INPUT_PII", True),
            check_input_content=_bool("CHECK_INPUT_CONTENT", True),
            check_input_injection=_bool("CHECK_INPUT_INJECTION", True),
            input_injection_threshold=_str("INPUT_INJECTION_THRESHOLD", "medium"),
            # Routing
            chat_model=_str("CHAT_MODEL", "gpt-4o-mini"),
            chat_max_tokens=_int("CHAT_MAX_TOKENS", 512),
            chat_temperature=_float("CHAT_TEMPERATURE", 0.7),
            chat_timeout=_int("CHAT_TIMEOUT", 15),
            rag_model=_str("RAG_MODEL", "gpt-4o"),
            rag_max_tokens=_int("RAG_MAX_TOKENS", 2_048),
            rag_temperature=_float("RAG_TEMPERATURE", 0.3),
            rag_timeout=_int("RAG_TIMEOUT", 45),
            agent_model=_str("AGENT_MODEL", "gpt-4o"),
            agent_max_tokens=_int("AGENT_MAX_TOKENS", 4_096),
            agent_temperature=_float("AGENT_TEMPERATURE", 0.2),
            agent_timeout=_int("AGENT_TIMEOUT", 120),
            agent_max_iterations=_int("AGENT_MAX_ITERATIONS", 10),
            # Resilience
            circuit_breaker_threshold=_int("CIRCUIT_BREAKER_THRESHOLD", 5),
            circuit_breaker_recovery=_float("CIRCUIT_BREAKER_RECOVERY", 120.0),
            circuit_breaker_window=_float("CIRCUIT_BREAKER_WINDOW", 60.0),
            llm_max_retries=_int("LLM_MAX_RETRIES", 3),
            llm_base_delay=_float("LLM_BASE_DELAY", 1.0),
            llm_max_delay=_float("LLM_MAX_DELAY", 60.0),
            llm_total_deadline=_float("LLM_TOTAL_DEADLINE", 300.0),
            tool_max_retries=_int("TOOL_MAX_RETRIES", 2),
            tool_base_delay=_float("TOOL_BASE_DELAY", 0.5),
            tool_timeout=_int("TOOL_TIMEOUT", 30),
            # Output
            validate_output_schema=_bool("VALIDATE_OUTPUT_SCHEMA", True),
            check_output_pii=_bool("CHECK_OUTPUT_PII", True),
            check_output_safety=_bool("CHECK_OUTPUT_SAFETY", True),
            check_output_leakage=_bool("CHECK_OUTPUT_LEAKAGE", True),
            check_output_hallucination=_bool("CHECK_OUTPUT_HALLUCINATION", True),
            check_output_facts=_bool("CHECK_OUTPUT_FACTS", False),
            block_on_hallucination=_bool("BLOCK_ON_HALLUCINATION", False),
            hallucination_confidence_threshold=_float("HALLUCINATION_THRESHOLD", 0.7),
            output_max_length=_int("OUTPUT_MAX_LENGTH", 50_000),
            # Human-in-the-loop
            approval_channels=_list("APPROVAL_CHANNELS", None),
            approval_high_value_refund_threshold=_float("APPROVAL_REFUND_THRESHOLD", 500.0),
            approval_external_communication=_bool("APPROVAL_EXTERNAL_COMM", True),
            approval_database_modification=_bool("APPROVAL_DB_MODIFICATION", True),
            approval_default_timeout=_float("APPROVAL_DEFAULT_TIMEOUT", 300.0),
            approval_critical_timeout=_float("APPROVAL_CRITICAL_TIMEOUT", 600.0),
            # Observability
            log_level=_str("LOG_LEVEL", "INFO"),
            trace_export_enabled=_bool("TRACE_EXPORT_ENABLED", True),
            metrics_export_interval=_int("METRICS_EXPORT_INTERVAL", 60),
            alert_on_circuit_open=_bool("ALERT_ON_CIRCUIT_OPEN", True),
            alert_on_fallback_exhaustion=_bool("ALERT_ON_FALLBACK_EXHAUSTION", True),
            alert_on_cost_spike=_bool("ALERT_ON_COST_SPIKE", True),
            cost_spike_threshold_multiplier=_float("COST_SPIKE_MULTIPLIER", 2.0),
        )

        logger.debug("HarnessConfig: %d values sourced from env: %s", len(sourced), sourced)
        return cfg

    @classmethod
    def from_yaml(cls, filepath: str) -> "HarnessConfig":
        """
        Load configuration from a YAML file.

        The file may use a top-level ``harness:`` key (nested) or a flat
        structure.  Missing fields fall back to dataclass defaults.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required fields have invalid types.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required: pip install pyyaml") from exc

        with open(filepath) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")

        data = raw.get("harness", raw)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, filepath: str) -> "HarnessConfig":
        """
        Load configuration from a JSON file.

        Supports both ``{"harness": {...}}`` and flat ``{...}`` formats.
        """
        with open(filepath) as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a JSON object, got {type(raw).__name__}")

        data = raw.get("harness", raw)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessConfig":
        """
        Create a HarnessConfig from a flat or mildly nested dictionary.

        Nested keys (e.g. ``{"input": {"max_input_length": 50000}}``) are
        flattened automatically.  Unknown keys are silently ignored.
        """
        flat: dict[str, Any] = {}

        for key, value in data.items():
            if isinstance(value, dict):
                # Merge one level of nesting
                for sub_key, sub_val in value.items():
                    flat[sub_key] = sub_val
            else:
                flat[key] = value

        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in flat.items() if k in known}
        return cls(**filtered)

    # =========================================================================
    # Export
    # =========================================================================

    def to_yaml(self, filepath: str) -> None:
        """
        Export configuration to a YAML file with section comments.

        Creates (or overwrites) *filepath*.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required: pip install pyyaml") from exc

        sections = {
            "identity": ["agent_id", "system_prompt", "tool_definitions"],
            "input": [
                "max_input_length", "min_input_length",
                "rate_limit_rpm", "rate_limit_rph", "rate_limit_rpd",
                "check_input_pii", "check_input_content", "check_input_injection",
                "input_injection_threshold",
            ],
            "routing": [
                "chat_model", "chat_max_tokens", "chat_temperature", "chat_timeout",
                "rag_model", "rag_max_tokens", "rag_temperature", "rag_timeout",
                "agent_model", "agent_max_tokens", "agent_temperature",
                "agent_timeout", "agent_max_iterations",
            ],
            "resilience": [
                "circuit_breaker_threshold", "circuit_breaker_recovery",
                "circuit_breaker_window",
                "llm_max_retries", "llm_base_delay", "llm_max_delay",
                "llm_total_deadline",
                "tool_max_retries", "tool_base_delay", "tool_timeout",
            ],
            "output": [
                "validate_output_schema",
                "check_output_pii", "check_output_safety", "check_output_leakage",
                "check_output_hallucination", "check_output_facts",
                "block_on_hallucination", "hallucination_confidence_threshold",
                "output_max_length",
            ],
            "approval": [
                "approval_channels",
                "approval_high_value_refund_threshold",
                "approval_external_communication",
                "approval_database_modification",
                "approval_default_timeout",
                "approval_critical_timeout",
            ],
            "observability": [
                "log_level", "trace_export_enabled", "metrics_export_interval",
                "alert_on_circuit_open", "alert_on_fallback_exhaustion",
                "alert_on_cost_spike", "cost_spike_threshold_multiplier",
            ],
        }

        flat = asdict(self)
        nested: dict[str, Any] = {}
        for section, keys in sections.items():
            nested[section] = {k: flat[k] for k in keys if k in flat}

        with open(filepath, "w") as f:
            yaml.dump({"harness": nested}, f, default_flow_style=False, sort_keys=False)

    def to_env_file(self, filepath: str) -> None:
        """
        Export configuration as a shell-compatible .env file.

        Each line has the format ``KEY=value``.  Useful for documenting
        the full set of supported environment variables.
        """
        entries: list[tuple[str, Any]] = [
            ("AGENT_ID", self.agent_id),
            ("SYSTEM_PROMPT", self.system_prompt),
            # Input
            ("MAX_INPUT_LENGTH", self.max_input_length),
            ("MIN_INPUT_LENGTH", self.min_input_length),
            ("RATE_LIMIT_RPM", self.rate_limit_rpm),
            ("RATE_LIMIT_RPH", self.rate_limit_rph),
            ("RATE_LIMIT_RPD", self.rate_limit_rpd),
            ("CHECK_INPUT_PII", str(self.check_input_pii).lower()),
            ("CHECK_INPUT_CONTENT", str(self.check_input_content).lower()),
            ("CHECK_INPUT_INJECTION", str(self.check_input_injection).lower()),
            ("INPUT_INJECTION_THRESHOLD", self.input_injection_threshold),
            # Routing
            ("CHAT_MODEL", self.chat_model),
            ("CHAT_MAX_TOKENS", self.chat_max_tokens),
            ("CHAT_TEMPERATURE", self.chat_temperature),
            ("CHAT_TIMEOUT", self.chat_timeout),
            ("RAG_MODEL", self.rag_model),
            ("RAG_MAX_TOKENS", self.rag_max_tokens),
            ("RAG_TEMPERATURE", self.rag_temperature),
            ("RAG_TIMEOUT", self.rag_timeout),
            ("AGENT_MODEL", self.agent_model),
            ("AGENT_MAX_TOKENS", self.agent_max_tokens),
            ("AGENT_TEMPERATURE", self.agent_temperature),
            ("AGENT_TIMEOUT", self.agent_timeout),
            ("AGENT_MAX_ITERATIONS", self.agent_max_iterations),
            # Resilience
            ("CIRCUIT_BREAKER_THRESHOLD", self.circuit_breaker_threshold),
            ("CIRCUIT_BREAKER_RECOVERY", self.circuit_breaker_recovery),
            ("CIRCUIT_BREAKER_WINDOW", self.circuit_breaker_window),
            ("LLM_MAX_RETRIES", self.llm_max_retries),
            ("LLM_BASE_DELAY", self.llm_base_delay),
            ("LLM_MAX_DELAY", self.llm_max_delay),
            ("LLM_TOTAL_DEADLINE", self.llm_total_deadline),
            ("TOOL_MAX_RETRIES", self.tool_max_retries),
            ("TOOL_BASE_DELAY", self.tool_base_delay),
            ("TOOL_TIMEOUT", self.tool_timeout),
            # Output
            ("VALIDATE_OUTPUT_SCHEMA", str(self.validate_output_schema).lower()),
            ("CHECK_OUTPUT_PII", str(self.check_output_pii).lower()),
            ("CHECK_OUTPUT_SAFETY", str(self.check_output_safety).lower()),
            ("CHECK_OUTPUT_LEAKAGE", str(self.check_output_leakage).lower()),
            ("CHECK_OUTPUT_HALLUCINATION", str(self.check_output_hallucination).lower()),
            ("CHECK_OUTPUT_FACTS", str(self.check_output_facts).lower()),
            ("BLOCK_ON_HALLUCINATION", str(self.block_on_hallucination).lower()),
            ("HALLUCINATION_THRESHOLD", self.hallucination_confidence_threshold),
            ("OUTPUT_MAX_LENGTH", self.output_max_length),
            # Human-in-the-loop
            ("APPROVAL_CHANNELS", ",".join(self.approval_channels or [])),
            ("APPROVAL_REFUND_THRESHOLD", self.approval_high_value_refund_threshold),
            ("APPROVAL_EXTERNAL_COMM", str(self.approval_external_communication).lower()),
            ("APPROVAL_DB_MODIFICATION", str(self.approval_database_modification).lower()),
            ("APPROVAL_DEFAULT_TIMEOUT", self.approval_default_timeout),
            ("APPROVAL_CRITICAL_TIMEOUT", self.approval_critical_timeout),
            # Observability
            ("LOG_LEVEL", self.log_level),
            ("TRACE_EXPORT_ENABLED", str(self.trace_export_enabled).lower()),
            ("METRICS_EXPORT_INTERVAL", self.metrics_export_interval),
            ("ALERT_ON_CIRCUIT_OPEN", str(self.alert_on_circuit_open).lower()),
            ("ALERT_ON_FALLBACK_EXHAUSTION", str(self.alert_on_fallback_exhaustion).lower()),
            ("ALERT_ON_COST_SPIKE", str(self.alert_on_cost_spike).lower()),
            ("COST_SPIKE_MULTIPLIER", self.cost_spike_threshold_multiplier),
        ]

        with open(filepath, "w") as f:
            f.write("# Production Harness Configuration\n")
            f.write("# Generated by HarnessConfig.to_env_file()\n\n")
            for key, value in entries:
                f.write(f"{key}={value}\n")

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a flat dictionary."""
        return asdict(self)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> list[str]:
        """
        Validate the configuration and return a list of warning strings.

        An empty list means the configuration is consistent.  Warnings are
        not errors — the harness will still start — but they indicate
        settings that may behave unexpectedly in production.
        """
        warnings: list[str] = []

        # Timeout ordering
        if self.chat_timeout >= self.agent_timeout:
            warnings.append(
                f"chat_timeout ({self.chat_timeout}s) ≥ agent_timeout ({self.agent_timeout}s). "
                "Simple-chat requests will time out before agent tasks."
            )
        if self.rag_timeout >= self.agent_timeout:
            warnings.append(
                f"rag_timeout ({self.rag_timeout}s) ≥ agent_timeout ({self.agent_timeout}s)."
            )
        if self.agent_timeout >= self.llm_total_deadline:
            warnings.append(
                f"agent_timeout ({self.agent_timeout}s) ≥ llm_total_deadline "
                f"({self.llm_total_deadline}s). Deadline is reached before agent times out."
            )

        # Circuit breaker recovery vs deadline
        if self.circuit_breaker_recovery < self.llm_total_deadline:
            warnings.append(
                f"circuit_breaker_recovery ({self.circuit_breaker_recovery}s) < "
                f"llm_total_deadline ({self.llm_total_deadline}s). "
                "Consider increasing circuit_breaker_recovery so the circuit tests "
                "recovery after the deadline has elapsed."
            )

        # Approval timeouts
        if self.approval_default_timeout > self.approval_critical_timeout:
            warnings.append(
                "approval_default_timeout > approval_critical_timeout. "
                "Low-risk actions would wait longer than critical ones."
            )

        # Confidence threshold
        if not (0.0 <= self.hallucination_confidence_threshold <= 1.0):
            warnings.append(
                f"hallucination_confidence_threshold ({self.hallucination_confidence_threshold}) "
                "must be between 0 and 1."
            )

        # Rate limits vs timeouts
        if self.rate_limit_rpm > 60 and self.chat_timeout < 5:
            warnings.append(
                "High rate_limit_rpm with very short chat_timeout may cause cascading retries."
            )

        # Refund threshold
        if self.approval_high_value_refund_threshold < 0:
            warnings.append("approval_high_value_refund_threshold is negative.")

        # Cost spike multiplier
        if self.cost_spike_threshold_multiplier < 1.0:
            warnings.append(
                f"cost_spike_threshold_multiplier ({self.cost_spike_threshold_multiplier}) < 1. "
                "All requests would trigger cost-spike alerts."
            )

        # Input length
        if self.min_input_length >= self.max_input_length:
            warnings.append(
                f"min_input_length ({self.min_input_length}) ≥ max_input_length "
                f"({self.max_input_length}): no valid input could pass structural validation."
            )

        # Injection threshold
        valid_thresholds = {"low", "medium", "high"}
        if self.input_injection_threshold not in valid_thresholds:
            warnings.append(
                f"input_injection_threshold {self.input_injection_threshold!r} is not one of "
                f"{valid_thresholds}."
            )

        # block_on_hallucination without check
        if self.block_on_hallucination and not self.check_output_hallucination:
            warnings.append(
                "block_on_hallucination is True but check_output_hallucination is False. "
                "The block flag has no effect without hallucination detection."
            )

        return warnings

    # =========================================================================
    # Presets
    # =========================================================================

    @classmethod
    def development(cls) -> "HarnessConfig":
        """
        Relaxed settings for local development.

        Rate limits are high, timeouts are short, logging is verbose,
        expensive guardrails are disabled.
        """
        return cls(
            agent_id="dev-agent",
            rate_limit_rpm=120,
            rate_limit_rph=5_000,
            rate_limit_rpd=50_000,
            chat_timeout=10,
            rag_timeout=20,
            agent_timeout=30,
            llm_total_deadline=60.0,
            llm_max_retries=1,
            check_output_facts=False,
            block_on_hallucination=False,
            approval_default_timeout=30.0,
            approval_critical_timeout=60.0,
            log_level="DEBUG",
            metrics_export_interval=10,
        )

    @classmethod
    def production(cls) -> "HarnessConfig":
        """
        Balanced settings for a production deployment.

        All guardrails enabled, conservative rate limits, full resilience.
        Fact-checking is off (too expensive) but hallucination detection is on.
        """
        return cls(
            agent_id="production-agent-v1",
            rate_limit_rpm=30,
            rate_limit_rph=500,
            chat_timeout=15,
            rag_timeout=45,
            agent_timeout=120,
            llm_total_deadline=300.0,
            llm_max_retries=3,
            check_output_hallucination=True,
            check_output_facts=False,
            block_on_hallucination=False,
            approval_default_timeout=300.0,
            approval_critical_timeout=600.0,
            log_level="INFO",
        )

    @classmethod
    def cost_optimized(cls) -> "HarnessConfig":
        """
        Prioritise cost over capability.

        Uses cheaper models, shorter contexts, fewer retries.  Expensive
        guardrail layers (fact-check, hallucination) are disabled.
        """
        return cls(
            agent_id="cost-optimized-agent",
            chat_model="gpt-4o-mini",
            rag_model="gpt-4o-mini",
            agent_model="gpt-4o-mini",
            chat_max_tokens=256,
            rag_max_tokens=1_024,
            agent_max_tokens=2_048,
            agent_max_iterations=5,
            llm_max_retries=2,
            check_output_hallucination=False,
            check_output_facts=False,
            log_level="WARNING",
            metrics_export_interval=300,
        )

    @classmethod
    def high_security(cls) -> "HarnessConfig":
        """
        Maximum security posture.

        Every guardrail at maximum sensitivity.  Human approval required
        for all non-trivial actions.  Circuit breaker opens faster.
        """
        return cls(
            agent_id="high-security-agent",
            rate_limit_rpm=10,
            rate_limit_rph=200,
            rate_limit_rpd=2_000,
            input_injection_threshold="low",
            circuit_breaker_threshold=3,
            circuit_breaker_recovery=300.0,
            check_output_hallucination=True,
            block_on_hallucination=True,
            check_output_facts=True,
            approval_external_communication=True,
            approval_database_modification=True,
            approval_high_value_refund_threshold=0.01,  # Approve every refund
            approval_default_timeout=600.0,
            approval_critical_timeout=1_800.0,
            log_level="INFO",
            alert_on_circuit_open=True,
            alert_on_fallback_exhaustion=True,
            alert_on_cost_spike=True,
        )

    # =========================================================================
    # Diff
    # =========================================================================

    def diff(self, other: "HarnessConfig") -> str:
        """
        Return a human-readable diff of two :class:`HarnessConfig` instances.

        Lines show ``  key: <self_value> → <other_value>``.
        Returns ``  (no differences)`` when the two configs are equal.
        """
        self_flat = asdict(self)
        other_flat = asdict(other)
        lines: list[str] = []

        for key in sorted(set(self_flat) | set(other_flat)):
            a = self_flat.get(key)
            b = other_flat.get(key)
            if a != b:
                lines.append(f"  {key:45s}: {a!r:30} → {b!r}")

        return "\n".join(lines) if lines else "  (no differences)"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    import tempfile, os

    w = 70

    print("\n" + "=" * w)
    print("  HARNESS CONFIGURATION SYSTEM DEMO")
    print("=" * w)

    # 1. Load from environment variables
    print("\n── 1. Load from environment variables ──────────────────────────")
    os.environ["AGENT_ID"] = "env-demo-agent"
    os.environ["CHAT_MODEL"] = "gpt-4o-mini"
    os.environ["RATE_LIMIT_RPM"] = "60"
    cfg_env = HarnessConfig.from_env()
    print(f"  agent_id        : {cfg_env.agent_id}")
    print(f"  chat_model      : {cfg_env.chat_model}")
    print(f"  rate_limit_rpm  : {cfg_env.rate_limit_rpm}")

    # 2. Load from YAML
    print("\n── 2. Load from YAML ───────────────────────────────────────────")
    yaml_content = """
harness:
  agent_id: yaml-demo-agent
  routing:
    chat_model: gpt-4o-mini
    agent_model: gpt-4o
  resilience:
    llm_max_retries: 5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    try:
        cfg_yaml = HarnessConfig.from_yaml(yaml_path)
        print(f"  agent_id        : {cfg_yaml.agent_id}")
        print(f"  chat_model      : {cfg_yaml.chat_model}")
        print(f"  llm_max_retries : {cfg_yaml.llm_max_retries}")
    except ImportError:
        print("  (pyyaml not installed — skipping YAML demo)")
    finally:
        os.unlink(yaml_path)

    # 3. Export to YAML and .env
    print("\n── 3. Export formats ───────────────────────────────────────────")
    with tempfile.NamedTemporaryFile(suffix=".env", delete=False) as f:
        env_path = f.name
    HarnessConfig.production().to_env_file(env_path)
    with open(env_path) as f:
        lines = f.readlines()
    print(f"  .env file: {len(lines)} lines")
    for line in lines[:5]:
        print(f"    {line.rstrip()}")
    print(f"    …")
    os.unlink(env_path)

    # 4. Validation
    print("\n── 4. Validation ───────────────────────────────────────────────")
    bad_cfg = HarnessConfig(
        chat_timeout=200,          # > agent_timeout
        agent_timeout=120,
        block_on_hallucination=True,
        check_output_hallucination=False,   # contradictory
        approval_default_timeout=999,
        approval_critical_timeout=300,      # less than default
    )
    warnings = bad_cfg.validate()
    print(f"  {len(warnings)} warning(s):")
    for w_msg in warnings:
        print(f"    ⚠  {w_msg}")

    # 5. Presets
    presets = {
        "development":   HarnessConfig.development(),
        "production":    HarnessConfig.production(),
        "cost_optimized":HarnessConfig.cost_optimized(),
        "high_security": HarnessConfig.high_security(),
    }
    print("\n── 5. Presets ──────────────────────────────────────────────────")
    header = f"  {'Preset':<20} {'agent_model':<18} {'rpm':>6} {'retries':>7} {'block_hall':>10}"
    print(header)
    print("  " + "-" * 65)
    for name, cfg in presets.items():
        print(
            f"  {name:<20} {cfg.agent_model:<18} {cfg.rate_limit_rpm:>6} "
            f"{cfg.llm_max_retries:>7} {str(cfg.block_on_hallucination):>10}"
        )

    # 6. Diff
    print("\n── 6. Diff: production vs cost_optimized ───────────────────────")
    diff = HarnessConfig.production().diff(HarnessConfig.cost_optimized())
    print(diff)

    print("\n" + "=" * w + "\n")


if __name__ == "__main__":
    _run_demo()
