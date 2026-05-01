"""Tool definition quality validator.

Checks tool definitions against documented best practices and produces a
quality score (0–100) with actionable warnings and fix suggestions.

Checks performed:

NAMING
  - snake_case format
  - Starts with a recognised verb prefix (get_, search_, create_, …)
  - Unique within a tool set
  - Under 64 characters

DESCRIPTION
  - At least 20 characters
  - Mentions what the tool returns
  - Is not merely a restatement of the tool name

PARAMETERS
  - Every parameter has a non-empty description
  - Every parameter description includes an example value
  - Parameter names are not vague (data, input, param, …)
  - Required array is consistent with properties

Run as a script to validate the tools in tools.py::

    python tool_validator.py

Import as a module to validate arbitrary tool definitions::

    from tool_validator import ToolValidator
    results = ToolValidator().validate_set(my_tools)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERB_PREFIXES = (
    "get_", "search_", "create_", "update_", "delete_", "generate_",
    "list_", "fetch_", "send_", "check_", "validate_", "calculate_",
    "convert_", "format_", "submit_", "cancel_", "approve_", "reject_",
)

_VAGUE_NAMES = frozenset(
    {"data", "input", "output", "param", "value", "info", "result",
     "thing", "item", "object", "payload"}
)

_RETURN_WORDS = (
    "return", "returns", "gives", "provides", "fetches", "retrieves",
    "outputs", "yields", "contains", "responds with",
)

_EXAMPLE_INDICATORS = (
    "e.g.", "e.g ", "eg ", "ex:", "example:", "for example",
    "like ", "such as", "for instance",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Warning:
    category: str   # "naming" | "description" | "parameters"
    message: str
    suggestion: str
    penalty: int    # points deducted from 100


@dataclass
class ValidationResult:
    tool_name: str
    score: int
    warnings: list[Warning] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.score >= 70

    def summary(self) -> str:
        grade = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [
            f"  Tool: '{self.tool_name}'  Score: {self.score}/100  {grade}"
        ]
        for w in self.warnings:
            lines.append(f"  [{w.category.upper()}] {w.message}")
            lines.append(f"    → Fix: {w.suggestion}")
        if not self.warnings:
            lines.append("  No issues found.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ToolValidator:
    """Validates tool definitions and returns scored :class:`ValidationResult` objects."""

    def validate(
        self,
        tool_def: dict,
        all_tool_names: Optional[list[str]] = None,
    ) -> ValidationResult:
        """Validate a single tool definition.

        Args:
            tool_def:       Full OpenAI format ``{"type": "function", "function": {...}}``
                            or the inner ``{"name": …, "description": …, "parameters": …}``
                            dict directly.
            all_tool_names: All names in the tool set, used to detect duplicates.

        Returns:
            A :class:`ValidationResult` with a 0–100 score and list of warnings.
        """
        func = tool_def.get("function", tool_def)
        name: str = func.get("name", "")
        description: str = func.get("description", "")
        parameters: dict = func.get("parameters", {})
        properties: dict = parameters.get("properties", {})
        required: list = parameters.get("required", [])

        warnings: list[Warning] = []

        # ------------------------------------------------------------------
        # NAMING
        # ------------------------------------------------------------------
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            warnings.append(
                Warning(
                    category="naming",
                    message=f"Name '{name}' is not snake_case",
                    suggestion="Use lowercase letters, digits, and underscores only. "
                               "Example: 'get_weather'",
                    penalty=10,
                )
            )

        if not any(name.startswith(v) for v in _VERB_PREFIXES):
            warnings.append(
                Warning(
                    category="naming",
                    message=f"Name '{name}' does not start with a recognised verb prefix",
                    suggestion=(
                        f"Prefix with one of: "
                        + ", ".join(_VERB_PREFIXES[:6])
                        + ", …"
                    ),
                    penalty=10,
                )
            )

        if len(name) > 64:
            warnings.append(
                Warning(
                    category="naming",
                    message=f"Name '{name}' is {len(name)} characters; maximum is 64",
                    suggestion="Shorten the tool name",
                    penalty=5,
                )
            )

        if all_tool_names and all_tool_names.count(name) > 1:
            warnings.append(
                Warning(
                    category="naming",
                    message=f"Name '{name}' appears more than once in the tool set",
                    suggestion="Each tool must have a unique name",
                    penalty=15,
                )
            )

        # ------------------------------------------------------------------
        # DESCRIPTION
        # ------------------------------------------------------------------
        if len(description) < 20:
            warnings.append(
                Warning(
                    category="description",
                    message=f"Description is too short ({len(description)} chars; minimum 20)",
                    suggestion=(
                        "Explain what the tool does, what it returns, "
                        "and when the model should call it"
                    ),
                    penalty=15,
                )
            )

        if not any(w in description.lower() for w in _RETURN_WORDS):
            warnings.append(
                Warning(
                    category="description",
                    message="Description does not mention what the tool returns",
                    suggestion=(
                        "Add a sentence like "
                        "'Returns temperature (C/F), humidity, and conditions.'"
                    ),
                    penalty=10,
                )
            )

        # Flag descriptions that merely restate the name
        name_words = set(name.replace("_", " ").lower().split())
        desc_lower = description.lower().split()
        if name_words and name_words.issubset(set(desc_lower)) and len(desc_lower) <= len(name_words) + 3:
            warnings.append(
                Warning(
                    category="description",
                    message="Description appears to just restate the tool name",
                    suggestion=(
                        "Explain the tool's behaviour, parameters, return value, "
                        "and when to use it vs. similar tools"
                    ),
                    penalty=10,
                )
            )

        # ------------------------------------------------------------------
        # PARAMETERS
        # ------------------------------------------------------------------
        for param_name, spec in properties.items():
            param_desc: str = spec.get("description", "")

            if not param_desc:
                warnings.append(
                    Warning(
                        category="parameters",
                        message=f"Parameter '{param_name}' has no description",
                        suggestion=(
                            f"Add a description with the format and an example value, "
                            f"e.g. \"City name with country code. Example: 'Tokyo, JP'\""
                        ),
                        penalty=8,
                    )
                )
            elif not any(ind in param_desc.lower() for ind in _EXAMPLE_INDICATORS):
                warnings.append(
                    Warning(
                        category="parameters",
                        message=f"Parameter '{param_name}' description has no example value",
                        suggestion=(
                            f"Append an example to the description, "
                            f"e.g. \"… e.g. 'Tokyo, JP'\""
                        ),
                        penalty=5,
                    )
                )

            if param_name.lower() in _VAGUE_NAMES:
                warnings.append(
                    Warning(
                        category="parameters",
                        message=f"Parameter name '{param_name}' is too vague",
                        suggestion=(
                            "Use a specific name that describes the content: "
                            "'city_name', 'order_id', 'user_email', etc."
                        ),
                        penalty=8,
                    )
                )

        # Required array consistency
        for req_name in required:
            if req_name not in properties:
                warnings.append(
                    Warning(
                        category="parameters",
                        message=(
                            f"Required parameter '{req_name}' is listed in 'required' "
                            "but missing from 'properties'"
                        ),
                        suggestion=f"Add '{req_name}' to the 'properties' object",
                        penalty=15,
                    )
                )

        total_penalty = sum(w.penalty for w in warnings)
        score = max(0, 100 - total_penalty)
        return ValidationResult(tool_name=name, score=score, warnings=warnings)

    def validate_set(
        self, tools: list[dict], all_tool_names: Optional[list[str]] = None
    ) -> list[ValidationResult]:
        """Validate a list of tool definitions, checking for name uniqueness."""
        names = all_tool_names or [
            t.get("function", t).get("name", "") for t in tools
        ]
        return [self.validate(t, names) for t in tools]


# ---------------------------------------------------------------------------
# Demo: run against tools.py
# ---------------------------------------------------------------------------


def _demo() -> None:
    try:
        from tools import TOOLS  # noqa: PLC0415
    except ImportError:
        print("Run this script from code/python/03-agent-loop/", file=sys.stderr)
        sys.exit(1)

    validator = ToolValidator()
    results = validator.validate_set(TOOLS)

    print("=" * 60)
    print("Tool Definition Quality Report")
    print("=" * 60)

    all_pass = True
    for result in results:
        print(result.summary())
        print()
        if not result.passed:
            all_pass = False

    overall = sum(r.score for r in results) // len(results) if results else 0
    print(f"Overall average score: {overall}/100")
    print("=" * 60)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    _demo()
