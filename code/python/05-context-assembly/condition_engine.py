"""Condition Engine — evaluate DSL conditions for dynamic prompt sections.

Supports a simple, safe condition DSL for including or excluding prompt
sections based on runtime variables.  No ``eval()`` is used; conditions are
parsed with regex-based tokenisation.

Simple conditions::

    "plan == 'premium'"
    "user.plan in ['premium', 'enterprise']"
    "sentiment_score > 0.7"
    "user.email contains '@enterprise'"
    "conversation_history exists"

Compound conditions::

    "plan == 'premium' AND country == 'US'"
    "country == 'DE' OR country == 'FR'"

See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operator table
# ---------------------------------------------------------------------------

_OP_TABLE: dict[str, Any] = {
    "eq":       lambda a, b: a == b,
    "neq":      lambda a, b: a != b,
    "in":       lambda a, b: a in b,
    "not_in":   lambda a, b: a not in b,
    "gt":       lambda a, b: a > b,
    "lt":       lambda a, b: a < b,
    "gte":      lambda a, b: a >= b,
    "lte":      lambda a, b: a <= b,
    "contains": lambda a, b: b in a,
    "exists":   lambda a, _: a is not None,
}

# Operator regex patterns — order matters: longer/multi-word first
_OP_PATTERNS: list[tuple[str, str]] = [
    (r"not\s+in",     "not_in"),
    (r"not_in",       "not_in"),
    (r">=",           "gte"),
    (r"<=",           "lte"),
    (r"==",           "eq"),
    (r"!=",           "neq"),
    (r">",            "gt"),
    (r"<",            "lt"),
    (r"\bcontains\b", "contains"),
    (r"\bexists\b",   "exists"),
    (r"\bin\b",       "in"),
]

_AND_SPLITTER = re.compile(r"\bAND\b")
_OR_SPLITTER  = re.compile(r"\bOR\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nested(variables: dict, key: str) -> Any:
    """Resolve a dotted key path: ``'user.plan'`` → ``variables['user']['plan']``."""
    obj: Any = variables
    for part in key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _parse_rhs(raw: str | None) -> Any:
    """Parse the RHS of a condition into a Python literal (str, int, float, list …)."""
    if raw is None:
        return None
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _parse_atom(atom_str: str) -> tuple[str, str, Any]:
    """Parse one atomic condition: ``'key op value'`` → ``(key, op_name, value)``.

    Args:
        atom_str: A single condition expression, e.g. ``"user.plan == 'premium'"``.

    Returns:
        Tuple of *(key, op_name, parsed_value)*.

    Raises:
        ValueError: If the condition cannot be parsed.
    """
    atom_str = atom_str.strip()
    for op_re, op_name in _OP_PATTERNS:
        pattern = rf"^([\w.]+)\s+{op_re}(?:\s+(.+))?$"
        m = re.match(pattern, atom_str)
        if m:
            key = m.group(1)
            raw_val = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else None
            return key, op_name, _parse_rhs(raw_val)
    raise ValueError(f"Cannot parse condition atom: {atom_str!r}")


def _evaluate_atom(key: str, op: str, value: Any, variables: dict) -> bool:
    """Evaluate a single atomic condition against the variables dict."""
    left = _get_nested(variables, key)
    fn = _OP_TABLE.get(op)
    if fn is None:
        raise ValueError(f"Unknown operator: {op!r}")
    try:
        return bool(fn(left, value))
    except TypeError as exc:
        raise TypeError(
            f"Type error evaluating '{key} {op} {value!r}': {exc} "
            f"(got {type(left).__name__})"
        ) from exc


# ---------------------------------------------------------------------------
# ConditionEngine
# ---------------------------------------------------------------------------

class ConditionEngine:
    """Evaluate conditions for including or excluding dynamic prompt sections.

    Uses a simple, **safe** DSL — no ``eval()`` is invoked.

    Example::

        engine = ConditionEngine()

        engine.evaluate("plan == 'premium'", {"plan": "premium"})
        # True

        engine.evaluate(
            "plan == 'premium' AND country == 'DE'",
            {"plan": "premium", "country": "US"},
        )
        # False — country doesn't match

        print(engine.explain(
            "plan == 'premium' AND score > 0.5",
            {"plan": "premium", "score": 0.85},
        ))
    """

    # Expose for introspection / documentation
    OPERATORS = _OP_TABLE

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, condition: str, variables: dict) -> bool:
        """Evaluate a condition string against variables.

        Supported formats:

        - ``"query_type == 'billing'"``
        - ``"user.plan in ['premium', 'enterprise']"``
        - ``"user.country not_in ['US', 'CA']"``
        - ``"sentiment_score > 0.7"``
        - ``"user.email contains '@enterprise'"``
        - ``"conversation_history exists"``
        - ``"plan == 'premium' AND country == 'US'"``
        - ``"country == 'DE' OR country == 'FR'"``

        Args:
            condition: Condition string to evaluate.
            variables: Variable dict to evaluate against.

        Returns:
            ``True`` if the condition matches, ``False`` otherwise.
        """
        condition = condition.strip()
        # OR has lower precedence than AND.
        # Split by OR first, then by AND within each branch.
        for or_branch in _OR_SPLITTER.split(condition):
            and_atoms = _AND_SPLITTER.split(or_branch)
            try:
                branch_true = all(
                    _evaluate_atom(*_parse_atom(a.strip()), variables)
                    for a in and_atoms
                )
            except Exception as exc:
                logger.warning("Condition evaluation error in %r: %s", condition, exc)
                branch_true = False
            if branch_true:
                return True
        return False

    def evaluate_all(
        self,
        conditions: dict[str, str],
        variables: dict,
    ) -> list[str]:
        """Evaluate multiple named conditions.

        Args:
            conditions: Mapping of condition name → condition string.
            variables:  Variable dict to evaluate against.

        Returns:
            Names of conditions that evaluated to ``True``.
        """
        return [
            name
            for name, cond in conditions.items()
            if self.evaluate(cond, variables)
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_condition(self, condition: str) -> tuple[bool, str]:
        """Validate a condition string for syntax errors.

        Does **not** evaluate the condition — only checks parseability.

        Args:
            condition: Condition string to validate.

        Returns:
            ``(True, "")`` if valid; ``(False, error_message)`` if not.
        """
        try:
            for or_branch in _OR_SPLITTER.split(condition):
                for atom in _AND_SPLITTER.split(or_branch):
                    _parse_atom(atom.strip())
            return True, ""
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def explain(self, condition: str, variables: dict) -> str:
        """Explain why a condition evaluated to True or False.

        Useful for debugging "why wasn't this section included?".

        Args:
            condition: Condition string to explain.
            variables: Variable dict to evaluate against.

        Returns:
            Human-readable multi-line explanation string.
        """
        lines: list[str] = [f"Evaluating: {condition!r}"]

        or_branches = _OR_SPLITTER.split(condition)
        branch_results: list[bool] = []

        for b_idx, or_branch in enumerate(or_branches, 1):
            or_branch = or_branch.strip()
            and_atoms = _AND_SPLITTER.split(or_branch)

            if len(or_branches) > 1:
                lines.append(f"\n  [OR branch {b_idx}] {or_branch!r}")

            atom_results: list[bool] = []
            for atom_str in and_atoms:
                atom_str = atom_str.strip()
                try:
                    key, op, value = _parse_atom(atom_str)
                    left = _get_nested(variables, key)
                    result = _evaluate_atom(key, op, value, variables)
                    atom_results.append(result)
                    check = "✓" if result else "✗"
                    lines.append(
                        f"    {atom_str!r:40s}  →  "
                        f"{key}={left!r}  {op}  {value!r}  [{check}]"
                    )
                except Exception as exc:
                    atom_results.append(False)
                    lines.append(f"    {atom_str!r}  →  ERROR: {exc}")

            branch_ok = all(atom_results) if atom_results else False
            branch_results.append(branch_ok)
            if len(or_branches) > 1:
                lines.append(
                    f"    Branch result: {'True ✓' if branch_ok else 'False ✗'}"
                )

        final = any(branch_results)
        lines.append(f"\nOverall: {'True ✓' if final else 'False ✗'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    engine = ConditionEngine()

    variables = {
        "plan": "premium",
        "country": "DE",
        "score": 0.85,
        "user": {"plan": "enterprise", "email": "alice@enterprise.com"},
    }

    print("=== Simple Conditions ===")
    simple_cases: list[tuple[str, bool]] = [
        ("plan == 'premium'",                    True),
        ("plan == 'free'",                       False),
        ("plan != 'free'",                       True),
        ("country in ['DE', 'FR', 'ES']",        True),
        ("country in ['US', 'CA']",              False),
        ("country not_in ['US', 'CA']",          True),
        ("score > 0.7",                          True),
        ("score < 0.5",                          False),
        ("plan exists",                          True),
        ("missing_key exists",                   False),
        ("user.plan == 'enterprise'",            True),
        ("user.email contains '@enterprise'",    True),
    ]
    for cond, expected in simple_cases:
        result = engine.evaluate(cond, variables)
        mark = "✓" if result == expected else "✗"
        print(f"  {mark} {cond!r:<45s} → {result}")

    print("\n=== Compound Conditions ===")
    compound_cases: list[tuple[str, bool]] = [
        ("plan == 'premium' AND country == 'DE'",  True),
        ("plan == 'premium' AND country == 'US'",  False),
        ("country == 'US' OR country == 'DE'",     True),
        ("country == 'US' OR country == 'CA'",     False),
    ]
    for cond, expected in compound_cases:
        result = engine.evaluate(cond, variables)
        mark = "✓" if result == expected else "✗"
        print(f"  {mark} {cond!r:<45s} → {result}")

    print("\n=== Explain ===")
    print(engine.explain(
        "plan == 'premium' AND score > 0.5",
        {"plan": "premium", "score": 0.85},
    ))

    print("\n=== Validation ===")
    valid_cases = [
        ("plan == 'premium'",    True),
        ("score > 0.7",          True),
        ("country in ['DE']",    True),
        ("plan is premium",      False),  # 'is' not in DSL
        ("plan",                 False),  # no operator
    ]
    for cond, expect_valid in valid_cases:
        ok, msg = engine.validate_condition(cond)
        mark = "✓" if ok == expect_valid else "✗"
        status = "valid" if ok else f"invalid: {msg}"
        print(f"  {mark} {cond!r:<30s} → {status}")

    print("\n=== evaluate_all ===")
    conditions = {
        "is_premium":       "plan == 'premium'",
        "is_eu":            "country in ['DE', 'FR', 'ES', 'IT']",
        "high_score":       "score > 0.9",     # False — score is 0.85
    }
    matched = engine.evaluate_all(conditions, variables)
    print(f"  Matched: {matched}")  # ['is_premium', 'is_eu']


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _demo()
