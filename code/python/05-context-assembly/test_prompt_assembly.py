"""Tests for dynamic prompt assembly modules.

Covers:
- PromptAssembler: template rendering, conditional sections, context sources,
  priority ordering, token budget enforcement, missing variable errors
- PromptLibrary: YAML loading, version tracking, hot-reload, validation
- ConditionEngine: equality, list membership, compound AND/OR, explain

Run offline (no API calls):
    pytest test_prompt_assembly.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from condition_engine import ConditionEngine
from prompt_assembler import (
    MissingVariableError,
    PromptAssembler,
    format_rag_results,
    format_user_profile,
    format_tool_results,
    format_conversation_summary,
    format_business_rules,
)
from prompt_library import PromptLibrary, RenderedPrompt
from context_budget import count_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_assembler() -> PromptAssembler:
    """Return a PromptAssembler with a minimal template pre-registered."""
    a = PromptAssembler()
    a.register_template("base", "Hello {name}, your plan is {plan}.")
    return a


def _rag_docs(n: int = 2) -> list[dict]:
    return [
        {"text": f"Policy doc {i}: " + "word " * 50, "score": 0.9, "metadata": {"source": f"doc-{i}.md"}}
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# PromptAssembler tests
# ---------------------------------------------------------------------------

class TestPromptAssembler:

    # 1. Base template renders with variables
    def test_base_template_renders_with_variables(self):
        a = _make_assembler()
        result = a.assemble("base", {"name": "Alice", "plan": "premium"})
        assert result == "Hello Alice, your plan is premium."

    # 2. Conditional section included when condition is True
    def test_conditional_section_included_when_true(self):
        a = _make_assembler()
        a.register_section(
            "premium",
            "You qualify for priority support.",
            condition=lambda v: v.get("plan") == "premium",
        )
        a.register_template("full", "Role: {role}\n{sections}")
        result = a.assemble("full", {"role": "engineer", "plan": "premium"})
        assert "You qualify for priority support." in result

    # 3. Conditional section excluded when condition is False
    def test_conditional_section_excluded_when_false(self):
        a = PromptAssembler()
        a.register_template("full", "Role: {role}\n{sections}")
        a.register_section(
            "premium",
            "You qualify for priority support.",
            condition=lambda v: v.get("plan") == "premium",
        )
        result = a.assemble("full", {"role": "engineer", "plan": "free"})
        assert "priority support" not in result

    # 4. Multiple conditions evaluated — only matching sections included
    def test_multiple_conditions_evaluated(self):
        a = PromptAssembler()
        a.register_template("t", "{sections}")
        a.register_section("s1", "SECTION_ONE",   condition=lambda v: v.get("a") is True)
        a.register_section("s2", "SECTION_TWO",   condition=lambda v: v.get("b") is True)
        a.register_section("s3", "SECTION_THREE", condition=lambda v: v.get("c") is True)

        result = a.assemble("t", {"a": True, "b": True, "c": False})
        assert "SECTION_ONE"   in result
        assert "SECTION_TWO"   in result
        assert "SECTION_THREE" not in result

    # 5. Context sources are formatted
    def test_context_sources_formatted(self):
        a = PromptAssembler()
        a.register_template("t", "{context}")
        a.register_source_formatter("rag", format_rag_results, priority=1)

        docs = [{"text": "Refund policy: 30 days.", "score": 0.95,
                  "metadata": {"source": "policy.md"}}]
        result = a.assemble("t", {}, {"rag": docs})
        assert "policy.md" in result
        assert "Refund policy: 30 days." in result

    # 6. Context sources sorted by priority (highest first)
    def test_context_sources_sorted_by_priority(self):
        a = PromptAssembler()
        a.register_template("t", "{context}")
        a.register_source_formatter("low",    lambda d: d, priority=1)
        a.register_source_formatter("medium", lambda d: d, priority=2)
        a.register_source_formatter("high",   lambda d: d, priority=3)

        result = a.assemble("t", {}, {
            "low":    "LOW_CONTENT",
            "medium": "MEDIUM_CONTENT",
            "high":   "HIGH_CONTENT",
        })
        high_pos   = result.index("HIGH_CONTENT")
        medium_pos = result.index("MEDIUM_CONTENT")
        low_pos    = result.index("LOW_CONTENT")
        assert high_pos < medium_pos < low_pos

    # 7. Token budget enforced — low-priority sources dropped first
    def test_token_budget_enforced(self):
        a = PromptAssembler()
        a.register_template("t", "{context}")
        # high-priority small source
        a.register_source_formatter("important", lambda d: d, priority=10)
        # low-priority very large source
        big_text = "word " * 2_000   # ~2000 tokens
        a.register_source_formatter("noise", lambda d: d, priority=1)

        result = a.assemble_with_budget(
            "t",
            {},
            {"important": "KEEP_THIS", "noise": big_text},
            max_tokens=50,
        )
        # Budget forces noise to be dropped
        assert "KEEP_THIS" in result
        assert count_tokens(result) <= 60   # small slack for formatting overhead

    # 8. Missing template variable raises MissingVariableError
    def test_missing_template_variable_raises(self):
        a = PromptAssembler()
        a.register_template("t", "Hello {name}, your score is {score}.")
        with pytest.raises(MissingVariableError):
            a.assemble("t", {"name": "Alice"})  # missing "score"

    # 9. Registering unknown template name raises KeyError
    def test_unknown_template_raises(self):
        a = PromptAssembler()
        with pytest.raises(KeyError):
            a.assemble("nonexistent", {})

    # 10. get_available_variables returns unique ordered variable names
    def test_get_available_variables(self):
        a = PromptAssembler()
        a.register_template("t", "Hello {name}! Plan: {plan}. Name again: {name}.")
        result = a.get_available_variables("t")
        assert result == ["name", "plan"]  # unique, in order of first appearance

    # 11. Source formatter with max_tokens truncates content
    def test_source_max_tokens_truncates(self):
        a = PromptAssembler()
        a.register_template("t", "{context}")
        large_text = "word " * 1_000
        a.register_source_formatter("src", lambda d: d, priority=1, max_tokens=20)
        result = a.assemble("t", {}, {"src": large_text})
        # The formatted source should be within max_tokens + header overhead
        assert count_tokens(result) < 100   # well under the original 1000+ tokens

    # 12. Sections injected even when template lacks {sections} placeholder
    def test_sections_appended_when_no_placeholder(self):
        a = PromptAssembler()
        a.register_template("t", "You are a bot.")
        a.register_section("extra", "BE HELPFUL.", condition=lambda v: True)
        result = a.assemble("t", {})
        assert "BE HELPFUL." in result


# ---------------------------------------------------------------------------
# Built-in formatter tests
# ---------------------------------------------------------------------------

class TestBuiltinFormatters:

    def test_format_rag_results_citation(self):
        docs = [
            {"text": "Return policy is 30 days.", "score": 0.92,
             "metadata": {"source": "policy.md"}},
        ]
        out = format_rag_results(docs)
        assert "policy.md" in out
        assert "92%" in out
        assert "Return policy is 30 days." in out

    def test_format_rag_results_empty(self):
        assert "no documents" in format_rag_results([])

    def test_format_user_profile(self):
        out = format_user_profile({"name": "Alice", "plan": "premium", "location": "NYC"})
        assert "Alice" in out
        assert "premium" in out
        assert "NYC" in out

    def test_format_tool_results_success_failure(self):
        results = [
            {"tool_name": "order_lookup", "success": True,  "summary": "Found order #123"},
            {"tool_name": "refund_api",   "success": False, "summary": "API timeout"},
        ]
        out = format_tool_results(results)
        assert "✓ order_lookup" in out
        assert "✗ refund_api" in out

    def test_format_conversation_summary(self):
        out = format_conversation_summary("User asked about refunds.")
        assert "User asked about refunds." in out

    def test_format_business_rules(self):
        out = format_business_rules(["No refunds after 90 days.", "Premium gets 2× limit."])
        assert "No refunds after 90 days." in out
        assert "Premium gets 2× limit." in out


# ---------------------------------------------------------------------------
# PromptLibrary tests
# ---------------------------------------------------------------------------

YAML_TEMPLATE_1 = """\
name: greet
version: 1.0.0
description: Simple greeting template

template: |
  Hello, {customer_name}! You are on the {plan} plan.

sections:
  premium_note:
    condition: "plan == 'premium'"
    content: |
      You have access to premium features.
"""

YAML_TEMPLATE_2 = """\
name: farewell
version: 1.0.0
description: Farewell template

template: |
  Goodbye, {customer_name}. Have a great day!
"""

YAML_TEMPLATE_3 = """\
name: info
version: 2.1.0
description: Info template

template: |
  Here is some info for {customer_name}.
"""

YAML_UPDATED_GREET = """\
name: greet
version: 1.1.0
description: Simple greeting template — updated

template: |
  Hi there, {customer_name}! You are on the {plan} plan.

sections:
  premium_note:
    condition: "plan == 'premium'"
    content: |
      You have access to premium features.
"""


@pytest.fixture()
def prompts_dir(tmp_path: Path) -> Path:
    (tmp_path / "greet.yaml").write_text(YAML_TEMPLATE_1)
    (tmp_path / "farewell.yaml").write_text(YAML_TEMPLATE_2)
    (tmp_path / "info.yaml").write_text(YAML_TEMPLATE_3)
    return tmp_path


# 13. Loads all YAML templates from directory
def test_loads_all_yaml_templates(prompts_dir: Path):
    lib = PromptLibrary(str(prompts_dir))
    assert set(lib.templates.keys()) == {"greet", "farewell", "info"}


# 14. Template version is tracked in RenderedPrompt
def test_template_version_tracked(prompts_dir: Path):
    lib = PromptLibrary(str(prompts_dir))
    result = lib.render("info", {"customer_name": "Alice"})
    assert isinstance(result, RenderedPrompt)
    assert result.template_version == "2.1.0"
    assert result.template_name == "info"


# 15. Hot-reload picks up updated template
def test_hot_reload_updates_template(prompts_dir: Path):
    lib = PromptLibrary(str(prompts_dir))

    r1 = lib.render("greet", {"customer_name": "Alice", "plan": "free"})
    assert "Hello" in r1.rendered_text  # v1.0.0 greeting

    # Update the file on disk
    (prompts_dir / "greet.yaml").write_text(YAML_UPDATED_GREET)
    lib.reload()

    r2 = lib.render("greet", {"customer_name": "Alice", "plan": "free"})
    assert "Hi there" in r2.rendered_text  # v1.1.0 greeting
    assert r2.template_version == "1.1.0"


# 16. Sections included / excluded based on condition
def test_sections_evaluated_on_render(prompts_dir: Path):
    lib = PromptLibrary(str(prompts_dir))

    r_premium = lib.render("greet", {"customer_name": "Alice", "plan": "premium"})
    assert "premium_note" in r_premium.sections_included
    assert "premium features" in r_premium.rendered_text

    r_free = lib.render("greet", {"customer_name": "Bob", "plan": "free"})
    assert "premium_note" not in r_free.sections_included
    assert "premium features" not in r_free.rendered_text


# 17. validate_all returns empty list for well-formed templates
def test_validate_all_clean(prompts_dir: Path):
    lib = PromptLibrary(str(prompts_dir))
    issues = lib.validate_all()
    assert issues == []


# 18. validate_all detects missing section condition
def test_validate_detects_missing_section_condition(tmp_path: Path):
    yaml_bad = """\
name: bad_template
version: 1.0.0
description: Template with broken section

template: |
  Hello {name}.

sections:
  broken_section:
    condition: ""
    content: |
      This section has no condition.
"""
    (tmp_path / "bad.yaml").write_text(yaml_bad)
    lib = PromptLibrary(str(tmp_path))
    issues = lib.validate_all()
    assert any("broken_section" in issue for issue in issues)


# 19. validate_all detects invalid condition syntax
def test_validate_detects_invalid_condition(tmp_path: Path):
    yaml_bad = """\
name: bad_cond
version: 1.0.0
description: Template with invalid condition

template: |
  Hello {name}.

sections:
  broken:
    condition: "plan is premium"
    content: |
      Something.
"""
    (tmp_path / "bad_cond.yaml").write_text(yaml_bad)
    lib = PromptLibrary(str(tmp_path))
    issues = lib.validate_all()
    assert any("broken" in issue for issue in issues)


# 20. diff shows changes between two loaded versions
def test_diff_shows_changes(prompts_dir: Path):
    lib = PromptLibrary(str(prompts_dir))  # loads v1.0.0

    # Update to v1.1.0 and reload
    (prompts_dir / "greet.yaml").write_text(YAML_UPDATED_GREET)
    lib.reload()

    diff = lib.diff("greet", "1.0.0", "1.1.0")
    assert "-" in diff   # removals
    assert "+" in diff   # additions
    assert "Hello" in diff or "Hi there" in diff


# 21. get() raises KeyError for unknown template
def test_get_unknown_template_raises(prompts_dir: Path):
    lib = PromptLibrary(str(prompts_dir))
    with pytest.raises(KeyError):
        lib.get("nonexistent_template")


# ---------------------------------------------------------------------------
# ConditionEngine tests
# ---------------------------------------------------------------------------

class TestConditionEngine:
    """Tests for the standalone ConditionEngine DSL evaluator."""

    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = ConditionEngine()

    # 13. Simple equality
    def test_simple_equality_true(self):
        assert self.engine.evaluate("plan == 'premium'", {"plan": "premium"}) is True

    def test_simple_equality_false(self):
        assert self.engine.evaluate("plan == 'premium'", {"plan": "free"}) is False

    def test_simple_inequality(self):
        assert self.engine.evaluate("plan != 'premium'", {"plan": "free"}) is True

    # 14. List membership
    def test_list_membership_true(self):
        assert self.engine.evaluate(
            "country in ['DE', 'FR']", {"country": "DE"}
        ) is True

    def test_list_membership_false(self):
        assert self.engine.evaluate(
            "country in ['DE', 'FR']", {"country": "US"}
        ) is False

    def test_not_in_membership(self):
        assert self.engine.evaluate(
            "country not_in ['US', 'CA']", {"country": "DE"}
        ) is True

    # 15. Numeric comparisons
    def test_greater_than(self):
        assert self.engine.evaluate("score > 0.7", {"score": 0.85}) is True
        assert self.engine.evaluate("score > 0.7", {"score": 0.5}) is False

    def test_less_than(self):
        assert self.engine.evaluate("count < 5", {"count": 3}) is True
        assert self.engine.evaluate("count < 5", {"count": 7}) is False

    # 16. Exists operator
    def test_exists_true(self):
        assert self.engine.evaluate("token exists", {"token": "abc"}) is True

    def test_exists_false_for_missing_key(self):
        assert self.engine.evaluate("missing_key exists", {}) is False

    # 17. Contains operator
    def test_contains(self):
        assert self.engine.evaluate(
            "email contains '@enterprise'", {"email": "alice@enterprise.com"}
        ) is True

    # 18. Nested variable access
    def test_nested_variable_access(self):
        assert self.engine.evaluate(
            "user.plan == 'enterprise'",
            {"user": {"plan": "enterprise"}},
        ) is True

    # 19. Compound AND — True only when both sub-conditions are True
    def test_compound_and_both_true(self):
        assert self.engine.evaluate(
            "plan == 'premium' AND country == 'US'",
            {"plan": "premium", "country": "US"},
        ) is True

    def test_compound_and_one_false(self):
        assert self.engine.evaluate(
            "plan == 'premium' AND country == 'US'",
            {"plan": "premium", "country": "DE"},
        ) is False

    # 20. Compound OR — True when either sub-condition is True
    def test_compound_or_first_true(self):
        assert self.engine.evaluate(
            "country == 'DE' OR country == 'FR'",
            {"country": "DE"},
        ) is True

    def test_compound_or_second_true(self):
        assert self.engine.evaluate(
            "country == 'DE' OR country == 'FR'",
            {"country": "FR"},
        ) is True

    def test_compound_or_both_false(self):
        assert self.engine.evaluate(
            "country == 'DE' OR country == 'FR'",
            {"country": "US"},
        ) is False

    # 21. Explain shows step-by-step evaluation
    def test_explain_shows_evaluation(self):
        explanation = self.engine.explain(
            "plan == 'premium' AND score > 0.5",
            {"plan": "premium", "score": 0.85},
        )
        assert "plan == 'premium'" in explanation
        assert "score > 0.5" in explanation
        assert "True" in explanation
        assert "Overall" in explanation

    def test_explain_shows_false_branch(self):
        explanation = self.engine.explain(
            "plan == 'premium' AND country == 'US'",
            {"plan": "premium", "country": "DE"},
        )
        assert "False" in explanation

    # 22. evaluate_all returns matching names
    def test_evaluate_all(self):
        conditions = {
            "is_premium": "plan == 'premium'",
            "is_eu":      "country in ['DE', 'FR', 'ES']",
            "high_score": "score > 0.9",
        }
        matched = self.engine.evaluate_all(
            conditions, {"plan": "premium", "country": "DE", "score": 0.5}
        )
        assert set(matched) == {"is_premium", "is_eu"}

    # 23. Validation catches invalid condition
    def test_validate_condition_valid(self):
        ok, msg = self.engine.validate_condition("plan == 'premium'")
        assert ok is True
        assert msg == ""

    def test_validate_condition_invalid(self):
        ok, msg = self.engine.validate_condition("plan is premium")
        assert ok is False
        assert msg != ""

    def test_validate_condition_no_operator(self):
        ok, msg = self.engine.validate_condition("just_a_key")
        assert ok is False
