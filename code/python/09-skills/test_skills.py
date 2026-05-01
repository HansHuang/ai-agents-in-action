"""pytest tests for the skill system.

All tests run without an LLM or real API. External services are mocked.

Test coverage:
  Skill Base (5 tests):
    1.  test_skill_execute_happy_path
    2.  test_skill_input_validation_blocks_bad_input
    3.  test_skill_fallback_on_tool_failure
    4.  test_skill_raises_without_fallback
    5.  test_skill_output_normalizer_transforms

  Skill Registry (5 tests):
    6.  test_registry_rejects_duplicate_skills
    7.  test_registry_resolves_dependencies
    8.  test_registry_detects_circular_dependencies
    9.  test_registry_detects_missing_dependency
    10. test_registry_find_by_tags

  Integration (3 tests):
    11. test_weather_skill_full_pipeline
    12. test_skill_dependency_chain
    13. test_skill_test_runner_reports_correctly

  SkilledAgent (2 tests):
    14. test_agent_builds_system_prompt_from_skills
    15. test_agent_uses_skill_tools
"""

from __future__ import annotations

import json
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from skill_base import (
    CircularDependencyError,
    MissingDependencyError,
    Skill,
    SkillInputError,
    SkillRegistry,
    SkillResult,
    SkillTest,
)
from skill_test_runner import SkillTestRunner
from skilled_agent import SkilledAgent
from skills.stock_analysis_skill import create_stock_analysis_skill
from skills.stock_price_skill import create_stock_price_skill
from skills.weather_skill import create_weather_skill

# ---------------------------------------------------------------------------
# Helpers — minimal skills for unit tests
# ---------------------------------------------------------------------------


def _make_echo_skill(**overrides) -> Skill:
    """A trivial skill that echoes its input — no external dependencies."""

    def echo(text: str) -> dict:
        return {"echo": text}

    return Skill(
        name=overrides.get("name", "echo"),
        description="Echo tool for testing",
        tool=echo,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"}
            },
            "required": ["text"],
        },
        prompt_fragment="Use echo when you want to test",
        fallback=lambda p, e: f"echo failed: {e}",
        **{k: v for k, v in overrides.items() if k not in ("name",)},
    )


def _make_failing_skill(raise_type=RuntimeError, has_fallback=True) -> Skill:
    """A skill whose tool always raises."""

    def always_fails(**kwargs) -> dict:
        raise raise_type("simulated failure")

    fb = (lambda p, e: "Service temporarily unavailable") if has_fallback else None

    return Skill(
        name="failing_skill",
        description="Always fails",
        tool=always_fails,
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string", "description": "ignored"}},
            "required": ["x"],
        },
        fallback=fb,
    )


def _mock_llm_tool_call(tool_name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(arguments)
    tc.id = f"call_{tool_name}"

    choice = MagicMock()
    choice.message.tool_calls = [tc]
    choice.message.content = None

    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_llm_text(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.tool_calls = None
    choice.message.content = content

    response = MagicMock()
    response.choices = [choice]
    return response


# ===========================================================================
# 1. Skill Base Tests
# ===========================================================================


class TestSkillExecute:
    def test_skill_execute_happy_path(self):
        """A skill with a working tool returns success=True with data."""
        skill = _make_echo_skill()
        result = skill.execute({"text": "hello"})

        assert result.success is True
        assert result.data == {"echo": "hello"}
        assert result.error is None
        assert result.execution_time_ms >= 0

    def test_skill_input_validation_blocks_bad_input(self):
        """A validator that rejects empty strings returns success=False, error_type='invalid_input'."""

        def reject_empty(params: dict) -> dict:
            if not params.get("text"):
                raise SkillInputError(
                    message="text must not be empty",
                    suggestion="Provide a non-empty string",
                )
            return params

        skill = _make_echo_skill(input_validator=reject_empty)
        result = skill.execute({"text": ""})

        assert result.success is False
        assert result.error_type == "invalid_input"
        assert "empty" in result.error
        assert result.suggestion == "Provide a non-empty string"

    def test_skill_fallback_on_tool_failure(self):
        """When the tool raises and a fallback is defined, success=False with fallback message."""
        skill = _make_failing_skill(has_fallback=True)
        result = skill.execute({"x": "trigger"})

        assert result.success is False
        assert result.error_type == "unavailable"
        assert "temporarily unavailable" in result.error

    def test_skill_raises_without_fallback(self):
        """When the tool raises and no fallback is defined, the exception propagates."""
        skill = _make_failing_skill(has_fallback=False)

        with pytest.raises(RuntimeError, match="simulated failure"):
            skill.execute({"x": "trigger"})

    def test_skill_output_normalizer_transforms(self):
        """The normaliser's output is what appears in result.data."""

        def add_processed(raw: dict) -> dict:
            return {**raw, "processed": True}

        skill = _make_echo_skill(output_normalizer=add_processed)
        result = skill.execute({"text": "world"})

        assert result.success is True
        assert result.data["processed"] is True
        assert result.data["echo"] == "world"


# ===========================================================================
# 2. Skill Registry Tests
# ===========================================================================


class TestSkillRegistry:
    def test_registry_rejects_duplicate_skills(self):
        """Registering the same skill name twice raises ValueError."""
        registry = SkillRegistry()
        registry.register(_make_echo_skill())

        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_echo_skill())

    def test_registry_resolves_dependencies(self):
        """A skill that depends on another returns both in topological order."""
        registry = SkillRegistry()

        dep = _make_echo_skill(name="base_skill")
        registry.register(dep)

        consumer = Skill(
            name="consumer_skill",
            description="Depends on base_skill",
            tool=lambda text: {"result": text},
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "text"}},
                "required": ["text"],
            },
            dependencies=["base_skill"],
        )
        registry.register(consumer)

        order = registry.resolve_dependencies(consumer)
        names = [s.name for s in order]

        assert names.index("base_skill") < names.index("consumer_skill")
        assert len(names) == 2

    def test_registry_detects_circular_dependencies(self):
        """Registering skills that form a cycle raises CircularDependencyError."""
        registry = SkillRegistry()

        skill_a = Skill(
            name="skill_a",
            description="A",
            tool=lambda **kw: {},
            parameters={"type": "object", "properties": {}},
            dependencies=["skill_b"],
        )
        skill_b = Skill(
            name="skill_b",
            description="B",
            tool=lambda **kw: {},
            parameters={"type": "object", "properties": {}},
            dependencies=["skill_a"],
        )

        # Register both — the second registration should detect the cycle
        registry._skills["skill_a"] = skill_a  # bypass register to set up cycle
        with pytest.raises(CircularDependencyError):
            registry.register(skill_b)

    def test_registry_detects_missing_dependency(self):
        """Registering a skill whose dependency isn't registered raises MissingDependencyError."""
        registry = SkillRegistry()

        orphan = Skill(
            name="orphan",
            description="Depends on nonexistent",
            tool=lambda **kw: {},
            parameters={"type": "object", "properties": {}},
            dependencies=["ghost_skill"],
        )

        with pytest.raises(MissingDependencyError, match="ghost_skill"):
            registry.register(orphan)

    def test_registry_find_by_tags(self):
        """find_by_tags returns only skills that carry at least one matching tag."""
        registry = SkillRegistry()

        weather = _make_echo_skill(name="w", tags=["weather", "real-time"])
        stock = _make_echo_skill(name="s", tags=["finance", "real-time"])
        other = _make_echo_skill(name="o", tags=["misc"])

        registry.register_many([weather, stock, other])

        weather_skills = registry.find_by_tags(["weather"])
        assert len(weather_skills) == 1
        assert weather_skills[0].name == "w"

        real_time_skills = registry.find_by_tags(["real-time"])
        assert len(real_time_skills) == 2

        no_match = registry.find_by_tags(["unknown-tag"])
        assert no_match == []


# ===========================================================================
# 3. Integration Tests
# ===========================================================================


class TestIntegration:
    def test_weather_skill_full_pipeline(self):
        """Weather skill: happy path, bad input, and unavailable city."""
        skill = create_weather_skill()

        # 1. Happy path — valid city
        result = skill.execute({"city": "Tokyo, JP"})
        assert result.success is True
        assert "Tokyo" in str(result.data)
        assert "°C" in str(result.data)
        assert "°F" in str(result.data)

        # 2. Invalid input — missing country code
        result = skill.execute({"city": "Tokyo"})
        assert result.success is False
        assert result.error_type == "invalid_input"
        assert "country code" in result.error

        # 3. Unavailable city — tool raises, fallback fires
        result = skill.execute({"city": "GhostCity, ZZ"})
        assert result.success is False
        assert result.error_type == "unavailable"
        assert "GhostCity" in result.error

    def test_skill_dependency_chain(self):
        """stock_analysis calls stock_price internally via the registry."""
        registry = SkillRegistry()
        registry.register(create_stock_price_skill())
        registry.register(create_stock_analysis_skill(registry))

        # Track calls to stock_price via mock.patch.object
        with mock.patch.object(
            registry, "execute", wraps=registry.execute
        ) as mock_exec:
            result = registry.execute("stock_analysis", {"ticker": "AAPL"})

        # stock_analysis was called once (outer call) and stock_price once (inner call)
        call_names = [c.args[0] for c in mock_exec.call_args_list]
        assert "stock_analysis" in call_names
        assert "stock_price" in call_names

        assert result.success is True
        data_str = str(result.data)
        assert "AAPL" in data_str
        assert "assessment" in data_str

    def test_skill_test_runner_reports_correctly(self):
        """SkillTestRunner counts 2 passing and 1 failing test correctly."""
        # Build a skill with a fixed outcome for each test case
        def echo(text: str) -> dict:
            return {"echo": text}

        skill_with_mixed_tests = Skill(
            name="mixed_test_skill",
            description="Test skill",
            tool=echo,
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "text"}},
                "required": ["text"],
            },
            fallback=lambda p, e: "failed",
            test_cases=[
                SkillTest(
                    input={"text": "hello"},
                    expect_output_contains=["hello"],
                ),  # passes
                SkillTest(
                    input={"text": "world"},
                    expect_output_contains=["world"],
                ),  # passes
                SkillTest(
                    input={"text": "test"},
                    expect_output_contains=["this-string-wont-appear"],
                ),  # fails
            ],
        )

        registry = SkillRegistry()
        registry.register(skill_with_mixed_tests)

        runner = SkillTestRunner(registry)
        report = runner.run_skill("mixed_test_skill")

        assert report.total_tests == 3
        assert report.passed == 2
        assert report.failed == 1
        assert len(report.failures) == 1
        assert "this-string-wont-appear" in report.failures[0]["reason"]


# ===========================================================================
# 4. SkilledAgent Tests
# ===========================================================================


class TestSkilledAgent:
    def test_agent_builds_system_prompt_from_skills(self):
        """System prompt contains prompt fragments from both loaded skills."""
        registry = SkillRegistry()
        registry.register(create_weather_skill())
        registry.register(create_stock_price_skill())

        agent = SkilledAgent(registry, client=MagicMock())
        agent.load_skills(["weather_reporting", "stock_price"])

        prompt = agent.build_system_prompt()

        # Both skill fragments should appear in the combined prompt
        assert "weather_reporting" in prompt
        assert "stock_price" in prompt
        # Key instructions from each skill's prompt_fragment
        assert "country code" in prompt
        assert "52-week" in prompt

    def test_agent_uses_skill_tools(self):
        """Agent calls weather skill, which runs through the full pipeline."""
        registry = SkillRegistry()
        registry.register(create_weather_skill())

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _mock_llm_tool_call("weather_reporting", {"city": "Tokyo, JP"}),
            _mock_llm_text("The weather in Tokyo is 22°C and partly cloudy."),
        ]

        agent = SkilledAgent(registry, client=mock_client)
        agent.load_skills(["weather_reporting"])

        result = agent.run("What's the weather in Tokyo?")

        # Final answer
        assert result.answer == "The weather in Tokyo is 22°C and partly cloudy."

        # One skill call
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["skill"] == "weather_reporting"
        assert result.tool_calls[0]["params"] == {"city": "Tokyo, JP"}

        # Skill pipeline ran successfully (validation + normalisation)
        assert len(result.skill_results) == 1
        skill_result: SkillResult = result.skill_results[0]["result"]
        assert skill_result.success is True
        # Normaliser ran: output contains the formatted temperature display
        assert "°C" in str(skill_result.data)
