"""pytest tests for plan_schema.py — PlanStep, AgentPlan, StepResult, validate_plan_quality."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from plan_schema import AgentPlan, PlanStep, StepResult, validate_plan_quality


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _step(
    n: int,
    *,
    tool: str | None = "get_weather",
    params: dict | None = None,
    depends_on: list[int] | None = None,
    expected: str = "Weather data for the city",
    description: str | None = None,
) -> PlanStep:
    return PlanStep(
        step_number=n,
        description=description or f"Step {n}: do something useful here",
        tool_name=tool,
        tool_params=params or ({"city": "Tokyo"} if tool else None),
        depends_on=depends_on or [],
        expected_output=expected,
    )


def _plan(*steps: PlanStep, question: str = "Test question") -> AgentPlan:
    return AgentPlan(user_question=question, steps=list(steps))


# ---------------------------------------------------------------------------
# PlanStep validation
# ---------------------------------------------------------------------------


def test_plan_step_minimal():
    step = _step(1)
    assert step.step_number == 1
    assert step.tool_name == "get_weather"


def test_plan_step_rejects_zero_step_number():
    with pytest.raises(ValidationError, match="greater_than_equal|greater than or equal"):
        PlanStep(
            step_number=0,
            description="Invalid step",
            expected_output="nothing",
        )


def test_plan_step_rejects_short_description():
    with pytest.raises(ValidationError, match="string_too_short|at least"):
        PlanStep(
            step_number=1,
            description="Too short",
            expected_output="something",
        )


def test_plan_step_rejects_short_expected_output():
    with pytest.raises(ValidationError, match="string_too_short|at least"):
        PlanStep(
            step_number=1,
            description="A valid description that is long enough",
            expected_output="ok",
        )


def test_plan_step_reasoning_step_has_no_tool():
    step = PlanStep(
        step_number=1,
        description="Synthesise all gathered information into a summary",
        tool_name=None,
        tool_params=None,
        depends_on=[],
        expected_output="A complete markdown summary table",
    )
    assert step.tool_name is None
    assert step.tool_params is None


# ---------------------------------------------------------------------------
# AgentPlan — sequential numbering
# ---------------------------------------------------------------------------


def test_plan_accepts_valid_sequential_steps():
    plan = _plan(_step(1), _step(2), _step(3))
    assert len(plan.steps) == 3


def test_plan_rejects_non_sequential_steps():
    with pytest.raises(ValidationError, match="sequential"):
        _plan(_step(1), _step(3))  # gap at 2


def test_plan_rejects_duplicate_step_numbers():
    with pytest.raises(ValidationError, match="sequential"):
        _plan(_step(1), _step(1))


def test_plan_rejects_steps_not_starting_at_one():
    with pytest.raises(ValidationError, match="sequential"):
        _plan(_step(2), _step(3))


# ---------------------------------------------------------------------------
# AgentPlan — dependency validation
# ---------------------------------------------------------------------------


def test_plan_accepts_valid_dependency():
    plan = _plan(_step(1), _step(2, depends_on=[1]))
    assert plan.steps[1].depends_on == [1]


def test_plan_rejects_dependency_on_nonexistent_step():
    with pytest.raises(ValidationError, match="does not exist"):
        _plan(_step(1), _step(2, depends_on=[99]))


def test_plan_rejects_forward_dependency():
    with pytest.raises(ValidationError, match="earlier steps"):
        _plan(_step(1, depends_on=[2]), _step(2))


def test_plan_accepts_multiple_valid_dependencies():
    plan = _plan(_step(1), _step(2), _step(3, depends_on=[1, 2]))
    assert plan.steps[2].depends_on == [1, 2]


# ---------------------------------------------------------------------------
# AgentPlan — circular dependency detection
# ---------------------------------------------------------------------------


def test_plan_rejects_self_dependency():
    # Self-dependency: step 1 depends on step 1 → forward dep check fires first
    with pytest.raises(ValidationError):
        _plan(_step(1, depends_on=[1]))


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


def test_step_result_success():
    result = StepResult(
        step_number=1,
        success=True,
        data={"temperature_c": 22},
        duration_ms=45,
    )
    assert result.success is True
    assert result.data == {"temperature_c": 22}


def test_step_result_failure():
    result = StepResult(
        step_number=2,
        success=False,
        error="City not found",
        duration_ms=10,
    )
    assert result.success is False
    assert result.error == "City not found"
    assert result.data is None


def test_step_result_duration_non_negative():
    with pytest.raises(ValidationError, match="greater_than_equal|greater than or equal"):
        StepResult(step_number=1, success=True, duration_ms=-1)


# ---------------------------------------------------------------------------
# validate_plan_quality
# ---------------------------------------------------------------------------


def test_quality_score_perfect_simple_plan():
    plan = _plan(
        _step(1, expected="Temperature, condition, and humidity for Tokyo"),
        PlanStep(
            step_number=2,
            description="Synthesise weather data into a user-facing answer",
            tool_name=None,
            tool_params=None,
            depends_on=[1],
            expected_output="A plain-English weather summary sentence",
        ),
    )
    result = validate_plan_quality(plan)
    assert result["score"] == 100
    assert result["warnings"] == []


def test_quality_warns_on_too_many_steps():
    steps = [_step(i, expected=f"Data for step {i} including key metric values") for i in range(1, 13)]
    plan = AgentPlan(user_question="big task", steps=steps)
    result = validate_plan_quality(plan)
    assert result["score"] < 100
    assert any("high" in w.lower() or "steps" in w.lower() for w in result["warnings"])


def test_quality_warns_on_vague_expected_output():
    # 8-char expected_output passes schema (min 5) but is caught as vague (< 15)
    plan = _plan(_step(1, expected="Raw data"))  # 8 chars
    result = validate_plan_quality(plan)
    assert any("vague" in w or "expected_output" in w for w in result["warnings"])
    assert result["score"] < 100


def test_quality_warns_on_missing_synthesis_step():
    plan = _plan(_step(1), _step(2))  # all tool steps, no reasoning
    result = validate_plan_quality(plan)
    assert any("synthesis" in w.lower() for w in result["warnings"])


def test_quality_warns_on_tool_step_without_params():
    step = PlanStep(
        step_number=1,
        description="Get weather without specifying parameters",
        tool_name="get_weather",
        tool_params=None,
        depends_on=[],
        expected_output="Weather data including temperature",
    )
    plan = _plan(step)
    result = validate_plan_quality(plan)
    assert any("tool_params" in w or "parameters" in w.lower() for w in result["warnings"])


def test_quality_score_is_clamped_to_zero():
    # Create a very bad plan to ensure score doesn't go negative
    steps = [
        PlanStep(
            step_number=i,
            description=f"Step {i} do something here at least ten chars",
            tool_name="get_weather",
            tool_params=None,  # missing params
            expected_output="Short",  # 5 chars passes schema but < 15 → deduction
            depends_on=[],
        )
        for i in range(1, 16)  # 15 steps → big deduction
    ]
    plan = AgentPlan(user_question="?", steps=steps)
    result = validate_plan_quality(plan)
    assert result["score"] >= 0
