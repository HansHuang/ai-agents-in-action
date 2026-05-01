"""Pydantic models and validation for agent plans.

Defines PlanStep, AgentPlan, and StepResult — the data contracts used by
PlanAndExecuteAgent. Includes dependency validation (missing references,
circular dependency detection) and a quality scorer.

See docs/02-the-agent-loop/03-planning-strategies.md — "Plan-and-Execute"
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single step in an agent execution plan."""

    step_number: int = Field(..., ge=1, description="1-based position in the plan.")
    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Human-readable description of what this step does.",
    )
    tool_name: Optional[str] = Field(
        None,
        description="Name of the tool to call. None for reasoning/synthesis steps.",
    )
    tool_params: Optional[dict[str, Any]] = Field(
        None,
        description="Arguments to pass to the tool. None when tool_name is None.",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="Step numbers this step must wait for before executing.",
    )
    expected_output: str = Field(
        ...,
        min_length=5,
        description="Brief description of what a successful result looks like.",
    )


class AgentPlan(BaseModel):
    """A complete agent execution plan with dependency tracking."""

    user_question: str = Field(..., description="The original user query.")
    steps: list[PlanStep] = Field(..., min_length=1)
    estimated_tool_calls: int = Field(
        0, ge=0, description="Total number of tool calls expected."
    )

    @model_validator(mode="after")
    def validate_steps_numbered_sequentially(self) -> "AgentPlan":
        """Step numbers must be 1, 2, 3, … with no gaps."""
        numbers = sorted(s.step_number for s in self.steps)
        expected = list(range(1, len(self.steps) + 1))
        if numbers != expected:
            raise ValueError(
                f"Step numbers must be sequential starting at 1. "
                f"Got: {numbers}, expected: {expected}"
            )
        return self

    @model_validator(mode="after")
    def validate_dependencies_exist(self) -> "AgentPlan":
        """All depends_on references must point to existing step numbers."""
        valid = {s.step_number for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in valid:
                    raise ValueError(
                        f"Step {step.step_number} depends on step {dep}, "
                        f"which does not exist. Valid steps: {sorted(valid)}"
                    )
                if dep >= step.step_number:
                    raise ValueError(
                        f"Step {step.step_number} depends on step {dep}, "
                        f"but a step can only depend on earlier steps."
                    )
        return self

    @model_validator(mode="after")
    def validate_no_circular_dependencies(self) -> "AgentPlan":
        """Detect dependency cycles using DFS."""
        adj: dict[int, list[int]] = {s.step_number: list(s.depends_on) for s in self.steps}

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[int, int] = {n: WHITE for n in adj}

        def dfs(node: int) -> None:
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color[neighbor] == GRAY:
                    raise ValueError(
                        f"Circular dependency detected involving step {node}"
                    )
                if color[neighbor] == WHITE:
                    dfs(neighbor)
            color[node] = BLACK

        for node in list(adj.keys()):
            if color[node] == WHITE:
                dfs(node)
        return self


# ---------------------------------------------------------------------------
# Step result model
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """The outcome of executing a single plan step."""

    step_number: int
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Quality scorer
# ---------------------------------------------------------------------------


def validate_plan_quality(plan: AgentPlan) -> dict[str, Any]:
    """Score an AgentPlan against best-practice heuristics.

    Args:
        plan: A validated AgentPlan instance.

    Returns:
        A dict with keys:
            score (int 0-100),
            warnings (list[str]),
            suggestions (list[str]).
    """
    warnings: list[str] = []
    suggestions: list[str] = []
    deductions = 0

    # Total step count
    n = len(plan.steps)
    if n > 10:
        warnings.append(f"Plan has {n} steps, which is high. Consider simplifying.")
        suggestions.append("Break the task into sub-tasks handled by separate agents.")
        deductions += min(20, (n - 10) * 2)

    # Per-step checks
    reasoning_steps = 0
    for step in plan.steps:
        if step.tool_name is None:
            reasoning_steps += 1

        # Expected output quality
        if len(step.expected_output) < 15:
            warnings.append(
                f"Step {step.step_number}: expected_output is too vague "
                f"('{step.expected_output}'). Add more detail."
            )
            suggestions.append(
                f"Step {step.step_number}: describe the key fields you expect "
                "in the tool result."
            )
            deductions += 5

        # Steps with tool_name but no tool_params
        if step.tool_name is not None and not step.tool_params:
            warnings.append(
                f"Step {step.step_number} calls tool '{step.tool_name}' "
                "but has no tool_params. The LLM will have to guess parameters."
            )
            deductions += 5

        # Unnecessary sequential dependencies
        if len(step.depends_on) > 1 and step.step_number - 1 in step.depends_on:
            # Depends on the immediate predecessor AND something else — flag if
            # the additional dependency seems implied by ordering anyway
            pass  # Would need semantic analysis; skip for heuristic check

    # At least one reasoning/synthesis step
    if reasoning_steps == 0 and n > 1:
        warnings.append("No synthesis step found. Consider adding a final reasoning step.")
        suggestions.append(
            "Add a final step with tool_name=None to synthesise all results."
        )
        deductions += 10

    score = max(0, 100 - deductions)
    return {"score": score, "warnings": warnings, "suggestions": suggestions}
