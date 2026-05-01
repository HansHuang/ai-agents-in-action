"""Skill base class and registry for composable agent capabilities.

A Skill bundles a tool with:
  - Input validation   (runs before the tool)
  - Output normalisation  (runs after the tool)
  - A fallback  (runs when the tool raises an exception)
  - A prompt fragment  (injected into the agent system prompt)
  - Test cases  (runnable without an LLM or API key)

See: docs/02-the-agent-loop/05-skills-composing-capabilities.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SkillInputError(Exception):
    """Raised by an input validator when parameters are invalid.

    Attributes:
        message:    Human-readable description of what is wrong.
        suggestion: Optional hint the model can use to self-correct.
        fix_action: Optional machine-readable action identifier.
    """

    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        fix_action: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.fix_action = fix_action


class CircularDependencyError(ValueError):
    """Raised when registration would introduce a dependency cycle."""


class MissingDependencyError(ValueError):
    """Raised when a skill declares a dependency that is not registered."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SkillResult:
    """The outcome of executing a skill."""

    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    #: "invalid_input" | "unavailable" | "internal"
    error_type: Optional[str] = None
    suggestion: Optional[str] = None
    execution_time_ms: int = 0


@dataclass
class TestResult:
    """The outcome of running a single SkillTest."""

    test_input: dict
    passed: bool
    reason: str = ""
    result: Optional[SkillResult] = None


@dataclass
class SkillTest:
    """A single test case for a skill.

    Attributes:
        input:                  Parameters passed to skill.execute().
        expect_success:         True if the skill should succeed.
        expect_output_contains: Substrings that must appear in str(result.data).
        expect_fallback:        True if the tool should raise and the fallback
                                message should be returned (error_type="unavailable").
    """

    input: dict
    expect_success: bool = True
    expect_output_contains: Optional[list[str]] = None
    expect_fallback: bool = False


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A composable, testable unit of agent capability.

    Bundles a tool with prompt instructions, input validation,
    output normalisation, fallback behaviour, and test cases.
    """

    name: str
    """Unique identifier, e.g. 'weather_reporting'."""

    description: str
    """When the agent should use this skill."""

    tool: Callable[..., Any]
    """The function the skill wraps."""

    parameters: dict
    """OpenAI function-calling parameter schema."""

    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    prompt_fragment: Optional[str] = None
    """Injected into the agent system prompt when this skill is loaded."""

    input_validator: Optional[Callable[[dict], dict]] = None
    """Validates and optionally corrects params before the tool runs."""

    output_normalizer: Optional[Callable[[Any], dict]] = None
    """Normalises the tool's raw return value into a consistent format."""

    fallback: Optional[Callable[[dict, Exception], str]] = None
    """Called when the tool raises; returns a user-facing message."""

    dependencies: list[str] = field(default_factory=list)
    """Names of skills that must be registered before this one."""

    test_cases: list[SkillTest] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Core execution pipeline
    # ------------------------------------------------------------------

    def execute(self, params: dict) -> SkillResult:
        """Execute the full validate → run → normalise → fallback pipeline.

        Returns:
            SkillResult with success=True and data on success.
            SkillResult with success=False and error details on failure.

        Raises:
            Any exception raised by the tool if no fallback is defined.
        """
        start = time.monotonic()

        try:
            logger.debug("[%s] execute params=%r", self.name, params)

            # 1. Validate input
            if self.input_validator:
                params = self.input_validator(params)

            # 2. Run tool
            raw = self.tool(**params)

            # 3. Normalise output
            if self.output_normalizer:
                raw = self.output_normalizer(raw)

            elapsed = int((time.monotonic() - start) * 1000)
            logger.info("[%s] success elapsed_ms=%d", self.name, elapsed)
            return SkillResult(success=True, data=raw, execution_time_ms=elapsed)

        except SkillInputError as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info("[%s] invalid_input: %s", self.name, exc.message)
            return SkillResult(
                success=False,
                error=exc.message,
                error_type="invalid_input",
                suggestion=exc.suggestion,
                execution_time_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning("[%s] tool error: %s", self.name, exc)
            if self.fallback:
                msg = self.fallback(params, exc)
                return SkillResult(
                    success=False,
                    error=msg,
                    error_type="unavailable",
                    execution_time_ms=elapsed,
                )
            raise

    # ------------------------------------------------------------------
    # Schema / prompt helpers
    # ------------------------------------------------------------------

    def get_openai_schema(self) -> dict:
        """Return the OpenAI function-calling schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def get_prompt_fragment(self) -> str:
        """Return the prompt fragment to inject into the agent system message."""
        if self.prompt_fragment:
            return self.prompt_fragment.strip()
        return f"Use {self.name} when: {self.description}"

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def run_tests(self) -> list[TestResult]:
        """Run all test cases in isolation. No agent or LLM required.

        Returns:
            A list of TestResult, one per test case.
        """
        results: list[TestResult] = []

        for test in self.test_cases:
            result = self.execute(test.input)
            passed = True
            reason = ""

            if test.expect_fallback:
                # Tool should have raised and been caught by the fallback
                if result.success or result.error_type == "invalid_input":
                    passed = False
                    reason = (
                        f"Expected fallback (unavailable) but got "
                        f"success={result.success}, error_type={result.error_type!r}"
                    )

            elif not test.expect_success:
                # Expect any failure
                if result.success:
                    passed = False
                    reason = "Expected failure but skill reported success"
                elif test.expect_output_contains:
                    combined = (result.error or "") + " " + (result.suggestion or "")
                    for kw in test.expect_output_contains:
                        if kw not in combined:
                            passed = False
                            reason = f"Expected '{kw}' in error/suggestion"
                            break

            else:
                # Expect success
                if not result.success:
                    passed = False
                    reason = f"Expected success but got error: {result.error}"
                elif test.expect_output_contains:
                    data_str = str(result.data or "")
                    for kw in test.expect_output_contains:
                        if kw not in data_str:
                            passed = False
                            reason = f"Expected '{kw}' in output data"
                            break

            results.append(
                TestResult(
                    test_input=test.input,
                    passed=passed,
                    reason=reason,
                    result=result,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of warnings about this skill's definition.

        Checks completeness without executing the skill. Use before
        registering skills in production to catch missing components.

        Returns:
            A list of warning strings (empty means no issues found).
        """
        warnings: list[str] = []

        if not self.description:
            warnings.append(f"[{self.name}] has no description")

        props = self.parameters.get("properties", {})
        if not props:
            warnings.append(f"[{self.name}] parameters has no properties defined")
        else:
            for param_name, schema in props.items():
                if not schema.get("description"):
                    warnings.append(
                        f"[{self.name}] parameter '{param_name}' has no description"
                    )

        if self.fallback is None:
            warnings.append(
                f"[{self.name}] no fallback defined — "
                "tool failures will propagate as exceptions"
            )

        if self.prompt_fragment is None:
            warnings.append(
                f"[{self.name}] no prompt_fragment — "
                "the agent will receive only the skill description"
            )

        return warnings


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Manages skill registration, discovery, and dependency resolution."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Register a skill. Validates dependencies and detects cycles.

        Raises:
            ValueError: If a skill with the same name is already registered.
            MissingDependencyError: If a declared dependency is not registered.
            CircularDependencyError: If registration would create a cycle.
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' is already registered")

        # Temporarily add so cycle detection can inspect the full graph
        self._skills[skill.name] = skill
        try:
            self._check_dependencies(skill)
        except Exception:
            del self._skills[skill.name]
            raise

    def register_many(self, skills: list[Skill]) -> None:
        """Register multiple skills in the given order."""
        for skill in skills:
            self.register(skill)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Skill:
        """Return a registered skill by name.

        Raises:
            KeyError: If no skill with that name is registered.
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' is not registered")
        return self._skills[name]

    def find_by_tags(self, tags: list[str]) -> list[Skill]:
        """Return all skills that share at least one of the given tags."""
        tag_set = set(tags)
        return [s for s in self._skills.values() if tag_set & set(s.tags)]

    def get_all_schemas(self) -> list[dict]:
        """Return OpenAI function-calling schemas for all registered skills."""
        return [s.get_openai_schema() for s in self._skills.values()]

    def get_combined_prompt(self, skill_names: list[str]) -> str:
        """Return concatenated prompt fragments for the given skills."""
        parts: list[str] = []
        for name in skill_names:
            skill = self.get(name)
            frag = skill.get_prompt_fragment()
            parts.append(f"### {skill.name}\n{frag}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, params: dict) -> SkillResult:
        """Execute a registered skill by name."""
        return self.get(name).execute(params)

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def resolve_dependencies(self, skill: Skill) -> list[Skill]:
        """Return the skill and all its transitive dependencies, topologically sorted.

        Dependencies appear before the skills that require them.

        Raises:
            MissingDependencyError: If any dependency is not registered.
            CircularDependencyError: If a cycle is detected.
        """
        order: list[str] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise CircularDependencyError(
                    f"Circular dependency detected involving '{name}'"
                )
            if name in visited:
                return
            visiting.add(name)
            s = self._skills.get(name)
            if s is None:
                raise MissingDependencyError(
                    f"Dependency '{name}' is not registered"
                )
            for dep in s.dependencies:
                visit(dep)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        visit(skill.name)
        return [self._skills[n] for n in order]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_dependencies(self, skill: Skill) -> None:
        """Verify declared dependencies exist and there are no cycles."""
        for dep in skill.dependencies:
            if dep not in self._skills:
                raise MissingDependencyError(
                    f"Skill '{skill.name}' depends on '{dep}' which is not registered"
                )

        # Full cycle detection via DFS from the new skill
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise CircularDependencyError(
                    f"Circular dependency detected involving '{name}'"
                )
            if name in visited:
                return
            visiting.add(name)
            for dep in self._skills[name].dependencies:
                if dep in self._skills:
                    visit(dep)
            visiting.remove(name)
            visited.add(name)

        visit(skill.name)
