"""Standalone skill test runner — no LLM, no API keys, no agent loop.

Tests skills in isolation: just the tool, the validator, the normaliser,
and the fallback. This is the "boundary between AI engineering and software
engineering" described in the Skills chapter.

See: docs/02-the-agent-loop/05-skills-composing-capabilities.md
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

from skill_base import Skill, SkillRegistry, TestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TestReport
# ---------------------------------------------------------------------------


@dataclass
class TestReport:
    """Aggregated result of running one skill's test suite."""

    skill_name: str
    total_tests: int
    passed: int
    failed: int
    failures: list[dict] = field(default_factory=list)
    execution_time_ms: int = 0

    def __str__(self) -> str:
        icon = "PASS" if self.failed == 0 else "FAIL"
        line = f"[{icon}] {self.skill_name}: {self.passed}/{self.total_tests} passed"
        for f in self.failures:
            line += f"\n       - input={f['test_input']!r}: {f['reason']}"
        return line


# ---------------------------------------------------------------------------
# SkillTestRunner
# ---------------------------------------------------------------------------


class SkillTestRunner:
    """Runs all test cases for registered skills.

    No LLM. No API keys. No agent loop.
    Skills are executed directly through their validate → run → normalise
    pipeline; everything external is expected to be mocked or use mock data.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    # ------------------------------------------------------------------
    # Per-skill run
    # ------------------------------------------------------------------

    def run_skill(self, skill_name: str) -> TestReport:
        """Run all test cases for a specific skill.

        Args:
            skill_name: Name of the registered skill to test.

        Returns:
            TestReport with pass/fail counts and failure details.
        """
        skill: Skill = self.registry.get(skill_name)
        start = time.monotonic()

        if not skill.test_cases:
            return TestReport(
                skill_name=skill_name,
                total_tests=0,
                passed=0,
                failed=0,
            )

        results: list[TestResult] = skill.run_tests()
        elapsed = int((time.monotonic() - start) * 1000)

        failures = [
            {"test_input": r.test_input, "reason": r.reason}
            for r in results
            if not r.passed
        ]

        return TestReport(
            skill_name=skill_name,
            total_tests=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=len(failures),
            failures=failures,
            execution_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Full-registry run
    # ------------------------------------------------------------------

    def run_all(self) -> list[TestReport]:
        """Run test cases for every skill in the registry.

        Returns:
            One TestReport per registered skill, in registration order.
        """
        return [self.run_skill(name) for name in list(self.registry._skills)]

    # ------------------------------------------------------------------
    # Integration test
    # ------------------------------------------------------------------

    def run_integration_test(
        self,
        skill_names: list[str],
        scenario: dict,
    ) -> bool:
        """Test multiple skills working together in sequence.

        Args:
            skill_names: Skills that are in scope for this test.
            scenario:    Dict with keys:
                           "steps": list of {"skill": str, "input": dict}
                           "expect_final_output_contains": list[str]

        Returns:
            True if all steps succeed and the final output contains all
            expected substrings; False otherwise.

        Example::

            runner.run_integration_test(
                skill_names=["weather_reporting"],
                scenario={
                    "steps": [
                        {"skill": "weather_reporting",
                         "input": {"city": "Tokyo, JP"}},
                    ],
                    "expect_final_output_contains": ["Tokyo", "22"],
                },
            )
        """
        steps = scenario.get("steps", [])
        expect_contains: list[str] = scenario.get("expect_final_output_contains", [])
        last_result = None

        for step in steps:
            name = step["skill"]
            skill = self.registry.get(name)
            result = skill.execute(step["input"])

            if not result.success:
                logger.error(
                    "Integration step failed: skill=%s error=%s", name, result.error
                )
                return False

            last_result = result

        if not expect_contains:
            return True

        final_str = str(last_result.data if last_result else "")
        for keyword in expect_contains:
            if keyword not in final_str:
                logger.error(
                    "Integration test: '%s' not found in output", keyword
                )
                return False

        return True


# ---------------------------------------------------------------------------
# Demo / CI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from skills.weather_skill import create_weather_skill
    from skills.stock_price_skill import create_stock_price_skill
    from skills.stock_analysis_skill import create_stock_analysis_skill

    registry = SkillRegistry()
    registry.register(create_weather_skill())
    registry.register(create_stock_price_skill())
    registry.register(create_stock_analysis_skill(registry))

    runner = SkillTestRunner(registry)

    print("=== Running all skill tests ===")
    reports = runner.run_all()
    for report in reports:
        print(report)

    print()
    print("=== Integration test: weather pipeline ===")
    passed = runner.run_integration_test(
        skill_names=["weather_reporting"],
        scenario={
            "steps": [
                {"skill": "weather_reporting", "input": {"city": "Tokyo, JP"}},
            ],
            "expect_final_output_contains": ["Tokyo", "22"],
        },
    )
    print(f"Integration test: {'PASSED' if passed else 'FAILED'}")

    all_failed = sum(r.failed for r in reports)
    if all_failed > 0 or not passed:
        print(f"\n{all_failed} unit test(s) failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    main()
