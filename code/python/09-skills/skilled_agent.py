"""SkilledAgent — an agent that loads capabilities from a SkillRegistry.

Instead of raw tools the agent uses Skills. Each Skill provides:
  - Its own OpenAI function-calling schema (no manual tool definitions)
  - Its own prompt fragment (injected into the system prompt)
  - Validation, normalisation, and fallback (transparent to the agent loop)

The agent code does not change when skills are added or removed.

See: docs/02-the-agent-loop/05-skills-composing-capabilities.md
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from skill_base import Skill, SkillRegistry, SkillResult

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

_BASE_PROMPT = """\
You are a helpful assistant with access to skills.
Answer questions using the skills provided. Always use a skill when the
question falls within its scope — never guess data that a skill can provide.
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SkillAgentResult:
    """The result of a single SkilledAgent.run() call."""

    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    skill_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SkilledAgent:
    """An agent that uses Skills instead of raw tools.

    Skills provide their own prompt fragments, validation, normalisation,
    and fallback behaviour — the agent loop doesn't need to know any of that.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        model: str = "gpt-4o",
        client: Optional[object] = None,
    ) -> None:
        self.registry = registry
        self.model = model
        # Accept an injected client to support testing without API keys
        self.client: OpenAI = client or OpenAI(  # type: ignore[assignment]
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.loaded_skills: list[Skill] = []

    # ------------------------------------------------------------------
    # Skill loading
    # ------------------------------------------------------------------

    def load_skills(self, skill_names: list[str]) -> None:
        """Load skills by name, resolving dependencies automatically.

        Each skill's dependencies are loaded before the skill itself.
        Already-loaded skills are skipped to avoid duplicates.
        """
        loaded_set: set[str] = {s.name for s in self.loaded_skills}
        for name in skill_names:
            skill = self.registry.get(name)
            for dep_skill in self.registry.resolve_dependencies(skill):
                if dep_skill.name not in loaded_set:
                    self.loaded_skills.append(dep_skill)
                    loaded_set.add(dep_skill.name)
        logger.info("Loaded skills: %s", [s.name for s in self.loaded_skills])

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Build system prompt from base prompt + all loaded skill fragments."""
        if not self.loaded_skills:
            return _BASE_PROMPT

        fragments = []
        for skill in self.loaded_skills:
            frag = skill.get_prompt_fragment()
            fragments.append(f"## Skill: {skill.name}\n{frag}")

        return _BASE_PROMPT + "\n\n" + "\n\n".join(fragments)

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> SkillAgentResult:
        """Run the ReAct loop using skill schemas as tools.

        The agent reasons, calls skills (which run through their full
        validate → execute → normalise pipeline), observes the results,
        and loops until it produces a final text answer.

        Args:
            user_input: The user's question or instruction.

        Returns:
            SkillAgentResult with the final answer and execution details.

        Raises:
            ValueError: If user_input is empty.
        """
        if not user_input.strip():
            raise ValueError("user_input must not be empty")

        tools = [s.get_openai_schema() for s in self.loaded_skills]
        skill_map = {s.name: s for s in self.loaded_skills}

        messages: list[dict] = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_input},
        ]

        all_tool_calls: list[dict] = []
        all_skill_results: list[dict] = []

        for iteration in range(MAX_ITERATIONS):
            logger.debug("Iteration %d", iteration)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
            )

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return SkillAgentResult(
                    answer=msg.content or "",
                    tool_calls=all_tool_calls,
                    skill_results=all_skill_results,
                )

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}

                logger.info("Calling skill '%s' params=%r", fn_name, params)
                all_tool_calls.append({"skill": fn_name, "params": params})

                skill = skill_map.get(fn_name)
                if skill:
                    result: SkillResult = skill.execute(params)
                else:
                    result = SkillResult(
                        success=False,
                        error=f"Unknown skill: {fn_name}",
                        error_type="internal",
                    )

                all_skill_results.append({"skill": fn_name, "result": result})

                content = (
                    json.dumps(result.data)
                    if result.success
                    else json.dumps(
                        {
                            "error": result.error,
                            "error_type": result.error_type,
                            "suggestion": result.suggestion,
                        }
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    }
                )

        logger.warning("Reached max iterations (%d)", MAX_ITERATIONS)
        return SkillAgentResult(
            answer="I wasn't able to complete the task within the iteration limit.",
            tool_calls=all_tool_calls,
            skill_results=all_skill_results,
        )


# ---------------------------------------------------------------------------
# Demo
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

    agent = SkilledAgent(registry)
    agent.load_skills(["weather_reporting", "stock_analysis"])

    print("=== System Prompt ===")
    print(agent.build_system_prompt())
    print()

    queries = [
        "What's the weather in Tokyo?",
        "Should I invest in AAPL?",
        "Compare the weather in London and New York",
    ]

    for query in queries:
        print(f"=== Query: {query} ===")
        result = agent.run(query)
        print(f"Answer: {result.answer}")
        print(f"Skills used: {[tc['skill'] for tc in result.tool_calls]}")
        print()


if __name__ == "__main__":
    main()
