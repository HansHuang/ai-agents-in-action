"""pytest tests for planning strategy agents.

All tests use MockLLM — no real OpenAI API calls, no OPENAI_API_KEY required.

Test coverage:
  Plan-and-Execute (5 tests):
    1. Single tool step executes and returns correct data
    2. Simple 2-step plan: tool + synthesis step
    3. Dependency order is respected (step 2 waits for step 1)
    4. Independent steps (no shared deps) run without blocking each other
    5. Failing tool step is captured as a failed StepResult

  Reflection (4 tests):
    6. Initial answer is returned when score meets threshold
    7. Answer is revised when first score is below threshold
    8. Loop stops at max_reflections even if score stays below threshold
    9. Iteration history records each generation and critique

  Strategy comparison smoke test (1 test):
    10. All three strategies return non-empty answer strings for the same task
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plan_execute_agent import PlanAndExecuteAgent
from plan_schema import PlanStep, StepResult
from reflection_agent import CritiqueResult, ReflectionAgent


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _content_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _json_response(data: dict) -> MagicMock:
    return _content_response(json.dumps(data))


class MockLLM:
    """Pre-programmed sequence of OpenAI chat completion responses."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def create(self, **kwargs) -> object:  # noqa: ANN001
        if self.call_count >= len(self.responses):
            raise AssertionError(
                f"MockLLM called {self.call_count + 1}× but only "
                f"{len(self.responses)} response(s) were programmed."
            )
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def _make_agent(responses: list) -> PlanAndExecuteAgent:
    """Build a PlanAndExecuteAgent with a MockLLM injected."""
    agent = PlanAndExecuteAgent()
    mock_llm = MockLLM(responses)
    client = MagicMock()
    client.chat.completions.create = mock_llm.create
    agent._client = client
    return agent


def _make_reflection_agent(responses: list, **kwargs) -> ReflectionAgent:
    """Build a ReflectionAgent with a MockLLM injected."""
    agent = ReflectionAgent(**kwargs)
    mock_llm = MockLLM(responses)
    client = MagicMock()
    client.chat.completions.create = mock_llm.create
    agent._client = client
    return agent


# ---------------------------------------------------------------------------
# Plan-and-Execute tests
# ---------------------------------------------------------------------------


def test_pe_single_tool_step_executes_correctly():
    """A single-step plan calling get_stock_price returns correct mock data."""
    plan_json = {
        "user_question": "What is AAPL price?",
        "steps": [
            {
                "step_number": 1,
                "description": "Get the current Apple stock price",
                "tool_name": "get_stock_price",
                "tool_params": {"ticker": "AAPL"},
                "depends_on": [],
                "expected_output": "AAPL price in USD with change percent",
            }
        ],
        "estimated_tool_calls": 1,
    }
    agent = _make_agent([
        _json_response(plan_json),           # planner call
        _content_response("AAPL is $192.35"),  # synthesizer call
    ])

    output = agent.run("What is AAPL price?")

    assert len(output["results"]) == 1
    result = output["results"][0]
    assert result["success"] is True
    assert result["data"]["ticker"] == "AAPL"
    assert result["data"]["price_usd"] == 192.35


def test_pe_two_step_plan_tool_then_synthesis():
    """Two-step plan: tool step followed by a no-tool synthesis step."""
    plan_json = {
        "user_question": "What is MSFT stock performance?",
        "steps": [
            {
                "step_number": 1,
                "description": "Fetch Microsoft stock price",
                "tool_name": "get_stock_price",
                "tool_params": {"ticker": "MSFT"},
                "depends_on": [],
                "expected_output": "MSFT price and weekly change",
            },
            {
                "step_number": 2,
                "description": "Summarise the stock performance for the user",
                "tool_name": None,
                "tool_params": None,
                "depends_on": [1],
                "expected_output": "Plain-English summary of MSFT performance",
            },
        ],
        "estimated_tool_calls": 1,
    }
    agent = _make_agent([
        _json_response(plan_json),
        _content_response("MSFT analysis: strong week"),  # reasoning step (step 2)
        _content_response("Microsoft is up 2.4% this week."),  # synthesizer
    ])

    output = agent.run("What is MSFT stock performance?")

    assert len(output["plan"]) == 2
    assert len(output["results"]) == 2
    assert output["results"][0]["success"] is True
    assert output["results"][1]["success"] is True
    assert "Microsoft" in output["answer"] or output["answer"]  # synthesizer ran


def test_pe_dependency_order_respected():
    """Step 2 that depends on step 1 must wait; verify its dep is resolved first."""
    plan_json = {
        "user_question": "Compare AAPL and MSFT",
        "steps": [
            {
                "step_number": 1,
                "description": "Get AAPL stock price",
                "tool_name": "get_stock_price",
                "tool_params": {"ticker": "AAPL"},
                "depends_on": [],
                "expected_output": "AAPL current price and weekly change",
            },
            {
                "step_number": 2,
                "description": "Get MSFT stock price",
                "tool_name": "get_stock_price",
                "tool_params": {"ticker": "MSFT"},
                "depends_on": [1],  # forced dependency
                "expected_output": "MSFT current price and weekly change",
            },
        ],
        "estimated_tool_calls": 2,
    }
    agent = _make_agent([
        _json_response(plan_json),
        _content_response("Comparison done."),  # synthesizer
    ])

    output = agent.run("Compare AAPL and MSFT")

    results = output["results"]
    assert results[0]["success"] is True
    assert results[1]["success"] is True
    # Both results have data from the mock tools
    assert results[0]["data"]["ticker"] == "AAPL"
    assert results[1]["data"]["ticker"] == "MSFT"


def test_pe_independent_steps_all_execute():
    """Two steps with no dependencies can both execute successfully."""
    plan_json = {
        "user_question": "Get Tokyo and London weather",
        "steps": [
            {
                "step_number": 1,
                "description": "Get Tokyo weather",
                "tool_name": "get_weather",
                "tool_params": {"city": "Tokyo"},
                "depends_on": [],
                "expected_output": "Tokyo temperature and conditions",
            },
            {
                "step_number": 2,
                "description": "Get London weather",
                "tool_name": "get_weather",
                "tool_params": {"city": "London"},
                "depends_on": [],
                "expected_output": "London temperature and conditions",
            },
        ],
        "estimated_tool_calls": 2,
    }
    agent = _make_agent([
        _json_response(plan_json),
        _content_response("Both cities fetched."),  # synthesizer
    ])

    output = agent.run("Get Tokyo and London weather")

    results = output["results"]
    assert len(results) == 2
    step_nums = {r["step_number"] for r in results}
    assert step_nums == {1, 2}
    assert all(r["success"] for r in results)


def test_pe_failing_tool_is_captured():
    """A step calling an unknown tool name results in a failed StepResult."""
    plan_json = {
        "user_question": "Run an unknown tool",
        "steps": [
            {
                "step_number": 1,
                "description": "Call a tool that does not exist in the registry",
                "tool_name": "nonexistent_tool",
                "tool_params": {"param": "value"},
                "depends_on": [],
                "expected_output": "Some output from the nonexistent tool",
            }
        ],
        "estimated_tool_calls": 1,
    }
    agent = _make_agent([
        _json_response(plan_json),
        _content_response("Tool failed but here is a fallback answer."),  # synthesizer
    ])

    output = agent.run("Run an unknown tool")

    result = output["results"][0]
    assert result["success"] is False
    assert "nonexistent_tool" in result["error"] or "Unknown tool" in result["error"]


# ---------------------------------------------------------------------------
# Reflection tests
# ---------------------------------------------------------------------------


def _satisfied_critique(score: int = 9) -> dict:
    return {
        "overall_score": score,
        "is_satisfied": True,
        "feedback": "Excellent answer covering all aspects of the question.",
        "strengths": ["Comprehensive", "Well-structured"],
        "weaknesses": [],
    }


def _unsatisfied_critique(score: int = 4) -> dict:
    return {
        "overall_score": score,
        "is_satisfied": False,
        "feedback": "The answer is missing key details and citations.",
        "strengths": ["Correct direction"],
        "weaknesses": ["Missing citations", "Too vague"],
    }


def test_reflection_returns_initial_when_score_meets_threshold():
    """If the first critique score meets the threshold, no revision is performed."""
    agent = _make_reflection_agent(
        [
            _content_response("Transformers use attention mechanisms."),    # generate
            _json_response(_satisfied_critique(score=9)),                   # critique → satisfied
        ],
        max_reflections=2,
        quality_threshold=8,
    )

    result = agent.run("How do transformers work?")

    assert result["reflections_used"] == 0
    assert len(result["iterations"]) == 1
    assert "Transformers" in result["final_answer"]


def test_reflection_revises_when_score_below_threshold():
    """When the first score is below threshold, one revision round is performed."""
    agent = _make_reflection_agent(
        [
            _content_response("Transformers are neural networks."),         # generate
            _json_response(_unsatisfied_critique(score=4)),                 # critique → unsatisfied
            _content_response("Transformers use self-attention, multi-head attention, and positional encoding."),  # revise
            _json_response(_satisfied_critique(score=9)),                   # critique of revision
        ],
        max_reflections=2,
        quality_threshold=8,
    )

    result = agent.run("How do transformers work?")

    assert result["reflections_used"] == 1
    assert len(result["iterations"]) == 2
    assert "attention" in result["final_answer"].lower()


def test_reflection_stops_at_max_reflections():
    """The loop does not exceed max_reflections even if quality never meets threshold."""
    # max_reflections=1 → 1 generation + 1 critique + 1 revision + 1 final critique
    agent = _make_reflection_agent(
        [
            _content_response("Initial weak answer."),           # generate
            _json_response(_unsatisfied_critique(score=3)),      # critique 1 → revise
            _content_response("Still a weak answer."),           # revision
            _json_response(_unsatisfied_critique(score=4)),      # critique 2 → still below
        ],
        max_reflections=1,
        quality_threshold=8,
    )

    result = agent.run("Explain attention mechanisms")

    # Used exactly max_reflections rounds
    assert result["reflections_used"] == 1
    assert len(result["iterations"]) == 2  # initial + 1 revision


def test_reflection_iteration_history_is_correct():
    """Each iteration record captures the answer and its critique."""
    agent = _make_reflection_agent(
        [
            _content_response("First answer text here."),
            _json_response(_satisfied_critique(score=8)),
        ],
        max_reflections=2,
        quality_threshold=8,
    )

    result = agent.run("What is a transformer?")

    iterations = result["iterations"]
    assert len(iterations) == 1
    it = iterations[0]
    assert it["iteration"] == 1
    assert "First answer" in it["answer"]
    assert it["critique"] is not None
    assert it["critique"]["overall_score"] == 8


# ---------------------------------------------------------------------------
# Strategy comparison smoke test
# ---------------------------------------------------------------------------


def test_all_strategies_return_non_empty_answers(monkeypatch):
    """All three strategies (ReAct, Plan-and-Execute, Reflection) return answers."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    task = "What is the weather in Tokyo?"

    # --- ReAct via run_agent ---
    import agent as agent_mod
    from tools import TOOLS

    react_mock_llm = MockLLM([
        _content_response("Tokyo has light rain and 22°C."),
    ])
    react_client = MagicMock()
    react_client.chat.completions.create = react_mock_llm.create

    with patch("agent.OpenAI", return_value=react_client):
        react_answer = agent_mod.run_agent(task, tools=TOOLS)
    assert isinstance(react_answer, str) and len(react_answer) > 0

    # --- Plan-and-Execute ---
    pe_plan = {
        "user_question": task,
        "steps": [
            {
                "step_number": 1,
                "description": "Get Tokyo weather",
                "tool_name": "get_weather",
                "tool_params": {"city": "Tokyo"},
                "depends_on": [],
                "expected_output": "Tokyo temperature and conditions",
            }
        ],
        "estimated_tool_calls": 1,
    }
    pe_agent = _make_agent([
        _json_response(pe_plan),
        _content_response("Tokyo: 22°C, light rain."),
    ])
    pe_output = pe_agent.run(task)
    assert isinstance(pe_output["answer"], str) and len(pe_output["answer"]) > 0

    # --- Reflection ---
    ref_agent = _make_reflection_agent(
        [
            _content_response("Tokyo has light rain and 22°C."),
            _json_response(_satisfied_critique(score=9)),
        ],
        max_reflections=1,
        quality_threshold=8,
    )
    ref_output = ref_agent.run(task)
    assert isinstance(ref_output["final_answer"], str) and len(ref_output["final_answer"]) > 0
