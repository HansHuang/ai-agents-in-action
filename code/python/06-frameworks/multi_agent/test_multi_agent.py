"""pytest tests for 06-multi-agent pattern agents.

All tests use MockLLM — no real OpenAI API calls, no OPENAI_API_KEY required.

Test coverage:
  Delegation (3 tests):
    1. Coordinator delegates to correct specialist based on task type
    2. Coordinator delegates to multiple specialists for cross-domain tasks
    3. Specialist failure is handled gracefully — final answer is non-empty

  Debate (2 tests):
    4. Debate improves answer when critic is initially unsatisfied
    5. Debate stops immediately when critic is satisfied on round 1

  Supervisor (2 tests):
    6. Supervisor decomposes a complex task into 3+ subtasks
    7. Supervisor reassigns on validation failure and provides feedback

  Swarm (2 tests):
    8. Swarm produces diverse outputs from multiple agents
    9. Swarm merger consolidates repeated ideas

  Communication patterns (1 test):
    10. Structured handoff preserves explicit context boundaries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from delegation_agent import CoordinatorAgent, HandoffResult, SpecialistAgent, build_specialists
from debate_agent import DebateSystem
from supervisor_agent import SubtaskStatus, SupervisorAgent, WorkerAgent, build_workers
from swarm_agent import AgentResponse, SwarmAgent
from structured_handoff import Handoff


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


def _tool_call_response(tool_name: str, arguments: dict) -> MagicMock:
    """Create a mock response that requests one tool call."""
    tc = MagicMock()
    tc.id = "call_mock_01"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(arguments)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


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


def _make_coordinator(specialist_responses: dict[str, list], coordinator_responses: list) -> CoordinatorAgent:
    """Build a CoordinatorAgent with mocked LLMs for coordinator and specialists."""
    specialists: dict[str, SpecialistAgent] = {}
    for name in ["finance", "research", "writing"]:
        spec_resps = specialist_responses.get(name, [_content_response("Mock specialist result.")])
        spec = SpecialistAgent(
            name=name,
            role=f"Mock {name} specialist",
            tools=[],
            system_prompt=f"Mock {name} prompt",
            task_guidance=f"Use for {name} tasks.",
        )
        mock_llm = MockLLM(spec_resps)
        client = MagicMock()
        client.chat.completions.create = mock_llm.create
        spec._client = client
        specialists[name] = spec

    coordinator = CoordinatorAgent(specialists)
    coord_mock = MockLLM(coordinator_responses)
    coord_client = MagicMock()
    coord_client.chat.completions.create = coord_mock.create
    coordinator._client = coord_client
    return coordinator


def _make_debate_system(responses: list, max_rounds: int = 3) -> DebateSystem:
    system = DebateSystem(max_rounds=max_rounds)
    mock_llm = MockLLM(responses)
    client = MagicMock()
    client.chat.completions.create = mock_llm.create
    system._client = client
    return system


def _make_supervisor(worker_responses: dict[str, list], supervisor_responses: list) -> SupervisorAgent:
    """Build a SupervisorAgent with mocked worker and supervisor LLMs."""
    workers: dict[str, WorkerAgent] = {}
    for name in ["research_worker", "analysis_worker", "writing_worker"]:
        w_resps = worker_responses.get(name, [_content_response("Mock worker result.")])
        worker = WorkerAgent(name=name, tools=[], system_prompt=f"Mock {name} prompt")
        mock_llm = MockLLM(w_resps)
        wc = MagicMock()
        wc.chat.completions.create = mock_llm.create
        worker._client = wc
        workers[name] = worker

    supervisor = SupervisorAgent(workers)
    sup_mock = MockLLM(supervisor_responses)
    sup_client = MagicMock()
    sup_client.chat.completions.create = sup_mock.create
    supervisor._client = sup_client
    return supervisor


def _make_swarm(per_agent_response: str | list[str], merge_response: str, swarm_size: int = 4) -> SwarmAgent:
    """Build a SwarmAgent with pre-programmed per-agent and merger responses."""
    agent = SwarmAgent(swarm_size=swarm_size)
    if isinstance(per_agent_response, str):
        agent_responses = [_content_response(per_agent_response)] * swarm_size
    else:
        agent_responses = [_content_response(r) for r in per_agent_response]
    all_responses = agent_responses + [_content_response(merge_response)]
    mock_llm = MockLLM(all_responses)
    client = MagicMock()
    client.chat.completions.create = mock_llm.create
    agent._client = client
    return agent


# ---------------------------------------------------------------------------
# Delegation tests
# ---------------------------------------------------------------------------


class TestCoordinatorDelegation:
    def test_coordinator_delegates_to_correct_specialist(self):
        """Coordinator calls the finance specialist for a financial task."""
        coordinator = _make_coordinator(
            specialist_responses={
                "finance": [_content_response("AAPL is trading at $192.35.")]
            },
            coordinator_responses=[
                _tool_call_response("delegate_to_finance_agent", {"task": "Get AAPL stock price"}),
                _content_response("Apple is trading at $192.35."),
            ],
        )
        result = coordinator.run("What is Apple's stock price?")

        assert result["answer"]
        delegation_agents = [d["agent"] for d in result["delegations"]]
        assert "finance" in delegation_agents
        assert "research" not in delegation_agents

    def test_coordinator_delegates_to_multiple_specialists(self):
        """Coordinator calls both research and finance specialists."""
        coordinator = _make_coordinator(
            specialist_responses={
                "finance":   [_content_response("Apple revenue: $391B TTM.")],
                "research":  [_content_response("Apple had record Q3 services revenue.")],
            },
            coordinator_responses=[
                _tool_call_response("delegate_to_research_agent", {"task": "Research Apple news"}),
                _tool_call_response("delegate_to_finance_agent", {"task": "Get Apple financials"}),
                _content_response("Apple had record services revenue and $391B TTM revenue."),
            ],
        )
        result = coordinator.run("What is Apple's financial performance and latest news?")

        delegation_agents = [d["agent"] for d in result["delegations"]]
        assert "research" in delegation_agents
        assert "finance" in delegation_agents
        assert result["answer"]

    def test_specialist_failure_handled_gracefully(self):
        """When a specialist raises, the coordinator continues and returns a non-empty answer."""
        # Make the finance specialist raise an exception
        finance_spec = SpecialistAgent(
            name="finance",
            role="Finance specialist",
            tools=[],
            system_prompt="Mock finance",
            task_guidance="",
        )
        failing_client = MagicMock()
        failing_client.chat.completions.create.side_effect = RuntimeError("API error")
        finance_spec._client = failing_client

        specialists = {
            "finance": finance_spec,
            "research": SpecialistAgent("research", "", [], "", ""),
            "writing":  SpecialistAgent("writing", "", [], "", ""),
        }
        for name in ["research", "writing"]:
            mock = MagicMock()
            mock.chat.completions.create = MockLLM([_content_response(f"Mock {name} result.")]).create
            specialists[name]._client = mock

        coordinator = CoordinatorAgent(specialists)
        coord_mock = MockLLM([
            _tool_call_response("delegate_to_finance_agent", {"task": "Get Apple financials"}),
            _content_response("I was unable to retrieve financials, but here is a general answer."),
        ])
        coord_client = MagicMock()
        coord_client.chat.completions.create = coord_mock.create
        coordinator._client = coord_client

        result = coordinator.run("What are Apple's financials?")

        assert result["answer"]
        # Delegation was attempted even though it failed
        assert any(d["agent"] == "finance" for d in result["delegations"])


# ---------------------------------------------------------------------------
# Debate tests
# ---------------------------------------------------------------------------


class TestDebateSystem:
    def test_debate_improves_answer(self):
        """When the critic is unsatisfied on round 1, the generator revises."""
        initial_answer = "Build a basic app."
        revised_answer = "Build an app with offline mode, smart notifications, and AI personalisation."

        system = _make_debate_system(
            responses=[
                # Initial generation
                _content_response(initial_answer),
                # Round 1 critique: unsatisfied
                _content_response("ISSUES: The answer is too vague. Add specific features and distribution strategy."),
                # Round 1 revised generation
                _content_response(revised_answer),
                # Round 2 critique: satisfied
                _content_response("NO_ISSUES: The answer is comprehensive and specific."),
            ],
            max_rounds=3,
        )
        result = system.run("Design a go-to-market strategy for an AI email client")

        assert result["final_answer"] != initial_answer
        assert result["rounds_completed"] >= 1
        assert result["final_answer"]

    def test_debate_stops_when_critic_satisfied(self):
        """When the critic signals NO_ISSUES on round 1, no further rounds occur."""
        system = _make_debate_system(
            responses=[
                _content_response("An excellent, comprehensive answer."),  # initial
                _content_response("NO_ISSUES: The answer covers all bases."),  # critique round 1
            ],
            max_rounds=5,
        )
        result = system.run("What is the capital of France?")

        assert result["rounds_completed"] == 1
        assert result["history"][0]["critic_satisfied"] is True


# ---------------------------------------------------------------------------
# Supervisor tests
# ---------------------------------------------------------------------------


class TestSupervisorAgent:
    def test_supervisor_decomposes_complex_task(self):
        """Supervisor generates a plan with at least 3 subtasks."""
        plan = {
            "subtasks": [
                {"id": "t1", "description": "Research AI code editor market players", "assigned_worker": "research_worker", "dependencies": []},
                {"id": "t2", "description": "Analyse market share data for top 3 players", "assigned_worker": "analysis_worker", "dependencies": ["t1"]},
                {"id": "t3", "description": "Write competitive analysis report", "assigned_worker": "writing_worker", "dependencies": ["t1", "t2"]},
            ]
        }
        supervisor = _make_supervisor(
            worker_responses={
                "research_worker": [_content_response("GitHub Copilot 35%, Cursor AI 20%, Replit 10%.")],
                "analysis_worker":  [_content_response("Copilot leads. Cursor growing fastest (+15% MoM).")],
                "writing_worker":   [_content_response("## AI Code Editor Market\n\nCopilot leads with 35% share...")],
            },
            supervisor_responses=[
                _json_response(plan),                           # decompose
                _json_response({"passes": True, "feedback": ""}),  # validate t1
                _json_response({"passes": True, "feedback": ""}),  # validate t2
                _json_response({"passes": True, "feedback": ""}),  # validate t3
                _content_response("## Competitive Analysis\n\nCopilot leads the market..."),  # synthesize
            ],
        )
        output = supervisor.run("Create a competitive analysis report for the AI code editor market")

        assert len(output["workflow"]) >= 3
        assert output["answer"]

    def test_supervisor_reassigns_on_failure(self):
        """Supervisor retries a subtask when validation fails, passing feedback."""
        plan = {
            "subtasks": [
                {"id": "t1", "description": "Research AI editor market", "assigned_worker": "research_worker", "dependencies": []},
            ]
        }
        # Worker will be called twice: first attempt fails validation, second passes
        supervisor = _make_supervisor(
            worker_responses={
                "research_worker": [
                    _content_response("Some vague info."),          # attempt 1
                    _content_response("Detailed: GitHub Copilot has 35% market share."),  # attempt 2
                ],
            },
            supervisor_responses=[
                _json_response(plan),
                # Validation attempt 1: fail
                _json_response({"passes": False, "feedback": "Too vague — include market share percentages."}),
                # Validation attempt 2: pass
                _json_response({"passes": True, "feedback": ""}),
                _content_response("GitHub Copilot leads the market."),  # synthesize
            ],
        )
        output = supervisor.run("Research the AI code editor market")

        reassigned = next(t for t in output["workflow"] if t["id"] == "t1")
        assert reassigned["attempts"] >= 2
        assert reassigned["status"] == SubtaskStatus.DONE.value


# ---------------------------------------------------------------------------
# Swarm tests
# ---------------------------------------------------------------------------


class TestSwarmAgent:
    def test_swarm_produces_diverse_outputs(self):
        """Four agents produce distinct responses."""
        per_agent = [
            "Idea A: Smart lighting that learns your schedule. Idea B: Voice-controlled appliances.",
            "Idea C: Predictive energy optimisation. Idea D: Automated grocery ordering.",
            "Idea E: Health monitoring integration. Idea F: Seamless multi-device scenes.",
            "Idea G: Carbon footprint tracking. Idea H: Pet care automation.",
        ]
        agent = _make_swarm(per_agent_response=per_agent, merge_response="Merged: A, B, C, D, E, F, G, H")
        result = agent.run("Feature ideas for a smart home app")

        individual = result["individual_responses"]
        assert len(individual) == 4
        # All four responses should be present and non-empty
        texts = {r["response"] for r in individual}
        assert len(texts) >= 3  # at least 3 distinct responses

    def test_swarm_merger_consolidates_ideas(self):
        """When agents repeat the same idea, the merger consolidates it."""
        repeated = "Smart lighting with schedule learning."
        merge_response = f"Consolidated: {repeated} This feature was mentioned by all agents."

        agent = _make_swarm(per_agent_response=repeated, merge_response=merge_response, swarm_size=4)
        result = agent.run("Feature ideas for a smart home app")

        assert "Consolidated" in result["merged_answer"]
        # The raw individual responses all contain the same idea
        for r in result["individual_responses"]:
            assert "Smart lighting" in r["response"]


# ---------------------------------------------------------------------------
# Structured handoff test
# ---------------------------------------------------------------------------


class TestStructuredHandoff:
    def test_structured_handoff_preserves_boundaries(self):
        """Handoff context contains only explicitly passed data."""
        handoff = Handoff(
            from_agent="coordinator",
            to_agent="research_specialist",
            task="Research the TypeScript ecosystem",
            context={
                "domain": "enterprise software",
                "constraints": "existing team knows JavaScript",
            },
        )
        handoff.validate()

        # Context only contains what was explicitly provided
        assert set(handoff.context.keys()) == {"domain", "constraints"}
        assert "coordinator" not in handoff.context
        assert "system_prompt" not in handoff.context

        # to_user_message includes task and context, not internal agent state
        msg = handoff.to_user_message()
        assert "Research the TypeScript ecosystem" in msg
        assert "enterprise software" in msg
        # Internal coordinator message history is NOT in the handoff message
        assert "system_prompt" not in msg
        assert "internal" not in msg.lower()
