"""AutoGen product design team: conversational multi-agent design.

Implements a five-agent group chat using AutoGen (Microsoft):

    ProductManager ─┐
    Designer        │
    Engineer        ├─→ GroupChatManager → consensus → DesignSpec
    Critic          │
    UserProxy       ┘

Solutions emerge from conversation rather than from a predefined task list.
That is the fundamental difference from CrewAI: here, no agent "plans" the
discussion — the dialogue itself produces the outcome.

When AutoGen is not installed a custom conversational loop provides the same
pattern using only the OpenAI SDK, so you can compare the two approaches.

Run:
    python autogen_design_team.py

See: docs/06-frameworks-in-practice/03-crewai-autogen.md
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional AutoGen import — pyautogen (v0.2/v0.3 API)
# ---------------------------------------------------------------------------
# AutoGen has gone through significant API changes.  We target the stable
# v0.2/v0.3 interface published as ``pyautogen``.  A v0.4+ port would use
# ``autogen_agentchat`` but the group-chat concepts are identical.
# ---------------------------------------------------------------------------

try:
    import autogen  # type: ignore[import]
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager  # type: ignore[import]

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared LLM client (also used by the from-scratch fallback)
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConversationAnalysis:
    """Statistical breakdown of a completed group-chat conversation."""

    speaker_turns: dict[str, int] = field(default_factory=dict)
    productive_rounds: int = 0
    circular_rounds: int = 0
    decisions: list[str] = field(default_factory=list)
    deadlocks_detected: int = 0
    total_rounds: int = 0


@dataclass
class DesignResult:
    """Structured output from any product design team implementation."""

    requirement: str
    final_spec: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    rounds_taken: int = 0
    decisions_made: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    token_usage: int = 0


# ---------------------------------------------------------------------------
# ProductDesignTeam — AutoGen implementation
# ---------------------------------------------------------------------------


class ProductDesignTeam:
    """An AutoGen group chat that designs a product feature through conversation.

    Agents:
    - ProductManager: Defines requirements and priorities
    - Designer: Proposes UX solutions
    - Engineer: Evaluates technical feasibility
    - Critic: Challenges assumptions and finds edge cases
    - UserProxy: Represents the user; can ask for human input

    The conversation proceeds until the team reaches consensus on a spec,
    or until max_rounds is exhausted.
    """

    TERMINATE_SIGNAL = "DESIGN_APPROVED"

    def __init__(self, model: str = "gpt-4o") -> None:
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "pyautogen is required for ProductDesignTeam. "
                "Install with: pip install pyautogen"
            )
        self.llm_config: dict = {
            "config_list": [
                {
                    "model": model,
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                }
            ],
            "temperature": 0.7,
        }
        self.agents = self._create_agents()

    def _create_agents(self) -> dict:
        """Create all agents with distinct system messages.

        Each agent has a clear personality and a well-defined area of expertise.
        This is what shapes the conversation: roles create productive tension.
        """
        product_manager = AssistantAgent(
            name="ProductManager",
            llm_config=self.llm_config,
            system_message=(
                "You are an experienced product manager. Your job is to clarify "
                "requirements, set priorities, and keep the team focused on user "
                "value. When the team proposes something technically interesting "
                "but user-unfriendly, push back. When the team reaches a decision, "
                "summarise it clearly. If the design is complete and the team agrees, "
                f"end your message with '{self.TERMINATE_SIGNAL}'."
            ),
        )

        designer = AssistantAgent(
            name="Designer",
            llm_config=self.llm_config,
            system_message=(
                "You are a senior UX designer. You propose user interface solutions "
                "grounded in usability principles and accessibility standards. You "
                "think about edge cases: what happens when two users edit the same "
                "thing simultaneously? What does the error state look like? Always "
                "sketch the happy path first, then cover failure modes."
            ),
        )

        engineer = AssistantAgent(
            name="Engineer",
            llm_config=self.llm_config,
            system_message=(
                "You are a senior software engineer. You evaluate design proposals "
                "for technical feasibility, performance implications, and security "
                "risks. Be specific: if something requires O(n²) syncing or "
                "introduces a race condition, say so. Propose concrete technical "
                "solutions, not just objections."
            ),
        )

        critic = AssistantAgent(
            name="Critic",
            llm_config=self.llm_config,
            system_message=(
                "You are a rigorous product critic. Your job is to find flaws that "
                "the team has missed: edge cases, unmet user needs, inconsistencies "
                "between requirements and proposed solutions. Be constructive — "
                "your goal is a better product, not winning the argument. Prioritise "
                "your objections; not everything is equally important."
            ),
        )

        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",  # Fully automated; change to "TERMINATE" for interactive
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda msg: self.TERMINATE_SIGNAL in msg.get("content", ""),
            code_execution_config=False,
        )

        return {
            "product_manager": product_manager,
            "designer": designer,
            "engineer": engineer,
            "critic": critic,
            "user_proxy": user_proxy,
        }

    def design_feature(
        self, requirement: str, max_rounds: int = 20
    ) -> DesignResult:
        """Run the design conversation and return structured results.

        Returns:
            DesignResult with requirement, final_spec, conversation_history,
            rounds_taken, decisions_made, execution_time, token_usage.
        """
        agents = self.agents
        groupchat = GroupChat(
            agents=[
                agents["user_proxy"],
                agents["product_manager"],
                agents["designer"],
                agents["engineer"],
                agents["critic"],
            ],
            messages=[],
            max_round=max_rounds,
            speaker_selection_method="round_robin",
        )
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config,
        )

        start = time.perf_counter()
        agents["user_proxy"].initiate_chat(
            manager,
            message=(
                f"We need to design the following product feature.\n\n"
                f"REQUIREMENT: {requirement}\n\n"
                "Please work through the design together. ProductManager: start "
                "by clarifying the requirements. Designer: propose solutions. "
                "Engineer: evaluate feasibility. Critic: challenge assumptions. "
                "Iterate until you reach consensus on a complete design spec."
            ),
        )
        elapsed = time.perf_counter() - start

        history = groupchat.messages
        decisions = _extract_decisions(history)
        final_spec = _build_final_spec(history, decisions)

        return DesignResult(
            requirement=requirement,
            final_spec=final_spec,
            conversation_history=[
                {"role": m.get("name", "unknown"), "content": m.get("content", "")}
                for m in history
            ],
            rounds_taken=len(history),
            decisions_made=decisions,
            execution_time=elapsed,
        )

    def analyze_conversation(self, history: list[dict]) -> ConversationAnalysis:
        """Analyze a conversation for participation, productivity, and decisions.

        Returns:
            ConversationAnalysis with speaker turns, productive vs. circular
            rounds, decisions made, deadlocks, and total rounds.
        """
        speaker_turns: dict[str, int] = {}
        decisions: list[str] = []
        seen_topics: set[str] = set()
        circular_count = 0
        productive_count = 0

        for msg in history:
            speaker = msg.get("role", msg.get("name", "unknown"))
            content = msg.get("content", "")

            speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1

            # Heuristic: detect decisions (ProductManager summaries, approvals)
            if "DESIGN_APPROVED" in content or re.search(
                r"\bdecid(ed|e|ing)\b|\bagreed?\b|\bwe will\b", content, re.I
            ):
                # Extract the first sentence as the decision
                first_sentence = content.split(".")[0].strip()
                if first_sentence and first_sentence not in decisions:
                    decisions.append(first_sentence[:120])

            # Heuristic: circular if the same key phrase appears again
            key_phrase = re.sub(r"\W+", " ", content[:60].lower()).strip()
            if key_phrase in seen_topics:
                circular_count += 1
            else:
                seen_topics.add(key_phrase)
                productive_count += 1

        deadlocks = max(0, circular_count - 2)  # 1-2 repeats are normal debate

        return ConversationAnalysis(
            speaker_turns=speaker_turns,
            productive_rounds=productive_count,
            circular_rounds=circular_count,
            decisions=decisions,
            deadlocks_detected=deadlocks,
            total_rounds=len(history),
        )


# ---------------------------------------------------------------------------
# CustomConversationalTeam — from-scratch fallback (no AutoGen required)
# ---------------------------------------------------------------------------


class CustomConversationalTeam:
    """Same five-agent conversation pattern using only the OpenAI SDK.

    Demonstrates that AutoGen's conversational model is conceptually simple:
    each agent is a system prompt, messages accumulate in shared history,
    and the manager decides who speaks next.
    """

    TERMINATE_SIGNAL = "DESIGN_APPROVED"

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.client = _get_client()
        self._system_prompts = self._build_system_prompts()

    def _build_system_prompts(self) -> dict[str, str]:
        return {
            "ProductManager": (
                "You are an experienced product manager. Clarify requirements, "
                "set priorities, and keep the team focused on user value. When "
                f"the design is complete and the team agrees, say '{self.TERMINATE_SIGNAL}'."
            ),
            "Designer": (
                "You are a senior UX designer. Propose user interface solutions "
                "grounded in usability principles. Cover the happy path first, "
                "then failure modes and edge cases."
            ),
            "Engineer": (
                "You are a senior software engineer. Evaluate design proposals for "
                "technical feasibility, performance, and security. Be specific about "
                "risks and propose concrete solutions."
            ),
            "Critic": (
                "You are a rigorous product critic. Find edge cases, inconsistencies, "
                "and unmet user needs the team has missed. Be constructive and "
                "prioritise your objections."
            ),
        }

    def design_feature(
        self, requirement: str, max_rounds: int = 20
    ) -> DesignResult:
        """Run the design conversation using a simple round-robin loop."""
        turn_order = ["ProductManager", "Designer", "Engineer", "Critic"]
        conversation: list[dict] = []
        token_total = 0
        decisions: list[str] = []

        # Seed the conversation
        conversation.append({
            "role": "user",
            "content": (
                f"We need to design the following product feature.\n\n"
                f"REQUIREMENT: {requirement}\n\n"
                "ProductManager: start by clarifying the requirements."
            ),
        })

        start = time.perf_counter()
        for round_num in range(max_rounds):
            speaker = turn_order[round_num % len(turn_order)]
            sys_prompt = self._system_prompts[speaker]

            messages_for_call = [{"role": "system", "content": sys_prompt}]
            # Include the last 8 turns for context (sliding window)
            messages_for_call.extend(conversation[-8:])

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_for_call,
            )
            reply = response.choices[0].message.content or ""
            token_total += response.usage.total_tokens if response.usage else 0

            conversation.append({"role": "assistant", "name": speaker, "content": reply})

            # Capture decisions
            if re.search(r"\bdecid(ed|e|ing)\b|\bagreed?\b|\bwe will\b", reply, re.I):
                sentence = reply.split(".")[0].strip()
                if sentence not in decisions:
                    decisions.append(sentence[:120])

            # Termination check
            if self.TERMINATE_SIGNAL in reply:
                break

        elapsed = time.perf_counter() - start
        final_spec = _build_final_spec(conversation, decisions)

        return DesignResult(
            requirement=requirement,
            final_spec=final_spec,
            conversation_history=[
                {
                    "role": m.get("name", m.get("role", "unknown")),
                    "content": m.get("content", ""),
                }
                for m in conversation
            ],
            rounds_taken=len(conversation),
            decisions_made=decisions,
            execution_time=elapsed,
            token_usage=token_total,
        )

    def analyze_conversation(self, history: list[dict]) -> ConversationAnalysis:
        """Delegate to the same analysis logic as ProductDesignTeam."""
        _team = ProductDesignTeam.__new__(ProductDesignTeam)
        return _team.analyze_conversation(history)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_decisions(history: list[dict]) -> list[str]:
    """Pull decision-like sentences from conversation history."""
    decisions: list[str] = []
    for msg in history:
        content = msg.get("content", "")
        if re.search(r"\bdecid(ed|e|ing)\b|\bagreed?\b|\bwe will\b|\bfinal spec\b", content, re.I):
            sentence = content.split(".")[0].strip()
            if sentence and sentence not in decisions:
                decisions.append(sentence[:120])
    return decisions


def _build_final_spec(history: list[dict], decisions: list[str]) -> str:
    """Construct a final spec string from the last few messages and decisions."""
    if not history:
        return "(no conversation recorded)"

    # Look for an APPROVED message first
    for msg in reversed(history):
        if "DESIGN_APPROVED" in msg.get("content", ""):
            return msg["content"]

    # Fallback: summarise the last message + decisions
    last_content = history[-1].get("content", "")
    decision_text = "\n".join(f"• {d}" for d in decisions) if decisions else "(none)"
    return (
        f"DESIGN SPEC (reconstructed from conversation)\n\n"
        f"Key decisions:\n{decision_text}\n\n"
        f"Final team message:\n{last_content}"
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _divider(title: str, width: int = 60) -> None:
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def _preview(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n  … [{len(text) - max_chars} more chars]"


if __name__ == "__main__":
    REQUIREMENT = "Add collaborative editing to our document editor"

    if AUTOGEN_AVAILABLE:
        print("AutoGen detected. Running ProductDesignTeam.\n")
        team: ProductDesignTeam | CustomConversationalTeam = ProductDesignTeam()
    else:
        print("AutoGen not installed. Running CustomConversationalTeam fallback.")
        print("To enable AutoGen: pip install pyautogen\n")
        team = CustomConversationalTeam()

    result = team.design_feature(REQUIREMENT, max_rounds=12)

    _divider("REQUIREMENT")
    print(f"  {result.requirement}")

    _divider("CONVERSATION")
    for i, turn in enumerate(result.conversation_history):
        speaker = turn.get("role", "unknown")
        content = turn.get("content", "")
        print(f"\n[{i+1}] {speaker.upper()}")
        print(_preview(content, max_chars=300))

    _divider("FINAL SPEC")
    print(result.final_spec)

    _divider("CONVERSATION ANALYSIS")
    analysis = team.analyze_conversation(result.conversation_history)
    print(f"  Total rounds    : {analysis.total_rounds}")
    print(f"  Productive      : {analysis.productive_rounds}")
    print(f"  Circular        : {analysis.circular_rounds}")
    print(f"  Deadlocks       : {analysis.deadlocks_detected}")
    print(f"\n  Speaker turns:")
    for speaker, turns in sorted(analysis.speaker_turns.items(), key=lambda x: -x[1]):
        print(f"    {speaker:<20} {turns} turns")
    print(f"\n  Decisions made  : {len(analysis.decisions)}")
    for d in analysis.decisions[:5]:
        print(f"    • {d}")

    _divider("WHERE THE CRITIC IMPROVED THE DESIGN")
    critic_turns = [
        t for t in result.conversation_history
        if t.get("role", "").lower() == "critic"
    ]
    if critic_turns:
        print(f"\n  Critic spoke {len(critic_turns)} time(s). Key contribution:")
        print(_preview(critic_turns[0].get("content", ""), max_chars=400))
    else:
        print("  (Critic did not contribute in this run)")

    print(f"\n  Execution time  : {result.execution_time:.1f}s")
    print(f"  Tokens used     : {result.token_usage:,}")
