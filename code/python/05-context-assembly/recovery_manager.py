"""Recovery Manager — goal detection and conversation health recovery.

When a long conversation drifts, stalls, or gets stuck in a repetition loop,
the recovery manager diagnoses what's wrong and injects the right intervention
into the next prompt.

Key classes:

- :class:`GoalDetector` — extracts and tracks user goals from messages.
- :class:`RecoveryManager` — diagnoses conversation health and generates
  recovery interventions.

Data classes:

- :class:`GoalResult` — output from goal detection.
- :class:`CompletionResult` — output from completion checking.
- :class:`ConversationIssue` — a diagnosed conversation problem.

See: docs/04-context-engineering/04-multi-turn-context-management.md
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from state_manager import StateManager


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GoalResult:
    """Output of :meth:`GoalDetector.detect_goal`.

    Attributes:
        goal:               One-sentence description of what the user wants.
        is_new_goal:        True if this is a new topic; False if continuing.
        supersedes_previous: True if this goal replaces an older goal.
        subtasks:           Ordered list of concrete steps to complete the goal.
        priority:           ``"high"`` / ``"medium"`` / ``"low"``.
        estimated_turns:    Rough estimate of how many agent turns are needed.
    """

    goal: str = ""
    is_new_goal: bool = True
    supersedes_previous: bool = False
    subtasks: list[str] = field(default_factory=list)
    priority: str = "medium"
    estimated_turns: int = 3


@dataclass
class CompletionResult:
    """Output of :meth:`GoalDetector.check_goal_completion`.

    Attributes:
        is_complete:        True if the goal appears to be accomplished.
        completion_pct:     Estimated percentage of completion (0–100).
        remaining_subtasks: Tasks that are not yet done.
        evidence:           Phrase from the conversation that indicates completion.
    """

    is_complete: bool = False
    completion_pct: float = 0.0
    remaining_subtasks: list[str] = field(default_factory=list)
    evidence: str = ""


@dataclass
class ConversationIssue:
    """A single diagnosed conversation problem.

    Attributes:
        type:             One of ``"goal_drift"``, ``"repetition"``,
                          ``"contradiction"``, ``"lost_context"``,
                          ``"frustration"``, ``"stalemate"``.
        severity:         ``"low"`` / ``"medium"`` / ``"high"``.
        description:      Human-readable description.
        suggested_action: One of the standard intervention action names.
    """

    type: str
    severity: str
    description: str
    suggested_action: str


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    try:
        import openai  # noqa: F401
        return bool(os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        return False


def _llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# GoalDetector
# ---------------------------------------------------------------------------

_GOAL_DETECTION_PROMPT = """\
Analyze the user's message and determine their goal.

Output JSON only — no other text:
{{
    "goal": "Clear one-sentence description of what the user wants",
    "is_new_goal": true,
    "supersedes_previous": false,
    "subtasks": ["step1", "step2"],
    "priority": "medium",
    "estimated_turns": 3
}}

Rules:
- is_new_goal: true if this is a new topic, false if continuing the previous goal
- supersedes_previous: true if this goal replaces the previous one
- subtasks: break complex goals into 2-5 concrete steps; empty list for simple goals
- priority: high if time-sensitive or critical; low if casual or exploratory
- estimated_turns: estimate how many assistant turns this goal requires

Previous conversation context:
{context}

User message:
{message}"""

_COMPLETION_CHECK_PROMPT = """\
Assess whether this conversation goal has been accomplished.

Goal: {goal}

Conversation summary:
{summary}

Output JSON only — no other text:
{{
    "is_complete": false,
    "completion_pct": 40.0,
    "remaining_subtasks": ["step still needed"],
    "evidence": "phrase that shows how far along we are"
}}"""

_TOPIC_RELEVANCE_PROMPT = """\
Score how much this user message relates to the stated goal.

Goal: {goal}
User message: {message}

Output a single JSON number between 0.0 and 1.0.
1.0 = directly on-topic.  0.0 = completely different subject.
Output the number only, no other text."""


class GoalDetector:
    """Detect and track user goals across conversation turns.

    Uses the OpenAI API when available; falls back to deterministic heuristics
    for offline use or testing.

    Args:
        model: OpenAI model to use for goal detection.
    """

    _GOAL_VERBS = re.compile(
        r"\b(?:need|want|would\s+like|must|have\s+to|trying\s+to|help\s+me"
        r"|please|can\s+you|could\s+you|looking\s+to)\b",
        re.I,
    )

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.use_llm = _llm_available()

    def detect_goal(
        self,
        user_message: str,
        conversation_context: str | None = None,
    ) -> GoalResult:
        """Detect the user's goal from a message.

        Args:
            user_message:          The raw user message.
            conversation_context:  Optional summary of previous turns for context.

        Returns:
            A :class:`GoalResult` describing the detected goal.
        """
        if self.use_llm:
            try:
                prompt = _GOAL_DETECTION_PROMPT.format(
                    context=conversation_context or "(none)",
                    message=user_message,
                )
                raw = _llm_call(prompt, self.model)
                # Strip markdown code fences if present
                raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                data = json.loads(raw)
                return GoalResult(
                    goal=data.get("goal", user_message[:80]),
                    is_new_goal=data.get("is_new_goal", True),
                    supersedes_previous=data.get("supersedes_previous", False),
                    subtasks=data.get("subtasks", []),
                    priority=data.get("priority", "medium"),
                    estimated_turns=data.get("estimated_turns", 3),
                )
            except Exception:
                pass  # Fall through to heuristic

        return self._heuristic_goal(user_message)

    def check_goal_completion(
        self, goal: str, conversation_summary: str
    ) -> CompletionResult:
        """Check whether *goal* has been accomplished.

        Args:
            goal:                  The stated goal string.
            conversation_summary:  Summary of the conversation so far.

        Returns:
            A :class:`CompletionResult`.
        """
        if self.use_llm:
            try:
                prompt = _COMPLETION_CHECK_PROMPT.format(
                    goal=goal,
                    summary=conversation_summary,
                )
                raw = _llm_call(prompt, self.model)
                raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                data = json.loads(raw)
                return CompletionResult(
                    is_complete=data.get("is_complete", False),
                    completion_pct=float(data.get("completion_pct", 0.0)),
                    remaining_subtasks=data.get("remaining_subtasks", []),
                    evidence=data.get("evidence", ""),
                )
            except Exception:
                pass

        return self._heuristic_completion(goal, conversation_summary)

    def detect_topic_change(
        self, current_goal: str, user_message: str
    ) -> float:
        """Score how much *user_message* relates to *current_goal* (0.0–1.0).

        Args:
            current_goal:  The currently active goal string.
            user_message:  The user's latest message.

        Returns:
            A float; 1.0 = directly related, 0.0 = completely different.
        """
        if self.use_llm:
            try:
                prompt = _TOPIC_RELEVANCE_PROMPT.format(
                    goal=current_goal,
                    message=user_message,
                )
                raw = _llm_call(prompt, self.model).strip()
                return max(0.0, min(1.0, float(raw)))
            except Exception:
                pass

        return self._heuristic_relevance(current_goal, user_message)

    # ------------------------------------------------------------------
    # Heuristic fallbacks (deterministic, no API key needed)
    # ------------------------------------------------------------------

    def _heuristic_goal(self, message: str) -> GoalResult:
        """Extract a goal from *message* using keyword heuristics."""
        is_goal = bool(self._GOAL_VERBS.search(message))
        # Simple goal: first sentence
        first_sentence = re.split(r"[.!?]", message)[0].strip()
        goal_text = first_sentence[:100] if first_sentence else message[:100]

        # Detect potential subtasks: imperatives after a comma or semicolon
        subtasks: list[str] = []
        for frag in re.split(r"[,;]", message):
            frag = frag.strip()
            if frag and len(frag) > 10 and frag != message:
                subtasks.append(frag[:60])
        subtasks = subtasks[:4]  # Cap at 4

        return GoalResult(
            goal=goal_text,
            is_new_goal=is_goal,
            supersedes_previous=False,
            subtasks=subtasks,
            priority="medium",
            estimated_turns=max(2, len(subtasks) + 1),
        )

    def _heuristic_completion(
        self, goal: str, summary: str
    ) -> CompletionResult:
        """Estimate goal completion from keyword presence in *summary*."""
        completion_words = re.compile(
            r"\b(?:confirmed|completed|done|booked|finished|ready|sent|paid|"
            r"success|all\s+set|thank\s+you|you're\s+all\s+set)\b",
            re.I,
        )
        matches = len(completion_words.findall(summary))
        pct = min(100.0, matches * 20.0)
        return CompletionResult(
            is_complete=pct >= 80.0,
            completion_pct=pct,
            remaining_subtasks=[],
            evidence=f"{matches} completion signal(s) in summary",
        )

    def _heuristic_relevance(self, goal: str, message: str) -> float:
        """Word-overlap relevance score."""
        goal_words  = set(re.findall(r"\b\w{4,}\b", goal.lower()))
        msg_words   = set(re.findall(r"\b\w{4,}\b", message.lower()))
        if not goal_words:
            return 1.0
        return len(goal_words & msg_words) / len(goal_words)


# ---------------------------------------------------------------------------
# RecoveryManager
# ---------------------------------------------------------------------------

# Intervention message templates
_INTERVENTIONS: dict[str, str] = {
    "goal_drift": (
        "[REMINDER: The current goal is: {goal}. "
        "Pending steps: {pending}. "
        "Please return focus to this goal, or ask the user if they want to change it.]"
    ),
    "repetition": (
        "[NOTE: The following information has already been provided by the user and "
        "should NOT be asked again: {already_asked}. "
        "Do not repeat these questions.]"
    ),
    "contradiction": (
        "[CONSISTENCY CHECK: Previous recommendations were: {recommendations}. "
        "Ensure any new suggestions are consistent with these, or explicitly "
        "acknowledge the change.]"
    ),
    "lost_context": (
        "[CONTEXT SUMMARY: Goal: {goal}. "
        "Completed: {completed}. Pending: {pending}. "
        "User info: {user_info}. Resume from this point.]"
    ),
    "frustration": (
        "[USER FRUSTRATION DETECTED: The user appears frustrated. "
        "Acknowledge their concern, apologise for any confusion, "
        "and focus on resolving their core need: {goal}.]"
    ),
    "stalemate": (
        "[STALEMATE: The conversation has not made progress for several turns. "
        "Offer a concrete next step or ask the user directly: "
        "'What would be most helpful right now?']"
    ),
}


class RecoveryManager:
    """Detect and recover from conversation health problems.

    Works alongside a :class:`~state_manager.StateManager` to monitor
    conversation health and generate targeted interventions.

    Args:
        state_manager: The conversation's :class:`~state_manager.StateManager`.

    Example::

        recovery = RecoveryManager(state_mgr)
        issues = recovery.diagnose()
        if issues:
            intervention = recovery.get_intervention(issues)
            messages.append({"role": "system", "content": intervention})
    """

    # Thresholds
    DRIFT_THRESHOLD: int = 5       # turns without goal mention → goal_drift
    FRUSTRATION_THRESHOLD: int = 2  # signals → frustration issue
    REPETITION_THRESHOLD: int = 3   # same questions asked → repetition issue
    STALEMATE_THRESHOLD: int = 8    # turns without progress → stalemate

    def __init__(self, state_manager: StateManager) -> None:
        self._sm = state_manager
        self._turns_without_progress: int = 0
        self._last_subtask_count: int = 0

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def diagnose(self) -> list[ConversationIssue]:
        """Diagnose current conversation health.

        Returns:
            A list of :class:`ConversationIssue` objects, sorted with the
            highest-severity issues first.  An empty list means the
            conversation is healthy.
        """
        issues: list[ConversationIssue] = []
        state = self._sm.state

        # ── Goal drift ──────────────────────────────────────────────────
        if (
            state.current_goal
            and state.turns_since_goal_mentioned >= self.DRIFT_THRESHOLD
        ):
            severity = "high" if state.turns_since_goal_mentioned >= 8 else "medium"
            issues.append(ConversationIssue(
                type="goal_drift",
                severity=severity,
                description=(
                    f"Conversation has drifted {state.turns_since_goal_mentioned} "
                    f"turns away from goal: '{state.current_goal}'"
                ),
                suggested_action="remind_goal",
            ))

        # ── User frustration ────────────────────────────────────────────
        if state.user_frustration_signals >= self.FRUSTRATION_THRESHOLD:
            severity = "high" if state.user_frustration_signals >= 3 else "medium"
            issues.append(ConversationIssue(
                type="frustration",
                severity=severity,
                description=(
                    f"User frustration detected: "
                    f"{state.user_frustration_signals} signal(s)"
                ),
                suggested_action="ask_user_to_clarify",
            ))

        # ── Repetition ──────────────────────────────────────────────────
        if len(state.agent_questions_asked) >= self.REPETITION_THRESHOLD:
            # Count how many questions have similar wording (crude dedup)
            unique_q: set[str] = set()
            duplicates: int = 0
            for q in state.agent_questions_asked:
                key = re.sub(r"\W+", " ", q.lower()).strip()[:30]
                if key in unique_q:
                    duplicates += 1
                else:
                    unique_q.add(key)
            if duplicates >= 2:
                issues.append(ConversationIssue(
                    type="repetition",
                    severity="medium",
                    description=f"Agent has repeated similar questions {duplicates} time(s)",
                    suggested_action="remind_goal",
                ))

        # ── Stalemate ───────────────────────────────────────────────────
        current_completed = len(state.subtasks_completed)
        if current_completed > self._last_subtask_count:
            self._last_subtask_count = current_completed
            self._turns_without_progress = 0
        elif state.current_goal:
            self._turns_without_progress += 1

        if self._turns_without_progress >= self.STALEMATE_THRESHOLD:
            issues.append(ConversationIssue(
                type="stalemate",
                severity="high",
                description=(
                    f"No progress in {self._turns_without_progress} turns; "
                    f"pending: {state.subtasks_pending}"
                ),
                suggested_action="inject_checkpoint",
            ))

        # Sort by severity
        _sev_order = {"high": 0, "medium": 1, "low": 2}
        issues.sort(key=lambda i: _sev_order.get(i.severity, 3))
        return issues

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------

    def get_intervention(self, issues: list[ConversationIssue]) -> str:
        """Generate an intervention message to inject into the prompt.

        Combines the most severe issue type with the current conversation
        state to produce a targeted message.

        Args:
            issues: List returned by :meth:`diagnose`.

        Returns:
            A string ready to append to ``messages`` as a ``system`` role
            message.  Returns an empty string when *issues* is empty.
        """
        if not issues:
            return ""

        state = self._sm.state
        # Use the highest-severity issue as the primary driver
        primary = issues[0]

        template = _INTERVENTIONS.get(primary.type, "")
        if not template:
            return f"[CONVERSATION HEALTH ISSUE: {primary.description}]"

        return template.format(
            goal=state.current_goal or "(unknown)",
            pending=", ".join(state.subtasks_pending) or "none",
            completed=", ".join(state.subtasks_completed) or "none",
            already_asked="; ".join(state.agent_questions_asked[-5:]) or "none",
            recommendations="; ".join(state.agent_recommendations[-5:]) or "none",
            user_info=", ".join(
                f"{k}={v}" for k, v in state.user_provided_info.items()
            ) or "none",
        )

    def should_reset(self, issues: list[ConversationIssue]) -> bool:
        """Return True if the conversation is beyond recovery.

        Resets are warranted when there are multiple high-severity issues
        or when frustration is combined with stalemate.

        Args:
            issues: List returned by :meth:`diagnose`.
        """
        high_count   = sum(1 for i in issues if i.severity == "high")
        issue_types  = {i.type for i in issues}
        stalemate    = "stalemate" in issue_types
        frustration  = "frustration" in issue_types
        return high_count >= 2 or (stalemate and frustration)

    def generate_progress_report(self) -> str:
        """Generate a user-facing progress summary.

        Returns:
            A short human-readable string describing what has been done
            and what remains.
        """
        state = self._sm.state
        parts: list[str] = []

        if state.current_goal:
            parts.append(f"Goal: {state.current_goal}")

        if state.subtasks_completed:
            parts.append("Done: " + ", ".join(state.subtasks_completed))

        if state.subtasks_pending:
            parts.append("Still to do: " + ", ".join(state.subtasks_pending))
        else:
            parts.append("All steps complete.")

        if state.agent_recommendations:
            parts.append(
                "Recommendations made: " + "; ".join(state.agent_recommendations[-3:])
            )

        return "\n".join(parts) if parts else "No active goal."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    """Simulate a 40-turn conversation and demonstrate recovery."""

    print("=" * 70)
    print("RECOVERY MANAGER DEMO")
    print("=" * 70)

    state_mgr = StateManager()
    recovery = RecoveryManager(state_mgr)
    detector = GoalDetector()

    state_mgr.set_goal(
        "Help user plan a vacation to Italy",
        subtasks=["choose cities", "book flights", "arrange accommodation", "plan itinerary"],
    )

    # Phase 1: On-topic turns (1–8)
    on_topic_turns = [
        ("I want to plan a trip to Italy next summer.", "Great! Which cities interest you most?"),
        ("I'd love to visit Rome and Florence.", "Excellent choices. How many days are you planning?"),
        ("About 10 days. My budget is $3000.", "For 10 days with $3000, you can do Rome (4 days) and Florence (6 days) comfortably."),
        ("I prefer direct flights from New York.", "Aleph Airlines and ITA fly direct JFK→FCO. Shall I check availability?"),
        ("Yes please. I need window seats.", "Searching direct flights..."),
        ("What's the best hotel in Rome?", "I recommend Hotel Eden near Villa Borghese — 5-star within budget."),
        ("Book it for June 15–19 please.", "Hotel Eden, Rome, June 15–19. Confirmed!"),
        ("Now Florence hotels.", "Looking at Florence — Hotel Lungarno on the Arno is excellent."),
    ]

    # Phase 2: Drift turns (9–18)
    drift_turns = [
        ("By the way, do you know good Italian recipes?", "Carbonara and Cacio e Pepe are Roman classics."),
        ("How do I make Carbonara?", "You need guanciale, eggs, Pecorino, and black pepper."),
        ("What about Tiramisu?", "Tiramisu uses mascarpone, ladyfingers, espresso, and cocoa."),
        ("Is Italian wine expensive?", "Chianti is affordable; Barolo is pricier."),
        ("What's the difference between Prosecco and Champagne?", "Prosecco is Italian; Champagne is French."),
        ("Should I learn Italian?", "Locals appreciate any attempt — buongiorno goes a long way."),
        ("How do I say 'where is the bathroom'?", "Dov'è il bagno? — and point with a smile."),
        ("What's the Italian word for train?", "Treno. The Trenitalia network connects all major cities."),
        ("Is it safe to drive in Italy?", "City driving is hectic; trains are safer for tourists."),
        ("What's the speed limit on Italian highways?", "130 km/h on autostrade."),
    ]

    # Phase 3: More drift + frustration (19–28)
    frustration_turns = [
        ("You already asked me about flight preferences!", "I apologise for repeating myself."),
        ("And you keep asking about window seats!", "You're right — I have that noted."),
        ("Why can't you remember what I said??", "I'm sorry for the confusion."),
        ("This is useless.", "I understand your frustration. Let me refocus."),
        ("What's the Vatican opening hours?", "The Vatican Museums open at 9 AM, last entry 4 PM."),
        ("Do I need to book in advance?", "Yes — Vatican tickets sell out weeks ahead."),
        ("How about the Uffizi Gallery?", "Book Uffizi tickets at least a week in advance."),
        ("Colosseum?", "Book Colosseum tickets online to skip the queue."),
        ("Is the Sistine Chapel inside the Vatican?", "Yes, it's inside the Vatican Museums complex."),
        ("How long does the Vatican tour take?", "Allow 3–4 hours for the museums and Sistine Chapel."),
    ]

    # Phase 4: Recovery (29–40)
    recovery_turns = [
        ("Sorry, let me get back to the actual trip planning.", "Of course! We had confirmed Rome hotels and needed Florence."),
        ("Book Hotel Lungarno Florence, June 19–25.", "Hotel Lungarno Florence, June 19–25. Confirmed!"),
        ("Now the flights.", "Searching JFK→FCO direct for June 15..."),
        ("Book the Aleph flight.", "Aleph Airlines JFK→FCO June 15. Booking..."),
        ("And the return, June 25.", "Return FCO→JFK June 25. Confirmed."),
        ("Great. What's left?", "Still pending: plan itinerary."),
        ("Can you create a day-by-day itinerary?", "Day 1 Rome: Colosseum + Forum. Day 2: Vatican. Day 3: Trastevere. Day 4: train to Florence. Day 5-10: Florence museums and day trips."),
        ("Perfect. Book a day trip to Cinque Terre.", "Day trip to Cinque Terre from Florence — booked!"),
        ("I think we're all set!", "Your Italy trip is fully planned — flights, hotels, and itinerary confirmed."),
        ("Thank you so much!", "You're welcome! Buon viaggio!"),
        ("One last thing — travel insurance?", "I recommend World Nomads for comprehensive EU coverage."),
        ("Add it.", "Travel insurance added. Total trip cost: $2,840. Enjoy Italy!"),
    ]

    all_turns = on_topic_turns + drift_turns + frustration_turns + recovery_turns

    print(f"\nSimulating {len(all_turns)}-turn conversation...\n")

    for turn_num, (user_msg, agent_msg) in enumerate(all_turns, start=1):
        state_mgr.process_user_turn(user_msg)
        state_mgr.process_agent_turn(agent_msg)

        # Mark some subtasks complete
        if turn_num == 8:
            state_mgr.mark_subtask_complete("choose cities")
        if turn_num == 30:
            state_mgr.mark_subtask_complete("arrange accommodation")
        if turn_num == 34:
            state_mgr.mark_subtask_complete("book flights")
        if turn_num == 37:
            state_mgr.mark_subtask_complete("plan itinerary")

        # Goal detection at key turns
        if turn_num in {1, 10, 20, 29}:
            result = detector.detect_goal(user_msg)
            print(f"[Turn {turn_num}] Goal detection: '{result.goal[:60]}'")
            print(f"           is_new_goal={result.is_new_goal}, "
                  f"estimated_turns={result.estimated_turns}")

        # Diagnose at turn 30
        if turn_num == 30:
            print(f"\n{'─'*60}")
            print(f"DIAGNOSIS AT TURN {turn_num}")
            print(f"{'─'*60}")
            issues = recovery.diagnose()
            if issues:
                for issue in issues:
                    print(f"  [{issue.severity.upper()}] {issue.type}: {issue.description}")
                intervention = recovery.get_intervention(issues)
                print(f"\nIntervention injected:\n{intervention}")
                print(f"\nShould reset: {recovery.should_reset(issues)}")
            else:
                print("  Conversation is healthy.")

    # Final progress report
    print(f"\n{'='*60}")
    print("FINAL PROGRESS REPORT")
    print(f"{'='*60}")
    print(recovery.generate_progress_report())

    print(f"\n{'='*60}")
    print("FINAL CONVERSATION STATE")
    print(f"{'='*60}")
    print(state_mgr.state.to_prompt_context())

    print("\nDemo complete.")


if __name__ == "__main__":
    _demo()
