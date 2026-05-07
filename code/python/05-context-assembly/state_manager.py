"""State Manager — explicit conversation state that survives context truncation.

The message list is temporary.  State is durable.

Every turn updates :class:`ConversationState`, which is then injected back
into the system prompt so the agent never loses track of goals, collected
user information, or recommendations it has already made.

Key classes:

- :class:`ConversationState` — serialisable state bag that survives truncation.
- :class:`StateManager` — updates and queries state; selects recovery actions.

See: docs/04-context-engineering/04-multi-turn-context-management.md
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from context_budget import count_tokens


# ---------------------------------------------------------------------------
# ConversationState
# ---------------------------------------------------------------------------


@dataclass
class ConversationState:
    """Explicit state that persists across message list truncations.

    This is the agent's durable memory — it survives context window limits.
    Inject :meth:`to_prompt_context` into every system prompt so the agent
    stays oriented even after the earliest turns are truncated away.
    """

    # Task tracking
    current_goal: Optional[str] = None
    subtasks_completed: list[str] = field(default_factory=list)
    subtasks_pending: list[str] = field(default_factory=list)
    goal_set_at_turn: int = 0

    # User context
    user_name: Optional[str] = None
    user_preferences: dict = field(default_factory=dict)
    user_provided_info: dict = field(default_factory=dict)

    # Agent context
    agent_recommendations: list[str] = field(default_factory=list)
    agent_questions_asked: list[str] = field(default_factory=list)
    agent_mode: str = "general"

    # Conversation health
    turns_since_goal_mentioned: int = 0
    user_frustration_signals: int = 0
    topic_changes: list[str] = field(default_factory=list)
    turn_count: int = 0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_prompt_context(self) -> str:
        """Generate a compact state summary for prompt injection.

        The output is intentionally terse — it will consume part of the
        system-prompt token budget on every call.

        Returns:
            A multi-line string ready to embed inside a ``## Conversation
            State`` section of the system prompt, or an empty string when
            the state carries no useful information.
        """
        parts: list[str] = []

        if self.user_name:
            parts.append(f"User: {self.user_name}")

        if self.current_goal:
            parts.append(f"Current goal: {self.current_goal}")

        if self.subtasks_completed:
            parts.append("Completed: " + "; ".join(self.subtasks_completed))

        if self.subtasks_pending:
            parts.append("Pending: " + "; ".join(self.subtasks_pending))

        if self.agent_recommendations:
            # Trim to avoid runaway growth
            recs = self.agent_recommendations[-5:]
            parts.append("Previous recommendations: " + "; ".join(recs))

        if self.agent_questions_asked:
            qs = self.agent_questions_asked[-5:]
            parts.append("Already asked about: " + "; ".join(qs))

        if self.user_provided_info:
            info_str = ", ".join(
                f"{k}={v}" for k, v in self.user_provided_info.items()
            )
            parts.append(f"User has provided: {info_str}")

        if self.user_preferences:
            pref_str = ", ".join(
                f"{k}={v}" for k, v in self.user_preferences.items()
            )
            parts.append(f"User preferences: {pref_str}")

        if self.agent_mode != "general":
            parts.append(f"Agent mode: {self.agent_mode}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialise state for persistence (JSON-safe)."""
        return {
            "current_goal": self.current_goal,
            "subtasks_completed": list(self.subtasks_completed),
            "subtasks_pending": list(self.subtasks_pending),
            "goal_set_at_turn": self.goal_set_at_turn,
            "user_name": self.user_name,
            "user_preferences": dict(self.user_preferences),
            "user_provided_info": dict(self.user_provided_info),
            "agent_recommendations": list(self.agent_recommendations),
            "agent_questions_asked": list(self.agent_questions_asked),
            "agent_mode": self.agent_mode,
            "turns_since_goal_mentioned": self.turns_since_goal_mentioned,
            "user_frustration_signals": self.user_frustration_signals,
            "topic_changes": list(self.topic_changes),
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        """Deserialise state from a persistence dictionary."""
        s = cls()
        s.current_goal = data.get("current_goal")
        s.subtasks_completed = data.get("subtasks_completed", [])
        s.subtasks_pending = data.get("subtasks_pending", [])
        s.goal_set_at_turn = data.get("goal_set_at_turn", 0)
        s.user_name = data.get("user_name")
        s.user_preferences = data.get("user_preferences", {})
        s.user_provided_info = data.get("user_provided_info", {})
        s.agent_recommendations = data.get("agent_recommendations", [])
        s.agent_questions_asked = data.get("agent_questions_asked", [])
        s.agent_mode = data.get("agent_mode", "general")
        s.turns_since_goal_mentioned = data.get("turns_since_goal_mentioned", 0)
        s.user_frustration_signals = data.get("user_frustration_signals", 0)
        s.topic_changes = data.get("topic_changes", [])
        s.turn_count = data.get("turn_count", 0)
        return s


# ---------------------------------------------------------------------------
# Information extraction helpers (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

# Patterns: "my <key> is <value>", "order number 12345", etc.
_INFO_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("order_number",   re.compile(r"\border\s*(?:number|#|num)?\s*(?:is\s*)?[:#]?\s*([A-Z0-9\-]{4,20})", re.I)),
    ("name",           re.compile(r"\b(?:my\s+name\s+is|i(?:'m| am)\s+called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I)),
    ("budget",         re.compile(r"\b(?:budget|spend|cost).*?[\$£€]?\s*([0-9][0-9,]*(?:\.[0-9]{2})?)\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)?", re.I)),
    ("date",           re.compile(r"\b(?:on|by|for|at)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", re.I)),
    ("location",       re.compile(r"\b(?:in|to|from|at|near)\s+([A-Z][a-zA-Z\s]{3,30}?)(?:\s+(?:airport|city|station|office|hotel)|\s*[,.])", re.I)),
    ("preference",     re.compile(r"\bi\s+prefer\s+([^.!?]{3,60})", re.I)),
    ("email",          re.compile(r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")),
]

_FRUSTRATION_SIGNALS = [
    r"\b(?:already\s+told|said\s+(?:this|that)|you\s+asked\s+(?:me\s+)?(?:that|this|about))\b",
    r"\b(?:again|again\?|for\s+the\s+\w+\s+time)\b",
    r"\b(?:stop\s+repeating|stop\s+asking|can.t\s+you|why\s+(?:do|are|is|can.t))\b",
    r"\b(?:frustrated|annoyed|useless|terrible|awful)\b",
    r"!{2,}",   # multiple exclamation marks
]
_FRUSTRATION_RE = re.compile("|".join(_FRUSTRATION_SIGNALS), re.I)

_GOAL_KEYWORDS = re.compile(
    r"\b(?:i\s+(?:need|want|would\s+like|must|have\s+to)|"
    r"(?:please|can\s+you|could\s+you|help\s+me\s+(?:to|with)?)\s+|"
    r"(?:my\s+goal|the\s+goal|objective|task)\s+is)\b",
    re.I,
)


def _extract_user_info(message: str) -> dict:
    """Extract structured information from a user message using regex patterns.

    Returns a dict of {key: value} pairs found in *message*.
    """
    found: dict = {}
    for key, pattern in _INFO_PATTERNS:
        m = pattern.search(message)
        if m:
            value = m.group(1).strip(" ,.")
            # Don't overwrite explicit keys with weaker matches
            if key not in found:
                found[key] = value
    return found


def _has_frustration(message: str) -> bool:
    """Return True if the message contains frustration signals."""
    return bool(_FRUSTRATION_RE.search(message))


def _goal_relevance(message: str, current_goal: Optional[str]) -> float:
    """Estimate how relevant *message* is to *current_goal* (0.0–1.0).

    Uses simple keyword overlap — good enough for drift detection without
    an LLM call.
    """
    if not current_goal:
        return 1.0
    goal_words = set(re.findall(r"\b\w{4,}\b", current_goal.lower()))
    msg_words  = set(re.findall(r"\b\w{4,}\b", message.lower()))
    if not goal_words:
        return 1.0
    overlap = len(goal_words & msg_words) / len(goal_words)
    return overlap


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------


class StateManager:
    """Manages conversation state across turns.

    Updates :class:`ConversationState` based on user input and agent output,
    and selects the appropriate recovery action when the conversation drifts.

    Example::

        mgr = StateManager()
        mgr.set_goal("Book a flight to London", subtasks=[
            "collect travel dates",
            "select seat preference",
            "confirm payment",
        ])
        changes = mgr.process_user_turn("I'd like to fly on June 15")
        print(changes)
    """

    # Drift threshold: turns without mentioning the goal
    DRIFT_TURNS_THRESHOLD: int = 5
    # Checkpoint every N turns
    CHECKPOINT_INTERVAL: int = 10
    # Max questions / recommendations to track
    MAX_TRACKED_ITEMS: int = 20

    def __init__(self) -> None:
        self.state = ConversationState()

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def set_goal(self, goal: str, subtasks: list[str] | None = None) -> None:
        """Set the current conversation goal.

        Args:
            goal:     One-sentence description of what the user wants.
            subtasks: Optional ordered list of steps to complete the goal.
        """
        self.state.current_goal = goal
        self.state.subtasks_pending = list(subtasks or [])
        self.state.subtasks_completed = []
        self.state.goal_set_at_turn = self.state.turn_count
        self.state.turns_since_goal_mentioned = 0

    def mark_subtask_complete(self, subtask: str) -> None:
        """Mark a subtask as completed, moving it from pending to completed.

        Accepts both exact matches and substring matches so callers don't
        need to know the precise subtask string.
        """
        # Exact match first
        if subtask in self.state.subtasks_pending:
            self.state.subtasks_pending.remove(subtask)
            self.state.subtasks_completed.append(subtask)
            return
        # Substring / case-insensitive match
        lower = subtask.lower()
        for item in list(self.state.subtasks_pending):
            if lower in item.lower() or item.lower() in lower:
                self.state.subtasks_pending.remove(item)
                self.state.subtasks_completed.append(item)
                return

    def check_goal_drift(self) -> bool:
        """Return True if the conversation has drifted from the current goal."""
        if not self.state.current_goal:
            return False
        return self.state.turns_since_goal_mentioned >= self.DRIFT_TURNS_THRESHOLD

    def check_goal_complete(self) -> bool:
        """Return True when all subtasks are completed (and a goal exists)."""
        if not self.state.current_goal:
            return False
        # If subtasks were defined, all must be completed
        if self.state.subtasks_pending or self.state.subtasks_completed:
            return len(self.state.subtasks_pending) == 0
        return False

    # ------------------------------------------------------------------
    # Per-turn updates
    # ------------------------------------------------------------------

    def process_user_turn(self, user_message: str) -> dict:
        """Process user input and return a dict describing state changes.

        Performs:
        1. Goal-relevance check → updates ``turns_since_goal_mentioned``.
        2. Info extraction → updates ``user_provided_info``.
        3. Frustration detection → increments ``user_frustration_signals``.
        4. Topic-change detection → appends to ``topic_changes``.
        5. Turn counter increment.

        Returns:
            A dict with keys: ``goal_detected``, ``info_extracted``,
            ``frustration_detected``, ``topic_changed``, ``turn``.
        """
        self.state.turn_count += 1
        changes: dict = {
            "goal_detected":      False,
            "info_extracted":     {},
            "frustration_detected": False,
            "topic_changed":      False,
            "turn":               self.state.turn_count,
        }

        # ── Relevance / drift tracking ──────────────────────────────────
        relevance = _goal_relevance(user_message, self.state.current_goal)
        if relevance < 0.25 and self.state.current_goal:
            self.state.turns_since_goal_mentioned += 1
            if self.state.turns_since_goal_mentioned >= self.DRIFT_TURNS_THRESHOLD:
                snippet = user_message[:80].strip()
                # Only record the first drift per distinct topic fragment
                if not self.state.topic_changes or self.state.topic_changes[-1] != snippet:
                    self.state.topic_changes.append(snippet)
                changes["topic_changed"] = True
        else:
            # Message is on-topic; reset the drift counter
            self.state.turns_since_goal_mentioned = 0

        # ── Goal detection from message ─────────────────────────────────
        if _GOAL_KEYWORDS.search(user_message):
            changes["goal_detected"] = True

        # ── Information extraction ──────────────────────────────────────
        extracted = _extract_user_info(user_message)
        if extracted:
            self.state.user_provided_info.update(extracted)
            # Also capture the user's name in the dedicated field
            if "name" in extracted and not self.state.user_name:
                self.state.user_name = extracted["name"]
            changes["info_extracted"] = extracted

        # ── Frustration signals ─────────────────────────────────────────
        if _has_frustration(user_message):
            self.state.user_frustration_signals += 1
            changes["frustration_detected"] = True

        return changes

    def process_agent_turn(
        self,
        agent_response: str,
        tool_calls: list | None = None,
    ) -> dict:
        """Process agent output and return a dict describing state changes.

        Performs:
        1. Recommendation tracking.
        2. Question tracking.
        3. Subtask-completion inference from tool calls.
        4. Goal-completion check.

        Returns:
            A dict with keys: ``recommendations_added``, ``questions_asked``,
            ``subtasks_completed``, ``goal_complete``.
        """
        changes: dict = {
            "recommendations_added": 0,
            "questions_asked":       0,
            "subtasks_completed":    0,
            "goal_complete":         False,
        }

        # ── Recommendations ─────────────────────────────────────────────
        rec_pattern = re.compile(
            r"\b(?:I\s+(?:recommend|suggest|advise|propose)|"
            r"you\s+(?:should|could|might\s+want\s+to))\s+([^.!?]{5,120})",
            re.I,
        )
        for m in rec_pattern.finditer(agent_response):
            rec = m.group(1).strip().rstrip(".,")
            if rec not in self.state.agent_recommendations:
                self.state.agent_recommendations.append(rec)
                changes["recommendations_added"] += 1
        # Cap list size
        if len(self.state.agent_recommendations) > self.MAX_TRACKED_ITEMS:
            self.state.agent_recommendations = (
                self.state.agent_recommendations[-self.MAX_TRACKED_ITEMS:]
            )

        # ── Questions asked ─────────────────────────────────────────────
        # Split on sentence boundaries to find individual questions
        sentences = re.split(r"(?<=[.!?])\s+", agent_response)
        for sentence in sentences:
            if "?" in sentence:
                q = sentence.strip()[:120]
                if q and q not in self.state.agent_questions_asked:
                    self.state.agent_questions_asked.append(q)
                    changes["questions_asked"] += 1
        if len(self.state.agent_questions_asked) > self.MAX_TRACKED_ITEMS:
            self.state.agent_questions_asked = (
                self.state.agent_questions_asked[-self.MAX_TRACKED_ITEMS:]
            )

        # ── Tool-call → subtask completion ──────────────────────────────
        if tool_calls:
            for call in tool_calls:
                fn_name = ""
                if isinstance(call, dict):
                    fn_name = call.get("function", {}).get("name", "") or call.get("name", "")
                elif hasattr(call, "function"):
                    fn_name = getattr(call.function, "name", "")
                if fn_name:
                    # Try to match against pending subtasks
                    for pending in list(self.state.subtasks_pending):
                        if fn_name.replace("_", " ").lower() in pending.lower():
                            self.mark_subtask_complete(pending)
                            changes["subtasks_completed"] += 1

        # ── Goal completion ─────────────────────────────────────────────
        if self.check_goal_complete():
            changes["goal_complete"] = True

        return changes

    # ------------------------------------------------------------------
    # Recovery actions
    # ------------------------------------------------------------------

    def get_recovery_action(self) -> str | None:
        """Return the most appropriate recovery action, or None.

        Priority order (highest first):

        1. ``"ask_user_to_clarify"``   — high frustration (≥ 3 signals)
        2. ``"remind_goal"``           — goal has drifted
        3. ``"inject_checkpoint"``     — periodic progress check
        4. ``"summarize_progress"``    — all subtasks done but goal not closed
        5. ``None``                    — conversation is healthy
        """
        if self.state.user_frustration_signals >= 3:
            return "ask_user_to_clarify"

        if self.check_goal_drift():
            return "remind_goal"

        if (
            self.state.turn_count > 0
            and self.state.turn_count % self.CHECKPOINT_INTERVAL == 0
            and self.state.current_goal
        ):
            return "inject_checkpoint"

        if (
            self.state.current_goal
            and not self.state.subtasks_pending
            and self.state.subtasks_completed
        ):
            return "summarize_progress"

        return None

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    def build_system_prompt_with_state(self, base_prompt: str) -> str:
        """Augment *base_prompt* with the current conversation state.

        The state summary is appended as a ``## Conversation State`` section.
        If there is no meaningful state yet the prompt is returned unchanged.
        """
        ctx = self.state.to_prompt_context()
        if not ctx:
            return base_prompt
        return (
            f"{base_prompt}\n\n"
            f"## Conversation State (maintained across turns)\n"
            f"{ctx}\n\n"
            "Use this state to maintain continuity. "
            "Do not repeat questions already asked. "
            "Do not contradict previous recommendations."
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    """Simulate a 30-turn conversation and demonstrate state management."""

    print("=" * 70)
    print("STATE MANAGER DEMO")
    print("=" * 70)

    mgr = StateManager()

    # ── Goal setting ────────────────────────────────────────────────────
    mgr.set_goal(
        "Book a round-trip flight to London for the user",
        subtasks=[
            "collect travel dates",
            "collect seat preference",
            "confirm passenger name",
            "process payment",
        ],
    )
    print(f"\nGoal set: {mgr.state.current_goal}")
    print(f"Subtasks: {mgr.state.subtasks_pending}\n")

    # ── Simulated turns ─────────────────────────────────────────────────
    conversation = [
        # Turns 1–5: on-topic
        ("I need to book a flight to London please.",
         "I can help with that. When would you like to travel?"),
        ("I want to fly on June 15 and return June 22.",
         "Got it — June 15 outbound, June 22 return. Do you prefer window or aisle?"),
        ("I prefer window seats please.",
         "Window it is! May I have your name for the booking?"),
        ("My name is Alice Johnson. My budget is $800.",
         "Thank you Alice. I recommend checking BA flight 123 within your $800 budget."),
        ("Great, let's go with that.",
         "To confirm payment, shall I use your saved card?"),
        # Turn 6 agent marks subtask complete
        # Turns 6–12: topic drift
        ("Actually, do you know any good restaurants in London?",
         "Some popular spots include The Ivy and Sketch."),
        ("What about museums?",
         "The British Museum and Tate Modern are excellent."),
        ("What's the weather like in London in June?",
         "London in June averages 18–22 °C with occasional rain."),
        ("I also need a hotel recommendation.",
         "I suggest The Savoy or a budget-friendly Premier Inn."),
        ("What's the best tube route from Heathrow?",
         "Take the Piccadilly line — it goes directly to central London."),
        ("Any tips for avoiding jet lag?",
         "Stay hydrated and adjust to local time immediately."),
        ("What currency should I bring?",
         "British Pounds (GBP). You can exchange at the airport."),
        # Turns 13–20: continued drift
        ("Is London expensive?",
         "It can be — budget around £80–£150 per day for accommodation."),
        ("What's the best time to visit Buckingham Palace?",
         "Morning queues are shorter; try arriving by 9 AM."),
        ("Do I need a visa?",
         "US citizens don't need a visa for short stays."),
        ("What power adapter do I need?",
         "A Type G adapter (3-pin) is required for UK sockets."),
        ("Can I use my phone there?",
         "Most modern phones work on UK networks; check roaming charges."),
        ("Is it safe to drink tap water?",
         "Yes, UK tap water is safe and of high quality."),
        ("What language do they speak?",
         "English is the primary language."),
        ("Any etiquette tips?",
         "Queuing is very important — never jump a queue."),
        # Turns 21–25: back on topic
        ("Sorry, let me get back to the flight booking.",
         "Of course! We were about to confirm payment for flight BA123."),
        ("Yes please, use my Visa card ending 4242.",
         "Processing payment..."),
        ("Is the booking confirmed?",
         "Yes, booking confirmed! Confirmation number: LDN-2024-8821."),
        ("Can I get an email confirmation?",
         "I recommend keeping the confirmation email for check-in."),
        ("Perfect, thanks!",
         "You're welcome, Alice. Have a wonderful trip to London!"),
        # Turns 26–30: new topic
        ("One more thing — can you book me a rental car?",
         "Of course! What dates would you need the car?"),
        ("Same dates, June 15–22.",
         "Do you prefer automatic or manual transmission?"),
        ("Automatic please, and I want GPS included.",
         "I suggest an Avis mid-size automatic with GPS — around $45/day."),
        ("That works!",
         "I'll proceed with the Avis booking."),
        ("Great, can you do it?",
         "Rental car booked — confirmation: CAR-8821-LDN."),
    ]

    for i, (user_msg, agent_msg) in enumerate(conversation, start=1):
        user_changes = mgr.process_user_turn(user_msg)
        agent_changes = mgr.process_agent_turn(agent_msg)

        # Mark some subtasks complete manually
        if i == 2:
            mgr.mark_subtask_complete("collect travel dates")
        if i == 3:
            mgr.mark_subtask_complete("collect seat preference")
        if i == 4:
            mgr.mark_subtask_complete("confirm passenger name")
        if i == 23:
            mgr.mark_subtask_complete("process payment")

        # Show state snapshots
        if i in {5, 15, 25}:
            print(f"\n{'─'*60}")
            print(f"TURN {i} STATE SNAPSHOT")
            print(f"{'─'*60}")
            print(mgr.state.to_prompt_context() or "(empty state)")
            print(f"\nTurn count: {mgr.state.turn_count}")
            print(f"Drift counter: {mgr.state.turns_since_goal_mentioned}")
            print(f"Frustration signals: {mgr.state.user_frustration_signals}")
            recovery = mgr.get_recovery_action()
            print(f"Recovery action: {recovery!r}")

        # Highlight drift detection
        if mgr.check_goal_drift():
            recovery = mgr.get_recovery_action()
            if recovery == "remind_goal":
                print(f"\n[Turn {i}] ⚠  DRIFT DETECTED — recovery: '{recovery}'")
                print(f"   Injecting: [REMINDER: Your goal is '{mgr.state.current_goal}']")

    # ── Final state ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL STATE")
    print(f"{'='*60}")
    print(mgr.state.to_prompt_context())

    # ── Serialisation roundtrip ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PERSISTENCE ROUNDTRIP")
    print(f"{'='*60}")
    serialised = json.dumps(mgr.state.to_dict(), indent=2)
    print(f"Serialised ({len(serialised)} bytes)")
    restored = ConversationState.from_dict(json.loads(serialised))
    assert restored.current_goal == mgr.state.current_goal
    assert restored.user_name == mgr.state.user_name
    assert restored.user_provided_info == mgr.state.user_provided_info
    print("✓ State restored — goal, user name, and user info match")

    # ── Recovery actions ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RECOVERY ACTION WALKTHROUGH")
    print(f"{'='*60}")

    test_mgr = StateManager()
    test_mgr.set_goal("Order pizza", subtasks=["choose toppings", "confirm address"])

    # Simulate drift
    for _ in range(6):
        test_mgr.process_user_turn("Tell me a joke")
    print(f"After 6 off-topic turns → {test_mgr.get_recovery_action()!r}")

    # Simulate frustration
    test_mgr2 = StateManager()
    test_mgr2.set_goal("Fix bug")
    test_mgr2.state.user_frustration_signals = 3
    print(f"After 3 frustration signals → {test_mgr2.get_recovery_action()!r}")

    # Simulate checkpoint
    test_mgr3 = StateManager()
    test_mgr3.set_goal("Plan vacation")
    test_mgr3.state.turn_count = 10
    print(f"At turn 10 (checkpoint interval) → {test_mgr3.get_recovery_action()!r}")

    # Simulate all subtasks complete
    test_mgr4 = StateManager()
    test_mgr4.set_goal("File tax return", subtasks=["gather docs"])
    test_mgr4.mark_subtask_complete("gather docs")
    print(f"All subtasks done → {test_mgr4.get_recovery_action()!r}")

    print("\nDemo complete.")


if __name__ == "__main__":
    _demo()
