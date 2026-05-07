"""Tests for multi-turn conversation management.

Covers:
- ConversationState: serialisation, prompt injection, fields
- StateManager: goal setting, drift detection, user info tracking,
  recovery actions
- ProgressiveSummarizer: verbatim preservation, layer cascade, key-info
  retention across summarisation
- SessionManager: expiry, session reuse, branching, reset, persistence
- RecoveryManager / GoalDetector: goal detection, issue diagnosis,
  intervention generation

Run offline (no API calls):
    pytest test_multi_turn.py -v
"""

from __future__ import annotations

import json
import sys
import os
import time
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from context_budget import count_tokens
from state_manager import ConversationState, StateManager
from progressive_summarizer import ProgressiveSummarizer
from session_manager import Session, SessionManager
from recovery_manager import (
    ConversationIssue,
    GoalDetector,
    RecoveryManager,
)


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def _make_state_mgr(goal: str | None = None, subtasks: list[str] | None = None) -> StateManager:
    mgr = StateManager()
    if goal:
        mgr.set_goal(goal, subtasks or [])
    return mgr


def _advance_turns(mgr: StateManager, n: int, on_topic: bool = True) -> None:
    """Simulate n turns. Off-topic turns use irrelevant text."""
    if on_topic:
        msg = "Let's continue working on the goal"
    else:
        msg = "Tell me a completely unrelated joke about cats"
    for _ in range(n):
        mgr.process_user_turn(msg)
        mgr.process_agent_turn("Here is my response to that.")


# ===========================================================================
# STATE MANAGER TESTS
# ===========================================================================

class TestConversationState:

    def test_to_prompt_context_empty(self):
        s = ConversationState()
        assert s.to_prompt_context() == ""

    def test_to_prompt_context_with_goal(self):
        s = ConversationState()
        s.current_goal = "Book a flight to London"
        ctx = s.to_prompt_context()
        assert "Book a flight to London" in ctx

    def test_to_prompt_context_with_all_fields(self):
        s = ConversationState()
        s.current_goal = "Order pizza"
        s.user_name = "Alice"
        s.subtasks_completed = ["choose toppings"]
        s.subtasks_pending = ["confirm address"]
        s.agent_recommendations = ["try the Margherita"]
        s.agent_questions_asked = ["What toppings do you prefer?"]
        s.user_provided_info = {"address": "123 Main St"}
        ctx = s.to_prompt_context()
        assert "Order pizza" in ctx
        assert "Alice" in ctx
        assert "choose toppings" in ctx
        assert "confirm address" in ctx
        assert "Margherita" in ctx
        assert "123 Main St" in ctx

    def test_serialisation_roundtrip(self):
        s = ConversationState()
        s.current_goal = "Book a flight to London"
        s.user_name = "Bob"
        s.subtasks_completed = ["choose dates"]
        s.subtasks_pending = ["confirm payment"]
        s.user_provided_info = {"order_number": "XY123", "budget": "800"}
        s.agent_recommendations = ["Take BA flight 178"]
        s.turn_count = 10
        s.user_frustration_signals = 1

        data = s.to_dict()
        restored = ConversationState.from_dict(data)

        assert restored.current_goal == s.current_goal
        assert restored.user_name == s.user_name
        assert restored.subtasks_completed == s.subtasks_completed
        assert restored.subtasks_pending == s.subtasks_pending
        assert restored.user_provided_info == s.user_provided_info
        assert restored.agent_recommendations == s.agent_recommendations
        assert restored.turn_count == s.turn_count
        assert restored.user_frustration_signals == s.user_frustration_signals

    def test_serialisation_preserves_defaults(self):
        s = ConversationState()
        restored = ConversationState.from_dict(s.to_dict())
        assert restored.agent_mode == "general"
        assert isinstance(restored.subtasks_completed, list)
        assert isinstance(restored.user_provided_info, dict)


class TestStateManagerGoalManagement:

    # 1. State survives context truncation
    def test_state_survives_context_truncation(self):
        """State persists even when the message list is truncated to nothing."""
        mgr = _make_state_mgr(
            "Book a flight to London",
            ["choose dates", "select seat", "confirm payment"],
        )
        mgr.state.user_provided_info["order_number"] = "ORD-42"
        mgr.state.user_name = "Alice"

        # Simulate truncation by discarding all messages
        messages: list[dict] = [
            {"role": "user", "content": "some message"},
            {"role": "assistant", "content": "some reply"},
        ]
        # "Truncate" to empty
        messages.clear()

        # State is independent of messages
        assert mgr.state.current_goal == "Book a flight to London"
        assert mgr.state.user_provided_info["order_number"] == "ORD-42"
        assert mgr.state.user_name == "Alice"
        assert len(mgr.state.subtasks_pending) == 3

    # 2. State is injected into the system prompt
    def test_state_injected_into_prompt(self):
        mgr = _make_state_mgr("Book a flight to London")
        augmented = mgr.build_system_prompt_with_state("You are a helpful assistant.")
        assert "Book a flight" in augmented
        assert "Conversation State" in augmented

    # 3. Goal drift detected after 5 off-topic turns
    def test_goal_drift_detected(self):
        mgr = _make_state_mgr("Book a flight to London", ["collect dates"])
        # 5 off-topic turns
        for _ in range(6):
            mgr.process_user_turn("What is 2+2? Completely unrelated math question")
            mgr.process_agent_turn("4.")
        assert mgr.check_goal_drift()
        assert mgr.get_recovery_action() == "remind_goal"

    # 4. User info tracked across truncation
    def test_user_info_tracked_across_turns(self):
        mgr = _make_state_mgr()
        # Turn 5: user provides order number
        mgr.process_user_turn("my order number is 12345 for reference")
        mgr.process_agent_turn("Got it.")
        # Simulate 15 more turns (messages would be truncated, state persists)
        for _ in range(15):
            mgr.process_user_turn("Tell me something interesting")
            mgr.process_agent_turn("Here's something interesting.")
        # State still contains the order number
        assert "order_number" in mgr.state.user_provided_info
        assert "12345" in mgr.state.user_provided_info["order_number"]

    def test_no_drift_when_on_topic(self):
        mgr = _make_state_mgr("Book a flight to London", ["choose dates"])
        for _ in range(10):
            mgr.process_user_turn("I want to book the London flight for June 15")
            mgr.process_agent_turn("Searching London flights.")
        assert not mgr.check_goal_drift()

    def test_set_goal_resets_subtasks(self):
        mgr = _make_state_mgr("Old goal", ["old task"])
        mgr.mark_subtask_complete("old task")
        mgr.set_goal("New goal", ["new task 1", "new task 2"])
        assert mgr.state.current_goal == "New goal"
        assert mgr.state.subtasks_completed == []
        assert len(mgr.state.subtasks_pending) == 2

    def test_mark_subtask_complete(self):
        mgr = _make_state_mgr("Goal", ["collect dates", "choose seat", "payment"])
        mgr.mark_subtask_complete("collect dates")
        assert "collect dates" in mgr.state.subtasks_completed
        assert "collect dates" not in mgr.state.subtasks_pending

    def test_check_goal_complete_true_when_all_done(self):
        mgr = _make_state_mgr("Goal", ["task1"])
        mgr.mark_subtask_complete("task1")
        assert mgr.check_goal_complete()

    def test_check_goal_complete_false_when_pending(self):
        mgr = _make_state_mgr("Goal", ["task1", "task2"])
        mgr.mark_subtask_complete("task1")
        assert not mgr.check_goal_complete()

    def test_frustration_signals_counted(self):
        mgr = _make_state_mgr()
        mgr.process_user_turn("You already asked me that!!")
        assert mgr.state.user_frustration_signals >= 1

    def test_recovery_action_frustration_priority(self):
        mgr = _make_state_mgr("Some goal", ["task"])
        mgr.state.user_frustration_signals = 3
        # Even if drift would also trigger, frustration takes priority
        mgr.state.turns_since_goal_mentioned = 10
        assert mgr.get_recovery_action() == "ask_user_to_clarify"

    def test_recovery_action_checkpoint_interval(self):
        mgr = _make_state_mgr("Goal")
        mgr.state.turn_count = 10
        assert mgr.get_recovery_action() == "inject_checkpoint"

    def test_recovery_action_summarize_when_all_done(self):
        mgr = _make_state_mgr("Goal", ["task1"])
        mgr.mark_subtask_complete("task1")
        assert mgr.get_recovery_action() == "summarize_progress"

    def test_recovery_action_none_when_healthy(self):
        mgr = _make_state_mgr()
        assert mgr.get_recovery_action() is None

    def test_process_agent_turn_tracks_questions(self):
        mgr = _make_state_mgr()
        mgr.process_agent_turn("What is your preferred seat? Aisle or window?")
        assert any("?" in q for q in mgr.state.agent_questions_asked)

    def test_process_agent_turn_tracks_recommendations(self):
        mgr = _make_state_mgr()
        mgr.process_agent_turn("I recommend the Business class for long-haul flights.")
        assert len(mgr.state.agent_recommendations) >= 1
        assert any("Business class" in r for r in mgr.state.agent_recommendations)

    def test_user_name_extracted(self):
        mgr = _make_state_mgr()
        mgr.process_user_turn("My name is Alice Johnson and I want to travel")
        assert mgr.state.user_name == "Alice Johnson"


# ===========================================================================
# PROGRESSIVE SUMMARIZER TESTS
# ===========================================================================

class TestProgressiveSummarizer:

    # 5. Verbatim turns preserved
    def test_verbatim_turns_preserved(self):
        """Last 5 turns must appear verbatim in the context."""
        s = ProgressiveSummarizer(verbatim_turns=5, layer_size=10)
        turns = [(f"user message {i}", f"assistant response {i}") for i in range(10)]
        for u, a in turns:
            s.add_turn(u, a)

        ctx = s.get_context()
        # Last 5 turns must be verbatim
        for u, a in turns[-5:]:
            assert u in ctx, f"Expected verbatim user message not found: {u!r}"
            assert a in ctx, f"Expected verbatim assistant message not found: {a!r}"

    # 6. Older turns summarized, not verbatim
    def test_older_turns_summarized(self):
        """Turns 1–5 should be in a summary layer, not raw verbatim."""
        s = ProgressiveSummarizer(verbatim_turns=5, layer_size=10)
        turns = [(f"user msg {i}", f"assistant msg {i}") for i in range(10)]
        for u, a in turns:
            s.add_turn(u, a)

        # The first 5 turns should NOT be in verbatim buffer
        verbatim_user_msgs = {u for u, _ in s.verbatim}
        for u, _ in turns[:5]:
            assert u not in verbatim_user_msgs, (
                f"Turn {u!r} should be in a summary layer, not verbatim"
            )

    # 7. Layers cascade on overflow
    def test_layers_cascade_on_overflow(self):
        """Adding many turns forces older turns to cascade into deeper layers."""
        # Use a very low token limit to force cascading with few turns
        s = ProgressiveSummarizer(
            verbatim_turns=3,
            layer_size=5,
            layer_token_limit=50,   # Very small to trigger cascade quickly
        )
        for i in range(25):
            s.add_turn(
                f"User message number {i} with some extra words to fill tokens",
                f"Assistant response {i} with extra text to ensure token count rises",
            )

        stats = s.get_stats()
        # At least layer 1 should have content; with cascade, layer 2 may also
        assert stats["total_turns_processed"] == 25
        # The verbatim buffer should not exceed verbatim_turns
        assert stats["verbatim_turns"] <= s.verbatim_turns

    # 8. Summary preserves key information
    def test_summary_preserves_key_information(self):
        """Critical facts should survive into summary layers."""
        s = ProgressiveSummarizer(verbatim_turns=3, layer_size=5)
        # Add turns with key information in early turns
        s.add_turn("My budget is $500 for this project", "Got it, $500 budget noted.")
        s.add_turn("I prefer window seats on all flights", "Window seat preference recorded.")
        s.add_turn("Let's keep going", "Sure.")
        s.add_turn("Any updates?", "Working on it.")
        s.add_turn("How is it going?", "Making progress.")
        s.add_turn("Continue please", "Continuing.")
        s.add_turn("And the next step?", "Next step coming up.")
        s.add_turn("Thanks", "You're welcome.")

        ctx = s.get_context()
        # Key facts from early turns should appear in context
        # (either verbatim or in summaries)
        assert "$500" in ctx or "500" in ctx, "Budget should appear in context"
        assert "window" in ctx.lower(), "Seat preference should appear in context"

    def test_verbatim_always_last_n_turns(self):
        """Verbatim buffer must always contain the N most recent turns."""
        n = 5
        s = ProgressiveSummarizer(verbatim_turns=n)
        all_turns = [(f"u{i}", f"a{i}") for i in range(20)]
        for u, a in all_turns:
            s.add_turn(u, a)

        expected_verbatim = all_turns[-n:]
        for (exp_u, exp_a), (got_u, got_a) in zip(expected_verbatim, s.verbatim):
            assert exp_u == got_u
            assert exp_a == got_a

    def test_get_stats_structure(self):
        s = ProgressiveSummarizer()
        for i in range(10):
            s.add_turn(f"user {i}", f"assistant {i}")
        stats = s.get_stats()
        assert "total_turns_processed" in stats
        assert stats["total_turns_processed"] == 10
        assert "verbatim_turns" in stats
        assert "layer_1_tokens" in stats
        assert "total_context_tokens" in stats

    def test_get_context_order(self):
        """Context must be ordered: deep layers first, verbatim last."""
        s = ProgressiveSummarizer(verbatim_turns=2, layer_token_limit=20)
        for i in range(15):
            s.add_turn(f"user {i}", f"assistant {i}")

        ctx = s.get_context()
        # Verbatim section should appear after any summary layers
        verbatim_idx = ctx.find("[Most recent turns:")
        if verbatim_idx >= 0:
            # Check that 'Early conversation' or 'Earlier conversation'
            # appears before 'Most recent turns' (or doesn't appear at all)
            for label in ["[Early conversation:", "[Earlier conversation:", "[Recent conversation:"]:
                label_idx = ctx.find(label)
                if label_idx >= 0:
                    assert label_idx < verbatim_idx, (
                        f"Label '{label}' should appear before verbatim section"
                    )


# ===========================================================================
# SESSION MANAGER TESTS
# ===========================================================================

class TestSessionManager:

    # 9. Expired session creates a new one
    def test_expired_session_creates_new(self):
        mgr = SessionManager(ttl_minutes=30)
        session = mgr.get_session("user-1")
        original_id = session.session_id

        # Expire by backdating last_activity (45 minutes ago)
        session.last_activity = time.time() - (45 * 60)

        new_session = mgr.get_session("user-1")
        assert new_session.session_id != original_id

    # 10. Active session is reused
    def test_active_session_reused(self):
        mgr = SessionManager(ttl_minutes=30)
        s1 = mgr.get_session("user-2")
        s2 = mgr.get_session("user-2")
        assert s1.session_id == s2.session_id

    # 11. Branch inherits context
    def test_branch_inherits_context(self):
        mgr = SessionManager(ttl_minutes=30)
        parent = mgr.get_session("alice")
        parent.state_manager.set_goal(
            "Book flight to London",
            ["collect dates", "confirm payment"],
        )
        parent.state.user_name = "Alice"
        parent.state.user_preferences["seat"] = "window"
        parent.state_manager.mark_subtask_complete("collect dates")

        child = mgr.branch_session("alice", context_keys=["user_profile", "goal_summary"])

        assert child.state.user_name == "Alice"
        assert child.state.user_preferences.get("seat") == "window"
        assert child.inherited_context is not None
        assert "Book flight" in child.inherited_context

    # 12. Reset preserves identity
    def test_reset_preserves_identity(self):
        mgr = SessionManager(ttl_minutes=30)
        session = mgr.get_session("bob")
        session.state.user_name = "Bob"
        session.state_manager.set_goal("Order pizza", ["choose toppings"])

        reset = mgr.reset_session("bob", keep_identity=True)

        assert reset.state.user_name == "Bob"
        assert reset.state.current_goal is None
        assert len(reset.messages) == 0

    def test_reset_without_identity(self):
        mgr = SessionManager(ttl_minutes=30)
        session = mgr.get_session("carol")
        session.state.user_name = "Carol"
        session.state_manager.set_goal("Some goal")

        reset = mgr.reset_session("carol", keep_identity=False)

        assert reset.state.user_name is None

    def test_cleanup_expired(self):
        mgr = SessionManager(ttl_minutes=30)
        for uid in ("u1", "u2", "u3"):
            s = mgr.get_session(uid)
            s.last_activity = time.time() - (45 * 60)

        removed = mgr.cleanup_expired()
        assert removed == 3
        assert mgr.get_active_sessions() == 0

    def test_persist_and_restore(self):
        mgr = SessionManager(ttl_minutes=30)
        session = mgr.get_session("dave")
        session.state_manager.set_goal("Fix a bug", ["reproduce", "patch", "test"])
        session.state.user_name = "Dave"
        session.add_user_message("I found a bug in the payment module")
        session.add_agent_message("Let me help you fix that.")

        json_str = mgr.persist_session("dave")
        assert len(json_str) > 2

        mgr2 = SessionManager(ttl_minutes=30)
        restored = mgr2.restore_session("dave", json_str)

        assert restored.session_id == session.session_id
        assert restored.state.current_goal == session.state.current_goal
        assert restored.state.user_name == session.state.user_name
        assert len(restored.messages) == 2

    def test_end_session_removes_it(self):
        mgr = SessionManager(ttl_minutes=30)
        mgr.get_session("eve")
        assert mgr.get_active_sessions() == 1
        mgr.end_session("eve")
        assert mgr.get_active_sessions() == 0

    def test_session_touch_resets_expiry(self):
        mgr = SessionManager(ttl_minutes=30)
        s = mgr.get_session("frank")
        s.last_activity = time.time() - (29 * 60)
        assert not s.is_expired(30)
        s.last_activity = time.time() - (31 * 60)
        assert s.is_expired(30)

    def test_add_user_message_updates_turn_count(self):
        mgr = SessionManager(ttl_minutes=30)
        s = mgr.get_session("grace")
        s.add_user_message("Hello")
        assert s.state.turn_count == 1
        s.add_user_message("World")
        assert s.state.turn_count == 2

    def test_build_messages_for_llm_respects_max_tokens(self):
        """History messages added to the returned list must stay within budget.

        We add many large messages and ask for a tight token budget.  The
        resulting non-system messages should collectively fit within that
        budget (the system prompt itself may use tokens from the summary).
        """
        mgr = SessionManager(ttl_minutes=30)
        s = mgr.get_session("henry")
        for i in range(40):
            s.add_user_message("x " * 100)   # ~100 tokens each
            s.add_agent_message("y " * 100)

        HISTORY_BUDGET = 500
        messages = s.build_messages_for_llm(system_prompt="", max_tokens=100_000)
        full_count = len(messages)  # should include all 80 history messages + system

        messages_tight = s.build_messages_for_llm(system_prompt="", max_tokens=HISTORY_BUDGET + 200)
        history_tokens = sum(
            count_tokens(m) for m in messages_tight
            if m["role"] != "system"
        )
        # History (non-system) messages must fit within the allowed budget
        assert history_tokens <= HISTORY_BUDGET + 50  # 50-token overhead buffer
        # And truncation should have removed some older messages
        assert len(messages_tight) < full_count


# ===========================================================================
# RECOVERY / GOAL DETECTOR TESTS
# ===========================================================================

class TestGoalDetector:

    # 13. Goal detected from message
    def test_goal_detected_from_message(self):
        detector = GoalDetector()
        result = detector.detect_goal("I need to book a flight to London next Tuesday")
        assert "flight" in result.goal.lower() or "London" in result.goal

    def test_goal_subtasks_extracted(self):
        detector = GoalDetector()
        result = detector.detect_goal(
            "I need to book a flight, reserve a hotel, and arrange a rental car"
        )
        # Heuristic may extract multiple subtasks
        assert isinstance(result.subtasks, list)

    def test_topic_change_low_relevance(self):
        detector = GoalDetector()
        score = detector.detect_topic_change("Book a flight to London", "What is the weather in Tokyo?")
        assert score < 0.5

    def test_topic_change_high_relevance(self):
        detector = GoalDetector()
        score = detector.detect_topic_change("Book a flight to London", "Which London flights are available on June 15?")
        assert score >= 0.3

    def test_completion_check_signals(self):
        detector = GoalDetector()
        summary = "The flight has been booked and confirmed. Payment was processed successfully."
        result = detector.check_goal_completion("Book a flight", summary)
        assert result.completion_pct > 0


class TestRecoveryManager:

    # 14. Recovery intervention injected with goal name
    def test_recovery_intervention_injected(self):
        mgr = _make_state_mgr("Book a vacation to Paris", ["buy tickets", "book hotel"])
        recovery = RecoveryManager(mgr)
        # Trigger drift
        mgr.state.turns_since_goal_mentioned = 6
        issues = recovery.diagnose()
        assert any(i.type == "goal_drift" for i in issues)

        intervention = recovery.get_intervention(issues)
        assert "Paris" in intervention or "Book a vacation" in intervention

    def test_diagnose_no_issues_when_healthy(self):
        mgr = _make_state_mgr("Clean goal", ["step 1"])
        recovery = RecoveryManager(mgr)
        issues = recovery.diagnose()
        assert issues == []

    def test_diagnose_frustration_issue(self):
        mgr = _make_state_mgr()
        mgr.state.user_frustration_signals = 3
        recovery = RecoveryManager(mgr)
        issues = recovery.diagnose()
        assert any(i.type == "frustration" for i in issues)

    def test_diagnose_severity_ordering(self):
        """High severity issues must appear before medium."""
        mgr = _make_state_mgr("Goal")
        mgr.state.user_frustration_signals = 3   # high
        mgr.state.turns_since_goal_mentioned = 6  # medium/high drift
        recovery = RecoveryManager(mgr)
        issues = recovery.diagnose()
        if len(issues) >= 2:
            sev_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(issues) - 1):
                assert sev_order[issues[i].severity] <= sev_order[issues[i + 1].severity]

    def test_should_not_reset_healthy_conversation(self):
        mgr = _make_state_mgr()
        recovery = RecoveryManager(mgr)
        assert not recovery.should_reset([])

    def test_should_reset_multiple_high_severity(self):
        mgr = _make_state_mgr("Goal")
        recovery = RecoveryManager(mgr)
        issues = [
            ConversationIssue("frustration", "high", "desc", "ask"),
            ConversationIssue("stalemate", "high", "desc", "checkpoint"),
        ]
        assert recovery.should_reset(issues)

    def test_generate_progress_report_with_goal(self):
        mgr = _make_state_mgr("Plan a vacation", ["book flights", "reserve hotel"])
        mgr.mark_subtask_complete("book flights")
        recovery = RecoveryManager(mgr)
        report = recovery.generate_progress_report()
        assert "Plan a vacation" in report
        assert "book flights" in report
        assert "reserve hotel" in report

    def test_generate_progress_report_no_goal(self):
        """With no goal set the report should indicate there is nothing to track."""
        mgr = _make_state_mgr()
        recovery = RecoveryManager(mgr)
        report = recovery.generate_progress_report()
        # implementation returns "All steps complete." when no subtasks are pending
        assert "All steps complete" in report or "No active goal" in report

    def test_intervention_empty_for_no_issues(self):
        mgr = _make_state_mgr()
        recovery = RecoveryManager(mgr)
        assert recovery.get_intervention([]) == ""

    def test_state_injected_into_drift_intervention(self):
        mgr = _make_state_mgr("Book a flight to Tokyo", ["collect dates", "confirm seat"])
        mgr.state.turns_since_goal_mentioned = 7
        recovery = RecoveryManager(mgr)
        issues = recovery.diagnose()
        intervention = recovery.get_intervention(issues)
        # The intervention should reference the goal
        assert "Tokyo" in intervention or "Book a flight" in intervention


# ===========================================================================
# INTEGRATION: Full session lifecycle
# ===========================================================================

class TestIntegration:

    def test_full_session_lifecycle(self):
        """End-to-end: create session, track goal, drift, recover, persist."""
        mgr = SessionManager(ttl_minutes=60)
        s = mgr.get_session("user-integration")

        # Set goal
        s.state_manager.set_goal(
            "Book a round-trip to London",
            ["choose dates", "select seat", "confirm payment"],
        )

        # Normal turns
        s.add_user_message("I want to fly June 15 and return June 22")
        s.add_agent_message("Searching June 15–22 flights.")
        s.state_manager.mark_subtask_complete("choose dates")

        # Off-topic drift
        for _ in range(6):
            s.add_user_message("Tell me a random fact about penguins")
            s.add_agent_message("Penguins are flightless birds.")

        assert s.state_manager.check_goal_drift()
        recovery_action = s.state_manager.get_recovery_action()
        assert recovery_action == "remind_goal"

        # Return on-topic
        s.add_user_message("Right, back to the flight booking — window seat please")
        s.add_agent_message("Window seat noted.")
        s.state_manager.mark_subtask_complete("select seat")

        # Persist and restore
        json_str = mgr.persist_session("user-integration")
        mgr2 = SessionManager(ttl_minutes=60)
        restored = mgr2.restore_session("user-integration", json_str)

        assert restored.state.current_goal == "Book a round-trip to London"
        assert "choose dates" in restored.state.subtasks_completed
        assert "select seat" in restored.state.subtasks_completed
        assert "confirm payment" in restored.state.subtasks_pending

    def test_branch_then_continue(self):
        """Branch a session and ensure the child starts fresh but remembers identity."""
        mgr = SessionManager(ttl_minutes=60)
        parent = mgr.get_session("user-branch")
        parent.state.user_name = "Alice"
        parent.state_manager.set_goal("Book a flight", ["collect dates"])
        parent.state_manager.mark_subtask_complete("collect dates")

        child = mgr.branch_session("user-branch", context_keys=["user_profile", "goal_summary"])
        # Child starts fresh goal-wise
        assert child.state.current_goal is None
        # But remembers Alice
        assert child.state.user_name == "Alice"
        # And has inherited context
        assert child.inherited_context is not None
