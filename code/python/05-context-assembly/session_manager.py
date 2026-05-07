"""Session Manager — multi-session conversation management with expiry,
branching, and reset.

Handles the full lifecycle of a conversation session: creation, context
accumulation, branch (inheriting context from a parent), explicit reset, and
expiry after a configurable period of inactivity.

Key classes:

- :class:`Session` — a single user conversation (state + messages + summarizer).
- :class:`SessionManager` — manages a collection of sessions, with TTL and
  persistence helpers.

See: docs/04-context-engineering/04-multi-turn-context-management.md
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from context_budget import count_tokens
from progressive_summarizer import ProgressiveSummarizer
from state_manager import ConversationState, StateManager


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """A single conversation session for one user.

    Owns a :class:`~state_manager.ConversationState`,
    a raw ``messages`` list for the LLM, and a
    :class:`~progressive_summarizer.ProgressiveSummarizer` for context
    compression.

    Args:
        user_id: Application-level user identifier (e.g. UUID, email hash).
    """

    def __init__(self, user_id: str) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.user_id: str = user_id
        self.created_at: float = time.time()
        self.last_activity: float = time.time()
        self.state_manager: StateManager = StateManager()
        self.messages: list[dict] = []
        self.summarizer: ProgressiveSummarizer = ProgressiveSummarizer()
        self.inherited_context: Optional[str] = None
        self.is_active: bool = True

    # Convenience property so callers can access state directly
    @property
    def state(self) -> ConversationState:
        return self.state_manager.state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_expired(self, ttl_minutes: int) -> bool:
        """Return True when the session has been idle longer than *ttl_minutes*."""
        return (time.time() - self.last_activity) > (ttl_minutes * 60)

    def touch(self) -> None:
        """Update last-activity timestamp."""
        self.last_activity = time.time()

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_user_message(self, message: str) -> None:
        """Append a user message and update conversation state.

        Args:
            message: Raw text of the user's message.
        """
        self.touch()
        self.messages.append({"role": "user", "content": message})
        self.state_manager.process_user_turn(message)

    def add_agent_message(
        self, message: str, tool_calls: list | None = None
    ) -> None:
        """Append an agent message and update conversation state.

        Args:
            message:    Raw text of the agent's response.
            tool_calls: Optional list of tool-call objects for subtask tracking.
        """
        self.touch()
        msg: dict = {"role": "assistant", "content": message}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self.state_manager.process_agent_turn(message, tool_calls=tool_calls)

        # Feed the turn pair to the progressive summarizer
        # Find the most recent user message
        user_content = ""
        for m in reversed(self.messages[:-1]):
            if m["role"] == "user":
                user_content = m.get("content", "")
                break
        self.summarizer.add_turn(user_content, message)

    def build_messages_for_llm(
        self,
        system_prompt: str = "",
        max_tokens: int = 100_000,
    ) -> list[dict]:
        """Build the messages array for the next LLM call.

        Performs token-aware truncation of the raw message history.
        Injects conversation state and progressive summaries into the
        system prompt.

        Args:
            system_prompt: Base system prompt text.
            max_tokens:    Hard cap on total tokens in the returned list.

        Returns:
            A ``list[dict]`` of ``{"role": ..., "content": ...}`` messages
            ready to pass directly to the chat completions API.
        """
        # Build an augmented system prompt
        augmented = self.state_manager.build_system_prompt_with_state(system_prompt)
        summary_context = self.summarizer.get_context()
        if summary_context:
            augmented = (
                f"{augmented}\n\n"
                f"## Conversation History Summary\n"
                f"{summary_context}"
            )
        if self.inherited_context:
            augmented = (
                f"{augmented}\n\n"
                f"## Inherited from Previous Session\n"
                f"{self.inherited_context}"
            )

        system_msg = {"role": "system", "content": augmented}
        system_tokens = count_tokens(system_msg)
        budget = max_tokens - system_tokens

        # Walk backwards through messages, keeping as many as fit
        kept: list[dict] = []
        running = 0
        for msg in reversed(self.messages):
            tokens = count_tokens(msg)
            if running + tokens > budget:
                break
            kept.append(msg)
            running += tokens

        kept.reverse()
        return [system_msg, *kept]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise session for storage (JSON-safe)."""
        return {
            "session_id":        self.session_id,
            "user_id":           self.user_id,
            "created_at":        self.created_at,
            "last_activity":     self.last_activity,
            "state":             self.state.to_dict(),
            "messages":          list(self.messages),
            "inherited_context": self.inherited_context,
            "is_active":         self.is_active,
            # Summarizer layers
            "summarizer": {
                "verbatim":        [(u, a) for u, a in self.summarizer.verbatim],
                "layers":          list(self.summarizer.layers),
                "total_turns":     self.summarizer._total_turns,
                "verbatim_turns":  self.summarizer.verbatim_turns,
                "layer_size":      self.summarizer.layer_size,
                "layer_token_limit": self.summarizer.layer_token_limit,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Restore a session from a serialised dictionary."""
        session = cls(user_id=data["user_id"])
        session.session_id    = data["session_id"]
        session.created_at    = data["created_at"]
        session.last_activity = data["last_activity"]
        session.state_manager.state = ConversationState.from_dict(data["state"])
        session.messages      = data.get("messages", [])
        session.inherited_context = data.get("inherited_context")
        session.is_active     = data.get("is_active", True)

        # Restore summarizer
        s_data = data.get("summarizer", {})
        summ = ProgressiveSummarizer(
            verbatim_turns    = s_data.get("verbatim_turns", 5),
            layer_size        = s_data.get("layer_size", 10),
            layer_token_limit = s_data.get("layer_token_limit", 1500),
        )
        summ.verbatim      = [tuple(pair) for pair in s_data.get("verbatim", [])]
        summ.layers        = s_data.get("layers", [""] * ProgressiveSummarizer.NUM_LAYERS)
        summ._total_turns  = s_data.get("total_turns", 0)
        session.summarizer = summ

        return session


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Manage conversation sessions across users and time.

    Handles creation, expiry, branching, reset, and JSON persistence.

    Args:
        ttl_minutes:  Minutes of inactivity before a session expires.
        max_sessions: Maximum number of concurrent sessions kept in memory.
                      Oldest sessions are evicted when the limit is reached.

    Example::

        mgr = SessionManager(ttl_minutes=30)
        session = mgr.get_session("user-123")
        session.add_user_message("I need to book a flight")
    """

    def __init__(
        self,
        ttl_minutes: int = 60,
        max_sessions: int = 10_000,
    ) -> None:
        self.ttl_minutes  = ttl_minutes
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------
    # Core session access
    # ------------------------------------------------------------------

    def get_session(self, user_id: str) -> Session:
        """Return the active session for *user_id*, creating a new one if
        the session has expired or does not exist.

        Args:
            user_id: Application-level user identifier.

        Returns:
            An active :class:`Session`.
        """
        session = self._sessions.get(user_id)

        if session is not None and session.is_expired(self.ttl_minutes):
            session = None  # Treat as non-existent

        if session is None:
            session = self.create_session(user_id)

        return session

    def create_session(
        self,
        user_id: str,
        inherit_from: str | None = None,
    ) -> Session:
        """Create a (possibly new) session, optionally inheriting context.

        If *inherit_from* names a user whose session still exists, the new
        session carries forward the parent's user profile and goal summary.

        Args:
            user_id:      Owner of the new session.
            inherit_from: User ID whose session to inherit context from.

        Returns:
            The newly created :class:`Session`.
        """
        self._evict_if_needed()
        session = Session(user_id=user_id)

        if inherit_from and inherit_from in self._sessions:
            parent = self._sessions[inherit_from]
            session.state.user_name        = parent.state.user_name
            session.state.user_preferences = dict(parent.state.user_preferences)
            if parent.state.current_goal:
                session.inherited_context = (
                    f"Previous conversation goal: {parent.state.current_goal}. "
                    f"Completed steps: {', '.join(parent.state.subtasks_completed) or 'none'}."
                )

        self._sessions[user_id] = session
        return session

    def branch_session(
        self,
        user_id: str,
        context_keys: list[str] | None = None,
    ) -> Session:
        """Create a new session that inherits selected context from the current one.

        *context_keys* controls which context is carried forward:

        - ``"user_profile"``  — name, preferences
        - ``"goal_summary"``  — current goal and completed steps
        - ``"preferences"``   — user preferences only

        Args:
            user_id:      User whose session to branch from.
            context_keys: Which context types to inherit.  Defaults to
                          ``["user_profile", "goal_summary"]``.

        Returns:
            The new branched :class:`Session`.
        """
        context_keys = context_keys or ["user_profile", "goal_summary"]
        parent = self._sessions.get(user_id)

        self._evict_if_needed()
        new_session = Session(user_id=user_id)

        if parent:
            if "user_profile" in context_keys or "preferences" in context_keys:
                new_session.state.user_name = parent.state.user_name
                new_session.state.user_preferences = dict(parent.state.user_preferences)

            if "user_profile" in context_keys:
                new_session.state.user_provided_info = dict(
                    parent.state.user_provided_info
                )

            if "goal_summary" in context_keys and parent.state.current_goal:
                completed_str = (
                    ", ".join(parent.state.subtasks_completed)
                    if parent.state.subtasks_completed
                    else "none"
                )
                new_session.inherited_context = (
                    f"Previous conversation goal: {parent.state.current_goal}. "
                    f"Completed: {completed_str}."
                )

        self._sessions[user_id] = new_session
        return new_session

    def reset_session(
        self, user_id: str, keep_identity: bool = True
    ) -> Session:
        """Reset a user's session, optionally preserving user identity.

        Args:
            user_id:       User whose session to reset.
            keep_identity: When True, the new session retains *user_name*
                           and *user_preferences* from the old session.

        Returns:
            The fresh :class:`Session`.
        """
        old = self._sessions.get(user_id)
        self._evict_if_needed()
        session = Session(user_id=user_id)

        if keep_identity and old:
            session.state.user_name        = old.state.user_name
            session.state.user_preferences = dict(old.state.user_preferences)

        self._sessions[user_id] = session
        return session

    def end_session(self, user_id: str) -> None:
        """Explicitly end a session.

        The session is marked inactive and removed from memory.

        Args:
            user_id: User whose session to end.
        """
        session = self._sessions.pop(user_id, None)
        if session:
            session.is_active = False

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove all expired sessions.

        Returns:
            The number of sessions removed.
        """
        expired = [
            uid for uid, s in self._sessions.items()
            if s.is_expired(self.ttl_minutes)
        ]
        for uid in expired:
            del self._sessions[uid]
        return len(expired)

    def get_active_sessions(self) -> int:
        """Return the count of sessions that have not yet expired."""
        return sum(
            1 for s in self._sessions.values()
            if not s.is_expired(self.ttl_minutes)
        )

    def _evict_if_needed(self) -> None:
        """Evict the oldest session when at capacity."""
        if len(self._sessions) >= self.max_sessions:
            oldest_uid = min(
                self._sessions,
                key=lambda uid: self._sessions[uid].last_activity,
            )
            del self._sessions[oldest_uid]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_session(self, user_id: str) -> str:
        """Serialise a session to a JSON string.

        Args:
            user_id: User whose session to serialise.

        Returns:
            A JSON string, or ``"{}"`` if the user has no session.
        """
        session = self._sessions.get(user_id)
        if not session:
            return "{}"
        return json.dumps(session.to_dict())

    def restore_session(self, user_id: str, data: str) -> Session:
        """Deserialise a session from a JSON string.

        Args:
            user_id: User to assign the restored session to.
            data:    JSON string previously produced by :meth:`persist_session`.

        Returns:
            The restored :class:`Session`.
        """
        parsed = json.loads(data)
        session = Session.from_dict(parsed)
        # Ensure user_id is consistent
        session.user_id = user_id
        self._sessions[user_id] = session
        return session


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    """Demonstrate session management: creation, branching, reset, expiry."""

    print("=" * 70)
    print("SESSION MANAGER DEMO")
    print("=" * 70)

    mgr = SessionManager(ttl_minutes=30)

    # ── Create sessions for 3 users ─────────────────────────────────────
    print("\n── Creating sessions for 3 users ──")
    for user_id in ("alice", "bob", "carol"):
        s = mgr.get_session(user_id)
        print(f"  {user_id}: session {s.session_id[:8]}…")

    print(f"  Active sessions: {mgr.get_active_sessions()}")

    # ── Alice books a flight ─────────────────────────────────────────────
    print("\n── Alice's conversation ──")
    alice = mgr.get_session("alice")
    alice.state_manager.set_goal(
        "Book a flight to London",
        subtasks=["choose dates", "confirm payment"],
    )
    alice.add_user_message("I need to book a flight to London for June 15.")
    alice.add_agent_message("I can help! What's your return date?")
    alice.add_user_message("June 22. My name is Alice. My budget is $800.")
    alice.add_agent_message("Got it Alice — searching BA flights within $800.")
    alice.state_manager.mark_subtask_complete("choose dates")

    print(f"  Alice state:\n{alice.state.to_prompt_context()}")

    # ── Bob starts a different conversation ──────────────────────────────
    print("\n── Bob's conversation ──")
    bob = mgr.get_session("bob")
    bob.add_user_message("What's the weather like in Tokyo?")
    bob.add_agent_message("Tokyo is sunny and 22°C today.")
    print(f"  Bob messages: {len(bob.messages)}")

    # ── Session branching ────────────────────────────────────────────────
    print("\n── Alice branches to a new session (related task) ──")
    alice2 = mgr.branch_session("alice", context_keys=["user_profile", "goal_summary"])
    print(f"  New session ID: {alice2.session_id[:8]}…")
    print(f"  Inherited user name: {alice2.state.user_name}")
    print(f"  Inherited context: {alice2.inherited_context}")

    # ── Session reset ────────────────────────────────────────────────────
    print("\n── Bob resets his session (keep identity) ──")
    bob.state.user_name = "Bob"
    bob_reset = mgr.reset_session("bob", keep_identity=True)
    print(f"  New session ID: {bob_reset.session_id[:8]}…")
    print(f"  User name preserved: {bob_reset.state.user_name}")
    print(f"  Goal cleared: {bob_reset.state.current_goal!r}")
    print(f"  Messages cleared: {len(bob_reset.messages)}")

    # ── Session expiry ────────────────────────────────────────────────────
    print("\n── Session expiry simulation ──")
    carol = mgr.get_session("carol")
    carol_id_before = carol.session_id

    # Manually expire by backdating last_activity
    carol.last_activity = time.time() - (45 * 60)  # 45 minutes ago
    print(f"  Carol's session expired: {carol.is_expired(30)}")

    carol_new = mgr.get_session("carol")
    print(f"  Got new session after expiry: {carol_new.session_id[:8]}…")
    print(f"  Different from old: {carol_new.session_id != carol_id_before}")

    # ── Persist and restore ───────────────────────────────────────────────
    print("\n── Persistence roundtrip for Alice ──")
    json_str = mgr.persist_session("alice")
    print(f"  Serialised: {len(json_str)} bytes")

    mgr2 = SessionManager(ttl_minutes=30)
    restored = mgr2.restore_session("alice", json_str)
    print(f"  Restored session ID: {restored.session_id[:8]}…")
    print(f"  Restored goal: {restored.state.current_goal}")
    print(f"  Restored user name: {restored.state.user_name}")
    print(f"  Restored messages: {len(restored.messages)}")

    # Verify key fields match
    original = mgr.get_session("alice")
    assert restored.session_id == original.session_id
    assert restored.state.current_goal == original.state.current_goal
    assert restored.state.user_name == original.state.user_name
    print("  ✓ Restored state matches original")

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("\n── Cleanup expired sessions ──")
    # Expire alice's sessions
    for uid in list(mgr._sessions):
        mgr._sessions[uid].last_activity = time.time() - (35 * 60)
    removed = mgr.cleanup_expired()
    print(f"  Sessions removed: {removed}")
    print(f"  Active sessions remaining: {mgr.get_active_sessions()}")

    print("\nDemo complete.")


if __name__ == "__main__":
    _demo()
