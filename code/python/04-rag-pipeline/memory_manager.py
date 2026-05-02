"""Token-aware conversation memory manager.

Manages the agent message list with three strategies for handling context
window overflow:
  - truncate:       Drop oldest complete turns, keep system prompt + recent.
  - summarize:      Compress old messages into a summary, keep recent verbatim.
  - sliding_window: Rolling summary of old messages + recent messages (default).

See: docs/03-memory-and-retrieval/01-short-term-memory.md
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import tiktoken
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tokens reserved for the model's output response
_OUTPUT_RESERVE = 4096

# Formatting overhead per message (role, separators, etc.)
_MSG_OVERHEAD = 4

# Priming overhead for the assistant turn header
_PRIMING_OVERHEAD = 2

_SUMMARIZER_PROMPT = """\
Summarize the following conversation between a user and an AI assistant.
Focus on information the assistant needs to continue helping the user.

INCLUDE:
- The user's original request and any changes to it
- Information gathered from tools (with specific values: numbers, dates, names)
- Decisions the assistant made and why
- Actions taken and their outcomes
- Pending tasks or unanswered questions
- User preferences or constraints mentioned

DO NOT INCLUDE:
- Greetings, small talk, pleasantries
- Exact wording of prompts unless critical
- Redundant or repeated information
- Tool call mechanics (just the results)

FORMAT:
Write as a dense paragraph in third person past tense. Be concise but complete.\
"""


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count tokens in a messages list, including per-message overhead.

    This matches the OpenAI token-counting method described in:
    https://platform.openai.com/docs/guides/text-generation/managing-tokens

    Args:
        messages: List of message dicts (role, content, tool_calls, …).
        model:    Model name for the correct tokeniser.

    Returns:
        Estimated token count.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        total += _MSG_OVERHEAD
        for key, value in msg.items():
            if value is None:
                continue
            if isinstance(value, str):
                total += len(enc.encode(value))
            elif isinstance(value, list):
                # tool_calls array — serialize as JSON
                total += len(enc.encode(json.dumps(value, default=str)))
            elif isinstance(value, dict):
                total += len(enc.encode(json.dumps(value, default=str)))

    total += _PRIMING_OVERHEAD
    return total


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Token-aware conversation memory with pluggable overflow strategies.

    Attributes:
        model:       LLM model name (used for tokenisation).
        max_tokens:  Hard limit for the input context sent to the API.
        messages:    Complete message history (including system prompt).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 100_000,
        system_prompt: str = "",
        client: Optional[object] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._client: OpenAI = client or OpenAI(  # type: ignore[assignment]
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
        self._summary_cache: Optional[str] = None
        self._summary_input_len: int = 0  # len(messages) when summary was built

    # ------------------------------------------------------------------
    # Message appenders
    # ------------------------------------------------------------------

    def add_message(self, message: dict) -> None:
        """Append a raw message dict to the history."""
        self.messages.append(message)
        # Invalidate summary cache whenever new messages arrive
        self._summary_cache = None

    def add_user_message(self, content: str) -> None:
        """Append a user message."""
        self.add_message({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list] = None,
    ) -> None:
        """Append an assistant message, optionally with tool calls."""
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.add_message(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Append a tool result message."""
        self.add_message(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    # ------------------------------------------------------------------
    # Token count / cost estimation
    # ------------------------------------------------------------------

    def token_count(self) -> int:
        """Return the current token count of the full message history."""
        return count_tokens(self.messages, self.model)

    def estimated_cost(
        self,
        input_price_per_1k: float = 0.0025,
        output_price_per_1k: float = 0.01,
    ) -> dict:
        """Estimate cost for the next API call based on current token count.

        Returns:
            Dict with keys: input_tokens, estimated_output_tokens,
            input_cost_usd, output_cost_usd, total_cost_usd.
        """
        input_tok = self.token_count()
        estimated_output = _OUTPUT_RESERVE
        return {
            "input_tokens": input_tok,
            "estimated_output_tokens": estimated_output,
            "input_cost_usd": round(input_tok / 1000 * input_price_per_1k, 6),
            "output_cost_usd": round(
                estimated_output / 1000 * output_price_per_1k, 6
            ),
            "total_cost_usd": round(
                input_tok / 1000 * input_price_per_1k
                + estimated_output / 1000 * output_price_per_1k,
                6,
            ),
        }

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, to_message_index: int) -> None:
        """Roll back the message list to a previous state.

        The system prompt (index 0) is always preserved.

        Args:
            to_message_index: Slice up to (exclusive). Index 1 removes
                              everything after the system prompt.

        Raises:
            ValueError: If the index is out of range.
        """
        if to_message_index < 1 or to_message_index > len(self.messages):
            raise ValueError(
                f"to_message_index must be in [1, {len(self.messages)}]; "
                f"got {to_message_index}"
            )
        self.messages = self.messages[:to_message_index]
        self._summary_cache = None

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def get_messages(
        self,
        strategy: str = "sliding_window",
        recent_count: int = 10,
    ) -> list[dict]:
        """Return the messages to send to the API, applying the chosen strategy.

        Args:
            strategy:     One of: "none", "truncate", "summarize",
                          "sliding_window".
            recent_count: Number of most-recent messages to keep verbatim
                          (used by "summarize" and "sliding_window").

        Returns:
            A list of message dicts safe to pass to the LLM.
        """
        current_tokens = self.token_count()

        if strategy == "none" or current_tokens <= self.max_tokens:
            return self.messages

        if strategy == "truncate":
            return self._apply_truncation()
        if strategy == "summarize":
            return self._apply_summarization(recent_count)
        if strategy == "sliding_window":
            return self._apply_sliding_window(recent_count)

        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose: none, truncate, summarize, sliding_window."
        )

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def _apply_truncation(self) -> list[dict]:
        """Keep system prompt + most-recent complete turns within max_tokens.

        A 'complete turn' is: user → assistant (with possible tool_calls)
        → tool results → final assistant answer.  We never remove a turn
        partially (which would leave orphaned tool results).
        """
        system_msg = self.messages[0]
        turns = _group_into_turns(self.messages[1:])

        budget = self.max_tokens - count_tokens([system_msg], self.model)
        kept: list[list[dict]] = []

        for turn in reversed(turns):
            turn_tokens = count_tokens(turn, self.model)
            if turn_tokens <= budget:
                kept.insert(0, turn)
                budget -= turn_tokens
            else:
                break

        result = [system_msg] + [msg for turn in kept for msg in turn]
        original_count = self.token_count()
        new_count = count_tokens(result, self.model)
        if new_count < original_count:
            logger.warning(
                "Truncation: reduced from %d to %d tokens (%d turns kept)",
                original_count,
                new_count,
                len(kept),
            )
        return result

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _apply_summarization(self, recent_count: int) -> list[dict]:
        """Summarize all but the last `recent_count` messages."""
        system_msg = self.messages[0]
        conversation = self.messages[1:]

        if len(conversation) <= recent_count:
            return self.messages

        to_summarize = conversation[:-recent_count]
        recent = conversation[-recent_count:]

        summary = self._get_or_build_summary(to_summarize)

        return [
            system_msg,
            {
                "role": "user",
                "content": f"[Conversation summary: {summary}]",
            },
            *recent,
        ]

    # ------------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------------

    def _apply_sliding_window(self, recent_count: int) -> list[dict]:
        """Rolling summary of old messages + verbatim recent messages."""
        system_msg = self.messages[0]
        conversation = self.messages[1:]

        if len(conversation) <= recent_count:
            return self.messages

        to_summarize = conversation[:-recent_count]
        recent = conversation[-recent_count:]

        # Use the cached summary when the "old" slice hasn't changed
        summary = self._get_or_build_summary(to_summarize)

        result = [system_msg]
        if summary:
            result.append(
                {
                    "role": "user",
                    "content": f"[Conversation so far: {summary}]",
                }
            )
        result.extend(recent)
        return result

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def _get_or_build_summary(self, messages: list[dict]) -> str:
        """Return cached summary or build a new one via LLM."""
        if (
            self._summary_cache is not None
            and self._summary_input_len == len(messages)
        ):
            logger.debug("Summary cache hit (%d messages)", len(messages))
            return self._summary_cache

        logger.info("Building summary for %d messages", len(messages))
        summary = self._call_summarizer(messages)
        self._summary_cache = summary
        self._summary_input_len = len(messages)
        return summary

    def _call_summarizer(self, messages: list[dict]) -> str:
        """Call the LLM to produce a conversation summary."""
        formatted = _format_messages_for_summary(messages)
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SUMMARIZER_PROMPT},
                {"role": "user", "content": formatted},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_into_turns(messages: list[dict]) -> list[list[dict]]:
    """Group a flat message list into logical conversation turns.

    A turn ends when an assistant message has plain text content (no tool
    calls) — indicating a final answer was given.  Incomplete final turns
    (e.g. in mid-flight) are kept as a single group.
    """
    turns: list[list[dict]] = []
    current: list[dict] = []

    for msg in messages:
        current.append(msg)
        if (
            msg["role"] == "assistant"
            and msg.get("content")
            and not msg.get("tool_calls")
        ):
            turns.append(current)
            current = []

    if current:
        turns.append(current)

    return turns


def _format_messages_for_summary(messages: list[dict]) -> str:
    """Render a message list as plain text for the summariser."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_str = json.dumps(tool_calls, default=str)
            lines.append(f"{role} [tool_call]: {tc_str}")
        elif content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    SYSTEM = "You are a helpful research assistant."
    mem = MemoryManager(
        model="gpt-4o",
        max_tokens=2000,  # Low threshold so strategies trigger quickly
        system_prompt=SYSTEM,
    )

    # Simulate 20 turns of conversation
    topics = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is its population?", "Paris has about 2.1 million people in the city."),
        ("What language do they speak?", "French is the official language."),
        (
            "What currency?",
            "France uses the Euro (€), part of the Eurozone since 2002.",
        ),
        ("Name a famous landmark.", "The Eiffel Tower is world-famous."),
    ]
    for i, (user_msg, asst_msg) in enumerate(topics * 4):
        mem.add_user_message(f"[Turn {i+1}] {user_msg}")
        mem.add_assistant_message(asst_msg)

    print(f"Full history: {mem.token_count()} tokens, {len(mem.messages)} messages")

    for strategy in ("truncate", "summarize", "sliding_window"):
        # Use a mock summarizer for demo (no API key needed)
        if strategy in ("summarize", "sliding_window"):
            mem._call_summarizer = lambda msgs: (  # type: ignore[method-assign]
                f"[DEMO SUMMARY: {len(msgs)} messages compressed]"
            )
        msgs = mem.get_messages(strategy=strategy, recent_count=6)
        print(f"  {strategy:15s}: {count_tokens(msgs, 'gpt-4o'):5d} tokens, {len(msgs):3d} messages")

    cost = mem.estimated_cost()
    print(f"Estimated cost: ${cost['total_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
