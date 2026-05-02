"""Conversation summarizer — compresses message history for memory management.

Uses a cheap model (gpt-4o-mini by default) to produce dense,
information-preserving summaries of conversation history.

See: docs/03-memory-and-retrieval/01-short-term-memory.md
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FULL_SUMMARY_PROMPT = """\
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

_INCREMENTAL_SUMMARY_PROMPT = """\
You have an existing summary of a conversation and new messages that followed.
Update the summary to incorporate the new information.

Rules:
- Keep all important information from the existing summary.
- Add new facts, decisions, and outcomes from the new messages.
- Remove information that is no longer relevant.
- Keep the output as a single dense paragraph in third person past tense.\
"""

_KEY_FACTS_PROMPT = """\
Extract a list of key facts from this conversation.

A key fact is:
- A specific number, date, name, price, or measurement
- A decision or preference stated by the user
- A result or outcome from a tool call
- An unresolved question or pending task

Format as a JSON list of strings. Each item should be a single short sentence.
Example: ["The user's budget is $500.", "AAPL stock price is $192.35.",
          "The user needs the report by Friday 5pm EST."]

Return ONLY the JSON array, no other text.\
"""


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class ConversationSummarizer:
    """Summarizes conversation history into a concise, information-dense format.

    Uses a cheap model to compress history. The summariser preserves:
    goals, decisions, key data (numbers/dates/names), and pending items.

    Args:
        model:  LLM model for summarisation. Defaults to gpt-4o-mini.
        client: Optional pre-configured OpenAI client (for testing).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Optional[object] = None,
    ) -> None:
        self.model = model
        self._client: OpenAI = client or OpenAI(  # type: ignore[assignment]
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, messages: list[dict]) -> str:
        """Summarize a list of messages into a single dense paragraph.

        Preserves: goals, decisions, key data, pending items.

        Args:
            messages: Conversation messages to summarize.

        Returns:
            A dense paragraph summary, or empty string if nothing to summarize.
        """
        if not messages:
            return ""

        formatted = _format_for_summarizer(messages)
        logger.info(
            "Summarizing %d messages (%d chars)", len(messages), len(formatted)
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _FULL_SUMMARY_PROMPT},
                {"role": "user", "content": formatted},
            ],
            max_tokens=512,
        )
        summary = response.choices[0].message.content or ""
        logger.info("Summary: %d chars (%.0f%% compression)",
                    len(summary), (1 - len(summary) / max(len(formatted), 1)) * 100)
        return summary

    def summarize_incremental(
        self,
        existing_summary: str,
        new_messages: list[dict],
    ) -> str:
        """Update an existing summary with new messages.

        Faster than re-summarizing everything from scratch. Use when
        new messages have been added since the last summarization.

        Args:
            existing_summary: The current summary string.
            new_messages:     Messages that came after the summary was built.

        Returns:
            An updated summary incorporating the new messages.
        """
        if not new_messages:
            return existing_summary

        formatted_new = _format_for_summarizer(new_messages)
        user_content = (
            f"EXISTING SUMMARY:\n{existing_summary}\n\n"
            f"NEW MESSAGES:\n{formatted_new}"
        )

        logger.info("Incremental summary: adding %d messages", len(new_messages))

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _INCREMENTAL_SUMMARY_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
        )
        return response.choices[0].message.content or existing_summary

    def extract_key_facts(self, messages: list[dict]) -> list[str]:
        """Extract a list of key facts from the conversation.

        Useful for injecting specific context into new branches without
        the full conversation history.

        Args:
            messages: Conversation messages to analyse.

        Returns:
            List of key fact strings, or empty list on parse failure.
        """
        if not messages:
            return []

        formatted = _format_for_summarizer(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _KEY_FACTS_PROMPT},
                {"role": "user", "content": formatted},
            ],
            max_tokens=256,
        )
        raw = response.choices[0].message.content or "[]"
        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [str(f) for f in facts]
        except json.JSONDecodeError:
            logger.warning("key_facts response was not valid JSON: %r", raw)
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_for_summarizer(messages: list[dict]) -> str:
    """Render messages as plain text for the summariser LLM."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            tc_str = json.dumps(tool_calls, default=str)
            lines.append(f"{role} [tool_call]: {tc_str}")
        elif role == "TOOL":
            tool_id = msg.get("tool_call_id", "")
            lines.append(f"TOOL RESULT [{tool_id}]: {content}")
        elif content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from memory_manager import count_tokens

    # Build a synthetic conversation
    conversation: list[dict] = [
        {"role": "user", "content": "I need to research Apple stock. My budget for investing is $500."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "get_stock_price", "arguments": '{"ticker":"AAPL"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"price": 192.35, "change_pct": 1.2}'},
        {"role": "assistant", "content": "AAPL is currently $192.35, up 1.2% today."},
        {"role": "user", "content": "What about Microsoft?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "function": {"name": "get_stock_price", "arguments": '{"ticker":"MSFT"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": '{"price": 415.10, "change_pct": 0.8}'},
        {"role": "assistant", "content": "MSFT is at $415.10, up 0.8% today."},
        {"role": "user", "content": "Which is a better buy for my $500 budget?"},
        {"role": "assistant", "content": "With $500 you can buy 2 shares of AAPL at $192.35 each ($384.70 total, $115.30 remaining), or 1 share of MSFT at $415.10 ($84.90 remaining). With your $500 budget, AAPL gives you more shares and diversification opportunity. However, I recommend consulting a financial advisor before investing."},
    ]

    orig_tokens = count_tokens(conversation)
    print(f"Original: {len(conversation)} messages, {orig_tokens} tokens")

    # Demo with a mock summarizer (no API key needed)
    class MockSummarizer(ConversationSummarizer):
        def summarize(self, messages):
            return (
                "The user requested stock research with a $500 investment budget. "
                "The assistant retrieved AAPL at $192.35 (+1.2%) and MSFT at $415.10 (+0.8%). "
                "The assistant recommended AAPL for 2 shares within the budget, "
                "noting professional advice is warranted."
            )
        def extract_key_facts(self, messages):
            return [
                "User's investment budget is $500.",
                "AAPL stock price is $192.35 (up 1.2%).",
                "MSFT stock price is $415.10 (up 0.8%).",
                "AAPL recommendation: 2 shares for $384.70.",
            ]

    s = MockSummarizer()
    summary = s.summarize(conversation)
    summary_tokens = count_tokens([{"role": "user", "content": summary}])
    print(f"Summary:  1 message, {summary_tokens} tokens ({summary_tokens/orig_tokens*100:.0f}% of original)")

    facts = s.extract_key_facts(conversation)
    print(f"Key facts: {len(facts)}")
    for f in facts:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
