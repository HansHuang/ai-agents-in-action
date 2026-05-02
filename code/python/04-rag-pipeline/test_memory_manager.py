"""Tests for memory management infrastructure.

12 tests covering:
  - Token counting accuracy
  - Truncation strategy (system prompt preserved, complete turns, oldest removed first)
  - Summarization strategy
  - Sliding window strategy
  - Branch manager
  - Token tracker
"""

from __future__ import annotations

import sys
import os

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from memory_manager import MemoryManager, count_tokens, _group_into_turns
from conversation_summarizer import ConversationSummarizer
from branch_manager import BranchManager
from token_tracker import TokenTracker, TokenUsage, PRICING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(summary_text: str = "Mock summary.") -> MagicMock:
    """Return a mock OpenAI client whose completions return summary_text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = summary_text
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _make_memory(
    max_tokens: int = 100_000,
    system: str = "You are a helpful assistant.",
    client: object = None,
) -> MemoryManager:
    """Create a MemoryManager with a test client (avoids real API calls)."""
    client = client or _make_mock_client()
    return MemoryManager(
        model="gpt-4o",
        max_tokens=max_tokens,
        system_prompt=system,
        client=client,
    )


def _make_turns(mem: MemoryManager, count: int = 5) -> None:
    """Add `count` complete user→assistant turns to mem."""
    for i in range(count):
        mem.add_user_message(f"User turn {i}: tell me something about topic {i}.")
        mem.add_assistant_message(f"Assistant answer {i}: here is information about topic {i}.")


# ---------------------------------------------------------------------------
# 1. Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_add_messages_increases_token_count(self):
        """Each message added should increase the token count."""
        mem = _make_memory()
        tokens_before = mem.token_count()

        mem.add_user_message("Hello, how are you?")
        tokens_after = mem.token_count()

        assert tokens_after > tokens_before, (
            f"Expected token count to increase; before={tokens_before}, after={tokens_after}"
        )

    def test_token_counting_includes_overhead(self):
        """count_tokens should include per-message overhead (4 tokens per message)."""
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")

        messages = [{"role": "user", "content": "hi"}]
        # Our implementation encodes all key values (role + content) plus:
        #   4 tokens per-message overhead + 2 tokens priming
        raw_role_tokens = len(enc.encode("user"))
        raw_content_tokens = len(enc.encode("hi"))
        total = count_tokens(messages, model="gpt-4o")

        # Must be strictly more than raw content alone
        assert total > raw_content_tokens, (
            f"Expected overhead to be counted; raw_content={raw_content_tokens}, total={total}"
        )
        # Exact formula: role_tokens + content_tokens + 4 (msg overhead) + 2 (priming)
        expected = raw_role_tokens + raw_content_tokens + 4 + 2
        assert total == expected, (
            f"Token count mismatch; expected {expected}, got {total}"
        )


# ---------------------------------------------------------------------------
# 2. Truncation strategy
# ---------------------------------------------------------------------------


class TestTruncationStrategy:
    def test_truncation_keeps_system_prompt(self):
        """Truncation must always preserve the system prompt as the first message."""
        system = "You are a very helpful assistant with detailed instructions."
        mem = _make_memory(max_tokens=200, system=system)
        _make_turns(mem, count=10)

        result = mem.get_messages(strategy="truncate")

        assert result[0]["role"] == "system", "First message must always be the system prompt"
        assert result[0]["content"] == system

    def test_truncation_keeps_complete_turns(self):
        """Truncation must not leave orphaned tool results — whole turns only."""
        mem = _make_memory(max_tokens=300)

        # Add a turn with a tool call + tool result
        mem.add_user_message("What is AAPL stock price?")
        mem.add_assistant_message(
            content=None,
            tool_calls=[{"id": "tc1", "function": {"name": "get_price", "arguments": '{"ticker":"AAPL"}'}}],
        )
        mem.add_tool_result("tc1", '{"price": 192.35}')
        mem.add_assistant_message("AAPL is $192.35.")
        # Second turn
        mem.add_user_message("And MSFT?")
        mem.add_assistant_message("MSFT is $415.")

        result = mem.get_messages(strategy="truncate")

        # Verify that no tool result message appears without a preceding tool call
        has_tool_result = any(m.get("role") == "tool" for m in result)
        if has_tool_result:
            roles = [m["role"] for m in result]
            for i, role in enumerate(roles):
                if role == "tool":
                    # The tool result must have a preceding assistant+tool_calls message
                    preceding_has_tool_call = any(
                        m.get("tool_calls") for m in result[:i]
                    )
                    assert preceding_has_tool_call, (
                        "Tool result found without preceding tool call in truncated history"
                    )

    def test_truncation_removes_oldest_first(self):
        """When truncating, the most recent turns are kept; old ones are dropped."""
        mem = _make_memory(max_tokens=350)

        for i in range(5):
            mem.add_user_message(f"Old question number {i}.")
            mem.add_assistant_message(f"Old answer number {i}.")

        # These should survive the truncation
        mem.add_user_message("Recent: what is 2+2?")
        mem.add_assistant_message("Recent: 2+2=4.")

        result = mem.get_messages(strategy="truncate")
        all_content = " ".join(m.get("content") or "" for m in result)

        assert "Recent: what is 2+2?" in all_content, "Most recent turn must be preserved"
        assert "Recent: 2+2=4." in all_content, "Most recent assistant answer must be preserved"


# ---------------------------------------------------------------------------
# 3. Summarization strategy
# ---------------------------------------------------------------------------


class TestSummarizationStrategy:
    def test_summarization_reduces_token_count(self):
        """The summarize strategy should produce a shorter context than the original."""
        mock_client = _make_mock_client(summary_text="Brief summary of the conversation.")
        mem = _make_memory(max_tokens=200, client=mock_client)
        _make_turns(mem, count=8)

        original_tokens = mem.token_count()
        result = mem.get_messages(strategy="summarize", recent_count=4)
        result_tokens = count_tokens(result, "gpt-4o")

        assert result_tokens < original_tokens, (
            f"Summarize should reduce tokens; original={original_tokens}, after={result_tokens}"
        )

    def test_summarization_preserves_key_information(self):
        """The summary injected by the strategy should contain the summary text."""
        key_info = "The user's budget is $500 and they prefer AAPL."
        mock_client = _make_mock_client(summary_text=key_info)
        mem = _make_memory(max_tokens=200, client=mock_client)
        _make_turns(mem, count=6)

        result = mem.get_messages(strategy="summarize", recent_count=2)

        # Find the injected summary message
        summary_msgs = [
            m for m in result
            if m.get("role") == "user" and "Conversation summary" in (m.get("content") or "")
        ]
        assert summary_msgs, "No summary message found in result"
        assert key_info in summary_msgs[0]["content"], (
            f"Expected key info in summary; got: {summary_msgs[0]['content']!r}"
        )


# ---------------------------------------------------------------------------
# 4. Sliding window strategy
# ---------------------------------------------------------------------------


class TestSlidingWindowStrategy:
    def test_sliding_window_combines_both(self):
        """Sliding window must include both a summary message and recent verbatim messages."""
        summary_text = "Summary of earlier conversation."
        mock_client = _make_mock_client(summary_text=summary_text)
        mem = _make_memory(max_tokens=200, client=mock_client)
        _make_turns(mem, count=6)

        mem.add_user_message("RECENT: what is the weather today?")
        mem.add_assistant_message("RECENT: it is sunny.")

        result = mem.get_messages(strategy="sliding_window", recent_count=4)

        # Should have a summary message
        summary_msgs = [
            m for m in result
            if m.get("role") == "user" and "Conversation so far" in (m.get("content") or "")
        ]
        assert summary_msgs, "No sliding-window summary found in result"

        # Most recent messages should be verbatim
        all_content = " ".join(m.get("content") or "" for m in result)
        assert "RECENT: what is the weather today?" in all_content
        assert "RECENT: it is sunny." in all_content


# ---------------------------------------------------------------------------
# 5. Branches
# ---------------------------------------------------------------------------


class TestBranchManager:
    def _make_bm(self, summary_text: str = "Branch summary.") -> BranchManager:
        mock_client = _make_mock_client(summary_text)
        return BranchManager(
            system_prompt="You are a research assistant.",
            client=mock_client,
        )

    def test_branches_have_independent_messages(self):
        """Messages added to one branch must not appear in other branches."""
        bm = self._make_bm()

        branch_a = bm.create_branch("a")
        branch_b = bm.create_branch("b")

        bm.add_to_branch(branch_a, {"role": "user", "content": "Branch A only."})
        bm.add_to_branch(branch_b, {"role": "user", "content": "Branch B only."})

        msgs_a = " ".join(m.get("content") or "" for m in bm.get_branch(branch_a).messages)
        msgs_b = " ".join(m.get("content") or "" for m in bm.get_branch(branch_b).messages)

        assert "Branch A only." in msgs_a, "Branch A message not found in A"
        assert "Branch A only." not in msgs_b, "Branch A message leaked into B"
        assert "Branch B only." in msgs_b, "Branch B message not found in B"
        assert "Branch B only." not in msgs_a, "Branch B message leaked into A"

    def test_merge_context_injects_summary(self):
        """merge_context must inject a source branch's summary into the target."""
        summary_text = "Source branch found AAPL at $192."
        bm = self._make_bm(summary_text=summary_text)

        source = bm.create_branch("source")
        bm.add_to_branch(source, {"role": "user", "content": "AAPL research."})
        bm.add_to_branch(source, {"role": "assistant", "content": "AAPL is $192."})

        target = bm.create_branch("target")
        bm.merge_context(target, [source])

        target_msgs = bm.get_branch(target).messages
        injected = [
            m for m in target_msgs
            if summary_text in (m.get("content") or "")
        ]
        assert injected, (
            f"Expected summary text to be injected into target; target messages: {target_msgs}"
        )


# ---------------------------------------------------------------------------
# 6. Token tracker
# ---------------------------------------------------------------------------


class TestTokenTracker:
    def test_token_tracker_accumulates_usage(self):
        """TokenTracker must accumulate tokens across multiple calls."""
        tracker = TokenTracker()

        tracker.record_call("gpt-4o", 1000, 200, purpose="step1")
        tracker.record_call("gpt-4o-mini", 500, 100, purpose="step2")

        assert tracker.total_input_tokens() == 1500
        assert tracker.total_output_tokens() == 300
        assert tracker.total_tokens() == 1800

        # Cost: gpt-4o: 1000/1000*0.0025 + 200/1000*0.01 = 0.0025+0.002 = 0.0045
        #       gpt-4o-mini: 500/1000*0.00015 + 100/1000*0.0006 = 0.000075+0.00006 = 0.000135
        expected_cost = (
            1000 / 1000 * PRICING["gpt-4o"]["input"]
            + 200 / 1000 * PRICING["gpt-4o"]["output"]
            + 500 / 1000 * PRICING["gpt-4o-mini"]["input"]
            + 100 / 1000 * PRICING["gpt-4o-mini"]["output"]
        )
        assert abs(tracker.total_cost() - expected_cost) < 1e-9

    def test_budget_warning_at_80_percent(self):
        """TokenTracker should log a WARNING when 80% of the budget is consumed."""
        import logging

        budget = 0.01
        tracker = TokenTracker(budget_cap=budget)

        # Spend exactly 90% of budget via gpt-4o input tokens:
        # total_input_cost = tokens/1000 * 0.0025
        # 0.009 = tokens/1000 * 0.0025 → tokens = 3600
        with patch.object(logging.getLogger("token_tracker"), "warning") as mock_warn:
            tracker.record_call("gpt-4o", 3600, 0, purpose="big_call")
            mock_warn.assert_called_once()
            warning_message = mock_warn.call_args[0][0]
            assert "Budget warning" in warning_message or "80" in str(mock_warn.call_args)

    def test_budget_exceeded_returns_true(self):
        """is_budget_exceeded must return True when total cost >= budget cap."""
        tracker = TokenTracker(budget_cap=0.001)

        # gpt-4o: 1000 input tokens → $0.0025 > $0.001
        tracker.record_call("gpt-4o", 1000, 0)

        assert tracker.is_budget_exceeded() is True

    def test_generate_report_contains_model_breakdown(self):
        """generate_report must include per-model lines."""
        tracker = TokenTracker()
        tracker.record_call("gpt-4o", 100, 50, purpose="test")
        tracker.record_call("gpt-4o-mini", 200, 30, purpose="summarize")

        report = tracker.generate_report()

        assert "gpt-4o" in report
        assert "gpt-4o-mini" in report
        assert "TOKEN USAGE REPORT" in report
