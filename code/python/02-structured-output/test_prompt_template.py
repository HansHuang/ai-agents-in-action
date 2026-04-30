"""Tests for the prompt_template module.

Run with: pytest test_prompt_template.py -v
No API key required — tests exercise template logic and token counting only.
"""

import pytest

from prompt_template import (
    build_messages,
    build_system_prompt,
    build_user_prompt,
    count_tokens,
)


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_contains_focus_area(self):
        result = build_system_prompt("practical implementation details")
        assert "practical implementation details" in result

    def test_no_leftover_braces(self):
        result = build_system_prompt("testing")
        assert "{" not in result and "}" not in result

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            build_system_prompt("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            build_system_prompt("   ")


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_contains_article_text(self):
        result = build_user_prompt("AI agents are fascinating.")
        assert "AI agents are fascinating." in result

    def test_no_leftover_braces(self):
        result = build_user_prompt("Some article text")
        assert "{" not in result and "}" not in result

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            build_user_prompt("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            build_user_prompt("   ")

    def test_very_long_input_does_not_crash(self):
        long_text = "word " * 2_000
        result = build_user_prompt(long_text)
        assert long_text in result


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_returns_two_messages(self):
        msgs = build_messages("focus", "text")
        assert len(msgs) == 2

    def test_first_role_is_system(self):
        msgs = build_messages("focus", "text")
        assert msgs[0]["role"] == "system"

    def test_second_role_is_user(self):
        msgs = build_messages("focus", "text")
        assert msgs[1]["role"] == "user"

    def test_system_contains_focus_area(self):
        msgs = build_messages("unique-focus-area", "text")
        assert "unique-focus-area" in msgs[0]["content"]

    def test_user_contains_article_text(self):
        msgs = build_messages("focus", "unique-article-content")
        assert "unique-article-content" in msgs[1]["content"]

    def test_no_leftover_braces_in_any_message(self):
        msgs = build_messages("focus", "text")
        for msg in msgs:
            assert "{" not in msg["content"] and "}" not in msg["content"]

    def test_empty_focus_raises(self):
        with pytest.raises(ValueError):
            build_messages("", "text")

    def test_empty_article_raises(self):
        with pytest.raises(ValueError):
            build_messages("focus", "")


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_returns_positive_integer(self):
        msgs = build_messages("focus area", "article text here")
        assert count_tokens(msgs) > 0

    def test_longer_input_yields_more_tokens(self):
        short = build_messages("focus", "short text")
        long = build_messages("focus", "short text " + "extra content " * 100)
        assert count_tokens(long) > count_tokens(short)

    def test_very_long_input_does_not_crash(self):
        msgs = build_messages("focus", "word " * 2_000)
        assert count_tokens(msgs) > 0

    def test_empty_messages_list_returns_integer(self):
        # Empty list still returns the reply-primer overhead (3 tokens).
        result = count_tokens([])
        assert isinstance(result, int)
        assert result == 3
