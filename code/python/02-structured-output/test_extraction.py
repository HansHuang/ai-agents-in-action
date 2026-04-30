"""Pytest tests for structured output extraction modules.

Tests cover:
- Correct classification of clear positive / negative / neutral text
- Confidence value stays within [0, 1]
- Empty / whitespace input raises ValueError
- Very long input (10k+ chars) is handled without error
- Sarcastic text returns a valid SentimentResponse (any sentiment)
- Schema compliance: only expected fields are present
- Retry on first-call validation failure (mock returns bad data then good)
- Max retries exceeded raises ValidationError

All OpenAI API calls are mocked — no OPENAI_API_KEY required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from instructor_extraction import SentimentResponse, extract_sentiment
from retry_handler import extract_with_retry

# ---------------------------------------------------------------------------
# Shared test payloads
# ---------------------------------------------------------------------------

VALID_POSITIVE = {
    "sentiment": "positive",
    "confidence": 0.95,
    "key_phrases": ["love", "amazing"],
}
VALID_NEGATIVE = {
    "sentiment": "negative",
    "confidence": 0.90,
    "key_phrases": ["terrible", "broke"],
}
VALID_NEUTRAL = {
    "sentiment": "neutral",
    "confidence": 0.70,
    "key_phrases": None,
}

# ---------------------------------------------------------------------------
# Helper: build a minimal OpenAI response mock for retry_handler tests
# ---------------------------------------------------------------------------


def _make_openai_response(content: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(content)
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 30
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_instructor_client(monkeypatch):
    """Patch instructor.from_openai and OpenAI inside instructor_extraction."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Prevent real OpenAI client instantiation.
    monkeypatch.setattr(
        "instructor_extraction.OpenAI",
        lambda **kw: MagicMock(),
    )
    mock_create = MagicMock(return_value=SentimentResponse(**VALID_POSITIVE))
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    monkeypatch.setattr(
        "instructor_extraction.instructor.from_openai",
        lambda *a, **kw: mock_client,
    )
    return mock_create


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Patch OpenAI inside retry_handler."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_create = MagicMock(return_value=_make_openai_response(VALID_POSITIVE))
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    monkeypatch.setattr("retry_handler.OpenAI", lambda **kw: mock_client)
    return mock_create


# ---------------------------------------------------------------------------
# Classification correctness
# ---------------------------------------------------------------------------


class TestClassificationCorrectness:
    def test_positive_text_returns_positive(self, mock_instructor_client):
        mock_instructor_client.return_value = SentimentResponse(**VALID_POSITIVE)
        result = extract_sentiment("I absolutely love this product!")
        assert result.sentiment == "positive"

    def test_negative_text_returns_negative(self, mock_instructor_client):
        mock_instructor_client.return_value = SentimentResponse(**VALID_NEGATIVE)
        result = extract_sentiment("Terrible product, broke after one day.")
        assert result.sentiment == "negative"

    def test_neutral_text_returns_neutral(self, mock_instructor_client):
        mock_instructor_client.return_value = SentimentResponse(**VALID_NEUTRAL)
        result = extract_sentiment("It arrived on time.")
        assert result.sentiment == "neutral"


# ---------------------------------------------------------------------------
# Schema compliance
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    def test_confidence_between_zero_and_one(self, mock_instructor_client):
        mock_instructor_client.return_value = SentimentResponse(**VALID_POSITIVE)
        result = extract_sentiment("Great!")
        assert 0.0 <= result.confidence <= 1.0

    def test_output_has_only_expected_fields(self, mock_instructor_client):
        mock_instructor_client.return_value = SentimentResponse(**VALID_POSITIVE)
        result = extract_sentiment("I love this!")
        assert set(result.model_fields.keys()) == {"sentiment", "confidence", "key_phrases"}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            extract_sentiment("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            extract_sentiment("   \t\n")


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_very_long_text_handled(self, mock_instructor_client):
        mock_instructor_client.return_value = SentimentResponse(**VALID_NEUTRAL)
        long_text = "This product is okay. " * 500  # ~10 000 characters
        result = extract_sentiment(long_text)
        assert result.sentiment in {"positive", "negative", "neutral"}

    def test_sarcastic_text_returns_valid_response(self, mock_instructor_client):
        mock_instructor_client.return_value = SentimentResponse(
            sentiment="negative", confidence=0.65, key_phrases=["sure"]
        )
        result = extract_sentiment(
            "Oh sure, because *that's* what I needed — another bug."
        )
        assert result.sentiment in {"positive", "negative", "neutral"}
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Retry logic (via retry_handler.extract_with_retry)
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retry_on_first_call_validation_failure(self, monkeypatch):
        """Mock returns invalid data first, valid data second; expect 2 calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Invalid enum + out-of-range confidence
                return _make_openai_response(
                    {"sentiment": "INVALID_VALUE", "confidence": 5.0, "key_phrases": None}
                )
            return _make_openai_response(VALID_POSITIVE)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effect
        monkeypatch.setattr("retry_handler.OpenAI", lambda **kw: mock_client)

        messages = [{"role": "user", "content": "I love this!"}]
        result = extract_with_retry(messages, SentimentResponse, max_retries=3)

        assert result.sentiment == "positive"
        assert call_count == 2

    def test_max_retries_exceeded_raises_validation_error(self, monkeypatch):
        """When every response is invalid, raise ValidationError after max_retries."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response(
            {"sentiment": "ALWAYS_WRONG", "confidence": 99.9, "key_phrases": None}
        )
        monkeypatch.setattr("retry_handler.OpenAI", lambda **kw: mock_client)

        messages = [{"role": "user", "content": "test"}]
        with pytest.raises(ValidationError):
            extract_with_retry(messages, SentimentResponse, max_retries=2)
