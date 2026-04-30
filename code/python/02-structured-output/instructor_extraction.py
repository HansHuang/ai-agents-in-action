"""Structured output extraction using Pydantic and Instructor.

Demonstrates:
- Defining a Pydantic model as the output schema with field descriptions
- Using instructor.from_openai() to patch the OpenAI client
- Automatic parse-validate-retry handled by Instructor (max_retries=2)
- Real-world fields: sentiment enum, confidence float, optional key phrases

See docs/01-foundations/03-structured-output.md — "Language-Specific Patterns"
"""

from __future__ import annotations

import os
from typing import Literal

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description=(
            "The overall sentiment of the text. Must be exactly one of: "
            "positive (favorable, happy, satisfied), "
            "negative (unfavorable, unhappy, dissatisfied), or "
            "neutral (neither clearly positive nor negative)."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are in the classification, as a float "
            "between 0.0 (no confidence) and 1.0 (completely certain)."
        ),
    )
    key_phrases: list[str] | None = Field(
        default=None,
        description=(
            "Up to 5 short phrases from the text that most influenced "
            "the sentiment classification. Omit if no clear phrases stand out."
        ),
    )


def extract_sentiment(text: str) -> SentimentResponse:
    """Extract sentiment from text with up to 2 automatic retries on failure.

    Instructor intercepts the API response, validates it against
    SentimentResponse, and retries with the validation error appended to
    the message history if it doesn't match.

    Args:
        text: The text to classify.

    Returns:
        A validated SentimentResponse instance.

    Raises:
        ValueError: If text is empty.
        instructor.exceptions.InstructorRetryException: After max_retries.
    """
    if not text.strip():
        raise ValueError("text must not be empty")

    client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise sentiment analysis engine. "
                    "Classify the sentiment of the user's text accurately."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_model=SentimentResponse,
        max_retries=2,
    )


def main() -> None:
    tests = [
        "I absolutely love this, it changed my life!",
        "It's fine I guess, nothing special.",
        "Terrible product, broke after one day.",
    ]
    for text in tests:
        result = extract_sentiment(text)
        print(f"Text       : {text!r}")
        print(f"Sentiment  : {result.sentiment}")
        print(f"Confidence : {result.confidence:.2f}")
        print(f"Key Phrases: {result.key_phrases}")
        print()


if __name__ == "__main__":
    main()
