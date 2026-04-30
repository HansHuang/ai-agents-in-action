"""Function Calling vs. Structured Output: side-by-side comparison.

Runs the same sentiment-extraction task through both API paths on 5 test texts
and prints a per-text table plus a summary comparing success rate, total tokens,
and average latency for each method.

See docs/01-foundations/03-structured-output.md
  — "Function Calling vs. Structured Output: The Real Difference"
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

MODEL = "gpt-4o"


class SentimentResponse(BaseModel):
    """Output schema shared by both extraction paths."""

    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0, le=1)
    key_phrases: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Function-calling definition (Path A)
# ---------------------------------------------------------------------------

FUNCTION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "classify_sentiment",
        "description": "Classify the sentiment of the provided text.",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "Overall sentiment.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score between 0 and 1.",
                },
                "key_phrases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                    "description": "Key phrases that influenced the classification.",
                },
            },
            "required": ["sentiment", "confidence"],
        },
    },
}

# ---------------------------------------------------------------------------
# JSON-schema response_format definition (Path B)
# ---------------------------------------------------------------------------

_SCHEMA = SentimentResponse.model_json_schema()
# Strict mode requires additionalProperties: false and all properties in required.
_SCHEMA.setdefault("additionalProperties", False)
_SCHEMA.setdefault("required", list(_SCHEMA.get("properties", {}).keys()))

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class Result:
    method: str
    text: str
    success: bool
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_function_calling(text: str, client: OpenAI) -> Result:
    """Extract sentiment via function calling (Path A)."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": text}],
            tools=[FUNCTION_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "classify_sentiment"},
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        parsed = SentimentResponse(**args)
        return Result(
            method="function_calling",
            text=text,
            success=True,
            sentiment=parsed.sentiment,
            confidence=parsed.confidence,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return Result(
            method="function_calling",
            text=text,
            success=False,
            error=str(exc),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def extract_structured_output(text: str, client: OpenAI) -> Result:
    """Extract sentiment via structured output / json_schema (Path B)."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": text}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_response",
                    "schema": _SCHEMA,
                    "strict": True,
                },
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.choices[0].message.content or ""
        parsed = SentimentResponse.model_validate_json(raw)
        return Result(
            method="structured_output",
            text=text,
            success=True,
            sentiment=parsed.sentiment,
            confidence=parsed.confidence,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return Result(
            method="structured_output",
            text=text,
            success=False,
            error=str(exc),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# Test texts
# ---------------------------------------------------------------------------

TEST_TEXTS = [
    "I absolutely love this product! Best purchase I've ever made.",
    "This is absolutely terrible. Complete waste of money.",
    "It arrived. Haven't tried it yet.",
    "Oh sure, because *that's* exactly what I needed — another broken feature.",  # sarcastic
    "Ce produit est fantastique, je le recommande vivement.",  # French
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    fc_results: list[Result] = []
    so_results: list[Result] = []

    for text in TEST_TEXTS:
        fc = extract_function_calling(text, client)
        so = extract_structured_output(text, client)
        fc_results.append(fc)
        so_results.append(so)

        agree = (
            fc.sentiment == so.sentiment
            if fc.success and so.success
            else "N/A"
        )
        print(f"\nText: {text[:65]!r}")
        print(f"  {'Method':<22} {'Result':<12} {'Tokens':>8} {'Latency':>10}  Status")
        print(f"  {'-'*58}")
        print(
            f"  {'function_calling':<22} {fc.sentiment or 'FAIL':<12} "
            f"{fc.prompt_tokens + fc.completion_tokens:>8} "
            f"{fc.latency_ms:>9.0f}ms  {'✓' if fc.success else '✗'}"
        )
        print(
            f"  {'structured_output':<22} {so.sentiment or 'FAIL':<12} "
            f"{so.prompt_tokens + so.completion_tokens:>8} "
            f"{so.latency_ms:>9.0f}ms  {'✓' if so.success else '✗'}"
        )
        print(f"  Results agree: {agree}")

    # Summary
    print("\n" + "=" * 58)
    print("SUMMARY")
    print("=" * 58)
    for label, results in [
        ("Function Calling", fc_results),
        ("Structured Output", so_results),
    ]:
        successes = [r for r in results if r.success]
        total_tok = sum(r.prompt_tokens + r.completion_tokens for r in successes)
        avg_tok = total_tok // max(len(successes), 1)
        avg_lat = sum(r.latency_ms for r in results) / max(len(results), 1)
        print(f"\n{label}:")
        print(f"  Success rate : {len(successes)}/{len(results)}")
        print(f"  Avg tokens   : {avg_tok}")
        print(f"  Avg latency  : {avg_lat:.0f}ms")


if __name__ == "__main__":
    main()
