"""Zero-shot vs few-shot classification comparison.

Demonstrates the reliability gain from few-shot examples for sentiment
classification. Prints both results side-by-side with token counts so
you can measure the reliability/cost trade-off directly.

See docs/01-foundations/02-prompt-engineering.md — "Few-Shot Prompting"

When few-shot is worth the extra tokens:
  - Classification where the output format MUST be a single word/label.
  - Tasks where the model often adds unwanted explanation.
  - Edge cases where zero-shot returns inconsistent casing or phrasing.
For anything more complex, switch to structured output (json_schema).
"""

from __future__ import annotations

import os

import tiktoken
from openai import OpenAI

MODEL = "gpt-4o"
ALLOWED_LABELS = {"Positive", "Negative", "Neutral"}

ZERO_SHOT_SYSTEM = (
    "Classify the sentiment of the following text as exactly one of: "
    "Positive, Negative, or Neutral."
)

FEW_SHOT_SYSTEM = """\
Classify the sentiment of the following text as exactly one of: \
Positive, Negative, or Neutral.
Respond with exactly one word.

Examples:
Text: "I love this product!" → Positive
Text: "This is absolutely terrible." → Negative
Text: "It arrived on time." → Neutral
"""

# Tricky input: mildly positive but hedged — models often disagree zero-shot.
TEST_INPUT = (
    "The new update is fine, I guess. Not bad, but nothing to get excited about."
)


def _count(messages: list[dict]) -> int:
    enc = tiktoken.encoding_for_model(MODEL)
    total = (
        sum(3 + sum(len(enc.encode(v)) for v in msg.values()) for msg in messages) + 3
    )
    return total


def classify(system_prompt: str, text: str, client: OpenAI) -> tuple[str, int, int]:
    """Return (classification_label, estimated_prompt_tokens, actual_prompt_tokens)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Text: "{text}"'},
    ]
    estimated_tokens = _count(messages)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=10,
    )
    actual_prompt_tokens = response.usage.prompt_tokens
    return response.choices[0].message.content.strip(), estimated_tokens, actual_prompt_tokens


def format_status(label: str) -> str:
    """Return whether the model followed the requested one-word label format."""
    return "ok" if label in ALLOWED_LABELS else "drift"


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f'Input: "{TEST_INPUT}"\n')
    print(
        f"{'Approach':<12} {'Result':<12} {'Format':<8} {'Est. prompt':>12} {'API prompt':>12}"
    )
    print("-" * 64)

    zero_result, zero_estimated, zero_actual = classify(
        ZERO_SHOT_SYSTEM, TEST_INPUT, client
    )
    print(
        f"{'Zero-shot':<12} {zero_result:<12} {format_status(zero_result):<8} "
        f"{zero_estimated:>12} {zero_actual:>12}"
    )

    few_result, few_estimated, few_actual = classify(FEW_SHOT_SYSTEM, TEST_INPUT, client)
    print(
        f"{'Few-shot':<12} {few_result:<12} {format_status(few_result):<8} "
        f"{few_estimated:>12} {few_actual:>12}"
    )

    overhead = few_estimated - zero_estimated
    print(f"\nFew-shot overhead: +{overhead} estimated prompt tokens per request")
    print("A drift status means the model ignored the exact one-word label format.")


if __name__ == "__main__":
    main()
