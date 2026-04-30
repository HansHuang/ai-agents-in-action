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


def classify(system_prompt: str, text: str, client: OpenAI) -> tuple[str, int]:
    """Return (classification_label, token_count)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Text: "{text}"'},
    ]
    tokens = _count(messages)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip(), tokens


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f'Input: "{TEST_INPUT}"\n')
    print(f"{'Approach':<12} {'Result':<12} {'Tokens sent':>12}")
    print("-" * 40)

    zero_result, zero_tokens = classify(ZERO_SHOT_SYSTEM, TEST_INPUT, client)
    print(f"{'Zero-shot':<12} {zero_result:<12} {zero_tokens:>12}")

    few_result, few_tokens = classify(FEW_SHOT_SYSTEM, TEST_INPUT, client)
    print(f"{'Few-shot':<12} {few_result:<12} {few_tokens:>12}")

    overhead = few_tokens - zero_tokens
    print(f"\nFew-shot overhead: +{overhead} tokens per request")


if __name__ == "__main__":
    main()
