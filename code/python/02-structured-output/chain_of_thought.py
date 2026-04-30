"""Chain-of-thought prompting demonstration.

Sends the same multi-step math problem with and without chain-of-thought
instructions. Both calls use temperature=0 for a fair, deterministic comparison.

See docs/01-foundations/02-prompt-engineering.md — "Chain-of-Thought"

Why chain-of-thought helps:
  - The model must commit to intermediate values before reaching the answer.
  - Wrong steps become visible and debuggable — critical in an agent loop.
  - For complex multi-step problems, CoT can lift accuracy by 20-40% at temp=0.
  - Trade-off: CoT costs more output tokens. Skip it for simple lookups.
"""

from __future__ import annotations

import os

from openai import OpenAI

MODEL = "gpt-4o"

# Multi-step problem that requires tracking multiple quantities — easy to
# get wrong without explicit step-by-step reasoning.
PROBLEM = (
    "A store sells apples for $1.20 each and bananas for $0.40 each. "
    "Alice buys 5 apples and 8 bananas. She pays with a $20 bill. "
    "How much change does she receive?"
)

WITHOUT_COT = [
    {"role": "system", "content": "You are a math assistant. Answer concisely."},
    {"role": "user", "content": PROBLEM},
]

WITH_COT = [
    {"role": "system", "content": "You are a math assistant."},
    {
        "role": "user",
        "content": PROBLEM + "\n\nThink step by step before giving the final answer.",
    },
]


def call(messages: list[dict], client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Problem:", PROBLEM)
    print("=" * 60)

    print("\n[WITHOUT chain-of-thought]")
    without = call(WITHOUT_COT, client)
    print(without)

    print("\n[WITH chain-of-thought]")
    with_cot = call(WITH_COT, client)
    print(with_cot)

    print("\n" + "=" * 60)
    print("Observation: the CoT response shows every arithmetic step.")
    print("If the answer is wrong, you can see exactly which step failed.")
    print("In an agent loop this means a wrong tool call is debuggable,")
    print("not just a black-box failure.")


if __name__ == "__main__":
    main()
