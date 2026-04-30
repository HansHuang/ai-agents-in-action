"""Prompt template example: variable substitution, token counting, and LLM call.

Demonstrates the pattern described in:
docs/01-foundations/02-prompt-engineering.md — "Prompt Templates"

Functions are importable so they can be reused and tested independently.
"""

from __future__ import annotations

import os

import tiktoken
from openai import OpenAI

MODEL = "gpt-4o"

SYSTEM_PROMPT_TEMPLATE = """\
You are a technical summarizer.
Summarize the following article in 3 bullet points.
Focus on: {focus_area}"""

USER_PROMPT_TEMPLATE = """\
Article:
{article_text}"""


def build_system_prompt(focus_area: str) -> str:
    """Return the filled system prompt for the given focus area."""
    if not focus_area.strip():
        raise ValueError("focus_area must not be empty")
    return SYSTEM_PROMPT_TEMPLATE.format(focus_area=focus_area)


def build_user_prompt(article_text: str) -> str:
    """Return the filled user prompt for the given article text."""
    if not article_text.strip():
        raise ValueError("article_text must not be empty")
    return USER_PROMPT_TEMPLATE.format(article_text=article_text)


def build_messages(focus_area: str, article_text: str) -> list[dict]:
    """Return a messages array ready to send to chat.completions.create."""
    return [
        {"role": "system", "content": build_system_prompt(focus_area)},
        {"role": "user", "content": build_user_prompt(article_text)},
    ]


def count_tokens(messages: list[dict], model: str = MODEL) -> int:
    """Return the token cost of a messages array (includes API overhead).

    Accounts for the per-message overhead (3 tokens) and reply primer
    (3 tokens) that the API adds automatically.
    See: https://platform.openai.com/docs/guides/chat/managing-tokens
    """
    enc = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(enc.encode(value))
            if key == "name":
                total += tokens_per_name
    total += 3  # reply is primed with <|start|>assistant<|message|>
    return total


def main() -> None:
    focus_area = "practical implementation details"
    article_text = (
        "AI agents are software systems that use large language models as their "
        "reasoning engine. Unlike chatbots, agents can take actions: call APIs, "
        "search the web, write code, and orchestrate other agents. The key "
        "architectural pattern is the agent loop: perceive, think, act, observe. "
        "Production agents require harness engineering — input validation, retry "
        "logic, output guardrails, and human-in-the-loop checkpoints."
    )

    messages = build_messages(focus_area, article_text)
    tokens = count_tokens(messages)
    print(f"Token count before sending: {tokens}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
    )
    print("\nResponse:")
    print(response.choices[0].message.content)
    print(
        f"\nActual tokens used — prompt: {response.usage.prompt_tokens}, "
        f"completion: {response.usage.completion_tokens}"
    )


if __name__ == "__main__":
    main()
