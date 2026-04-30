"""Token counting for OpenAI-compatible models using tiktoken.

Shows how to count tokens for a plain string and for a messages array
(the format used by chat.completions.create). The messages array count
mirrors what the API actually charges you for.
"""

from __future__ import annotations

import tiktoken

MODEL = "gpt-4o"


def count_tokens(text: str, model: str = MODEL) -> int:
    """Return the number of tokens in *text* for the given model."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict], model: str = MODEL) -> int:
    """Return the token cost of a messages array for chat completions.

    Accounts for the per-message overhead (3 tokens) and reply primer (3
    tokens) that the API adds automatically.
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
    text = "The quick brown fox jumps over the lazy dog."
    print(f"Text  : {text!r}")
    print(f"Tokens: {count_tokens(text)}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    print(f"\nMessages array token count: {count_messages_tokens(messages)}")


if __name__ == "__main__":
    main()
