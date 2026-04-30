"""Reusable parse-validate-retry handler for structured LLM extraction.

extract_with_retry() is a generic function that:
- Derives a JSON Schema from a Pydantic model via model_json_schema()
- Calls the OpenAI chat completions API with json_schema response_format
- Parses the raw JSON response into the target Pydantic model
- On ValidationError, appends a human-readable error to the message history
  and retries, giving the model a chance to self-correct
- Logs each attempt (attempt number, success/failure, error details)
- Raises ValidationError when max_retries is exhausted

Import and reuse across any extraction task in this repo.

See docs/01-foundations/03-structured-output.md — "The Parse-Validate-Retry Pattern"
"""

from __future__ import annotations

import logging
import os
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_with_retry(
    messages: list[dict],
    output_model: Type[T],
    max_retries: int = 3,
    model: str = "gpt-4o",
) -> T:
    """Call the LLM and parse into output_model, retrying on validation failure.

    Args:
        messages:     The message array to send.  Extended in-place on retry.
        output_model: Pydantic model class that defines the expected schema.
        max_retries:  Maximum number of attempts (default 3).
        model:        OpenAI model name to use.

    Returns:
        A validated instance of output_model.

    Raises:
        ValidationError: If every attempt fails validation.
        openai.OpenAIError: On API errors — these are not retried (fail fast).
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Build the strict-compatible schema: require all properties and forbid extras.
    schema = _make_strict_schema(output_model.model_json_schema())

    working_messages = list(messages)

    for attempt in range(1, max_retries + 1):
        logger.info("Attempt %d/%d", attempt, max_retries)

        response = client.chat.completions.create(
            model=model,
            messages=working_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__.lower(),
                    "schema": schema,
                    "strict": True,
                },
            },
        )

        raw = response.choices[0].message.content or ""

        try:
            result = output_model.model_validate_json(raw)
            logger.info("Attempt %d succeeded", attempt)
            return result
        except ValidationError as exc:
            logger.warning(
                "Attempt %d failed — %d validation error(s): %s",
                attempt,
                exc.error_count(),
                exc,
            )
            if attempt == max_retries:
                raise

            # Build a human-readable error message and let the model self-correct.
            human_readable = "; ".join(
                "{}: {}".format(
                    " → ".join(str(loc) for loc in e["loc"]) or "root",
                    e["msg"],
                )
                for e in exc.errors()
            )
            working_messages = [
                *working_messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your response did not match the required schema. "
                        f"Errors: {human_readable}. "
                        "Please correct your output and respond with valid JSON only."
                    ),
                },
            ]

    # Unreachable — loop always raises or returns inside.
    raise RuntimeError("extract_with_retry: unreachable code path")  # pragma: no cover


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_strict_schema(schema: dict) -> dict:
    """Post-process a Pydantic JSON schema to satisfy OpenAI strict mode.

    Strict mode requires:
    - Every object has ``additionalProperties: false``
    - Every property of an object appears in ``required``

    Pydantic v2 uses ``anyOf: [T, {type: null}]`` for Optional fields.
    We leave those intact — they satisfy strict mode as-is.
    """
    schema = dict(schema)

    if schema.get("type") == "object" and "properties" in schema:
        schema.setdefault("additionalProperties", False)
        schema.setdefault("required", list(schema["properties"].keys()))
        schema["properties"] = {
            k: _make_strict_schema(v) for k, v in schema["properties"].items()
        }

    # Recurse into $defs so nested models are also strict.
    if "$defs" in schema:
        schema["$defs"] = {k: _make_strict_schema(v) for k, v in schema["$defs"].items()}

    return schema
