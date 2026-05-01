"""ReAct agent orchestration loop.

Implements the Reason → Act → Observe cycle described in:
docs/02-the-agent-loop/01-anatomy-of-an-agent.md

The loop is intentionally simple and self-contained:
  1. Send messages + tools to the LLM (Reason).
  2. If the response contains tool_calls, execute each one (Act).
  3. Append results to messages and loop back (Observe).
  4. If the response contains only content (no tool_calls), return it.
  5. Safety: abort after MAX_ITERATIONS to prevent infinite loops.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from openai import OpenAI

from tool_dispatcher import ToolRegistry, dispatch_tool
from tools import TOOLS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are an AI assistant with access to tools.

## Your Process
1. When the user asks a question, determine if you need a tool to answer it.
2. If yes, call the appropriate tool with the correct parameters.
3. Wait for the tool result, then determine if you need more tools or can answer.
4. Never guess tool results. Always wait for the actual result.
5. If a tool fails, explain the failure to the user and suggest alternatives.

## Tool Usage Rules
- Call only one tool at a time unless they are independent.
- If you don't have enough information to call a tool, ask the user.
- Never make up parameters. If unsure, ask for clarification.

## Answer Format
- Use the tool results to answer the user's question directly.
- Cite specific data from tool results.
- If multiple tools were used, synthesize the information."""


def run_agent(
    user_input: str,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    registry: Optional[ToolRegistry] = None,
) -> str:
    """Run the ReAct loop until a final answer is reached.

    Args:
        user_input: The user's question or instruction.
        messages:   Existing message history. A fresh history (with the system
                    prompt) is created when None. Extended in-place on each turn.
        tools:      Tool definitions to pass to the LLM. Defaults to TOOLS.
        registry:   Tool execution registry. Defaults to the built-in registry
                    in tool_dispatcher.py.

    Returns:
        The agent's final answer string.

    Raises:
        ValueError: If user_input is empty or whitespace.
    """
    if not user_input.strip():
        raise ValueError("user_input must not be empty")

    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if tools is None:
        tools = TOOLS

    messages.append({"role": "user", "content": user_input})
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.debug("Agent iteration %d/%d", iteration, MAX_ITERATIONS)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Always append the assistant's turn BEFORE processing tool calls.
        # The API requires the assistant message (with tool_calls) to be present
        # in the history before any of the corresponding tool result messages.
        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            logger.debug("Agent finished after %d iteration(s)", iteration)
            return msg.content or ""

        # Execute every tool call and append results to history.
        for tool_call in msg.tool_calls:
            tool_msg = dispatch_tool(tool_call, registry)
            messages.append(tool_msg)

    # Safety valve: MAX_ITERATIONS exhausted without a final answer.
    logger.warning("Agent exceeded MAX_ITERATIONS (%d) without finishing", MAX_ITERATIONS)
    return (
        "I was unable to complete your request within the allowed number of "
        "steps. Please try rephrasing your question or breaking it into "
        "smaller parts."
    )


def main() -> None:
    """Run two demo queries and print full conversation history for each."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    queries = [
        "What's the weather in Shanghai?",
        "Should I invest in Apple stock right now?",
    ]

    for query in queries:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        answer = run_agent(query, messages=messages, tools=TOOLS)

        print(f"\nFinal Answer:\n{answer}")
        print("\n--- Full Conversation History ---")

        for msg in messages:
            role = msg["role"].upper()
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    print(f"  [{role}] → tool call: {fn['name']}({fn['arguments']})")
            elif role == "TOOL":
                content = json.loads(msg["content"])
                print(f"  [{role}] ← result: {content}")
            elif role == "SYSTEM":
                preview = msg["content"][:60].replace("\n", " ")
                print(f"  [{role}] {preview}…")
            else:
                content = (msg.get("content") or "")
                preview = content[:80].replace("\n", " ") + ("…" if len(content) > 80 else "")
                print(f"  [{role}] {preview}")
