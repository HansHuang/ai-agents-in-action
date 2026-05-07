"""Context Budget Manager — zone-based token allocation and enforcement.

Every LLM call passes through the budget enforcer which ensures each zone
stays within its allocated token quota.  When a zone overflows, the enforcer
applies the appropriate compression strategy and records what it did in a
structured audit trail.

See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def _get_encoding(model: str = "gpt-4o") -> tiktoken.Encoding:
    """Return the tiktoken encoding for *model*, falling back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(
    content: "str | list[dict] | dict | None",
    model: str = "gpt-4o",
) -> int:
    """Count tokens for strings, message lists, or tool definition dicts.

    Follows OpenAI chat-completion token counting rules:
    - 3 overhead tokens per message
    - +1 token when a ``name`` field is present
    - 3 primer tokens appended to the total for the reply

    Args:
        content: A string, a list of message dicts, a single dict, or ``None``.
        model:   Model name used to select the correct BPE encoding.

    Returns:
        Integer token count.
    """
    if content is None:
        return 0

    enc = _get_encoding(model)

    if isinstance(content, str):
        return len(enc.encode(content))

    if isinstance(content, dict):
        return count_tokens(json.dumps(content), model)

    if isinstance(content, list):
        if not content:
            return 0

        # Message list (dicts with "role")
        if all(isinstance(m, dict) and "role" in m for m in content):
            tokens_per_message = 3
            tokens_per_name = 1
            total = 0
            for msg in content:
                total += tokens_per_message
                for key, val in msg.items():
                    if isinstance(val, str):
                        total += len(enc.encode(val))
                    elif isinstance(val, (dict, list)):
                        total += len(enc.encode(json.dumps(val)))
                    if key == "name":
                        total += tokens_per_name
            total += 3  # reply primer
            return total

        # List of tool definitions or other dicts
        return sum(count_tokens(item, model) for item in content)

    return 0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ZoneAudit:
    """Audit record for a single context zone."""

    zone: str
    original_tokens: int
    budget_tokens: int
    final_tokens: int
    action_taken: str  # "within_budget" | "truncated" | "sliding_window" | "filtered" | "reserved"

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.final_tokens)


@dataclass
class EnforceResult:
    """Result of budget enforcement with a full audit trail.

    Attributes:
        system_prompt:    Compressed (or original) system prompt text.
        messages:         Compressed (or original) message list.
        dynamic_context:  Compressed (or original) dynamic context string.
        tool_definitions: Compressed (or original) tool definition list.
        audit:            Per-zone :class:`ZoneAudit` records.
        warnings:         Human-readable warning strings for zones that were compressed.
    """

    system_prompt: str
    messages: list[dict]
    dynamic_context: str
    tool_definitions: list[dict]
    audit: dict[str, ZoneAudit] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_tokens_saved(self) -> int:
        """Sum of tokens saved across all zones."""
        return sum(a.tokens_saved for a in self.audit.values())

    @property
    def total_tokens_used(self) -> int:
        """Sum of final tokens across all zones."""
        return sum(a.final_tokens for a in self.audit.values())


class BudgetExceededError(Exception):
    """Raised when content cannot be compressed enough to fit the budget."""


# ---------------------------------------------------------------------------
# ContextBudget
# ---------------------------------------------------------------------------


class ContextBudget:
    """Define and enforce token allocation across context zones.

    Every LLM call should run through :meth:`enforce` to keep each zone
    within its allocated quota.  Zones that overflow are compressed using
    zone-specific strategies; everything is recorded in an audit trail.

    Example::

        budget = ContextBudget(total_tokens=128_000)
        result = budget.enforce(
            system_prompt,
            messages,
            dynamic_context=rag_text,
            tool_definitions=tools,
        )
        # Use result.messages, result.system_prompt, etc.
        print(result.total_tokens_saved, "tokens saved")

    Attributes:
        total_tokens: Maximum context window size in tokens.
        model:        Model name used for token counting.
        allocations:  Mapping of zone name → fraction of total window (0–1).
    """

    def __init__(self, total_tokens: int = 128_000, model: str = "gpt-4o") -> None:
        self.total_tokens = total_tokens
        self.model = model
        self.allocations: dict[str, float] = {
            "system_prompt":        0.02,
            "tool_definitions":     0.05,
            "dynamic_context":      0.45,
            "conversation_history": 0.33,
            "response_buffer":      0.15,
        }

    # ------------------------------------------------------------------
    # Allocation management
    # ------------------------------------------------------------------

    def set_allocation(self, zone: str, percentage: float) -> None:
        """Set the allocation fraction for *zone*.

        The sum of all allocations must remain <= 1.0.

        Args:
            zone:       One of the five zone names.
            percentage: New fraction in [0, 1].

        Raises:
            ValueError: If *zone* is unknown, *percentage* is out of range,
                        or the new total would exceed 1.0.
        """
        if zone not in self.allocations:
            raise ValueError(
                f"Unknown zone '{zone}'. Valid zones: {list(self.allocations)}"
            )
        if not (0.0 <= percentage <= 1.0):
            raise ValueError(f"Percentage must be in [0, 1]; got {percentage}")
        new_allocs = dict(self.allocations)
        new_allocs[zone] = percentage
        total = sum(new_allocs.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Allocation total would be {total:.3f} > 1.0 after setting "
                f"'{zone}' to {percentage}. Reduce another zone first."
            )
        self.allocations[zone] = percentage

    def get_token_budget(self, zone: str) -> int:
        """Return the token budget for *zone*.

        Args:
            zone: One of the five zone names.

        Returns:
            Integer token count.

        Raises:
            ValueError: If *zone* is unknown.
        """
        if zone not in self.allocations:
            raise ValueError(f"Unknown zone '{zone}'")
        return int(self.total_tokens * self.allocations[zone])

    def get_all_budgets(self) -> dict[str, int]:
        """Return all zone token budgets as a ``{zone: tokens}`` dict."""
        return {z: self.get_token_budget(z) for z in self.allocations}

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_zone(
        self,
        zone: str,
        content: "str | list[dict] | None",
    ) -> int:
        """Return the token count for *content* in the context of *zone*.

        Args:
            zone:    Zone name (used for routing; not validated here).
            content: String, message list, tool list, or ``None``.

        Returns:
            Integer token count.
        """
        return count_tokens(content, self.model)

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def enforce(
        self,
        system_prompt: str,
        messages: list[dict],
        dynamic_context: str = "",
        tool_definitions: Optional[list[dict]] = None,
    ) -> EnforceResult:
        """Enforce the budget on all zones.

        For each zone that exceeds its allocation the method:
        1. Logs a warning with current vs budget token counts.
        2. Applies the zone-specific compression strategy.
        3. Records the action in the audit trail.

        Compression strategies:
        - **system_prompt**: Truncate from the end; preserves the first
          instructions which are typically the most important.
        - **tool_definitions**: Keep the tools that fit; truncate descriptions
          of large tools to squeeze in more.
        - **dynamic_context**: Truncate from the end (caller should pre-rank).
        - **conversation_history**: Sliding window — drop oldest messages
          while keeping at least the most recent user turn.
        - **response_buffer**: Purely a reservation; nothing to compress.

        Args:
            system_prompt:    System prompt string.
            messages:         Full message list (may include a system message).
            dynamic_context:  RAG results, tool outputs, etc.
            tool_definitions: List of OpenAI-format tool dicts.

        Returns:
            :class:`EnforceResult` with compressed content and audit trail.
        """
        tool_definitions = tool_definitions or []
        result = EnforceResult(
            system_prompt=system_prompt,
            messages=list(messages),
            dynamic_context=dynamic_context,
            tool_definitions=list(tool_definitions),
        )
        budgets = self.get_all_budgets()

        # ---- 1. System prompt ----------------------------------------
        sp_tokens = self.measure_zone("system_prompt", system_prompt)
        sp_budget = budgets["system_prompt"]
        if sp_tokens <= sp_budget:
            result.audit["system_prompt"] = ZoneAudit(
                "system_prompt", sp_tokens, sp_budget, sp_tokens, "within_budget"
            )
        else:
            compressed = self._compress_system_prompt(system_prompt, sp_budget)
            final_tokens = self.measure_zone("system_prompt", compressed)
            result.system_prompt = compressed
            msg = (
                f"system_prompt: {sp_tokens:,} tokens exceeded budget "
                f"{sp_budget:,}; truncated to {final_tokens:,} tokens."
            )
            result.warnings.append(msg)
            result.audit["system_prompt"] = ZoneAudit(
                "system_prompt", sp_tokens, sp_budget, final_tokens, "truncated"
            )
            logger.warning(msg)

        # ---- 2. Tool definitions -------------------------------------
        td_tokens = self.measure_zone("tool_definitions", tool_definitions)
        td_budget = budgets["tool_definitions"]
        if td_tokens <= td_budget:
            result.audit["tool_definitions"] = ZoneAudit(
                "tool_definitions", td_tokens, td_budget, td_tokens, "within_budget"
            )
        else:
            compressed_tools = self._compress_tool_definitions(tool_definitions, td_budget)
            final_tokens = self.measure_zone("tool_definitions", compressed_tools)
            result.tool_definitions = compressed_tools
            msg = (
                f"tool_definitions: {td_tokens:,} tokens exceeded budget "
                f"{td_budget:,}; trimmed to {final_tokens:,} tokens "
                f"({len(compressed_tools)}/{len(tool_definitions)} tools kept)."
            )
            result.warnings.append(msg)
            result.audit["tool_definitions"] = ZoneAudit(
                "tool_definitions", td_tokens, td_budget, final_tokens, "filtered"
            )
            logger.warning(msg)

        # ---- 3. Dynamic context --------------------------------------
        dc_tokens = self.measure_zone("dynamic_context", dynamic_context)
        dc_budget = budgets["dynamic_context"]
        if dc_tokens <= dc_budget:
            result.audit["dynamic_context"] = ZoneAudit(
                "dynamic_context", dc_tokens, dc_budget, dc_tokens, "within_budget"
            )
        else:
            compressed = self._compress_dynamic_context(dynamic_context, dc_budget)
            final_tokens = self.measure_zone("dynamic_context", compressed)
            result.dynamic_context = compressed
            msg = (
                f"dynamic_context: {dc_tokens:,} tokens exceeded budget "
                f"{dc_budget:,}; truncated to {final_tokens:,} tokens."
            )
            result.warnings.append(msg)
            result.audit["dynamic_context"] = ZoneAudit(
                "dynamic_context", dc_tokens, dc_budget, final_tokens, "truncated"
            )
            logger.warning(msg)

        # ---- 4. Conversation history ---------------------------------
        history = [m for m in result.messages if m.get("role") != "system"]
        hist_tokens = self.measure_zone("conversation_history", history)
        hist_budget = budgets["conversation_history"]
        if hist_tokens <= hist_budget:
            result.audit["conversation_history"] = ZoneAudit(
                "conversation_history", hist_tokens, hist_budget, hist_tokens, "within_budget"
            )
        else:
            compressed_hist = self._compress_history(history, hist_budget)
            final_tokens = self.measure_zone("conversation_history", compressed_hist)
            system_msgs = [m for m in result.messages if m.get("role") == "system"]
            result.messages = system_msgs + compressed_hist
            msg = (
                f"conversation_history: {hist_tokens:,} tokens exceeded budget "
                f"{hist_budget:,}; sliding window applied, "
                f"{final_tokens:,} tokens kept."
            )
            result.warnings.append(msg)
            result.audit["conversation_history"] = ZoneAudit(
                "conversation_history", hist_tokens, hist_budget, final_tokens, "sliding_window"
            )
            logger.warning(msg)

        # ---- 5. Response buffer (reservation only) -------------------
        rb_budget = budgets["response_buffer"]
        result.audit["response_buffer"] = ZoneAudit(
            "response_buffer", 0, rb_budget, 0, "reserved"
        )

        return result

    # ------------------------------------------------------------------
    # Compression strategies
    # ------------------------------------------------------------------

    def _compress_system_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate *prompt* from the end, preserving the first instructions."""
        enc = _get_encoding(self.model)
        tokens = enc.encode(prompt)
        if len(tokens) <= max_tokens:
            return prompt
        return enc.decode(tokens[:max_tokens])

    def _compress_tool_definitions(
        self, tools: list[dict], max_tokens: int
    ) -> list[dict]:
        """Keep tools that fit within *max_tokens*, trimming descriptions first."""
        kept: list[dict] = []
        used = 0
        for tool in tools:
            tool_tokens = count_tokens(tool, self.model)
            if used + tool_tokens <= max_tokens:
                kept.append(tool)
                used += tool_tokens
            else:
                trimmed = self._trim_tool_description(tool, max_tokens - used)
                if trimmed is not None:
                    t = count_tokens(trimmed, self.model)
                    kept.append(trimmed)
                    used += t
                if max_tokens - used < 50:
                    break
        return kept

    def _trim_tool_description(
        self, tool: dict, max_tokens: int
    ) -> Optional[dict]:
        """Return a copy of *tool* with a shortened description, or ``None``."""
        if max_tokens < 20:
            return None
        trimmed = copy.deepcopy(tool)
        func = trimmed.get("function", trimmed)
        if "description" in func:
            enc = _get_encoding(self.model)
            desc_tokens = enc.encode(func["description"])
            # How many description tokens can we afford?
            non_desc_tokens = count_tokens(trimmed, self.model) - len(desc_tokens)
            available = max_tokens - non_desc_tokens
            if available < 10:
                return None
            func["description"] = enc.decode(desc_tokens[: max(10, available - 5)])
        return trimmed

    def _compress_dynamic_context(self, context: str, max_tokens: int) -> str:
        """Truncate *context* to fit within *max_tokens*."""
        enc = _get_encoding(self.model)
        tokens = enc.encode(context)
        if len(tokens) <= max_tokens:
            return context
        return enc.decode(tokens[:max_tokens])

    def _compress_history(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict]:
        """Apply a sliding window: drop oldest messages until the history fits.

        Always retains at least the most recent message so the conversation
        remains coherent.
        """
        if not messages:
            return messages

        kept: list[dict] = []
        used = 0
        for msg in reversed(messages):
            msg_tokens = count_tokens([msg], self.model)
            if used + msg_tokens <= max_tokens:
                kept.insert(0, msg)
                used += msg_tokens
            else:
                break

        # Always keep at least the last message
        if not kept and messages:
            kept = [messages[-1]]

        return kept

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    # Pricing per 1 000 tokens (USD)
    _PRICING: dict[str, dict[str, float]] = {
        "gpt-4o":            {"input": 0.0025,   "output": 0.010},
        "gpt-4o-mini":       {"input": 0.00015,  "output": 0.0006},
        "claude-3.5-sonnet": {"input": 0.003,    "output": 0.015},
        "claude-3-haiku":    {"input": 0.00025,  "output": 0.00125},
        "gemini-1.5-pro":    {"input": 0.0035,   "output": 0.0105},
        "gemini-1.5-flash":  {"input": 0.000075, "output": 0.0003},
    }

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
    ) -> float:
        """Return the estimated USD cost for *input_tokens* + *output_tokens*.

        Args:
            input_tokens:  Number of prompt tokens.
            output_tokens: Number of completion tokens.
            model:         Model name; defaults to ``self.model``.

        Returns:
            Estimated cost in USD, or 0.0 if the model is not in the table.
        """
        m = (model or self.model).lower()
        pricing = self._PRICING.get(m)
        if pricing is None:
            logger.debug("No pricing data for model '%s'; returning 0.0", m)
            return 0.0
        return (
            input_tokens  / 1_000 * pricing["input"]
            + output_tokens / 1_000 * pricing.get("output", 0.0)
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------


def _make_demo_messages(n: int = 30) -> list[dict]:
    """Generate *n* fake conversation turns (user + assistant)."""
    messages: list[dict] = []
    for i in range(1, n + 1):
        messages.append({
            "role": "user",
            "content": f"Turn {i}: " + "Tell me about context engineering. " * 10,
        })
        messages.append({
            "role": "assistant",
            "content": (
                f"Turn {i} reply: "
                + "Context engineering is the practice of managing the LLM "
                  "context window as a finite resource. " * 8
            ),
        })
    return messages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Context Budget Manager Demo ===\n")

    budget = ContextBudget(total_tokens=8_000)   # small window for demo

    system_prompt = (
        "You are a customer support agent. Use the knowledge base to answer. "
        "If unsure, escalate. Format: direct answer, then sources.\n"
    ) * 5   # artificially long

    messages = _make_demo_messages(n=10)

    rag_context = (
        "## Knowledge Base Article: Context Windows\n"
        "Context windows are the amount of text an LLM can process at once. "
        "They are measured in tokens. GPT-4o has 128K tokens.\n\n"
    ) * 20   # artificially large

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": (
                    "Search the knowledge base for relevant articles. "
                    "Returns the most relevant articles ranked by similarity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ] * 8   # many duplicate tools to trigger compression

    print("Budget allocations:")
    for zone, tokens in budget.get_all_budgets().items():
        print(
            f"  {zone:25s}  {tokens:6,d} tokens  "
            f"({budget.allocations[zone] * 100:.0f}%)"
        )

    print(f"\nTotal window: {budget.total_tokens:,d} tokens\n")

    print("Before enforcement:")
    print(f"  system_prompt      : {budget.measure_zone('system_prompt', system_prompt):6,d} tokens")
    print(f"  messages (history) : {budget.measure_zone('conversation_history', messages):6,d} tokens")
    print(f"  dynamic_context    : {budget.measure_zone('dynamic_context', rag_context):6,d} tokens")
    print(f"  tool_definitions   : {budget.measure_zone('tool_definitions', tools):6,d} tokens")

    result = budget.enforce(
        system_prompt,
        messages,
        dynamic_context=rag_context,
        tool_definitions=tools,
    )

    print("\nAfter enforcement:")
    for zone, audit in result.audit.items():
        print(
            f"  {zone:25s}: {audit.original_tokens:6,d} → "
            f"{audit.final_tokens:5,d}  ({audit.action_taken})"
        )

    print(f"\nTotal tokens saved : {result.total_tokens_saved:,d}")
    print(f"Total tokens used  : {result.total_tokens_used:,d}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  ⚠  {w}")

    cost_before = budget.estimate_cost(
        input_tokens=(
            budget.measure_zone("system_prompt", system_prompt)
            + budget.measure_zone("conversation_history", messages)
            + budget.measure_zone("dynamic_context", rag_context)
        ),
        output_tokens=500,
    )
    cost_after = budget.estimate_cost(
        input_tokens=result.total_tokens_used,
        output_tokens=500,
    )
    print(f"\nEstimated cost (single call): ${cost_before:.4f} → ${cost_after:.4f}")
