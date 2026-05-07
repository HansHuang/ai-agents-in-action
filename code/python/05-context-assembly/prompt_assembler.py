"""Prompt Assembler — dynamic prompt construction from templates, conditional
sections, and multi-source context injection.

Separates the **structure** of a prompt (templates) from its **content**
(runtime variables and context sources) so prompts adapt to every request
without becoming a configuration nightmare.

Key concepts:

- **Templates**: ``{variable}``-style base strings, registered by name.
- **Conditional sections**: extra instructions included only when needed.
- **Context sources**: RAG results, user profiles, tool outputs — each
  formatted by a registered formatter and sorted by priority.
- **Budget enforcement**: ``assemble_with_budget`` drops low-priority sources
  when the total token count would exceed a cap.

See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import tiktoken

from context_budget import count_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate *text* to at most *max_tokens* using the model's tokeniser."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens]) + "…"


# ---------------------------------------------------------------------------
# Template variable regex
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"\{(\w+)\}")


class _StrictFormatMap(dict):
    """dict subclass that raises MissingVariableError for unknown keys."""

    def __missing__(self, key: str) -> str:
        raise MissingVariableError(key)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MissingVariableError(KeyError):
    """Raised when a required template variable is not provided."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key

    def __str__(self) -> str:
        return f"Template variable '{{{self.key}}}' not provided"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptSection:
    """A conditional prompt section.

    The section is appended to the assembled prompt only when
    :meth:`should_include` returns ``True``.
    """

    name: str
    content: str
    condition: Callable[[dict], bool]

    def should_include(self, variables: dict) -> bool:
        """Evaluate the condition against current variables."""
        try:
            return bool(self.condition(variables))
        except Exception as exc:
            logger.warning("Section %r condition raised: %s", self.name, exc)
            return False


@dataclass
class ContextSource:
    """A registered context source with its formatter and priority."""

    name: str
    formatter: Callable[[Any], str]
    priority: int = 0
    max_tokens: int | None = None


# ---------------------------------------------------------------------------
# Built-in source formatters
# ---------------------------------------------------------------------------

def format_rag_results(documents: list[dict]) -> str:
    """Format retrieved documents with source citations.

    Args:
        documents: List of ``{"text": ..., "score": ..., "metadata": {...}}``.
    """
    if not documents:
        return "(no documents retrieved)"
    parts: list[str] = []
    for i, doc in enumerate(documents, 1):
        meta   = doc.get("metadata", {})
        source = meta.get("source", "unknown")
        score  = doc.get("score", 0.0)
        text   = doc.get("text", "")
        parts.append(f"[{i}] Source: {source} (relevance: {score:.0%})\n{text}")
    return "\n\n---\n\n".join(parts)


def format_user_profile(profile: dict) -> str:
    """Format user profile data for prompt insertion.

    Args:
        profile: Dict with optional keys: name, plan, location, preferences,
                 member_since, recent_orders, open_tickets.
    """
    lines: list[str] = []
    for key, label in [
        ("name",          "Name"),
        ("plan",          "Plan"),
        ("location",      "Location"),
        ("preferences",   "Preferences"),
        ("member_since",  "Member since"),
        ("recent_orders", "Recent orders"),
        ("open_tickets",  "Open tickets"),
    ]:
        val = profile.get(key)
        if val is not None:
            lines.append(f"{label}: {val}")
    return "\n".join(lines) if lines else "(no profile data)"


def format_tool_results(results: list[dict]) -> str:
    """Format tool execution results.

    Args:
        results: List of ``{"tool_name": ..., "success": bool, "summary": ...}``.
    """
    if not results:
        return "(no tool results)"
    parts: list[str] = []
    for r in results:
        mark    = "✓" if r.get("success") else "✗"
        name    = r.get("tool_name", "unknown")
        summary = r.get("summary", "")
        parts.append(f"{mark} {name}: {summary}")
    return "\n".join(parts)


def format_conversation_summary(summary: str) -> str:
    """Format a conversation history summary."""
    if not summary:
        return "(no conversation history)"
    return f"Previous conversation summary:\n{summary}"


def format_business_rules(rules: list[str]) -> str:
    """Format a list of business rule strings."""
    if not rules:
        return "(no business rules)"
    return "\n".join(f"- {rule}" for rule in rules)


# ---------------------------------------------------------------------------
# PromptAssembler
# ---------------------------------------------------------------------------

class PromptAssembler:
    """Assemble prompts dynamically from templates, conditional sections,
    and multiple context sources.

    Example::

        assembler = PromptAssembler()

        assembler.register_template("support", \"\"\"
        You are a support agent for {company}.
        {sections}
        {context}
        \"\"\".strip())

        assembler.register_section(
            "premium_user",
            "- Provide priority service to this premium customer.",
            condition=lambda v: v.get("plan") == "premium",
        )

        assembler.register_source_formatter(
            "rag", format_rag_results, priority=3, max_tokens=4000
        )

        prompt = assembler.assemble(
            "support",
            {"company": "Acme Corp", "plan": "premium"},
            context_sources={"rag": my_rag_docs},
        )
    """

    def __init__(self) -> None:
        self.base_templates: dict[str, str] = {}
        self.sections: dict[str, PromptSection] = {}
        self.source_formatters: dict[str, ContextSource] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_template(self, name: str, template: str) -> None:
        """Register a base template with ``{placeholder}`` variables.

        Args:
            name:     Unique template identifier.
            template: Template string with ``{variable}`` placeholders.
        """
        self.base_templates[name] = template
        logger.debug("Registered template %r (%d chars)", name, len(template))

    def register_section(
        self,
        name: str,
        content: str,
        condition: Callable[[dict], bool],
    ) -> None:
        """Register a conditional prompt section.

        Args:
            name:      Unique section identifier.
            content:   Section text to inject when the condition is True.
            condition: Callable that receives the variables dict and returns
                       ``True`` when this section should be included.
        """
        self.sections[name] = PromptSection(
            name=name, content=content, condition=condition
        )
        logger.debug("Registered section %r", name)

    def register_source_formatter(
        self,
        name: str,
        formatter: Callable[[Any], str],
        priority: int = 0,
        max_tokens: int | None = None,
    ) -> None:
        """Register a context source formatter.

        Args:
            name:       Source identifier (e.g. ``"rag"``, ``"user_profile"``).
            formatter:  Callable that converts raw data to a prompt-ready string.
            priority:   Higher priority sources are included first and survive
                        budget cuts longest.
            max_tokens: Truncate this source to this many tokens before injection.
        """
        self.source_formatters[name] = ContextSource(
            name=name,
            formatter=formatter,
            priority=priority,
            max_tokens=max_tokens,
        )
        logger.debug("Registered source formatter %r (priority=%d)", name, priority)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        template_name: str,
        variables: dict,
        context_sources: dict[str, Any] | None = None,
    ) -> str:
        """Assemble a complete prompt.

        Steps:

        1. Look up the base template.
        2. Evaluate all conditional sections; collect those that match.
        3. Format all context sources using their registered formatters.
        4. Sort context sources by priority (highest first).
        5. Build the ``{context}`` block from sorted sources.
        6. Build the ``{sections}`` block from active sections.
        7. Fill the template with all variables.
        8. Log: template used, sections included, context sources used, token count.

        Args:
            template_name:   Name of the registered base template.
            variables:       Runtime variables for template filling.
            context_sources: Mapping of source name → raw data.

        Returns:
            Fully assembled prompt string.

        Raises:
            KeyError:              If *template_name* is not registered.
            MissingVariableError:  If a ``{variable}`` in the template has no value.
        """
        if template_name not in self.base_templates:
            raise KeyError(f"Template {template_name!r} not registered")

        template = self.base_templates[template_name]

        # 2. Evaluate conditional sections
        active_sections = [
            s for s in self.sections.values() if s.should_include(variables)
        ]

        # 3 + 4. Format and sort context sources
        formatted: list[tuple[int, str, str]] = []   # (priority, name, text)
        for src_name, data in (context_sources or {}).items():
            if src_name not in self.source_formatters:
                logger.warning("No formatter registered for source %r — skipping", src_name)
                continue
            source = self.source_formatters[src_name]
            text   = source.formatter(data)
            if source.max_tokens and count_tokens(text) > source.max_tokens:
                text = _truncate_to_tokens(text, source.max_tokens)
            formatted.append((source.priority, src_name, text))

        formatted.sort(key=lambda x: x[0], reverse=True)   # highest priority first

        # 5. Build context block
        context_block = "\n\n".join(
            f"## {name}\n{text}" for _, name, text in formatted
        )

        # 6. Build sections block
        sections_block = "\n\n".join(s.content for s in active_sections)

        # 7. Fill template
        fill_vars: dict[str, Any] = {
            **variables,
            "context":  context_block,
            "sections": sections_block,
        }
        result = template.format_map(_StrictFormatMap(fill_vars))

        # Append sections and context that have no placeholder in the template
        if "{sections}" not in template and sections_block:
            result = result.rstrip() + "\n\n" + sections_block
        if "{context}" not in template and context_block:
            result = result.rstrip() + "\n\n" + context_block

        # 8. Log
        token_count = count_tokens(result)
        logger.info(
            "Assembled prompt | template=%s | sections=[%s] | sources=[%s] | tokens=%d",
            template_name,
            ", ".join(s.name for s in active_sections),
            ", ".join(n for _, n, _ in formatted),
            token_count,
        )

        return result

    def assemble_with_budget(
        self,
        template_name: str,
        variables: dict,
        context_sources: dict[str, Any] | None = None,
        max_tokens: int = 100_000,
    ) -> str:
        """Assemble a prompt while enforcing a total token budget.

        Low-priority context sources are dropped first when the assembled
        prompt would exceed *max_tokens*.

        Args:
            template_name:   Name of the registered base template.
            variables:       Runtime variables for template filling.
            context_sources: Mapping of source name → raw data.
            max_tokens:      Hard token cap for the final prompt.

        Returns:
            Assembled prompt string within the token budget (best effort).
        """
        result = self.assemble(template_name, variables, context_sources)
        if count_tokens(result) <= max_tokens:
            return result

        if not context_sources:
            return result  # nothing to drop

        # Sort sources ascending by priority (drop lowest first)
        def _priority(name: str) -> int:
            src = self.source_formatters.get(name)
            return src.priority if src else 0

        drop_order = sorted(context_sources.keys(), key=_priority)
        remaining  = dict(context_sources)

        for name in drop_order:
            del remaining[name]
            logger.info(
                "Budget (%d tokens) exceeded — dropping source %r", max_tokens, name
            )
            result = self.assemble(template_name, variables, remaining)
            if count_tokens(result) <= max_tokens:
                return result

        return result  # best effort: no sources remain

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_available_variables(self, template_name: str) -> list[str]:
        """Parse a template and return all ``{variable}`` names in order.

        Args:
            template_name: Name of the registered base template.

        Returns:
            Unique variable names in order of first appearance.

        Raises:
            KeyError: If *template_name* is not registered.
        """
        if template_name not in self.base_templates:
            raise KeyError(f"Template {template_name!r} not registered")
        seen: set[str] = set()
        result: list[str] = []
        for m in _VAR_RE.finditer(self.base_templates[template_name]):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Support agent with 3 templates, 5 sections, 4 sources, 4 scenarios."""
    assembler = PromptAssembler()

    # ── Templates ──────────────────────────────────────────────────────────
    assembler.register_template("general", (
        "You are a customer support agent for {company}.\n\n"
        "Guidelines:\n"
        "- Be helpful, clear, and concise.\n"
        "- Escalate to a human agent if you cannot resolve the issue.\n"
        "{sections}\n\n"
        "{context}"
    ).strip())

    assembler.register_template("billing", (
        "You are a billing support specialist for {company}.\n\n"
        "Responsibilities:\n"
        "- Handle payment inquiries, invoice questions, and account charges.\n"
        "- Process refunds according to the refund policy.\n"
        "- Explain subscription tiers and pricing clearly.\n"
        "{sections}\n\n"
        "{context}"
    ).strip())

    assembler.register_template("technical", (
        "You are a technical support engineer for {company}.\n\n"
        "Responsibilities:\n"
        "- Debug integration issues and API errors.\n"
        "- Guide customers through configuration and setup.\n"
        "- Escalate critical bugs to the engineering team.\n"
        "{sections}\n\n"
        "{context}"
    ).strip())

    # ── Conditional sections ───────────────────────────────────────────────
    assembler.register_section(
        "premium_user",
        "Additional instructions (premium customer):\n"
        "- Provide priority service with SLA: 2-hour response.\n"
        "- Offer dedicated account manager contact.",
        condition=lambda v: v.get("plan") == "premium",
    )
    assembler.register_section(
        "frustrated_user",
        "Additional instructions (frustrated customer):\n"
        "- Acknowledge the customer's frustration before solving the issue.\n"
        "- Offer a goodwill discount code if appropriate.",
        condition=lambda v: v.get("sentiment") == "frustrated",
    )
    assembler.register_section(
        "international",
        "Additional instructions (international customer):\n"
        "- Provide international shipping times where relevant.\n"
        "- Note any country-specific policies.",
        condition=lambda v: v.get("country", "US") not in ("US", "CA"),
    )
    assembler.register_section(
        "multi_turn",
        "Additional instructions (multi-turn conversation):\n"
        "- Maintain continuity with previous context.\n"
        "- Do not repeat information already provided.",
        condition=lambda v: v.get("conversation_turns", 0) > 1,
    )
    assembler.register_section(
        "gdpr_required",
        "Additional instructions (EU customer — GDPR):\n"
        "- Include data processing notice.\n"
        "- Inform the customer of their right to data deletion.",
        condition=lambda v: v.get("country", "US") in ("DE", "FR", "ES", "IT", "AT", "NL"),
    )

    # ── Context source formatters ──────────────────────────────────────────
    assembler.register_source_formatter("rag",                  format_rag_results,          priority=3, max_tokens=4_000)
    assembler.register_source_formatter("user_profile",         format_user_profile,         priority=2, max_tokens=500)
    assembler.register_source_formatter("tool_results",         format_tool_results,         priority=2)
    assembler.register_source_formatter("conversation_summary", format_conversation_summary, priority=1, max_tokens=1_000)

    # ── Sample data ────────────────────────────────────────────────────────
    rag_docs = [
        {
            "text":     "Refund policy: full refund within 30 days; partial refund within 60 days.",
            "score":    0.92,
            "metadata": {"source": "refund-policy.md"},
        },
        {
            "text":     "Premium plans include priority support with a 2-hour SLA.",
            "score":    0.87,
            "metadata": {"source": "pricing.md"},
        },
    ]

    divider = "\n" + "─" * 60

    # ── Scenario 1: Free user, billing question ────────────────────────────
    print(f"{divider}\nScenario 1: Free user, billing question")
    p1 = assembler.assemble(
        "billing",
        {"company": "Acme Corp", "plan": "free", "country": "US"},
        {"rag": rag_docs},
    )
    print(p1)
    print(f"\n[{count_tokens(p1)} tokens]")

    # ── Scenario 2: Premium user, billing question ─────────────────────────
    print(f"{divider}\nScenario 2: Premium user, billing question")
    p2 = assembler.assemble(
        "billing",
        {"company": "Acme Corp", "plan": "premium", "country": "US"},
        {
            "rag":          rag_docs,
            "user_profile": {"name": "Alice", "plan": "Premium", "location": "San Francisco"},
        },
    )
    print(p2)
    print(f"\n[{count_tokens(p2)} tokens]")

    # ── Scenario 3: German user ────────────────────────────────────────────
    print(f"{divider}\nScenario 3: German user (international + GDPR)")
    p3 = assembler.assemble(
        "general",
        {"company": "Acme Corp", "plan": "free", "country": "DE"},
        {"rag": rag_docs},
    )
    print(p3)
    print(f"\n[{count_tokens(p3)} tokens]")

    # ── Scenario 4: Frustrated premium German user with history ───────────
    print(f"{divider}\nScenario 4: Frustrated premium user with history (everything)")
    p4 = assembler.assemble(
        "billing",
        {
            "company":            "Acme Corp",
            "plan":               "premium",
            "country":            "DE",
            "sentiment":          "frustrated",
            "conversation_turns": 3,
        },
        {
            "rag":                  rag_docs,
            "user_profile":         {"name": "Klaus", "plan": "Premium", "location": "Berlin"},
            "conversation_summary": (
                "Customer has been struggling with a billing discrepancy for 2 weeks. "
                "Previous agents escalated but no resolution was reached."
            ),
        },
    )
    print(p4)
    print(f"\n[{count_tokens(p4)} tokens]")

    # ── get_available_variables ────────────────────────────────────────────
    print(f"{divider}\nTemplate variables in 'billing':")
    print(assembler.get_available_variables("billing"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _demo()
