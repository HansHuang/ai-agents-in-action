"""Context Assembler — dynamic context assembly from multiple sources.

Assembles the LLM context string from:
  1. RAG retrieved documents
  2. Tool execution results
  3. User profile / preferences
  4. Conversation summary
  5. Template variables

Integrates with :class:`~context_budget.ContextBudget` for budget enforcement
and :class:`~context_optimizer.ContextOptimizer` for structure optimisation.

See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from string import Template
from typing import Optional

import tiktoken

from context_budget import ContextBudget, EnforceResult, count_tokens
from context_optimizer import ContextOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ContextConfig:
    """Declarative configuration for context assembly.

    Example::

        config = ContextConfig(
            template="You are a $role. Use the documents to answer.",
            template_vars={"role": "support_agent"},
            include_sources=["rag", "profile"],
            max_tokens_per_source={"rag": 50_000},
            format="markdown",
        )
    """

    template: str
    template_vars: dict[str, str] = field(default_factory=dict)
    include_sources: list[str] = field(
        default_factory=lambda: ["rag", "tools", "profile", "summary"]
    )
    max_tokens_per_source: dict[str, int] = field(default_factory=dict)
    format: str = "markdown"   # "markdown" | "plain" | "json"
    priority_order: list[str] = field(
        default_factory=lambda: ["rag", "tools", "profile", "summary"]
    )


# ---------------------------------------------------------------------------
# Assembly result
# ---------------------------------------------------------------------------


@dataclass
class AssemblyResult:
    """Assembled context with per-source token accounting.

    Attributes:
        context:          Final assembled context string.
        token_breakdown:  Token count per source section.
        enforce_result:   :class:`~context_budget.EnforceResult` if enforcement ran.
        sources_included: Source names that made it into the context.
        sources_excluded: Source names that were dropped due to budget.
    """

    context: str
    token_breakdown: dict[str, int] = field(default_factory=dict)
    enforce_result: Optional[EnforceResult] = None
    sources_included: list[str] = field(default_factory=list)
    sources_excluded: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Sum of tokens across all included sections."""
        return sum(self.token_breakdown.values())


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Assemble context for LLM calls from multiple sources.

    Template-driven, budget-aware, and optimizable.

    Example::

        budget    = ContextBudget(total_tokens=128_000)
        assembler = ContextAssembler(budget)

        result = assembler.assemble(
            template="You are a $role. Answer using the documents.",
            variables={"role": "support agent"},
            retrieved_docs=rag_docs,
            user_profile={"name": "Alice", "plan": "Pro"},
            query="How do I cancel my subscription?",
        )
        # result.context → final assembled string
        # result.token_breakdown → {"template": 45, "rag": 3200, "profile": 60}
        # result.sources_excluded → [] or ["summary", ...]
    """

    _SECTION_HEADERS: dict[str, str] = {
        "rag":     "## Retrieved Documents\n",
        "tools":   "## Tool Results\n",
        "profile": "## User Profile\n",
        "summary": "## Conversation Summary\n",
    }

    def __init__(
        self,
        budget: ContextBudget,
        optimizer: Optional[ContextOptimizer] = None,
        model: str = "gpt-4o",
    ) -> None:
        self.budget    = budget
        self.optimizer = optimizer or ContextOptimizer(model=model)
        self.model     = model

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def assemble(
        self,
        template: str,
        variables: Optional[dict[str, str]] = None,
        retrieved_docs: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
        user_profile: Optional[dict] = None,
        conversation_summary: Optional[str] = None,
        query: str = "",
        optimize: bool = True,
    ) -> AssemblyResult:
        """Assemble the dynamic context for an LLM call.

        Sources are assembled in priority order:

        1. Retrieved documents (RAG)
        2. Tool execution results
        3. User profile / preferences
        4. Conversation summary
        5. Template variables

        Each source section is added while there is remaining budget.  When a
        section is too large to fit in full it is truncated to what fits.
        If there is no budget for a source it is excluded entirely.

        When *optimize* is ``True`` and a *query* is provided the RAG section
        is additionally structured and position-optimised before the full
        context is assembled.

        Args:
            template:             Python :class:`string.Template` template.
            variables:            Mapping for template substitution.
            retrieved_docs:       RAG result dicts with ``text``/``metadata``.
            tool_results:         Tool output dicts with ``tool``/``result``.
            user_profile:         Free-form user profile dict.
            conversation_summary: Plain-text summary of prior conversation turns.
            query:                User's question; used for optimisation.
            optimize:             Whether to apply :class:`ContextOptimizer`.

        Returns:
            :class:`AssemblyResult`.
        """
        variables = variables or {}

        # Render template
        try:
            rendered_template = Template(template).safe_substitute(variables)
        except Exception as exc:
            logger.warning("Template render failed: %s; using raw template", exc)
            rendered_template = template

        # Build source content map
        sections: dict[str, str] = {}

        if retrieved_docs:
            if optimize and query:
                structured = self.optimizer.structure_for_retrieval(retrieved_docs)
            else:
                structured = "\n\n".join(d.get("text", "") for d in retrieved_docs)
            sections["rag"] = structured

        if tool_results:
            tool_lines: list[str] = []
            for tr in tool_results:
                tool_name = tr.get("tool", "unknown")
                result = tr.get("result", "")
                if isinstance(result, (dict, list)):
                    result = json.dumps(result, indent=2)
                tool_lines.append(f"**{tool_name}**:\n{result}")
            sections["tools"] = "\n\n".join(tool_lines)

        if user_profile:
            lines = [f"- **{k}**: {v}" for k, v in user_profile.items()]
            sections["profile"] = "\n".join(lines)

        if conversation_summary:
            sections["summary"] = conversation_summary

        # Assemble in priority order within the dynamic_context budget
        dc_budget = self.budget.get_token_budget("dynamic_context")
        context_parts: list[str] = [rendered_template]
        token_breakdown: dict[str, int] = {
            "template": count_tokens(rendered_template, self.model)
        }
        sources_included: list[str] = []
        sources_excluded: list[str] = []

        for source in ["rag", "tools", "profile", "summary"]:
            if source not in sections:
                continue

            content = sections[source]
            header  = self._SECTION_HEADERS.get(source, f"## {source.title()}\n")
            section_text   = header + content
            section_tokens = count_tokens(section_text, self.model)
            used_so_far    = sum(token_breakdown.values())

            if used_so_far + section_tokens <= dc_budget:
                context_parts.append(section_text)
                token_breakdown[source] = section_tokens
                sources_included.append(source)
            else:
                # Try to fit a truncated version
                available = (
                    dc_budget
                    - used_so_far
                    - count_tokens(header, self.model)
                )
                if available > 100:
                    truncated    = self._truncate_to_tokens(content, available)
                    section_text = header + truncated
                    tok          = count_tokens(section_text, self.model)
                    context_parts.append(section_text)
                    token_breakdown[source] = tok
                    sources_included.append(source)
                    logger.warning(
                        "Source '%s' truncated: %d → %d tokens (budget %d)",
                        source, section_tokens, tok, dc_budget,
                    )
                else:
                    sources_excluded.append(source)
                    logger.warning(
                        "Source '%s' excluded: no budget remaining (%d / %d tokens used)",
                        source, used_so_far, dc_budget,
                    )

        final_context = "\n\n".join(context_parts)

        # Position-optimise the query-relevant RAG content
        if optimize and query and "rag" in sections:
            final_context = self.optimizer.optimize_for_query(final_context, query)

        return AssemblyResult(
            context=final_context,
            token_breakdown=token_breakdown,
            sources_included=sources_included,
            sources_excluded=sources_excluded,
        )

    def assemble_from_config(
        self,
        config: ContextConfig,
        retrieved_docs: Optional[list[dict]] = None,
        tool_results: Optional[list[dict]] = None,
        user_profile: Optional[dict] = None,
        conversation_summary: Optional[str] = None,
        query: str = "",
    ) -> AssemblyResult:
        """Assemble context from a declarative :class:`ContextConfig`.

        The config's ``include_sources`` gate and ``max_tokens_per_source``
        limits are applied before delegating to :meth:`assemble`.

        Args:
            config:               Assembly configuration.
            retrieved_docs:       RAG result dicts.
            tool_results:         Tool output dicts.
            user_profile:         Profile dict.
            conversation_summary: Summary string.
            query:                User's question.

        Returns:
            :class:`AssemblyResult`.
        """
        included = set(config.include_sources or ["rag", "tools", "profile", "summary"])

        filtered_docs    = retrieved_docs      if "rag"     in included else None
        filtered_tools   = tool_results        if "tools"   in included else None
        filtered_profile = user_profile        if "profile" in included else None
        filtered_summary = conversation_summary if "summary" in included else None

        # Apply per-source token limits
        for source, max_tok in (config.max_tokens_per_source or {}).items():
            if source == "rag" and filtered_docs is not None:
                filtered_docs = self._clip_docs_to_budget(filtered_docs, max_tok)

        return self.assemble(
            template=config.template,
            variables=config.template_vars or {},
            retrieved_docs=filtered_docs,
            tool_results=filtered_tools,
            user_profile=filtered_profile,
            conversation_summary=filtered_summary,
            query=query,
            optimize=(config.format == "markdown"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Return *text* truncated to at most *max_tokens* tokens."""
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])

    def _clip_docs_to_budget(
        self, docs: list[dict], max_tokens: int
    ) -> list[dict]:
        """Return the subset of *docs* that fits within *max_tokens*."""
        kept: list[dict] = []
        used = 0
        for doc in docs:
            t = count_tokens(doc.get("text", ""), self.model)
            if used + t <= max_tokens:
                kept.append(doc)
                used += t
            else:
                break
        return kept


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _build_demo_rag_docs() -> list[dict]:
    return [
        {
            "text": (
                "## Account Management\n"
                "To update your account email, navigate to Settings → Account → Email "
                "Address.  A verification email will be sent to your new address."
            ),
            "metadata": {"source": "help-center/account-management.md", "score": 0.91},
        },
        {
            "text": (
                "## Billing & Subscriptions\n"
                "Invoices are generated on the 1st of each month.  Download them from "
                "Settings → Billing → Invoice History.  Invoices are retained for 7 years."
            ),
            "metadata": {"source": "help-center/billing.md", "score": 0.88},
        },
        {
            "text": (
                "## Cancellation Policy\n"
                "You may cancel your subscription at any time.  Cancellation takes effect "
                "at the end of the current billing period.  No refunds for partial months."
            ),
            "metadata": {"source": "help-center/cancellation.md", "score": 0.85},
        },
        {
            "text": (
                "## Data Export\n"
                "Export all your data as a ZIP archive from Settings → Privacy → Export "
                "Data.  The export includes messages, files, and account metadata."
            ),
            "metadata": {"source": "help-center/data-export.md", "score": 0.80},
        },
        {
            "text": (
                "## Two-Factor Authentication\n"
                "Enable 2FA in Settings → Security.  We support TOTP apps (Google "
                "Authenticator, Authy) and hardware security keys (FIDO2/WebAuthn)."
            ),
            "metadata": {"source": "help-center/security.md", "score": 0.78},
        },
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Context Assembler Demo ===\n")

    budget    = ContextBudget(total_tokens=16_000)
    assembler = ContextAssembler(budget)

    rag_docs     = _build_demo_rag_docs()
    user_profile = {
        "name":         "Alice Johnson",
        "plan":         "Pro (annual)",
        "member_since": "2022-03-15",
        "open_tickets": "2",
    }
    prior_summary = (
        "Alice asked how to change her password (resolved: Settings → Security) "
        "and enquired about invoice downloads (resolved: Settings → Billing)."
    )
    tool_results = [
        {
            "tool": "lookup_account",
            "result": {
                "user_id":      "usr_alice_123",
                "subscription": "pro_annual",
                "next_invoice": "2026-06-01",
                "2fa_enabled":  True,
            },
        }
    ]

    query = "How do I cancel my subscription and download my last invoice?"

    result = assembler.assemble(
        template=(
            "You are a $role support agent. "
            "Answer the user's question using only the provided documents.\n"
        ),
        variables={"role": "customer"},
        retrieved_docs=rag_docs,
        tool_results=tool_results,
        user_profile=user_profile,
        conversation_summary=prior_summary,
        query=query,
    )

    print(f"Sources included : {result.sources_included}")
    print(f"Sources excluded : {result.sources_excluded}")
    print(f"Total tokens     : {result.total_tokens:,d}")
    print("\nToken breakdown by source:")
    for src, tok in result.token_breakdown.items():
        print(f"  {src:10s}: {tok:5,d} tokens")

    print("\n--- Assembled context (first 1 000 chars) ---")
    print(result.context[:1_000])

    print("\n\n--- assemble_from_config() demo ---")
    config = ContextConfig(
        template="You are a $role. Answer based on the documents only.",
        template_vars={"role": "support agent"},
        include_sources=["rag", "profile"],
        max_tokens_per_source={"rag": 2_000},
        format="markdown",
    )
    cfg_result = assembler.assemble_from_config(
        config,
        retrieved_docs=rag_docs,
        user_profile=user_profile,
        query=query,
    )
    print(f"Config result total tokens : {cfg_result.total_tokens:,d}")
    print(f"Sources included           : {cfg_result.sources_included}")
