"""Token Cost Calculator — cost modeling and optimization for LLM context.

Calculates and projects token costs across models and providers, identifies
waste in system prompts and conversation history, and suggests concrete
optimisations with projected savings.

See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from context_budget import count_tokens, _get_encoding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 M tokens, May 2026)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "gpt-4o":                 {"input": 2.50,   "output": 10.00},
        "gpt-4o-mini":            {"input": 0.15,   "output": 0.60},
        "text-embedding-3-small": {"input": 0.02},
        "text-embedding-3-large": {"input": 0.13},
    },
    "anthropic": {
        "claude-3.5-sonnet": {"input": 3.00,  "output": 15.00},
        "claude-3-haiku":    {"input": 0.25,  "output": 1.25},
    },
    "google": {
        "gemini-1.5-pro":   {"input": 3.50,   "output": 10.50},
        "gemini-1.5-flash": {"input": 0.075,  "output": 0.30},
    },
}

# Flat lookup for convenience
_FLAT: dict[str, dict[str, float]] = {
    model: pricing
    for provider in PRICING.values()
    for model, pricing in provider.items()
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CostEstimate:
    """Cost estimate for a single API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ProjectionResult:
    """Cost projection over daily / monthly / annual horizons."""

    model: str
    calls_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    cost_per_call: float
    daily: float
    monthly: float
    annual: float


@dataclass
class AuditResult:
    """Audit of a message context for cost optimisation opportunities.

    Attributes:
        total_tokens:             Total tokens in the audited context.
        by_role:                  Token counts keyed by message role.
        system_prompt_efficiency: Fraction of system prompt that is constraint
                                  text (0 = all padding, 1 = pure constraints).
        wasted_tokens:            Estimated tokens that could be removed without
                                  meaningful impact.
        suggestions:              Actionable optimisation suggestions.
        projected_savings:        Token and USD savings at the given call volume.
    """

    total_tokens: int
    by_role: dict[str, int]
    system_prompt_efficiency: float
    wasted_tokens: int
    suggestions: list[str]
    projected_savings: dict[str, float]


# ---------------------------------------------------------------------------
# TokenCostCalculator
# ---------------------------------------------------------------------------


class TokenCostCalculator:
    """Calculate and optimise token costs across models and providers.

    Example::

        calc = TokenCostCalculator()
        cost = calc.calculate_call_cost("gpt-4o", 50_000, 5_000)
        print(f"${cost:.4f}")

        print(calc.compare_models(50_000, 5_000, calls_per_day=1_000))
    """

    PRICING = PRICING  # expose for inspection and extension

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def calculate_call_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """Calculate the USD cost for a single API call.

        Args:
            model:         Model name (e.g. ``"gpt-4o"``).
            input_tokens:  Number of prompt / input tokens.
            output_tokens: Number of completion / output tokens.

        Returns:
            Cost in USD.

        Raises:
            KeyError: If the model is not in the pricing table.
        """
        pricing = self._get_pricing(model)
        cost = input_tokens / 1_000_000 * pricing["input"]
        if output_tokens and "output" in pricing:
            cost += output_tokens / 1_000_000 * pricing["output"]
        return cost

    def calculate_daily_cost(
        self,
        model: str,
        calls_per_day: int,
        avg_input_tokens: int,
        avg_output_tokens: int = 0,
    ) -> ProjectionResult:
        """Project daily, monthly, and annual costs.

        Args:
            model:             Model name.
            calls_per_day:     Number of API calls per day.
            avg_input_tokens:  Average input tokens per call.
            avg_output_tokens: Average output tokens per call.

        Returns:
            :class:`ProjectionResult`.
        """
        cost_per_call = self.calculate_call_cost(
            model, avg_input_tokens, avg_output_tokens
        )
        daily   = cost_per_call * calls_per_day
        monthly = daily * 30
        annual  = daily * 365
        return ProjectionResult(
            model=model,
            calls_per_day=calls_per_day,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            cost_per_call=cost_per_call,
            daily=daily,
            monthly=monthly,
            annual=annual,
        )

    def compare_models(
        self,
        input_tokens: int,
        output_tokens: int,
        calls_per_day: int,
    ) -> str:
        """Return a formatted comparison table of costs across all models.

        Args:
            input_tokens:  Average input tokens per call.
            output_tokens: Average output tokens per call.
            calls_per_day: Number of API calls per day.

        Returns:
            Multi-line string table.

        Example output::

            Model                      |       Daily |     Monthly |      Annual
            ---------------------------|-------------|-------------|-------------
            gpt-4o                     |      $24.00 |     $720.00 |   $8,640.00
            gpt-4o-mini                |       $1.44 |      $43.20 |     $518.40
        """
        c_model   = 28
        c_daily   = 12
        c_monthly = 12
        c_annual  = 13

        header = (
            f"{'Model':<{c_model}} | "
            f"{'Daily':>{c_daily}} | "
            f"{'Monthly':>{c_monthly}} | "
            f"{'Annual':>{c_annual}}"
        )
        sep = "-" * (c_model + c_daily + c_monthly + c_annual + 9)
        rows: list[str] = [header, sep]

        for model in _FLAT:
            try:
                proj = self.calculate_daily_cost(
                    model, calls_per_day, input_tokens, output_tokens
                )
            except KeyError:
                continue
            rows.append(
                f"{model:<{c_model}} | "
                f"${proj.daily:>{c_daily - 1},.2f} | "
                f"${proj.monthly:>{c_monthly - 1},.2f} | "
                f"${proj.annual:>{c_annual - 1},.2f}"
            )
        return "\n".join(rows)

    def optimize_system_prompt(
        self,
        prompt: str,
        target_reduction_pct: float = 0.50,
    ) -> str:
        """Return a condensed version of *prompt* using rule-based compression.

        Rules applied in order:

        1. Collapse runs of blank lines to at most one.
        2. Remove duplicate lines (case-insensitive).
        3. Strip common verbose filler phrases.
        4. Truncate to the target token count if still over budget.

        No external LLM call is made.

        Args:
            prompt:                System prompt text.
            target_reduction_pct:  Desired reduction as a fraction in [0, 1].
                                   0.50 means "reduce to 50% of original size".

        Returns:
            Compressed prompt string.
        """
        target_tokens = int(count_tokens(prompt) * (1.0 - target_reduction_pct))

        # Rule 1: collapse blank lines
        lines = prompt.splitlines()
        collapsed: list[str] = []
        prev_blank = False
        for line in lines:
            blank = line.strip() == ""
            if blank and prev_blank:
                continue
            prev_blank = blank
            collapsed.append(line)

        # Rule 2: remove duplicate lines
        seen: set[str] = set()
        deduped: list[str] = []
        for line in collapsed:
            key = line.strip().lower()
            if key and key in seen:
                continue
            seen.add(key)
            deduped.append(line)

        # Rule 3: strip filler phrases
        _FILLER = [
            r"\bplease note that\b",
            r"\bit is important to remember that\b",
            r"\bas an AI language model\b",
            r"\bfeel free to\b",
            r"\bdon't hesitate to\b",
            r"\bI hope this helps\b",
            r"\bof course\b",
            r"\bsure\b[,!]\s*",
            r"\babsolutely\b[,!]\s*",
            r"\bcertainly\b[,!]\s*",
        ]
        result_lines: list[str] = []
        for line in deduped:
            for pat in _FILLER:
                line = re.sub(pat, "", line, flags=re.IGNORECASE)
            result_lines.append(line.rstrip())

        result = "\n".join(result_lines).strip()

        # Rule 4: truncate if still over target
        if count_tokens(result) > target_tokens:
            enc = _get_encoding()
            tokens = enc.encode(result)
            result = enc.decode(tokens[:target_tokens])

        return result

    def audit_context(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        calls_per_day: int = 1_000,
        model: Optional[str] = None,
    ) -> AuditResult:
        """Audit a message context for cost optimisation opportunities.

        Analyses token usage per message role and produces actionable
        suggestions with projected savings.

        Args:
            messages:      List of message dicts (OpenAI format).
            tools:         List of tool definition dicts.
            calls_per_day: Used for projection calculations.
            model:         Model for token counting and pricing
                           (defaults to ``self.model``).

        Returns:
            :class:`AuditResult`.
        """
        m = model or self.model
        by_role: dict[str, int] = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            by_role[role] = by_role.get(role, 0) + count_tokens(
                msg.get("content", ""), m
            )

        tool_tokens = count_tokens(tools or [], m)
        if tool_tokens:
            by_role["tools"] = tool_tokens

        total = sum(by_role.values())

        # --- System prompt efficiency ---
        sys_text = next(
            (msg.get("content", "") for msg in messages if msg.get("role") == "system"),
            "",
        )
        sys_tokens = by_role.get("system", 0)
        # Heuristic: short non-comment lines are constraint text; everything else
        # is likely padding (personality, examples, etc.)
        constraint_line_words = sum(
            len(ln.split())
            for ln in sys_text.splitlines()
            if 5 < len(ln.strip()) < 120 and not ln.strip().startswith("#")
        )
        est_constraint_tokens = constraint_line_words * 1.3
        efficiency = min(1.0, max(0.0, est_constraint_tokens / max(sys_tokens, 1)))

        # --- Identify waste sources ---
        suggestions: list[str] = []
        wasted = 0

        # Examples in system prompt
        example_count = len(
            re.findall(
                r"(Example|e\.g\.|For example|For instance)",
                sys_text,
                re.IGNORECASE,
            )
        )
        if example_count > 2:
            est = example_count * 200
            wasted += est
            suggestions.append(
                f"System prompt: Move {example_count} examples to dynamic context "
                f"(save ~{est:,} tokens per call)."
            )

        # Old assistant turns
        assistant_msgs = [msg for msg in messages if msg.get("role") == "assistant"]
        if len(assistant_msgs) > 6:
            old_turns = assistant_msgs[:-3]
            old_tokens = sum(
                count_tokens(msg.get("content", ""), m) for msg in old_turns
            )
            if old_tokens > 2_000:
                est = int(old_tokens * 0.70)
                wasted += est
                suggestions.append(
                    f"Messages: Summarise turns 1–{len(old_turns)} "
                    f"(save ~{est:,} tokens)."
                )

        # Verbose tool descriptions
        if tools:
            for tool in tools:
                func = tool.get("function", {})
                desc = func.get("description", "")
                word_count = len(desc.split())
                if word_count > 25:
                    est = max(0, word_count - 15)
                    suggestions.append(
                        f"Tool '{func.get('name', '?')}': Shorten description "
                        f"({word_count} words → 15; save ~{est} tokens)."
                    )

        if not suggestions:
            suggestions.append("No significant optimisation opportunities detected.")

        # --- Projected savings ---
        wasted_daily = wasted * calls_per_day
        try:
            pricing = self._get_pricing(m)
            usd_per_day = wasted_daily / 1_000_000 * pricing["input"]
        except KeyError:
            usd_per_day = 0.0

        projected_savings: dict[str, float] = {
            "tokens_per_call": wasted,
            "tokens_daily":    wasted_daily,
            "tokens_monthly":  wasted_daily * 30,
            "tokens_annual":   wasted_daily * 365,
            "usd_daily":       usd_per_day,
            "usd_monthly":     usd_per_day * 30,
            "usd_annual":      usd_per_day * 365,
        }

        return AuditResult(
            total_tokens=total,
            by_role=by_role,
            system_prompt_efficiency=efficiency,
            wasted_tokens=wasted,
            suggestions=suggestions,
            projected_savings=projected_savings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pricing(self, model: str) -> dict[str, float]:
        """Return the pricing dict for *model*.

        Raises:
            KeyError: If the model is not in the pricing table.
        """
        pricing = _FLAT.get(model.lower())
        if pricing is None:
            raise KeyError(
                f"Unknown model '{model}'. Available: {sorted(_FLAT)}"
            )
        return pricing


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Token Cost Calculator Demo ===\n")

    calc = TokenCostCalculator()

    # 1. Single call cost
    cost = calc.calculate_call_cost("gpt-4o", input_tokens=50_000, output_tokens=5_000)
    expected = (50_000 / 1_000_000 * 2.50) + (5_000 / 1_000_000 * 10.00)
    print(f"Single call (GPT-4o, 50K in / 5K out):")
    print(f"  Calculated : ${cost:.4f}")
    print(f"  Expected   : ${expected:.4f}\n")

    # 2. Verbose system prompt
    verbose_prompt = (
        "You are a helpful, friendly, and knowledgeable customer support assistant.\n"
        "You always greet the user warmly and ask how you can help them today.\n"
        "Please note that you should always be polite and professional.\n"
        "It is important to remember that you represent our company brand.\n"
        "Feel free to use examples to clarify your answers.\n\n"
        "Example 1: When a user asks about returns, explain the 30-day policy.\n"
        "Example 2: When a user asks about billing, direct them to the billing portal.\n"
        "Example 3: For technical issues, offer to escalate to tier 2 support.\n"
        "Example 4: When a user mentions a competitor, redirect to our product benefits.\n"
        "Example 5: When a user is upset, empathise and offer a goodwill gesture.\n\n"
        "Of course, verify the customer's account before sharing account details.\n"
        "Absolutely follow GDPR regulations. Certainly do not share data without consent.\n"
        "As an AI language model you may not always have the latest information.\n"
        "Don't hesitate to ask clarifying questions if the request is ambiguous.\n\n"
        "Core rules:\n"
        "1. Answer using only information from the knowledge base.\n"
        "2. If unsure, escalate the ticket.\n"
        "3. Format: direct answer, then sources.\n"
    ) * 3   # repeat 3× to simulate a bloated prompt

    orig_tokens = count_tokens(verbose_prompt)
    print(f"Verbose system prompt: {orig_tokens:,d} tokens")

    optimized_prompt = calc.optimize_system_prompt(verbose_prompt, target_reduction_pct=0.60)
    opt_tokens = count_tokens(optimized_prompt)
    print(
        f"Optimized prompt    : {opt_tokens:,d} tokens "
        f"({100 * (1 - opt_tokens / orig_tokens):.0f}% reduction)\n"
    )

    # 3. Model comparison
    print("Model cost comparison (50K input + 5K output, 1,000 calls/day):\n")
    print(calc.compare_models(50_000, 5_000, 1_000))

    # 4. Context audit
    messages = [
        {"role": "system", "content": verbose_prompt},
        *[
            {"role": role, "content": f"Turn {i}: " + "context engineering " * 50}
            for i in range(1, 12)
            for role in ("user", "assistant")
        ],
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": (
                    "Search the internal knowledge base for relevant articles, "
                    "policies, FAQs, and product documentation.  Returns the "
                    "top-k most relevant documents ranked by semantic similarity. "
                    "Use this for every customer question before replying."
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
    ]

    audit = calc.audit_context(messages, tools=tools, calls_per_day=1_000)
    print(f"\nContext audit (total {audit.total_tokens:,d} tokens):")
    print("  Token breakdown by role:")
    for role, tok in audit.by_role.items():
        print(f"    {role:12s}: {tok:6,d} tokens")
    print(f"  System prompt efficiency : {audit.system_prompt_efficiency:.0%}")
    print(f"  Estimated wasted tokens  : {audit.wasted_tokens:,d}")
    print("\n  Optimisation suggestions:")
    for s in audit.suggestions:
        print(f"    • {s}")
    print(f"\n  Projected savings @ 1,000 calls/day:")
    print(f"    Tokens/day : {audit.projected_savings['tokens_daily']:,.0f}")
    print(f"    USD/month  : ${audit.projected_savings['usd_monthly']:,.2f}")
    print(f"    USD/year   : ${audit.projected_savings['usd_annual']:,.2f}")

    # 5. Original vs optimised cost comparison
    orig_proj  = calc.calculate_daily_cost("gpt-4o", 1_000, orig_tokens, 500)
    optim_proj = calc.calculate_daily_cost("gpt-4o", 1_000, opt_tokens,  500)
    saving     = orig_proj.monthly - optim_proj.monthly
    print(f"\nOriginal prompt cost  (1K calls/day): ${orig_proj.monthly:,.2f}/month")
    print(f"Optimised prompt cost (1K calls/day): ${optim_proj.monthly:,.2f}/month")
    print(f"Monthly savings                     : ${saving:,.2f}")
