"""Multi-agent over-engineering detector.

Analyses a project description and warns when a multi-agent architecture
is likely overkill — adding cost, complexity, and latency without a
proportional benefit.

The detector works in two stages:
    1. Rule-based pattern matching against known over-engineering signals
    2. Optional LLM-as-judge for nuanced cases (requires OPENAI_API_KEY)

Run:
    python over_engineering_detector.py

See: docs/06-frameworks-in-practice/03-crewai-autogen.md
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Cost model (gpt-4o, May 2026 pricing)
# ---------------------------------------------------------------------------

TOKENS_PER_AGENT_PER_CALL = 3_000    # average tokens per agent round-trip
CALLS_PER_AGENT_PER_DAY = 200        # average daily queries
COST_PER_1K_TOKENS = 0.005           # blended input + output rate (USD)
DAYS_PER_MONTH = 30

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OverEngineeringReport:
    """Full analysis of a proposed multi-agent design."""

    risk_level: str = "low"          # "low" | "medium" | "high"
    agent_count: int = 0             # Proposed agent count (extracted from description)
    suggested_agent_count: int = 0   # What the detector recommends
    warnings: list[str] = field(default_factory=list)
    simplification_suggestions: list[str] = field(default_factory=list)
    cost_savings_estimate: float = 0.0   # Monthly USD saved by simplifying
    llm_verdict: Optional[str] = None   # Set when LLM-as-judge is used


# ---------------------------------------------------------------------------
# OverEngineeringDetector
# ---------------------------------------------------------------------------


class OverEngineeringDetector:
    """Detect when a project might be over-engineered with multi-agent.

    Usage::

        detector = OverEngineeringDetector()
        report = detector.analyze("5-agent system for a customer support chatbot")
        print(detector.suggest_simplification(report))
    """

    # Patterns map (signal_key → user-facing warning message)
    WARNING_PATTERNS: dict[str, str] = {
        "faq_bot": (
            "An FAQ or Q&A bot doesn't need multiple agents. "
            "One agent with RAG retrieval handles this completely."
        ),
        "crud_app": (
            "CRUD operations (create, read, update, delete) don't benefit from "
            "agent collaboration. Use deterministic code with tools, not agents."
        ),
        "simple_workflow": (
            "A strictly linear workflow (step A then B then C) doesn't justify "
            "multi-agent. A single Plan-and-Execute agent is simpler and cheaper."
        ),
        "single_domain": (
            "All tasks appear to be in the same knowledge domain. "
            "Specialized agents add overhead without adding specialization value."
        ),
        "no_critique_needed": (
            "The output doesn't require adversarial review or quality checking. "
            "Skip the Critic agent — it adds tokens and latency for no benefit."
        ),
        "agents_as_tools": (
            "Some 'agents' here are doing one thing and returning a result. "
            "Those are tools, not agents. Fold them into a single agent's tool list."
        ),
        "tightly_coupled_pipeline": (
            "Every agent depends on the previous one's full output. "
            "This is a sequential pipeline — a single agent with structured steps "
            "or a Plan-and-Execute loop does the same thing with less overhead."
        ),
        "real_time_requirement": (
            "Real-time or low-latency requirements conflict with multi-agent "
            "coordination. Each agent-to-agent handoff adds 1–5 seconds of latency."
        ),
        "simple_classification": (
            "Classification or routing tasks don't need multiple agents. "
            "A single LLM call with a structured output schema handles this."
        ),
        "single_user_turn": (
            "If the system is stateless (one user turn, one reply), multi-agent "
            "adds infrastructure complexity for no conversational benefit."
        ),
    }

    # Patterns that JUSTIFY multi-agent (reduce false-positive risk level)
    JUSTIFICATION_PATTERNS: list[str] = [
        "adversarial",
        "debate",
        "multiple domain",
        "different expertise",
        "parallel",
        "independent sub-task",
        "human in the loop",
        "critique",
        "quality check",
        "role specializ",
        "research.*analys.*writ",  # classic research pipeline
        "long-running",
        "autonomous",
    ]

    def __init__(self, use_llm_judge: bool = False) -> None:
        """
        Args:
            use_llm_judge: If True and OPENAI_API_KEY is set, use an LLM
                           to provide a nuanced second opinion. Requires openai.
        """
        self.use_llm_judge = use_llm_judge
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def _extract_agent_count(self, description: str) -> int:
        """Try to extract an explicit agent count from the description."""
        # Matches: "5-agent", "five agents", "5 agents", "a team of 3"
        patterns = [
            r"(\d+)[\s\-]agent",
            r"(\d+)\s+agents?",
            r"team\s+of\s+(\d+)",
            r"(\d+)\s+specialized",
            r"(one|two|three|four|five|six|seven|eight|nine|ten)\s+agents?",
        ]
        word_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        for pattern in patterns:
            m = re.search(pattern, description, re.I)
            if m:
                raw = m.group(1).lower()
                return word_map.get(raw, int(raw) if raw.isdigit() else 0)
        return 0

    def _check_signal(self, description: str, key: str) -> bool:
        """Return True if the description matches the named signal pattern."""
        lower = description.lower()
        checks: dict[str, list[str]] = {
            "faq_bot":              ["faq", "frequently asked", "q&a", "q & a", "knowledge base answer"],
            "crud_app":             ["crud", "create update delete", "database", "form", "record management"],
            "simple_workflow":      ["linear", "step by step", "step-by-step", "one after another",
                                     "sequential pipeline", "waterfall"],
            "single_domain":        [],  # heuristic — checked separately
            "no_critique_needed":   ["no review", "no quality check", "single pass", "no edit"],
            "agents_as_tools":      ["returns a result", "calls an api", "fetch", "lookup", "query the"],
            "tightly_coupled_pipeline": ["each agent", "previous agent", "passes to the next",
                                         "chain of agents", "pipeline of agents"],
            "real_time_requirement": ["real-time", "real time", "low latency", "< 1 second",
                                      "sub-second", "millisecond", "streaming response"],
            "simple_classification": ["classify", "route", "intent detection", "categoris",
                                      "categoriz", "label"],
            "single_user_turn":     ["one-shot", "single turn", "stateless", "no conversation",
                                     "no history", "no memory"],
        }
        if key in checks:
            return any(kw in lower for kw in checks[key])
        return False

    def _check_single_domain(self, description: str) -> bool:
        """Heuristic: if all role words map to the same domain, flag it."""
        domains = {
            "customer_support": ["support", "helpdesk", "ticket", "complaint", "inquiry"],
            "data_entry":       ["fill", "form", "entry", "transcrib", "extract fields"],
            "simple_qa":        ["answer", "question", "faq", "lookup"],
            "notification":     ["notify", "alert", "send email", "send message", "push notification"],
        }
        lower = description.lower()
        for domain_keywords in domains.values():
            if sum(1 for kw in domain_keywords if kw in lower) >= 2:
                return True
        return False

    def _count_justifications(self, description: str) -> int:
        lower = description.lower()
        return sum(
            1 for pat in self.JUSTIFICATION_PATTERNS if re.search(pat, lower)
        )

    def analyze(self, project_description: str) -> OverEngineeringReport:
        """Analyse a project for over-engineering signals.

        Signals evaluated:
        - Agent count vs. task complexity
        - Agents with overlapping capabilities
        - Agents that only do one thing (should be tools)
        - Sequential tasks that could be a single agent with planning
        - No clear benefit from specialization

        Returns:
            OverEngineeringReport with risk_level, warnings, and suggestions.
        """
        warnings: list[str] = []

        # --- Check each signal ---
        for key, message in self.WARNING_PATTERNS.items():
            if key == "single_domain":
                if self._check_single_domain(project_description):
                    warnings.append(message)
            elif self._check_signal(project_description, key):
                warnings.append(message)

        # --- Agent count heuristic ---
        proposed = self._extract_agent_count(project_description)
        justifications = self._count_justifications(project_description)

        # Reduce warnings for well-justified designs
        max_justified_agents = 2 + justifications  # base 2 + 1 per justification
        agent_count_warning = ""
        if proposed > 0 and proposed > max_justified_agents:
            agent_count_warning = (
                f"The design proposes {proposed} agents but the described task "
                f"appears to justify at most {max_justified_agents}. "
                f"Each extra agent adds token overhead and latency."
            )
            warnings.append(agent_count_warning)

        # --- Determine risk level ---
        n_warnings = len(warnings)
        if n_warnings == 0:
            risk_level = "low"
        elif n_warnings <= 2:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Reduce risk if there are strong justifications
        if justifications >= 3 and risk_level == "high":
            risk_level = "medium"
        if justifications >= 5:
            risk_level = "low"

        # --- Suggested agent count ---
        if proposed == 0:
            suggested = 1
        elif risk_level == "low":
            suggested = proposed
        elif risk_level == "medium":
            suggested = max(1, proposed - 1)
        else:
            suggested = max(1, min(2, proposed // 2))

        # --- Simplification suggestions ---
        suggestions = self._build_suggestions(project_description, warnings, proposed, suggested)

        # --- Cost savings estimate ---
        cost_savings = self._estimate_cost_savings(proposed, suggested)

        report = OverEngineeringReport(
            risk_level=risk_level,
            agent_count=proposed,
            suggested_agent_count=suggested,
            warnings=warnings,
            simplification_suggestions=suggestions,
            cost_savings_estimate=cost_savings,
        )

        # --- Optional LLM-as-judge ---
        if self.use_llm_judge and os.environ.get("OPENAI_API_KEY"):
            report.llm_verdict = self._llm_judge(project_description, report)

        return report

    def _build_suggestions(
        self,
        description: str,
        warnings: list[str],
        proposed: int,
        suggested: int,
    ) -> list[str]:
        """Build specific simplification suggestions based on detected signals."""
        suggestions: list[str] = []

        if any("FAQ" in w or "Q&A" in w for w in warnings):
            suggestions.append(
                "Replace the multi-agent system with a single agent + RAG pipeline. "
                "Use a vector database to retrieve relevant docs and a single LLM call to answer."
            )

        if any("CRUD" in w for w in warnings):
            suggestions.append(
                "Replace agents with deterministic code. Use function tools for "
                "database operations; reserve the LLM for natural-language interpretation only."
            )

        if any("linear" in w.lower() or "pipeline" in w.lower() for w in warnings):
            suggestions.append(
                "Use a single Plan-and-Execute agent instead of a chain of agents. "
                "The agent plans the steps internally and executes them with tools."
            )

        if any("tool" in w.lower() for w in warnings):
            suggestions.append(
                "Convert single-purpose agents into tools. A tool is a Python function "
                "with a docstring — simpler, faster, and cheaper than a full agent."
            )

        if any("real-time" in w.lower() or "latency" in w.lower() for w in warnings):
            suggestions.append(
                "For real-time requirements, pre-compute results or use a single fast "
                "LLM call. Multi-agent coordination typically adds 3–10 seconds of latency."
            )

        if proposed > 0 and suggested < proposed:
            savings_pct = int((1 - suggested / proposed) * 100)
            suggestions.append(
                f"Reducing from {proposed} to {suggested} agents saves approximately "
                f"{savings_pct}% of token usage and improves reliability (fewer failure points)."
            )

        if not suggestions:
            suggestions.append(
                "The design appears reasonable. Proceed with the multi-agent approach, "
                "but monitor token costs closely in the first two weeks."
            )

        return suggestions

    def _estimate_cost_savings(self, proposed: int, suggested: int) -> float:
        """Estimate monthly USD savings from reducing the agent count."""
        if proposed <= 0 or suggested >= proposed:
            return 0.0
        proposed_tokens = proposed * TOKENS_PER_AGENT_PER_CALL * CALLS_PER_AGENT_PER_DAY * DAYS_PER_MONTH
        suggested_tokens = suggested * TOKENS_PER_AGENT_PER_CALL * CALLS_PER_AGENT_PER_DAY * DAYS_PER_MONTH
        savings_tokens = proposed_tokens - suggested_tokens
        return round(savings_tokens / 1000 * COST_PER_1K_TOKENS, 2)

    def suggest_simplification(self, report: OverEngineeringReport) -> str:
        """Return a formatted simplification recommendation as a string."""
        lines = [
            f"OVER-ENGINEERING ASSESSMENT",
            f"{'─'*40}",
            f"Risk level       : {report.risk_level.upper()}",
            f"Proposed agents  : {report.agent_count or 'unknown'}",
            f"Suggested agents : {report.suggested_agent_count}",
            f"Monthly savings  : ${report.cost_savings_estimate:.2f} (est.)",
            "",
        ]
        if report.warnings:
            lines.append("WARNINGS:")
            for w in report.warnings:
                lines.append(f"  ⚠  {w}")
            lines.append("")

        if report.simplification_suggestions:
            lines.append("SIMPLIFICATION SUGGESTIONS:")
            for i, s in enumerate(report.simplification_suggestions, 1):
                lines.append(f"  {i}. {s}")
            lines.append("")

        if report.llm_verdict:
            lines.append("LLM VERDICT:")
            lines.append(f"  {report.llm_verdict}")

        return "\n".join(lines)

    def _llm_judge(self, description: str, report: OverEngineeringReport) -> str:
        """Ask an LLM to provide a one-paragraph verdict on the design."""
        warnings_text = "\n".join(f"- {w}" for w in report.warnings) or "None"
        prompt = (
            f"A developer proposes this multi-agent system:\n\n"
            f"{description}\n\n"
            f"A rule-based checker raised these concerns:\n{warnings_text}\n\n"
            "In one paragraph (max 100 words), give a balanced verdict: "
            "Is this system over-engineered? What is the key risk? "
            "What is the simplest version that achieves the same goal?"
        )
        try:
            response = self._get_client().chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior AI systems architect. "
                            "Give concise, practical advice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content or "(no response)"
        except Exception:  # noqa: BLE001
            return "(LLM judge unavailable)"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo(detector: OverEngineeringDetector) -> None:
    test_cases = [
        (
            "5-agent system for a customer support chatbot that answers FAQs "
            "about our product. Agents: Greeter, IntentClassifier, KnowledgeRetriever, "
            "ResponseWriter, ResponseReviewer.",
            "Customer support FAQ bot",
            True,   # expect high risk
        ),
        (
            "A 3-agent research crew: Researcher (web search + database), "
            "FactChecker (verify sources), Writer (produce structured report). "
            "Used for daily competitive intelligence reports with adversarial review.",
            "Research + analysis + writing team",
            False,  # expect low risk
        ),
        (
            "A 4-agent pipeline to fill out insurance claim forms: "
            "DataExtractor, FormFiller, ValidatorAgent, SubmitterAgent. "
            "Linear workflow, one-shot per claim, no critique needed.",
            "Insurance form filling pipeline",
            True,   # expect high risk
        ),
        (
            "An 8-agent content marketing system: IdeaGenerator, TopicResearcher, "
            "OutlineWriter, DraftWriter, EditorAgent, SEOOptimizer, ImagePromptAgent, "
            "SchedulerAgent. Each depends on the previous agent's output.",
            "Content marketing pipeline",
            True,   # expect medium-high risk
        ),
    ]

    for description, label, _expect_risky in test_cases:
        print(f"\n{'═'*60}")
        print(f"  TEST CASE: {label}")
        print(f"{'═'*60}")
        print(f"\nDescription:\n  {description[:120]}{'…' if len(description) > 120 else ''}\n")
        report = detector.analyze(description)
        print(detector.suggest_simplification(report))

    # Cost comparison: 5-agent vs. 1-agent customer support bot
    print(f"\n{'═'*60}")
    print("  COST COMPARISON: 5-agent vs. 1-agent customer support bot")
    print(f"{'═'*60}")
    proposed = 5
    simplified = 1
    proposed_monthly = (
        proposed * TOKENS_PER_AGENT_PER_CALL * CALLS_PER_AGENT_PER_DAY * DAYS_PER_MONTH
        / 1000 * COST_PER_1K_TOKENS
    )
    simplified_monthly = (
        simplified * TOKENS_PER_AGENT_PER_CALL * CALLS_PER_AGENT_PER_DAY * DAYS_PER_MONTH
        / 1000 * COST_PER_1K_TOKENS
    )
    print(f"\n  Assumptions: {CALLS_PER_AGENT_PER_DAY} queries/day, "
          f"{TOKENS_PER_AGENT_PER_CALL:,} tokens/agent/call\n")
    print(f"  5-agent monthly cost  : ${proposed_monthly:.2f}")
    print(f"  1-agent monthly cost  : ${simplified_monthly:.2f}")
    print(f"  Monthly savings       : ${proposed_monthly - simplified_monthly:.2f}")
    print(f"  Annual savings        : ${(proposed_monthly - simplified_monthly) * 12:.2f}")
    print(
        f"\n  For 200 daily queries, the 1-agent + RAG design saves "
        f"~${(proposed_monthly - simplified_monthly) * 12:.0f}/year with equal "
        f"or better accuracy and significantly lower latency."
    )


if __name__ == "__main__":
    use_llm = bool(os.environ.get("OPENAI_API_KEY"))
    detector = OverEngineeringDetector(use_llm_judge=use_llm)
    _run_demo(detector)
