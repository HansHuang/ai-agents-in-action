"""CrewAI research crew: structured multi-agent research pipeline.

Implements a four-agent research crew using CrewAI:

    Researcher → Analyst ──────────────┐
                                        ├─→ Writer → Reviewer
    Researcher → FactChecker ──────────┘

Each agent has a distinct role, goal, and backstory that shapes its approach.
Tasks are chained via ``context`` dependencies so outputs flow automatically.

Also includes a from-scratch equivalent so you can compare the two approaches
side-by-side:

    • Code complexity   — how many lines does each approach need?
    • Execution time    — does the framework add overhead?
    • Token usage       — does CrewAI make extra calls?
    • Output quality    — does structure help?

Run:
    python crewai_research_crew.py

See: docs/06-frameworks-in-practice/03-crewai-autogen.md
"""

from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional CrewAI imports — scripts degrade gracefully when not installed
# ---------------------------------------------------------------------------

try:
    from crewai import Agent, Task, Crew, Process  # type: ignore[import]

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared LLM client
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


# ---------------------------------------------------------------------------
# Simulated tools (CrewAI can call Python callables directly as tools)
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:
    """Simulate web search results for a query."""
    return (
        f"[web_search] Top results for {query!r}:\n"
        "1. Nature (2026): Recent advances confirm the trend.\n"
        "2. MIT Tech Review (2026): Industry adoption accelerating.\n"
        "3. arXiv preprint: New benchmark results released.\n"
        "4. IEEE Spectrum: Engineering challenges remain.\n"
        "5. Wired: Commercial applications expanding rapidly."
    )


def database_lookup(query: str) -> str:
    """Simulate an internal document database lookup."""
    return (
        f"[database] 3 documents found for {query!r}:\n"
        "• Internal report Q1-2026: market size $42B, 38% YoY growth.\n"
        "• Patent analysis 2025: 1,200 new filings in this domain.\n"
        "• Analyst note: three companies control 70% of the market."
    )


def fact_verify(claim: str) -> str:
    """Simulate fact verification against trusted sources."""
    snippet = claim[:80].replace("\n", " ")
    return (
        f"[fact_verify] Checked: '{snippet}...'\n"
        "Status: VERIFIED — cross-referenced against two independent sources."
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResearchResult:
    """Structured output from any research implementation."""

    topic: str
    report: str = ""
    sources: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    agent_outputs: dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    token_usage: int = 0


# ---------------------------------------------------------------------------
# ResearchCrew — CrewAI implementation
# ---------------------------------------------------------------------------


class ResearchCrew:
    """A crew that researches a topic, analyzes findings, and produces a report.

    Agents:
    - Researcher: Gathers information from web and databases
    - Analyst: Analyzes data, identifies trends and patterns
    - FactChecker: Verifies claims and sources
    - Writer: Produces the final report

    Process: Sequential with quality control loop
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "crewai is required for ResearchCrew. "
                "Install with: pip install crewai"
            )
        self.model = model
        self.agents = self._create_agents()
        self.crew: Optional[Crew] = None

    def _create_agents(self) -> dict[str, "Agent"]:
        """Create all agents with distinct roles, goals, and backstories.

        Each agent has:
        - A specific role with expertise level
        - A clear goal that focuses its outputs
        - A backstory that shapes how it approaches problems
        - Appropriate tools for its task
        """
        researcher = Agent(
            role="Senior Research Scientist",
            goal=(
                "Gather comprehensive, accurate information from multiple sources "
                "on the given technology topic, with at least five cited facts."
            ),
            backstory=(
                "You are a PhD-level research scientist with 15 years of experience "
                "tracking technology trends. You are known for thorough literature "
                "reviews and for finding non-obvious connections between research "
                "areas. You never present a claim without a source."
            ),
            tools=[web_search, database_lookup],
            verbose=True,
            allow_delegation=False,
        )

        analyst = Agent(
            role="Technology Trend Analyst",
            goal=(
                "Synthesize research data into meaningful insights, identify "
                "patterns, and highlight strategic implications for the next "
                "three to five years."
            ),
            backstory=(
                "You are a technology strategist who has advised Fortune 500 "
                "companies on technology adoption. You excel at separating signal "
                "from noise and identifying which shifts will have lasting impact. "
                "You think in systems and second-order effects."
            ),
            verbose=True,
            allow_delegation=False,
        )

        fact_checker = Agent(
            role="Scientific Fact Checker",
            goal=(
                "Verify all claims, statistics, and attributions in the research "
                "to ensure accuracy and reliability before publication."
            ),
            backstory=(
                "You are a former science journalist who now specialises in research "
                "integrity. You have a keen eye for unsupported claims, misquoted "
                "statistics, and outdated data. You are constructive but "
                "uncompromising on accuracy."
            ),
            tools=[fact_verify, web_search],
            verbose=True,
            allow_delegation=False,
        )

        writer = Agent(
            role="Technical Report Writer",
            goal=(
                "Transform research and analysis into a clear, well-structured "
                "report that is accessible to informed non-specialists."
            ),
            backstory=(
                "You are a science communicator with an engineering background. "
                "You have written for MIT Technology Review, Nature, and Wired. "
                "You believe clarity is never sacrificed for accuracy — complex "
                "ideas can always be explained simply."
            ),
            verbose=True,
            allow_delegation=False,
        )

        return {
            "researcher": researcher,
            "analyst": analyst,
            "fact_checker": fact_checker,
            "writer": writer,
        }

    def _create_tasks(self, topic: str) -> list["Task"]:
        """Create tasks with clear dependencies.

        Dependency graph:
            1. Research task    (no dependencies)
            2. Analysis task    (depends on research)
            3. Fact-check task  (depends on research)
            4. Writing task     (depends on analysis + fact-check)
            5. Review task      (depends on writing)
        """
        agents = self.agents

        research_task = Task(
            description=(
                f"Research the following technology topic: {topic!r}.\n\n"
                "Your deliverable must include:\n"
                "1. Current state of the technology (2025–2026)\n"
                "2. Key players and recent breakthroughs\n"
                "3. At least five concrete facts with sources\n"
                "4. Timeline of major milestones\n"
                "5. Raw data or statistics where available"
            ),
            agent=agents["researcher"],
            expected_output=(
                "A structured research brief with numbered facts, source "
                "citations, a timeline, and a list of key players."
            ),
        )

        analysis_task = Task(
            description=(
                f"Analyze the research findings on {topic!r}.\n\n"
                "Your deliverable must include:\n"
                "1. Three to five key trends identified in the research\n"
                "2. Strategic implications for the next 3–5 years\n"
                "3. Comparison with adjacent technologies\n"
                "4. Risk factors and open questions\n"
                "5. A confidence rating for each major claim: High / Medium / Low"
            ),
            agent=agents["analyst"],
            expected_output=(
                "An analytical memo with trend identification, strategic "
                "implications, and confidence-rated claims."
            ),
            context=[research_task],
        )

        fact_check_task = Task(
            description=(
                f"Fact-check the research brief on {topic!r}.\n\n"
                "For each major claim:\n"
                "1. Verify the claim is supported by the cited source\n"
                "2. Check whether statistics are current (within 18 months)\n"
                "3. Flag claims that need correction or qualification\n"
                "4. Confirm key players and their roles are accurately described\n"
                "5. Assign a status: Verified / Needs Correction / Unverified"
            ),
            agent=agents["fact_checker"],
            expected_output=(
                "A fact-check ledger: each claim with its source, status "
                "(Verified / Needs Correction / Unverified), and correction notes."
            ),
            context=[research_task],
        )

        writing_task = Task(
            description=(
                f"Write a professional research report on {topic!r}.\n\n"
                "Requirements:\n"
                "• Executive summary (≈100 words)\n"
                "• Current state section (≈200 words)\n"
                "• Key findings: three to five bullet points\n"
                "• Strategic outlook section (≈150 words)\n"
                "• Sources list at the end\n\n"
                "Incorporate all corrections from the fact-check. "
                "Only include Verified or corrected claims."
            ),
            agent=agents["writer"],
            expected_output=(
                "A polished 500–600 word research report with executive summary, "
                "current state, key findings, strategic outlook, and sources."
            ),
            context=[analysis_task, fact_check_task],
        )

        review_task = Task(
            description=(
                "Review the draft research report for quality. Check:\n"
                "1. Does the executive summary capture the three most important points?\n"
                "2. Are key findings concrete and specific (not vague)?\n"
                "3. Is the strategic outlook grounded in the research (not speculative)?\n"
                "4. Is the writing clear and free of unexplained jargon?\n"
                "5. Are all sources listed?\n\n"
                "If corrections are needed, provide specific edits. "
                "If the report meets all quality criteria, respond with APPROVED."
            ),
            agent=agents["analyst"],
            expected_output=(
                "Either 'APPROVED' with a brief quality note, or a numbered "
                "list of specific corrections with the section that needs revision."
            ),
            context=[writing_task],
        )

        return [
            research_task,
            analysis_task,
            fact_check_task,
            writing_task,
            review_task,
        ]

    def research(self, topic: str) -> ResearchResult:
        """Run the full research crew and return structured results.

        Returns:
            ResearchResult with topic, report, sources, key_findings,
            agent_outputs, execution_time, and token_usage.
        """
        tasks = self._create_tasks(topic)
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        start = time.perf_counter()
        raw_result = self.crew.kickoff()
        elapsed = time.perf_counter() - start

        # Collect per-task outputs (CrewAI stores them on task.output)
        task_names = ["research", "analysis", "fact_check", "writing", "review"]
        agent_outputs: dict[str, str] = {}
        for name, task in zip(task_names, tasks):
            output = getattr(task, "output", None)
            if output is None:
                agent_outputs[name] = ""
            elif hasattr(output, "raw"):
                agent_outputs[name] = output.raw
            else:
                agent_outputs[name] = str(output)

        report_text = str(raw_result)
        sources = _extract_sources(report_text)

        return ResearchResult(
            topic=topic,
            report=report_text,
            sources=sources,
            agent_outputs=agent_outputs,
            execution_time=elapsed,
        )

    def compare_to_from_scratch(self, topic: str) -> dict:
        """Run research with CrewAI and from-scratch; compare key metrics.

        Returns a dict with "crewai" and "from_scratch" sub-dicts containing:
        execution_time, report_length (words), sources, lines_of_code.
        """
        print("=== Running CrewAI implementation ===\n")
        t0 = time.perf_counter()
        crewai_result = self.research(topic)
        crewai_time = time.perf_counter() - t0

        print("\n=== Running from-scratch implementation ===\n")
        scratch = FromScratchResearcher()
        t0 = time.perf_counter()
        scratch_result = scratch.research(topic)
        scratch_time = time.perf_counter() - t0

        return {
            "topic": topic,
            "crewai": {
                "execution_time_s": round(crewai_time, 2),
                "report_length_words": len(crewai_result.report.split()),
                "sources_cited": len(crewai_result.sources),
                "token_usage": crewai_result.token_usage,
                "lines_of_code": _count_class_lines(ResearchCrew),
            },
            "from_scratch": {
                "execution_time_s": round(scratch_time, 2),
                "report_length_words": len(scratch_result.report.split()),
                "sources_cited": len(scratch_result.sources),
                "token_usage": scratch_result.token_usage,
                "lines_of_code": _count_class_lines(FromScratchResearcher),
            },
        }


# ---------------------------------------------------------------------------
# FromScratchResearcher — pure OpenAI SDK equivalent
# ---------------------------------------------------------------------------


def _call_llm(
    client: OpenAI,
    system: str,
    user: str,
    model: str = "gpt-4o",
    token_counter: Optional[list[int]] = None,
) -> tuple[str, int]:
    """Make a single chat completion call; return (text, tokens_used)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    if token_counter is not None:
        token_counter[0] += tokens
    return text, tokens


class FromScratchResearcher:
    """The same four-agent pipeline with zero framework dependencies.

    You explicitly control every interaction. The orchestration code is
    visible and modifiable — no magic happens behind the scenes.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.client = _get_client()

    def research(self, topic: str) -> ResearchResult:
        """Run the full research pipeline and return structured results."""
        tokens: list[int] = [0]
        agent_outputs: dict[str, str] = {}

        # Step 1: Researcher
        research_notes, _ = _call_llm(
            self.client,
            system=(
                "You are a senior research scientist. Produce a structured research "
                "brief with numbered facts, source citations, a timeline, and "
                "a list of key players."
            ),
            user=f"Research this technology topic in depth: {topic}",
            model=self.model,
            token_counter=tokens,
        )
        agent_outputs["research"] = research_notes

        # Step 2: Analyst (depends on research)
        analysis, _ = _call_llm(
            self.client,
            system=(
                "You are a technology trend analyst. Identify three to five key "
                "trends, strategic implications, and rate each claim "
                "High / Medium / Low confidence."
            ),
            user=f"Analyze these research findings:\n\n{research_notes}",
            model=self.model,
            token_counter=tokens,
        )
        agent_outputs["analysis"] = analysis

        # Step 3: FactChecker (depends on research, runs alongside analysis)
        fact_check, _ = _call_llm(
            self.client,
            system=(
                "You are a scientific fact checker. For each major claim in the "
                "research, assign a status: Verified / Needs Correction / Unverified. "
                "Provide correction notes for anything that needs fixing."
            ),
            user=f"Fact-check this research brief:\n\n{research_notes}",
            model=self.model,
            token_counter=tokens,
        )
        agent_outputs["fact_check"] = fact_check

        # Step 4: Writer (depends on analysis + fact_check)
        report, _ = _call_llm(
            self.client,
            system=(
                "You are a technical report writer. Produce a polished 500–600 word "
                "report with: executive summary, current state, key findings, "
                "strategic outlook, and sources."
            ),
            user=(
                f"Write a report based on:\n\n"
                f"ANALYSIS:\n{analysis}\n\n"
                f"FACT CHECK:\n{fact_check}"
            ),
            model=self.model,
            token_counter=tokens,
        )
        agent_outputs["writing"] = report

        # Step 5: Reviewer (depends on writing)
        review, _ = _call_llm(
            self.client,
            system=(
                "You are a senior editor. Review the report for quality, clarity, "
                "and accuracy. If it meets all criteria, respond with 'APPROVED' "
                "and a brief note. Otherwise, list specific corrections."
            ),
            user=f"Review this research report:\n\n{report}",
            model=self.model,
            token_counter=tokens,
        )
        agent_outputs["review"] = review

        sources = _extract_sources(report)
        return ResearchResult(
            topic=topic,
            report=report,
            sources=sources,
            agent_outputs=agent_outputs,
            execution_time=0.0,  # Caller measures wall time
            token_usage=tokens[0],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_sources(text: str) -> list[str]:
    """Heuristically extract source lines from a report."""
    sources: list[str] = []
    in_sources = False
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith(("sources", "references", "bibliography")):
            in_sources = True
            continue
        if in_sources and stripped and not stripped.startswith("#"):
            sources.append(stripped)
        if in_sources and not stripped:
            # A blank line after a sources section might mean we're done,
            # but keep collecting until the next heading.
            pass
    # Fallback: grab any http/bullet lines if no sources section found
    if not sources:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("http", "[", "•", "-")) and len(s) > 15:
                sources.append(s)
    return sources[:10]  # Cap at 10


def _count_class_lines(cls: type) -> int:
    """Return the number of lines in a class definition."""
    try:
        return len(inspect.getsource(cls).splitlines())
    except (OSError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _divider(title: str, width: int = 60) -> None:
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def _preview(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n  … [{len(text) - max_chars} more chars]"


if __name__ == "__main__":
    TOPIC = "The state of quantum computing in 2026"

    if not CREWAI_AVAILABLE:
        print("CrewAI not installed. Running from-scratch implementation only.")
        print("To enable CrewAI: pip install crewai\n")
        researcher: FromScratchResearcher | ResearchCrew = FromScratchResearcher()
        result = researcher.research(TOPIC)
    else:
        print("CrewAI detected. Running full research crew.\n")
        researcher = ResearchCrew()
        result = researcher.research(TOPIC)

    _divider("TOPIC")
    print(f"  {result.topic}")

    for agent_name, output in result.agent_outputs.items():
        _divider(f"AGENT: {agent_name.upper()}")
        print(_preview(output))

    _divider("FINAL REPORT")
    print(_preview(result.report, max_chars=1200))

    _divider("METADATA")
    print(f"  Execution time : {result.execution_time:.1f}s")
    print(f"  Token usage    : {result.token_usage:,}")
    print(f"  Sources found  : {len(result.sources)}")
    print(f"  Report length  : {len(result.report.split())} words")
    if result.sources:
        print("\n  Sources:")
        for src in result.sources[:5]:
            print(f"    {src}")

    if CREWAI_AVAILABLE:
        _divider("CODE COMPLEXITY COMPARISON")
        crewai_lines = _count_class_lines(ResearchCrew)
        scratch_lines = _count_class_lines(FromScratchResearcher)
        print(f"  ResearchCrew (CrewAI)    : {crewai_lines} lines")
        print(f"  FromScratchResearcher    : {scratch_lines} lines")
        print(
            f"\n  CrewAI adds ~{max(0, crewai_lines - scratch_lines)} lines of "
            "definition overhead in exchange for automatic context passing, "
            "role-based delegation, and process management."
        )
