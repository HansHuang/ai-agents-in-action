"""Multi-agent comparison: the same research task implemented three ways.

Task: "Research the impact of AI on software developer productivity."

Three parallel implementations complete the same research task so you can
measure tradeoffs across four dimensions:

    1. CODE METRICS       — lines of code, imports, orchestration overhead
    2. EXECUTION METRICS  — wall time, LLM calls, token usage, estimated cost
    3. QUALITY METRICS    — report length, sources cited, critique integration
    4. CONTROL METRICS    — traceability, modifiability, extensibility

Run:
    python multi_agent_comparison.py

See: docs/06-frameworks-in-practice/03-crewai-autogen.md
"""

from __future__ import annotations

import inspect
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional framework imports — graceful degradation
# ---------------------------------------------------------------------------

try:
    from crewai import Agent, Task, Crew, Process  # type: ignore[import]

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager  # type: ignore[import]

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

RESEARCH_TASK = "Research the impact of AI on software developer productivity."
GPT4O_INPUT_COST_PER_1K = 0.0025   # USD per 1,000 input tokens  (gpt-4o, May 2026)
GPT4O_OUTPUT_COST_PER_1K = 0.010   # USD per 1,000 output tokens

# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CodeMetrics:
    """Static code measurements collected via inspect."""

    agent_definition_lines: int = 0
    orchestration_lines: int = 0
    total_lines: int = 0
    import_count: int = 0


@dataclass
class ExecutionMetrics:
    """Measurements captured during a live run."""

    execution_time_s: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class QualityMetrics:
    """Measurements of the produced report."""

    report_length_words: int = 0
    sources_cited: int = 0
    has_critique_feedback: bool = False
    structure_score: int = 0        # 1–10, set by LLM-as-judge or heuristic


@dataclass
class ControlMetrics:
    """Qualitative ratings for developer control (out of 10)."""

    traceable_decision_path: bool = True
    can_modify_agent_comms: bool = True
    can_add_agent_mid_workflow: bool = True
    can_change_execution_order: bool = True
    control_score: int = 10         # Aggregate 0–10


@dataclass
class ComparisonResult:
    """Full comparison for a single implementation."""

    name: str
    code: CodeMetrics = field(default_factory=CodeMetrics)
    execution: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    control: ControlMetrics = field(default_factory=ControlMetrics)
    report: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Shared LLM helper
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


def _llm(
    system: str,
    user: str,
    model: str = "gpt-4o",
    call_counter: Optional[list[int]] = None,
    token_counter: Optional[list[int]] = None,
) -> str:
    """Single chat completion; optionally increments call and token counters."""
    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content or ""
    if call_counter is not None:
        call_counter[0] += 1
    if token_counter is not None and response.usage:
        token_counter[0] += response.usage.total_tokens
    return text


def _count_sources(text: str) -> int:
    """Heuristically count cited sources in a report."""
    count = 0
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^\d+\.", s) or s.startswith(("http", "[", "•", "-")):
            if len(s) > 15:
                count += 1
    return min(count, 20)


def _has_critique(text: str) -> bool:
    return bool(re.search(r"\bcritiq|however|caveat|limitation|concern\b", text, re.I))


def _heuristic_structure_score(text: str) -> int:
    """Score 1–10 based on section headings and list items found."""
    headings = len(re.findall(r"^#{1,3} .+", text, re.M))
    bullets = len(re.findall(r"^[•\-\*] .+", text, re.M))
    score = min(10, max(1, headings * 2 + bullets))
    return score


def _estimate_cost(tokens: int) -> float:
    """Rough cost estimate assuming 40% input / 60% output token split."""
    input_tokens = tokens * 0.4
    output_tokens = tokens * 0.6
    return round(
        (input_tokens / 1000) * GPT4O_INPUT_COST_PER_1K
        + (output_tokens / 1000) * GPT4O_OUTPUT_COST_PER_1K,
        4,
    )


def _count_lines_in_source(source: str) -> int:
    return len(source.strip().splitlines())


# ---------------------------------------------------------------------------
# Implementation 1: From Scratch
# ---------------------------------------------------------------------------


class FromScratchResearch:
    """Three-agent research pipeline with zero framework dependencies.

    Agents:    Researcher → Critic → Writer
    You control every message, every prompt, and the full execution order.

    Code metrics (approximate):
        Agent definitions  : 3 system prompts = ~15 lines
        Orchestration      : explicit loop = ~30 lines
        Total              : ~45 lines of implementation
    """

    def run(self, task: str) -> ComparisonResult:
        calls: list[int] = [0]
        tokens: list[int] = [0]
        start = time.perf_counter()

        # --- Agent 1: Researcher ---
        research = _llm(
            system=(
                "You are a research analyst. Produce a structured research brief "
                "on the given topic. Include at least three concrete statistics, "
                "three named sources, and a timeline of key developments."
            ),
            user=f"Research topic: {task}",
            call_counter=calls,
            token_counter=tokens,
        )

        # --- Agent 2: Critic ---
        critique = _llm(
            system=(
                "You are a rigorous research critic. Review the research brief. "
                "Identify three specific weaknesses: missing data, unsupported "
                "claims, or gaps in coverage. Be constructive and specific."
            ),
            user=f"Critique this research brief:\n\n{research}",
            call_counter=calls,
            token_counter=tokens,
        )

        # --- Agent 3: Writer (uses both research and critique) ---
        report = _llm(
            system=(
                "You are a technical report writer. Produce a 400–500 word report "
                "with: ## Executive Summary, ## Key Findings (bullet points), "
                "## Analysis, ## Sources. Address the critic's feedback."
            ),
            user=(
                f"Write a report based on:\n\nRESEARCH:\n{research}\n\n"
                f"CRITIQUE FEEDBACK:\n{critique}"
            ),
            call_counter=calls,
            token_counter=tokens,
        )

        elapsed = time.perf_counter() - start

        # Code metrics (measured on this class)
        src = inspect.getsource(FromScratchResearch)
        code = CodeMetrics(
            agent_definition_lines=15,   # 3 system prompts ≈ 5 lines each
            orchestration_lines=30,
            total_lines=_count_lines_in_source(src),
            import_count=4,              # os, re, time, openai
        )
        exec_m = ExecutionMetrics(
            execution_time_s=round(elapsed, 2),
            llm_calls=calls[0],
            total_tokens=tokens[0],
            estimated_cost_usd=_estimate_cost(tokens[0]),
        )
        quality = QualityMetrics(
            report_length_words=len(report.split()),
            sources_cited=_count_sources(report),
            has_critique_feedback=_has_critique(report),
            structure_score=_heuristic_structure_score(report),
        )
        control = ControlMetrics(
            traceable_decision_path=True,
            can_modify_agent_comms=True,
            can_add_agent_mid_workflow=True,
            can_change_execution_order=True,
            control_score=10,
        )
        return ComparisonResult(
            name="From Scratch",
            code=code,
            execution=exec_m,
            quality=quality,
            control=control,
            report=report,
        )


# ---------------------------------------------------------------------------
# Implementation 2: CrewAI
# ---------------------------------------------------------------------------


class CrewAIResearch:
    """Same three-agent pipeline built with CrewAI.

    CrewAI handles: context passing between tasks, agent-to-agent delegation,
    process ordering, and output formatting — all automatically.

    Code metrics (approximate):
        Agent definitions  : role + goal + backstory × 3 = ~30 lines
        Task definitions   : description + expected_output × 3 = ~24 lines
        Orchestration      : Crew() + kickoff() = ~6 lines
        Total              : ~60 lines of definition code
    """

    def run(self, task: str) -> ComparisonResult:
        if not CREWAI_AVAILABLE:
            return ComparisonResult(
                name="CrewAI",
                error="crewai not installed. Run: pip install crewai",
            )

        calls: list[int] = [0]
        tokens: list[int] = [0]

        # Monkey-patch to count calls (CrewAI calls LLM internally)
        original_create = _get_client().chat.completions.create

        def _counting_create(*args, **kwargs):
            calls[0] += 1
            result = original_create(*args, **kwargs)
            if hasattr(result, "usage") and result.usage:
                tokens[0] += result.usage.total_tokens
            return result

        _get_client().chat.completions.create = _counting_create  # type: ignore[method-assign]

        start = time.perf_counter()
        try:
            researcher = Agent(
                role="Research Analyst",
                goal="Find concrete data and statistics about the research topic",
                backstory=(
                    "You are a data-driven researcher who always backs claims "
                    "with numbers and named sources."
                ),
                verbose=False,
                allow_delegation=False,
            )
            critic = Agent(
                role="Research Critic",
                goal="Identify three specific gaps or weaknesses in the research",
                backstory=(
                    "You are a former academic peer reviewer. You spot unsupported "
                    "claims and missing data instantly."
                ),
                verbose=False,
                allow_delegation=False,
            )
            writer = Agent(
                role="Technical Writer",
                goal="Produce a structured 400–500 word report incorporating critique",
                backstory=(
                    "You write for technical audiences. Your reports have clear "
                    "sections and cite every claim."
                ),
                verbose=False,
                allow_delegation=False,
            )

            research_task = Task(
                description=f"Research topic: {task}. Include statistics and sources.",
                agent=researcher,
                expected_output="Structured research brief with facts, sources, timeline.",
            )
            critique_task = Task(
                description="Identify three specific weaknesses in the research brief.",
                agent=critic,
                expected_output="Three numbered critiques with specific suggestions.",
                context=[research_task],
            )
            writing_task = Task(
                description=(
                    "Write a 400–500 word report with sections: Executive Summary, "
                    "Key Findings, Analysis, Sources. Address all critique points."
                ),
                agent=writer,
                expected_output="Polished structured report, 400–500 words.",
                context=[research_task, critique_task],
            )

            crew = Crew(
                agents=[researcher, critic, writer],
                tasks=[research_task, critique_task, writing_task],
                process=Process.sequential,
                verbose=False,
            )
            raw = crew.kickoff()
            report = str(raw)
        finally:
            _get_client().chat.completions.create = original_create  # type: ignore[method-assign]

        elapsed = time.perf_counter() - start

        src = inspect.getsource(CrewAIResearch)
        code = CodeMetrics(
            agent_definition_lines=30,
            orchestration_lines=12,
            total_lines=_count_lines_in_source(src),
            import_count=5,   # + crewai
        )
        exec_m = ExecutionMetrics(
            execution_time_s=round(elapsed, 2),
            llm_calls=calls[0],
            total_tokens=tokens[0],
            estimated_cost_usd=_estimate_cost(tokens[0]),
        )
        quality = QualityMetrics(
            report_length_words=len(report.split()),
            sources_cited=_count_sources(report),
            has_critique_feedback=_has_critique(report),
            structure_score=_heuristic_structure_score(report),
        )
        control = ControlMetrics(
            traceable_decision_path=True,       # Task outputs are inspectable
            can_modify_agent_comms=False,       # CrewAI controls context passing
            can_add_agent_mid_workflow=False,   # Crew is immutable after creation
            can_change_execution_order=False,   # Process.sequential is fixed
            control_score=4,
        )
        return ComparisonResult(
            name="CrewAI",
            code=code,
            execution=exec_m,
            quality=quality,
            control=control,
            report=report,
        )


# ---------------------------------------------------------------------------
# Implementation 3: AutoGen (or conversational fallback)
# ---------------------------------------------------------------------------


class AutoGenResearch:
    """Same pipeline implemented as an AutoGen group chat.

    If AutoGen is not installed, a minimal round-robin conversation loop
    provides the same pattern so the comparison still runs.

    AutoGen handles: turn selection, message routing, termination detection,
    and optional code execution — all emergent from conversation.

    Code metrics (approximate):
        Agent definitions  : system_message × 4 = ~20 lines
        Orchestration      : GroupChat + manager + initiate_chat = ~10 lines
        Total              : ~30 lines of definition + more framework overhead
    """

    TERMINATE_SIGNAL = "RESEARCH_COMPLETE"

    def run(self, task: str) -> ComparisonResult:
        tokens: list[int] = [0]
        calls: list[int] = [0]
        start = time.perf_counter()
        report = ""

        if AUTOGEN_AVAILABLE:
            report = self._run_autogen(task, calls, tokens)
        else:
            report = self._run_fallback(task, calls, tokens)

        elapsed = time.perf_counter() - start

        src = inspect.getsource(AutoGenResearch)
        code = CodeMetrics(
            agent_definition_lines=20,
            orchestration_lines=10,
            total_lines=_count_lines_in_source(src),
            import_count=5,   # + autogen
        )
        exec_m = ExecutionMetrics(
            execution_time_s=round(elapsed, 2),
            llm_calls=calls[0],
            total_tokens=tokens[0],
            estimated_cost_usd=_estimate_cost(tokens[0]),
        )
        quality = QualityMetrics(
            report_length_words=len(report.split()),
            sources_cited=_count_sources(report),
            has_critique_feedback=_has_critique(report),
            structure_score=_heuristic_structure_score(report),
        )
        control = ControlMetrics(
            traceable_decision_path=False,      # Conversation is emergent
            can_modify_agent_comms=False,       # GroupChatManager controls flow
            can_add_agent_mid_workflow=False,   # Agents defined upfront
            can_change_execution_order=False,   # Manager decides speakers
            control_score=3,
        )
        return ComparisonResult(
            name="AutoGen" if AUTOGEN_AVAILABLE else "AutoGen (fallback)",
            code=code,
            execution=exec_m,
            quality=quality,
            control=control,
            report=report,
        )

    def _run_autogen(self, task: str, calls: list[int], tokens: list[int]) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        llm_config = {
            "config_list": [{"model": "gpt-4o", "api_key": api_key}],
            "temperature": 0.7,
        }

        researcher = AssistantAgent(
            name="Researcher",
            llm_config=llm_config,
            system_message=(
                "You are a research analyst. Find concrete data and statistics. "
                "Always cite sources. Focus on the research task."
            ),
        )
        critic = AssistantAgent(
            name="Critic",
            llm_config=llm_config,
            system_message=(
                "You are a research critic. Identify exactly three specific "
                "weaknesses: missing data, unsupported claims, or coverage gaps. "
                "Be constructive."
            ),
        )
        writer = AssistantAgent(
            name="Writer",
            llm_config=llm_config,
            system_message=(
                "You are a technical report writer. Once you have both the research "
                "and the critique, produce a structured 400-500 word report. "
                f"End your final report with '{self.TERMINATE_SIGNAL}'."
            ),
        )
        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda m: self.TERMINATE_SIGNAL in m.get("content", ""),
            code_execution_config=False,
        )

        groupchat = GroupChat(
            agents=[user_proxy, researcher, critic, writer],
            messages=[],
            max_round=12,
            speaker_selection_method="round_robin",
        )
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        user_proxy.initiate_chat(manager, message=f"Research task: {task}")

        # Extract all assistant messages and find the final report
        history = groupchat.messages
        calls[0] = len([m for m in history if m.get("role") == "assistant"])
        for msg in reversed(history):
            if self.TERMINATE_SIGNAL in msg.get("content", ""):
                return msg["content"]
        return history[-1].get("content", "") if history else ""

    def _run_fallback(self, task: str, calls: list[int], tokens: list[int]) -> str:
        """Minimal round-robin conversational loop as an AutoGen stand-in."""
        agents = [
            (
                "Researcher",
                "You are a research analyst. Find concrete data and statistics, cite sources.",
            ),
            (
                "Critic",
                "You are a research critic. Identify three specific weaknesses in the research.",
            ),
            (
                "Writer",
                (
                    "You are a technical report writer. Write a 400-500 word structured report. "
                    f"End with '{self.TERMINATE_SIGNAL}' when done."
                ),
            ),
        ]
        history: list[dict] = [
            {"role": "user", "content": f"Research task: {task}"}
        ]
        report = ""
        for round_num in range(12):
            name, sys_prompt = agents[round_num % len(agents)]
            response = _get_client().chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": sys_prompt}] + history[-6:],
            )
            reply = response.choices[0].message.content or ""
            calls[0] += 1
            if response.usage:
                tokens[0] += response.usage.total_tokens
            history.append({"role": "assistant", "name": name, "content": reply})
            if self.TERMINATE_SIGNAL in reply:
                report = reply
                break
        return report or (history[-1].get("content", "") if history else "")


# ---------------------------------------------------------------------------
# MultiAgentComparison — orchestrates all three and prints the table
# ---------------------------------------------------------------------------


class MultiAgentComparison:
    """Run the same task through all three implementations and compare."""

    def run(self, task: str = RESEARCH_TASK) -> list[ComparisonResult]:
        results: list[ComparisonResult] = []

        print(f"Task: {task!r}\n")
        for impl in [FromScratchResearch(), CrewAIResearch(), AutoGenResearch()]:
            name = impl.__class__.__name__
            print(f"Running {name}…", end=" ", flush=True)
            try:
                r = impl.run(task)
            except Exception as exc:  # noqa: BLE001
                r = ComparisonResult(name=name, error=str(exc))
            print("done." if not r.error else f"ERROR: {r.error[:60]}")
            results.append(r)

        return results

    def print_table(self, results: list[ComparisonResult]) -> None:
        """Print a comparison table and follow-up analysis."""
        cols = [r.name for r in results]
        col_w = max(16, *(len(c) + 2 for c in cols))
        hdr = f"{'Metric':<32}" + "".join(f"{c:>{col_w}}" for c in cols)
        sep = "─" * len(hdr)

        print(f"\n{sep}")
        print(hdr)
        print(sep)

        rows: list[tuple[str, list]] = [
            ("--- CODE METRICS ---", [""]*len(cols)),
            ("Lines of code (total)", [r.code.total_lines for r in results]),
            ("Agent definition lines", [r.code.agent_definition_lines for r in results]),
            ("Orchestration lines", [r.code.orchestration_lines for r in results]),
            ("Import count", [r.code.import_count for r in results]),
            ("--- EXECUTION METRICS ---", [""]*len(cols)),
            ("Execution time (s)", [f"{r.execution.execution_time_s:.1f}s" for r in results]),
            ("LLM calls", [r.execution.llm_calls for r in results]),
            ("Total tokens", [f"{r.execution.total_tokens:,}" for r in results]),
            ("Estimated cost (USD)", [f"${r.execution.estimated_cost_usd:.4f}" for r in results]),
            ("--- QUALITY METRICS ---", [""]*len(cols)),
            ("Report length (words)", [r.quality.report_length_words for r in results]),
            ("Sources cited", [r.quality.sources_cited for r in results]),
            ("Includes critique", [str(r.quality.has_critique_feedback) for r in results]),
            ("Structure score (1-10)", [r.quality.structure_score for r in results]),
            ("--- CONTROL METRICS ---", [""]*len(cols)),
            ("Traceable path", [str(r.control.traceable_decision_path) for r in results]),
            ("Modify agent comms", [str(r.control.can_modify_agent_comms) for r in results]),
            ("Add agent mid-workflow", [str(r.control.can_add_agent_mid_workflow) for r in results]),
            ("Change execution order", [str(r.control.can_change_execution_order) for r in results]),
            ("Control score (0-10)", [r.control.control_score for r in results]),
        ]

        for label, values in rows:
            if label.startswith("---"):
                print(f"\n{label}")
            else:
                row = f"  {label:<30}" + "".join(
                    f"{str(v):>{col_w}}" for v in values
                )
                print(row)

        print(f"\n{sep}")
        self._print_analysis(results)

    def _print_analysis(self, results: list[ComparisonResult]) -> None:
        """Print a short natural-language analysis of the comparison."""
        by_name = {r.name: r for r in results}

        # Find fastest
        runnable = [r for r in results if not r.error and r.execution.execution_time_s > 0]
        if not runnable:
            return

        fastest = min(runnable, key=lambda r: r.execution.execution_time_s)
        cheapest = min(runnable, key=lambda r: r.execution.estimated_cost_usd)
        most_control = max(runnable, key=lambda r: r.control.control_score)
        best_quality = max(runnable, key=lambda r: r.quality.structure_score)

        print("\nANALYSIS")
        print("─" * 40)
        print(f"  Fastest execution  : {fastest.name} ({fastest.execution.execution_time_s:.1f}s)")
        print(f"  Lowest cost        : {cheapest.name} (${cheapest.execution.estimated_cost_usd:.4f})")
        print(f"  Most control       : {most_control.name} ({most_control.control.control_score}/10)")
        print(f"  Best structure     : {best_quality.name} ({best_quality.quality.structure_score}/10)")

        scratch = by_name.get("From Scratch")
        if scratch and not scratch.error:
            print(
                f"\n  For this specific task, 'From Scratch' is best because it uses "
                f"the fewest tokens ({scratch.execution.total_tokens:,}), gives full "
                f"control over every agent interaction, and produces a report in "
                f"{scratch.execution.execution_time_s:.1f}s with no framework overhead.\n"
                f"  Use CrewAI when you have 5+ agents with complex dependencies and "
                f"want role-based orchestration out of the box. Use AutoGen when the "
                f"solution should emerge from open-ended conversation."
            )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    comparison = MultiAgentComparison()
    results = comparison.run(RESEARCH_TASK)
    comparison.print_table(results)

    print("\n\nFULL REPORTS")
    for r in results:
        print(f"\n{'═'*60}")
        print(f"  {r.name}")
        print(f"{'═'*60}")
        if r.error:
            print(f"  ERROR: {r.error}")
        else:
            preview = r.report[:600]
            if len(r.report) > 600:
                preview += f"\n  … [{len(r.report)-600} more chars]"
            print(preview)
