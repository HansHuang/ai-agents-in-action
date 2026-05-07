"""Multi-agent research-and-writing workflow built with LangGraph.

Workflow:
    START → research → fact_check → writer → editor
                                              ↓ (needs revision?)
                                           writer  ← loop
                                              ↓ (approved)
                                            END

Each agent is a specialised LLM node with specific instructions.
The editor decides whether to approve the report or send it back
for revision (up to MAX_REVISION_CYCLES times).

Run:
    python langgraph_multi_agent.py

See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Annotated, Optional

# ---------------------------------------------------------------------------
# Optional LangGraph imports (fail gracefully)
# ---------------------------------------------------------------------------

try:
    from typing import TypedDict

    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

MAX_REVISION_CYCLES = 3

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict):
    """State shared across all nodes in the research-and-writing workflow."""

    topic: str
    research_notes: str          # Output from research agent
    verified_facts: str          # Output from fact-checker
    draft_report: str            # Output from writer
    editor_feedback: str         # Output from editor (if revision needed)
    approved: bool               # True when editor approves
    revision_count: int          # Number of write→edit cycles
    sources: list[str]           # Accumulated source references
    workflow_trace: list[dict]   # Step-by-step execution log


# ---------------------------------------------------------------------------
# Mock tool: web search
# ---------------------------------------------------------------------------

_SEARCH_MOCK: dict[str, list[dict]] = {
    "default": [
        {
            "title": "AI in Software Engineering 2026",
            "snippet": (
                "By 2026, AI coding assistants handle ~40% of routine code generation. "
                "GitHub Copilot, Cursor, and similar tools are used daily by 60% of professional developers. "
                "AI-assisted code review catches 35% more bugs than manual review alone."
            ),
            "url": "https://example.com/ai-software-2026",
        },
        {
            "title": "LLM-Driven Development Practices",
            "snippet": (
                "Test-driven development has evolved into prompt-driven development (PDD). "
                "Teams use LLMs to generate test cases, refactor legacy code, "
                "and produce API documentation automatically."
            ),
            "url": "https://example.com/pdd-2026",
        },
        {
            "title": "AI Agent Adoption in Engineering Teams",
            "snippet": (
                "Multi-agent systems now handle CI/CD pipeline configuration, "
                "security scanning, and performance profiling. "
                "84% of Fortune 500 tech teams report using some form of AI agent in their workflow."
            ),
            "url": "https://example.com/agent-adoption",
        },
    ]
}


def mock_web_search(query: str) -> list[dict]:
    """Simulate a web search. Returns mock results for demo purposes."""
    return _SEARCH_MOCK.get(query, _SEARCH_MOCK["default"])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """Structured output from :meth:`ResearchWritingWorkflow.run`."""

    report: str
    sources: list[str]
    revisions: int
    workflow_trace: list[dict]
    elapsed_ms: float


# ---------------------------------------------------------------------------
# ResearchWritingWorkflow
# ---------------------------------------------------------------------------


class ResearchWritingWorkflow:
    """Multi-agent research-and-writing workflow.

    Four specialised agents collaborate via LangGraph StateGraph:
      1. Research Agent   — gathers information, cites sources
      2. Fact-Checker     — verifies claims against research notes
      3. Writer Agent     — produces a structured Markdown report
      4. Editor Agent     — reviews quality; approves or sends back for revision

    Args:
        model:               OpenAI chat model for all agents.
        max_revision_cycles: Maximum writer → editor → writer loops.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_revision_cycles: int = MAX_REVISION_CYCLES,
    ) -> None:
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required. Install with:\n"
                "  pip install langgraph langchain-openai"
            )
        self.model = model
        self.max_revision_cycles = max_revision_cycles
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.graph = self._build_workflow()

    # ------------------------------------------------------------------
    # Agent builders
    # ------------------------------------------------------------------

    def _create_research_agent(self):
        """Return an LLM callable configured as a research agent."""
        system = SystemMessage(content=(
            "You are a research agent. Your job is to gather comprehensive, "
            "accurate information on a given topic. "
            "Simulate web searches and database queries. "
            "Always cite your sources with URLs. "
            "Output your findings as structured notes with clear sections. "
            "Format: use bullet points, be factual, cite everything."
        ))

        def research(state: WorkflowState) -> WorkflowState:
            topic = state["topic"]

            # Simulate web search results
            search_results = mock_web_search(topic)
            search_context = "\n".join(
                f"- [{r['title']}]({r['url']}): {r['snippet']}"
                for r in search_results
            )

            messages = [
                system,
                HumanMessage(content=(
                    f"Research this topic thoroughly: {topic!r}\n\n"
                    f"Available search results:\n{search_context}\n\n"
                    "Produce detailed research notes with key facts, statistics, "
                    "and source citations."
                )),
            ]
            response = self.llm.invoke(messages)
            notes = response.content

            sources = [r["url"] for r in search_results]
            trace_entry = {
                "step": "research",
                "agent": "ResearchAgent",
                "summary": f"Gathered {len(search_results)} sources on: {topic}",
                "output_preview": notes[:200] + "...",
            }

            return {
                **state,
                "research_notes": notes,
                "sources": state["sources"] + sources,
                "workflow_trace": state["workflow_trace"] + [trace_entry],
            }

        return research

    def _create_fact_checker(self):
        """Return an LLM callable configured as a fact-checker."""
        system = SystemMessage(content=(
            "You are a fact-checker. Your job is to verify claims in research notes. "
            "Cross-check each major claim. Flag any unsupported assertions. "
            "Output a verified facts summary, noting what was confirmed and what was uncertain."
        ))

        def fact_check(state: WorkflowState) -> WorkflowState:
            messages = [
                system,
                HumanMessage(content=(
                    f"Fact-check these research notes about: {state['topic']!r}\n\n"
                    f"{state['research_notes']}\n\n"
                    "Produce a verified facts summary. "
                    "Format: list confirmed facts, list uncertain claims."
                )),
            ]
            response = self.llm.invoke(messages)
            verified = response.content

            trace_entry = {
                "step": "fact_check",
                "agent": "FactChecker",
                "summary": "Verified research notes",
                "output_preview": verified[:200] + "...",
            }

            return {
                **state,
                "verified_facts": verified,
                "workflow_trace": state["workflow_trace"] + [trace_entry],
            }

        return fact_check

    def _create_writer(self):
        """Return an LLM callable configured as a writer."""
        system = SystemMessage(content=(
            "You are a technical writer. Produce clear, well-structured reports. "
            "Use Markdown formatting: headers, bullet points, and code blocks where relevant. "
            "Each factual claim must reference the research. "
            "If given editor feedback, address every point in your revision."
        ))

        def write(state: WorkflowState) -> WorkflowState:
            feedback_section = ""
            if state.get("editor_feedback"):
                feedback_section = (
                    f"\n\nEditor feedback to address:\n{state['editor_feedback']}\n"
                    "Please revise the report to address ALL points above."
                )
            revision_note = (
                f" (Revision {state['revision_count'] + 1})"
                if state["revision_count"] > 0
                else ""
            )

            messages = [
                system,
                HumanMessage(content=(
                    f"Write a comprehensive report{revision_note} on: {state['topic']!r}\n\n"
                    f"Research notes:\n{state['research_notes']}\n\n"
                    f"Verified facts:\n{state['verified_facts']}"
                    f"{feedback_section}\n\n"
                    "Produce a complete, publication-ready Markdown report."
                )),
            ]
            response = self.llm.invoke(messages)
            draft = response.content

            is_revision = state["revision_count"] > 0
            trace_entry = {
                "step": "write",
                "agent": "Writer",
                "summary": (
                    f"Wrote revision {state['revision_count'] + 1}"
                    if is_revision
                    else "Wrote initial draft"
                ),
                "output_preview": draft[:200] + "...",
            }

            return {
                **state,
                "draft_report": draft,
                "revision_count": state["revision_count"] + 1,
                "workflow_trace": state["workflow_trace"] + [trace_entry],
            }

        return write

    def _create_editor(self):
        """Return an LLM callable configured as an editor."""
        system = SystemMessage(content=(
            "You are a senior editor. Review reports for clarity, accuracy, completeness, "
            "and professional quality. "
            "If the report is ready to publish, respond with exactly: APPROVED\n"
            "followed by a brief note of what's good. "
            "If it needs work, respond with exactly: REVISION NEEDED\n"
            "followed by specific, actionable feedback. "
            "Be demanding but fair — only approve reports that meet publication standards."
        ))

        def edit(state: WorkflowState) -> WorkflowState:
            messages = [
                system,
                HumanMessage(content=(
                    f"Review this report about: {state['topic']!r}\n\n"
                    f"{state['draft_report']}\n\n"
                    f"This is revision {state['revision_count']} of {self.max_revision_cycles} allowed."
                )),
            ]
            response = self.llm.invoke(messages)
            feedback = response.content

            approved = feedback.strip().upper().startswith("APPROVED")

            # Force approval when max revisions reached
            if state["revision_count"] >= self.max_revision_cycles:
                approved = True
                feedback = f"APPROVED (max revisions reached)\n{feedback}"

            trace_entry = {
                "step": "edit",
                "agent": "Editor",
                "summary": "Approved report" if approved else "Requested revision",
                "output_preview": feedback[:200] + "...",
            }

            return {
                **state,
                "editor_feedback": feedback if not approved else "",
                "approved": approved,
                "workflow_trace": state["workflow_trace"] + [trace_entry],
            }

        return edit

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_workflow(self):
        """Declare the research-and-writing graph."""
        research_fn = self._create_research_agent()
        fact_check_fn = self._create_fact_checker()
        writer_fn = self._create_writer()
        editor_fn = self._create_editor()

        workflow = StateGraph(WorkflowState)

        workflow.add_node("research", research_fn)
        workflow.add_node("fact_check", fact_check_fn)
        workflow.add_node("writer", writer_fn)
        workflow.add_node("editor", editor_fn)

        workflow.set_entry_point("research")
        workflow.add_edge("research", "fact_check")
        workflow.add_edge("fact_check", "writer")

        workflow.add_conditional_edges(
            "editor",
            lambda state: "end" if state["approved"] else "writer",
            {"writer": "writer", "end": END},
        )
        workflow.add_edge("writer", "editor")

        return workflow.compile()

    @staticmethod
    def visualize() -> str:
        """Return an ASCII diagram of the workflow."""
        return """\
Multi-Agent Research-and-Writing Workflow
──────────────────────────────────────────

┌─────────┐
│  START  │
└────┬────┘
     ▼
┌──────────┐
│ research │  Gathers information, cites sources
└────┬─────┘
     ▼
┌────────────┐
│ fact_check │  Verifies claims
└─────┬──────┘
      ▼
┌────────┐
│ writer │ ◀──────────────────────────────────────────────────┐
└────┬───┘                                                     │
     ▼                                                         │
┌────────┐    REVISION NEEDED (revision_count < max)          │
│ editor │ ──────────────────────────────────────────────────▶ ┘
└────┬───┘
     │ APPROVED (or max revisions reached)
     ▼
┌─────────┐
│   END   │
└─────────┘
"""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, topic: str) -> WorkflowResult:
        """Research and write a report on *topic*.

        Args:
            topic: The research topic.

        Returns:
            :class:`WorkflowResult` with the final report, sources,
            revision count, and full workflow trace.
        """
        start = time.monotonic()

        initial_state: WorkflowState = {
            "topic": topic,
            "research_notes": "",
            "verified_facts": "",
            "draft_report": "",
            "editor_feedback": "",
            "approved": False,
            "revision_count": 0,
            "sources": [],
            "workflow_trace": [],
        }

        final_state = self.graph.invoke(initial_state)
        elapsed = (time.monotonic() - start) * 1000

        return WorkflowResult(
            report=final_state["draft_report"],
            sources=final_state["sources"],
            revisions=final_state["revision_count"],
            workflow_trace=final_state["workflow_trace"],
            elapsed_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


def print_workflow_trace(trace: list[dict]) -> None:
    """Pretty-print the workflow trace."""
    print("\nWorkflow Trace:")
    print("─" * 60)
    for i, entry in enumerate(trace, 1):
        agent = entry.get("agent", "unknown")
        summary = entry.get("summary", "")
        preview = entry.get("output_preview", "")
        print(f"  Step {i} [{agent}]: {summary}")
        print(f"    Preview: {preview[:100]}...")
        print()


def main() -> None:
    """Run the demo on a real research topic."""
    print(ResearchWritingWorkflow.visualize())

    if not LANGGRAPH_AVAILABLE:
        print("LangGraph is not installed. Install with:")
        print("  pip install langgraph langchain-openai")
        return

    topic = "The impact of AI on software engineering in 2026"
    print(f"Topic: {topic!r}\n")
    print("Running multi-agent workflow...")
    print("(This makes real LLM calls — ensure OPENAI_API_KEY is set)\n")

    workflow = ResearchWritingWorkflow(max_revision_cycles=2)
    result = workflow.run(topic)

    print_workflow_trace(result.workflow_trace)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.report)
    print()
    print(f"Sources ({len(result.sources)}):")
    for src in result.sources:
        print(f"  • {src}")
    print(f"\nRevisions: {result.revisions}")
    print(f"Total time: {result.elapsed_ms:.0f} ms")


if __name__ == "__main__":
    main()
