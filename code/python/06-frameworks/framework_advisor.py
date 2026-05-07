"""Interactive framework advisor: answer questions, get a tailored recommendation.

Asks eight questions about your project and generates:
    • A primary framework recommendation with rationale
    • Secondary integrations framework (for vector DBs, loaders, etc.)
    • Frameworks to avoid and why
    • Migration path as the project evolves
    • ASCII architecture diagram for the recommended approach

Run:
    python framework_advisor.py             # interactive mode
    python framework_advisor.py --preset expert   # non-interactive demo

See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Question definitions
# ---------------------------------------------------------------------------

_QUESTIONS: list[dict] = [
    {
        "id": "team_size",
        "question": "How many developers on your team?",
        "type": "int",
        "valid_range": (1, 1000),
    },
    {
        "id": "ai_experience",
        "question": "Team's AI/LLM experience level?",
        "options": ["beginner", "intermediate", "expert"],
    },
    {
        "id": "use_case",
        "question": "Primary use case?",
        "options": [
            "simple_rag",
            "complex_workflows",
            "multi_agent",
            "chatbot",
            "code_generation",
            "data_extraction",
        ],
    },
    {
        "id": "scale",
        "question": "Expected scale (requests/day)?",
        "options": ["<100", "100-1000", "1000-10000", "10000+"],
    },
    {
        "id": "lifetime",
        "question": "Expected project lifetime?",
        "options": [
            "prototype (weeks)",
            "medium (months)",
            "long-term (years)",
        ],
    },
    {
        "id": "streaming",
        "question": "Is real-time streaming to a UI critical?",
        "type": "bool",
    },
    {
        "id": "multi_provider",
        "question": "Will you use multiple LLM providers (e.g. OpenAI + Anthropic)?",
        "type": "bool",
    },
    {
        "id": "existing_stack",
        "question": "Primary tech stack?",
        "options": ["python", "typescript", "go", "other"],
    },
]

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FrameworkRecommendation:
    """Complete recommendation produced by :class:`FrameworkAdvisor.recommend`."""

    primary: str
    """One of: 'from_scratch', 'langchain', 'langgraph', 'crewai', 'vercel_ai'."""

    for_integrations: str
    """What to use for vector DBs, document loaders, observability."""

    avoid: list[str]
    """Frameworks to skip for this use case."""

    migration_path: str
    """Guidance on how the architecture should evolve over time."""

    explanation: str
    """Full narrative reasoning behind the recommendation."""

    architecture_diagram: str
    """ASCII art diagram tailored to the recommendation."""


# ---------------------------------------------------------------------------
# Preset answers for non-interactive demos
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "expert": {
        "team_size": 4,
        "ai_experience": "expert",
        "use_case": "simple_rag",
        "scale": "10000+",
        "lifetime": "long-term (years)",
        "streaming": False,
        "multi_provider": True,
        "existing_stack": "python",
    },
    "beginner": {
        "team_size": 2,
        "ai_experience": "beginner",
        "use_case": "simple_rag",
        "scale": "<100",
        "lifetime": "prototype (weeks)",
        "streaming": False,
        "multi_provider": False,
        "existing_stack": "python",
    },
    "streaming_ts": {
        "team_size": 3,
        "ai_experience": "intermediate",
        "use_case": "chatbot",
        "scale": "100-1000",
        "lifetime": "medium (months)",
        "streaming": True,
        "multi_provider": True,
        "existing_stack": "typescript",
    },
    "multi_agent": {
        "team_size": 2,
        "ai_experience": "intermediate",
        "use_case": "multi_agent",
        "scale": "<100",
        "lifetime": "prototype (weeks)",
        "streaming": False,
        "multi_provider": False,
        "existing_stack": "python",
    },
    "complex_workflow": {
        "team_size": 5,
        "ai_experience": "intermediate",
        "use_case": "complex_workflows",
        "scale": "1000-10000",
        "lifetime": "long-term (years)",
        "streaming": False,
        "multi_provider": False,
        "existing_stack": "python",
    },
}

# ---------------------------------------------------------------------------
# ASCII architecture diagrams
# ---------------------------------------------------------------------------

_DIAGRAMS: dict[str, str] = {
    "from_scratch": textwrap.dedent("""\
        ┌────────────────────────────────────────────────────┐
        │                FROM-SCRATCH AGENT                  │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ YOUR CODE (everything)                       │ │
        │  │  • Agent orchestration loop                  │ │
        │  │  • Tool design and execution                 │ │
        │  │  • Context assembly and management           │ │
        │  │  • Memory and state                          │ │
        │  │  • Error handling                            │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ THIN HELPERS (optional, replaceable)         │ │
        │  │  • openai SDK  (direct API calls)            │ │
        │  │  • numpy       (cosine similarity)           │ │
        │  │  • httpx       (custom HTTP if needed)       │ │
        │  └──────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────┘
        Dependency count: 2-5  |  Debugging: direct stack trace
        Best for: production systems, long-term projects, experts
    """),

    "langchain": textwrap.dedent("""\
        ┌────────────────────────────────────────────────────┐
        │               LANGCHAIN AGENT                      │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ LANGCHAIN (chains, retrievers, prompts)      │ │
        │  │  • create_retrieval_chain()                  │ │
        │  │  • create_stuff_documents_chain()            │ │
        │  │  • FAISS / Chroma vector store               │ │
        │  │  • 700+ document loaders                     │ │
        │  │  • LangSmith tracing                         │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ YOUR CODE (customisation)                    │ │
        │  │  • Domain-specific prompts                   │ │
        │  │  • Business logic around results             │ │
        │  └──────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────┘
        Dependency count: 12-20  |  Debugging: 7-layer stack
        Best for: prototypes, RAG + many integrations, beginners
    """),

    "langgraph": textwrap.dedent("""\
        ┌────────────────────────────────────────────────────┐
        │               LANGGRAPH AGENT                      │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ LANGGRAPH StateGraph                         │ │
        │  │  • Node: retrieve → augment → generate       │ │
        │  │  • Conditional edges for branching           │ │
        │  │  • Built-in checkpointing                    │ │
        │  │  • LangGraph Studio visualisation            │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ YOUR CODE (node implementations)             │ │
        │  │  • Each node is a plain Python function      │ │
        │  │  • State schema is yours to define           │ │
        │  └──────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────┘
        Dependency count: 14-22  |  Debugging: per-node state
        Best for: complex workflows, branching logic, persistence
    """),

    "crewai": textwrap.dedent("""\
        ┌────────────────────────────────────────────────────┐
        │                 CREWAI AGENT                       │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ CREWAI Crew                                  │ │
        │  │  • Agent(role="researcher", ...)             │ │
        │  │  • Agent(role="writer", ...)                 │ │
        │  │  • Task(description="...", agent=...)        │ │
        │  │  • Crew(agents=[...], tasks=[...])           │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ YOUR CODE                                    │ │
        │  │  • Role descriptions and goals               │ │
        │  │  • Task definitions and expected outputs     │ │
        │  └──────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────┘
        Dependency count: 8-15  |  Debugging: moderate
        Best for: multi-agent prototypes, role-based thinking
    """),

    "vercel_ai": textwrap.dedent("""\
        ┌────────────────────────────────────────────────────┐
        │             VERCEL AI SDK AGENT                    │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ VERCEL AI SDK                                │ │
        │  │  • streamText() / generateText()             │ │
        │  │  • useChat() React hook                      │ │
        │  │  • Unified: OpenAI / Anthropic / Google      │ │
        │  │  • Edge-runtime compatible                   │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ YOUR CODE                                    │ │
        │  │  • Next.js API routes / app routes           │ │
        │  │  • Tool definitions                          │ │
        │  │  • RAG retrieval logic                       │ │
        │  └──────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────┘
        Dependency count: 3-6  |  Debugging: moderate
        Best for: full-stack TypeScript, streaming UX, multi-provider
    """),

    "hybrid": textwrap.dedent("""\
        ┌────────────────────────────────────────────────────┐
        │             HYBRID AGENT (recommended)             │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ YOUR CODE (the important parts)              │ │
        │  │  • Agent orchestration loop                  │ │
        │  │  • Context assembly and management           │ │
        │  │  • Memory and state                          │ │
        │  │  • Error handling and fallback logic         │ │
        │  └──────────────────────────────────────────────┘ │
        │                                                    │
        │  ┌──────────────────────────────────────────────┐ │
        │  │ FRAMEWORK (commodity parts only)             │ │
        │  │  • Vector DB connectors  (LangChain)         │ │
        │  │  • Document loaders      (LangChain)         │ │
        │  │  • Streaming             (Vercel AI SDK)     │ │
        │  │  • Observability         (LangSmith / Arize) │ │
        │  └──────────────────────────────────────────────┘ │
        └────────────────────────────────────────────────────┘
        Rule: Your agent's BRAIN is custom. Inputs/outputs can use frameworks.
    """),
}


# ---------------------------------------------------------------------------
# FrameworkAdvisor
# ---------------------------------------------------------------------------


class FrameworkAdvisor:
    """Interactive questionnaire → personalised framework recommendation."""

    def __init__(self) -> None:
        self._questions = _QUESTIONS

    # ------------------------------------------------------------------
    # Input collection
    # ------------------------------------------------------------------

    def ask_all(self) -> dict:
        """Ask every question interactively and return the answers dict."""
        print("\n" + "=" * 60)
        print("  FRAMEWORK ADVISOR — answer 8 questions")
        print("=" * 60 + "\n")

        answers: dict = {}
        for q in self._questions:
            answers[q["id"]] = self._ask_one(q)
        return answers

    def _ask_one(self, q: dict):
        """Ask a single question and validate the answer."""
        print(f"  {q['question']}")

        if q.get("type") == "int":
            lo, hi = q.get("valid_range", (1, 10000))
            while True:
                raw = input(f"    Enter a number ({lo}-{hi}): ").strip()
                try:
                    val = int(raw)
                    if lo <= val <= hi:
                        return val
                    print(f"    Please enter a number between {lo} and {hi}.")
                except ValueError:
                    print("    That doesn't look like a number. Try again.")

        elif q.get("type") == "bool":
            while True:
                raw = input("    [y/n]: ").strip().lower()
                if raw in {"y", "yes", "true", "1"}:
                    return True
                if raw in {"n", "no", "false", "0"}:
                    return False
                print("    Please enter y or n.")

        else:
            options: list[str] = q["options"]
            for i, opt in enumerate(options, 1):
                print(f"    {i}. {opt}")
            while True:
                raw = input(f"    Choose 1-{len(options)}: ").strip()
                try:
                    idx = int(raw) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                except ValueError:
                    pass
                print(f"    Please enter a number between 1 and {len(options)}.")

    # ------------------------------------------------------------------
    # Recommendation engine
    # ------------------------------------------------------------------

    def recommend(self, answers: dict) -> FrameworkRecommendation:
        """Derive a recommendation from *answers*.

        Rules are applied in priority order. The first matching rule wins.
        """
        stack = answers.get("existing_stack", "python")
        experience = answers.get("ai_experience", "intermediate")
        use_case = answers.get("use_case", "simple_rag")
        lifetime = answers.get("lifetime", "prototype (weeks)")
        streaming = answers.get("streaming", False)
        multi_provider = answers.get("multi_provider", False)
        scale = answers.get("scale", "<100")
        team_size = answers.get("team_size", 1)

        is_long_term = lifetime == "long-term (years)"
        is_prototype = "prototype" in lifetime
        is_ts = stack == "typescript"
        is_go = stack == "go"
        is_expert = experience == "expert"
        is_beginner = experience == "beginner"

        # ---- Rule set (first match wins) -----------------------------------

        # Go: no major frameworks, custom all the way
        if is_go:
            return self._rec_go(answers)

        # TypeScript + streaming → Vercel AI SDK
        if is_ts and (streaming or multi_provider):
            return self._rec_vercel_ai(answers)

        # TypeScript without strong streaming need
        if is_ts:
            return self._rec_vercel_ai(answers)

        # Multi-agent prototype → CrewAI
        if use_case == "multi_agent" and is_prototype:
            return self._rec_crewai(answers)

        # Complex workflows in production → LangGraph
        if use_case == "complex_workflows" and not is_prototype:
            return self._rec_langgraph(answers)

        # Expert + long-term Python → from scratch (hybrid for integrations)
        if is_expert and is_long_term:
            return self._rec_from_scratch(answers)

        # Beginner + prototype → LangChain
        if is_beginner and is_prototype:
            return self._rec_langchain(answers)

        # High scale + long-term → from scratch (full control)
        if scale == "10000+" and is_long_term:
            return self._rec_from_scratch(answers)

        # Default: hybrid approach
        return self._rec_hybrid(answers)

    # ------------------------------------------------------------------
    # Individual recommendation constructors
    # ------------------------------------------------------------------

    def _rec_from_scratch(self, a: dict) -> FrameworkRecommendation:
        return FrameworkRecommendation(
            primary="from_scratch",
            for_integrations="langchain (for vector DB connectors only)",
            avoid=["crewai (premature)", "full langchain (over-engineered)"],
            migration_path=(
                "Phase 1: Build core agent from scratch — done.\n"
                "Phase 2: Add LangChain for specific integrations (e.g. new vector DB).\n"
                "Phase 3: Add LangSmith / Arize for observability.\n"
                "Phase 4: Consider LangGraph only if workflow branching becomes complex."
            ),
            explanation=self._explain_from_scratch(a),
            architecture_diagram=_DIAGRAMS["hybrid"],
        )

    def _rec_langchain(self, a: dict) -> FrameworkRecommendation:
        return FrameworkRecommendation(
            primary="langchain",
            for_integrations="langchain (already included)",
            avoid=["langgraph (too complex for prototype)", "from_scratch (too slow to start)"],
            migration_path=(
                "Phase 1: Use LangChain chains for rapid prototyping.\n"
                "Phase 2: Extract agent loop to custom code as requirements clarify.\n"
                "Phase 3: Keep LangChain for connectors; build custom orchestration.\n"
                "Phase 4: Replace problematic LangChain components one at a time."
            ),
            explanation=self._explain_langchain(a),
            architecture_diagram=_DIAGRAMS["langchain"],
        )

    def _rec_langgraph(self, a: dict) -> FrameworkRecommendation:
        return FrameworkRecommendation(
            primary="langgraph",
            for_integrations="langchain (for vector DB and document loaders)",
            avoid=["crewai (lacks fine-grained control)", "full custom (StateGraph saves significant effort)"],
            migration_path=(
                "Phase 1: Build StateGraph with LangGraph for workflow orchestration.\n"
                "Phase 2: Implement individual nodes as custom Python functions.\n"
                "Phase 3: Add checkpointing for persistence across steps.\n"
                "Phase 4: Extract nodes that become problematic to custom code."
            ),
            explanation=self._explain_langgraph(a),
            architecture_diagram=_DIAGRAMS["langgraph"],
        )

    def _rec_crewai(self, a: dict) -> FrameworkRecommendation:
        return FrameworkRecommendation(
            primary="crewai",
            for_integrations="langchain (for document loading and vector search)",
            avoid=["langgraph (overkill for prototype)", "full custom (too slow for prototype)"],
            migration_path=(
                "Phase 1: Use CrewAI to explore multi-agent patterns quickly.\n"
                "Phase 2: Identify which agents/tasks need custom logic.\n"
                "Phase 3: Replace CrewAI orchestration with custom code for those.\n"
                "Phase 4: Evaluate LangGraph if state management becomes a pain point."
            ),
            explanation=self._explain_crewai(a),
            architecture_diagram=_DIAGRAMS["crewai"],
        )

    def _rec_vercel_ai(self, a: dict) -> FrameworkRecommendation:
        return FrameworkRecommendation(
            primary="vercel_ai",
            for_integrations="langchain.js (for document loading) or custom",
            avoid=["python langchain (wrong language)", "crewai (no TypeScript support)"],
            migration_path=(
                "Phase 1: Use Vercel AI SDK for streaming and provider abstraction.\n"
                "Phase 2: Add RAG with LangChain.js or a direct vector DB client.\n"
                "Phase 3: Build agent loop as custom TypeScript functions.\n"
                "Phase 4: Extract provider-specific code behind your own interface."
            ),
            explanation=self._explain_vercel_ai(a),
            architecture_diagram=_DIAGRAMS["vercel_ai"],
        )

    def _rec_go(self, a: dict) -> FrameworkRecommendation:
        return FrameworkRecommendation(
            primary="from_scratch",
            for_integrations="Qdrant Go SDK or Weaviate Go client for vector search",
            avoid=["langchain (no mature Go port)", "crewai (Python only)", "langgraph (Python only)"],
            migration_path=(
                "Phase 1: Build agent loop in Go with direct OpenAI SDK.\n"
                "Phase 2: Add Qdrant or Weaviate Go client for vector search.\n"
                "Phase 3: Use pgvector with database/sql if you already run Postgres.\n"
                "Phase 4: Consider LangChain4j (JVM) if you move to the JVM ecosystem."
            ),
            explanation=self._explain_go(a),
            architecture_diagram=_DIAGRAMS["from_scratch"],
        )

    def _rec_hybrid(self, a: dict) -> FrameworkRecommendation:
        return FrameworkRecommendation(
            primary="from_scratch",
            for_integrations="langchain (for commodity connectors only)",
            avoid=["full langchain ownership of agent logic"],
            migration_path=(
                "Phase 1: Build agent loop from scratch — learn the concepts.\n"
                "Phase 2: Add LangChain for specific connectors you need.\n"
                "Phase 3: Evaluate LangGraph if workflow complexity grows.\n"
                "Phase 4: Never let a framework own your agent's brain."
            ),
            explanation=self._explain_hybrid(a),
            architecture_diagram=_DIAGRAMS["hybrid"],
        )

    # ------------------------------------------------------------------
    # Explanation generators
    # ------------------------------------------------------------------

    def _explain_from_scratch(self, a: dict) -> str:
        return (
            f"Your team is expert-level with a long-term production commitment at "
            f"{a.get('scale', 'high')} scale. The highest ROI is full control: "
            f"you debug with plain stack traces, upgrade dependencies on your own "
            f"schedule, and keep the dependency footprint minimal. Use LangChain "
            f"only for vector DB connectors — a small, isolated surface that's easy "
            f"to replace if LangChain breaks a release."
        )

    def _explain_langchain(self, a: dict) -> str:
        return (
            f"Your team is getting started with LLMs and needs a working prototype "
            f"quickly. LangChain's pre-built chains reduce the learning curve and "
            f"time-to-demo. The trade-off — harder debugging and more dependencies — "
            f"is acceptable at the prototype stage. Plan to extract the agent loop "
            f"to custom code as the project matures."
        )

    def _explain_langgraph(self, a: dict) -> str:
        return (
            f"Your use case is {a.get('use_case', 'complex workflows')} and you're "
            f"building for {a.get('lifetime', 'medium')} use at "
            f"{a.get('scale', 'moderate')} scale. LangGraph's StateGraph maps "
            f"directly to branching agent logic, provides per-node state inspection, "
            f"and has built-in checkpointing. Individual nodes are plain Python "
            f"functions, so your business logic stays framework-independent."
        )

    def _explain_crewai(self, a: dict) -> str:
        return (
            f"You need multi-agent collaboration and you're in prototype mode. "
            f"CrewAI's role-based mental model (Agent + Task + Crew) is the fastest "
            f"path to a working multi-agent demo. Expect to outgrow it as you need "
            f"fine-grained control over agent communication — at that point, migrate "
            f"the orchestration to LangGraph or custom code."
        )

    def _explain_vercel_ai(self, a: dict) -> str:
        streaming_note = " Streaming is critical for your UX," if a.get("streaming") else ""
        mp_note = " and you need multi-provider flexibility," if a.get("multi_provider") else ""
        return (
            f"You're building in TypeScript.{streaming_note}{mp_note} which is "
            f"exactly what Vercel AI SDK is designed for. Its unified interface "
            f"across OpenAI, Anthropic, and Google means you can switch providers "
            f"without touching your application logic. The React hooks (useChat, "
            f"useCompletion) give you production-ready streaming UI in minutes."
        )

    def _explain_go(self, a: dict) -> str:
        return (
            f"Go has no mature, production-ready AI agent framework equivalent to "
            f"LangChain. The Go OpenAI SDK is solid. For vector search, Qdrant and "
            f"Weaviate both have official Go clients. Build the agent loop as custom "
            f"Go code — you'll get idiomatic concurrency, strong typing, and fast "
            f"compilation. This is actually an advantage: Go teams often produce "
            f"cleaner, more maintainable agent code than Python teams using frameworks."
        )

    def _explain_hybrid(self, a: dict) -> str:
        return (
            f"Your project sits in the middle of the decision matrix. The hybrid "
            f"approach gives you the best of both worlds: use LangChain for the "
            f"commodity parts (vector DB connectors, document loaders) and write "
            f"custom code for the parts that differentiate your agent. This keeps "
            f"your agent's brain under your control while avoiding the boilerplate "
            f"of writing every integration from scratch."
        )

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def explain(self, rec: FrameworkRecommendation) -> str:
        """Format the full recommendation as a printable string."""
        lines = [
            "",
            "=" * 65,
            "  FRAMEWORK RECOMMENDATION",
            "=" * 65,
            f"\n  PRIMARY:            {rec.primary.upper().replace('_', ' ')}",
            f"  FOR INTEGRATIONS:   {rec.for_integrations}",
            f"  AVOID:              {', '.join(rec.avoid)}",
            "",
            "  REASONING:",
            textwrap.fill(rec.explanation, width=62, initial_indent="  ", subsequent_indent="  "),
            "",
            "  MIGRATION PATH:",
        ]
        for step in rec.migration_path.splitlines():
            lines.append(f"    {step}")

        lines += [
            "",
            "  ARCHITECTURE:",
            "",
        ]
        for arch_line in rec.architecture_diagram.splitlines():
            lines.append(f"  {arch_line}")

        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework Advisor")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help=(
            "Use a preset answer set instead of interactive mode. "
            f"Choices: {', '.join(PRESETS.keys())}"
        ),
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Run all presets and print all recommendations (demo mode).",
    )
    args = parser.parse_args()

    advisor = FrameworkAdvisor()

    if args.all_presets:
        for name, preset in PRESETS.items():
            print(f"\n{'─' * 65}")
            print(f"  PRESET: {name.upper()}")
            print(f"  Answers: {preset}")
            rec = advisor.recommend(preset)
            print(advisor.explain(rec))
        return

    if args.preset:
        answers = PRESETS[args.preset]
        print(f"\n[Using preset '{args.preset}': {answers}]")
    else:
        answers = advisor.ask_all()

    rec = advisor.recommend(answers)
    print(advisor.explain(rec))


if __name__ == "__main__":
    main()
