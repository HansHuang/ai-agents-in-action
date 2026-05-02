"""RAG integrated as a tool in an agent loop.

The agent decides when to search the knowledge base versus answering
directly from general knowledge.  RAG is exposed as one tool among many;
the model routes each query appropriately.

See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
     docs/02-the-agent-loop/01-anatomy-of-an-agent.md
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

from rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RAG_AGENT_SYSTEM_PROMPT = """\
You are a helpful assistant with access to a company knowledge base and general knowledge.

DECISION RULES:
1. For general knowledge (math, common facts, science, coding): answer directly.
2. For company-specific information (policies, procedures, products, HR):
   ALWAYS call search_knowledge_base first.
3. If search_knowledge_base returns that it has no information, tell the user
   you couldn't find it in the knowledge base and offer to help another way.
4. Always cite the knowledge base when you use it.

EXAMPLES:
  "What's 2+2?"              → answer directly: "4"
  "What's our return policy?" → search_knowledge_base("return policy")
  "How do I reset my password?" → search_knowledge_base("password reset procedure")
  "Write a Python function"  → answer directly (coding knowledge)
  "What's the capital of France?" → answer directly (general geography)
"""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """Result of a single :meth:`RAGAgent.run` call."""

    answer: str
    used_rag: bool
    rag_queries: list[str] = field(default_factory=list)
    rag_sources: list[str] = field(default_factory=list)
    tool_calls_made: int = 0
    decision_trail: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------


class RAGAgent:
    """An agent that uses RAG as one of its tools.

    The agent model decides autonomously whether to answer directly or to
    call ``search_knowledge_base``.  The decision is observable through
    :attr:`AgentResponse.decision_trail`.

    Args:
        rag_pipeline: Ingested :class:`RAGPipeline`.
        model:        OpenAI chat model.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        model: str = "gpt-4o",
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.model = model
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._tools = self._build_tools()

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": (
                        "Search the company knowledge base for information. "
                        "Use this when the user asks about policies, procedures, "
                        "products, HR topics, or any company-specific information. "
                        "Do NOT use for general knowledge questions like math, "
                        "common facts, science, or coding."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Specific, targeted search query. "
                                    "Example: 'return policy for damaged electronics'"
                                ),
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: str) -> str:
        """Execute a tool call and return the result as a string."""
        if tool_name == "search_knowledge_base":
            args = json.loads(arguments)
            query = args.get("query", "")
            rag_response = self.rag_pipeline.query(query)
            return json.dumps({
                "answer": rag_response.answer,
                "sources": rag_response.sources,
                "scores": [round(s, 3) for s in rag_response.similarity_scores],
            })
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> AgentResponse:
        """Run the agent loop for a single user turn.

        The agent may call tools multiple times before producing a final answer.
        A maximum of 5 tool-call rounds is enforced to prevent runaway loops.

        Args:
            user_input: The user's message.

        Returns:
            :class:`AgentResponse` with answer and decision trail.
        """
        messages = [
            {"role": "system", "content": RAG_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        used_rag = False
        rag_queries: list[str] = []
        rag_sources: list[str] = []
        tool_calls_made = 0
        decision_trail: list[str] = []

        decision_trail.append(f"USER: {user_input}")

        max_rounds = 5
        for _ in range(max_rounds):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._tools,
                tool_choice="auto",
            )

            choice = response.choices[0]
            message = choice.message

            # Append the assistant turn (with any tool_calls)
            messages.append(message.model_dump(exclude_unset=True))

            # No tool calls → final answer
            if not message.tool_calls:
                decision_trail.append(
                    f"AGENT: answered {'using RAG' if used_rag else 'directly (no RAG)'}"
                )
                return AgentResponse(
                    answer=message.content or "",
                    used_rag=used_rag,
                    rag_queries=rag_queries,
                    rag_sources=rag_sources,
                    tool_calls_made=tool_calls_made,
                    decision_trail=decision_trail,
                )

            # Execute each tool call
            for tc in message.tool_calls:
                tool_name = tc.function.name
                args_raw = tc.function.arguments

                # Parse query for trail
                try:
                    args_parsed = json.loads(args_raw)
                    query_str = args_parsed.get("query", args_raw)
                except (json.JSONDecodeError, AttributeError):
                    query_str = args_raw

                decision_trail.append(f"TOOL CALL: {tool_name}({query_str!r})")
                result = self._execute_tool(tool_name, args_raw)
                decision_trail.append(f"TOOL RESULT: {result[:120]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                tool_calls_made += 1
                if tool_name == "search_knowledge_base":
                    used_rag = True
                    rag_queries.append(query_str)
                    try:
                        result_data = json.loads(result)
                        rag_sources.extend(result_data.get("sources", []))
                    except json.JSONDecodeError:
                        pass

        # Safety: return last assistant content if loop exhausted
        last_content = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"),
            "Agent loop limit reached without a final answer.",
        )
        decision_trail.append("AGENT: loop limit reached")
        return AgentResponse(
            answer=last_content or "Agent loop limit reached.",
            used_rag=used_rag,
            rag_queries=rag_queries,
            rag_sources=rag_sources,
            tool_calls_made=tool_calls_made,
            decision_trail=decision_trail,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    from embedding_generator import EmbeddingGenerator
    from simple_vector_store import SimpleVectorStore

    print("=" * 70)
    print("RAG AGENT DEMO")
    print("=" * 70)

    embedder = EmbeddingGenerator(model="text-embedding-3-small")
    vector_store = SimpleVectorStore()
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        model="gpt-4o",
        chunk_size=200,
        overlap=40,
        similarity_threshold=0.5,
    )

    COMPANY_DOCS = {
        "vacation-policy.md": """\
# Vacation Policy
Full-time employees accrue 15 days of paid vacation per year.
Vacation must be approved by your manager at least 2 weeks in advance.
Unused vacation of up to 5 days may be rolled over to the following year.
Vacation payout is available upon termination.
""",
        "expense-policy.md": """\
# Expense Reporting
Employees must submit expense reports within 30 days of the expense.
Receipts are required for all expenses over $25.
Meals: up to $75/person for client entertainment; $30/person for internal meals.
Travel: economy class for flights under 6 hours; business class allowed for longer.
Submit reports via the Finance portal at expenses.internal.example.com.
""",
        "it-support.md": """\
# IT Support Procedures
To reset your password: visit https://password.internal.example.com or call IT at ext. 4357.
For new software requests, submit a ticket at helpdesk.internal.example.com.
Approved hardware requests are fulfilled within 5 business days.
For security incidents, email security@internal.example.com immediately.
""",
    }

    for source, text in COMPANY_DOCS.items():
        n = pipeline.ingest_text(text, metadata={"source": source})
        print(f"Ingested {source}: {n} chunks")

    agent = RAGAgent(rag_pipeline=pipeline, model="gpt-4o")

    QUERIES = [
        ("1. General knowledge — should NOT use RAG", "What is Python?"),
        ("2. Company policy — SHOULD use RAG", "What is our vacation policy?"),
        ("3. Math — should NOT use RAG", "Calculate 15% tip on $45."),
        ("4. Company procedure — SHOULD use RAG", "How do I submit an expense report?"),
    ]

    for label, question in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"Q: {question}")
        print("-" * 70)

        response = agent.run(question)

        print("Decision trail:")
        for step in response.decision_trail:
            print(f"  → {step}")

        print(f"\nAnswer:\n{response.answer}")
        print(f"\nUsed RAG: {response.used_rag}")
        if response.rag_queries:
            print(f"RAG queries: {response.rag_queries}")
            print(f"RAG sources: {list(set(response.rag_sources))}")


if __name__ == "__main__":
    _run_demo()
