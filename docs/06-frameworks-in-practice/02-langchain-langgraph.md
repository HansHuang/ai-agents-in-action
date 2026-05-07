# LangChain and LangGraph

## What You'll Learn
- LangChain's core abstractions: chains, retrievers, agents, and tools
- How LangGraph extends LangChain with stateful, graph-based workflows
- Building the same agent you built from scratch — now with LangGraph
- When LangChain helps and when it hurts — real-world guidance
- The LangChain ecosystem: LangSmith, LangServe, and LangFuse

## Prerequisites
- [When to Use Frameworks](01-when-to-use-frameworks.md) — the build vs. buy decision
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — you've built agents from scratch
- [RAG from Scratch](../03-memory-and-retrieval/03-rag-from-scratch.md) — LangChain's primary use case

---

## What Is LangChain?

LangChain is an open-source framework for building LLM-powered applications. It provides:

1. **Abstractions**: Chains, agents, retrievers, tools — pre-built components you compose
2. **Integrations**: 700+ connectors for LLMs, vector databases, document loaders, and tools
3. **Off-the-shelf chains**: Ready-made pipelines for RAG, summarization, and Q&A

It's the most popular AI framework. It's also the most controversial. This chapter explains why both are true.

---

## LangChain's Core Abstractions

### Chains

A chain is a sequence of operations. The simplest chain is an LLM call. A more complex chain might be: load documents → embed → store → retrieve → prompt → generate.

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# The simplest chain: prompt → LLM → output
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | llm  # The "|" is the LangChain Expression Language (LCEL)
result = chain.invoke({"topic": "programmers"})
```

### Retrievers

A retriever is an interface for fetching documents. LangChain has retrievers for every vector database:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create a retriever from a vector store
vector_store = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Build a complete RAG chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using this context: {context}"),
    ("human", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)
result = qa_chain.invoke({"input": "What's our return policy?"})
```

### Agents

LangChain's agent abstraction wraps the ReAct loop you built from scratch:

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Define tools
tools = [
    StructuredTool.from_function(get_weather),
    StructuredTool.from_function(get_stock_price),
]

# The prompt must include a {agent_scratchpad} placeholder
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Create agent — create_tool_calling_agent works with any model that supports tool calling
agent = create_tool_calling_agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    prompt=agent_prompt,
)

# Run with automatic loop management
agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
result = agent_executor.invoke({"input": "What's the weather in Tokyo?"})
```

The `AgentExecutor` handles the orchestration loop, tool execution, message management, and stop conditions — all the things you built manually in Chapter 02.

### Tools

LangChain tools are wrappers around functions:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. City must include country code."""
    return f"Weather in {city}: 22°C, sunny"

# The @tool decorator automatically:
# - Reads the function signature for parameter schema
# - Uses the docstring as the tool description seen by the LLM
# - Wraps everything in LangChain's BaseTool interface
```

---

## LangChain Expression Language (LCEL)

LCEL is LangChain's composition syntax. The `|` operator chains components:

```python
# LCEL: compose a RAG pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Equivalent to:
# 1. Retrieve documents and format them
# 2. Fill the prompt template with context and question
# 3. Send to the LLM
# 4. Parse the output as a string

result = rag_chain.invoke("What's our return policy?")
```

LCEL is elegant for linear pipelines. For branching logic, LangGraph is the answer.

---

## What Is LangGraph?

LangGraph extends LangChain with **stateful, graph-based workflows**. Instead of a linear chain, you define a graph of nodes (steps) and edges (transitions).

### Why LangGraph?

Linear chains can't handle:
- **Conditional branching**: "If the user is premium, do X; otherwise do Y"
- **Loops**: "Keep searching until you find the answer"
- **State persistence**: "Remember what happened in step 1 when you're on step 5"
- **Human-in-the-loop**: "Pause here and wait for approval"

LangGraph handles all of these with a state machine approach.

---

## Building an Agent with LangGraph

Here's the same ReAct agent from Chapter 02, now built with LangGraph:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 1. Define the state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int

# 2. Define the tools
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, sunny."

@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price for a ticker symbol."""
    return f"{ticker}: $182.52 (+1.2%)"

tools = [get_weather, get_stock_price]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# 3. Define the nodes (steps)
def agent_node(state: AgentState) -> AgentState:
    """The LLM decides what to do."""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}

def tool_node(state: AgentState) -> AgentState:
    """Execute tool calls from the last message."""
    last_message = state["messages"][-1]
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find and execute the tool
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
    
    return {"messages": tool_results, "iteration_count": state["iteration_count"]}

# 4. Define the routing logic
def should_continue(state: AgentState) -> str:
    """Decide: continue with tools, or end with final answer?"""
    # Safety: max iterations (check BEFORE inspecting tool calls)
    if state["iteration_count"] >= 10:
        return "end"

    last_message = state["messages"][-1]

    # If the LLM called tools, route to tool_node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done
    return "end"

# 5. Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")  # After tools, loop back to agent

# 6. Compile and run
# compile() returns a CompiledGraph — a callable that accepts initial state
graph = workflow.compile()

result = graph.invoke({
    "messages": [HumanMessage(content="What's the weather in Tokyo?")],
    "iteration_count": 0,
})

print(result["messages"][-1].content)
```

### Visualizing the Graph

LangGraph can visualize your agent as a diagram:

```
┌─────────┐
│  START  │
└────┬────┘
     ▼
┌─────────┐     has_tool_calls     ┌─────────┐
│  agent  │────────────────────────▶│  tools  │
└─────────┘                        └─────────┘
     │                                   │
     │ no_tool_calls                     │
     ▼                                   │
┌─────────┐                              │
│   END   │◄─────────────────────────────┘
└─────────┘
```

This is the exact same ReAct loop — agent reasons, tools execute, agent reasons again, until done. The difference: it's now a declared graph, not imperative code.

---

## LangGraph's Killer Features

### 1. State Persistence (Checkpointing)

LangGraph automatically saves state after each step. If the agent crashes, it resumes where it left off:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)

# Run with a thread ID for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke({"messages": [HumanMessage(content="Research AI chips")]}, config)

# Even if the process crashes here, the next call resumes from the checkpoint
result = agent.invoke(
    {"messages": [HumanMessage(content="Now compare their prices")]},
    config  # Same thread_id → resumes from previous state
)
```

### 2. Human-in-the-Loop

LangGraph can pause execution and wait for human approval:

```python
from langgraph.checkpoint.memory import MemorySaver

workflow.add_node("human_approval", human_approval_node)
workflow.add_conditional_edges(
    "agent",
    lambda state: "human_approval" if state.get("requires_approval") else "tools",
    {"human_approval": "human_approval", "tools": "tools"},
)

# interrupt_before pauses the graph BEFORE executing the listed node
graph = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_approval"],
)

# Start a run — graph pauses before human_approval
config = {"configurable": {"thread_id": "review-123"}}
graph.invoke({"messages": [...]}, config)

# A human reviews, then resumes by updating state and invoking again
graph.update_state(config, {"approved": True})
graph.invoke(None, config)  # Resume from checkpoint
```

### 3. Subgraphs

Complex agents can be composed of sub-agents:

```python
# A research sub-agent
research_graph = create_research_agent()

# A writing sub-agent
writing_graph = create_writing_agent()

# Compose them
main_workflow.add_node("research", research_graph.compile())
main_workflow.add_node("write", writing_graph.compile())
main_workflow.add_edge("research", "write")
```

---

## When LangChain/LangGraph Works Well

| Scenario | Why It's Good |
|:---|:---|
| **Standard RAG pipelines** | `create_retrieval_chain` handles 90% of the boilerplate |
| **Complex workflows with branching** | LangGraph's state machine is the right abstraction |
| **Multi-step agent with persistence** | Checkpointing is built-in, not bolted on |
| **Need many integrations** | 700+ connectors save weeks of development |
| **Team already knows LangChain** | Don't fight the team's expertise |
| **Human-in-the-loop requirements** | LangGraph has native support |

## When LangChain/LangGraph Works Poorly

| Scenario | Why It Hurts |
|:---|:---|
| **Simple agents (1-3 tool calls)** | Framework overhead exceeds the problem complexity |
| **You need complete control** | LangChain makes decisions for you that you might disagree with |
| **Debugging is critical** | Seven layers of abstraction between you and the error |
| **API stability matters** | LangChain's history of breaking changes is real |
| **You're learning AI engineering** | LangChain obscures the concepts you need to understand |
| **Your use case is non-standard** | LangChain optimizes for common patterns |

---

## The LangChain Ecosystem

| Tool | Purpose | When to Use |
|:---|:---|:---|
| **LangChain** | Core framework, chains, agents, tools | Building LLM applications |
| **LangGraph** | Stateful graph-based workflows | Complex branching, persistence, HITL |
| **LangSmith** | Tracing, evaluation, monitoring | Works with any agent — LangChain or custom |
| **LangServe** | Deploy chains as REST APIs | Putting LangChain in production (deprecated in favour of FastAPI + streaming) |
| **LangFuse** | Open-source tracing (self-hosted) | Teams that can't send data to LangSmith's cloud |

---

## Practical Advice from Production

### 1. Use LangChain for Integration, Not Orchestration

LangChain's best feature is its ecosystem. Use its document loaders, vector store connectors, and tool integrations. Write your own agent loop.

### 2. If You Use LangGraph, Keep State Explicit

LangGraph's automatic state management is powerful and dangerous. A state key that accidentally accumulates data can balloon your context. Audit your state at each step.

### 3. Pin Your Versions

```txt
# requirements.txt — pin everything
langchain==0.3.25
langchain-core==0.3.56
langchain-openai==0.3.14
langchain-community==0.3.24
langgraph==0.4.0
```

Never use `langchain>=0.3.0` in production. A minor version bump can break your agent. The `langchain-core` version matters most — it defines the base types. Pin all five packages together.

### 4. Test Without LangChain First

Build a minimal version of your agent from scratch. Understand the concepts. Then decide if LangChain adds value. Never start with LangChain when you're learning.

---

## Common Pitfalls

- **"I use LangChain's AgentExecutor without understanding the loop"**: When the agent gets stuck in a loop, you won't know how to fix it because you don't know what the loop is doing. Understand the ReAct pattern before using anyone's executor. `AgentExecutor` is also deprecated since LangChain 0.3 — prefer LangGraph for new agents.
- **"I upgraded LangChain without reading the changelog"**: Breaking changes are common. Read the migration guide. Better yet, wait a month after a major release before upgrading.
- **"I use LangChain for a 20-line script"**: `pip install langchain` adds 50+ dependencies. For a simple LLM call, use the OpenAI SDK directly. Add LangChain when you need its integrations.
- **"I treat LangGraph state like a dumpster"**: Every key in your state graph persists across steps. A key that grows by 1,000 tokens per step will exceed your context window by step 100. Use `add_messages` (which trims via `RemoveMessage`) or implement a summarisation step to keep state bounded.
- **"I ignore the LangChain source code"**: When something breaks (and it will), you need to read LangChain's source to understand why. The documentation is often outdated; the source code is the truth.

## What's Next

You understand the dominant framework. Next: multi-agent frameworks — CrewAI and AutoGen, which specialize in agent-to-agent collaboration.
→ [CrewAI and AutoGen](03-crewai-autogen.md)