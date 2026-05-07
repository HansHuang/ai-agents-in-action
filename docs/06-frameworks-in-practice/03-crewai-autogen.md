# CrewAI and AutoGen

## What You'll Learn
- Why multi-agent frameworks exist: making collaboration turnkey
- CrewAI: role-based agent orchestration with crews and tasks
- AutoGen: conversational multi-agent patterns from Microsoft
- How CrewAI and AutoGen differ in philosophy and implementation
- When these frameworks help and when they add unnecessary complexity
- The thin line between multi-agent and over-engineered

## Prerequisites
- [Multi-Agent Patterns](../02-the-agent-loop/04-multi-agent-patterns.md) — the patterns these frameworks automate
- [When to Use Frameworks](01-when-to-use-frameworks.md) — the build vs. buy decision
- [LangChain and LangGraph](02-langchain-langgraph.md) — the other major framework approach

---

## The Multi-Agent Framework Pitch

You've built multi-agent systems from scratch. You understand delegation, debate, supervisor-worker, and swarm patterns. You know how to structure agent communication and handle handoffs.

Multi-agent frameworks say: *"That's a lot of boilerplate. Let us handle it."*

CrewAI and AutoGen take different approaches to the same problem. CrewAI thinks in terms of roles and tasks. AutoGen thinks in terms of conversations.

---

## CrewAI: Agents with Jobs

CrewAI models multi-agent systems as **crews** — teams of agents with defined roles working on assigned tasks.

### Core Concepts

| Concept | What It Is | Example |
|:---|:---|:---|
| **Agent** | An AI with a role, goal, and backstory | "Senior Financial Analyst with 20 years of experience" |
| **Task** | A unit of work assigned to an agent | "Analyze Q3 earnings reports for tech companies" |
| **Crew** | A team of agents with a shared objective | "Financial Research Crew" |
| **Process** | How the crew organizes work | Sequential (one after another) or Hierarchical (manager assigns) |

### Building a Crew

```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles and personalities
researcher = Agent(
    role="Financial Researcher",
    goal="Find and extract key financial data from earnings reports",
    backstory="""You are a senior financial researcher with 20 years of experience.
    You can read through dense financial documents and extract the numbers
    that matter most: revenue, profit margins, growth rates, and guidance.""",
    tools=[web_search_tool, sec_filing_tool],
    verbose=True
)

analyst = Agent(
    role="Financial Analyst",
    goal="Analyze financial data and identify trends, risks, and opportunities",
    backstory="""You are a CFA charterholder who specializes in tech sector analysis.
    You take raw financial data and transform it into actionable insights.
    You're known for spotting trends before they become obvious.""",
    tools=[calculator_tool, chart_tool],
    verbose=True
)

writer = Agent(
    role="Report Writer",
    goal="Create clear, concise financial reports for investors",
    backstory="""You are a financial journalist turned investor communications
    specialist. You take complex analysis and make it accessible to
    sophisticated but non-specialist readers.""",
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the Q3 2026 earnings for Apple, Microsoft, and Google. "
                "Extract revenue, EPS, growth rates, and forward guidance.",
    agent=researcher,
    expected_output="Structured data with key metrics for each company"
)

analysis_task = Task(
    description="Analyze the research data. Compare the three companies. "
                "Identify which is performing best and why. Note any risks.",
    agent=analyst,
    expected_output="Comparative analysis with clear winner and supporting evidence",
    context=[research_task]  # Depends on research_task
)

writing_task = Task(
    description="Write a 500-word investor briefing based on the analysis. "
                "Include an executive summary, key findings, and outlook.",
    agent=writer,
    expected_output="Professional investor briefing with clear sections",
    context=[analysis_task]  # Depends on analysis_task
)

# Form the crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,  # Execute tasks in order
    verbose=True
)

# Run
result = crew.kickoff()
print(result)
```

### What CrewAI Handles Automatically

- **Task assignment**: Each task is explicitly assigned to one agent at definition time; agents with `allow_delegation=True` can sub-delegate to teammates
- **Context passing**: Output from one task automatically becomes context for tasks that list it in `context=[...]`
- **Sequential execution**: Tasks run in dependency order; `Process.hierarchical` adds a manager agent that assigns sub-tasks dynamically
- **Agent collaboration**: Agents with `allow_delegation=True` can ask teammates for help mid-task
- **Output guidance**: The `expected_output` field guides the LLM toward a specific format (it is a natural-language description, not a schema enforcer)

### The CrewAI Philosophy

CrewAI's mental model is **organizational**. You define roles, assign tasks, and let the crew figure out how to collaborate. It's like managing a team: you don't tell people how to do their jobs, you tell them what needs to be done and who's responsible.

---

## AutoGen: Agents in Conversation

AutoGen (from Microsoft) models multi-agent systems as **conversations**. Agents talk to each other, and solutions emerge from the dialogue.

### Core Concepts

| Concept | What It Is | Example |
|:---|:---|:---|
| **Agent** | A conversational entity with a role | "User Proxy", "Assistant", "Critic" |
| **Conversation** | A multi-turn dialogue between agents | Assistant proposes, User asks clarifying questions |
| **GroupChat** | Multiple agents in a shared conversation | Product team discussion |
| **Code Execution** | Agents can write and execute Python code via `UserProxyAgent` | Assistant writes a script; UserProxy runs it and returns the output |

### Building an AutoGen Team

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define the LLM configuration
llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "..."}],
    "temperature": 0.7
}

# Create agents
assistant = AssistantAgent(
    name="ResearchAssistant",
    llm_config=llm_config,
    system_message="""You are a research assistant. You find information,
    analyze data, and present findings. You can write Python code to
    analyze data when needed. Always verify your sources.""",
)

critic = AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""You are a rigorous critic. Your job is to find flaws
    in the assistant's work. Question assumptions. Check facts. Identify
    missing information. Be constructive but thorough.""",
)

user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",  # Ask for human input when a termination message is detected
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "workspace", "use_docker": False},
)

# Create a group chat
groupchat = GroupChat(
    agents=[user_proxy, assistant, critic],
    messages=[],
    max_round=15
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Start the conversation
user_proxy.initiate_chat(
    manager,
    message="""Research the top 3 AI chip companies. Compare their:
    1. Market share
    2. Key products
    3. Financial performance
    Provide a recommendation for investment."""
)
```

### What AutoGen Handles Automatically

- **Turn-taking**: The manager decides who speaks next
- **Code execution**: Agents can write and run Python code
- **Human intervention**: UserProxy can ask for human input at any point
- **Conversation management**: The group chat tracks the full dialogue
- **Tool integration**: Agents can call external APIs and execute code

### The AutoGen Philosophy

AutoGen's mental model is **conversational**. Agents talk through problems. Solutions emerge from dialogue, not from a predefined workflow. It's like a meeting: people discuss, debate, and eventually reach a conclusion.

---

## CrewAI vs. AutoGen: The Philosophical Divide

| | CrewAI | AutoGen |
|:---|:---|:---|
| **Mental model** | Organization (roles, tasks, hierarchy) | Conversation (dialogue, turn-taking, emergence) |
| **Workflow** | Predefined: tasks have dependencies | Emergent: conversation flows naturally |
| **Control** | Sequential or hierarchical process | Group chat with a manager |
| **Best for** | Structured projects with clear deliverables | Open-ended problems requiring discussion |
| **Predictability** | High: tasks execute in defined order | Lower: conversation can go anywhere |
| **Setup complexity** | Moderate: define roles and tasks | Lower: define agents and start talking |
| **Output format** | Structured: expected_output defined | Unstructured: whatever the conversation produces |
| **Human involvement** | Via tools or task assignment | Via UserProxy agent in the conversation |
| **TypeScript/Go support** | None (Python only) | None (Python only) |

---

## When CrewAI Shines

### Structured Research Projects
```
Task: "Create a competitive analysis report for the AI code editor market"

Crew:
├── Market Researcher: Gathers data on 5 competitors
├── Competitive Analyst: Compares features, pricing, positioning
├── SWOT Specialist: Creates SWOT analysis for each competitor
└── Report Writer: Compiles everything into a professional report

Process: Sequential → Each step feeds the next
```

### When Outcomes Are Well-Defined
You know exactly what the final deliverable looks like. CrewAI's `expected_output` field keeps agents focused on producing specific results.

### When You Need Predictable Execution
Tasks run in order. Dependencies are explicit. You can trace exactly what happened and when. Good for regulated industries.

---

## When AutoGen Shines

### Open-Ended Problem Solving
```
Task: "How should we price our new AI feature?"

Conversation:
Assistant: "Let me analyze competitor pricing..."
Critic: "Your analysis ignores the freemium tier trends."
Assistant: "Good point. Let me revise..."
UserProxy: "What about enterprise customers specifically?"
Assistant: "Enterprise typically expects..."
Critic: "But our enterprise customers have different needs because..."
```

The solution emerges from the discussion. Neither agent "plans" the conversation.

### When You Need Code Execution
AutoGen agents can write and execute code. The assistant writes a Python script to analyze data, the system runs it, and the results feed back into the conversation.

### When Human Input Is Needed Mid-Conversation
UserProxy can pause the conversation and ask the human for input. The conversation resumes with the human's contribution.

---

## When Neither Framework Helps

| Scenario | Why Frameworks Fail |
|:---|:---|:---|
| **Simple single-agent tasks** | Both frameworks add overhead for what a single function can do |
| **Real-time applications** | Multi-agent conversations take seconds to minutes |
| **Strict latency requirements** | Agent-to-agent communication adds unpredictable delays |
| **You need fine-grained control** | Both frameworks make decisions about communication that you might disagree with |
| **Your use case doesn't fit the mold** | CrewAI assumes tasks. AutoGen assumes conversation. If your pattern is different, you're fighting the framework || **TypeScript or Go projects** | Neither framework has production-ready TypeScript or Go support (May 2026); build from scratch or use LangChain.js |
| **Tight token budget** | Framework overhead typically adds 30–80% more tokens vs. an equivalent from-scratch implementation |
---

## A Comparison: Same Task, Three Ways

Let's build the same system — a research agent that finds information, critiques it, and produces a report — three ways.

### From Scratch (Your Code)
```python
# You explicitly control every interaction
research_result = research_agent.run(topic)
critique = critic_agent.run(research_result)
if critique.has_issues:
    research_result = research_agent.run(f"Revise: {critique.feedback}")
report = writer_agent.run(research_result)
# ~45 lines total (agent prompts + orchestration loop)
# Full control, full visibility, fewest tokens
```

### CrewAI
```python
# Agent and task definitions: ~45 lines (role, goal, backstory, description per agent/task)
crew = Crew(
    agents=[researcher, critic, writer],
    tasks=[research_task, critique_task, writing_task],
    process=Process.sequential
)
result = crew.kickoff()
# ~6 lines of invocation code (plus ~45 lines of definitions = ~51 total)
# Less control over communication; task dependencies are explicit
```

### AutoGen
```python
# Agent definitions: ~15 lines (system_message per agent)
groupchat = GroupChat(
    agents=[user_proxy, assistant, critic],
    messages=[],
    max_round=15
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
user_proxy.initiate_chat(manager, message=topic)
# ~8 lines of invocation code (plus ~15 lines of definitions = ~23 total)
# Least control; solution emerges from conversation; most tokens consumed
```

---

## The Multi-Agent Trap

The biggest risk with multi-agent frameworks is building a system that's more complex than necessary.

You do not need multi-agent for:
- A chatbot with a knowledge base → Single agent with RAG
- A system that calls a few APIs → Single agent with tools
- A form-filling assistant → Single agent with structured output
- A simple Q&A system → Single agent with system prompt

You might benefit from multi-agent for:
- Complex research projects with multiple stakeholders
- Systems requiring adversarial review (generation + critique)
- Workflows with clear role specialization (researcher, analyst, writer)
- Tasks where different agents genuinely need different tools and expertise

**Rule of thumb:** If you can't explain why you need multiple agents in one sentence, you probably don't need multiple agents.

---

## Common Pitfalls

- **"I built a 5-agent system to answer FAQs"**: Your FAQ bot doesn't need a research agent, an analysis agent, a writing agent, an editing agent, and a publishing agent. One agent with good retrieval is enough.
- **"My CrewAI tasks are so tightly coupled they might as well be one function"**: If every task depends on every other task, you've created a sequential pipeline with extra overhead. Use a single agent with Plan-and-Execute.
- **"My AutoGen conversation goes in circles"**: Without a clear stop condition, agents will debate forever. Set `max_round`. Make sure at least one agent can emit a termination message that your `is_termination_msg` function recognises.
- **"I use multi-agent because it sounds more advanced"**: Multi-agent is a tradeoff, not an upgrade. You're trading simplicity, speed, and cost for specialization and collaboration. Only make that trade if you need what you're getting.
- **"I don't track costs in multi-agent systems"**: Every agent-to-agent conversation burns tokens. A 15-round group chat with 3 agents = 45 LLM calls minimum. Framework overhead typically adds 30–80% more tokens than an equivalent from-scratch implementation. Track costs from day one — you might be surprised.

## What's Next

You've seen the two major multi-agent frameworks. Next: the Vercel AI SDK — a fundamentally different approach built for the full-stack TypeScript ecosystem.
→ [Vercel AI SDK](04-vercel-ai-sdk.md)