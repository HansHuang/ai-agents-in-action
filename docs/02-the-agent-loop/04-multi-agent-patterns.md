# Multi-Agent Patterns

## What You'll Learn
- When one agent isn't enough: the case for multi-agent systems
- The four fundamental multi-agent patterns: delegation, debate, supervisor-worker, and swarm
- How agents communicate: shared messages, structured handoffs, and message buses
- The difference between multi-agent and multi-turn single-agent
- When multi-agent is overkill (and when it isn't)

## Prerequisites
- [Anatomy of an AI Agent](01-anatomy-of-an-agent.md) — the single-agent loop
- [Planning Strategies](03-planning-strategies.md) — Plan-and-Execute is the gateway to multi-agent
- [Tool Design Patterns](02-tool-design-patterns.md) — agents communicate through tools

---

## Why Multiple Agents?

A single agent with a ReAct loop can handle most tasks. But as complexity grows, you hit limits:

| Single Agent Limit | Multi-Agent Solution |
|:---|:---|
| Too many tools confuse the model | Each agent gets only its relevant tools |
| One system prompt can't serve all use cases | Each agent has a specialized system prompt |
| Context window fills with irrelevant history | Each agent carries only its own context |
| One model isn't optimal for all tasks | Different agents use different models |
| Hard to test and debug | Each agent can be tested in isolation |

The core insight: **multi-agent is organizational design, not AI architecture.** You're splitting responsibilities the same way you'd split a monolith into microservices.

---

## Pattern 1: Delegation (Agent-to-Agent Tool Call)

The simplest multi-agent pattern. One agent treats another agent as a tool.

```
User: "Summarize our Q3 financials and create a slide deck outline"

Coordinator Agent:
  → "I'll delegate the financial summary to the Finance Agent"
  → [calls finance_agent as a tool]
  → Finance Agent returns summary
  → "Now I'll delegate the slide outline to the Presentation Agent"
  → [calls presentation_agent as a tool]
  → Presentation Agent returns outline
  → Synthesizes both results for the user
```

### Implementation

```python
class DelegationAgent:
    def __init__(self):
        self.specialist_agents = {
            "finance": FinanceAgent(),
            "presentation": PresentationAgent(),
            "research": ResearchAgent()
        }
        # Each specialist is exposed as a tool
        self.tools = self._build_tool_definitions()
    
    def _build_tool_definitions(self) -> list[dict]:
        tools = []
        for name, agent in self.specialist_agents.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": f"delegate_to_{name}_agent",
                    "description": agent.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": f"Task for the {name} agent. {agent.task_guidance}"
                            }
                        },
                        "required": ["task"]
                    }
                }
            })
        return tools
    
    def run(self, user_input: str) -> str:
        # Standard ReAct loop, but tools are other agents
        messages = [
            {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        while True:
            response = call_llm(messages, self.tools)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    agent_name = tool_call.function.name.replace("delegate_to_", "").replace("_agent", "")
                    task = json.loads(tool_call.function.arguments)["task"]
                    result = self.specialist_agents[agent_name].run(task)
                    # Append the tool result to messages so the coordinator sees it
                    # (`call_llm()` is a thin wrapper around the OpenAI SDK — see delegation_agent.py)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                     "content": json.dumps({"result": result})})
            else:
                return response.content
```

### The Coordinator System Prompt

```markdown
You are a coordinator agent with access to specialist agents.
Your job is NOT to answer questions directly. Your job is to:

1. Understand the user's request
2. Identify which specialist(s) can help
3. Delegate to them with clear, specific tasks
4. Synthesize their results into a coherent response

Available specialists:
- finance_agent: Financial analysis, Q&A about numbers, reports
- presentation_agent: Slide decks, outlines, visual structure
- research_agent: Web research, fact-finding, data gathering

Delegation rules:
- Delegate as early as possible. Don't try to answer yourself.
- If a task spans multiple domains, delegate to multiple specialists.
- Give each specialist a complete, self-contained task.
- After receiving results, check if you need more information.
```

### When Delegation Works
- Tasks naturally split by domain (finance, legal, engineering)
- Each specialist needs different tools and prompts
- The coordinator's job is routing and synthesis

> **Code Reference:** [Python](../../code/python/06-multi-agent/delegation_agent.py) · [Node.js](../../code/nodejs/06-multi-agent/delegation_agent.ts) · [Go](../../code/go/06-multi-agent/delegation_agent.go)  
> The delegation example implements a coordinator with three specialist agents, each with domain-specific tools and system prompts.

---

## Pattern 2: Debate (Adversarial Collaboration)

Two or more agents critique each other's outputs to produce a better result. One generates, the other challenges.

```
User: "Design a pricing strategy for our SaaS product"

Generator Agent:
  → "Here's a tiered pricing model: $10/mo basic, $50/mo pro, $200/mo enterprise..."

Critic Agent:
  → "Your enterprise tier is underpriced. Competitors charge $500+. Also,
     you didn't consider usage-based pricing as an alternative."

Generator Agent:
  → "Revised: Added usage-based option. Adjusted enterprise to $500/mo.
     Added annual discount structure..."

Critic Agent:
  → "Better. One remaining issue: the jump from $50 to $500 creates a
     'dead zone' where mid-market customers have no good option."

Generator Agent:
  → "Final version: Added a $150/mo 'growth' tier between pro and enterprise."
```

### Implementation

```python
class DebateAgents:
    def run(self, task: str, rounds: int = 3) -> dict:
        generator_messages = [
            {"role": "system", "content": GENERATOR_PROMPT},
            {"role": "user", "content": task}
        ]
        critic_messages = [
            {"role": "system", "content": CRITIC_PROMPT}
        ]
        
        history = []
        
        for round_num in range(rounds):
            # Generator produces
            gen_response = call_llm(generator_messages)
            history.append({"role": "generator", "round": round_num, 
                           "output": gen_response})
            
            # Critic evaluates
            critic_messages.append({
                "role": "user", 
                "content": f"Evaluate this output:\n\n{gen_response}"
            })
            critic_response = call_llm(critic_messages)
            history.append({"role": "critic", "round": round_num,
                           "feedback": critic_response})
            
            # Feed critique back to generator
            generator_messages.append({
                "role": "assistant", "content": gen_response
            })
            generator_messages.append({
                "role": "user",
                "content": f"Critique: {critic_response}\n\nRevise your response."
            })
            
            # Check if critic is satisfied
            if "NO_ISSUES" in critic_response:
                break
        
        return {
            "final_output": history[-1]["output"],
            "rounds": round_num + 1,
            "history": history
        }
```

### The Generator Prompt
```markdown
You are a strategy consultant. Given a task, produce the best possible
answer. Be thorough, specific, and actionable. Include data and examples
where possible. After receiving critique, revise your answer to address
every point raised.
```

### The Critic Prompt
```markdown
You are a rigorous reviewer. Your job is to find flaws in the generator's
output. Look for:

1. Logical errors or contradictions
2. Missing considerations or edge cases
3. Weak assumptions
4. Implementation gaps
5. Anything a competitor or stakeholder would challenge

Be specific. Say exactly what's wrong and how to fix it.
If the output has no significant flaws, respond with "NO_ISSUES" and explain why.
```

### When Debate Works
- Strategic decisions with no single right answer
- Content that will face scrutiny (proposals, reports, public statements)
- Tasks where the generator might have blind spots

### When Debate Fails
- Simple factual queries (unnecessary overhead)
- Tasks requiring creativity over correctness
- When the critic isn't given a different perspective than the generator

> **Code Reference:** [Python](../../code/python/06-multi-agent/debate_agent.py)  
> The `DebateSystem` class separates generator and critic into independent message histories.  
> Two stop conditions: the critic prefixes its response with `"NO_ISSUES"` **or** `_similarity()` detects that the answer barely changed between rounds (avoids cycling without progress).

---

## Pattern 3: Supervisor-Worker

A supervisor agent decomposes a task, assigns subtasks to workers, and validates results. Unlike delegation, the supervisor can reassign, request revisions, and track progress.

```
Supervisor Agent
    │
    ├── Worker A (Research)
    │   └── "Find top 3 competitors in AI code editors"
    │
    ├── Worker B (Analysis)
    │   └── "Compare features and pricing of the 3 competitors"
    │
    └── Worker C (Writing)
        └── "Write a competitive analysis report from Worker B's data"
```

### Implementation Sketch

```python
class SupervisorAgent:
    def __init__(self, workers: dict[str, Agent]):
        self.workers = workers
        self.task_queue = []
        self.completed_tasks = {}
    
    def run(self, goal: str) -> str:
        # Phase 1: Decompose goal into subtasks
        subtasks = self._decompose(goal)
        
        # Phase 2: Assign to workers
        for subtask in subtasks:
            worker = self._select_worker(subtask)
            self.task_queue.append((worker, subtask))
        
        # Phase 3: Execute with oversight
        while self.task_queue:
            worker, task = self.task_queue.pop(0)
            result = worker.run(task)
            
            # Supervisor validates
            if self._validate_result(task, result):
                self.completed_tasks[task.id] = result
            else:
                # Reassign with feedback
                feedback = self._generate_feedback(task, result)
                self.task_queue.append((worker, task.with_feedback(feedback)))
        
        # Phase 4: Synthesize final output
        return self._synthesize(goal, self.completed_tasks)
```

### The Supervisor Prompt
```markdown
You are a project manager. Given a goal:

1. DECOMPOSE: Break it into independent subtasks. Each subtask should be
   completable by a single worker with a single toolset.
2. ASSIGN: Match each subtask to the best worker based on their capabilities.
3. VALIDATE: Review each worker's output. If it doesn't meet requirements,
   provide specific feedback and request revision.
4. SYNTHESIZE: When all subtasks are complete, combine results into a final
   deliverable that addresses the original goal.

Available workers:
- researcher: Web search, data gathering, fact-checking
- analyst: Data analysis, comparisons, calculations
- writer: Content creation, summarization, formatting
```

### When Supervisor-Worker Works
- Complex projects with multiple deliverables
- Tasks requiring quality control at each step
- When different subtasks need different expertise

### When It's Overkill
- The task can be done by one agent with a Plan-and-Execute strategy
- The overhead of coordination exceeds the benefit of specialization

> **Code Reference:** [Python](../../code/python/06-multi-agent/supervisor_agent.py)  
> `SupervisorAgent` decomposes, assigns, validates (with LLM scoring), and synthesises.  
> Failed validations trigger a retry with specific feedback; max 2 reassignments per subtask.

---

## Pattern 4: Swarm (Emergent Collaboration)

Multiple identical agents work on the same problem independently, then merge results. No hierarchy. No assignment. Each agent sees the problem and contributes.

```
User: "Generate 10 creative names for our AI productivity app"

Agent 1: "CogniFlow, ThinkSpark, MindForge..."
Agent 2: "NeuralPath, BrainWave, Synapse..."
Agent 3: "Intellecta, Cerebro, Cortex..."
Agent 4: "AetherMind, FluxThink, QuantumTask..."

Merger Agent:
  → "Top 10 across all suggestions, with duplicates removed and
     categorized by theme: cognitive, nature, tech, abstract"
```

### Implementation

```python
class SwarmAgent:
    def run(self, task: str, swarm_size: int = 4) -> str:
        # Phase 1: All agents work independently
        results = []
        for i in range(swarm_size):
            # Optional: give each agent a slightly different perspective
            variant_prompt = self._get_variant_prompt(i, swarm_size)
            result = call_llm(
                system=variant_prompt,
                user=task
            )
            results.append(result)
        
        # Phase 2: Merge results
        merged = call_llm(
            system="You are a synthesizer. Given multiple responses to the "
                   "same question, identify the best ideas, remove duplicates, "
                   "and produce a single consolidated answer.",
            user=f"Task: {task}\n\nResponses:\n" + 
                 "\n---\n".join(results)
        )
        return merged
    
    def _get_variant_prompt(self, index: int, total: int) -> str:
        perspectives = [
            "Focus on practicality and ease of use.",
            "Focus on innovation and uniqueness.",
            "Focus on simplicity and elegance.",
            "Focus on scalability and performance."
        ]
        return f"You are a creative assistant. {perspectives[index % len(perspectives)]}"
```

### When Swarm Works
- Creative tasks (brainstorming, naming, ideation)
- Tasks where diversity of thought improves the outcome
- When you want to reduce individual agent bias

### When Swarm Fails
- Sequential tasks where each step depends on previous results
- When cost is a concern (4-5x the API calls)
- Tasks requiring specialized knowledge that a general agent lacks

> **Code Reference:** [Python](../../code/python/06-multi-agent/swarm_agent.py)  
> `SwarmAgent` runs all agents in parallel with `ThreadPoolExecutor`, each with a different perspective.  
> **Tip:** Swarm is particularly effective when combined with Reflection — run the swarm first, then apply a Reflection loop to polish the merged output.

---

## Choosing a Pattern

```
Is the task splittable by domain?
    → Delegation (finance agent + legal agent + ...)

Does the output need to withstand scrutiny?
    → Debate (generator + critic, 2-3 rounds)

Is this a complex project with dependencies?
    → Supervisor-Worker (plan → assign → execute → validate → synthesize)

Is this a creative task where more ideas = better?
    → Swarm (multiple agents, merge results)

Can one agent with a good plan handle this?
    → Don't use multi-agent. Use Plan-and-Execute.
```

| Your Situation | Pattern | Why |
|:---|:---|:---|
| Multiple domains, one answer | Delegation | Each specialist handles its domain |
| High-stakes decisions | Debate | Adversarial review catches blind spots |
| Complex projects | Supervisor-Worker | Structured task management with QC |
| Brainstorming, ideation | Swarm | Diversity of ideas, then consolidate |
| Everything else | Single agent | Don't add complexity you don't need |

---

## Communication Patterns

How agents talk to each other is as important as how they think.

### Pattern A: Shared Message History (Tight Coupling)
All agents append to the same messages list. Simple but risky — any agent can corrupt the context.

### Pattern B: Structured Handoff (Medium Coupling)
Agents pass structured data objects with explicit `from_agent`, `to_agent`, and `task` fields. The receiving agent starts with a clean context.

```python
@dataclass
class Handoff:
    from_agent: str
    to_agent: str
    task: str
    context: dict  # Only the data the receiving agent needs
    reply_to: str  # Where to send the result
```

> **Code Reference:** [structured_handoff.py](../../code/python/06-multi-agent/structured_handoff.py)  
> Demonstrates coordinator → specialist → coordinator with validated `Handoff` objects.

### Pattern C: Message Bus (Loose Coupling)
Agents publish messages to a bus. Other agents subscribe to relevant topics. Best for complex systems with many agents.

```python
class AgentBus:
    def publish(self, topic: str, message: dict): ...
    def subscribe(self, topic: str, agent: Agent): ...
    def request(self, target_agent: str, task: str) -> dict: ...
```

For most applications, Pattern B (Structured Handoff) is the right balance of simplicity and safety.

---

## The Multi-Agent System Prompt Pattern

Every agent in a multi-agent system needs to know its role and its boundaries:

```markdown
You are the [ROLE] agent in a multi-agent system.

## Your Responsibility
[Clear, narrow scope. What you do and what you DON'T do.]

## Your Tools
[Only the tools this agent needs.]

## Communication Protocol
- You receive tasks from: [coordinator / supervisor / message bus]
- You respond to: [coordinator / supervisor / message bus]
- Response format: {"status": "complete", "result": ...} or
                   {"status": "need_clarification", "question": ...}

## Boundaries
- Do not attempt tasks outside your responsibility.
- If you receive a task you can't handle, respond with status "out_of_scope"
  and suggest the appropriate agent.
- Do not communicate with other agents directly unless instructed.
```

This boundary enforcement prevents agents from going rogue.

---

## Common Pitfalls

- **"My agents keep talking to each other forever"**: You don't have a stop condition. Each agent conversation needs a max turns limit and a clear "task complete" signal. In code: `for iteration in range(MAX_ITERATIONS): ... if not msg.tool_calls: return msg.content`. Without the guard, the loop never terminates.
- **"The coordinator doesn't actually delegate"**: Your system prompt says "You are a helpful assistant" instead of "Your ONLY job is to delegate." The model defaults to answering directly.
- **"Specialist agents have overlapping capabilities"**: If both the finance agent and the research agent can look up stock prices, the coordinator delegates randomly. Make capabilities mutually exclusive.
- **"Multi-agent is slower than single-agent"**: Usually true — this is a design tradeoff, not a bug. Multi-agent trades latency for quality, maintainability, and specialization. If speed matters more than correctness, stay single-agent.
- **"I built a 10-agent system for a todo app"**: Multi-agent is for complex, multi-domain tasks. For a CRUD app with AI features, one agent with good tools is enough.
- **"Agent B uses Agent A's internal context"**: Use structured handoffs. Agent B should only see what Agent A explicitly passes. Sharing full message history creates coupling that's hard to debug.

## What's Next

You can now design systems with multiple collaborating agents. Next: the evolution from tools to skills — composing tools, prompts, and validation into reusable, testable capability units.
→ [Skills: Composing Capabilities](05-skills-composing-capabilities.md)