# Planning Strategies

## What You'll Learn
- Why ReAct is the starting point, not the destination
- Plan-and-Execute: separating strategy from execution
- Reflection: when the agent critiques its own work
- Self-Critique and revision loops
- Choosing the right strategy for your task
- How planning strategy affects cost, latency, and reliability

## Prerequisites
- [Anatomy of an AI Agent](01-anatomy-of-an-agent.md) — the orchestration loop
- [Tool Design Patterns](02-tool-design-patterns.md) — tools are the actions plans execute
- [Structured Output](../01-foundations/03-structured-output.md) — plans are structured output

---

## Beyond ReAct: Why Planning Matters

In the previous chapter, you built a ReAct agent. It sees the user's question, decides on one action, executes it, observes the result, and repeats. This works for simple tasks:

```
User: "What's the weather in Shanghai?"
Agent: [calls get_weather] → "Shanghai is 22°C and sunny."
```

But real tasks require multiple steps with dependencies:

```
User: "Research the top 3 AI chip companies, compare their revenues,
       and create a summary table."
```

A ReAct agent would fumble through this step-by-step, potentially losing the thread after the second company lookup. A planning agent would:

1. First, generate a plan
2. Then, execute each step
3. Finally, synthesize the results

This chapter covers the strategies that make this possible.

---

## The Three Planning Strategies

| Strategy | How It Works | Best For | Cost |
|:---|:---|:---|:---|
| **ReAct** | Reason → Act → Observe → Repeat | Simple tasks, 1-3 tool calls | Low |
| **Plan-and-Execute** | Generate full plan first, then execute each step | Multi-step tasks with dependencies | Medium |
| **Reflection** | Execute, then critique own output, then revise | Tasks where quality matters more than speed | High |

> Reflection can wrap any other strategy: run Plan-and-Execute to get a structured answer, then add a Reflection pass to catch factual errors or missing sections.

You don't have to pick one. You can combine them. An agent might use Plan-and-Execute for the overall task, with ReAct inside each execution step, and a final Reflection pass to catch mistakes.

---

## Strategy 1: ReAct (Reasoning + Acting)

You already know this one. It's the default. Here's when it shines and when it doesn't.

### When ReAct Works
- Tasks with 1-3 tool calls
- Tasks where each step depends on the result of the previous step
- Interactive tasks where the user is in the loop
- Tasks where the path isn't clear upfront

### When ReAct Fails
- Tasks requiring 5+ tool calls: the model loses the thread
- Tasks with independent subtasks: ReAct does them sequentially when they could be parallel
- Tasks requiring a structured output at the end: ReAct often forgets the original question

### The ReAct System Prompt

```markdown
You are an assistant that solves problems step by step.

For each user request:
1. Think about what information you need
2. Call the appropriate tool to get that information
3. Analyze the result
4. Decide if you need more information or can answer
5. When you have everything, provide a complete answer

Always explain your reasoning before calling a tool.
Never call a tool without explaining why.
```

The key phrase: "Always explain your reasoning." Without it, the model calls tools silently and you can't debug the chain of decisions.

---

## Strategy 2: Plan-and-Execute

Plan-and-Execute separates the *what* from the *how*. The model first generates a complete plan, then executes each step. This prevents the "wandering agent" problem where a ReAct loop drifts off-task after several tool calls.

### How It Works

```
Phase 1: PLAN
─────────────
User: "Compare the weather in Shanghai, London, and New York"

Planner LLM generates:
Step 1: Get weather for Shanghai
Step 2: Get weather for London
Step 3: Get weather for New York
Step 4: Create comparison table

Phase 2: EXECUTE
─────────────
Executor LLM runs Step 1 → gets Shanghai data
Executor LLM runs Step 2 → gets London data
Executor LLM runs Step 3 → gets New York data

Phase 3: SYNTHESIZE
─────────────
Synthesizer LLM creates the comparison table from all results
```

The Planner and Executor can be the same model with different system prompts, or different models entirely (a smart model for planning, a fast model for executing).

### Implementation

```python
class PlanAndExecuteAgent:
    def run(self, user_input: str) -> str:
        # Phase 1: Generate the plan
        plan = self._generate_plan(user_input)
        # plan = ["Get weather for Shanghai", "Get weather for London", 
        #         "Get weather for New York", "Create comparison table"]
        
        # Phase 2: Execute each step
        results = []
        for step in plan:
            result = self._execute_step(step)
            results.append(result)
        
        # Phase 3: Synthesize
        final_answer = self._synthesize(user_input, plan, results)
        return final_answer
    
    def _generate_plan(self, user_input: str) -> list[str]:
        # call_llm wraps client.chat.completions.create(); response_format
        # requests structured JSON output so the plan is always machine-readable.
        response = call_llm(
            system="You are a planner. Break down the user's request into "
                   "sequential steps. Each step should be a single action. "
                   "Output a JSON array of step descriptions.",
            user=user_input,
            response_format={"type": "json_object"}
        )
        return response.steps
    
    def _execute_step(self, step: str) -> dict:
        # This can use a ReAct loop internally for complex steps
        response = call_llm(
            system="You are an executor. Complete the given step using "
                   "available tools. Return the result.",
            user=step,
            tools=self.tools
        )
        return response.content
    
    def _synthesize(self, question: str, plan: list[str], 
                    results: list[dict]) -> str:
        context = "\n".join([
            f"Step: {step}\nResult: {result}"
            for step, result in zip(plan, results)
        ])
        response = call_llm(
            system="You are a synthesizer. Using the execution results, "
                   "answer the user's original question completely.",
            user=f"Question: {question}\n\nExecution Results:\n{context}"
        )
        return response.content
```

### When Plan-and-Execute Shines
- Multi-step research tasks
- Tasks with independent subtasks (can execute in parallel)
- Tasks where you need to show the user the plan upfront
- Tasks where cost control matters (estimate tokens from the plan)

### When Plan-and-Execute Fails
- Tasks where each step's result changes the plan
- Highly interactive tasks
- Creative tasks where structure constrains quality

> **Code Reference:** [Python](../../code/python/03-agent-loop/plan_execute_agent.py) · [Node.js](../../code/nodejs/03-agent-loop/plan_execute_agent.ts) · [Go](../../code/go/03-agent-loop/plan_execute_agent.go)  
> Each implementation includes the full Plan-and-Execute agent with a configurable planner, executor, and synthesizer.

---

## Strategy 3: Reflection

Reflection adds a quality control pass after the initial answer. The agent generates a response, then switches to "critic mode" to evaluate its own work, then revises based on the critique.

### How It Works

```
Step 1: GENERATE
Agent answers the question normally.

Step 2: REFLECT
Agent reviews its own answer:
- Did I answer all parts of the question?
- Are my facts correct? Did I cite sources?
- Is the format what the user asked for?
- What could be improved?

Step 3: REVISE
Agent rewrites the answer addressing the critique.

Step 4 (optional): REFLECT AGAIN
Agent checks the revised answer. Loop until satisfied or max iterations.
```

### The CritiqueResult Model

```python
class CritiqueResult(BaseModel):
    overall_score: int = Field(..., ge=1, le=10)
    is_satisfied: bool
    feedback: str = Field(..., min_length=10)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
```

Using a Pydantic model (rather than parsing raw strings) ensures the agent
always has a numeric score to compare against the quality threshold.

### Implementation

```python
class ReflectionAgent:
    def run(self, user_input: str, max_reflections: int = 2) -> str:
        # Step 1: Generate initial answer
        answer = self._generate(user_input)
        
        for i in range(max_reflections):
            # Step 2: Reflect
            critique = self._reflect(user_input, answer)
            
            # Check if the critic is satisfied
            if critique.is_satisfied:
                break
            
            # Step 3: Revise
            answer = self._revise(user_input, answer, critique.feedback)
        
        return answer
    
    def _generate(self, user_input: str) -> str:
        return call_llm(
            system="Answer the user's question thoroughly and accurately.",
            user=user_input
        )
    
    def _reflect(self, user_input: str, answer: str) -> CritiqueResult:
        response = call_llm(
            system="You are a strict critic. Review the answer against the "
                   "original question. Check for: completeness, factual "
                   "accuracy, clarity, formatting. Output JSON with: "
                   "{is_satisfied: bool, feedback: string, score: 1-10}",
            user=f"Question: {user_input}\n\nAnswer to review:\n{answer}",
            response_format={"type": "json_schema", "schema": CRITIQUE_SCHEMA}
        )
        return CritiqueResult(**response)
    
    def _revise(self, user_input: str, original: str, feedback: str) -> str:
        return call_llm(
            system="Revise your answer based on the critique. Address every "
                   "point in the feedback. Maintain the original's strengths.",
            user=f"Question: {user_input}\n\n"
                 f"Original answer:\n{original}\n\n"
                 f"Critique:\n{feedback}\n\n"
                 f"Revised answer:"
        )
```

### The Reflection System Prompt (Critic Mode)

```markdown
You are a strict quality reviewer. Evaluate the answer against the original question.

Check for:
1. COMPLETENESS: Does it answer all parts of the question?
2. ACCURACY: Are there any factual errors or unsupported claims?
3. CLARITY: Is the answer easy to understand?
4. STRUCTURE: Is it well-organized with appropriate formatting?
5. ACTIONABILITY: Can the user act on this information?

Output your review as:
{
  "is_satisfied": true/false,
  "score": 1-10,
  "feedback": "Specific, actionable critique. If satisfied, explain why.",
  "missing_elements": ["list", "of", "missing", "things"]
}

Be honest but constructive. A score of 10 means the answer is perfect.
```

### When Reflection Shines
- High-stakes answers (medical, legal, financial)
- Content creation (blog posts, reports, documentation)
- Tasks where quality matters more than latency
- Debugging: the critique tells you why the answer is bad

### When Reflection Fails
- Real-time applications (doubles or triples latency)
- Simple factual queries (unnecessary overhead)
- Tasks where the model can't judge its own quality

> **Code Reference:** [Python](../../code/python/03-agent-loop/reflection_agent.py)  
> Includes the full `ReflectionAgent` class with `CritiqueResult`, `max_reflections`, and `quality_threshold`.

---

## Choosing a Strategy

```
Start with ReAct. Upgrade when it breaks.

Is the task multi-step with clear dependencies?
    → Plan-and-Execute

Is quality more important than speed?
    → Add Reflection to your existing strategy

Does the plan change based on intermediate results?
    → Stick with ReAct (Plan-and-Execute over-plans)

Are subtasks independent?
    → Plan-and-Execute with parallel execution

Is the user waiting in real-time?
    → ReAct (fastest time-to-first-response)
```

| Your Situation | Strategy | Why |
|:---|:---|:---|
| Building your first agent | ReAct | Simplest. Ship something. |
| Agent gets lost on complex tasks | Plan-and-Execute | Structure keeps it on track. |
| Agent produces sloppy answers | Reflection | Self-critique catches errors. |
| Need both | Plan-and-Execute + Reflection | Plan first, execute, then reflect. |

---

## Combining Strategies

Real agents combine strategies. Here's a production pattern:

```
1. CLASSIFY the user's request (simple vs complex)
2. If simple → ReAct (fast path)
3. If complex → Plan-and-Execute (structured path)
4. After final answer → Reflection (quality gate)
5. If reflection score < 7 → Revise and reflect again
```

This is the router pattern, covered in [Harness Engineering](../04-harness-engineering/03-routing-and-intent-classification.md). The harness decides which strategy to use based on the task.

---

## Common Pitfalls

- **"ReAct gets stuck in a loop"**: The model keeps calling the same tool with the same parameters. Track the last three tool calls; if all three share the same name *and* arguments, inject a system message: `"You have called {tool} with the same parameters three times. Explain what is missing and try a different approach."`
- **"The plan is too vague"**: "Research AI chips" isn't a step. Force the planner to specify exactly which tool to call with which parameters. Use structured output with a step schema.
- **"Reflection makes it worse"**: The critic is too strict or too lenient. Too strict: the agent revises forever. Too lenient: no improvement. Tune the reflection prompt and set a max reflection count (2 is usually right).
- **"Plan-and-Execute is too slow"**: You're executing steps sequentially that could run in parallel. Mark independent steps in the plan and use parallel tool calls.
- **"The agent over-plans for simple questions"**: "What's 2+2?" doesn't need a 5-step plan. Classify the task before choosing a strategy. Simple tasks go straight to ReAct.

## What's Next

You can now design an agent that plans, executes, and critiques its own work. Next: scaling from one agent to many — when and how to use multiple agents.
→ [Multi-Agent Patterns](04-multi-agent-patterns.md)