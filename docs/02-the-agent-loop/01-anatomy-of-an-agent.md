# Anatomy of an AI Agent

## What You'll Learn
- The four components every agent has: Brain, Hands, Memory, Loop
- The orchestration loop: the deterministic engine that drives the probabilistic brain
- How an agent decides what to do next (and when to stop)
- The ReAct pattern: reasoning and acting in a single loop
- Building your first agent from scratch — no frameworks

## Prerequisites
- [Structured Output](../01-foundations/03-structured-output.md) — the agent's decisions must be parseable
- [Prompt Engineering](../01-foundations/02-prompt-engineering.md) — the system prompt is the agent's constitution
- [How LLMs Actually Work](../01-foundations/01-how-llms-work.md) — tokens, messages, the API call

---

## What Is an AI Agent?

An AI agent is a system where an LLM serves as the reasoning engine, deciding what actions to take and in what order, to achieve a goal.

Here's the definition that matters for engineers:

> **An agent is an orchestration loop that repeatedly calls an LLM, parses its output, executes tools based on that output, feeds the results back, and stops when the goal is reached.**

That's it. No magic. Just a `while` loop with an LLM inside it.

---

## The Four Components

Every agent, from a 50-line script to a production system, has four parts:

```
┌────────────────────────────────────────────────┐
│                   AGENT                        │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐    │
│  │  BRAIN   │  │  HANDS   │  │   MEMORY   │    │
│  │  (LLM)   │  │ (Tools)  │  │ (Messages) │    │
│  └─────┬────┘  └────┬─────┘  └─────┬──────┘    │
│        │            │              │           │
│        └────────────┼──────────────┘           │
│                     │                          │
│              ┌──────┴──────┐                   │
│              │    LOOP     │                   │
│              │ (orchestr.) │                   │
│              └─────────────┘                   │
└────────────────────────────────────────────────┘
```

### 1. The Brain (LLM)

The brain doesn't *do* anything. It only decides. Given the current state (messages, available tools, conversation history), it outputs one of two things:
- **A function call:** "I need to run `get_weather` with `city='Shanghai'`"
- **A final answer:** "Here's what I found: Shanghai is 22°C."

Your job is to interpret that decision.

### 2. The Hands (Tools)

Tools are the functions you give the agent. The LLM can't send an email or query a database. It can only request that *you* do it on its behalf.

A tool is a contract:
```python
def get_weather(city: str) -> dict:
    """
    Gets current weather for a city.
    
    Args:
        city: City name and country code, e.g. "Shanghai, CN"
    
    Returns:
        dict with keys: temperature, condition, humidity
    """
    # Your API call here
```

The docstring is not documentation for developers. It's instructions for the LLM. Every word matters.

### 3. The Memory (Message History)

The agent's short-term memory is the message list. Every API call sends the entire history:

```python
messages = [
    {"role": "system", "content": "You are a professional weather assistant with tools."},
    {"role": "user", "content": "What's the weather in Shanghai?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "Shanghai, CN"}'}}
    ]},
    {"role": "tool", "content": '{"temperature": 22, "condition": "sunny"}', 
     "tool_call_id": "call_123"},
    {"role": "assistant", "content": "Shanghai is 22°C and sunny."}
]
```

This list grows with every interaction. When it exceeds the context window, you must trim, summarize, or compress — which is why memory management (Chapter 03) exists.

### 4. The Loop (Orchestrator)

The loop is the only deterministic component. It's a `while` loop that cycles through four phases until the goal is met:

```
while not agent.is_finished:
    response = call_llm(messages, tools)     # Phase 1: Reason
    if response has tool_calls:              # Phase 2: Decide
        for tool_call in tool_calls:
            result = execute_tool(tool_call) # Phase 3: Act
            messages.append(result)          # Phase 4: Observe
    else:
        agent.is_finished = True             # Final answer received
```

---

## The Orchestration Loop in Detail

Let's trace a single turn. A user asks: *"What's the weather in Shanghai and should I take an umbrella?"*

### Turn 1: The Initial Call

**Messages sent to LLM:**
```
System: You are a helpful assistant. Use tools when needed.
User: What's the weather in Shanghai and should I take an umbrella?
```

**LLM responds:**
```json
{
  "tool_calls": [{
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"Shanghai, CN\"}"
    }
  }]
}
```

The LLM didn't answer the question. It asked for a tool. Your loop must handle this.

### Turn 2: Execute and Observe

**Your loop executes the tool:**
```python
result = get_weather(city="Shanghai, CN")
# result = {"temperature": 22, "condition": "rain", "humidity": 85}
```

**Your loop appends the result to messages:**
```python
messages.append({
    "role": "tool",
    "content": json.dumps(result),
    "tool_call_id": tool_call.id
})
```

### Turn 3: The Follow-Up Call

**Messages now sent to LLM (entire history):**
```
System: You are a helpful assistant. Use tools when needed.
User: What's the weather in Shanghai and should I take an umbrella?
Assistant: [called get_weather("Shanghai, CN")]
Tool: {"temperature": 22, "condition": "rain", "humidity": 85}
```

**LLM responds (final answer):**
```
Shanghai is currently 22°C with rain and 85% humidity. 
Yes, you should take an umbrella.
```

**Your loop detects:** No tool call → this is the final answer → exit loop → return to user.

---

## The Minimal Agent: 30 Lines of Python

```python
import json
from openai import OpenAI

client = OpenAI()
messages = [{"role": "system", "content": "You are helpful. Use tools when needed."}]

def get_weather(city: str) -> dict:
    # In reality, call a weather API
    return {"temperature": 22, "condition": "rain", "city": city}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Shanghai, CN'"}
            },
            "required": ["city"]
        }
    }
}]

def run_agent(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools
        )
        msg = response.choices[0].message
        
        if msg.tool_calls:
            # IMPORTANT: append the assistant message BEFORE any tool results.
            # Skipping this step causes an API error:
            # "tool messages must be a response to a preceding message with tool_calls"
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = get_weather(**args)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })
            continue  # Back to the top of the loop
        
        return msg.content  # Final answer
```

This is an agent. It reasons, calls tools, observes results, and delivers an answer. Everything else — multi-agent systems, RAG, harnesses — is just adding more tools, more memory, and more control to this loop.

> **Code Reference:** [Python](../../code/python/03-agent-loop/) · [Node.js](../../code/nodejs/03-agent-loop/) · [Go](../../code/go/03-agent-loop/)  
> Each implementation contains the full working agent with weather and stock price tools, error handling, and a test harness that verifies the loop exits correctly.

---

## The ReAct Pattern: Reasoning and Acting

The pattern you just saw has a name: **ReAct** (Reasoning + Acting). It's the default architecture for single-agent systems.

```
User: "Book me a flight to London next Tuesday"
     │
     ▼
┌─────────────┐
│   REASON    │  LLM thinks: "I need to search flights first"
└──────┬──────┘
       ▼
┌─────────────┐
│    ACT      │  Execute: search_flights(destination="London", date="...")
└──────┬──────┘
       ▼
┌─────────────┐
│   OBSERVE   │  Result: "3 flights found, prices $200-$500"
└──────┬──────┘
       ▼
┌─────────────┐
│   REASON    │  LLM thinks: "I have flights, now I should present options"
└──────┬──────┘
       ▼
     Final answer to user
```

ReAct isn't the only pattern — Chapter 03 covers planning strategies including Plan-and-Execute and Reflection. But ReAct is where every agent starts.

---

## When Does the Agent Stop?

The loop ends when one of these happens:

| Stop Condition | How to Detect |
|:---|:---|
| **Final answer** | LLM response has `content` and no `tool_calls` |
| **Max iterations** | Loop counter exceeds a limit (prevents infinite loops) |
| **Timeout** | Wall-clock time exceeds a threshold |
| **Explicit stop** | LLM calls a special `task_complete` tool |
| **Human intervention** | Harness interrupts and asks for approval |

A production agent always has a safety limit. Without `max_iterations`, a confused model can loop forever, burning tokens and dollars.

```python
MAX_ITERATIONS = 10
for i in range(MAX_ITERATIONS):
    response = call_llm(messages, tools)
    if not response.tool_calls:
        return response.content
    # ... execute tools ...
raise Exception(f"Agent exceeded {MAX_ITERATIONS} iterations without finishing")
```

---

## System Prompt: The Agent's Constitution

The system prompt for an agent is fundamentally different from a regular prompt. It must teach the model how to *be* an agent:

```markdown
You are an AI assistant with access to tools.

## Your Process
1. When the user asks a question, determine if you need a tool to answer it.
2. If yes, call the appropriate tool with the correct parameters.
3. Wait for the tool result, then determine if you need more tools or can answer.
4. Never guess tool results. Always wait for the actual result.
5. If a tool fails, explain the failure to the user and suggest alternatives.

## Tool Usage Rules
- Call only one tool at a time unless they are independent.
- If you don't have enough information to call a tool, ask the user.
- Never make up parameters. If unsure, ask for clarification.

## Answer Format
- Use the tool results to answer the user's question directly.
- Cite specific data from tool results.
- If multiple tools were used, synthesize the information.
```

This prompt transforms a general-purpose LLM into an agent that understands the loop.

---

## Common Pitfalls

- **"The agent calls the same tool over and over"**: Your tool isn't returning useful enough information, or the model doesn't understand the result. Add more context to the tool's return value, or include guidance in the system prompt about when to stop calling a particular tool.
- **"The agent never calls tools"**: Your tool descriptions are too vague, or your system prompt doesn't explicitly instruct the model to use tools. The model defaults to answering from its training data.
- **"The agent calls the wrong tool"**: Your tool names are too similar, or your descriptions overlap. Tool names must be distinct. Tool descriptions must make the difference obvious.
- **"Messages list grows forever"**: You're not managing context. After the agent finishes, you should either start a new conversation or summarize the old one. Context management starts in Chapter 03.
- **"The agent stops too early"**: The model thinks it has enough information when it doesn't. Your system prompt should include: "Before giving a final answer, verify you have all the information you need. If uncertain, ask the user or call another tool."

## What's Next

You've built an agent that can reason and act. Now learn how to design the tools that make it useful — the contract between your code and the model's decisions.
→ [Tool Design Patterns](02-tool-design-patterns.md)
