# Tool Design Patterns

## What You'll Learn
- The anatomy of a tool: name, description, parameters, and return value
- Why the description is the most important code you'll write
- Designing tools the model won't misuse: distinct names, clear boundaries, explicit constraints
- Handling errors: when tools fail, the model must know why
- Composing tools: how many is too many, and how to group them
- Tools vs. Skills: the unit of reuse (teaser for Chapter 05)

## Prerequisites
- [Anatomy of an AI Agent](01-anatomy-of-an-agent.md) — the orchestration loop that calls tools
- [Structured Output](../01-foundations/03-structured-output.md) — tool calls are structured output

---

## A Tool Is a Contract

A tool is an API that your code exposes to an LLM. But unlike a REST API consumed by a human developer, the LLM:

- Cannot read your source code
- Cannot inspect your runtime
- Cannot ask clarifying questions (unless you design for it)
- Will guess parameters if it's unsure

This means your tool definition must be **entirely self-documenting**. The LLM decides whether to call your tool based solely on three things:

1. The tool's **name**
2. The tool's **description**
3. The **parameter schemas**

If any of these are ambiguous, the model will guess. And it will guess wrong.

---

## The Anatomy of a Tool

```python
{
    "type": "function",
    "function": {
        "name": "search_customer_orders",
        "description": "Search for customer orders by email address. Returns up to 50 most recent orders with order ID, date, status, and total amount.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Customer's email address. Must be a valid email format. Example: 'customer@example.com'"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "shipped", "delivered", "cancelled"],
                    "description": "Filter orders by status. If omitted, returns all statuses."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of orders to return. Defaults to 10 if not specified."
                }
            },
            "required": ["email", "status", "limit"],
            "additionalProperties": False
        }
    }
}
```

Every field earns its place:

| Field | Why It Matters |
|:---|:---|
| `name` | Must be unique across all tools. The model uses this to decide *which* tool to call. |
| `description` | Explains *what* the tool does and *when* to use it. Include what it returns. |
| `strict: true` | Forces the model to use *exactly* the declared schema — no invented keys, no omitted required fields. Requires all parameters to be in `required` and `additionalProperties: false`. |
| `parameters.properties` | Each parameter needs its own description with an example value. |
| `parameters.required` | Tells the model which arguments are mandatory. With `strict: true`, every parameter must be listed here — use `null` as the type for truly optional ones. |
| `additionalProperties: false` | Required for `strict: true`. Prevents the model from sending keys you didn't declare. |
| `enum` | Constrains the model to valid values. Without this, the model invents statuses. |
| `minimum`/`maximum` | Prevents nonsensical values. The model will ask for 10,000 orders unless you stop it. |

---

## The Description Is the Most Important Code You'll Write

A bad description:
```
"Gets the weather."
```
The model doesn't know: What parameters does it need? What format does the city need to be in? Does it return Celsius or Fahrenheit? Current weather or forecast?

A good description:
```
"Get current weather conditions for a city. Returns temperature (Celsius and Fahrenheit), humidity percentage, wind speed (km/h), and a brief condition description (e.g. 'partly cloudy'). The city parameter must include the country code for accuracy, e.g. 'Shanghai, CN' not just 'Shanghai'."
```

### The Description Checklist

Every tool description must answer:
- [ ] **What does this tool do?** (One sentence summary at the start)
- [ ] **What does it return?** (List the key fields in the response)
- [ ] **When should the model use it?** (What user question triggers this tool?)
- [ ] **When should the model NOT use it?** (Boundaries — "This does not return forecast data")
- [ ] **Are there gotchas?** ("City must include country code", "Date must be in YYYY-MM-DD format")

---

## Parameter Design: Make Ambiguity Impossible

The model will fill in parameters based on the user's question. If the parameter description is vague, the model invents values.

### Bad Parameters
```json
{
    "city": {
        "type": "string",
        "description": "The city"
    }
}
```
The model will send `"New York"` when it should send `"New York, NY, US"`. Your weather API returns results for York, England.

### Good Parameters
```json
{
    "city": {
        "type": "string",
        "description": "City name with country code for disambiguation. Format: 'City, CountryCode'. Examples: 'Shanghai, CN', 'New York, US', 'London, GB'"
    }
}
```
Now the model knows the format. It infers `"New York, US"` from context.

### The Parameter Checklist

Every parameter must specify:
- [ ] **Type** (string, number, integer, boolean, array, object)
- [ ] **Format** with an example value
- [ ] **Constraints** (enum, minimum, maximum, pattern)
- [ ] **Default behavior** when omitted (in the description)
- [ ] **Required or optional** (in the `required` array)

---

## Error Handling: Teach the Model to Recover

When a tool fails, the model needs to understand *why* it failed and *what to do next*. A generic error message produces a confused agent. A descriptive error produces a self-correcting agent.

### Bad Error Response
```python
# Tool raises an exception, your dispatcher catches it:
return {"error": "Tool execution failed"}
```
The model's response: *"I'm sorry, something went wrong. Please try again."* — It doesn't know what to fix.

### Good Error Response
```python
# Tool catches the specific error, returns structured feedback:
return {
    "error": "city_not_found",
    "message": "Could not find weather data for 'Shahai, CN'. Did you mean 'Shanghai, CN'?",
    "suggestion": "Shanghai, CN"
}
```
The model's response: *"I couldn't find weather for 'Shahai'. Let me try 'Shanghai, CN' instead."* — It self-corrects.

### Error Response Design Pattern

Every tool should return a consistent error structure:

```python
def tool_with_error_handling(params):
    try:
        result = call_external_api(params)
        return {"success": True, "data": result}
    except NotFoundError as e:
        return {
            "success": False,
            "error": "not_found",
            "message": str(e),
            "suggestion": e.best_match  # What the model should try instead
        }
    except InvalidInputError as e:
        return {
            "success": False,
            "error": "invalid_input",
            "message": str(e),
            "valid_values": e.allowed_values  # What the model should choose from
        }
```

> **Code Reference:** [Python](../../code/python/03-agent-loop/tool_registry.py) · [Node.js](../../code/nodejs/03-agent-loop/tool_dispatcher.ts) · [Go](../../code/go/03-agent-loop/tool_dispatcher.go)  
> `ToolRegistry.execute_tool()` (Python) and `DispatchTool()` (Go/TypeScript) implement this exact pattern: `NotFoundError` maps to `not_found`, `InvalidArgsError` to `invalid_args`, and any other exception to `internal_error`.

---

## Naming Tools: Distinct and Discoverable

The model uses tool names to decide which tool to call. Similar names cause confusion.

### Bad Naming
```
get_order
get_order_details
get_order_info
```
The model doesn't know which to use. It guesses. Sometimes it calls all three.

### Good Naming
```
search_orders       # Search by customer, returns list
get_order_by_id     # Fetch one order, requires exact ID
cancel_order        # Action, not a query
```
Each name implies: **what it does** + **what it needs**. The model can distinguish them.

### Naming Conventions

| Pattern | Example | When to Use |
|:---|:---|:---|
| `get_<entity>` | `get_weather` | Simple lookup with one identifier |
| `search_<entities>` | `search_products` | Query with filters, returns multiple |
| `create_<entity>` | `create_ticket` | Creating a new resource |
| `update_<entity>` | `update_order_status` | Modifying an existing resource |
| `delete_<entity>` | `delete_subscription` | Removing a resource |
| `generate_<thing>` | `generate_report` | Computation or generation |

Pick a convention and enforce it across all tools. The model learns patterns from your naming.

---

## How Many Tools Is Too Many?

Every tool in the definition is sent to the model on every API call. Tools consume context window tokens. More tools = slower responses, higher costs, and more opportunity for the model to call the wrong one.

### The Numbers That Matter

These are conservative baselines for well-named, well-described tools. Frontier models (GPT-4o, o3, Gemini 2.x) handle the upper end of each range reliably; smaller or fine-tuned models may struggle at lower counts.

| Tool Count | Behavior | Recommendation |
|:---|:---|:---|
| 1–5 | Model handles easily | No restrictions needed |
| 6–15 | Model still performs well | Group similar tools, use clear naming |
| 16–30 | Occasional wrong-tool selection | Consider intent routing; pre-filter to the 5–10 relevant tools |
| 30+ | Selection accuracy drops | Split into multiple agents or use embedding-based tool retrieval |

### When You Have Too Many Tools

**Option A: Group and Route**
Instead of giving the model 50 tools, classify the user's intent first, then give the model only the 5 relevant tools.

**Option B: Specialized Agents**
Create separate agents for separate domains. A "Customer Support Agent" gets customer tools. A "Data Analysis Agent" gets analytics tools.

**Option C: Tool Retrieval**
Store tool descriptions in a vector database. Retrieve the 5 most relevant tools based on the user's query. Only send those to the model.

> Multi-agent routing is covered in [Multi-Agent Patterns](04-multi-agent-patterns.md). Tool retrieval is covered in [RAG from Scratch](../03-memory-and-retrieval/03-rag-from-scratch.md).

---

## Tools vs. Skills: A Preview

A tool is a single function. A **skill** is a tool wrapped with its own prompt fragment, validation logic, fallback behavior, and test suite.

```python
# A TOOL: raw, single-purpose
def get_weather(city: str) -> dict:
    return weather_api.get(city)

# A SKILL: tool + system prompt + validation + fallback
weather_skill = Skill(
    name="weather_reporting",
    tool=get_weather,
    prompt_fragment="When reporting weather, always include temperature in both C and F.",
    input_validator=validate_city_format,
    fallback=lambda city: f"Could not get weather for {city}. Try a different city name.",
    test_cases=[
        ("Shanghai, CN", {"temp_c": 22, "condition": "sunny"}),
        ("InvalidCity", "fallback_message")
    ]
)
```

Skills are the unit of reuse in production agent systems. They are covered in detail in [Skills: The Unit of Agent Capability](05-skills-composing-capabilities.md).

---

## Tool Definition Template

Use this template for every tool you build. Fill it out before writing code.

```markdown
## Tool: [name]

### Purpose
[One sentence: what does this tool do?]

### When to Use
[What user intent triggers this tool?]

### When NOT to Use
[Boundaries: what this tool does NOT do]

### Parameters
| Name | Type | Required | Description | Example |
|:---|:---|:---|:---|:---|
| [param1] | string | yes | [description with format] | "example_value" |
| [param2] | integer | no | [description with default] | 42 |

### Success Response
```json
{"success": true, "data": {...}}
```

### Error Responses
| Error Code | Meaning | Model Should... |
|:---|:---|:---|
| not_found | Resource doesn't exist | Ask user for clarification |
| invalid_input | Parameter format wrong | Retry with corrected parameter |
| unavailable | External service down | Tell user, suggest retry later |
```

> **Code Reference:** [Python `tool_builder.py`](../../code/python/03-agent-loop/tool_builder.py) · [Node.js `tool_builder.ts`](../../code/nodejs/03-agent-loop/tool_builder.ts) · [Go `tool_builder.go`](../../code/go/03-agent-loop/tool_builder.go)  
> `ToolDef` / `Param` implement this template as code: `to_openai_schema()` generates the JSON definition and `validate_args()` enforces the constraints at runtime. Use `tool_validator.py` to score an existing definition against this checklist.

---

## Common Pitfalls

- **"The model calls the wrong tool"**: Your tool descriptions overlap. If two tools could both answer "find me my order," the model guesses. Make descriptions mutually exclusive: "This tool searches by email. For order ID lookup, use get_order_by_id."
- **"The model invents parameters"**: Your parameter descriptions lack constraints. Add enums, min/max, and format examples. If the model sends a date as "next Tuesday" instead of "2026-05-07", your description didn't specify the format.
- **"The model never uses my tool"**: Your description doesn't match how users ask questions. If users say "what's my balance" but your tool is called `get_account_balance`, add "Use this when users ask about their balance, funds, or account status."
- **"Tool calls succeed but the model ignores the result"**: Your return value is too verbose or too sparse. Return the data the model needs to answer the user — no more, no less. 50KB of order data is noise. A clean summary with order IDs is signal.
- **"Tool errors crash the agent"**: You're letting exceptions propagate. Every tool must catch exceptions and return structured error messages. The model cannot recover from a stack trace.

## What's Next

You can now design tools that the model uses correctly. Next: how the agent decides *which* action to take and in what order — planning strategies that go beyond simple ReAct.
→ [Planning Strategies](03-planning-strategies.md)