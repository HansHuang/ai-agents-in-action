# Skills: The Unit of Agent Capability

## What You'll Learn
- Why raw tools aren't enough for production
- The anatomy of a skill: tool + prompt + validation + fallback
- How skills make agents reusable, testable, and composable
- Skill discovery and registration patterns
- Testing skills in isolation before adding them to an agent
- The skill as the boundary between AI engineering and software engineering

## Prerequisites
- [Tool Design Patterns](02-tool-design-patterns.md) — tools are the foundation skills wrap
- [Anatomy of an AI Agent](01-anatomy-of-an-agent.md) — skills plug into the agent loop
- [Structured Output](../01-foundations/03-structured-output.md) — skill outputs are structured

---

## The Problem Skills Solve

You've built tools. You've designed them carefully with clear descriptions, parameter schemas, and error handling. But in production, you discover:

```python
# What you built:
def get_weather(city: str) -> dict:
    return weather_api.get(city)

# What production actually needs:
# - City name validation and disambiguation
# - Unit conversion (the user asked for Fahrenheit, the API returns Celsius)
# - Graceful fallback when the API is down
# - Caching for repeated queries
# - Logging for observability
# - A specific response format the agent expects
# - Instructions for the LLM on how to interpret the result
```

You can't put all of this in a tool. Tools are functions. What you need is a **skill**: a self-contained capability unit that bundles everything the agent needs to use that capability correctly.

---

## Tool vs. Skill: The Difference

| | Tool | Skill |
|:---|:---|:---|
| **What it is** | A single function | A composed capability |
| **What it contains** | Function + parameter schema | Tool + prompt fragment + input validation + output normalization + fallback + tests |
| **Who uses it** | The LLM calls it directly | The agent loads it as a capability module |
| **Reusability** | Across calls in one agent | Across agents, projects, and teams |
| **Testing** | Unit test the function | Test the tool, the prompt, the validation, and the integration |
| **Ownership** | Developer who wrote it | Can be owned by a domain expert |

```python
# A TOOL: raw, single-purpose
def get_weather(city: str) -> dict:
    return weather_api.get(city)

# A SKILL: complete, production-ready capability
weather_skill = Skill(
    name="weather_reporting",
    description="Get current weather conditions and present them clearly",
    
    # The raw tool
    tool=get_weather,
    
    # What the agent learns about this skill
    prompt_fragment="""
    When reporting weather:
    - Always include temperature in both Celsius and Fahrenheit
    - Mention humidity if it's above 80% or below 20%
    - Add a brief recommendation (umbrella? sunscreen? jacket?)
    - Format as a concise paragraph, not bullet points
    """,
    
    # Validate inputs before the tool runs
    input_validator=validate_city_with_country_code,
    
    # Normalize outputs for the agent
    output_normalizer=standardize_weather_response,
    
    # What to do when everything fails
    fallback=fallback_weather_message,
    
    # Prove it works
    test_cases=[
        SkillTest(input={"city": "Tokyo, JP"},
                  expect_output_contains=["temperature", "°C", "°F"]),
        SkillTest(input={"city": "NonexistentCity, XX"},
                  expect_fallback=True)
    ]
)
```

> **Code Reference:** [Python skill_base.py](../../code/python/09-skills/skill_base.py) · [Node.js skill_base.ts](../../code/nodejs/09-skills/skill_base.ts) · [Go skill_base.go](../../code/go/09-skills/skill_base.go)  
> Each implementation includes a Skill base class, the weather skill, a stock analysis skill (which depends on the stock price skill), and a skill registry that demonstrates testing skills independently.

---

## The Anatomy of a Skill

Every skill has six components. Not all are required, but production skills include all of them.

### 1. Metadata (Required)

```python
name: str              # Unique identifier, e.g. "weather_reporting"
description: str       # When the agent should use this skill
version: str           # Semantic version for dependency management
tags: list[str]        # For discovery: ["weather", "real-time", "public-data"]
```

### 2. Tool Definition (Required)

The function the skill wraps. Same tool definition format from Chapter 02, but now it's one piece of a larger package.

### 3. Prompt Fragment (Optional but Recommended)

Most prompts fail because they don't tell the model *how to use* a tool's output. The prompt fragment is appended to the agent's system prompt when this skill is loaded:

```python
prompt_fragment = """
When using the weather_reporting skill:
- The city parameter MUST include country code (e.g., "Tokyo, JP")
- Temperature is returned in Celsius. Convert to Fahrenheit for US users.
- If humidity > 80%, recommend an umbrella even if no rain is forecast.
- If the API returns an error, tell the user the specific issue, not a generic message.
"""
```

This is what transforms a generic agent into one that uses your tool expertly.

### 4. Input Validator (Recommended)

Runs before the tool. Catches bad parameters before they reach your API:

```python
def input_validator(params: dict) -> dict:
    city = params.get("city", "")
    if "," not in city:
        raise SkillInputError(
            message="City must include country code",
            suggestion=f"Did you mean '{city}, JP'?",
            fix_action="append_country_code"
        )
    return params
```

The validator returns corrected parameters or raises a `SkillInputError` with a suggestion the model can use to self-correct.

### 5. Output Normalizer (Recommended)

Runs after the tool. Ensures the agent always sees a consistent format:

```python
def output_normalizer(raw_result: dict) -> dict:
    return {
        "location": raw_result.get("city", "Unknown"),
        "temperature": {
            "celsius": raw_result.get("temp_c"),
            "fahrenheit": round(raw_result.get("temp_c", 0) * 9/5 + 32)
        },
        "conditions": raw_result.get("condition", "Unknown"),
        "humidity_percent": raw_result.get("humidity"),
        "reported_at": datetime.utcnow().isoformat()
    }
```

The normalizer handles missing fields, adds computed values, and strips unnecessary data.

### 6. Fallback (Optional but Critical for Production)

When the tool fails and retries are exhausted, what does the agent tell the user?

```python
def fallback(params: dict, error: Exception) -> str:
    city = params.get("city", "the specified location")
    return (
        f"Weather data for {city} is temporarily unavailable. "
        f"This could be due to an API outage or rate limiting. "
        f"Try again in a few minutes, or check a weather website directly."
    )
```

A good fallback: acknowledges the failure, explains why, and offers alternatives. The agent can pass this directly to the user without looking incompetent.

---

## The Skill Class

Here's the complete base class:

```python
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class SkillTest:
    input: dict
    expect_output_contains: list[str] | None = None
    expect_fallback: bool = False

@dataclass
class Skill:
    name: str
    description: str
    tool: Callable
    parameters: dict  # OpenAI function-calling schema
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    prompt_fragment: str | None = None
    input_validator: Callable | None = None
    output_normalizer: Callable | None = None
    fallback: Callable | None = None
    dependencies: list[str] = field(default_factory=list)
    test_cases: list[SkillTest] = field(default_factory=list)
    
    def execute(self, params: dict) -> dict:
        """Execute the skill with full validation/normalization/fallback."""
        try:
            # Validate input
            if self.input_validator:
                params = self.input_validator(params)
            
            # Execute tool
            result = self.tool(**params)
            
            # Normalize output
            if self.output_normalizer:
                result = self.output_normalizer(result)
            
            return {"success": True, "data": result}
            
        except SkillInputError as e:
            return {
                "success": False,
                "error": "invalid_input",
                "message": e.message,
                "suggestion": e.suggestion
            }
        except Exception as e:
            if self.fallback:
                return {
                    "success": False,
                    "error": "unavailable",
                    "message": self.fallback(params, e)
                }
            raise
    
    def run_tests(self) -> list[dict]:
        """Run all test cases in isolation. No agent needed."""
        results = []
        for test in self.test_cases:
            result = self.execute(test.input)
            passed = True
            if test.expect_fallback:
                passed = not result["success"]
            elif test.expect_output_contains:
                output_str = str(result.get("data", result))
                passed = all(
                    keyword in output_str 
                    for keyword in test.expect_output_contains
                )
            results.append({
                "test_input": test.input,
                "passed": passed,
                "result": result
            })
        return results
```

The key insight: `run_tests()` lets you test a skill completely independently, without an agent, without an LLM, without an API call. This is the boundary between AI engineering and software engineering.

> **Code Reference:** [skill_test_runner.py](../../code/python/09-skills/skill_test_runner.py)  
> `SkillTestRunner.run_all()` loops the registry, calls each skill's `run_tests()`, and reports pass/fail counts.  
> `run_integration_test()` chains multiple skills in sequence and checks the final output.
> Exit code 1 on failure — plug it directly into CI.

---

## Skill Composition: Skills That Use Other Skills

Skills can depend on other skills. This creates a dependency graph:

```python
weather_skill = Skill(
    name="weather_reporting",
    dependencies=["geocoding"],  # Requires geocoding skill
    tool=get_weather,
    ...
)

geocoding_skill = Skill(
    name="geocoding",
    description="Convert city names to coordinates",
    tool=geocode_city,
    ...
)

# The skill registry resolves dependencies at load time
registry = SkillRegistry()
registry.register(geocoding_skill)  # Must be registered first
registry.register(weather_skill)    # Depends on geocoding

# When executing weather_skill, the registry injects geocoding
result = registry.execute("weather_reporting", {"city": "Tokyo"})
# Internally: city → geocoding → coordinates → weather API → result
```

Dependencies are declared explicitly. The registry validates the graph at registration time — circular dependencies are rejected immediately.

> **Code Reference:** [stock_analysis_skill.py](../../code/python/09-skills/skills/stock_analysis_skill.py)  
> `create_stock_analysis_skill(registry)` captures the registry in a closure. When executed, it calls  
> `registry.execute("stock_price", ...)` internally — dependency composition without inheritance.
> [stock_price_skill.py](../../code/python/09-skills/skills/stock_price_skill.py) is the skill it depends on.

---

## Skill Discovery: How Agents Find Skills

In a single-agent system, you register skills manually. In a multi-agent system or platform, skills need to be discoverable.

### Static Registration (Simple)
```python
agent = Agent()
agent.load_skills([weather_skill, stock_skill, news_skill])
```

### Tag-Based Discovery (Medium)
```python
registry = SkillRegistry()
registry.register_many(all_skills)

# Agent requests: "I need real-time data skills"
relevant_skills = registry.find_by_tags(["real-time", "public-data"])
agent.load_skills(relevant_skills)
```

### Vector-Based Discovery (Advanced)
```python
# Embed skill descriptions into a vector database
# Given a user query, retrieve the most relevant skills
# "What's the weather?" → cosine similarity → weather_skill (0.97), news_skill (0.32)
```

Tag-based discovery is the right starting point for most systems. Vector-based discovery becomes valuable when you have 50+ skills.

> **Code Reference:** [skilled_agent.py](../../code/python/09-skills/skilled_agent.py)  
> `SkilledAgent.load_skills()` calls `registry.resolve_dependencies()` for each requested skill,  
> ensuring dependencies are always loaded before the skills that need them.

---

## Testing Skills: The Killer Feature

You can test a skill without an agent. No LLM calls. No API keys. This is the superpower of the skill abstraction.

```python
def test_weather_skill():
    skill = SkillRegistry().get("weather_reporting")
    
    # Test 1: Happy path
    result = skill.execute({"city": "Tokyo, JP"})
    assert result.success
    assert "temperature" in result.data
    assert "fahrenheit" in str(result.data)
    
    # Test 2: Invalid input
    result = skill.execute({"city": "Tokyo"})  # No country code
    assert not result.success
    assert result.error_type == "invalid_input"
    assert "country code" in result.error
    
    # Test 3: API failure
    with mock.patch.object(skill, 'tool', side_effect=TimeoutError):
        result = skill.execute({"city": "Tokyo, JP"})
        assert not result.success
        assert result.error_type == "unavailable"
        assert "temporarily unavailable" in result.error
    
    # Test 4: Output format
    result = skill.execute({"city": "London, GB"})
    assert result.data["temperature"]["celsius"] is not None
    assert result.data["temperature"]["fahrenheit"] is not None
```

No agent. No LLM. Just a skill and its contract. This is how you ship reliable AI features.

---

## When to Make a Skill vs. Keep a Tool

| Keep as a Tool | Make a Skill |
|:---|:---|
| Prototyping | Production deployment |
| The tool has no failure modes | The tool needs graceful degradation |
| One agent, one use case | Reusing across multiple agents |
| You own all the code | Teams contribute capabilities independently |
| Simple data lookup | Complex data processing with formatting rules |
| Internal use only | External-facing or customer-impacting |

**Rule of thumb:** If you would write a design doc for this capability, it's a skill. If you'd just write a function, it's a tool.

---

## The Skill Manifesto

1. **A skill is testable in isolation.** No agent required.
2. **A skill owns its prompt.** The agent doesn't need to know how to use the tool — the skill teaches it.
3. **A skill handles its own failures.** The agent never sees a raw stack trace.
4. **A skill has a defined contract.** Input types, output types, error types — all explicit.
5. **A skill is versioned.** Breaking changes to a skill's contract require a new version.
6. **A skill is composable.** It can depend on other skills via the registry, building capability layers without inheritance.

---

## Common Pitfalls

- **"My skill's prompt fragment contradicts the agent's system prompt"**: The skill's prompt fragment and the agent's system prompt must be compatible. If the agent says "Be concise" and the skill says "Provide detailed weather analysis," the model gets confused. Test skills with the agent they'll be used with.
- **"I made everything a skill and now it's slow"**: Skills have overhead — validation, normalization, logging. A simple `get_time()` function doesn't need to be a skill. Reserve skills for capabilities with meaningful failure modes.
- **"My skill's fallback is worse than no fallback"**: "An error occurred" is not a fallback. A good fallback gives the agent something specific to tell the user and suggests next steps.
- **"Skill dependencies create a dependency hell"**: Keep skill dependency graphs shallow (max 2 levels). If a skill depends on a skill that depends on a skill that depends on a skill, reconsider your design.
- **"I test my skill without mocking external APIs"**: Your skill tests should mock everything. Skills wrap external services; tests should not require those services to be running.

## What's Next

Skills are the unit of reuse. Next, we move to the ecosystem around agents: model providers, vector databases, and the observability tools you need to run agents in production.
→ [Model Providers](../05-the-tool-ecosystem/01-model-providers.md)