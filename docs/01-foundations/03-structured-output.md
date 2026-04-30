
# Structured Output

## What You'll Learn
- Why "please respond in JSON" is not enough
- Function calling: the standard that changed everything
- JSON mode vs. schema-constrained generation
- How to enforce output schemas with Pydantic (Python), Zod (Node.js), and struct tags (Go)
- The parse-validate-retry pattern: when the model still gets it wrong

## Prerequisites
- [Prompt Engineering](02-prompt-engineering.md) — why prompt engineering alone is fragile
- [How LLMs Actually Work](01-how-llms-work.md) — tokens, API calls, the message array

---

## The Problem: Prompts Are Probabilistic

You wrote the perfect prompt:
```
Please respond in valid JSON with the following structure:
{"sentiment": "positive" | "negative" | "neutral", "confidence": 0.0-1.0}
```

The model responds:
```json
Here's my analysis:
{"sentiment": "positive", "confidence": 0.87}
```

That leading text — `Here's my analysis:` — just broke your `JSON.parse()`. This happens constantly in production. The fix is not a better prompt. The fix is structured output.

---

## The Three Levels of Output Control

| Level | Mechanism | Reliability | When to Use |
|:---|:---|:---|:---|
| **1. Prompt Engineering** | "Please respond in JSON" | ~90% | Prototyping, internal tools |
| **2. JSON Mode** | `response_format: { type: "json_object" }` | ~98% | When structure is simple and you validate anyway |
| **3. Structured Output / Function Calling** | `tools` with schema, or `response_format` with `json_schema` | ~99.9% | Production. Always. |

You want Level 3 for anything that touches an agent, a tool, or another system.

---

## Function Calling: The Accidental Structured Output Standard

Function calling was designed to let models request tool executions. But it has a side effect that turned out to be more important: **it forces the model to output parseable JSON that matches your schema.**

### How It Works

You don't ask the model to call a function. You tell the model: *"You have access to this function. If you need it, output a function call in this exact format."*

```python
tools = [{
    "type": "function",
    "function": {
        "name": "classify_sentiment",
        "description": "Classify the sentiment of a text",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["sentiment", "confidence"]
        }
    }
}]
```

The model's response is not text. It's a structured function call:
```json
{
  "name": "classify_sentiment",
  "arguments": "{\"sentiment\":\"positive\",\"confidence\":0.87}"
}
```

You never parse this by hand. The SDK gives you a parsed object.

> **The key insight:** You don't have to execute the function. You can define a function purely to get structured output, then ignore the "call" and just read the arguments. This is the standard pattern for structured extraction.

### When to Actually Call the Function vs. Just Extract Data

| Pattern | You Define a Function | You Execute It | Example |
|:---|:---|:---|:---|
| **Structured Extraction** | Yes | No — you read the arguments | Sentiment analysis, entity extraction |
| **Tool Use** | Yes | Yes — you call your code, return result | Database query, API call, file read |

---

## JSON Mode vs. Schema-Constrained Generation

OpenAI offers two paths to structured output. Understand the difference:

### JSON Mode
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={"type": "json_object"}
)
```
- Model outputs valid JSON (no leading text)
- No schema enforcement — the model decides the keys and types
- You must still validate the structure yourself

### Schema-Constrained (Structured Outputs)
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_response",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"}
                },
                "required": ["sentiment", "confidence"],
                "additionalProperties": False
            }
        }
    }
)
```
- Model outputs JSON matching your exact schema
- `additionalProperties: false` means no surprise keys
- `enum` means only those exact values
- The API rejects the model's output if it doesn't match — before you ever see it

> **Rule of thumb:** If you have a schema, use schema-constrained. JSON mode is for when you genuinely need flexible output that happens to be JSON.

---

## The Parse-Validate-Retry Pattern

Even at 99.9% reliability, 0.1% of requests will fail. In a system handling millions of calls, that's thousands of failures. You need a fallback.

```python
import json
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def extract_structured(
    messages: list[dict],
    output_model: Type[T],
    max_retries: int = 3
) -> T:
    """
    Extract structured data with automatic retry on validation failure.
    
    On failure, appends the validation error to messages and retries,
    giving the model a chance to self-correct.
    """
    for attempt in range(max_retries):
        response = call_llm_with_schema(messages, output_model)
        raw = response.choices[0].message.content
        
        try:
            return output_model.model_validate_json(raw)
        except ValidationError as e:
            if attempt == max_retries - 1:
                raise
            # Give the model its own error and let it fix it
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Invalid output. Errors: {e.errors()}. Please fix and retry."
            })
```

> **Code Reference:** [Python `retry_handler.py`](../../code/python/02-structured-output/retry_handler.py) · [Node.js `retry_handler.ts`](../../code/nodejs/02-structured-output/retry_handler.ts) · [Go `retry_handler.go`](../../code/go/02-structured-output/retry_handler.go)  
> Each implementation is a standalone reusable module with logging, human-readable error injection, and language-appropriate validation (Pydantic, Zod, struct tags).

---

## Language-Specific Patterns

### Python: Pydantic + Instructor

[Instructor](https://python.useinstructor.com/) wraps the API call and handles parsing, validation, and retries automatically:

```python
import instructor
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0, le=1)

client = instructor.from_openai(OpenAI())
result = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "I love this product!"}],
    response_model=SentimentResponse,
    max_retries=2,
)
# result is a validated SentimentResponse instance. No parsing needed.
```

> **Code Reference:** [`instructor_extraction.py`](../../code/python/02-structured-output/instructor_extraction.py) — full implementation with field descriptions and a `main()` demo.

### Node.js: Zod + OpenAI SDK

```typescript
import { z } from "zod";
import OpenAI from "openai";

const SentimentSchema = z.object({
  sentiment: z.enum(["positive", "negative", "neutral"]),
  confidence: z.number().min(0).max(1),
});

const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "I love this product!" }],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "sentiment",
      schema: zodToJsonSchema(SentimentSchema), // use zod-to-json-schema package
      strict: true,                             // reject output that doesn't match
    },
  },
});

const result = SentimentSchema.safeParse(
  JSON.parse(response.choices[0].message.content)
);
if (!result.success) throw new Error(result.error.message);
```

> **Code Reference:** [`zod_extraction.ts`](../../code/nodejs/02-structured-output/zod_extraction.ts) — full TypeScript implementation with retry loop. [`retry_handler.ts`](../../code/nodejs/02-structured-output/retry_handler.ts) — generic reusable handler.

### Go: Struct Tags + JSON Schema

```go
type SentimentResponse struct {
    Sentiment  string   `json:"sentiment"            jsonschema:"enum=positive,enum=negative,enum=neutral"`
    Confidence float64  `json:"confidence"           jsonschema:"minimum=0,maximum=1"`
    KeyPhrases []string `json:"key_phrases,omitempty" jsonschema:"description=Up to 5 key phrases"`
}

// GenerateJSONSchema() reads jsonschema struct tags via reflection to build
// a strict JSON Schema. Pass the result to the API as json_schema.schema.
// After json.Unmarshal, validate enum values and numeric ranges manually.
```

> **Code Reference:** [`structured_extraction.go`](../../code/go/02-structured-output/structured_extraction.go) — full implementation including `GenerateJSONSchema`, manual validation, and retry loop.

---

## Function Calling vs. Structured Output: The Real Difference

In late 2024, OpenAI introduced a dedicated `response_format` with `json_schema`. Anthropic has native structured output. But function calling still works everywhere. Here's when to use each:

| Use Function Calling When... | Use Structured Output When... |
|:---|:---|
| You need compatibility across providers | You're on a provider that supports it natively |
| The output might trigger an actual tool execution | You purely want structured data extraction |
| Your framework (LangChain, etc.) uses function calling internally | You want the simplest possible code |
| The model needs to choose between multiple output schemas | You always want the same schema |

For agents, function calling is the standard because agents *are* tool-calling systems. The structured output is a side effect of the tool contract.

---

## Common Pitfalls

- **"I'll just use JSON mode and validate"**: JSON mode only guarantees valid JSON, not your schema. `{"foo": "bar"}` is valid JSON. It's not your `SentimentResponse`. Use schema-constrained or function calling.
- **"The model keeps outputting the wrong enum value"**: Your enum descriptions are ambiguous. The model doesn't know what "positive" means unless you define it. Add descriptions to every enum value.
- **"The schema works in dev but fails in production"**: You're not using `additionalProperties: false`. The model hallucinated a new field and your parser silently ignored it — until it didn't.
- **"Validation errors don't help the model self-correct"**: Your error messages are too technical. Instead of `"Field 'confidence' violates minimum 0"`, send `"Confidence must be a number between 0 and 1. You sent -0.5."`
- **"I'm using structured output so I don't need error handling"**: The 0.1% will happen. Always have a retry loop. Always have a fallback.

## What's Next

You can now extract guaranteed structured data from an LLM. The next step is building the loop that uses this data to make decisions and take actions.
→ [Anatomy of an Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md)