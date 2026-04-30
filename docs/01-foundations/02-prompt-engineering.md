# Prompt Engineering

## What You'll Learn
- The anatomy of a prompt: system, user, and assistant messages
- Why the system prompt is the most important piece of text you'll write
- Few-shot prompting: teaching by example
- Chain-of-thought: making the model show its work
- When prompt engineering ends and structured output begins

## Prerequisites
- [How LLMs Actually Work](01-how-llms-work.md) — tokens, temperature, the API call lifecycle

---

## The Anatomy of a Prompt

Every LLM API call sends an array of messages. Each message has a `role` and `content`. There are only three roles that matter:

| Role | Who | Purpose | Example |
|:---|:---|:---|:---|
| `system` | You (the developer) | Set the rules of engagement | "You are a helpful assistant that answers in JSON." |
| `user` | The end user | Ask the question | "What's the weather in Shanghai?" |
| `assistant` | The model | Previous responses (for multi-turn) | "The weather in Shanghai is 22°C, partly cloudy." |

```python
messages = [
    {"role": "system", "content": "You are a concise weather reporter."},
    {"role": "user", "content": "What's the weather in Shanghai?"}
]
# The model fills in: {"role": "assistant", "content": "..."}
```

### Why Three Roles Exist

The model was trained on conversations structured exactly this way. The `system` role was specifically fine-tuned to carry more weight — models are trained to obey the system prompt more than the user prompt. This is your superpower. Use it.

> **Why this matters for agents:** In an agent loop, you append every tool call and tool result as messages. The `assistant` role carries the model's own reasoning forward. The `user` role can come from either a human or from tool output injected back into the conversation.

---

## The System Prompt: Your Highest-Leverage Line of Code

The system prompt is the instruction set that runs before every single message. It consumes tokens on every API call, so every word must earn its place.

### A Bad System Prompt
```
You are a helpful assistant.
```
**Problem:** Vague. The model decides what "helpful" means. No constraints. No format. No personality.

### A Good System Prompt
```
You are a customer support agent for an e-commerce platform.
Answer questions about orders, returns, and shipping.
If the user asks about a specific order, ask for the order ID.
Never make up order details. If you don't know, say so.
Respond in plain text, under 100 words.
```
**Why it works:** Role clarity. Domain boundaries. Explicit fallback behavior. Output constraint.

### The System Prompt Checklist

Every system prompt you write should answer:

- [ ] **Role:** Who is the model? (support agent, code reviewer, translator)
- [ ] **Domain:** What can it talk about? What's out of bounds?
- [ ] **Tone:** Formal? Casual? Technical?
- [ ] **Format:** Plain text? JSON? Markdown?
- [ ] **Fallback:** What happens when it doesn't know?
- [ ] **Constraints:** Word limit? No speculation? Required disclaimers?

---

## Few-Shot Prompting: Teaching by Example

Models understand instructions. They understand examples better.

### Zero-Shot (No Example)
```
User: Classify this tweet as positive, negative, or neutral: "The new update is fine, I guess."
```
The model might output: `"neutral"` — or it might output: `"This tweet expresses mild satisfaction mixed with indifference, which I would classify as neutral."` You don't know.

### Few-Shot (With Examples)
```
System: Classify tweets as positive, negative, or neutral.
Respond with exactly one word.

User: "I love this product!"        → Positive
User: "This is the worst."          → Negative
User: "It's okay, nothing special." → Neutral
User: "The new update is fine, I guess."
```
Now the model knows: respond with one word. The examples are part of the prompt, so they consume tokens, but the reliability gain is almost always worth it.

### When to Use Few-Shot
| Scenario | Use Few-Shot? |
|:---|:---|
| Simple classification | 2-3 examples, then switch to structured output (next chapter) |
| Format teaching | 1-2 examples showing exact output format |
| Edge case handling | Show the tricky case in your examples |
| Every request is unique | Not worth the tokens |

> **Code** → [`few_shot_comparison.py`](../../code/python/02-structured-output/few_shot_comparison.py) · [Node.js](../../code/nodejs/02-structured-output/) · [Go](../../code/go/02-structured-output/)  
> `few_shot_comparison.py` runs zero-shot and few-shot on the same input and prints both labels with their token counts side by side — so you can measure the reliability/cost trade-off directly.

---

## Chain-of-Thought: Making the Model Show Its Work

The single most effective prompting technique: ask the model to reason step by step before giving the final answer.

### Without Chain-of-Thought
```
User: If a shirt costs $25 and is 20% off, and I buy 3, how much do I pay?
```
Model: `"$60."` (Sometimes wrong. You can't check the reasoning.)

### With Chain-of-Thought
```
User: If a shirt costs $25 and is 20% off, and I buy 3, how much do I pay?
Think step by step before giving the final answer.
```
Model:
```
Step 1: Original price per shirt = $25
Step 2: 20% discount = $25 × 0.20 = $5 off
Step 3: Discounted price per shirt = $25 - $5 = $20
Step 4: 3 shirts = $20 × 3 = $60
Final answer: $60
```

The answer is the same, but now you can verify every step. For agents, this is critical — if the model's reasoning produces a wrong tool call, you need to see exactly where it went wrong.

### When Chain-of-Thought Helps Most

- Math and logic problems
- Multi-step reasoning
- Planning (which tool to call next)
- Any decision where you need to audit the model's reasoning

> **Why this matters for agents:** In a ReAct agent, the model's "thinking" is the plan. Chain-of-thought makes the plan visible and debuggable. Without it, you're staring at a wrong tool call with no idea why.

> **Code** → [`chain_of_thought.py`](../../code/python/02-structured-output/chain_of_thought.py) — sends the same math problem with and without CoT at `temperature=0`, printing both responses side by side.

---

## Prompt Engineering Ends Where Structured Output Begins

This is the single most important concept in this chapter.

Prompt engineering asks: *"Please format your response like this."*  
Structured output demands: *"Your response must match this schema."*

| Prompt Engineering | Structured Output |
|:---|:---|
| "Please respond in JSON" | `response_format={ "type": "json_schema", "schema": {...} }` |
| Model might add extra text | Model cannot output anything outside the schema |
| You parse the output with regex | You parse the output with a JSON parser |
| 95% reliable | 99.9% reliable |

For agents, structured output is non-negotiable. An agent calling a tool needs a guaranteed parseable function call, not a "please." The next chapter covers this in detail.

---

## Prompt Templates: The First Step Toward Context Engineering

A prompt template separates structure from content. The template is the skeleton; the variables are the data.

```python
# Bad: Hardcoded
prompt = "Summarize this article about AI agents..."

# Good: Templated
PROMPT_TEMPLATE = """
You are a technical summarizer.
Summarize the following article in 3 bullet points.
Focus on: {focus_area}

Article: {article_text}
"""

prompt = PROMPT_TEMPLATE.format(
    focus_area="practical implementation details",
    article_text=article_text
)
```

This pattern is the foundation of context engineering (Chapter 04). When you retrieve documents via RAG, you inject them into a template. When you assemble a multi-turn conversation, you're filling a template with history.

> **Security note — prompt injection:** Never interpolate raw user input directly into a system prompt template. A malicious user can enter `Ignore all previous instructions` or embed `{article_text}` as literal text to manipulate the model's behavior. Validate and sanitize all inputs before templating. This is covered in depth in [Input Guardrails](../07-harness-engineering/02-input-guardrails-and-validation.md).

> **Code** → [`prompt_template.py`](../../code/python/02-structured-output/prompt_template.py) · [`prompt_template.js`](../../code/nodejs/02-structured-output/prompt_template.js) · [`prompt_template.go`](../../code/go/02-structured-output/prompt_template.go)  
> Each shows a `PROMPT_TEMPLATE` constant with placeholder substitution, token counting before sending, and the full API call. Tests are in `test_prompt_template.py` / `prompt_template.test.js` / `prompt_template_test.go`.

---

## Common Pitfalls

- **"The model ignores my instructions"**: Your system prompt is too long or your instructions are buried. Critical instructions go first — models pay most attention to the beginning and end of prompts.
- **"It worked yesterday, not today"**: You're relying on prompt engineering for something that should be structured output. Prompts are probabilistic; schemas are deterministic.
- **"My few-shot examples made it worse"**: You showed the model what *not* to do. Few-shot examples should only show correct behavior. Never include examples of failures.
- **"The system prompt is 5,000 tokens"**: You've written documentation, not a prompt. Every token in the system prompt is a tax on every single request. Be ruthless.

## What's Next

Prompt engineering is powerful but fragile. The solution is structured output — guaranteed parseable responses that the harness can validate.
→ [Structured Output](03-structured-output.md)
→ [Structured Output](03-structured-output.md)
