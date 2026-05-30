# Prompt Engineering

## What You'll Learn
- The anatomy of a prompt: system, user, and assistant messages
- Why the system prompt is the most important piece of text you'll write
- Few-shot prompting: teaching by example
- Chain-of-thought: making the model show its work
- When prompt engineering ends and structured output begins

## Prerequisites
- [How LLMs Actually Work](01-how-llms-work.md) — tokens and the basic API message structure

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

The model was trained on conversations structured exactly this way. In practice, the `system` role usually carries more weight than the user prompt, so it is the best place to define behavior, format, and boundaries. That does **not** make it magic: conflicting instructions, overloaded prompts, and weak examples can still cause failures.

> **Why this matters for agents:** We'll build agent loops in the next section. For now, know that an agent keeps appending tool calls and tool results as messages. The `assistant` role carries the model's prior reasoning forward, and the `user` role can come from either a human or from tool output injected back into the conversation.

---

## The System Prompt: Your Highest-Leverage Line of Code

The system prompt is the instruction set that runs before every single message. It has outsized influence on behavior, but it also consumes tokens on every API call, so every word must earn its place.

### A Minimal System Prompt
```
You are a helpful assistant.
```
**When it works:** Quick prototyping, generic chat, or simple one-off tasks.

**Where it fails:** The model decides what "helpful" means. There are no domain boundaries, no format constraints, and no fallback behavior.

### A Good System Prompt
```
You are a customer support agent for an e-commerce platform.
Answer questions about orders, returns, and shipping.
If the user asks about a specific order, ask for the order ID.
Never make up order details. If you don't know, say so.
Respond in plain text, under 100 words.
```
**Why it works:** Role clarity. Domain boundaries. Explicit fallback behavior. Output constraint.

The point is not to make every system prompt long. The point is to make it specific enough for the failure modes you actually care about.

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

### Zero-Shot (Instruction Only)
```
System: Classify tweets as positive, negative, or neutral.
Respond with exactly one word.

User: "The new update is fine, I guess."
```
This often works, but it is still brittle. The model might output `"Neutral"`, `"neutral"`, or `"This seems neutral."` The instruction teaches the format; it does not guarantee the model will follow it every time.

### Few-Shot (With Examples)
```
System: Classify tweets as positive, negative, or neutral.
Respond with exactly one word.

User: "I love this product!"        → Positive
User: "This is the worst."          → Negative
User: "It's okay, nothing special." → Neutral
User: "The new update is fine, I guess."
```
Now the model has both the instruction **and** examples of the desired behavior. The examples reinforce format, casing, and edge-case handling. They also consume tokens, so the question is not "Is few-shot good?" but "Is the extra reliability worth the extra cost for this task?"

Example output from the code sample:

```text
Approach     Result        Tokens sent
--------------------------------------
Zero-shot    Neutral                37
Few-shot     Neutral                67

Few-shot overhead: +30 tokens per request
```

The exact numbers vary by model and prompt wording, but the pattern is stable: examples usually buy consistency by spending more prompt tokens.

### When to Use Few-Shot
| Scenario | Use Few-Shot? |
|:---|:---|
| Simple classification | 2-3 examples if format drift matters, then switch to structured output when the label must be machine-parseable |
| Format teaching | 1-2 examples showing exact output format |
| Edge case handling | Show the tricky case in your examples |
| Every request is unique | Sometimes still useful if the output shape must stay consistent |

> **Code** → [`few_shot_comparison.py`](../../code/python/02-structured-output/few_shot_comparison.py) · [`few_shot_comparison.ts`](../../code/nodejs/02-structured-output/few_shot_comparison.ts) · [`few_shot_comparison.go`](../../code/go/02-structured-output/few_shot_comparison.go)  
> `few_shot_comparison.py` runs zero-shot and few-shot on the same input and prints both labels with their token counts side by side — so you can measure the reliability/cost trade-off directly.

---

## Chain-of-Thought: Making the Model Show Its Work

For reasoning-heavy tasks, one effective prompt pattern is to ask the model to reason step by step before giving the final answer.

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

The trade-off is cost. Chain-of-thought usually produces many more output tokens than a direct answer, so use it when you need auditability or multi-step reasoning, not for simple lookups.

### When Chain-of-Thought Helps Most

- Math and logic problems
- Multi-step reasoning
- Planning (which tool to call next)
- Any decision where you need to audit the model's reasoning

### When Not to Use Chain-of-Thought

- Simple factual lookups
- Classification tasks where a short label is enough
- High-volume paths where extra output tokens would dominate cost
- Any workflow where structured output already captures the decision you need

> **Why this matters for agents:** In a ReAct agent, the model's "thinking" is the plan. Chain-of-thought makes the plan visible and debuggable. Without it, you're staring at a wrong tool call with no idea why.

> **Code** → [`chain_of_thought.py`](../../code/python/02-structured-output/chain_of_thought.py) · [`chain_of_thought.ts`](../../code/nodejs/02-structured-output/chain_of_thought.ts) · [`chain_of_thought.go`](../../code/go/02-structured-output/chain_of_thought.go)  
> Each sends the same math problem with and without CoT at `temperature=0`, printing both responses side by side.

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
| Reliability depends on prompt wording and model behavior | Reliability is much higher because the response must satisfy the schema |

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

Two practical rules matter here:

- Keep reusable instructions in the template, but keep large retrieved text and user-provided content in the user message when possible.
- Remember that template text is a recurring tax. A 400-token system prompt costs 400 tokens on every request before the model answers anything.

> **Security note — prompt injection:** Never interpolate raw user input directly into a system prompt template. A malicious user can enter `Ignore all previous instructions` or include prompt-like content inside `{article_text}` to manipulate the model's behavior. Prefer placing raw user data in the user message, and validate or sanitize inputs before templating. This is covered in depth in [Input Guardrails](../07-harness-engineering/02-input-guardrails-and-validation.md).

> **Code** → [`prompt_template.py`](../../code/python/02-structured-output/prompt_template.py) · [`prompt_template.js`](../../code/nodejs/02-structured-output/prompt_template.js) · [`prompt_template.go`](../../code/go/02-structured-output/prompt_template.go)  
> Each shows a `PROMPT_TEMPLATE` constant with placeholder substitution, token counting before sending, and the full API call. Tests are in `test_prompt_template.py` / `prompt_template.test.js` / `prompt_template_test.go`.

---

## Common Pitfalls

- **"The model ignores my instructions"**: Usually this means the instruction is weak, conflicts with another instruction, or is buried too far from the task. Diagnostic: move the most important rule into the system prompt and restate the output format directly above the task.
- **"It worked yesterday, not today"**: You're relying on prompt wording for something that should be schema-validated. Diagnostic: ask whether your harness needs a reliable structure or just decent prose. If it needs structure, use structured output.
- **"My few-shot examples made it worse"**: The examples may be inconsistent, noisy, or teaching the wrong pattern. Diagnostic: check whether every example shows the exact behavior you want, including casing and format.
- **"The system prompt is 5,000 tokens"**: You've written documentation, not a prompt. Diagnostic: count tokens before sending and cut any instruction that does not change model behavior.

Prompt iteration is an engineering workflow, not an act of faith: write a baseline prompt, test it against a small set of representative inputs, inspect failures, and only then decide whether you need better wording, better examples, or a schema.

## What's Next

Prompt engineering is powerful but fragile. The solution is structured output — guaranteed parseable responses that the harness can validate.
→ [Structured Output](03-structured-output.md)
