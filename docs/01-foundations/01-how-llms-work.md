# How LLMs Actually Work

## What You'll Learn

* What happens when you call `openai.chat.completions.create()`
* Tokens, context windows, and why they matter
* Temperature, top-p, and controlling randomness
* The difference between base models and instruction-tuned models
* Why context engineering and memory management exist

---

# The 30-Second Mental Model

An LLM fundamentally works by predicting the next token from prior tokens. Modern reasoning models add additional inference-time techniques on top of this, but next-token prediction remains the core mechanism.

You give the model a sequence of tokens (words or word fragments), and it predicts the most likely next token.

```text
Input: "The capital of France is"
Output: " Paris"

Input: "The capital of France is Paris"
Output: "."
```

ChatGPT's eloquence, Claude's reasoning, Gemini's creativity, and AI agents themselves all emerge from this iterative prediction loop.

Every other capability — agents, tools, RAG, memory, workflows — is engineering layered on top of this fundamental process.



---

# What Happens When You Call the API

Let's trace exactly what happens when your code executes this line:

```python
response = openai.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100
)
```

## Step by Step

### 1. Client-Side Preparation

Your SDK serializes the request into JSON:

```json
{
  "model": "gpt-5.5",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

This JSON is sent as an HTTPS POST request to OpenAI's API endpoint.

### 2. Server-Side Tokenization

The server receives your request and tokenizes all the text. The system prompt and user message are converted from human-readable text into sequences of integers (token IDs):

```
System prompt: "You are a helpful assistant."
→ [2675, 527, 264, 12520, 11190, 13]

User message: "What is the capital of France?"
→ [3634, 374, 279, 5618, 315, 6051, 30]
```

These token IDs are what the model actually processes — not text.

### 3. Context Assembly

The tokenized messages are assembled into a single sequence. The model sees:

```
[system_prompt_tokens] [user_message_tokens]
```

At this point, the model has no memory of any previous conversation. It only sees what's in this sequence.

### 4. Forward Pass Through the Neural Network

The token sequence flows through the model's transformer architecture. Each token attends to every other token in the sequence through the attention mechanism. The model computes a probability distribution over its entire vocabulary for what the next token should be.

For our example, after processing "What is the capital of France?", the model might assign:

```
" Paris"    → 94.2% probability
" The"      →  2.1% probability
" France"   →  1.3% probability
" It"       →  0.8% probability
... (over 100,000 possible tokens)
```

### 5. Sampling

The `temperature` parameter (0.7 in our example) influences which token is actually selected. With temperature 0.7, the distribution is moderately sharpened — high-probability tokens become slightly more likely, low-probability tokens become slightly less likely, but there's still room for variation.

The model samples from this distribution and selects `" Paris"` as the next token.

### 6. Autoregressive Generation

The selected token is appended to the sequence:

```
[...previous tokens..., " Paris"]
```

Now the model runs again, predicting the next token after "Paris":

```
"...Paris" → "." (89.3% probability)
```

This loop continues — predict, sample, append — until one of these happens:
- The model generates a special end-of-sequence token
- The `max_tokens` limit (100 in our example) is reached
- A stop sequence is triggered

For our query, the full generation might be:

```
" Paris" → "." → " It" → " is" → " also" → " known" → " as" → " the" → " City" → " of" → " Light" → "."
```

### 7. Detokenization

The generated token IDs are converted back to text:

```
[7707, 13, 1107, 374, 1552, 1901, 374, 279, 3204, 315, 3077, 13]
→ "Paris. It is also known as the City of Light."
```

### 8. Response Assembly

The server packages everything into a response:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1715472000,
  "model": "gpt-5.5",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Paris. It is also known as the City of Light."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 12,
    "total_tokens": 69
  }
}
```

### 9. Your Code Receives the Response

Your SDK deserializes the JSON into a response object:

```python
print(response.choices[0].message.content)
# Output: Paris. It is also known as the City of Light.

print(response.usage.total_tokens)
# Output: 69
```

## Why Understanding This Matters will deal


Every concept in this repository builds on this flow:

| Concept | Where It Fits in the Flow |
|:---|:---|
| **Prompt Engineering** | Steps 1 & 3 — what you put in `messages` and how it's assembled |
| **Token Budgeting** | Steps 2 & 6 — how many tokens you send and generate |
| **Context Engineering** | Step 3 — what's in the assembled sequence and how it got there |
| **Structured Output** | Steps 5-6 — constraining what tokens can be generated |
| **Streaming** | Steps 6-7 — receiving tokens incrementally as they're generated |
| **Temperature Tuning** | Step 5 — controlling the probability distribution during sampling |
| **Agent Loop** | Steps 1-9 — the entire flow repeats for each turn of the agent's orchestration loop |

Later chapters add more to this flow:
- **Tool Calling** — adding `tools` to Step 1 so the model can request function execution
- **Retrieval (RAG)** — injecting external documents into Step 3
- **Memory Management** — controlling what conversation history appears in Step 1

This entire flow happens in milliseconds to seconds, and your agent well trigger it dozens or hundreds of times per conversation. Understanding each step is what separates someone who "uses AI" from someone who engineers with it.


---

# Tokens: The Atoms of Language Models

## What Is a Token?

A token is a chunk of text used internally by the model.

In English:

* `"Hello world"` → ~2 tokens
* `"inexplicable"` → may be 4 token
* punctuation and spaces can also be tokens

Rules vary by tokenizer and language:

* Common Chinese characters are often 1 token each
* Rare Chinese characters may use multiple tokens
* Long German compound words can become many tokens
* Code tends to tokenize densely

A useful mental estimate:

* 1 token ≈ 0.75 English words
* 1 token ≈ 4 characters

> Different models use different tokenizers. The same text may be 100 tokens with GPT but 130 tokens with Claude. Always count tokens with the specific tokenizer for your chosen model.

## Why Tokens Matter

Tokens directly affect nearly every production concern.

### Cost

Most providers charge per token:

* Input tokens
* Output tokens
* Sometimes cached/reasoning tokens separately

More tokens = higher cost.

### Latency

More tokens generally mean:

* longer upload time
* more model computation
* slower generation

### Context Window Limits

Models have a maximum context window. If your conversation exceeds it:

* older messages may be truncated
* requests may fail
* important information may disappear

### Hidden Tokens

Many developers underestimate hidden token usage.

These all consume tokens on every request:

* system prompts
* tool definitions
* schemas
* retrieved documents
* conversation history

A 2,000-token system prompt plus a 500-token tool definition means every user request starts 2,500 tokens in the hole — even if the user only says:

```text
Hello
```

---

# Token Counting

In production systems, estimate or count tokens before requests whenever cost, latency, or truncation matters.

Most providers expose tokenizer libraries compatible with their models.

## Python — `tiktoken`

[tiktoken](https://github.com/openai/tiktoken)

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-5.5")

text = "Your text here"

tokens = enc.encode(text)

print(len(tokens))
```

> **Code Examples**: [Python](../../code/python/01-basic-llm-call/) | [Node.js](../../code/nodejs/01-basic-llm-call/) | [Go](../../code/go/01-basic-llm-call/), each example counts tokens for
>
> * plain strings
> * full `messages` arrays
>
> This mirrors what APIs actually bill you for.

---

# The Context Window: The Model's Working Memory

The context window is the maximum number of tokens the model can see at once.

Modern frontier models range from roughly ~100K tokens to 1M+ tokens depending on provider and model family.

## What Counts Against the Window

Everything visible to the model consumes context space:

* System prompt
* Conversation history
* User message
* Tool definitions
* Retrieved documents (RAG)
* Structured output schemas
* The model's own generated output

A critical detail many beginners miss:

> Output tokens consume the same context window.

If a model has a 128K context window and your input already uses 127K tokens, the model only has room to generate roughly 1K output tokens before hitting the limit.

## Why Long Contexts Are Expensive

Long contexts do not just increase bandwidth, they increase computation.

Transformer models perform attention across the visible token sequence during generation. As context grows:

* latency increases
* compute increases
* inference becomes more expensive

This is why context engineering becomes a core discipline for AI agents.

## System Prompts Are Still Just Tokens

A common misconception is that system prompts are somehow "outside" the model. They are not.

Internally, system prompts are still tokens inside the context window. Providers may structure or prioritize them differently, but the model ultimately processes them as part of the same sequence.

This becomes important later when discussing:

* prompt injection
* instruction hierarchy
* context poisoning
* agent security

---

# Why Context Engineering Exists

You are always managing a scarce resource:

* context space
* latency budget
* token cost
* model attention

This is why disciplines like:

* context engineering
* memory management
* retrieval design

exist in modern AI systems.

Briefly:

* **Context engineering** = deciding what enters the context window
* **Memory management** = deciding what persists over time
* **Retrieval (RAG)** = fetching external information dynamically

---

# Temperature: Controlling Randomness

Temperature controls sampling randomness during token generation.

Lower temperatures make outputs more predictable. Higher temperatures increase variation and exploration.

| Temperature | Behavior                              | Common Use Cases                     |
| ----------- | ------------------------------------- | ------------------------------------ |
| 0           | Lowest randomness, usually repeatable | Structured output, extraction, tools |
| 0.3–0.7     | Balanced                              | General assistants                   |
| 0.8–1.2     | More varied and creative              | Brainstorming, writing               |
| 1.5+        | Chaotic, often incoherent             | Rarely useful                        |

## Important Nuance About Temperature 0

Temperature 0 reduces randomness but does not guarantee identical outputs.

You may still observe variation due to:

* provider infrastructure
* backend routing
* batching
* floating-point nondeterminism
* model updates

For production agents:

* use low temperature for reliability
* never assume perfect determinism

## Rule of Thumb for Agents

For most agent systems:

* `temperature = 0`

  * function calling
  * JSON generation
  * structured output
  * deterministic workflows

* `temperature = 0.3–0.7`

  * conversational responses
  * summaries
  * explanations

Higher temperatures are usually counterproductive for autonomous systems.

---

# Top-p: Alternative Sampling Control

Temperature is not the only sampling parameter.

`top_p` (nucleus sampling) limits token selection to the smallest probability set whose cumulative probability exceeds `p`.

Example:

* `top_p = 0.9`
* Model only samples from tokens covering the top 90% probability mass

In practice:

* Use temperature for general control over randomness (simpler mental model)
* Use top-p when you want to dynamically adapt to the probability distribution (e.g., `top_p=0.9` means "consider only the most likely tokens that together represent 90% of the probability mass")
* Many providers recommend adjusting either `temperature` or `top_p`, not both simultaneously

---

# Base Models vs. Instruction-Tuned Models

| Base Model                        | Instruction-Tuned Model           |
| --------------------------------- | --------------------------------- |
| Trained to continue internet text | Fine-tuned to follow instructions |
| Raw next-token continuation       | Optimized for helpful interaction |
| Unpredictable                     | Conversational                    |
| Often unsafe/unfiltered           | Aligned and constrained           |
| Used for research/fine-tuning     | Used for real applications        |

Example:

| Prompt                    | Base Model                         | Instruction Model                   |
| ------------------------- | ---------------------------------- | ----------------------------------- |
| `"The capital of France"` | `" is Paris, a city known for..."` | `"The capital of France is Paris."` |

Today, virtually every major production model: GPT, Claude, Gemini, Kimi, DeepSeek is instruction-tuned.

The raw base model is usually an internal artifact rather than the product developers interact with directly.

---

# Streaming Responses

Most modern APIs support token streaming.

Instead of waiting for the full response:

* the server streams tokens incrementally as they're generated
* the UI renders text progressively
* tool calls can be streamed as they're decided

Streaming improves:

* perceived responsiveness (users see activity immediately)
* conversational UX (feels like real-time dialogue)
* agent interactivity (tool calls and results appear as they happen)

Even though the model generates tokens sequentially internally, streaming transforms the user experience from "wait and see" to "watch it think."

Streaming is foundational to the Vercel AI SDK (Chapter 06) and is a key deployment consideration (Chapter 09).

---

# The Five Parameters Every Developer Should Know

Before optimizing exotic settings, master these five:

### 1. `model`

Determines: capability, speed, context size, cost
  
### 2. `messages`

The conversation itself:

```python
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."}
]
```

This is the primary interface to the model.

### 3. `temperature`

Controls output randomness and variation.

* Lower: more predictable
* Higher: more diverse

### 4. `max_tokens`

Controls maximum output length. This is your primary defense against:

* runaway agent loops
* unexpected cost spikes
* excessive response latency
* context window overflow from overly long generations

Always set a `max_tokens` limit appropriate for your use case. Never rely on the model to "know when to stop."

### 5. `tools` (optional but essential for agents)

Defines functions the model can request execution of. Covered in depth in [Tool Design Patterns](../02-the-agent-loop/02-tool-design-patterns.md).

---

# Common Pitfalls

## "The model is hallucinating"

Hallucination emerges naturally from probabilistic token prediction under uncertainty.

The model is optimized for:

* plausibility
* coherence
* continuation quality

—not truth.

This is why:

* retrieval systems
* validation layers
* structured outputs
* harness engineering

exist in production AI systems.

## "My prompt doesn't work"

Usually this means:

* insufficient constraints
* ambiguous instructions
* unclear output format
* missing examples

LLMs are highly sensitive to context structure.

Structured output and prompt engineering dramatically improve reliability.

## "It's too slow"

Check:

* token count
* context size
* retrieved documents
* conversation history
* unnecessary tools

Many slow agents are simply sending too much context.

## "The model forgot earlier information"

The model has no persistent memory by default.

Every request only sees:

* what fits in the current context window
* what you explicitly resend

Memory systems are external infrastructure layered around the model.

---

# The Core Insight

LLMs are not databases.
They are not reasoning engines in the classical sense.
They are not truth machines.

They are probabilistic token predictors operating over context.

Once you deeply understand that, modern AI systems become much easier to reason about:

* prompts
* agents
* tools
* memory
* RAG
* structured output
* evaluation
* guardrails

All of them are mechanisms for shaping the token prediction process.

**This entire repository is about building reliable engineering around that fundamental probabilistic core.**

---

# What's Next

Now that you understand the engine, the next step is learning how to control it effectively.

→ [Prompt Engineering](02-prompt-engineering.md)
