# How LLMs Actually Work

## What You'll Learn
- What happens when you call `openai.chat.completions.create()`
- Tokens, context windows, and why they matter
- Temperature, top-p, and controlling randomness
- The difference between base models and instruction-tuned models

---

## The 30-Second Mental Model

An LLM is a **token prediction engine**. You give it a sequence of tokens (words or word fragments), and it predicts the next most likely token. That's it. ChatGPT's eloquence, Claude's reasoning, Gemini's creativity — all emerge from this one mechanism.

```
Input: "The capital of France is"
Output: " Paris"
Input: "The capital of France is Paris"
Output: "."
```

Every other capability — agents, tools, RAG — is just engineering layered on top of this fundamental loop.

## Tokens: The Atoms of Language Models

### What Is a Token?
A token is roughly 0.75 words or 4 characters in English. "Hello world" = 2 tokens. "Inexplicable" = 1 token. Languages like Chinese use more tokens per concept.

### Why Tokens Matter
- **Cost**: You pay per token (input + output)
- **Latency**: More tokens = slower response
- **Context Window**: There's a hard limit on input tokens
- **Hidden tokens**: Your system prompt and tool definitions consume tokens on every single request. A 2,000-token system prompt with a 500-token tool definition means every user message starts 2,500 tokens in the hole — even for a one-word query like "Hello."

### Token Counting

Always count tokens *before* sending a request, not after. Libraries wrap the same BPE tokenizer used by the models.

**Python** — [`tiktoken`](https://github.com/openai/tiktoken)
```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
print(len(enc.encode("Your text here")))  # → token count
```

**Node.js** — [`js-tiktoken`](https://github.com/dqbd/tiktoken)
```js
import { encodingForModel } from "js-tiktoken";

const enc = encodingForModel("gpt-4o");
console.log(enc.encode("Your text here").length);
```

**Go** — [`tiktoken-go`](https://github.com/pkoukk/tiktoken-go)
```go
import tiktoken "github.com/pkoukk/tiktoken-go"

enc, _ := tiktoken.EncodingForModel("gpt-4o")
fmt.Println(len(enc.Encode("Your text here", nil, nil)))
```

> **Code** → [Python](../../code/python/01-basic-llm-call/) · [Node.js](../../code/nodejs/01-basic-llm-call/) · [Go](../../code/go/01-basic-llm-call/)  
> Each folder contains a full example that counts tokens for both a plain string and a `messages` array, mirroring what the API actually charges you for.

## The Context Window: The Model's Working Memory

The context window is the maximum number of tokens the model can "see" at once. For GPT-5.4 it's 272K tokens, for Claude 4.6 or DeepSeek v4 it's 1M tokens.

### What Counts Against the Window
- System prompt
- Conversation history
- User's current message
- Tool definitions
- Retrieved documents (RAG)
- The model's own output (as it generates)

### The Engineering Implication
You are always budgeting a scarce resource. This is why context engineering (Chapter 04) and memory management (Chapter 03) exist.

## Temperature: Controlling Creativity

Temperature controls how "random" the model's outputs are. Most providers use a scale from 0 to 2 (OpenAI) or 0 to 1 (Anthropic), but the behavior is consistent across ranges.

| Temp | Behavior | Use Case |
|:---|:---|:---|
| 0 | Deterministic, always picks highest-probability token | Structured data extraction, math |
| 0.3-0.7 | Balanced | General assistant tasks |
| 0.8-1.2 | Creative, varied | Brainstorming, creative writing |
| 1.5+ | Chaotic, often incoherent | Rarely useful |

> **Rule of thumb**: For agents, use temperature 0 for function calling and structured output. Use 0.3-0.7 for natural language responses.

## Base Models vs. Instruction-Tuned Models

| Base Model | Instruction-Tuned Model |
|:---|:---|
| Trained to predict next token on internet text | Fine-tuned to follow instructions |
| You prompt: "The capital of France" | You prompt: "What is the capital of France?" |
| It continues: "is Paris, a city known for..." | It answers: "The capital of France is Paris." |
| Raw, unpredictable | Helpful, conversational |
| Use for: custom fine-tuning | Use for: 99% of applications |

Today, every major model you interact with (Claude, GPT-4o, Gemini, Kimi) is instruction-tuned. The base model is an internal artifact; the instruction-tuned version is the product you use.

## The API Call Deconstructed

Here's what actually happens when your code calls the API:

```
1. You send: model, messages, temperature, tools (optional)
2. Server tokenizes your input
3. Model processes the full context through its neural network
4. Model predicts the next token
5. That token is added to the context
6. Repeat steps 4-5 until model predicts an "end" token or hits max_tokens
7. Server detokenizes the output tokens back to text
8. You receive: response with content, token usage, finish reason
```

## The Three Parameters Every Developer Should Know

1. **`model`**: Which model to use. Determines capability, cost, speed.
2. **`messages`**: The conversation array. `[{role, content}, ...]`
3. **`temperature`**: Control knob for randomness.

Everything else (top_p, frequency_penalty, presence_penalty) is optional fine-tuning. Master these three first.

## Common Pitfalls

- **"The model is hallucinating"**: It's not hallucinating — it's doing exactly what it was built to do: predict plausible tokens. Hallucination is a feature of the architecture, not a bug. Harness engineering (Chapter 07) exists to catch it.
- **"My prompt doesn't work"**: You're not giving the model enough constraints. Structured output (next chapter) solves this.
- **"It's too slow"**: Check your token count. Are you sending the entire conversation history every time?

## What's Next

Now that you understand the engine, learn how to control it:
→ [Prompt Engineering](02-prompt-engineering.md)
