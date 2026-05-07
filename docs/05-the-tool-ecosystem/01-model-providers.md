# Model Providers

## What You'll Learn
- The LLM provider landscape: OpenAI, Anthropic, Google, and open-source
- How to choose a model: capability, cost, latency, and context window tradeoffs
- Multi-provider architecture: writing provider-agnostic agent code
- The OpenAI-compatible API standard and why it matters
- When to use cloud APIs vs. self-hosted models
- Model routing: using different models for different tasks

## Prerequisites
- [How LLMs Actually Work](../01-foundations/01-how-llms-work.md) — tokens, context windows, temperature
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — the agent loop that calls models
- [Structured Output](../01-foundations/03-structured-output.md) — function calling across providers

---

## The Provider Landscape

Every LLM API has the same fundamental interface: send messages, get a response. But providers differ in capability, cost, speed, and reliability.

### The Major Providers

| Provider | Flagship Model | Context Window | Strengths | Weaknesses |
|:---|:---|:---|:---|:---|
| **OpenAI** | GPT-4o | 128K | Function calling, structured output, ecosystem | Cost at scale, rate limits |
| **Anthropic** | Claude 3.5 Sonnet | 200K | Long context reasoning, safety, nuanced instruction following | Fewer embedding options |
| **Google** | Gemini 1.5 Pro | 1M+ | Massive context, multimodal, competitive pricing | API stability, documentation |
| **Meta (Open-Source)** | Llama 3.1 405B | 128K | Self-hosted, privacy, no rate limits | Requires GPU infrastructure |
| **Mistral (Open-Source)** | Mistral Large | 128K | Efficient, multilingual, strong at code | Smaller ecosystem |

### The Long Tail

Beyond the majors, specialized providers fill specific niches:

| Provider | Specialty | Best For |
|:---|:---|:---|
| **Cohere** | Embeddings and RAG | Enterprise search, classification |
| **Voyage AI** | Embeddings | Retrieval-optimized embeddings |
| **Together AI** | Open-source hosting | Running Llama, Mistral, etc. without infrastructure |
| **Groq** | Speed | Ultra-low latency inference (300+ tok/s) |
| **Fireworks AI** | Fine-tuned models | Specialized, optimized open-source models |
| **DeepSeek** | Cost | Very competitive pricing, strong reasoning |

---

## The OpenAI-Compatible API Standard

OpenAI's API format has become the de facto standard. Most providers offer an OpenAI-compatible endpoint:

```python
# OpenAI native
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# Together AI (OpenAI-compatible)
client = OpenAI(
    api_key="together-...",
    base_url="https://api.together.xyz/v1"
)

# Groq (OpenAI-compatible)
client = OpenAI(
    api_key="gsk-...",
    base_url="https://api.groq.com/openai/v1"
)

# Local Ollama (OpenAI-compatible)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require a real key
)

# The SAME code works for all of them
response = client.chat.completions.create(
    model="llama-3.1-70b",  # or "mixtral-8x7b", or "gemma-2-9b"
    messages=[{"role": "user", "content": "Hello"}]
)
```

This is the single most important interoperability feature in the LLM ecosystem. It means your agent code can be provider-agnostic.

---

## Writing Provider-Agnostic Agent Code

Don't couple your agent to a specific provider. Abstract the LLM call behind an interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict] | None
    token_usage: dict
    model: str
    finish_reason: str
    latency_ms: int  # wall-clock milliseconds for the API call

class LLMProvider(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,  # e.g. {"type": "json_object"}
    ) -> LLMResponse:
        """Send a chat completion request."""
        ...
    
    @abstractmethod
    def supports_function_calling(self) -> bool:
        """Does this provider support native function calling?"""
        ...
    
    @abstractmethod
    def get_context_window(self) -> int:
        """Maximum context window size."""
        ...

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens
        )
        msg = response.choices[0].message
        return LLMResponse(
            content=msg.content,
            tool_calls=msg.tool_calls,
            token_usage=response.usage.__dict__,
            model=self.model,
            finish_reason=response.choices[0].finish_reason
        )
    
    def supports_function_calling(self):
        return True
    
    def get_context_window(self):
        return 128000

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
        # Anthropic uses a different API format — adapt internally
        system_msg = None
        if messages[0]["role"] == "system":
            system_msg = messages[0]["content"]
            messages = messages[1:]
        
        response = self.client.messages.create(
            model=self.model,
            system=system_msg,
            messages=messages,
            tools=self._convert_tools(tools),
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Normalize to our standard LLMResponse
        return self._normalize_response(response)
    
    def _convert_tools(self, tools):
        """Convert OpenAI tool format to Anthropic input_schema format."""
        if not tools:
            return None
        # Anthropic uses {name, description, input_schema} instead of
        # {type, function: {name, description, parameters}}
        return [
            {
                "name":         t["function"]["name"],
                "description":  t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]
    
    def _normalize_response(self, response):
        """Convert Anthropic response to our standard LLMResponse."""
        ...
```

### The Factory Pattern

Choose providers at runtime based on configuration:

```python
class LLMFactory:
    """Create LLM providers from configuration."""
    
    PROVIDER_MAP = {
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google":    GoogleProvider,
        "together":  TogetherProvider,
        "ollama":    OllamaProvider,
    }
    
    @classmethod
    def create(cls, provider: str, **kwargs) -> LLMProvider:
        provider_class = cls.PROVIDER_MAP.get(provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider}. "
                           f"Available: {list(cls.PROVIDER_MAP.keys())}")
        return provider_class(**kwargs)

# Usage
provider = LLMFactory.create(
    "openai",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o"
)

# Switch to Anthropic by changing one config value
provider = LLMFactory.create(
    "anthropic",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-3-5-sonnet-20241022"
)

# Your agent code doesn't change
agent = Agent(llm_provider=provider)
```

> **Code Reference:** [Python](../../code/python/03-agent-loop/) · [Node.js](../../code/nodejs/03-agent-loop/) · [Go](../../code/go/03-agent-loop/)  
> The agent-loop implementations include an `LLMProvider` abstraction with support for OpenAI, Anthropic, and Ollama.

---

## Choosing a Model: The Decision Matrix

For each agent task, evaluate models across five dimensions:

### 1. Capability

| Task | Recommended Models | Why |
|:---|:---|:---|
| Simple chat, Q&A | GPT-4o-mini, Claude 3 Haiku, Gemini 1.5 Flash | Fast, cheap, good enough |
| Complex reasoning | GPT-4o, Claude 3.5 Sonnet | Multi-step reasoning, tool use |
| Function calling / structured output | GPT-4o, Claude 3.5 Sonnet | Best tool-calling reliability |
| Long document analysis | Gemini 1.5 Pro, Claude 3.5 Sonnet | 1M+ and 200K context windows |
| Code generation | Claude 3.5 Sonnet, GPT-4o | Top coding benchmarks |
| Multilingual | GPT-4o, Gemini 1.5 Pro | Strong multilingual performance |

### 2. Cost

Per 1M tokens (approximate, check current pricing):

| Model | Input | Output | 100K input cost |
|:---|:---|:---|:---|
| GPT-4o | $2.50 | $10.00 | $0.25 |
| GPT-4o-mini | $0.15 | $0.60 | $0.015 |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $0.30 |
| Claude 3 Haiku | $0.25 | $1.25 | $0.025 |
| Gemini 1.5 Pro | $1.25 | $5.00 | $0.125 |
| Gemini 1.5 Flash | $0.075 | $0.30 | $0.0075 |
| Llama 3.1 70B (Together) | $0.90 | $0.90 | $0.09 |
| DeepSeek-V3 | $0.27 | $1.10 | $0.027 |

### 3. Latency

| Model | Typical TTFT* | Tokens/second | Best For |
|:---|:---|:---|:---|
| Groq (Llama 3.1 70B) | <200ms | 300+ | Real-time, streaming |
| GPT-4o-mini | ~500ms | ~100 | Interactive chat |
| Claude 3 Haiku | ~500ms | ~80 | Fast responses |
| GPT-4o | ~1s | ~80 | Balanced |
| Claude 3.5 Sonnet | ~1.5s | ~60 | Quality over speed |

*TTFT = Time to First Token

### 4. Reliability

| Provider | Uptime History | Rate Limits | Retry Behavior |
|:---|:---|:---|:---|
| OpenAI | Strong, occasional degradation | Tiered by spend | 429 errors, exponential backoff |
| Anthropic | Strong | Tiered by spend | 429 errors |
| Google | Improving | Generous | Quota-based |
| Self-hosted | Your responsibility | None | You control it |

### 5. Context Window

More context isn't always better. The model's ability to use context degrades with length:

| Window Size | Models | Practical Limit |
|:---|:---|:---|
| 128K | GPT-4o, Llama 3.1 | ~80K for reliable recall |
| 200K | Claude 3.5 Sonnet | ~120K for reliable recall |
| 1M+ | Gemini 1.5 Pro | ~500K for reliable recall |

---

## Model Routing: Different Models for Different Tasks

A single agent doesn't need a single model. Route tasks to the right model:

```python
class ModelRouter:
    """
    Route tasks to the most appropriate model based on requirements.
    """
    
    def __init__(self):
        self.models = {
            "fast": LLMFactory.create("groq", model="llama-3.1-70b"),
            "smart": LLMFactory.create("openai", model="gpt-4o"),
            "cheap": LLMFactory.create("openai", model="gpt-4o-mini"),
            "long_context": LLMFactory.create("anthropic", model="claude-3-5-sonnet"),
        }
    
    def route(self, task: dict) -> LLMProvider:
        """
        Route a task to the best model.
        
        task = {
            "type": "chat" | "reasoning" | "classification" | "summarization",
            "complexity": "low" | "medium" | "high",
            "context_size": int,  # tokens
            "priority": "latency" | "cost" | "quality"
        }
        """
        if task["priority"] == "latency":
            return self.models["fast"]
        
        if task["context_size"] > 100000:
            return self.models["long_context"]
        
        if task["complexity"] == "low":
            return self.models["cheap"]
        
        return self.models["smart"]
    
    def route_with_fallback(self, task: dict) -> LLMProvider:
        """
        Route with fallback: if primary model fails, try the next best.
        """
        primary = self.route(task)
        fallback = self.models["cheap"]  # Fallback for everything
        
        return FallbackProvider(primary=primary, fallbacks=[fallback])

class FallbackProvider(LLMProvider):
    """Try primary provider; on failure, iterate through fallbacks in order."""
    
    def __init__(self, primary: LLMProvider, fallbacks: list[LLMProvider]):
        self.primary = primary
        self.fallbacks = fallbacks  # list so you can chain more than one backup
    
    def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
        for provider in [self.primary, *self.fallbacks]:
            try:
                return provider.chat(messages, tools, temperature, max_tokens)
            except Exception as e:
                logging.warning(f"{provider.get_model_name()} failed: {e} — trying next")
        raise RuntimeError("All providers failed")
```

---

## Self-Hosted Models: When to Go Off-Cloud

Cloud APIs are the default. Self-hosting makes sense when:

| Reason | Example |
|:---|:---|
| **Privacy** | Medical, legal, financial data that can't leave your infrastructure |
| **Cost at scale** | 1B+ tokens/month — your GPU costs may be lower than API costs |
| **Latency** | Sub-50ms requirements that cloud APIs can't meet |
| **Control** | Need specific model versions, fine-tuning, or no rate limits |
| **Air-gapped** | Environments without internet access |

### Self-Hosting Options

| Tool | Best For | Setup Complexity |
|:---|:---|:---|
| **Ollama** | Development, small-scale | One command: `ollama run llama3.1` |
| **vLLM** | Production, high throughput | Moderate |
| **LM Studio** | Desktop, no-code | Minimal |
| **LocalAI** | OpenAI-compatible, Docker | Moderate |
| **llama.cpp** | CPU-only, edge devices | Moderate |

### The Ollama Quick Start

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b

# Run it (OpenAI-compatible endpoint on localhost:11434)
ollama serve

# Use it with the OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Common Pitfalls

- **"I use the same model for everything"**: A 2-cent classification task doesn't need GPT-4o. A complex reasoning task might. Route tasks to appropriate models.
- **"I hardcode the provider in my agent"**: Your agent shouldn't know whether it's talking to OpenAI or Anthropic. Use the `LLMProvider` abstraction. Switching providers should be a one-line config change.
- **"I assume all providers support function calling the same way"**: Anthropic's tool format differs from OpenAI's. Google's function calling has quirks. Abstract these differences in your provider implementations.
- **"I don't have a fallback provider"**: APIs go down. Rate limits get hit. Always have a fallback — even if it's just a cheaper model from the same provider.
- **"I ignore rate limits until I hit them"**: Production traffic hitting 429 errors means dropped user requests. Implement exponential backoff and rate limit awareness from day one.
- **"I assume self-hosting is always cheaper"**: GPUs are expensive. Cloud APIs are cheap at low-to-medium volume. Do the math for your specific scale.

## What's Next

You can now choose and integrate any LLM provider into your agent. Next: where to store the embeddings and vectors that power RAG — the vector database landscape.
→ [Vector Databases](02-vector-databases.md)