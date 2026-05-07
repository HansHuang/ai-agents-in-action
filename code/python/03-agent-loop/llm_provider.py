"""Provider-agnostic LLM abstraction layer.

Implements a uniform interface over multiple LLM providers so that agent
code never depends on a specific SDK or API format.

Supported providers:
  - OpenAI        (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
  - Anthropic     (claude-3-5-sonnet, claude-3-haiku)
  - Google        (gemini-1.5-pro, gemini-1.5-flash)
  - Ollama        (any locally-pulled model)
  - Together AI   (llama-3.1-70b, mixtral-8x7b, …)
  - FallbackProvider (wraps any two providers)

Usage:
    provider = LLMFactory.create("openai",
                                  api_key=os.environ["OPENAI_API_KEY"],
                                  model="gpt-4o")
    response = provider.chat([{"role": "user", "content": "Hello"}])
    print(response.content)

See: docs/05-the-tool-ecosystem/01-model-providers.md
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Normalised response returned by every provider."""
    content: str | None
    tool_calls: list[dict] | None
    token_usage: dict          # {prompt_tokens, completion_tokens, total_tokens}
    model: str
    finish_reason: str
    latency_ms: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate when a proper tokenizer isn't available.

    Approximation: ~4 characters per token (matches GPT-family ballpark).
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    # ------------------------------------------------------------------
    # Must-implement
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages:        OpenAI-format message list.
            tools:           OpenAI-format tool definitions (optional).
            temperature:     Sampling temperature (0–2).
            max_tokens:      Maximum tokens in the completion.
            response_format: e.g. ``{"type": "json_object"}`` for JSON mode.

        Returns:
            Normalised :class:`LLMResponse`.
        """
        ...

    @abstractmethod
    def supports_function_calling(self) -> bool:
        """True when the underlying model supports native tool/function calling."""
        ...

    @abstractmethod
    def supports_structured_output(self) -> bool:
        """True when the provider supports guaranteed JSON/schema output."""
        ...

    @abstractmethod
    def get_context_window(self) -> int:
        """Maximum context window in tokens."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        ...

    # ------------------------------------------------------------------
    # Optional override
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens for *text*.

        Override in subclasses that have access to a proper tokenizer.
        Default: character-based approximation.
        """
        return estimate_tokens(text)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI chat completions (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)."""

    _CONTEXT_WINDOWS = {
        "gpt-4o":          128_000,
        "gpt-4o-mini":     128_000,
        "gpt-3.5-turbo":   16_385,
    }

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None):
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        self._client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
        self._model = model

    # -- interface --

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.monotonic()
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        msg = resp.choices[0].message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            token_usage={
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens":      resp.usage.total_tokens,
            },
            model=self._model,
            finish_reason=resp.choices[0].finish_reason,
            latency_ms=latency_ms,
        )

    def supports_function_calling(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return True

    def get_context_window(self) -> int:
        return self._CONTEXT_WINDOWS.get(self._model, 128_000)

    def get_model_name(self) -> str:
        return self._model

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken  # type: ignore[import-untyped]
            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except Exception:
            return estimate_tokens(text)


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic Messages API (Claude 3.5 Sonnet, Claude 3 Haiku)."""

    _CONTEXT_WINDOWS = {
        "claude-3-5-sonnet-20241022": 200_000,
        "claude-3-5-sonnet-20240620": 200_000,
        "claude-3-haiku-20240307":    200_000,
        "claude-3-opus-20240229":     200_000,
    }

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    # -- format conversion --

    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Split out the system prompt; convert the rest to Anthropic format."""
        system: str | None = None
        rest: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "tool":
                # OpenAI tool result → Anthropic tool_result block
                rest.append({
                    "role": "user",
                    "content": [{"type": "tool_result",
                                 "tool_use_id": msg.get("tool_call_id", ""),
                                 "content": msg["content"]}],
                })
            else:
                rest.append({"role": msg["role"], "content": msg["content"]})
        return system, rest

    @staticmethod
    def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool definitions to Anthropic input_schema format."""
        result = []
        for tool in tools:
            fn = tool.get("function", {})
            result.append({
                "name":         fn.get("name", ""),
                "description":  fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _normalize_response(self, resp: Any, latency_ms: int) -> LLMResponse:
        """Convert Anthropic response to LLMResponse."""
        content: str | None = None
        tool_calls: list[dict] | None = None

        text_parts = []
        tc_parts: list[dict] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                import json
                tc_parts.append({
                    "id":   block.id,
                    "type": "function",
                    "function": {
                        "name":      block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
        if text_parts:
            content = "\n".join(text_parts)
        if tc_parts:
            tool_calls = tc_parts

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage={
                "prompt_tokens":     resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens":      resp.usage.input_tokens + resp.usage.output_tokens,
            },
            model=self._model,
            finish_reason=resp.stop_reason or "stop",
            latency_ms=latency_ms,
        )

    # -- interface --

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        system, converted = self._to_anthropic_messages(messages)

        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=converted,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        t0 = time.monotonic()
        resp = self._client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        return self._normalize_response(resp, latency_ms)

    def supports_function_calling(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        # Anthropic supports JSON output via prompting but not schema enforcement
        return False

    def get_context_window(self) -> int:
        return self._CONTEXT_WINDOWS.get(self._model, 200_000)

    def get_model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Google provider
# ---------------------------------------------------------------------------

class GoogleProvider(LLMProvider):
    """Google Generative AI (Gemini 1.5 Pro, Gemini 1.5 Flash)."""

    _CONTEXT_WINDOWS = {
        "gemini-1.5-pro":   1_048_576,
        "gemini-1.5-flash":   1_048_576,
        "gemini-2.0-flash": 1_048_576,
    }

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Install google-generativeai: pip install google-generativeai"
            ) from exc
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model

    @staticmethod
    def _to_gemini_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Split system prompt; convert to Gemini 'parts' format."""
        system: str | None = None
        history = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [msg["content"]]})
        return system, history

    @staticmethod
    def _to_gemini_tools(tools: list[dict]) -> list[Any]:
        """Convert OpenAI tool definitions to Gemini FunctionDeclaration objects."""
        try:
            from google.generativeai.types import FunctionDeclaration, Tool  # type: ignore
        except ImportError:
            return []
        decls = []
        for t in tools:
            fn = t.get("function", {})
            decls.append(FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            ))
        return [Tool(function_declarations=decls)]

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        import json

        system, history = self._to_gemini_messages(messages)
        model = self._genai.GenerativeModel(
            self._model_name,
            system_instruction=system,
            tools=self._to_gemini_tools(tools) if tools else None,
        )
        chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
        last = history[-1]["parts"][0] if history else ""

        t0 = time.monotonic()
        resp = chat.send_message(
            last,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        content: str | None = None
        tool_calls: list[dict] | None = None

        try:
            content = resp.text
        except Exception:
            pass

        fc_calls = []
        for part in resp.parts:
            if part.function_call.name:
                fc_calls.append({
                    "id":   f"call_{part.function_call.name}",
                    "type": "function",
                    "function": {
                        "name":      part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args)),
                    },
                })
        if fc_calls:
            tool_calls = fc_calls

        usage = resp.usage_metadata
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage={
                "prompt_tokens":     getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens":      getattr(usage, "total_token_count", 0),
            },
            model=self._model_name,
            finish_reason="stop",
            latency_ms=latency_ms,
        )

    def supports_function_calling(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return False

    def get_context_window(self) -> int:
        return self._CONTEXT_WINDOWS.get(self._model_name, 1_048_576)

    def get_model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# Ollama provider  (uses OpenAI-compatible local endpoint)
# ---------------------------------------------------------------------------

# Models known to support tool calling in Ollama
_OLLAMA_TOOL_MODELS = {
    "llama3.1", "llama3.1:8b", "llama3.1:70b",
    "llama3.2", "llama3.2:3b",
    "mistral", "mistral-nemo",
    "qwen2.5", "qwen2.5:7b",
    "command-r",
}


class OllamaProvider(LLMProvider):
    """Ollama local inference via its OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
    ):
        self._inner = OpenAIProvider(api_key="ollama", model=model, base_url=base_url)
        self._model = model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        # Only pass tools to models that support them
        effective_tools = tools if self.supports_function_calling() else None
        return self._inner.chat(messages, effective_tools, temperature, max_tokens, response_format)

    def supports_function_calling(self) -> bool:
        base = self._model.split(":")[0]
        return base in _OLLAMA_TOOL_MODELS or self._model in _OLLAMA_TOOL_MODELS

    def supports_structured_output(self) -> bool:
        return False

    def get_context_window(self) -> int:
        return 128_000

    def get_model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Together AI provider  (OpenAI-compatible cloud endpoint)
# ---------------------------------------------------------------------------

class TogetherProvider(LLMProvider):
    """Together AI hosted open-source models via OpenAI-compatible API."""

    _TOOL_MODELS = {
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    _CONTEXT_WINDOWS = {
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 128_000,
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":  128_000,
        "mistralai/Mixtral-8x7B-Instruct-v0.1":          32_768,
        "deepseek-ai/DeepSeek-V3":                       131_072,
    }

    def __init__(self, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        self._inner = OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url="https://api.together.xyz/v1",
        )
        self._model = model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        return self._inner.chat(messages, tools, temperature, max_tokens, response_format)

    def supports_function_calling(self) -> bool:
        return self._model in self._TOOL_MODELS

    def supports_structured_output(self) -> bool:
        return False

    def get_context_window(self) -> int:
        return self._CONTEXT_WINDOWS.get(self._model, 32_768)

    def get_model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# FallbackProvider  (decorator / wrapper)
# ---------------------------------------------------------------------------

class FallbackProvider(LLMProvider):
    """Try providers in order; fall back to the next on any exception.

    Construct a chain::

        provider = FallbackProvider(
            primary=gpt4o,
            fallbacks=[claude_haiku, ollama],
        )
    """

    def __init__(self, primary: LLMProvider, fallbacks: list[LLMProvider]):
        self._primary = primary
        self._fallbacks = fallbacks

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> LLMResponse:
        candidates = [self._primary, *self._fallbacks]
        last_exc: Exception | None = None
        for provider in candidates:
            try:
                return provider.chat(messages, tools, temperature, max_tokens, response_format)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Provider %s failed (%s: %s) — trying next fallback",
                    provider.get_model_name(),
                    type(exc).__name__,
                    exc,
                )
        raise RuntimeError(
            f"All providers failed. Last error: {last_exc}"
        ) from last_exc

    def supports_function_calling(self) -> bool:
        return self._primary.supports_function_calling()

    def supports_structured_output(self) -> bool:
        return self._primary.supports_structured_output()

    def get_context_window(self) -> int:
        return self._primary.get_context_window()

    def get_model_name(self) -> str:
        primary_name = self._primary.get_model_name()
        fallback_names = [f.get_model_name() for f in self._fallbacks]
        return f"{primary_name} → {' → '.join(fallback_names)}"


# ---------------------------------------------------------------------------
# LLMFactory
# ---------------------------------------------------------------------------

class LLMFactory:
    """Create :class:`LLMProvider` instances from simple configuration."""

    PROVIDER_MAP: dict[str, type[LLMProvider]] = {
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google":    GoogleProvider,
        "ollama":    OllamaProvider,
        "together":  TogetherProvider,
    }

    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> LLMProvider:
        """Create a provider by name.

        Args:
            provider: One of ``"openai"``, ``"anthropic"``, ``"google"``,
                      ``"ollama"``, ``"together"``.
            **kwargs: Forwarded to the provider's constructor.

        Raises:
            ValueError: If *provider* is unknown.
        """
        provider_cls = cls.PROVIDER_MAP.get(provider)
        if provider_cls is None:
            available = ", ".join(sorted(cls.PROVIDER_MAP))
            raise ValueError(
                f"Unknown provider: {provider!r}. Available: {available}"
            )
        return provider_cls(**kwargs)

    @classmethod
    def create_from_config(cls, config: dict) -> LLMProvider:
        """Create a provider from a configuration dict.

        Config format::

            {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",   # env var name
                "fallback": {                       # optional
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "api_key_env": "ANTHROPIC_API_KEY"
                }
            }

        The ``api_key_env`` field is looked up in the environment; the resolved
        value is passed as ``api_key`` to the constructor.
        """
        def _make_one(cfg: dict) -> LLMProvider:
            kwargs: dict[str, Any] = {}
            if "model" in cfg:
                kwargs["model"] = cfg["model"]
            if "api_key_env" in cfg:
                key = os.environ.get(cfg["api_key_env"], "")
                if cfg["provider"] not in ("ollama",) and not key:
                    logger.warning(
                        "Environment variable %s is not set.", cfg["api_key_env"]
                    )
                kwargs["api_key"] = key
            # Forward any other keys directly (e.g. base_url)
            for k, v in cfg.items():
                if k not in ("provider", "model", "api_key_env", "fallback"):
                    kwargs[k] = v
            return cls.create(cfg["provider"], **kwargs)

        primary = _make_one(config)

        fallback_cfg = config.get("fallback")
        if fallback_cfg:
            # Support nested chaining
            fallback_provider = cls.create_from_config(fallback_cfg)
            return FallbackProvider(primary=primary, fallbacks=[fallback_provider])

        return primary


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Compare providers and demonstrate fallback behaviour."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    question = "What is 17 × 23? Answer with just the number."

    providers: list[tuple[str, LLMProvider | None]] = []

    # OpenAI
    oai_key = os.environ.get("OPENAI_API_KEY")
    if oai_key:
        providers.append(("OpenAI gpt-4o-mini",
                          LLMFactory.create("openai", api_key=oai_key, model="gpt-4o-mini")))

    # Anthropic
    ant_key = os.environ.get("ANTHROPIC_API_KEY")
    if ant_key:
        providers.append(("Anthropic claude-3-haiku",
                          LLMFactory.create("anthropic", api_key=ant_key,
                                            model="claude-3-haiku-20240307")))

    # Ollama (always try; fails gracefully if server isn't running)
    providers.append(("Ollama llama3.1:8b",
                      LLMFactory.create("ollama", model="llama3.1:8b")))

    if not providers:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run the demo.")
        return

    # ------------------------------------------------------------------
    # 1. Query all providers
    # ------------------------------------------------------------------
    print("\n=== Provider Comparison ===")
    print(f"Question: {question}\n")
    print(f"{'Provider':<30}  {'Answer':<20}  {'Tokens':<10}  {'Latency (ms)'}")
    print("-" * 80)

    messages = [{"role": "user", "content": question}]
    for name, provider in providers:
        try:
            resp = provider.chat(messages, temperature=0.0, max_tokens=20)
            answer = (resp.content or "").strip()
            print(f"{name:<30}  {answer:<20}  "
                  f"{resp.token_usage.get('total_tokens', 0):<10}  {resp.latency_ms}")
        except Exception as exc:
            print(f"{name:<30}  ERROR: {exc}")

    # ------------------------------------------------------------------
    # 2. Fallback demonstration
    # ------------------------------------------------------------------
    print("\n=== Fallback Demonstration ===")

    class _AlwaysFailProvider(LLMProvider):
        """Stub that always raises to simulate an outage."""
        def chat(self, *_a, **_kw):
            raise RuntimeError("Simulated API outage")
        def supports_function_calling(self): return False
        def supports_structured_output(self): return False
        def get_context_window(self): return 0
        def get_model_name(self): return "always-fail"

    real = providers[0][1]  # use the first real provider as fallback
    fallback_provider = FallbackProvider(
        primary=_AlwaysFailProvider(),
        fallbacks=[real],
    )
    resp = fallback_provider.chat(messages, temperature=0.0, max_tokens=20)
    print(f"Fallback result: {resp.content!r} (from {real.get_model_name()})")

    # ------------------------------------------------------------------
    # 3. Provider switching via config
    # ------------------------------------------------------------------
    print("\n=== Config-Based Switching ===")
    if oai_key:
        config: dict = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        }
        p = LLMFactory.create_from_config(config)
        resp = p.chat(messages, temperature=0.0, max_tokens=20)
        print(f"Config-created provider ({p.get_model_name()}): {resp.content!r}")


if __name__ == "__main__":
    _demo()
