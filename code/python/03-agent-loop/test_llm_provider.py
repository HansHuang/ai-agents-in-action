"""Tests for llm_provider.py, model_router.py, and LLMFactory.

All tests use unittest.mock — no real API keys are needed.

Run:
    pytest code/python/03-agent-loop/test_llm_provider.py -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Helpers to create realistic mock SDK objects
# ---------------------------------------------------------------------------

def _make_openai_response(
    content: str = "Hello",
    finish_reason: str = "stop",
    tool_calls=None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Return a minimal object that looks like an openai ChatCompletion."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_anthropic_response(
    text: str = "Hi there",
    stop_reason: str = "end_turn",
    input_tokens: int = 8,
    output_tokens: int = 4,
    tool_blocks=None,
):
    """Return a minimal object that looks like an Anthropic Message."""
    content_blocks = []
    if text:
        block = MagicMock()
        block.type = "text"
        block.text = text
        content_blocks.append(block)
    if tool_blocks:
        content_blocks.extend(tool_blocks)

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Test 1 — OpenAIProvider.chat returns a well-formed LLMResponse
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def test_openai_provider_chat_returns_llm_response(self):
        """OpenAIProvider.chat should return an LLMResponse with content."""
        mock_resp = _make_openai_response(content="96", prompt_tokens=15, completion_tokens=3)

        with patch("llm_provider.OpenAIProvider.__init__", return_value=None):
            from llm_provider import OpenAIProvider, LLMResponse

            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._model = "gpt-4o-mini"

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            provider._client = mock_client

            messages = [{"role": "user", "content": "What is 8 × 12?"}]
            result = provider.chat(messages, temperature=0.0, max_tokens=10)

        assert isinstance(result, LLMResponse)
        assert result.content == "96"
        assert result.tool_calls is None
        assert result.model == "gpt-4o-mini"
        assert result.finish_reason == "stop"
        assert result.token_usage["total_tokens"] == 18
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# Test 2 — AnthropicProvider normalises to LLMResponse
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def test_anthropic_provider_normalizes_to_llm_response(self):
        """AnthropicProvider.chat should return an LLMResponse with correct structure."""
        mock_resp = _make_anthropic_response(text="Paris", stop_reason="end_turn")

        with patch("llm_provider.AnthropicProvider.__init__", return_value=None):
            from llm_provider import AnthropicProvider, LLMResponse

            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._model = "claude-3-haiku-20240307"

            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_resp
            provider._client = mock_client

            messages = [{"role": "user", "content": "Capital of France?"}]
            result = provider.chat(messages, temperature=0.0, max_tokens=10)

        assert isinstance(result, LLMResponse)
        assert result.content == "Paris"
        assert result.tool_calls is None
        assert result.model == "claude-3-haiku-20240307"
        assert result.finish_reason == "end_turn"

    # ------------------------------------------------------------------
    # Test 3 — Anthropic converts OpenAI tool format to input_schema
    # ------------------------------------------------------------------

    def test_anthropic_converts_tools_format(self):
        """_to_anthropic_tools() should convert 'parameters' to 'input_schema'."""
        from llm_provider import AnthropicProvider

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        result = AnthropicProvider._to_anthropic_tools(openai_tools)

        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "get_weather"
        assert "input_schema" in tool
        assert "parameters" not in tool
        assert tool["input_schema"]["properties"]["city"]["type"] == "string"


# ---------------------------------------------------------------------------
# Test 4 — OllamaProvider delegates to OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def test_ollama_provider_uses_openai_compatible_endpoint(self):
        """OllamaProvider should initialise an inner OpenAIProvider with localhost base URL."""
        mock_resp = _make_openai_response(content="42")

        captured_base_url = []

        original_init = None

        def mock_openai_init(self, api_key, model="gpt-4o", base_url=None):
            self._model = model
            self._base_url = base_url
            captured_base_url.append(base_url)
            self._client = MagicMock()
            self._client.chat.completions.create.return_value = mock_resp

        with patch("llm_provider.OpenAIProvider.__init__", mock_openai_init):
            from llm_provider import OllamaProvider

            provider = OllamaProvider("llama3.1:8b")

        assert any("localhost:11434" in (url or "") for url in captured_base_url), (
            f"Expected ollama base URL in {captured_base_url}"
        )


# ---------------------------------------------------------------------------
# Tests 5–7 — FallbackProvider behaviour
# ---------------------------------------------------------------------------

class TestFallbackProvider:
    def _make_provider(self, name: str, response=None, raises=False):
        from llm_provider import LLMProvider, LLMResponse

        class _TestProvider(LLMProvider):
            def chat(self, messages, **kwargs):
                if raises:
                    raise RuntimeError(f"{name} is unavailable")
                return response

            def supports_function_calling(self): return False
            def supports_structured_output(self): return False
            def get_context_window(self): return 4096
            def get_model_name(self): return name

        return _TestProvider()

    def _dummy_response(self, model: str):
        from llm_provider import LLMResponse
        return LLMResponse(
            content="ok", tool_calls=None,
            token_usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            model=model, finish_reason="stop", latency_ms=10,
        )

    def test_fallback_uses_primary_when_available(self):
        """FallbackProvider should call only the primary when it succeeds."""
        from llm_provider import FallbackProvider

        primary_resp = self._dummy_response("primary-model")
        primary   = self._make_provider("primary-model", response=primary_resp)
        fallback  = self._make_provider("fallback-model", raises=True)

        provider = FallbackProvider(primary=primary, fallbacks=[fallback])
        result = provider.chat([{"role": "user", "content": "Hi"}])

        assert result.model == "primary-model"

    def test_fallback_switches_to_fallback_on_failure(self):
        """FallbackProvider should use the next provider when the primary fails."""
        from llm_provider import FallbackProvider

        fallback_resp = self._dummy_response("fallback-model")
        primary  = self._make_provider("primary-model", raises=True)
        fallback = self._make_provider("fallback-model", response=fallback_resp)

        provider = FallbackProvider(primary=primary, fallbacks=[fallback])
        result = provider.chat([{"role": "user", "content": "Hi"}])

        assert result.model == "fallback-model"

    def test_fallback_logs_failover_event(self):
        """FallbackProvider should log a warning when falling over to the next provider."""
        import logging
        from llm_provider import FallbackProvider

        fallback_resp = self._dummy_response("fallback-model")
        primary  = self._make_provider("primary-model", raises=True)
        fallback = self._make_provider("fallback-model", response=fallback_resp)

        provider = FallbackProvider(primary=primary, fallbacks=[fallback])

        with patch("llm_provider.logger") as mock_logger:
            provider.chat([{"role": "user", "content": "Hi"}])
            mock_logger.warning.assert_called_once()
            warning_text = mock_logger.warning.call_args[0][0] % (
                mock_logger.warning.call_args[0][1:]
                if len(mock_logger.warning.call_args[0]) > 1 else ()
            )
            assert "primary-model" in warning_text or "primary-model" in str(mock_logger.warning.call_args)


# ---------------------------------------------------------------------------
# Tests 8–10 — ModelRouter task routing
# ---------------------------------------------------------------------------

class TestModelRouter:
    def _build_router(self):
        """Create a router with two registered providers (cheap mini + smart large)."""
        from llm_provider import LLMResponse
        from model_router import ModelRouter, RouterConfig, ProviderCapabilities

        class _FakeProvider:
            def __init__(self, name, ctx):
                self._name = name
                self._ctx = ctx
            def get_model_name(self):     return self._name
            def get_context_window(self): return self._ctx
            def supports_function_calling(self): return True
            def supports_structured_output(self): return True
            def chat(self, messages, **kwargs):
                return LLMResponse("ok", None,
                                   {"prompt_tokens":5,"completion_tokens":3,"total_tokens":8},
                                   self._name, "stop", 100)
            def count_tokens(self, text): return max(1, len(text) // 4)

        config = RouterConfig()
        router = ModelRouter(config)

        router.register_provider(
            "gpt-4o-mini", _FakeProvider("gpt-4o-mini", 128_000),
            capabilities=["chat"],
            cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
            typical_latency_ms=800,
        )
        router.register_provider(
            "gpt-4o", _FakeProvider("gpt-4o", 128_000),
            capabilities=["chat", "reasoning"],
            cost_per_1k_input=0.0025, cost_per_1k_output=0.010,
            typical_latency_ms=1200,
        )
        # Add a 200K-context provider for the context-window test
        router.register_provider(
            "claude-3-5-sonnet", _FakeProvider("claude-3-5-sonnet", 200_000),
            capabilities=["chat", "reasoning"],
            cost_per_1k_input=0.003, cost_per_1k_output=0.015,
            typical_latency_ms=1500,
        )
        return router

    def test_router_selects_cheapest_for_simple_task(self):
        """With priority=cost a simple chat task should route to gpt-4o-mini."""
        from model_router import RoutingTask, RouterConfig, ModelRouter

        router = self._build_router()
        router.config.priority_order = ["cheap", "fast", "smart"]

        task = RoutingTask(
            messages=[{"role": "user", "content": "Say hello"}],
            task_type="simple_chat",
            estimated_input_tokens=20,
            estimated_output_tokens=20,
            priority="cheap",
        )
        selected_name, _ = router.route(task)
        assert "mini" in selected_name or selected_name == "gpt-4o-mini"

    def test_router_selects_smartest_for_complex_task(self):
        """With priority=quality a reasoning task should not route to gpt-4o-mini."""
        from model_router import RoutingTask, RouterConfig, ModelRouter

        router = self._build_router()
        router.config.priority_order = ["smart", "cheap", "fast"]

        task = RoutingTask(
            messages=[{"role": "user", "content": "Solve this reasoning puzzle"}],
            task_type="reasoning",
            estimated_input_tokens=500,
            estimated_output_tokens=500,
            priority="smart",
        )
        selected_name, _ = router.route(task)
        assert selected_name != "gpt-4o-mini", (
            f"Expected a 'smart' model, got '{selected_name}'"
        )

    def test_router_filters_by_context_window(self):
        """A task requiring >128K tokens should filter out gpt-4o and gpt-4o-mini."""
        from model_router import RoutingTask, RouterConfig, ModelRouter

        router = self._build_router()

        task = RoutingTask(
            messages=[{"role": "user", "content": "A very long document…"}],
            task_type="summarisation",
            estimated_input_tokens=150_000,  # exceeds 128K window
            estimated_output_tokens=500,
            priority="cheap",
        )
        selected_name, _ = router.route(task)
        # Only claude-3-5-sonnet (200K) should survive the context filter
        assert "claude" in selected_name, (
            f"Expected claude (200K ctx) to be selected, got '{selected_name}'"
        )


# ---------------------------------------------------------------------------
# Tests 11–13 — LLMFactory
# ---------------------------------------------------------------------------

class TestLLMFactory:
    def test_factory_creates_correct_provider(self):
        """LLMFactory.create('openai', …) should return an OpenAIProvider instance."""
        with patch("llm_provider.OpenAIProvider.__init__", return_value=None):
            from llm_provider import LLMFactory, OpenAIProvider

            provider = LLMFactory.create("openai", api_key="sk-test", model="gpt-4o-mini")

        assert isinstance(provider, OpenAIProvider)

    def test_factory_raises_on_unknown_provider(self):
        """LLMFactory.create('unknown', …) should raise ValueError with a useful message."""
        from llm_provider import LLMFactory

        with pytest.raises(ValueError, match="unknown"):
            LLMFactory.create("unknown_provider_xyz", api_key="key")

    def test_factory_creates_from_config_with_fallback(self):
        """create_from_config with a 'fallback' key should return a FallbackProvider."""
        with (
            patch("llm_provider.OpenAIProvider.__init__", return_value=None),
            patch("llm_provider.AnthropicProvider.__init__", return_value=None),
        ):
            from llm_provider import LLMFactory, FallbackProvider

            config = {
                "provider": "openai",
                "api_key":  "sk-test",
                "model":    "gpt-4o",
                "fallback": {
                    "provider": "anthropic",
                    "api_key":  "ant-test",
                    "model":    "claude-3-haiku-20240307",
                },
            }
            provider = LLMFactory.create_from_config(config)

        assert isinstance(provider, FallbackProvider)
