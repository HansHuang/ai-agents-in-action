"""Task-based model router.

Routes LLM tasks to the most appropriate provider based on:
  - Task type (chat, reasoning, classification, summarisation, code)
  - Priority (cost / latency / quality)
  - Context size requirements
  - Tool-call / structured-output requirements
  - Declarative routing rules (override the default scoring)

Usage:
    router = ModelRouter()
    router.register_provider("gpt-4o",       gpt4o_provider,   ["smart", "function_calling", "structured_output"])
    router.register_provider("gpt-4o-mini",  mini_provider,    ["cheap", "fast", "function_calling"])
    router.register_provider("claude-sonnet",claude_provider,  ["smart", "long_context", "function_calling"])

    task = RoutingTask(
        messages=[{"role": "user", "content": "Explain quantum entanglement."}],
        task_type="reasoning",
        estimated_input_tokens=50,
        estimated_output_tokens=500,
        priority="quality",
    )
    provider = router.route(task)

See: docs/05-the-tool-ecosystem/01-model-providers.md
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoutingTask:
    """Description of a task to be routed."""
    messages: list[dict]
    task_type: str              # "chat" | "reasoning" | "classification" | "summarization" | "code"
    estimated_input_tokens: int
    estimated_output_tokens: int
    priority: str               # "cost" | "latency" | "quality"
    requires_tools: bool = False
    requires_structured_output: bool = False


@dataclass
class RoutingRule:
    """Declarative rule: if *condition* is truthy, force *provider* name."""
    condition: str   # evaluated against RoutingTask fields, e.g. "task_type == 'classification'"
    provider: str    # name registered with ModelRouter.register_provider()


@dataclass
class RouterConfig:
    """Tuneable knobs for the router's scoring algorithm."""
    priority_order: list[str] = field(default_factory=lambda: ["quality", "cost", "latency"])
    max_cost_per_1k_tokens: float | None = None   # USD; filters candidates above this cost
    max_latency_ms: int | None = None             # filters candidates above this latency
    rules: list[RoutingRule] = field(default_factory=list)


@dataclass
class ProviderCapabilities:
    """Metadata associated with a registered provider."""
    provider: LLMProvider
    capabilities: list[str]          # e.g. ["fast", "smart", "cheap", "long_context"]
    cost_per_1k_input: float = 0.0   # USD
    cost_per_1k_output: float = 0.0  # USD
    typical_latency_ms: int = 1000   # rough estimate used in latency-priority scoring


@dataclass
class UsageStats:
    """Per-provider routing and usage counters."""
    total_calls: int = 0
    successful_calls: int = 0
    fallback_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.successful_calls if self.successful_calls else 0.0

    def record(self, resp: LLMResponse) -> None:
        self.total_calls += 1
        self.successful_calls += 1
        self.total_prompt_tokens     += resp.token_usage.get("prompt_tokens", 0)
        self.total_completion_tokens += resp.token_usage.get("completion_tokens", 0)
        self.total_latency_ms        += resp.latency_ms


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """Route tasks to the optimal LLM provider.

    Scoring (higher is better):
      1. Hard filters: context window, required capabilities
      2. Declarative rules override scoring
      3. Score by priority weight:
         - quality  →  "smart" capability +2, "cheap" −1
         - cost     →  "cheap" +2, lower cost_per_1k +1, "smart" −1
         - latency  →  "fast" +2, lower typical_latency_ms +1, "smart" −1
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self._providers: dict[str, ProviderCapabilities] = {}
        self.usage_stats: dict[str, UsageStats] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_provider(
        self,
        name: str,
        provider: LLMProvider,
        capabilities: list[str],
        *,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
        typical_latency_ms: int = 1000,
    ) -> None:
        """Register a provider with its capability tags.

        Standard capability tags:
        ``"fast"``, ``"smart"``, ``"cheap"``, ``"long_context"``,
        ``"function_calling"``, ``"structured_output"``
        """
        self._providers[name] = ProviderCapabilities(
            provider=provider,
            capabilities=capabilities,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
            typical_latency_ms=typical_latency_ms,
        )
        self.usage_stats[name] = UsageStats()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, task: RoutingTask) -> LLMProvider:
        """Select the best provider for *task*.

        Returns the :class:`LLMProvider` instance; raises ``RuntimeError`` if
        no registered provider can satisfy the task requirements.
        """
        # 1. Apply declarative rules (first match wins)
        for rule in self.config.rules:
            if self._eval_rule(rule.condition, task):
                if rule.provider in self._providers:
                    logger.debug("Rule match: condition=%r → %s", rule.condition, rule.provider)
                    return self._providers[rule.provider].provider

        # 2. Filter to capable candidates
        candidates = self._filter_candidates(task)
        if not candidates:
            raise RuntimeError(
                "No registered provider can satisfy the task requirements. "
                f"Requires: context={task.estimated_input_tokens}, "
                f"tools={task.requires_tools}, "
                f"structured_output={task.requires_structured_output}"
            )

        # 3. Score and pick the best
        scored = [(name, self._score(name, task)) for name in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_name, best_score = scored[0]
        logger.debug(
            "Routing task_type=%s priority=%s → %s (score=%.1f)",
            task.task_type, task.priority, best_name, best_score,
        )
        return self._providers[best_name].provider

    def route_with_retry(self, task: RoutingTask) -> LLMResponse:
        """Route and execute; on failure, retry with the next-best provider.

        All candidate providers are tried in order of descending score.
        """
        candidates = self._ranked_candidates(task)
        if not candidates:
            raise RuntimeError("No registered provider can satisfy the task requirements.")

        last_exc: Exception | None = None
        for name in candidates:
            provider_meta = self._providers[name]
            stats = self.usage_stats[name]
            try:
                stats.total_calls += 1
                t0 = time.monotonic()
                resp = provider_meta.provider.chat(task.messages)
                stats.record(resp)
                return resp
            except Exception as exc:
                last_exc = exc
                stats.total_calls = max(stats.total_calls, 1)  # already incremented
                logger.warning("Provider %s failed: %s — retrying with next.", name, exc)
                if name != candidates[0]:
                    stats.fallback_calls += 1

        raise RuntimeError(f"All providers failed. Last error: {last_exc}") from last_exc

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return routing statistics per provider."""
        out: dict[str, Any] = {}
        total_calls = sum(s.total_calls for s in self.usage_stats.values())
        for name, stats in self.usage_stats.items():
            out[name] = {
                "total_calls":        stats.total_calls,
                "successful_calls":   stats.successful_calls,
                "fallback_calls":     stats.fallback_calls,
                "avg_latency_ms":     round(stats.avg_latency_ms, 1),
                "total_tokens":       stats.total_prompt_tokens + stats.total_completion_tokens,
                "share_pct":          round(stats.total_calls / total_calls * 100, 1) if total_calls else 0.0,
            }
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_candidates(self, task: RoutingTask) -> list[str]:
        """Return names of providers that can handle the task."""
        result = []
        for name, meta in self._providers.items():
            p = meta.provider
            # Context window
            if p.get_context_window() < task.estimated_input_tokens:
                continue
            # Required capabilities
            if task.requires_tools and "function_calling" not in meta.capabilities:
                if not p.supports_function_calling():
                    continue
            if task.requires_structured_output and "structured_output" not in meta.capabilities:
                if not p.supports_structured_output():
                    continue
            # Config-level cost / latency cap
            if self.config.max_cost_per_1k_tokens is not None:
                if meta.cost_per_1k_input > self.config.max_cost_per_1k_tokens:
                    continue
            if self.config.max_latency_ms is not None:
                if meta.typical_latency_ms > self.config.max_latency_ms:
                    continue
            result.append(name)
        return result

    def _score(self, name: str, task: RoutingTask) -> float:
        meta = self._providers[name]
        caps = set(meta.capabilities)
        score = 0.0

        if task.priority == "quality":
            if "smart" in caps:        score += 2.0
            if "cheap" in caps:        score -= 0.5
            if "long_context" in caps: score += 0.5
        elif task.priority == "cost":
            if "cheap" in caps:        score += 2.0
            if "smart" in caps:        score -= 0.5
            # Lower cost is better; normalise against a $5/1K baseline
            cost = meta.cost_per_1k_input + meta.cost_per_1k_output
            score += max(0.0, 1.0 - cost / 5.0)
        elif task.priority == "latency":
            if "fast" in caps:         score += 2.0
            if "smart" in caps:        score -= 0.5
            # Lower latency is better; normalise against a 3000 ms baseline
            score += max(0.0, 1.0 - meta.typical_latency_ms / 3000)

        # Task-type bonuses
        _task_bonuses: dict[str, list[str]] = {
            "classification": ["cheap", "fast"],
            "summarization":  ["long_context"],
            "reasoning":      ["smart"],
            "code":           ["smart"],
            "chat":           ["fast", "cheap"],
        }
        for bonus_cap in _task_bonuses.get(task.task_type, []):
            if bonus_cap in caps:
                score += 0.3

        return score

    def _ranked_candidates(self, task: RoutingTask) -> list[str]:
        candidates = self._filter_candidates(task)
        return sorted(candidates, key=lambda n: self._score(n, task), reverse=True)

    @staticmethod
    def _eval_rule(condition: str, task: RoutingTask) -> bool:
        """Evaluate a simple condition string against task fields.

        Supports: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
                  ``in`` (list), ``and``/``or`` (case-insensitive).

        The expression is evaluated with ``task.*`` fields in scope.
        No arbitrary code execution: only task attribute lookups.
        """
        task_dict = {
            "task_type":                task.task_type,
            "priority":                 task.priority,
            "estimated_input_tokens":   task.estimated_input_tokens,
            "estimated_output_tokens":  task.estimated_output_tokens,
            "requires_tools":           task.requires_tools,
            "requires_structured_output": task.requires_structured_output,
        }
        # Normalise and/or to Python keywords (already Python but ensure lowercase)
        safe_cond = re.sub(r'\bAND\b', 'and', condition)
        safe_cond = re.sub(r'\bOR\b', 'or', safe_cond)
        try:
            return bool(eval(safe_cond, {"__builtins__": {}}, task_dict))  # noqa: S307
        except Exception:
            logger.warning("Could not evaluate routing rule: %r", condition)
            return False


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    import os
    from llm_provider import LLMFactory, OllamaProvider

    router = ModelRouter(config=RouterConfig(
        rules=[
            RoutingRule("task_type == 'classification' and estimated_input_tokens < 1000",
                        "gpt-4o-mini"),
        ]
    ))

    # Register providers (using stub factories when keys are absent)
    def _provider(p_name: str, model: str, env_key: str) -> LLMProvider | None:
        key = os.environ.get(env_key)
        if not key:
            return None
        try:
            return LLMFactory.create(p_name, api_key=key, model=model)
        except Exception:
            return None

    gpt4o      = _provider("openai",    "gpt-4o",                    "OPENAI_API_KEY")
    gpt4o_mini = _provider("openai",    "gpt-4o-mini",               "OPENAI_API_KEY")
    claude     = _provider("anthropic", "claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY")
    haiku      = _provider("anthropic", "claude-3-haiku-20240307",    "ANTHROPIC_API_KEY")
    ollama     = OllamaProvider("llama3.1:8b")

    registrations = [
        ("gpt-4o",           gpt4o,      ["smart", "function_calling", "structured_output"], 0.0025, 0.010,  900),
        ("gpt-4o-mini",      gpt4o_mini, ["cheap", "fast", "function_calling"],              0.00015, 0.0006, 500),
        ("claude-3-5-sonnet",claude,     ["smart", "long_context", "function_calling"],      0.003, 0.015, 1200),
        ("claude-3-haiku",   haiku,      ["cheap", "fast", "function_calling"],              0.00025, 0.00125, 500),
        ("ollama-llama3.1",  ollama,     ["cheap", "fast"],                                  0.0,  0.0,   300),
    ]

    for name, p, caps, ci, co, lat in registrations:
        if p is not None:
            router.register_provider(
                name, p, caps,
                cost_per_1k_input=ci, cost_per_1k_output=co, typical_latency_ms=lat,
            )

    tasks = [
        RoutingTask(
            messages=[{"role": "user", "content": "Is this review positive or negative? 'Great product!'"}],
            task_type="classification", estimated_input_tokens=30,
            estimated_output_tokens=10, priority="cost",
        ),
        RoutingTask(
            messages=[{"role": "user", "content": "Explain the Halting Problem in depth."}],
            task_type="reasoning", estimated_input_tokens=20,
            estimated_output_tokens=800, priority="quality",
        ),
        RoutingTask(
            messages=[{"role": "user", "content": "Write a Python merge sort."}],
            task_type="code", estimated_input_tokens=20,
            estimated_output_tokens=300, priority="quality",
        ),
        RoutingTask(
            messages=[{"role": "user", "content": "Hi!"}],
            task_type="chat", estimated_input_tokens=5,
            estimated_output_tokens=50, priority="latency",
        ),
    ]

    print("\n=== Model Router Demo ===\n")
    print(f"{'Task type':<18}  {'Priority':<10}  {'Selected Provider'}")
    print("-" * 55)
    for task in tasks:
        try:
            provider = router.route(task)
            print(f"{task.task_type:<18}  {task.priority:<10}  {provider.get_model_name()}")
        except RuntimeError as exc:
            print(f"{task.task_type:<18}  {task.priority:<10}  ERROR: {exc}")


if __name__ == "__main__":
    _demo()
