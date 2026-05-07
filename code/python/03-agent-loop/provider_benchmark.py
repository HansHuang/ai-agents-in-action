"""Provider benchmarking tool.

Measures and compares LLM providers across latency, cost, and capability.
Produces a formatted comparison report with a recommendation.

Usage:
    benchmark = ProviderBenchmark({
        "GPT-4o":            gpt4o_provider,
        "GPT-4o-mini":       mini_provider,
        "Claude-3.5-Sonnet": claude_provider,
        "Claude-3-Haiku":    haiku_provider,
    })
    results = benchmark.run_all(messages, iterations=5)
    print(benchmark.generate_report(results))

See: docs/05-the-tool-ecosystem/01-model-providers.md
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_provider import LLMProvider, LLMResponse, estimate_tokens


# ---------------------------------------------------------------------------
# Per-provider token pricing (USD per 1 000 tokens, approximate)
# Update when providers revise pricing.
# ---------------------------------------------------------------------------

KNOWN_PRICING: dict[str, dict[str, float]] = {
    # model fragment → {input, output}
    "gpt-4o":            {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini":       {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo":     {"input": 0.0005, "output": 0.0015},
    "claude-3-5-sonnet": {"input": 0.003,  "output": 0.015},
    "claude-3-haiku":    {"input": 0.00025, "output": 0.00125},
    "claude-3-opus":     {"input": 0.015,  "output": 0.075},
    "gemini-1.5-pro":    {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash":  {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash":  {"input": 0.0001,  "output": 0.0004},
    "llama-3.1-70b":     {"input": 0.0009,  "output": 0.0009},
    "mixtral-8x7b":      {"input": 0.0006,  "output": 0.0006},
    "deepseek":          {"input": 0.00027, "output": 0.0011},
}

def _lookup_pricing(model_name: str) -> dict[str, float]:
    model_lower = model_name.lower()
    for fragment, pricing in KNOWN_PRICING.items():
        if fragment in model_lower:
            return pricing
    return {"input": 0.0, "output": 0.0}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    min_ms: int
    max_ms: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: list[int] = field(default_factory=list, repr=False)


@dataclass
class CostResult:
    cost_per_call_usd: float
    cost_per_1k_input_usd: float
    cost_per_1k_output_usd: float
    input_tokens: int
    estimated_output_tokens: int


@dataclass
class CapabilityTest:
    """A single capability test case."""
    name: str
    messages: list[dict]
    # For function-calling tests
    expected_tool: str | None = None
    # For structured-output tests (expected keys)
    expected_keys: list[str] | None = None
    # Generic success predicate — overrides name-based defaults when set
    success_criteria: Callable[[LLMResponse], bool] | None = None


@dataclass
class CapabilityResult:
    test_name: str
    passed: int
    total: int
    success_rate: float


@dataclass
class BenchmarkResults:
    provider_name: str
    model_name: str
    latency: LatencyResult | None = None
    cost: CostResult | None = None
    capabilities: list[CapabilityResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ProviderBenchmark
# ---------------------------------------------------------------------------

class ProviderBenchmark:
    """Benchmark multiple LLM providers across latency, cost, and capability."""

    def __init__(self, providers: dict[str, LLMProvider]) -> None:
        self.providers = providers

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    def benchmark_latency(
        self,
        messages: list[dict],
        iterations: int = 5,
    ) -> dict[str, LatencyResult]:
        """Measure response latency for each provider.

        Args:
            messages:   Messages to send on each call.
            iterations: Number of calls per provider.

        Returns:
            Mapping of provider name → :class:`LatencyResult`.
        """
        results: dict[str, LatencyResult] = {}
        for name, provider in self.providers.items():
            samples: list[int] = []
            for _ in range(iterations):
                try:
                    t0 = time.monotonic()
                    provider.chat(messages, temperature=0.0, max_tokens=100)
                    samples.append(int((time.monotonic() - t0) * 1000))
                except Exception:
                    pass  # skip failed calls from stats

            if not samples:
                continue

            sorted_s = sorted(samples)
            results[name] = LatencyResult(
                min_ms=sorted_s[0],
                max_ms=sorted_s[-1],
                avg_ms=round(statistics.mean(samples), 1),
                p50_ms=round(statistics.median(samples), 1),
                p95_ms=round(_percentile(sorted_s, 95), 1),
                p99_ms=round(_percentile(sorted_s, 99), 1),
                samples=samples,
            )
        return results

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def benchmark_cost(
        self,
        messages: list[dict],
        estimated_output_tokens: int = 500,
    ) -> dict[str, CostResult]:
        """Estimate cost per call based on token counts and known pricing.

        Counts input tokens from *messages*; uses *estimated_output_tokens*
        for the output side.

        Returns:
            Mapping of provider name → :class:`CostResult`.
        """
        results: dict[str, CostResult] = {}
        for name, provider in self.providers.items():
            # Count input tokens
            input_text = " ".join(
                msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)
            )
            input_tokens = provider.count_tokens(input_text)

            pricing = _lookup_pricing(provider.get_model_name())
            cost_input  = (input_tokens / 1000) * pricing["input"]
            cost_output = (estimated_output_tokens / 1000) * pricing["output"]

            results[name] = CostResult(
                cost_per_call_usd=round(cost_input + cost_output, 6),
                cost_per_1k_input_usd=pricing["input"],
                cost_per_1k_output_usd=pricing["output"],
                input_tokens=input_tokens,
                estimated_output_tokens=estimated_output_tokens,
            )
        return results

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------

    def benchmark_capability(
        self,
        test_cases: list[CapabilityTest],
    ) -> dict[str, list[CapabilityResult]]:
        """Run capability tests on each provider.

        Returns:
            Mapping of provider name → list of :class:`CapabilityResult`.
        """
        results: dict[str, list[CapabilityResult]] = {n: [] for n in self.providers}

        for test in test_cases:
            for name, provider in self.providers.items():
                passed = 0
                total = 1
                try:
                    resp = provider.chat(test.messages, temperature=0.0, max_tokens=500)

                    if test.success_criteria:
                        passed = int(test.success_criteria(resp))
                    elif test.expected_tool:
                        # Function-calling test
                        if resp.tool_calls:
                            called = [tc["function"]["name"] for tc in resp.tool_calls]
                            passed = int(test.expected_tool in called)
                    elif test.expected_keys:
                        # Structured-output test: check JSON keys
                        import json
                        try:
                            data = json.loads(resp.content or "{}")
                            passed = int(all(k in data for k in test.expected_keys))
                        except Exception:
                            passed = 0
                    else:
                        # Generic: passes as long as there's non-empty content
                        passed = int(bool(resp.content and resp.content.strip()))
                except Exception:
                    passed = 0

                results[name].append(CapabilityResult(
                    test_name=test.name,
                    passed=passed,
                    total=total,
                    success_rate=passed / total,
                ))
        return results

    # ------------------------------------------------------------------
    # Combined run
    # ------------------------------------------------------------------

    def run_all(
        self,
        messages: list[dict],
        iterations: int = 5,
        estimated_output_tokens: int = 500,
        capability_tests: list[CapabilityTest] | None = None,
    ) -> dict[str, BenchmarkResults]:
        """Run all benchmarks and return consolidated results."""
        latency_results   = self.benchmark_latency(messages, iterations)
        cost_results      = self.benchmark_cost(messages, estimated_output_tokens)
        cap_results       = self.benchmark_capability(capability_tests or [])

        combined: dict[str, BenchmarkResults] = {}
        for name, provider in self.providers.items():
            combined[name] = BenchmarkResults(
                provider_name=name,
                model_name=provider.get_model_name(),
                latency=latency_results.get(name),
                cost=cost_results.get(name),
                capabilities=cap_results.get(name, []),
            )
        return combined

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, results: dict[str, BenchmarkResults]) -> str:
        """Produce a formatted ASCII comparison table and recommendation."""
        lines: list[str] = []
        lines.append("\n" + "=" * 90)
        lines.append("LLM PROVIDER BENCHMARK REPORT")
        lines.append("=" * 90)

        # -- Latency table --
        if any(r.latency for r in results.values()):
            lines.append("\n## Latency (lower is better)\n")
            hdr = f"{'Provider':<26}  {'p50 (ms)':<10}  {'p95 (ms)':<10}  {'avg (ms)':<10}  {'min–max'}"
            lines.append(hdr)
            lines.append("-" * 72)
            for name, r in sorted(results.items(),
                                   key=lambda x: x[1].latency.p50_ms if x[1].latency else 999_999):
                if r.latency:
                    rng = f"{r.latency.min_ms}–{r.latency.max_ms} ms"
                    lines.append(
                        f"{name:<26}  {r.latency.p50_ms:<10}  {r.latency.p95_ms:<10}"
                        f"  {r.latency.avg_ms:<10}  {rng}"
                    )

        # -- Cost table --
        if any(r.cost for r in results.values()):
            lines.append("\n## Cost (approximate, lower is better)\n")
            hdr2 = f"{'Provider':<26}  {'Cost/call ($)':<15}  {'Input $/1K':<12}  {'Output $/1K'}"
            lines.append(hdr2)
            lines.append("-" * 72)
            for name, r in sorted(results.items(),
                                   key=lambda x: x[1].cost.cost_per_call_usd if x[1].cost else 999.0):
                if r.cost:
                    lines.append(
                        f"{name:<26}  {r.cost.cost_per_call_usd:<15.6f}"
                        f"  {r.cost.cost_per_1k_input_usd:<12.5f}  {r.cost.cost_per_1k_output_usd:.5f}"
                    )

        # -- Capability table --
        all_tests: list[str] = []
        for r in results.values():
            for cr in r.capabilities:
                if cr.test_name not in all_tests:
                    all_tests.append(cr.test_name)

        if all_tests:
            lines.append("\n## Capability Tests\n")
            col_w = 14
            header = f"{'Provider':<26}  " + "  ".join(f"{t[:col_w]:<{col_w}}" for t in all_tests)
            lines.append(header)
            lines.append("-" * (28 + len(all_tests) * (col_w + 2)))
            for name, r in results.items():
                test_map = {cr.test_name: cr for cr in r.capabilities}
                row = f"{name:<26}  "
                cols = []
                for t in all_tests:
                    cr = test_map.get(t)
                    if cr:
                        icon = "✓" if cr.success_rate >= 1.0 else "✗"
                        cols.append(f"{icon} {int(cr.success_rate * 100):>3}%")
                    else:
                        cols.append(f"{'N/A':<{col_w}}")
                row += "  ".join(f"{c:<{col_w}}" for c in cols)
                lines.append(row)

        # -- Recommendation --
        lines.append("\n## Recommendation\n")
        rec = self._recommend(results)
        lines.append(rec)
        lines.append("=" * 90)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recommend(self, results: dict[str, BenchmarkResults]) -> str:
        """Generate a workload-based recommendation string."""
        if not results:
            return "No results available."

        # Find cheapest and fastest
        cost_ranked = sorted(
            [(n, r) for n, r in results.items() if r.cost],
            key=lambda x: x[1].cost.cost_per_call_usd,  # type: ignore[union-attr]
        )
        latency_ranked = sorted(
            [(n, r) for n, r in results.items() if r.latency],
            key=lambda x: x[1].latency.p50_ms,  # type: ignore[union-attr]
        )

        cheapest_name = cost_ranked[0][0] if cost_ranked else None
        fastest_name  = latency_ranked[0][0] if latency_ranked else None

        # "Smart" heuristic: pick the best overall by combined rank
        combined_rank: dict[str, int] = {n: 0 for n in results}
        for rank, (n, _) in enumerate(cost_ranked):
            combined_rank[n] += rank
        for rank, (n, _) in enumerate(latency_ranked):
            combined_rank[n] += rank
        best_overall = min(combined_rank, key=combined_rank.__getitem__)

        # Compute relative cost vs most expensive
        if len(cost_ranked) >= 2:
            most_expensive_cost = cost_ranked[-1][1].cost.cost_per_call_usd  # type: ignore
            cheapest_cost       = cost_ranked[0][1].cost.cost_per_call_usd   # type: ignore
            if most_expensive_cost > 0:
                pct = int(cheapest_cost / most_expensive_cost * 100)
                cost_note = (
                    f"'{cheapest_name}' is the most cost-efficient: "
                    f"{pct}% the cost of '{cost_ranked[-1][0]}' per call."
                )
            else:
                cost_note = f"'{cheapest_name}' is the lowest-cost option."
        else:
            cost_note = ""

        return (
            f"For the tested workload, '{best_overall}' offers the best balance "
            f"of latency and cost. {cost_note}\n"
            f"If latency is critical, prefer '{fastest_name}'. "
            f"If cost is critical, prefer '{cheapest_name}'."
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[int | float], p: float) -> float:
    """Return the *p*-th percentile of already-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import os
    import logging
    logging.basicConfig(level=logging.WARNING)

    from llm_provider import LLMFactory, OllamaProvider

    providers: dict[str, LLMProvider] = {}

    oai_key = os.environ.get("OPENAI_API_KEY")
    ant_key = os.environ.get("ANTHROPIC_API_KEY")

    if oai_key:
        providers["GPT-4o"]       = LLMFactory.create("openai", api_key=oai_key, model="gpt-4o")
        providers["GPT-4o-mini"]  = LLMFactory.create("openai", api_key=oai_key, model="gpt-4o-mini")
    if ant_key:
        providers["Claude-Sonnet"] = LLMFactory.create("anthropic", api_key=ant_key,
                                                        model="claude-3-5-sonnet-20241022")
        providers["Claude-Haiku"]  = LLMFactory.create("anthropic", api_key=ant_key,
                                                        model="claude-3-haiku-20240307")
    providers["Ollama-Llama3.1"] = OllamaProvider("llama3.1:8b")

    if not providers:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run the benchmark.")
        return

    benchmark = ProviderBenchmark(providers)

    messages = [{"role": "user", "content": "What is 12 × 8? Answer with just the number."}]

    # Tool for function-calling test
    weather_tool = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    fn_call_test = CapabilityTest(
        name="function_calling",
        messages=[{"role": "user", "content": "What's the weather in London?"}],
        expected_tool="get_weather",
    )
    structured_test = CapabilityTest(
        name="structured_output",
        messages=[{
            "role": "user",
            "content": 'Reply with JSON: {"answer": <number>, "unit": "number"}'
        }],
        expected_keys=["answer", "unit"],
    )

    print("Running benchmark (5 iterations per provider, may take a minute)…")
    results = benchmark.run_all(
        messages,
        iterations=5,
        estimated_output_tokens=500,
        capability_tests=[fn_call_test, structured_test],
    )
    print(benchmark.generate_report(results))


if __name__ == "__main__":
    _demo()
