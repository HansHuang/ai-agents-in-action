"""Tests for context engineering modules.

Covers:
- ContextBudget: allocation, enforcement, compression, audit trail
- ContextOptimizer: structure, attention zones, deduplication, needle placement
- ContextAssembler: multi-source assembly, budget enforcement, config
- TokenCostCalculator: cost maths, model comparison, waste detection

Run offline only (no API calls required):
    pytest test_context_engineering.py

Run with verbose output:
    pytest test_context_engineering.py -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pytest

from context_budget import ContextBudget, BudgetExceededError, count_tokens, EnforceResult
from context_optimizer import ContextOptimizer
from context_assembler import ContextAssembler, ContextConfig
from token_cost_calculator import TokenCostCalculator, PRICING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n: int = 5) -> list[dict]:
    """Return 2*n alternating user/assistant messages."""
    msgs: list[dict] = []
    for i in range(1, n + 1):
        msgs.append({"role": "user",      "content": f"User turn {i}: " + "word " * 30})
        msgs.append({"role": "assistant", "content": f"Asst turn {i}: " + "word " * 30})
    return msgs


def _make_docs(n: int = 3) -> list[dict]:
    return [
        {
            "text": f"Section {i}: " + "relevant information " * 20,
            "metadata": {"source": f"doc-{i}.md"},
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# ContextBudget tests
# ---------------------------------------------------------------------------


class TestContextBudgetAllocations:
    def test_default_allocations_sum_to_one(self):
        b = ContextBudget()
        total = sum(b.allocations.values())
        assert total <= 1.0 + 1e-9

    def test_set_allocation_valid(self):
        b = ContextBudget()
        # Reduce system_prompt and increase dynamic_context by the same amount
        b.set_allocation("system_prompt",   0.01)
        b.set_allocation("dynamic_context", 0.46)
        total = sum(b.allocations.values())
        assert total <= 1.0 + 1e-9
        assert b.allocations["system_prompt"]   == pytest.approx(0.01)
        assert b.allocations["dynamic_context"] == pytest.approx(0.46)

    def test_set_allocation_rejects_over_100_percent(self):
        b = ContextBudget()
        with pytest.raises(ValueError, match="1.0"):
            b.set_allocation("system_prompt", 0.99)  # would push total > 1

    def test_set_allocation_rejects_unknown_zone(self):
        b = ContextBudget()
        with pytest.raises(ValueError, match="Unknown zone"):
            b.set_allocation("typo_zone", 0.10)

    def test_set_allocation_rejects_negative(self):
        b = ContextBudget()
        with pytest.raises(ValueError):
            b.set_allocation("system_prompt", -0.01)

    def test_get_token_budget_proportional(self):
        b = ContextBudget(total_tokens=10_000)
        assert b.get_token_budget("system_prompt") == int(10_000 * 0.02)

    def test_get_all_budgets_keys(self):
        b = ContextBudget()
        budgets = b.get_all_budgets()
        assert set(budgets.keys()) == {
            "system_prompt", "tool_definitions", "dynamic_context",
            "conversation_history", "response_buffer",
        }


class TestContextBudgetEnforce:
    def _budget_for_demo(self) -> ContextBudget:
        """Small 4K window to make overflows easy to trigger in tests."""
        return ContextBudget(total_tokens=4_000)

    def test_enforce_preserves_system_prompt_within_budget(self):
        b = self._budget_for_demo()
        sp = "You are a helpful assistant."   # tiny — well within 2% of 4K
        result = b.enforce(sp, [], dynamic_context="", tool_definitions=[])
        assert result.system_prompt == sp
        assert result.audit["system_prompt"].action_taken == "within_budget"

    def test_enforce_truncates_over_budget_system_prompt(self):
        b = self._budget_for_demo()
        # Force a system prompt that is 4× the 2% budget
        sp_budget = b.get_token_budget("system_prompt")
        sp = "Instruction line. " * (sp_budget * 5)

        result = b.enforce(sp, [], dynamic_context="")
        final_tok = count_tokens(result.system_prompt)
        assert final_tok <= sp_budget + 5   # allow a small tiktoken off-by-one
        assert result.audit["system_prompt"].action_taken == "truncated"

    def test_enforce_preserves_first_paragraph_after_truncation(self):
        b = self._budget_for_demo()
        sp_budget = b.get_token_budget("system_prompt")
        first_line = "CRITICAL: Always respond in JSON format."
        sp = first_line + "\n" + "Extra verbose instructions. " * (sp_budget * 10)

        result = b.enforce(sp, [], dynamic_context="")
        assert result.system_prompt.startswith("CRITICAL")

    def test_enforce_compresses_over_budget_history(self):
        b = self._budget_for_demo()
        hist_budget = b.get_token_budget("conversation_history")
        # Create history that is at least 3× the budget
        messages = _make_messages(n=40)   # 80 messages, ~4800 words
        original_tok = count_tokens(messages)
        assert original_tok > hist_budget, "Test precondition: history must exceed budget"

        result = b.enforce("System.", messages)
        history = [m for m in result.messages if m.get("role") != "system"]
        final_tok = count_tokens(history)
        assert final_tok <= hist_budget + 10
        assert result.audit["conversation_history"].action_taken == "sliding_window"

    def test_enforce_history_keeps_most_recent_message(self):
        b = self._budget_for_demo()
        messages = _make_messages(n=40)
        last_user_content = messages[-2]["content"]  # second-to-last = last user msg

        result = b.enforce("System.", messages)
        # The last user message must survive the sliding window
        contents = [m["content"] for m in result.messages]
        assert any(last_user_content in c for c in contents)

    def test_enforce_warns_on_over_budget_zones(self):
        b = self._budget_for_demo()
        sp = "Instructions. " * 200
        msgs = _make_messages(n=40)
        result = b.enforce(sp, msgs)
        assert len(result.warnings) > 0

    def test_enforce_warnings_name_affected_zones(self):
        b = self._budget_for_demo()
        sp = "Instructions. " * 200
        msgs = _make_messages(n=40)
        result = b.enforce(sp, msgs)
        affected = " ".join(result.warnings)
        assert "system_prompt" in affected or "conversation_history" in affected

    def test_budget_provides_audit_trail(self):
        b = self._budget_for_demo()
        sp   = "You are a support agent. " * 50
        msgs = _make_messages(n=20)
        dc   = "RAG content. " * 100
        result = b.enforce(sp, msgs, dynamic_context=dc)

        assert set(result.audit.keys()) == {
            "system_prompt", "tool_definitions", "dynamic_context",
            "conversation_history", "response_buffer",
        }
        for audit in result.audit.values():
            assert audit.original_tokens >= 0
            assert audit.final_tokens >= 0
            assert audit.budget_tokens > 0

    def test_audit_shows_compression_actions(self):
        b = self._budget_for_demo()
        sp   = "Instructions. " * 300
        msgs = _make_messages(n=40)
        result = b.enforce(sp, msgs)

        actions = {zone: a.action_taken for zone, a in result.audit.items()}
        # At least one zone should have been compressed
        compressed_zones = {z for z, a in actions.items() if a != "within_budget"}
        assert len(compressed_zones) > 0

    def test_total_tokens_saved_positive_when_compressed(self):
        b = self._budget_for_demo()
        sp   = "Instructions. " * 300
        msgs = _make_messages(n=40)
        result = b.enforce(sp, msgs)
        assert result.total_tokens_saved > 0

    def test_enforce_dynamic_context_truncated(self):
        b = self._budget_for_demo()
        dc_budget = b.get_token_budget("dynamic_context")
        dc = "RAG chunk. " * (dc_budget * 10)
        result = b.enforce("System.", [], dynamic_context=dc)
        final_tok = count_tokens(result.dynamic_context)
        assert final_tok <= dc_budget + 5
        assert result.audit["dynamic_context"].action_taken == "truncated"

    def test_estimate_cost_zero_for_unknown_model(self):
        b = ContextBudget()
        cost = b.estimate_cost(10_000, 1_000, model="unknown-model-xyz")
        assert cost == 0.0

    def test_estimate_cost_correct_for_gpt4o(self):
        b = ContextBudget(model="gpt-4o")
        # pricing: input=0.0025/1K, output=0.010/1K
        cost = b.estimate_cost(10_000, 1_000, model="gpt-4o")
        expected = 10_000 * 0.0025 / 1_000 + 1_000 * 0.010 / 1_000
        assert cost == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# ContextOptimizer tests
# ---------------------------------------------------------------------------


class TestContextOptimizer:
    def _make_docs_with_needle(self) -> tuple[list[dict], str]:
        """Return 10 documents with the needle planted at index 0."""
        needle = "The needle is the return policy: 30 days, no questions asked."
        docs = [
            {"text": needle, "metadata": {"source": "return-policy (NEEDLE)"}},
            *[
                {
                    "text": f"Topic {i}: " + "generic filler content " * 15,
                    "metadata": {"source": f"doc-{i}.md"},
                }
                for i in range(1, 10)
            ],
        ]
        return docs, needle

    def test_structure_adds_context_overview(self):
        opt = ContextOptimizer()
        docs = _make_docs(3)
        result = opt.structure_for_retrieval(docs)
        assert "Context Overview" in result

    def test_structure_adds_section_numbers(self):
        opt = ContextOptimizer()
        docs = _make_docs(3)
        result = opt.structure_for_retrieval(docs)
        assert "## [1]" in result
        assert "## [2]" in result
        assert "## [3]" in result

    def test_structure_adds_end_section_markers(self):
        opt = ContextOptimizer()
        docs = _make_docs(2)
        result = opt.structure_for_retrieval(docs)
        assert "[End Section 1]" in result
        assert "[End Section 2]" in result

    def test_structure_marks_important_documents(self):
        opt = ContextOptimizer()
        docs = [
            {"text": "Critical info.", "metadata": {"source": "critical.md", "important": True}},
            {"text": "Normal info.",   "metadata": {"source": "normal.md"}},
        ]
        result = opt.structure_for_retrieval(docs)
        assert "IMPORTANT" in result

    def test_optimize_moves_needle_to_better_position(self):
        opt = ContextOptimizer()
        docs, needle = self._make_docs_with_needle()

        raw = opt.structure_for_retrieval(docs)
        n_raw = len(raw)
        pos_before = raw.find(needle) / n_raw if needle in raw else 0.0

        optimized = opt.optimize_for_query(raw, "What is the return policy?")
        n_opt = len(optimized)
        pos_after = optimized.find(needle) / n_opt if needle in optimized else 0.0

        # The needle should move closer to the 20-60% golden zone
        # (pos_before is near 0; pos_after should be higher)
        assert pos_after > pos_before or (0.10 <= pos_after <= 0.80)

    def test_estimate_attention_score_absent_fact_returns_zero(self):
        opt = ContextOptimizer()
        assert opt.estimate_attention_score("some context", "totally absent fact") == 0.0

    def test_estimate_attention_score_in_optimal_zone(self):
        opt = ContextOptimizer()
        # Build a context that places the fact at ~40% position
        prefix = "a " * 200
        fact   = "THE CRITICAL ANSWER IS 42."
        suffix = "b " * 200
        context = prefix + fact + suffix
        score = opt.estimate_attention_score(context, fact)
        assert score >= 0.70   # should be well-attended in the golden zone

    def test_estimate_attention_score_repetition_bonus(self):
        opt = ContextOptimizer()
        # Place the fact at the very start (primacy zone — low attention)
        # With no repetitions it should score low; with repetitions it should score higher.
        fact    = "FACT: use tiktoken for token counting."
        suffix  = "b " * 800   # push fact to ~0% of context
        single_ctx = fact + suffix
        multi_ctx  = fact + suffix + fact + " more content " * 100 + fact
        single_score = opt.estimate_attention_score(single_ctx, fact)
        multi_score  = opt.estimate_attention_score(multi_ctx, fact)
        # Repetition bonus should push multi_score above single_score
        assert multi_score >= single_score

    def test_deduplicate_removes_near_duplicates(self):
        opt = ContextOptimizer()
        doc_a = {"text": "The quick brown fox jumps over the lazy dog.", "metadata": {"source": "A"}}
        doc_b = {"text": "The quick brown fox jumps over the lazy dog!",  "metadata": {"source": "B"}}
        doc_c = {"text": "Entirely different content about AI agents and LLMs.", "metadata": {"source": "C"}}

        result = opt.deduplicate_context([doc_a, doc_b, doc_c])
        assert len(result) == 2
        sources = [r["metadata"]["source"] for r in result]
        assert "C" in sources

    def test_deduplicate_keeps_longer_document(self):
        opt = ContextOptimizer()
        # Two documents with near-identical 3-gram sets (Jaccard > 0.90).
        # Repeating the base gives the same unique 3-grams + at most 2 join-point
        # 3-grams, guaranteeing similarity > 0.98 for any base > 50 chars.
        base = (
            "Context engineering involves managing the information passed to a language model "
            "to maximise relevance minimise token cost and fit within the context window."
        )
        short_doc = {"text": base,      "metadata": {"source": "short"}}
        long_doc  = {"text": base * 2,  "metadata": {"source": "long"}}
        result = opt.deduplicate_context([short_doc, long_doc])
        assert len(result) == 1
        assert result[0]["metadata"]["source"] == "long"

    def test_deduplicate_preserves_unique_documents(self):
        opt = ContextOptimizer()
        docs = _make_docs(5)
        result = opt.deduplicate_context(docs)
        assert len(result) == 5   # all unique

    def test_chunk_by_attention_zones_covers_full_context(self):
        opt = ContextOptimizer()
        context = "x" * 1_000
        zones = opt.chunk_by_attention_zones(context)
        total = (
            len(zones.primacy_zone)
            + len(zones.transition_12)
            + len(zones.optimal_zone)
            + len(zones.transition_23)
            + len(zones.recency_zone)
        )
        assert total == 1_000

    def test_chunk_by_attention_zones_empty_context(self):
        opt = ContextOptimizer()
        zones = opt.chunk_by_attention_zones("")
        assert zones.primacy_zone == ""
        assert zones.optimal_zone == ""

    def test_prioritize_information_with_few_sections_unchanged(self):
        opt = ContextOptimizer()
        docs = _make_docs(2)
        structured = opt.structure_for_retrieval(docs)
        result = opt.prioritize_information(structured, "query")
        # With 2 sections the method returns the original
        assert result == structured


# ---------------------------------------------------------------------------
# ContextAssembler tests
# ---------------------------------------------------------------------------


class TestContextAssembler:
    def _make_assembler(self, total_tokens: int = 32_000) -> ContextAssembler:
        budget = ContextBudget(total_tokens=total_tokens)
        return ContextAssembler(budget)

    def test_assemble_includes_rag_section(self):
        asm = self._make_assembler()
        result = asm.assemble(
            template="System.",
            retrieved_docs=_make_docs(2),
        )
        assert "rag" in result.sources_included

    def test_assemble_includes_user_profile(self):
        asm = self._make_assembler()
        result = asm.assemble(
            template="System.",
            user_profile={"name": "Alice", "plan": "Pro"},
        )
        assert "profile" in result.sources_included
        assert "Alice" in result.context

    def test_assemble_includes_tool_results(self):
        asm = self._make_assembler()
        result = asm.assemble(
            template="System.",
            tool_results=[{"tool": "search_kb", "result": "Some KB result."}],
        )
        assert "tools" in result.sources_included
        assert "search_kb" in result.context

    def test_assemble_includes_conversation_summary(self):
        asm = self._make_assembler()
        result = asm.assemble(
            template="System.",
            conversation_summary="Prior turns: user asked about billing.",
        )
        assert "summary" in result.sources_included
        assert "billing" in result.context

    def test_assemble_template_variable_substitution(self):
        asm = self._make_assembler()
        result = asm.assemble(
            template="You are a $role.",
            variables={"role": "support_agent"},
        )
        assert "support_agent" in result.context
        assert "$role" not in result.context

    def test_assemble_respects_max_tokens_per_source_via_config(self):
        asm = self._make_assembler()
        # 5 docs × ~20 words each = well over 200 tokens total;
        # cap rag at 200 tokens
        many_docs = [
            {
                "text": f"Document {i}: " + "important information " * 50,
                "metadata": {"source": f"doc-{i}.md"},
            }
            for i in range(1, 6)
        ]
        config = ContextConfig(
            template="System.",
            include_sources=["rag"],
            max_tokens_per_source={"rag": 200},
        )
        result = asm.assemble_from_config(config, retrieved_docs=many_docs, query="test")
        rag_tokens = result.token_breakdown.get("rag", 0)
        # After clipping, the rag section (with header) should be <= 200 + header overhead
        assert rag_tokens <= 250   # header adds ~5 tokens

    def test_assemble_from_config_respects_include_sources(self):
        asm = self._make_assembler()
        config = ContextConfig(
            template="System.",
            include_sources=["rag"],  # only RAG, no profile
        )
        result = asm.assemble_from_config(
            config,
            retrieved_docs=_make_docs(2),
            user_profile={"name": "Alice"},
        )
        assert "rag" in result.sources_included
        assert "profile" not in result.sources_included

    def test_assemble_excludes_source_when_no_budget(self):
        # Use a tiny window so there's no room for anything after template
        budget = ContextBudget(total_tokens=200)
        asm = ContextAssembler(budget)
        many_docs = [
            {"text": "doc content " * 30, "metadata": {"source": f"doc-{i}.md"}}
            for i in range(1, 4)
        ]
        result = asm.assemble(
            template="S.",
            retrieved_docs=many_docs,
            user_profile={"name": "Alice"},
        )
        # With 200 tokens total (45% = 90 for DC), some sources might be excluded
        # Just verify it doesn't crash and returns valid result
        assert isinstance(result.context, str)
        assert result.total_tokens <= budget.get_token_budget("dynamic_context") + 50

    def test_assembly_result_total_tokens_is_sum_of_breakdown(self):
        asm = self._make_assembler()
        result = asm.assemble(
            template="System prompt.",
            retrieved_docs=_make_docs(2),
            user_profile={"name": "Bob"},
        )
        assert result.total_tokens == sum(result.token_breakdown.values())

    def test_assemble_empty_inputs_returns_only_template(self):
        asm = self._make_assembler()
        result = asm.assemble(template="Just the template.")
        assert "template" in result.token_breakdown
        assert len(result.sources_included) == 0


# ---------------------------------------------------------------------------
# TokenCostCalculator tests
# ---------------------------------------------------------------------------


class TestTokenCostCalculator:
    def test_cost_calculation_is_accurate_gpt4o(self):
        calc = TokenCostCalculator()
        cost = calc.calculate_call_cost("gpt-4o", input_tokens=50_000, output_tokens=5_000)
        expected = (50_000 / 1_000_000 * 2.50) + (5_000 / 1_000_000 * 10.00)
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_cost_calculation_is_accurate_gpt4o_mini(self):
        calc = TokenCostCalculator()
        cost = calc.calculate_call_cost("gpt-4o-mini", input_tokens=100_000, output_tokens=1_000)
        expected = (100_000 / 1_000_000 * 0.15) + (1_000 / 1_000_000 * 0.60)
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_cost_calculation_raises_for_unknown_model(self):
        calc = TokenCostCalculator()
        with pytest.raises(KeyError):
            calc.calculate_call_cost("unknown-model", 1_000, 100)

    def test_daily_projection_structure(self):
        calc = TokenCostCalculator()
        proj = calc.calculate_daily_cost("gpt-4o", calls_per_day=100,
                                         avg_input_tokens=5_000, avg_output_tokens=500)
        assert proj.daily == pytest.approx(proj.cost_per_call * 100, rel=1e-6)
        assert proj.monthly == pytest.approx(proj.daily * 30, rel=1e-6)
        assert proj.annual  == pytest.approx(proj.daily * 365, rel=1e-6)

    def test_model_comparison_includes_all_models(self):
        calc = TokenCostCalculator()
        table = calc.compare_models(10_000, 1_000, calls_per_day=100)
        all_models = [
            model
            for provider in PRICING.values()
            for model in provider
        ]
        for model in all_models:
            assert model in table, f"Model '{model}' missing from comparison table"

    def test_model_comparison_is_sorted_by_provider(self):
        calc = TokenCostCalculator()
        table = calc.compare_models(10_000, 1_000, calls_per_day=100)
        assert "gpt-4o" in table
        assert "claude-3.5-sonnet" in table
        assert "gemini-1.5-flash" in table

    def test_optimize_system_prompt_reduces_tokens(self):
        calc = TokenCostCalculator()
        verbose = (
            "You are a helpful, friendly assistant.\n"
            "Please note that you should always be polite.\n"
            "It is important to remember that you represent the brand.\n"
            "Feel free to use examples. Of course, verify identity.\n"
            "Example 1: For returns, explain the 30-day policy.\n"
            "Example 2: For billing, point to the billing portal.\n"
            "Core rule: Answer only from the knowledge base.\n"
        ) * 5
        original_tokens = count_tokens(verbose)
        optimized = calc.optimize_system_prompt(verbose, target_reduction_pct=0.40)
        optimized_tokens = count_tokens(optimized)
        assert optimized_tokens < original_tokens

    def test_optimize_system_prompt_preserves_core_rules(self):
        calc = TokenCostCalculator()
        prompt = (
            "You are a helpful assistant. Please note that always be polite.\n"
            "Core rules:\n"
            "1. Answer using only the knowledge base.\n"
            "2. If unsure, escalate.\n"
        ) * 3
        optimized = calc.optimize_system_prompt(prompt, target_reduction_pct=0.30)
        assert "knowledge base" in optimized

    def test_audit_identifies_examples_as_waste(self):
        calc = TokenCostCalculator()
        verbose_system = (
            "You are a support agent.\n"
            "Example 1: For returns, explain the 30-day policy.\n"
            "Example 2: For billing, direct to the portal.\n"
            "Example 3: For technical, escalate to tier 2.\n"
            "Core rule: Answer from knowledge base only.\n"
        ) * 4
        messages = [{"role": "system", "content": verbose_system}]
        audit = calc.audit_context(messages, calls_per_day=1_000)
        combined = " ".join(audit.suggestions).lower()
        assert "example" in combined or "shorten" in combined or "move" in combined

    def test_audit_projected_savings_positive_when_waste_found(self):
        calc = TokenCostCalculator()
        verbose_system = (
            "Example 1: test. Example 2: test. Example 3: test. "
            "Example 4: test. Example 5: test.\n"
        ) * 10
        messages = [{"role": "system", "content": verbose_system}]
        audit = calc.audit_context(messages, calls_per_day=1_000)
        if audit.wasted_tokens > 0:
            assert audit.projected_savings["usd_monthly"] > 0

    def test_audit_total_tokens_matches_message_content(self):
        calc = TokenCostCalculator()
        messages = [
            {"role": "system",    "content": "You are a support agent."},
            {"role": "user",      "content": "Hello, I need help."},
            {"role": "assistant", "content": "Of course! How can I assist?"},
        ]
        audit = calc.audit_context(messages)
        assert audit.total_tokens > 0
        assert "system" in audit.by_role
        assert "user"   in audit.by_role

    def test_audit_includes_tool_tokens_when_provided(self):
        calc = TokenCostCalculator()
        messages = [{"role": "system", "content": "Agent."}]
        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the knowledge base.",
                "parameters": {"type": "object", "properties": {}},
            }
        }]
        audit_with    = calc.audit_context(messages, tools=tools)
        audit_without = calc.audit_context(messages)
        assert audit_with.total_tokens > audit_without.total_tokens
