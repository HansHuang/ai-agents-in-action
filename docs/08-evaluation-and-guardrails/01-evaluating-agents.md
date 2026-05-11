# Evaluating Agents

## What You'll Learn
- Why "it looks right" is not evaluation — and what to do instead
- The evaluation framework: retrieval metrics, generation metrics, and end-to-end metrics
- Building a test set: golden queries, expected outputs, and edge cases
- LLM-as-judge: using AI to evaluate AI (and when not to)
- Offline evaluation vs. online evaluation vs. A/B testing
- Continuous evaluation: catching regressions before users do

## Prerequisites
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — the system you're evaluating
- [RAG from Scratch](../03-memory-and-retrieval/03-rag-from-scratch.md) — retrieval evaluation
- [The Harness Mindset](../07-harness-engineering/01-the-harness-mindset.md) — evaluation is part of the harness

---

## The Evaluation Crisis

You've built an agent. You ask it a few questions. The answers look good. You ship it.

A week later, users report that it:
- Makes up refund policies that don't exist
- Forgets the conversation after 5 turns
- Recommends products you don't sell
- Costs 3x what you budgeted per query
- Works great in English but terribly in Spanish

What went wrong? **You didn't evaluate. You eyeballed.**

Evaluation is not optional. It's the difference between "it worked in my test" and "it works in production." This chapter gives you a three-level framework that catches retrieval failures, generation failures, and end-to-end failures — at every stage of development and in CI.

---

## The Three Levels of Evaluation

| Level | What It Measures | Example Question |
|:---|:---|:---|
| **Retrieval** | Are we finding the right documents? | "Did the retriever find the return policy document for this query?" |
| **Generation** | Is the answer good? | "Is this answer factually correct, complete, and well-formatted?" |
| **End-to-End** | Does the agent solve the user's problem? | "Did the user accomplish their goal without escalating to a human?" |

Each level catches different failures. You need all three.

---

## Level 1: Retrieval Evaluation

If your agent uses RAG, retrieval quality determines everything downstream. Garbage in, garbage out.

### Building a Retrieval Test Set

```python
@dataclass
class RetrievalTestCase:
    """A test case for retrieval evaluation."""
    query: str                        # The user's question
    relevant_doc_ids: list[str]       # Documents that SHOULD be retrieved
    partially_relevant_doc_ids: list[str] = None  # Nice to have
    irrelevant_doc_ids: list[str] = None          # Should NOT be retrieved
    min_results_expected: int = 1     # At least this many relevant docs should appear
```

Example test cases:

```python
retrieval_tests = [
    RetrievalTestCase(
        query="What's your return policy for damaged items?",
        relevant_doc_ids=["return-policy.md", "damaged-goods-policy.md"],
        irrelevant_doc_ids=["pricing.md", "careers.md"],
        min_results_expected=2,
    ),
    RetrievalTestCase(
        query="How do I reset my password?",
        relevant_doc_ids=["account-faq.md"],
        partially_relevant_doc_ids=["security-policy.md"],
        irrelevant_doc_ids=["shipping-info.md"],
        min_results_expected=1,
    ),
    RetrievalTestCase(
        query="Do you ship to Germany?",
        relevant_doc_ids=["international-shipping.md"],
        irrelevant_doc_ids=["domestic-shipping.md", "return-policy.md"],
        min_results_expected=1,
    ),
    RetrievalTestCase(
        query="Tell me about your company history",
        relevant_doc_ids=["about-us.md", "company-history.md"],
        irrelevant_doc_ids=["pricing.md", "api-docs.md"],
        min_results_expected=1,
    ),
    # Edge case: query with no relevant documents
    RetrievalTestCase(
        query="What's the capital of France?",
        relevant_doc_ids=[],  # Not in our knowledge base
        min_results_expected=0,
    ),
]
```

> **Precision edge case:** These formulas assume the retriever returns no duplicate document IDs. If duplicates are possible, deduplicate `retrieved_ids` before computing precision.

### Retrieval Metrics

```python
class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard information retrieval metrics.
    """
    
    def __init__(self, retriever, test_cases: list[RetrievalTestCase]):
        self.retriever = retriever
        self.test_cases = test_cases
    
    def evaluate(self, k: int = 5) -> RetrievalReport:
        """Evaluate retrieval across all test cases."""
        results = []
        
        for test in self.test_cases:
            retrieved = self.retriever.search(test.query, k=k)
            retrieved_ids = [doc["id"] for doc in retrieved]
            
            # Calculate per-query metrics
            metrics = self._calculate_metrics(test, retrieved_ids, k)
            results.append(metrics)
        
        # Aggregate
        return self._aggregate(results)
    
    def _calculate_metrics(self, test: RetrievalTestCase,
                          retrieved_ids: list[str], k: int) -> dict:
        """Calculate all retrieval metrics for a single query."""
        relevant = set(test.relevant_doc_ids)
        retrieved = set(retrieved_ids[:k])
        
        # Hit Rate: Did we find at least one relevant document?
        hit = 1 if relevant & retrieved else 0
        
        # Precision@K: What fraction of retrieved documents are relevant?
        precision = len(relevant & retrieved) / len(retrieved) if retrieved else 0
        
        # Recall@K: What fraction of all relevant documents were retrieved?
        recall = len(relevant & retrieved) / len(relevant) if relevant else 1.0
        
        # MRR: Mean Reciprocal Rank
        # How high does the first relevant document rank?
        reciprocal_rank = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant:
                reciprocal_rank = 1 / (i + 1)
                break
        
        # NDCG@K: Normalized Discounted Cumulative Gain
        # Accounts for position and graded relevance
        ndcg = self._calculate_ndcg(test, retrieved_ids, k)
        
        return {
            "query": test.query,
            "hit": hit,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_k": ndcg,
            "relevant_found": len(relevant & retrieved),
            "relevant_total": len(relevant),
            "retrieved_ids": retrieved_ids,
        }
    
    def _calculate_ndcg(self, test: RetrievalTestCase,
                       retrieved_ids: list[str], k: int) -> float:
        """
        Calculate NDCG with graded relevance:
        - relevant: score 2
        - partially_relevant: score 1
        - irrelevant: score 0
        """
        relevance_scores = {}
        for doc_id in test.relevant_doc_ids:
            relevance_scores[doc_id] = 2
        if test.partially_relevant_doc_ids:
            for doc_id in test.partially_relevant_doc_ids:
                relevance_scores[doc_id] = 1
        
        # DCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = relevance_scores.get(doc_id, 0)
            # Rank is 1-indexed: rank = i+1. Standard DCG uses log2(rank+1) = log2(i+2).
            dcg += relevance / math.log2(i + 2)
        
        # IDCG (ideal DCG — best possible ranking)
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def _aggregate(self, results: list[dict]) -> RetrievalReport:
        """Aggregate per-query metrics into a report."""
        n = len(results)
        
        return RetrievalReport(
            hit_rate=sum(r["hit"] for r in results) / n,
            precision_at_k=sum(r["precision_at_k"] for r in results) / n,
            recall_at_k=sum(r["recall_at_k"] for r in results) / n,
            mrr=sum(r["reciprocal_rank"] for r in results) / n,
            ndcg_at_k=sum(r["ndcg_at_k"] for r in results) / n,
            total_queries=n,
            queries_with_zero_results=sum(1 for r in results if r["relevant_found"] == 0 and r["relevant_total"] > 0),
            per_query=results,
        )

@dataclass
class RetrievalReport:
    hit_rate: float           # Should be > 0.90
    precision_at_k: float     # Should be > 0.70
    recall_at_k: float        # Should be > 0.80
    mrr: float                # Should be > 0.60
    ndcg_at_k: float          # Should be > 0.70
    total_queries: int
    queries_with_zero_results: int
    per_query: list[dict]
    
    def to_string(self) -> str:
        return f"""
RETRIEVAL EVALUATION REPORT
============================
Total Queries: {self.total_queries}

Hit Rate:        {self.hit_rate:.2%}  (target: > 90%)
Precision@5:     {self.precision_at_k:.2%}  (target: > 70%)
Recall@5:        {self.recall_at_k:.2%}  (target: > 80%)
MRR:             {self.mrr:.2%}  (target: > 60%)
NDCG@5:          {self.ndcg_at_k:.2%}  (target: > 70%)

Queries with zero relevant results: {self.queries_with_zero_results}
"""
```

---

## Level 2: Generation Evaluation

Retrieval finds documents. Generation produces answers. Both need evaluation.

### What to Measure in Generated Answers

| Dimension | What It Means | Example Failure |
|:---|:---|:---|
| **Faithfulness** | Is the answer grounded in the provided documents? | Answer mentions a policy that doesn't exist in any document |
| **Relevance** | Does the answer address the user's question? | User asks about returns, answer talks about shipping |
| **Completeness** | Does the answer cover all parts of the question? | User asks "compare A and B," answer only covers A |
| **Accuracy** | Are the facts correct? | Answer says "30-day return" when policy says "14-day" |
| **Coherence** | Is the answer well-written and logical? | Answer is a jumble of unrelated sentences |
| **Safety** | Does the answer avoid harmful content? | Answer provides dangerous advice |

### Building a Generation Test Set

```python
@dataclass
class GenerationTestCase:
    """A test case for generation evaluation."""
    query: str
    expected_answer_contains: list[str] = None    # Key facts that MUST appear
    expected_answer_not_contains: list[str] = None # Facts that MUST NOT appear
    expected_sources: list[str] = None             # Sources that should be cited
    min_answer_length: int = 20
    max_answer_length: int = 2000
    
    # For LLM-as-judge evaluation
    evaluation_criteria: str = None  # Custom criteria for this specific query
    reference_answer: str = None     # Golden answer for comparison

generation_tests = [
    GenerationTestCase(
        query="What's your return policy?",
        expected_answer_contains=[
            "30 days",
            "original packaging",
            "receipt",
        ],
        expected_answer_not_contains=[
            "60 days",  # Old policy
            "no questions asked",  # Not our policy
        ],
        expected_sources=["return-policy.md"],
    ),
    GenerationTestCase(
        query="How much does shipping cost?",
        expected_answer_contains=[
            "free shipping",
            "$50",
        ],
        expected_answer_not_contains=[
            "international shipping",  # Different policy
        ],
        expected_sources=["shipping-info.md"],
    ),
    GenerationTestCase(
        query="Compare the Pro and Enterprise plans",
        expected_answer_contains=[
            "Pro",
            "Enterprise",
            "$",  # Should include pricing
        ],
        min_answer_length=100,  # Comparison should be substantial
    ),
]
```

**Two complementary approaches:** Rule-based checks are fast, zero-cost, and deterministic — ideal for facts that can be verified literally (required phrases, forbidden phrases, length bounds, source citations). LLM judges are slower and cost tokens but catch nuances rules cannot: hallucinated facts that sound plausible, answers that technically contain all required phrases but are incoherent, or responses that address a different reading of the question. Run rule checks first; apply judge checks on queries that pass rules, or sample 10–20% for cost control.

### LLM-as-Judge Evaluation

For nuanced evaluation, use an LLM to judge the quality of another LLM's output:

> **Before using LLM-as-judge:** Judges inherit the biases of the model they run on — they prefer longer answers, can be fooled by confident-sounding errors, and add API cost per query. Always pair judge checks with at least one rule-based check, and validate judge outputs with manual review on a 5–10% sample before trusting aggregate scores.

```python
class LLMJudge:
    """
    Use an LLM to evaluate the quality of agent responses.
    This is the most scalable approach but has its own biases.
    """
    
    FAITHFULNESS_PROMPT = """You are evaluating whether an AI response is faithful 
to the provided source documents.

Faithfulness means: Every factual claim in the response is directly supported by 
at least one of the source documents. The response does not add information not 
found in the sources.

SOURCE DOCUMENTS:
{source_documents}

RESPONSE TO EVALUATE:
{response}

Evaluate the response for faithfulness. Think step by step before writing your JSON.

Output JSON:
{{
    "is_faithful": true/false,
    "score": 1-5,
    "unsupported_claims": ["claim1", "claim2"],
    "explanation": "Brief explanation of your evaluation"
}}
"""

    RELEVANCE_PROMPT = """You are evaluating whether an AI response is relevant 
to the user's question.

Relevance means: The response directly addresses what the user asked. It does not 
go off-topic or provide unnecessary information.

USER QUESTION:
{user_question}

RESPONSE TO EVALUATE:
{response}

Evaluate the response for relevance. Think step by step before writing your JSON.

Output JSON:
{{
    "is_relevant": true/false,
    "score": 1-5,
    "off_topic_parts": ["part1", "part2"],
    "explanation": "Brief explanation"
}}
"""

    COMPLETENESS_PROMPT = """You are evaluating whether an AI response completely 
answers the user's question.

Completeness means: The response addresses ALL parts of the user's question. 
If the user asked multiple questions, all are answered. If the user asked for 
a comparison, both sides are covered.

USER QUESTION:
{user_question}

RESPONSE TO EVALUATE:
{response}

Evaluate the response for completeness. Think step by step before writing your JSON.

Output JSON:
{{
    "is_complete": true/false,
    "score": 1-5,
    "missing_parts": ["unanswered question 1", "unanswered question 2"],
    "explanation": "Brief explanation"
}}
"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
    
    async def evaluate_faithfulness(self, response: str,
                                   source_documents: list[str]) -> JudgeResult:
        """Evaluate if response is faithful to source documents."""
        sources_text = "\n\n---\n\n".join(
            f"[Document {i+1}]\n{doc}" 
            for i, doc in enumerate(source_documents)
        )
        
        prompt = self.FAITHFULNESS_PROMPT.format(
            source_documents=sources_text[:10000],  # Capped at 10K chars; for very long docs, chunk at semantic boundaries rather than truncating blindly
            response=response[:5000],
        )
        
        result = await self._call_judge(prompt)
        
        return JudgeResult(
            dimension="faithfulness",
            passed=result["is_faithful"],
            score=result["score"],
            issues=result.get("unsupported_claims", []),
            explanation=result.get("explanation", ""),
        )
    
    async def evaluate_relevance(self, response: str,
                                user_question: str) -> JudgeResult:
        """Evaluate if response is relevant to the question."""
        prompt = self.RELEVANCE_PROMPT.format(
            user_question=user_question,
            response=response[:5000],
        )
        
        result = await self._call_judge(prompt)
        
        return JudgeResult(
            dimension="relevance",
            passed=result["is_relevant"],
            score=result["score"],
            issues=result.get("off_topic_parts", []),
            explanation=result.get("explanation", ""),
        )
    
    async def evaluate_completeness(self, response: str,
                                   user_question: str) -> JudgeResult:
        """Evaluate if response completely answers the question."""
        prompt = self.COMPLETENESS_PROMPT.format(
            user_question=user_question,
            response=response[:5000],
        )
        
        result = await self._call_judge(prompt)
        
        return JudgeResult(
            dimension="completeness",
            passed=result["is_complete"],
            score=result["score"],
            issues=result.get("missing_parts", []),
            explanation=result.get("explanation", ""),
        )
    
    async def _call_judge(self, prompt: str) -> dict:
        """Call the judge LLM."""
        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent judging
        )
        return json.loads(response.content)

class GenerationEvaluator:
    """
    Evaluate generation quality using both rule-based checks and LLM-as-judge.
    """
    
    def __init__(self, agent, test_cases: list[GenerationTestCase],
                judge: LLMJudge = None):
        self.agent = agent
        self.test_cases = test_cases
        self.judge = judge or LLMJudge()
    
    async def evaluate(self) -> GenerationReport:
        """Evaluate generation across all test cases."""
        results = []
        
        for test in self.test_cases:
            response = await self.agent.run(test.query)
            
            # Rule-based checks
            rule_checks = self._rule_based_checks(test, response.content)
            
            # LLM-as-judge checks
            judge_checks = {}
            if self.judge:
                judge_checks["faithfulness"] = await self.judge.evaluate_faithfulness(
                    response.content,
                    response.metadata.get("retrieved_documents", [])
                )
                judge_checks["relevance"] = await self.judge.evaluate_relevance(
                    response.content, test.query
                )
                judge_checks["completeness"] = await self.judge.evaluate_completeness(
                    response.content, test.query
                )
            
            results.append({
                "query": test.query,
                "response": response.content,
                "rule_checks": rule_checks,
                "judge_checks": judge_checks,
                "overall_pass": rule_checks.get("all_passed", False) and 
                               all(j.passed for j in judge_checks.values()),
            })
        
        return self._aggregate(results)
    
    def _rule_based_checks(self, test: GenerationTestCase, 
                          response: str) -> dict:
        """Perform deterministic checks on the response."""
        checks = {}
        
        # Check for required content
        if test.expected_answer_contains:
            checks["contains_required"] = all(
                phrase.lower() in response.lower()
                for phrase in test.expected_answer_contains
            )
            if not checks["contains_required"]:
                missing = [p for p in test.expected_answer_contains 
                          if p.lower() not in response.lower()]
                checks["missing_required"] = missing
        
        # Check for forbidden content
        if test.expected_answer_not_contains:
            checks["avoids_forbidden"] = all(
                phrase.lower() not in response.lower()
                for phrase in test.expected_answer_not_contains
            )
            if not checks["avoids_forbidden"]:
                found = [p for p in test.expected_answer_not_contains
                        if p.lower() in response.lower()]
                checks["found_forbidden"] = found
        
        # Check length
        checks["length_ok"] = (
            len(response) >= test.min_answer_length and
            len(response) <= test.max_answer_length
        )
        
        # Check sources cited
        if test.expected_sources:
            checks["sources_cited"] = all(
                source.lower() in response.lower()
                for source in test.expected_sources
            )
        
        checks["all_passed"] = all(checks.values())
        return checks
    
    def _aggregate(self, results: list[dict]) -> GenerationReport:
        """Aggregate per-query results into a report."""
        n = len(results)
        
        return GenerationReport(
            overall_pass_rate=sum(1 for r in results if r["overall_pass"]) / n,
            contains_required_rate=sum(1 for r in results 
                                      if r["rule_checks"].get("contains_required", True)) / n,
            avoids_forbidden_rate=sum(1 for r in results 
                                     if r["rule_checks"].get("avoids_forbidden", True)) / n,
            faithfulness_pass_rate=sum(1 for r in results 
                                      if r["judge_checks"].get("faithfulness", JudgeResult(dimension="", passed=True, score=5, issues=[], explanation="")).passed) / n if "faithfulness" in results[0].get("judge_checks", {}) else None,
            relevance_pass_rate=sum(1 for r in results 
                                   if r["judge_checks"].get("relevance", JudgeResult(dimension="", passed=True, score=5, issues=[], explanation="")).passed) / n if "relevance" in results[0].get("judge_checks", {}) else None,
            total_queries=n,
            per_query=results,
        )
```

---

## Level 3: End-to-End Evaluation

Does the agent actually solve user problems?

### Task Success Rate

```python
@dataclass
class EndToEndTestCase:
    """A test case for end-to-end evaluation."""
    scenario: str                          # Description of the scenario
    user_messages: list[str]               # Multi-turn conversation
    expected_outcome: str                  # "resolved", "escalated", "information_provided"
    expected_tools_called: list[str] = None # Tools that should be used
    max_turns_expected: int = 5            # Should resolve within this many turns
    forbidden_behaviors: list[str] = None   # Things the agent should NOT do

class EndToEndEvaluator:
    """
    Evaluate the agent on realistic multi-turn scenarios.
    """
    
    def __init__(self, agent, test_cases: list[EndToEndTestCase]):
        self.agent = agent
        self.test_cases = test_cases
    
    async def evaluate(self) -> EndToEndReport:
        """Run all end-to-end scenarios."""
        results = []
        
        for test in self.test_cases:
            conversation = []
            tools_called = []
            outcome = "unknown"
            
            for i, user_message in enumerate(test.user_messages):
                response = await self.agent.run(
                    user_message,
                    conversation_history=conversation,
                )
                
                conversation.append({"role": "user", "content": user_message})
                conversation.append({"role": "assistant", "content": response.content})
                
                if response.tool_calls:
                    tools_called.extend([tc.name for tc in response.tool_calls])
                
                # Check if conversation reached a resolution
                if self._is_resolved(response, test):
                    outcome = "resolved"
                    break
            
            if outcome == "unknown":
                outcome = "unresolved" if len(test.user_messages) >= test.max_turns_expected else "incomplete"
            
            results.append({
                "scenario": test.scenario,
                "outcome": outcome,
                "expected_outcome": test.expected_outcome,
                "turns_taken": i + 1,
                "tools_called": tools_called,
                "expected_tools": test.expected_tools_called,
                "success": outcome == test.expected_outcome,
            })
        
        return EndToEndReport(
            task_success_rate=sum(1 for r in results if r["success"]) / len(results),
            avg_turns_to_resolution=sum(r["turns_taken"] for r in results) / len(results),
            total_scenarios=len(results),
            per_scenario=results,
        )
    
    def _is_resolved(self, response, test: EndToEndTestCase) -> bool:
        """Check if the agent's response indicates resolution."""
        resolution_markers = [
            "Is there anything else",
            "I hope that helps",
            "Your request has been",
            "I've completed",
            "Would you like me to",
        ]
        return any(marker in response.content for marker in resolution_markers)
```

---

## Continuous Evaluation Pipeline

Evaluation isn't a one-time event. It's a pipeline:

```python
class ContinuousEvaluationPipeline:
    """
    Run evaluation on every change and alert on regressions.

    The ``harness`` parameter ensures evaluation runs through the same routing,
    retry, and guardrail logic as production — measuring the system users
    actually hit, not a stripped-down test version.
    """
    
    def __init__(self, harness: ProductionHarness,
                retrieval_evaluator: RetrievalEvaluator,
                generation_evaluator: GenerationEvaluator,
                end_to_end_evaluator: EndToEndEvaluator):
        self.harness = harness
        self.retrieval_evaluator = retrieval_evaluator
        self.generation_evaluator = generation_evaluator
        self.end_to_end_evaluator = end_to_end_evaluator
        self.baseline = None  # Stored baseline metrics
    
    async def set_baseline(self):
        """Set the current performance as the baseline."""
        self.baseline = await self.run_all()
        logger.info(f"Baseline set: {self.baseline.summary()}")
    
    async def run_all(self) -> FullEvaluationReport:
        """Run all three evaluation levels."""
        retrieval = await self.retrieval_evaluator.evaluate()
        generation = await self.generation_evaluator.evaluate()
        end_to_end = await self.end_to_end_evaluator.evaluate()
        
        return FullEvaluationReport(
            retrieval=retrieval,
            generation=generation,
            end_to_end=end_to_end,
        )
    
    async def check_regression(self) -> RegressionCheck:
        """
        Compare current performance against baseline.
        Alert if any metric has regressed significantly.
        """
        if not self.baseline:
            raise ValueError("No baseline set. Run set_baseline() first.")
        
        current = await self.run_all()
        regressions = []
        
        # Check retrieval regressions
        if current.retrieval.hit_rate < self.baseline.retrieval.hit_rate - 0.05:
            regressions.append(f"Hit rate dropped from {self.baseline.retrieval.hit_rate:.2%} to {current.retrieval.hit_rate:.2%}")
        
        if current.retrieval.mrr < self.baseline.retrieval.mrr - 0.05:
            regressions.append(f"MRR dropped from {self.baseline.retrieval.mrr:.2%} to {current.retrieval.mrr:.2%}")
        
        # Check generation regressions
        if current.generation.overall_pass_rate < self.baseline.generation.overall_pass_rate - 0.05:
            regressions.append(f"Generation pass rate dropped from {self.baseline.generation.overall_pass_rate:.2%} to {current.generation.overall_pass_rate:.2%}")
        
        # Check end-to-end regressions
        if current.end_to_end.task_success_rate < self.baseline.end_to_end.task_success_rate - 0.05:
            regressions.append(f"Task success rate dropped from {self.baseline.end_to_end.task_success_rate:.2%} to {current.end_to_end.task_success_rate:.2%}")
        
        return RegressionCheck(
            has_regressions=len(regressions) > 0,
            regressions=regressions,
            baseline=self.baseline,
            current=current,
        )
```

> **Tip — Persisting the Baseline in CI:** `self.baseline` is in-memory and lost when the process exits. Serialize the `FullEvaluationReport` to JSON and commit it to the repository (or store it as a CI artifact). On subsequent runs, load it before calling `check_regression()` to measure regressions against a known-good commit.

> **Tip — CI Pipeline Duration:** Running LLM-judge evaluation on every query for every PR is slow and expensive. In CI, run retrieval evaluation on the full test set (it's fast) and LLM-judge evaluation on a sampled 20–30 queries. Reserve the full generation evaluation for nightly runs or release gates.

> **Regression Threshold:** The 5% threshold is a starting point. For small test sets (< 20 queries), results are noisy — consider 8–10%. For safety-critical metrics such as faithfulness pass rate, consider 0%: any regression fails the build. Adjust thresholds per metric rather than globally.

---

## Common Pitfalls

- **"I evaluate with 5 hand-picked queries and call it done"**: Five queries catch obvious failures. They don't catch the edge case that affects 2% of users. Build a test set of at least 50 queries covering all intent categories.
- **"I only evaluate retrieval, not generation"**: Retrieval accuracy is necessary but not sufficient. A perfect retriever with a hallucinating generator produces garbage. Evaluate both.
- **"I use LLM-as-judge for everything"**: LLM judges have biases. They prefer longer answers. They can be fooled by confident-sounding falsehoods. Validate judge results with human review on a sample.
- **"I set a baseline and never re-evaluate"**: Models change. Prompts change. Documents change. Re-evaluate on every significant change. Run the evaluation pipeline in CI/CD.
- **"I only evaluate in English"**: If your users speak multiple languages, evaluate in all of them. Performance varies dramatically across languages.
- **"I don't evaluate cost and latency"**: An agent that's 100% accurate but costs $5 per query and takes 30 seconds is not production-ready. Cost and latency are evaluation metrics too. Track: input/output token count per query, P50 and P95 latency, and cost per query (tokens × model price). Set targets before launch — for example, P95 latency < 3 seconds, cost < $0.02/query — and include them in the regression check.

## What's Next

You can now measure whether your agent works. Next: ensuring it works safely — guardrails, content policies, and the safety evaluation framework.
→ [Guardrails and Safety](02-guardrails-and-safety.md)
```

