// agent_evaluator_test.go
// =======================
// Table-driven tests for the Go evaluation framework.
package main

import (
	"context"
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type staticRetriever struct {
	results []string
}

func (r *staticRetriever) Search(_ context.Context, _ string, k int) ([]map[string]string, error) {
	out := r.results
	if k < len(out) {
		out = out[:k]
	}
	result := make([]map[string]string, len(out))
	for i, id := range out {
		result[i] = map[string]string{"id": id}
	}
	return result, nil
}

type staticAgent struct {
	content string
}

func (a *staticAgent) Run(_ context.Context, _ string, _ []map[string]string) (AgentResponse, error) {
	return AgentResponse{
		Content:            a.content,
		RetrievedDocuments: []string{"doc excerpt"},
	}, nil
}

func newEval(retriever Retriever, tc []RetrievalTestCase) *RetrievalEvaluator {
	return &RetrievalEvaluator{Retriever: retriever, TestCases: tc}
}

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// ---------------------------------------------------------------------------
// Retrieval evaluator tests
// ---------------------------------------------------------------------------

func TestHitRatePerfect(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a"}, MinResultsExpected: 1}}
	eval := newEval(&staticRetriever{results: []string{"a"}}, tc)
	report, err := eval.Evaluate(context.Background(), 5)
	if err != nil {
		t.Fatal(err)
	}
	if report.HitRate != 1.0 {
		t.Errorf("expected hit_rate 1.0, got %.2f", report.HitRate)
	}
}

func TestHitRateZero(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a"}, MinResultsExpected: 1}}
	eval := newEval(&staticRetriever{results: []string{"b", "c"}}, tc)
	report, err := eval.Evaluate(context.Background(), 5)
	if err != nil {
		t.Fatal(err)
	}
	if report.HitRate != 0.0 {
		t.Errorf("expected hit_rate 0.0, got %.2f", report.HitRate)
	}
}

func TestPrecisionPerfect(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a", "b"}}}
	eval := newEval(&staticRetriever{results: []string{"a", "b"}}, tc)
	report, _ := eval.Evaluate(context.Background(), 2)
	if !approxEqual(report.PrecisionAtK, 1.0, 0.01) {
		t.Errorf("expected precision 1.0, got %.2f", report.PrecisionAtK)
	}
}

func TestPrecisionMixed(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a"}}}
	eval := newEval(&staticRetriever{results: []string{"a", "b"}}, tc)
	report, _ := eval.Evaluate(context.Background(), 2)
	if !approxEqual(report.PrecisionAtK, 0.5, 0.01) {
		t.Errorf("expected precision 0.5, got %.2f", report.PrecisionAtK)
	}
}

func TestRecallPerfect(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a", "b"}}}
	eval := newEval(&staticRetriever{results: []string{"a", "b", "c"}}, tc)
	report, _ := eval.Evaluate(context.Background(), 5)
	if !approxEqual(report.RecallAtK, 1.0, 0.01) {
		t.Errorf("expected recall 1.0, got %.2f", report.RecallAtK)
	}
}

func TestMRRFirstPosition(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a"}}}
	eval := newEval(&staticRetriever{results: []string{"a", "b", "c"}}, tc)
	report, _ := eval.Evaluate(context.Background(), 5)
	if !approxEqual(report.MRR, 1.0, 0.01) {
		t.Errorf("expected MRR 1.0, got %.2f", report.MRR)
	}
}

func TestMRRThirdPosition(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"c"}}}
	eval := newEval(&staticRetriever{results: []string{"a", "b", "c"}}, tc)
	report, _ := eval.Evaluate(context.Background(), 5)
	expected := 1.0 / 3.0
	if !approxEqual(report.MRR, expected, 0.01) {
		t.Errorf("expected MRR %.3f, got %.3f", expected, report.MRR)
	}
}

func TestMRRNoMatch(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"x"}}}
	eval := newEval(&staticRetriever{results: []string{"a", "b", "c"}}, tc)
	report, _ := eval.Evaluate(context.Background(), 5)
	if report.MRR != 0.0 {
		t.Errorf("expected MRR 0.0, got %.2f", report.MRR)
	}
}

func TestNDCGPerfectRanking(t *testing.T) {
	tc := RetrievalTestCase{Query: "q", RelevantDocIDs: []string{"a", "b"}}
	eval := &RetrievalEvaluator{}
	ndcg := eval.calculateNDCG(tc, []string{"a", "b"}, 5)
	if !approxEqual(ndcg, 1.0, 0.01) {
		t.Errorf("expected NDCG 1.0 for perfect ranking, got %.4f", ndcg)
	}
}

func TestNDCGGradedRelevance(t *testing.T) {
	tc := RetrievalTestCase{
		Query:                   "q",
		RelevantDocIDs:          []string{"a"},
		PartiallyRelevantDocIDs: []string{"b"},
	}
	eval := &RetrievalEvaluator{}
	// Retrieved in order: b(1), a(2) — sub-optimal ordering
	ndcg := eval.calculateNDCG(tc, []string{"b", "a"}, 5)
	// Should be < 1 because a (score 2) is ranked below b (score 1)
	if ndcg >= 1.0 {
		t.Errorf("expected NDCG < 1.0 for suboptimal ranking, got %.4f", ndcg)
	}
	if ndcg <= 0.0 {
		t.Errorf("expected NDCG > 0.0, got %.4f", ndcg)
	}
}

func TestEmptyRelevantDocsRecallIsOne(t *testing.T) {
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{}, MinResultsExpected: 0}}
	eval := newEval(&staticRetriever{results: []string{"a"}}, tc)
	report, _ := eval.Evaluate(context.Background(), 5)
	if !approxEqual(report.RecallAtK, 1.0, 0.01) {
		t.Errorf("expected recall 1.0 for empty relevant set, got %.2f", report.RecallAtK)
	}
}

func TestAggregateAverages(t *testing.T) {
	tc := []RetrievalTestCase{
		{Query: "q1", RelevantDocIDs: []string{"a"}},
		{Query: "q2", RelevantDocIDs: []string{"a"}},
	}
	// q1: hits, q2: misses
	corpus := map[string][]string{
		"q1": {"a"},
		"q2": {"b"},
	}
	type mapRetriever struct{ corpus map[string][]string }
	eval := &RetrievalEvaluator{
		Retriever: &demoRetriever{corpus: corpus},
		TestCases: tc,
	}
	report, _ := eval.Evaluate(context.Background(), 5)
	if !approxEqual(report.HitRate, 0.5, 0.01) {
		t.Errorf("expected aggregated hit_rate 0.5, got %.2f", report.HitRate)
	}
}

// ---------------------------------------------------------------------------
// Generation evaluator tests
// ---------------------------------------------------------------------------

func newGenEval(response string, tcs []GenerationTestCase) *GenerationEvaluator {
	return &GenerationEvaluator{Agent: &staticAgent{content: response}, TestCases: tcs}
}

func TestRuleBasedContainsAll(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", ExpectedAnswerContains: []string{"hello", "world"}}
	rc := eval.ruleBasedChecks(tc, "hello world answer")
	if !rc.ContainsRequired {
		t.Error("expected ContainsRequired=true")
	}
}

func TestRuleBasedMissingRequired(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", ExpectedAnswerContains: []string{"hello", "missing"}}
	rc := eval.ruleBasedChecks(tc, "hello world answer")
	if rc.ContainsRequired {
		t.Error("expected ContainsRequired=false")
	}
	if len(rc.MissingRequired) == 0 || rc.MissingRequired[0] != "missing" {
		t.Errorf("expected MissingRequired=[missing], got %v", rc.MissingRequired)
	}
}

func TestRuleBasedContainsForbidden(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", ExpectedAnswerNotContains: []string{"bad"}}
	rc := eval.ruleBasedChecks(tc, "this is bad content")
	if rc.AvoidsForbidden {
		t.Error("expected AvoidsForbidden=false")
	}
}

func TestRuleBasedAvoidsAllForbidden(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", ExpectedAnswerNotContains: []string{"bad", "evil"}}
	rc := eval.ruleBasedChecks(tc, "this is a good safe answer")
	if !rc.AvoidsForbidden {
		t.Error("expected AvoidsForbidden=true")
	}
}

func TestRuleBasedLengthOK(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", MinAnswerLength: 5, MaxAnswerLength: 100}
	rc := eval.ruleBasedChecks(tc, "hello world")
	if !rc.LengthOK {
		t.Error("expected LengthOK=true")
	}
}

func TestRuleBasedTooShort(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", MinAnswerLength: 50, MaxAnswerLength: 200}
	rc := eval.ruleBasedChecks(tc, "short")
	if rc.LengthOK {
		t.Error("expected LengthOK=false for too-short response")
	}
}

func TestRuleBasedTooLong(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", MinAnswerLength: 1, MaxAnswerLength: 5}
	rc := eval.ruleBasedChecks(tc, "this is a very long response that exceeds the maximum")
	if rc.LengthOK {
		t.Error("expected LengthOK=false for too-long response")
	}
}

func TestRuleBasedSourcesCited(t *testing.T) {
	eval := newGenEval("", nil)
	tc := GenerationTestCase{Query: "q", ExpectedSources: []string{"source.md"}}
	rc := eval.ruleBasedChecks(tc, "The answer is in source.md")
	if !rc.SourcesCited {
		t.Error("expected SourcesCited=true")
	}
}

// ---------------------------------------------------------------------------
// End-to-End evaluator tests
// ---------------------------------------------------------------------------

func TestScenarioResolved(t *testing.T) {
	agent := &staticAgent{content: "Is there anything else I can help you with?"}
	e2e := &EndToEndEvaluator{
		Agent: agent,
		TestCases: []EndToEndTestCase{
			{Scenario: "s", UserMessages: []string{"hi"}, ExpectedOutcome: "resolved", MaxTurnsExpected: 3},
		},
	}
	report, err := e2e.Evaluate(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report.TaskSuccessRate != 1.0 {
		t.Errorf("expected success rate 1.0, got %.2f", report.TaskSuccessRate)
	}
}

func TestScenarioNotResolved(t *testing.T) {
	agent := &staticAgent{content: "I don't know."}
	e2e := &EndToEndEvaluator{
		Agent: agent,
		TestCases: []EndToEndTestCase{
			{Scenario: "s", UserMessages: []string{"hi"}, ExpectedOutcome: "resolved", MaxTurnsExpected: 1},
		},
	}
	report, _ := e2e.Evaluate(context.Background())
	if report.TaskSuccessRate != 0.0 {
		t.Errorf("expected success rate 0.0, got %.2f", report.TaskSuccessRate)
	}
}

func TestResolutionDetection(t *testing.T) {
	markers := []string{
		"Is there anything else I can help you with?",
		"I hope that helps.",
		"Your request has been processed.",
		"I've completed the task.",
		"Would you like me to do anything else?",
	}
	for _, content := range markers {
		if !isResolved(content) {
			t.Errorf("expected isResolved=true for %q", content)
		}
	}
}

// ---------------------------------------------------------------------------
// Continuous evaluation pipeline tests
// ---------------------------------------------------------------------------

func TestNoRegression(t *testing.T) {
	ctx := context.Background()
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a"}}}
	retriever := &staticRetriever{results: []string{"a"}}
	agent := &staticAgent{content: "Is there anything else I can help you with?"}

	pipeline := &ContinuousEvaluationPipeline{
		RetrievalEvaluator:  &RetrievalEvaluator{Retriever: retriever, TestCases: tc},
		GenerationEvaluator: &GenerationEvaluator{Agent: agent, TestCases: []GenerationTestCase{{Query: "q"}}},
		EndToEndEvaluator:   &EndToEndEvaluator{Agent: agent, TestCases: []EndToEndTestCase{{Scenario: "s", UserMessages: []string{"hi"}, ExpectedOutcome: "resolved"}}},
	}

	if err := pipeline.SetBaseline(ctx); err != nil {
		t.Fatal(err)
	}
	check, err := pipeline.CheckRegression(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if check.HasRegressions {
		t.Errorf("expected no regressions, got: %v", check.Regressions)
	}
}

func TestRegressionDetected(t *testing.T) {
	ctx := context.Background()
	tc := []RetrievalTestCase{
		{Query: "q1", RelevantDocIDs: []string{"a"}},
		{Query: "q2", RelevantDocIDs: []string{"a"}},
		{Query: "q3", RelevantDocIDs: []string{"a"}},
		{Query: "q4", RelevantDocIDs: []string{"a"}},
		{Query: "q5", RelevantDocIDs: []string{"a"}},
		{Query: "q6", RelevantDocIDs: []string{"a"}},
		{Query: "q7", RelevantDocIDs: []string{"a"}},
		{Query: "q8", RelevantDocIDs: []string{"a"}},
		{Query: "q9", RelevantDocIDs: []string{"a"}},
		{Query: "q10", RelevantDocIDs: []string{"a"}},
	}
	goodRetriever := &staticRetriever{results: []string{"a"}}
	badRetriever := &staticRetriever{results: []string{"b"}}
	agent := &staticAgent{content: "Is there anything else I can help you with?"}
	genTCs := []GenerationTestCase{{Query: "q"}}
	e2eTCs := []EndToEndTestCase{{Scenario: "s", UserMessages: []string{"hi"}, ExpectedOutcome: "resolved"}}

	pipeline := &ContinuousEvaluationPipeline{
		RetrievalEvaluator:  &RetrievalEvaluator{Retriever: goodRetriever, TestCases: tc},
		GenerationEvaluator: &GenerationEvaluator{Agent: agent, TestCases: genTCs},
		EndToEndEvaluator:   &EndToEndEvaluator{Agent: agent, TestCases: e2eTCs},
	}
	if err := pipeline.SetBaseline(ctx); err != nil {
		t.Fatal(err)
	}

	pipeline.RetrievalEvaluator = &RetrievalEvaluator{Retriever: badRetriever, TestCases: tc}
	check, err := pipeline.CheckRegression(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !check.HasRegressions {
		t.Error("expected regression to be detected")
	}
}

func TestRegressionThresholdNotExceeded(t *testing.T) {
	ctx := context.Background()
	// Use a test case where baseline and current are identical (no drop)
	tc := []RetrievalTestCase{{Query: "q", RelevantDocIDs: []string{"a"}}}
	retriever := &staticRetriever{results: []string{"a"}}
	agent := &staticAgent{content: "Is there anything else I can help you with?"}
	genTCs := []GenerationTestCase{{Query: "q"}}
	e2eTCs := []EndToEndTestCase{{Scenario: "s", UserMessages: []string{"hi"}, ExpectedOutcome: "resolved"}}

	pipeline := &ContinuousEvaluationPipeline{
		RetrievalEvaluator:  &RetrievalEvaluator{Retriever: retriever, TestCases: tc},
		GenerationEvaluator: &GenerationEvaluator{Agent: agent, TestCases: genTCs},
		EndToEndEvaluator:   &EndToEndEvaluator{Agent: agent, TestCases: e2eTCs},
	}
	if err := pipeline.SetBaseline(ctx); err != nil {
		t.Fatal(err)
	}
	check, err := pipeline.CheckRegression(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// Drop within threshold should not be flagged
	if check.HasRegressions {
		t.Errorf("small drop within threshold should not be flagged as regression, got: %v", check.Regressions)
	}
}
