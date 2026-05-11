// agent_evaluator.go
// ==================
// Three-level agent evaluation framework — Go port.
//
// Evaluates AI agents across three dimensions:
//  1. Retrieval   — Hit Rate, Precision@K, Recall@K, MRR, NDCG@K
//  2. Generation  — Rule-based checks + LLM-as-judge
//  3. End-to-End  — Task success rate across multi-turn scenarios
//
// Also provides a ContinuousEvaluationPipeline with regression detection.
//
// See: docs/08-evaluation-and-guardrails/01-evaluating-agents.md
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
)

// ===========================================================================
// LEVEL 1 — RETRIEVAL EVALUATION
// ===========================================================================

// RetrievalTestCase holds a single retrieval test case.
type RetrievalTestCase struct {
	Query                    string
	RelevantDocIDs           []string
	PartiallyRelevantDocIDs  []string
	IrrelevantDocIDs         []string
	MinResultsExpected       int
}

// RetrievalMetrics contains per-query evaluation metrics.
type RetrievalMetrics struct {
	Query           string
	Hit             int
	PrecisionAtK    float64
	RecallAtK       float64
	ReciprocalRank  float64
	NDCGAtK         float64
	RelevantFound   int
	RelevantTotal   int
	RetrievedIDs    []string
}

// RetrievalReport aggregates retrieval metrics across all test cases.
type RetrievalReport struct {
	HitRate                float64
	PrecisionAtK           float64
	RecallAtK              float64
	MRR                    float64
	NDCGAtK                float64
	TotalQueries           int
	QueriesWithZeroResults int
	PerQuery               []RetrievalMetrics
}

// String returns a human-readable report.
func (r RetrievalReport) String() string {
	return fmt.Sprintf(`
RETRIEVAL EVALUATION REPORT
============================
Total Queries: %d

Hit Rate:        %.2f%%  (target: > 90%%)
Precision@5:     %.2f%%  (target: > 70%%)
Recall@5:        %.2f%%  (target: > 80%%)
MRR:             %.2f%%  (target: > 60%%)
NDCG@5:          %.2f%%  (target: > 70%%)

Queries with zero relevant results: %d
`,
		r.TotalQueries,
		r.HitRate*100, r.PrecisionAtK*100, r.RecallAtK*100,
		r.MRR*100, r.NDCGAtK*100,
		r.QueriesWithZeroResults,
	)
}

// Retriever is the interface a retriever implementation must satisfy.
type Retriever interface {
	Search(ctx context.Context, query string, k int) ([]map[string]string, error)
}

// RetrievalEvaluator evaluates retrieval quality.
type RetrievalEvaluator struct {
	Retriever Retriever
	TestCases []RetrievalTestCase
}

// Evaluate runs all test cases and returns an aggregated report.
func (e *RetrievalEvaluator) Evaluate(ctx context.Context, k int) (RetrievalReport, error) {
	var results []RetrievalMetrics
	for _, tc := range e.TestCases {
		docs, err := e.Retriever.Search(ctx, tc.Query, k)
		if err != nil {
			return RetrievalReport{}, fmt.Errorf("search error for query %q: %w", tc.Query, err)
		}
		ids := make([]string, len(docs))
		for i, d := range docs {
			ids[i] = d["id"]
		}
		results = append(results, e.calculateMetrics(tc, ids, k))
	}
	return e.aggregate(results), nil
}

func (e *RetrievalEvaluator) calculateMetrics(tc RetrievalTestCase, retrieved []string, k int) RetrievalMetrics {
	relevant := toSet(tc.RelevantDocIDs)
	retrievedSet := toSet(retrieved[:min(k, len(retrieved))])

	hit := 0
	tp := 0
	for id := range relevant {
		if retrievedSet[id] {
			tp++
		}
	}
	if tp > 0 {
		hit = 1
	}

	precision := 0.0
	if len(retrievedSet) > 0 {
		precision = float64(tp) / float64(len(retrievedSet))
	}

	recall := 1.0
	if len(relevant) > 0 {
		recall = float64(tp) / float64(len(relevant))
	}

	rr := 0.0
	for i, id := range retrieved {
		if relevant[id] {
			rr = 1.0 / float64(i+1)
			break
		}
	}

	ndcg := e.calculateNDCG(tc, retrieved, k)

	return RetrievalMetrics{
		Query:          tc.Query,
		Hit:            hit,
		PrecisionAtK:   precision,
		RecallAtK:      recall,
		ReciprocalRank: rr,
		NDCGAtK:        ndcg,
		RelevantFound:  tp,
		RelevantTotal:  len(relevant),
		RetrievedIDs:   retrieved,
	}
}

// calculateNDCG computes NDCG with graded relevance (relevant=2, partial=1).
func (e *RetrievalEvaluator) calculateNDCG(tc RetrievalTestCase, retrieved []string, k int) float64 {
	scores := make(map[string]float64)
	for _, id := range tc.RelevantDocIDs {
		scores[id] = 2
	}
	for _, id := range tc.PartiallyRelevantDocIDs {
		if _, exists := scores[id]; !exists {
			scores[id] = 1
		}
	}

	// DCG
	dcg := 0.0
	limit := min(k, len(retrieved))
	for i := 0; i < limit; i++ {
		rel := scores[retrieved[i]]
		dcg += rel / math.Log2(float64(i+2))
	}

	// IDCG — sort scores descending
	idealScores := sortedValues(scores)
	idcg := 0.0
	for i, rel := range idealScores {
		if i >= k {
			break
		}
		idcg += rel / math.Log2(float64(i+2))
	}

	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func (e *RetrievalEvaluator) aggregate(results []RetrievalMetrics) RetrievalReport {
	n := float64(len(results))
	r := RetrievalReport{TotalQueries: len(results)}
	for _, m := range results {
		r.HitRate += float64(m.Hit)
		r.PrecisionAtK += m.PrecisionAtK
		r.RecallAtK += m.RecallAtK
		r.MRR += m.ReciprocalRank
		r.NDCGAtK += m.NDCGAtK
		if m.RelevantFound == 0 && m.RelevantTotal > 0 {
			r.QueriesWithZeroResults++
		}
	}
	r.HitRate /= n
	r.PrecisionAtK /= n
	r.RecallAtK /= n
	r.MRR /= n
	r.NDCGAtK /= n
	r.PerQuery = results
	return r
}

// ===========================================================================
// LEVEL 2 — GENERATION EVALUATION
// ===========================================================================

// GenerationTestCase holds a single generation test case.
type GenerationTestCase struct {
	Query                       string
	ExpectedAnswerContains      []string
	ExpectedAnswerNotContains   []string
	ExpectedSources             []string
	MinAnswerLength             int
	MaxAnswerLength             int
	ReferenceAnswer             string
	EvaluationCriteria          string
}

// JudgeResult is the output from a single LLM judge dimension.
type JudgeResult struct {
	Dimension   string
	Passed      bool
	Score       int
	Issues      []string
	Explanation string
}

// RuleChecks holds the results of deterministic rule-based checks.
type RuleChecks struct {
	ContainsRequired  bool
	MissingRequired   []string
	AvoidsForbidden   bool
	FoundForbidden    []string
	LengthOK          bool
	SourcesCited      bool
	AllPassed         bool
}

// GenerationQueryResult holds evaluation results for one query.
type GenerationQueryResult struct {
	Query       string
	Response    string
	RuleChecks  RuleChecks
	JudgeChecks map[string]JudgeResult
	OverallPass bool
}

// GenerationReport aggregates generation evaluation metrics.
type GenerationReport struct {
	OverallPassRate        float64
	ContainsRequiredRate   float64
	AvoidsForbiddenRate    float64
	FaithfulnessPassRate   *float64
	RelevancePassRate      *float64
	CompletenessPassRate   *float64
	TotalQueries           int
	PerQuery               []GenerationQueryResult
}

// String returns a human-readable report.
func (g GenerationReport) String() string {
	b := &strings.Builder{}
	fmt.Fprintf(b, "\nGENERATION EVALUATION REPORT\n")
	fmt.Fprintf(b, "==============================\n")
	fmt.Fprintf(b, "Total Queries: %d\n\n", g.TotalQueries)
	fmt.Fprintf(b, "Overall Pass Rate:       %.2f%%\n", g.OverallPassRate*100)
	fmt.Fprintf(b, "Contains Required:       %.2f%%\n", g.ContainsRequiredRate*100)
	fmt.Fprintf(b, "Avoids Forbidden:        %.2f%%\n", g.AvoidsForbiddenRate*100)
	if g.FaithfulnessPassRate != nil {
		fmt.Fprintf(b, "Faithfulness (judge):    %.2f%%\n", *g.FaithfulnessPassRate*100)
	}
	if g.RelevancePassRate != nil {
		fmt.Fprintf(b, "Relevance (judge):       %.2f%%\n", *g.RelevancePassRate*100)
	}
	if g.CompletenessPassRate != nil {
		fmt.Fprintf(b, "Completeness (judge):    %.2f%%\n", *g.CompletenessPassRate*100)
	}
	return b.String()
}

// AgentResponse is the response returned by a demo or real agent.
type AgentResponse struct {
	Content             string
	RetrievedDocuments  []string
	ToolCalls           []struct{ Name string }
}

// Agent is the interface an agent implementation must satisfy.
type Agent interface {
	Run(ctx context.Context, query string, history []map[string]string) (AgentResponse, error)
}

// LLMJudge uses an LLM to evaluate response quality.
type LLMJudge struct {
	Model string
}

const faithfulnessPrompt = `You are evaluating whether an AI response is faithful to the provided source documents.

Faithfulness means: Every factual claim in the response is directly supported by at least one of the source documents. The response does not add information not found in the sources.

SOURCE DOCUMENTS:
%s

RESPONSE TO EVALUATE:
%s

Evaluate the response for faithfulness. Output JSON:
{
    "is_faithful": true/false,
    "score": 1-5,
    "unsupported_claims": ["claim1", "claim2"],
    "explanation": "Brief explanation of your evaluation"
}`

const relevancePrompt = `You are evaluating whether an AI response is relevant to the user's question.

Relevance means: The response directly addresses what the user asked. It does not go off-topic or provide unnecessary information.

USER QUESTION:
%s

RESPONSE TO EVALUATE:
%s

Evaluate the response for relevance. Output JSON:
{
    "is_relevant": true/false,
    "score": 1-5,
    "off_topic_parts": ["part1", "part2"],
    "explanation": "Brief explanation"
}`

const completenessPrompt = `You are evaluating whether an AI response completely answers the user's question.

Completeness means: The response addresses ALL parts of the user's question. If the user asked multiple questions, all are answered. If the user asked for a comparison, both sides are covered.

USER QUESTION:
%s

RESPONSE TO EVALUATE:
%s

Evaluate the response for completeness. Output JSON:
{
    "is_complete": true/false,
    "score": 1-5,
    "missing_parts": ["unanswered question 1", "unanswered question 2"],
    "explanation": "Brief explanation"
}`

// EvaluateFaithfulness evaluates whether a response is grounded in source documents.
func (j *LLMJudge) EvaluateFaithfulness(ctx context.Context, response string, sourceDocs []string) (JudgeResult, error) {
	sources := strings.Join(sourceDocs, "\n\n---\n\n")
	prompt := fmt.Sprintf(faithfulnessPrompt, truncate(sources, 10000), truncate(response, 5000))
	raw, err := j.callJudge(ctx, prompt)
	if err != nil {
		return JudgeResult{}, err
	}
	return JudgeResult{
		Dimension:   "faithfulness",
		Passed:      getBool(raw, "is_faithful"),
		Score:       getInt(raw, "score"),
		Issues:      getStringSlice(raw, "unsupported_claims"),
		Explanation: getString(raw, "explanation"),
	}, nil
}

// EvaluateRelevance evaluates whether a response is relevant to the question.
func (j *LLMJudge) EvaluateRelevance(ctx context.Context, response, userQuestion string) (JudgeResult, error) {
	prompt := fmt.Sprintf(relevancePrompt, userQuestion, truncate(response, 5000))
	raw, err := j.callJudge(ctx, prompt)
	if err != nil {
		return JudgeResult{}, err
	}
	return JudgeResult{
		Dimension:   "relevance",
		Passed:      getBool(raw, "is_relevant"),
		Score:       getInt(raw, "score"),
		Issues:      getStringSlice(raw, "off_topic_parts"),
		Explanation: getString(raw, "explanation"),
	}, nil
}

// EvaluateCompleteness evaluates whether a response completely answers the question.
func (j *LLMJudge) EvaluateCompleteness(ctx context.Context, response, userQuestion string) (JudgeResult, error) {
	prompt := fmt.Sprintf(completenessPrompt, userQuestion, truncate(response, 5000))
	raw, err := j.callJudge(ctx, prompt)
	if err != nil {
		return JudgeResult{}, err
	}
	return JudgeResult{
		Dimension:   "completeness",
		Passed:      getBool(raw, "is_complete"),
		Score:       getInt(raw, "score"),
		Issues:      getStringSlice(raw, "missing_parts"),
		Explanation: getString(raw, "explanation"),
	}, nil
}

// callJudge calls the judge LLM and parses the JSON response.
// In production replace this stub with a real OpenAI API call.
func (j *LLMJudge) callJudge(_ context.Context, _ string) (map[string]interface{}, error) {
	// Stub — replace with real API call in production.
	return map[string]interface{}{
		"is_faithful": true,
		"is_relevant": true,
		"is_complete": true,
		"score":       float64(4),
		"unsupported_claims": []interface{}{},
		"off_topic_parts":    []interface{}{},
		"missing_parts":      []interface{}{},
		"explanation":        "Stub judge — replace with real LLM call in production.",
	}, nil
}

// GenerationEvaluator evaluates generation quality.
type GenerationEvaluator struct {
	Agent     Agent
	TestCases []GenerationTestCase
	Judge     *LLMJudge
}

// Evaluate runs all test cases and returns an aggregated report.
func (e *GenerationEvaluator) Evaluate(ctx context.Context) (GenerationReport, error) {
	var results []GenerationQueryResult
	for _, tc := range e.TestCases {
		resp, err := e.Agent.Run(ctx, tc.Query, nil)
		if err != nil {
			return GenerationReport{}, fmt.Errorf("agent error for query %q: %w", tc.Query, err)
		}
		ruleChecks := e.ruleBasedChecks(tc, resp.Content)
		judgeChecks := make(map[string]JudgeResult)
		if e.Judge != nil {
			f, err := e.Judge.EvaluateFaithfulness(ctx, resp.Content, resp.RetrievedDocuments)
			if err == nil {
				judgeChecks["faithfulness"] = f
			}
			r, err := e.Judge.EvaluateRelevance(ctx, resp.Content, tc.Query)
			if err == nil {
				judgeChecks["relevance"] = r
			}
			c, err := e.Judge.EvaluateCompleteness(ctx, resp.Content, tc.Query)
			if err == nil {
				judgeChecks["completeness"] = c
			}
		}
		overallPass := ruleChecks.AllPassed
		for _, jr := range judgeChecks {
			if !jr.Passed {
				overallPass = false
			}
		}
		results = append(results, GenerationQueryResult{
			Query:       tc.Query,
			Response:    resp.Content,
			RuleChecks:  ruleChecks,
			JudgeChecks: judgeChecks,
			OverallPass: overallPass,
		})
	}
	return e.aggregate(results), nil
}

func (e *GenerationEvaluator) ruleBasedChecks(tc GenerationTestCase, response string) RuleChecks {
	lower := strings.ToLower(response)
	rc := RuleChecks{
		ContainsRequired: true,
		AvoidsForbidden:  true,
		LengthOK:         true,
		SourcesCited:     true,
	}

	if len(tc.ExpectedAnswerContains) > 0 {
		for _, phrase := range tc.ExpectedAnswerContains {
			if !strings.Contains(lower, strings.ToLower(phrase)) {
				rc.ContainsRequired = false
				rc.MissingRequired = append(rc.MissingRequired, phrase)
			}
		}
	}

	if len(tc.ExpectedAnswerNotContains) > 0 {
		for _, phrase := range tc.ExpectedAnswerNotContains {
			if strings.Contains(lower, strings.ToLower(phrase)) {
				rc.AvoidsForbidden = false
				rc.FoundForbidden = append(rc.FoundForbidden, phrase)
			}
		}
	}

	maxLen := tc.MaxAnswerLength
	if maxLen == 0 {
		maxLen = 2000
	}
	minLen := tc.MinAnswerLength
	if minLen == 0 {
		minLen = 20
	}
	rc.LengthOK = len(response) >= minLen && len(response) <= maxLen

	if len(tc.ExpectedSources) > 0 {
		for _, src := range tc.ExpectedSources {
			if !strings.Contains(lower, strings.ToLower(src)) {
				rc.SourcesCited = false
			}
		}
	}

	rc.AllPassed = rc.ContainsRequired && rc.AvoidsForbidden && rc.LengthOK && rc.SourcesCited
	return rc
}

func (e *GenerationEvaluator) aggregate(results []GenerationQueryResult) GenerationReport {
	n := float64(len(results))
	r := GenerationReport{TotalQueries: len(results)}
	faithPass, relPass, compPass := 0, 0, 0
	hasJudge := false
	for _, qr := range results {
		if qr.OverallPass {
			r.OverallPassRate++
		}
		if qr.RuleChecks.ContainsRequired {
			r.ContainsRequiredRate++
		}
		if qr.RuleChecks.AvoidsForbidden {
			r.AvoidsForbiddenRate++
		}
		if len(qr.JudgeChecks) > 0 {
			hasJudge = true
			if qr.JudgeChecks["faithfulness"].Passed {
				faithPass++
			}
			if qr.JudgeChecks["relevance"].Passed {
				relPass++
			}
			if qr.JudgeChecks["completeness"].Passed {
				compPass++
			}
		}
	}
	r.OverallPassRate /= n
	r.ContainsRequiredRate /= n
	r.AvoidsForbiddenRate /= n
	if hasJudge {
		fp := float64(faithPass) / n
		rp := float64(relPass) / n
		cp := float64(compPass) / n
		r.FaithfulnessPassRate = &fp
		r.RelevancePassRate = &rp
		r.CompletenessPassRate = &cp
	}
	r.PerQuery = results
	return r
}

// ===========================================================================
// LEVEL 3 — END-TO-END EVALUATION
// ===========================================================================

// EndToEndTestCase holds a multi-turn end-to-end test scenario.
type EndToEndTestCase struct {
	Scenario             string
	UserMessages         []string
	ExpectedOutcome      string
	ExpectedToolsCalled  []string
	MaxTurnsExpected     int
	ForbiddenBehaviors   []string
}

// ScenarioResult holds evaluation results for one scenario.
type ScenarioResult struct {
	Scenario         string
	Outcome          string
	ExpectedOutcome  string
	TurnsTaken       int
	ToolsCalled      []string
	ExpectedTools    []string
	Success          bool
}

// EndToEndReport aggregates end-to-end evaluation results.
type EndToEndReport struct {
	TaskSuccessRate        float64
	AvgTurnsToResolution  float64
	TotalScenarios        int
	PerScenario           []ScenarioResult
}

// String returns a human-readable report.
func (e EndToEndReport) String() string {
	b := &strings.Builder{}
	fmt.Fprintf(b, "\nEND-TO-END EVALUATION REPORT\n")
	fmt.Fprintf(b, "=============================\n")
	fmt.Fprintf(b, "Total Scenarios: %d\n\n", e.TotalScenarios)
	fmt.Fprintf(b, "Task Success Rate:         %.2f%%  (target: > 85%%)\n", e.TaskSuccessRate*100)
	fmt.Fprintf(b, "Avg Turns to Resolution:   %.1f\n\n", e.AvgTurnsToResolution)
	for _, s := range e.PerScenario {
		mark := "✅"
		if !s.Success {
			mark = "❌"
		}
		fmt.Fprintf(b, "  %s  %s  (%d turn(s), outcome: %s)\n",
			mark, truncate(s.Scenario, 60), s.TurnsTaken, s.Outcome)
	}
	return b.String()
}

var resolutionMarkers = []string{
	"is there anything else",
	"i hope that helps",
	"your request has been",
	"i've completed",
	"would you like me to",
}

// EndToEndEvaluator evaluates multi-turn scenarios.
type EndToEndEvaluator struct {
	Agent     Agent
	TestCases []EndToEndTestCase
}

// Evaluate runs all scenarios and returns an aggregated report.
func (e *EndToEndEvaluator) Evaluate(ctx context.Context) (EndToEndReport, error) {
	var results []ScenarioResult
	for _, tc := range e.TestCases {
		var history []map[string]string
		var toolsCalled []string
		outcome := "unknown"
		turnsTaken := 0
		maxTurns := tc.MaxTurnsExpected
		if maxTurns == 0 {
			maxTurns = 5
		}
		for i, msg := range tc.UserMessages {
			resp, err := e.Agent.Run(ctx, msg, history)
			if err != nil {
				return EndToEndReport{}, err
			}
			turnsTaken = i + 1
			history = append(history,
				map[string]string{"role": "user", "content": msg},
				map[string]string{"role": "assistant", "content": resp.Content},
			)
			for _, tc2 := range resp.ToolCalls {
				toolsCalled = append(toolsCalled, tc2.Name)
			}
			if isResolved(resp.Content) {
				outcome = "resolved"
				break
			}
		}
		if outcome == "unknown" {
			if turnsTaken >= maxTurns {
				outcome = "unresolved"
			} else {
				outcome = "incomplete"
			}
		}
		results = append(results, ScenarioResult{
			Scenario:        tc.Scenario,
			Outcome:         outcome,
			ExpectedOutcome: tc.ExpectedOutcome,
			TurnsTaken:      turnsTaken,
			ToolsCalled:     toolsCalled,
			ExpectedTools:   tc.ExpectedToolsCalled,
			Success:         outcome == tc.ExpectedOutcome,
		})
	}
	n := float64(len(results))
	successCount := 0
	totalTurns := 0
	for _, r := range results {
		if r.Success {
			successCount++
		}
		totalTurns += r.TurnsTaken
	}
	return EndToEndReport{
		TaskSuccessRate:       float64(successCount) / n,
		AvgTurnsToResolution:  float64(totalTurns) / n,
		TotalScenarios:        len(results),
		PerScenario:           results,
	}, nil
}

func isResolved(content string) bool {
	lower := strings.ToLower(content)
	for _, marker := range resolutionMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// ===========================================================================
// CONTINUOUS EVALUATION PIPELINE
// ===========================================================================

const regressionThreshold = 0.05

// FullEvaluationReport combines all three evaluation levels.
type FullEvaluationReport struct {
	Retrieval  RetrievalReport
	Generation GenerationReport
	EndToEnd   EndToEndReport
}

// String returns a combined human-readable report.
func (f FullEvaluationReport) String() string {
	return f.Retrieval.String() + f.Generation.String() + f.EndToEnd.String()
}

// Summary returns a brief one-line summary.
func (f FullEvaluationReport) Summary() string {
	return fmt.Sprintf("hit_rate=%.1f%%  pass_rate=%.1f%%  task_success=%.1f%%",
		f.Retrieval.HitRate*100,
		f.Generation.OverallPassRate*100,
		f.EndToEnd.TaskSuccessRate*100,
	)
}

// RegressionCheck holds the results of a regression check.
type RegressionCheck struct {
	HasRegressions bool
	Regressions    []string
	Baseline       FullEvaluationReport
	Current        FullEvaluationReport
}

// String returns a human-readable regression check result.
func (rc RegressionCheck) String() string {
	if !rc.HasRegressions {
		return "\n✅  No regressions detected. All metrics within acceptable range.\n"
	}
	b := &strings.Builder{}
	fmt.Fprintln(b, "\n❌  REGRESSIONS DETECTED")
	fmt.Fprintln(b, strings.Repeat("=", 30))
	for _, r := range rc.Regressions {
		fmt.Fprintf(b, "  • %s\n", r)
	}
	return b.String()
}

// ContinuousEvaluationPipeline orchestrates all evaluators and detects regressions.
type ContinuousEvaluationPipeline struct {
	Harness             interface{}
	RetrievalEvaluator  *RetrievalEvaluator
	GenerationEvaluator *GenerationEvaluator
	EndToEndEvaluator   *EndToEndEvaluator
	baseline            *FullEvaluationReport
}

// SetBaseline runs all evaluators and stores the result as the baseline.
func (p *ContinuousEvaluationPipeline) SetBaseline(ctx context.Context) error {
	report, err := p.RunAll(ctx)
	if err != nil {
		return err
	}
	p.baseline = &report
	log.Printf("Baseline set: %s", report.Summary())
	return nil
}

// RunAll runs all three evaluators.
func (p *ContinuousEvaluationPipeline) RunAll(ctx context.Context) (FullEvaluationReport, error) {
	ret, err := p.RetrievalEvaluator.Evaluate(ctx, 5)
	if err != nil {
		return FullEvaluationReport{}, err
	}
	gen, err := p.GenerationEvaluator.Evaluate(ctx)
	if err != nil {
		return FullEvaluationReport{}, err
	}
	e2e, err := p.EndToEndEvaluator.Evaluate(ctx)
	if err != nil {
		return FullEvaluationReport{}, err
	}
	return FullEvaluationReport{Retrieval: ret, Generation: gen, EndToEnd: e2e}, nil
}

// CheckRegression compares current metrics against the baseline.
func (p *ContinuousEvaluationPipeline) CheckRegression(ctx context.Context) (RegressionCheck, error) {
	if p.baseline == nil {
		return RegressionCheck{}, fmt.Errorf("no baseline set: call SetBaseline first")
	}
	current, err := p.RunAll(ctx)
	if err != nil {
		return RegressionCheck{}, err
	}
	var regressions []string
	check := func(name string, base, curr float64) {
		if curr < base-regressionThreshold {
			regressions = append(regressions,
				fmt.Sprintf("%s dropped from %.1f%% to %.1f%%", name, base*100, curr*100))
		}
	}
	check("Hit Rate", p.baseline.Retrieval.HitRate, current.Retrieval.HitRate)
	check("MRR", p.baseline.Retrieval.MRR, current.Retrieval.MRR)
	check("NDCG@5", p.baseline.Retrieval.NDCGAtK, current.Retrieval.NDCGAtK)
	check("Generation Pass Rate", p.baseline.Generation.OverallPassRate, current.Generation.OverallPassRate)
	check("Task Success Rate", p.baseline.EndToEnd.TaskSuccessRate, current.EndToEnd.TaskSuccessRate)

	return RegressionCheck{
		HasRegressions: len(regressions) > 0,
		Regressions:    regressions,
		Baseline:       *p.baseline,
		Current:        current,
	}, nil
}

// ===========================================================================
// DEMO STUBS
// ===========================================================================

type demoRetriever struct {
	corpus   map[string][]string
	degraded bool
}

func (r *demoRetriever) Search(_ context.Context, query string, k int) ([]map[string]string, error) {
	docs := r.corpus[query]
	if r.degraded {
		reversed := make([]string, len(docs))
		for i, d := range docs {
			reversed[len(docs)-1-i] = d
		}
		docs = reversed
	}
	if k < len(docs) {
		docs = docs[:k]
	}
	result := make([]map[string]string, len(docs))
	for i, id := range docs {
		result[i] = map[string]string{"id": id}
	}
	return result, nil
}

type demoAgent struct{}

var agentResponses = map[string]string{
	"What's your return policy?": "You may return items within 30 days in original packaging with a receipt. Source: return-policy.md Is there anything else I can help you with?",
	"How much does shipping cost?": "Shipping is free on orders over $50. Source: shipping-info.md Is there anything else I can help you with?",
	"Do you ship to Germany?": "Yes, we ship internationally including Germany. Source: international-shipping.md Is there anything else I can help you with?",
	"I received a damaged item and want to return it.": "I'm sorry to hear that. Please return it within 30 days with the receipt. Is there anything else I can help you with?",
}

func (a *demoAgent) Run(_ context.Context, query string, _ []map[string]string) (AgentResponse, error) {
	content, ok := agentResponses[query]
	if !ok {
		content = fmt.Sprintf("I can help with that. Is there anything else I can help you with?")
	}
	return AgentResponse{
		Content:            content,
		RetrievedDocuments: []string{"Relevant document excerpt."},
	}, nil
}

func buildRetrievalTests() []RetrievalTestCase {
	return []RetrievalTestCase{
		{Query: "What's your return policy for damaged items?", RelevantDocIDs: []string{"return-policy.md", "damaged-goods-policy.md"}, IrrelevantDocIDs: []string{"pricing.md"}, MinResultsExpected: 2},
		{Query: "How do I reset my password?", RelevantDocIDs: []string{"account-faq.md"}, PartiallyRelevantDocIDs: []string{"security-policy.md"}, MinResultsExpected: 1},
		{Query: "Do you ship to Germany?", RelevantDocIDs: []string{"international-shipping.md"}, MinResultsExpected: 1},
		{Query: "Tell me about your company history", RelevantDocIDs: []string{"about-us.md", "company-history.md"}, MinResultsExpected: 1},
		{Query: "What's the capital of France?", RelevantDocIDs: []string{}, MinResultsExpected: 0},
	}
}

func buildCorpus(degraded bool) map[string][]string {
	corpus := map[string][]string{
		"What's your return policy for damaged items?": {"return-policy.md", "damaged-goods-policy.md", "pricing.md"},
		"How do I reset my password?":                 {"account-faq.md", "security-policy.md"},
		"Do you ship to Germany?":                     {"international-shipping.md", "domestic-shipping.md"},
		"Tell me about your company history":          {"about-us.md", "company-history.md"},
		"What's the capital of France?":               {},
	}
	if degraded {
		result := make(map[string][]string)
		i := 0
		for k, v := range corpus {
			if i%2 == 0 {
				reversed := make([]string, len(v))
				for j, d := range v {
					reversed[len(v)-1-j] = d
				}
				result[k] = reversed
			} else {
				result[k] = v
			}
			i++
		}
		return result
	}
	return corpus
}

func buildGenerationTests() []GenerationTestCase {
	return []GenerationTestCase{
		{Query: "What's your return policy?", ExpectedAnswerContains: []string{"30 days", "original packaging", "receipt"}, ExpectedAnswerNotContains: []string{"60 days"}, ExpectedSources: []string{"return-policy.md"}},
		{Query: "How much does shipping cost?", ExpectedAnswerContains: []string{"free", "$50"}, ExpectedSources: []string{"shipping-info.md"}},
		{Query: "Do you ship to Germany?", ExpectedAnswerContains: []string{"Germany"}, ExpectedSources: []string{"international-shipping.md"}},
	}
}

func buildE2ETests() []EndToEndTestCase {
	return []EndToEndTestCase{
		{Scenario: "Customer wants to return a damaged item", UserMessages: []string{"I received a damaged item and want to return it."}, ExpectedOutcome: "resolved", MaxTurnsExpected: 3},
		{Scenario: "Customer asks about shipping to Germany", UserMessages: []string{"Do you ship to Germany?"}, ExpectedOutcome: "resolved", MaxTurnsExpected: 2},
		{Scenario: "Customer asks about shipping cost", UserMessages: []string{"How much does shipping cost?"}, ExpectedOutcome: "resolved", MaxTurnsExpected: 2},
	}
}

// ===========================================================================
// HELPERS
// ===========================================================================

func toSet(items []string) map[string]bool {
	s := make(map[string]bool, len(items))
	for _, v := range items {
		s[v] = true
	}
	return s
}

func sortedValues(m map[string]float64) []float64 {
	vals := make([]float64, 0, len(m))
	for _, v := range m {
		vals = append(vals, v)
	}
	// Simple selection sort (small N)
	for i := 0; i < len(vals); i++ {
		for j := i + 1; j < len(vals); j++ {
			if vals[j] > vals[i] {
				vals[i], vals[j] = vals[j], vals[i]
			}
		}
	}
	return vals
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func getBool(m map[string]interface{}, key string) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}

func getInt(m map[string]interface{}, key string) int {
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	return 0
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getStringSlice(m map[string]interface{}, key string) []string {
	raw, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	result := make([]string, 0, len(raw))
	for _, v := range raw {
		if s, ok := v.(string); ok {
			result = append(result, s)
		}
	}
	return result
}

// Suppress unused import warning for json in test builds.
var _ = json.Marshal

// ===========================================================================
// MAIN (DEMO)
// ===========================================================================

func main() {
	ctx := context.Background()
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("AGENT EVALUATION FRAMEWORK — Go Demo")
	fmt.Println(strings.Repeat("=", 60))

	retriever := &demoRetriever{corpus: buildCorpus(false)}
	agent := &demoAgent{}
	judge := &LLMJudge{Model: "gpt-4o"}

	pipeline := &ContinuousEvaluationPipeline{
		RetrievalEvaluator:  &RetrievalEvaluator{Retriever: retriever, TestCases: buildRetrievalTests()},
		GenerationEvaluator: &GenerationEvaluator{Agent: agent, TestCases: buildGenerationTests(), Judge: judge},
		EndToEndEvaluator:   &EndToEndEvaluator{Agent: agent, TestCases: buildE2ETests()},
	}

	fmt.Println("\n[ Stage 1 ] Setting baseline…")
	if err := pipeline.SetBaseline(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Println(pipeline.baseline.String())

	fmt.Println("[ Stage 2 ] Simulating retrieval regression…")
	// Use random seed for reproducible demo shuffling
	rand.New(rand.NewSource(42))
	pipeline.RetrievalEvaluator = &RetrievalEvaluator{
		Retriever: &demoRetriever{corpus: buildCorpus(true)},
		TestCases: buildRetrievalTests(),
	}

	check, err := pipeline.CheckRegression(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(check.String())
	fmt.Println("\n[ Stage 3 ] Full combined report:")
	fmt.Println(check.Current.String())
}
