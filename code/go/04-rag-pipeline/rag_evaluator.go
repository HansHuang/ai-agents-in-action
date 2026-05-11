// rag_evaluator.go — Systematic evaluation of RAG pipeline quality.
//
// Evaluation has two orthogonal dimensions:
//   - Retrieval: does the pipeline surface the right chunks?
//     Metrics: Hit Rate, Mean Reciprocal Rank (MRR), Precision@K, Recall@K
//   - Generation: does the LLM produce faithful, grounded, relevant answers?
//     Metrics: Faithfulness, Relevance, Groundedness  (LLM-as-judge, 0–1 scale)
//
// LLM judge calls use JSON-mode responses.
//
// See: docs/08-evaluation-and-guardrails/
package ragpipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// ---------------------------------------------------------------------------
// Test case types
// ---------------------------------------------------------------------------

// RetrievalTestCase holds a query and the IDs of documents that should appear.
type RetrievalTestCase struct {
	Query          string
	RelevantDocIDs []string
}

// GenerationTestCase holds a query plus facts that must / must not appear.
type GenerationTestCase struct {
	Query          string
	ExpectedFacts  []string
	ForbiddenFacts []string
}

// ---------------------------------------------------------------------------
// RAGEvaluator
// ---------------------------------------------------------------------------

// RAGEvaluator evaluates a RAGPipeline on retrieval and generation quality.
type RAGEvaluator struct {
	Pipeline *RAGPipeline
	Model    string
	apiKey   string
}

// NewRAGEvaluator creates an evaluator for the given pipeline.
func NewRAGEvaluator(pipeline *RAGPipeline, model string) *RAGEvaluator {
	if model == "" {
		model = "gpt-4o-mini"
	}
	return &RAGEvaluator{
		Pipeline: pipeline,
		Model:    model,
		apiKey:   os.Getenv("OPENAI_API_KEY"),
	}
}

// ---------------------------------------------------------------------------
// Retrieval evaluation
// ---------------------------------------------------------------------------

// EvaluateRetrieval computes Hit Rate, MRR, Precision@K, Recall@K.
func (e *RAGEvaluator) EvaluateRetrieval(ctx context.Context, testCases []RetrievalTestCase, k int) (map[string]float64, error) {
	if k <= 0 {
		k = 5
	}

	hitCount := 0
	mrrTotal := 0.0
	precisionTotal := 0.0
	recallTotal := 0.0

	for _, tc := range testCases {
		emb, err := e.Pipeline.Embedder.Embed(ctx, tc.Query)
		if err != nil {
			return nil, fmt.Errorf("embed(%q): %w", tc.Query, err)
		}
		results, err := e.Pipeline.VectorStore.Search(emb, k, nil)
		if err != nil {
			return nil, err
		}

		// Build a set of relevant IDs.
		relevant := make(map[string]bool, len(tc.RelevantDocIDs))
		for _, id := range tc.RelevantDocIDs {
			relevant[id] = true
		}

		// Build set of retrieved IDs and compute metrics.
		hits := 0
		recipRank := 0.0
		for rank, r := range results {
			if relevant[r.ID] {
				hits++
				if recipRank == 0 {
					recipRank = 1.0 / float64(rank+1)
				}
			}
		}

		if hits > 0 {
			hitCount++
		}
		mrrTotal += recipRank
		precisionTotal += float64(hits) / float64(len(results))
		if len(relevant) > 0 {
			recallTotal += float64(hits) / float64(len(relevant))
		}
	}

	n := float64(len(testCases))
	if n == 0 {
		return map[string]float64{
			"hit_rate":    0,
			"mrr":         0,
			"precision_k": 0,
			"recall_k":    0,
		}, nil
	}
	return map[string]float64{
		"hit_rate":    float64(hitCount) / n,
		"mrr":         mrrTotal / n,
		"precision_k": precisionTotal / n,
		"recall_k":    recallTotal / n,
	}, nil
}

// ---------------------------------------------------------------------------
// Generation evaluation (LLM-as-judge)
// ---------------------------------------------------------------------------

type judgeScore struct {
	Score     float64 `json:"score"`
	Reasoning string  `json:"reasoning"`
}

func (e *RAGEvaluator) judgeMetric(ctx context.Context, systemPrompt, userPrompt string) (float64, string, error) {
	jsonFmt := &jsonFmtType{Type: "json_object"}
	text, _, err := callBasicChat(e.apiKey, basicChatReq{
		Model:          e.Model,
		Messages:       []chatMessage{{Role: "system", Content: systemPrompt}, {Role: "user", Content: userPrompt}},
		Temperature:    0,
		MaxTokens:      256,
		ResponseFormat: jsonFmt,
	})
	if err != nil {
		return 0, "", err
	}
	var result judgeScore
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return 0, text, nil
	}
	return math.Max(0, math.Min(1, result.Score)), result.Reasoning, nil
}

func faithfulnessPrompts(query, answer string, sources []string) (string, string) {
	sys := `You evaluate whether an answer is faithful to its sources.
Score: 1.0 = fully supported by sources, 0.0 = contradicts or ignores sources.
Return JSON: {"score": <float 0-1>, "reasoning": "<brief explanation>"}`
	usr := fmt.Sprintf("QUERY: %s\nANSWER: %s\nSOURCES:\n%s",
		query, answer, strings.Join(sources, "\n---\n"))
	return sys, usr
}

func relevancePrompts(query, answer string) (string, string) {
	sys := `You evaluate whether an answer is relevant and helpful for the query.
Score: 1.0 = completely addresses the query, 0.0 = off-topic.
Return JSON: {"score": <float 0-1>, "reasoning": "<brief explanation>"}`
	usr := fmt.Sprintf("QUERY: %s\nANSWER: %s", query, answer)
	return sys, usr
}

func groundednessPrompts(query, answer string, expectedFacts, forbiddenFacts []string) (string, string) {
	sys := `You evaluate whether an answer contains the expected facts and avoids forbidden content.
Score: 1.0 = all expected facts present and no forbidden content, 0.0 = major failures.
Return JSON: {"score": <float 0-1>, "reasoning": "<brief explanation>"}`
	usr := fmt.Sprintf("QUERY: %s\nANSWER: %s\nEXPECTED FACTS: %s\nFORBIDDEN: %s",
		query, answer,
		strings.Join(expectedFacts, "; "),
		strings.Join(forbiddenFacts, "; "))
	return sys, usr
}

// EvaluateGeneration returns faithfulness, relevance, and groundedness scores.
func (e *RAGEvaluator) EvaluateGeneration(ctx context.Context, testCases []GenerationTestCase) (map[string]float64, error) {
	faithTotal, relTotal, groundTotal := 0.0, 0.0, 0.0
	n := float64(len(testCases))

	for _, tc := range testCases {
		ragResp, err := e.Pipeline.Query(ctx, tc.Query, 0, 0)
		if err != nil {
			return nil, fmt.Errorf("query(%q): %w", tc.Query, err)
		}
		answer := ragResp.Answer

		sys, usr := faithfulnessPrompts(tc.Query, answer, ragResp.Sources)
		s, _, err := e.judgeMetric(ctx, sys, usr)
		if err != nil {
			return nil, err
		}
		faithTotal += s

		sys, usr = relevancePrompts(tc.Query, answer)
		s, _, err = e.judgeMetric(ctx, sys, usr)
		if err != nil {
			return nil, err
		}
		relTotal += s

		sys, usr = groundednessPrompts(tc.Query, answer, tc.ExpectedFacts, tc.ForbiddenFacts)
		s, _, err = e.judgeMetric(ctx, sys, usr)
		if err != nil {
			return nil, err
		}
		groundTotal += s
	}

	if n == 0 {
		return map[string]float64{"faithfulness": 0, "relevance": 0, "groundedness": 0}, nil
	}
	return map[string]float64{
		"faithfulness": faithTotal / n,
		"relevance":    relTotal / n,
		"groundedness": groundTotal / n,
	}, nil
}

// ---------------------------------------------------------------------------
// Pipeline comparison
// ---------------------------------------------------------------------------

// ComparePipelines evaluates multiple pipelines on the same retrieval test cases
// and returns a per-pipeline metrics map.
func (e *RAGEvaluator) ComparePipelines(ctx context.Context, pipelines []*RAGPipeline, testCases []RetrievalTestCase, labels []string, k int) (map[string]map[string]float64, error) {
	if len(labels) != len(pipelines) {
		return nil, fmt.Errorf("labels length (%d) must match pipelines length (%d)", len(labels), len(pipelines))
	}
	results := make(map[string]map[string]float64, len(pipelines))
	for i, pipeline := range pipelines {
		eval := NewRAGEvaluator(pipeline, e.Model)
		metrics, err := eval.EvaluateRetrieval(ctx, testCases, k)
		if err != nil {
			return nil, fmt.Errorf("pipeline %q: %w", labels[i], err)
		}
		results[labels[i]] = metrics
	}
	return results, nil
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

// GenerateReport formats evaluation results into a human-readable report.
func (e *RAGEvaluator) GenerateReport(results map[string]interface{}) string {
	var sb strings.Builder
	sb.WriteString(strings.Repeat("=", 60) + "\n")
	sb.WriteString("RAG EVALUATION REPORT\n")
	sb.WriteString(strings.Repeat("=", 60) + "\n\n")

	if retrieval, ok := results["retrieval"].(map[string]float64); ok {
		sb.WriteString("RETRIEVAL METRICS\n")
		sb.WriteString(strings.Repeat("-", 30) + "\n")
		for metric, val := range retrieval {
			sb.WriteString(fmt.Sprintf("  %-20s %.4f\n", metric, val))
		}
		sb.WriteString("\n")
	}

	if generation, ok := results["generation"].(map[string]float64); ok {
		sb.WriteString("GENERATION METRICS\n")
		sb.WriteString(strings.Repeat("-", 30) + "\n")
		for metric, val := range generation {
			sb.WriteString(fmt.Sprintf("  %-20s %.4f\n", metric, val))
		}
		sb.WriteString("\n")
	}

	if comparison, ok := results["comparison"].(map[string]map[string]float64); ok {
		sb.WriteString("PIPELINE COMPARISON\n")
		sb.WriteString(strings.Repeat("-", 30) + "\n")
		for label, metrics := range comparison {
			sb.WriteString(fmt.Sprintf("  [%s]\n", label))
			for k, v := range metrics {
				sb.WriteString(fmt.Sprintf("    %-20s %.4f\n", k, v))
			}
		}
	}

	return sb.String()
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunRAGEvaluator demonstrates pipeline evaluation on synthetic test cases.
func RunRAGEvaluator() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("RAG EVALUATOR DEMO")
	fmt.Println(strings.Repeat("=", 70))

	embedder := NewEmbeddingGenerator("text-embedding-3-small", 0)
	vectorStore := NewSimpleVectorStore()
	pipeline := NewRAGPipeline(vectorStore, embedder, "gpt-4o-mini", 200, 40, 5, 0.4)
	evaluator := NewRAGEvaluator(pipeline, "gpt-4o-mini")

	ctx := context.Background()

	// Ingest sample docs.
	docs := map[string]string{
		"vacation-policy": "Employees earn 15 days of paid vacation per year. Vacation accrues monthly.",
		"expense-policy":  "Submit expense reports within 30 days. Receipts required over $25.",
	}
	for source, text := range docs {
		_, err := pipeline.IngestText(ctx, text, map[string]interface{}{"source": source})
		if err != nil {
			fmt.Printf("Error ingesting %s: %v\n", source, err)
		}
	}

	retrievalCases := []RetrievalTestCase{
		{Query: "vacation days per year", RelevantDocIDs: []string{"vacation-policy"}},
		{Query: "expense report deadline", RelevantDocIDs: []string{"expense-policy"}},
	}

	fmt.Println("\nRunning retrieval evaluation...")
	metrics, err := evaluator.EvaluateRetrieval(ctx, retrievalCases, 5)
	if err != nil {
		fmt.Printf("Retrieval eval error: %v\n", err)
		return
	}

	results := map[string]interface{}{
		"retrieval": metrics,
	}
	fmt.Println(evaluator.GenerateReport(results))
}
