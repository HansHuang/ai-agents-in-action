// advanced_retriever.go — Advanced retrieval techniques for RAG pipelines.
//
// Implements four strategies that improve over naive nearest-neighbour search:
//   - Standard:   baseline nearest-neighbour retrieval.
//   - HyDE:       embed a hypothetical answer instead of the raw query.
//   - MultiQuery: rephrase and retrieve for each variant, merge results.
//   - Decompose:  break complex question into sub-questions, retrieve each.
//   - Contextual: enrich query from conversation history before retrieval.
//
// See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
package ragpipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// ---------------------------------------------------------------------------
// AdvancedRetriever
// ---------------------------------------------------------------------------

// AdvancedRetriever wraps a SimpleVectorStore and EmbeddingGenerator with
// advanced retrieval strategies.
type AdvancedRetriever struct {
	VectorStore *SimpleVectorStore
	Embedder    *EmbeddingGenerator
	Model       string
	apiKey      string
}

// NewAdvancedRetriever creates an AdvancedRetriever.
func NewAdvancedRetriever(vectorStore *SimpleVectorStore, embedder *EmbeddingGenerator, model string) *AdvancedRetriever {
	if model == "" {
		model = "gpt-4o"
	}
	return &AdvancedRetriever{
		VectorStore: vectorStore,
		Embedder:    embedder,
		Model:       model,
		apiKey:      os.Getenv("OPENAI_API_KEY"),
	}
}

// ---------------------------------------------------------------------------
// Standard baseline
// ---------------------------------------------------------------------------

// StandardRetrieve performs basic nearest-neighbour retrieval.
func (ar *AdvancedRetriever) StandardRetrieve(ctx context.Context, question string, k int) ([]VectorSearchResult, error) {
	emb, err := ar.Embedder.Embed(ctx, question)
	if err != nil {
		return nil, err
	}
	return ar.VectorStore.Search(emb, k, nil)
}

// ---------------------------------------------------------------------------
// HyDE
// ---------------------------------------------------------------------------

// HydeRetrieve uses Hypothetical Document Embeddings retrieval.
// 1. Generate a hypothetical answer with the LLM.
// 2. Embed the hypothetical answer.
// 3. Search the vector store with that embedding.
func (ar *AdvancedRetriever) HydeRetrieve(ctx context.Context, question string, k int) ([]VectorSearchResult, error) {
	prompt := fmt.Sprintf(
		"Write a detailed, factual answer to the following question. "+
			"Use the style of a technical document or FAQ entry.\n\nQuestion: %s", question)
	hypoAnswer, _, err := callBasicChat(ar.apiKey, basicChatReq{
		Model:       ar.Model,
		Messages:    []chatMessage{{Role: "user", Content: prompt}},
		Temperature: 0.3,
	})
	if err != nil {
		return nil, fmt.Errorf("hyde generate: %w", err)
	}

	emb, err := ar.Embedder.Embed(ctx, hypoAnswer)
	if err != nil {
		return nil, fmt.Errorf("hyde embed: %w", err)
	}
	results, err := ar.VectorStore.Search(emb, k, nil)
	if err != nil {
		return nil, err
	}
	for i := range results {
		if results[i].Metadata == nil {
			results[i].Metadata = make(map[string]interface{})
		}
		results[i].Metadata["_retrieval_method"] = "hyde"
	}
	return results, nil
}

// ---------------------------------------------------------------------------
// Multi-query
// ---------------------------------------------------------------------------

// MultiQueryRetrieve generates multiple search queries, retrieves for each,
// and deduplicates the results by highest score.
func (ar *AdvancedRetriever) MultiQueryRetrieve(ctx context.Context, question string, nQueries, kPerQuery int) ([]VectorSearchResult, error) {
	prompt := fmt.Sprintf(
		"Generate exactly %d alternative search queries for the following question. "+
			"Each query should use different vocabulary and phrasing to maximise retrieval coverage.\n\n"+
			"Original question: %s\n\n"+
			`Output a JSON object with a "queries" key containing an array of strings.`,
		nQueries, question)

	raw, _, err := callBasicChat(ar.apiKey, basicChatReq{
		Model:          ar.Model,
		Messages:       []chatMessage{{Role: "user", Content: prompt}},
		Temperature:    0.5,
		ResponseFormat: &jsonFmtType{Type: "json_object"},
	})
	if err != nil {
		return nil, fmt.Errorf("multi-query generate: %w", err)
	}

	var parsed map[string]interface{}
	queries := []string{question}
	if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
		for _, v := range parsed {
			if arr, ok := v.([]interface{}); ok {
				queries = nil
				for _, item := range arr {
					if s, ok := item.(string); ok {
						queries = append(queries, s)
					}
				}
				break
			}
		}
	}

	seen := make(map[string]VectorSearchResult)
	for _, q := range queries {
		if len(queries) > nQueries {
			queries = queries[:nQueries]
		}
		emb, err := ar.Embedder.Embed(ctx, q)
		if err != nil {
			continue
		}
		results, err := ar.VectorStore.Search(emb, kPerQuery, nil)
		if err != nil {
			continue
		}
		for _, r := range results {
			if existing, ok := seen[r.Text]; !ok || r.Score > existing.Score {
				if r.Metadata == nil {
					r.Metadata = make(map[string]interface{})
				}
				r.Metadata["_retrieval_method"] = "multi_query"
				seen[r.Text] = r
			}
		}
	}

	merged := make([]VectorSearchResult, 0, len(seen))
	for _, r := range seen {
		merged = append(merged, r)
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].Score > merged[j].Score })
	return merged, nil
}

// ---------------------------------------------------------------------------
// Decompose-and-retrieve
// ---------------------------------------------------------------------------

// DecomposeAndRetrieve breaks a complex question into sub-questions and
// retrieves for each, merging and deduplicating results.
func (ar *AdvancedRetriever) DecomposeAndRetrieve(ctx context.Context, complexQuestion string, k int) ([]VectorSearchResult, error) {
	prompt := fmt.Sprintf(
		"Break the following question into simple, focused sub-questions. "+
			"Each sub-question should be answerable from a single document.\n\n"+
			"Question: %s\n\n"+
			`Output a JSON object with a "sub_questions" key containing an array of strings.`,
		complexQuestion)

	raw, _, err := callBasicChat(ar.apiKey, basicChatReq{
		Model:          ar.Model,
		Messages:       []chatMessage{{Role: "user", Content: prompt}},
		Temperature:    0.3,
		ResponseFormat: &jsonFmtType{Type: "json_object"},
	})
	if err != nil {
		return nil, fmt.Errorf("decompose generate: %w", err)
	}

	subQuestions := []string{complexQuestion}
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
		for _, v := range parsed {
			if arr, ok := v.([]interface{}); ok {
				subQuestions = nil
				for _, item := range arr {
					if s, ok := item.(string); ok {
						subQuestions = append(subQuestions, s)
					}
				}
				break
			}
		}
	}

	seen := make(map[string]VectorSearchResult)
	for _, sq := range subQuestions {
		emb, err := ar.Embedder.Embed(ctx, sq)
		if err != nil {
			continue
		}
		results, err := ar.VectorStore.Search(emb, 3, nil)
		if err != nil {
			continue
		}
		for _, r := range results {
			if existing, ok := seen[r.Text]; !ok || r.Score > existing.Score {
				if r.Metadata == nil {
					r.Metadata = make(map[string]interface{})
				}
				r.Metadata["_retrieval_method"] = "decompose"
				r.Metadata["_sub_question"] = sq
				seen[r.Text] = r
			}
		}
	}

	merged := make([]VectorSearchResult, 0, len(seen))
	for _, r := range seen {
		merged = append(merged, r)
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].Score > merged[j].Score })
	if k > 0 && len(merged) > k {
		merged = merged[:k]
	}
	return merged, nil
}

// ---------------------------------------------------------------------------
// Contextual retrieve
// ---------------------------------------------------------------------------

// ContextualRetrieve enriches the query from conversation history before retrieving.
func (ar *AdvancedRetriever) ContextualRetrieve(ctx context.Context, question string, conversationHistory []chatMessage, k int) ([]VectorSearchResult, error) {
	if len(conversationHistory) == 0 {
		return ar.StandardRetrieve(ctx, question, k)
	}

	// Use last 6 messages (3 exchanges).
	history := conversationHistory
	if len(history) > 6 {
		history = history[len(history)-6:]
	}
	var historyText string
	for _, m := range history {
		historyText += fmt.Sprintf("%s: %s\n", m.Role, m.Content)
	}

	prompt := fmt.Sprintf(
		"Given the following conversation, rewrite the last question as a fully "+
			"self-contained search query. Resolve all pronouns and implicit references.\n\n"+
			"Conversation:\n%s\nLast question: %s\n\nOutput only the rewritten query, nothing else.",
		historyText, question)

	enriched, _, err := callBasicChat(ar.apiKey, basicChatReq{
		Model:    ar.Model,
		Messages: []chatMessage{{Role: "user", Content: prompt}},
	})
	if err != nil {
		return ar.StandardRetrieve(ctx, question, k)
	}
	if enriched == "" {
		enriched = question
	}

	emb, err := ar.Embedder.Embed(ctx, enriched)
	if err != nil {
		return nil, err
	}
	results, err := ar.VectorStore.Search(emb, k, nil)
	if err != nil {
		return nil, err
	}
	for i := range results {
		if results[i].Metadata == nil {
			results[i].Metadata = make(map[string]interface{})
		}
		results[i].Metadata["_retrieval_method"] = "contextual"
		results[i].Metadata["_enriched_query"] = enriched
	}
	return results, nil
}

// ---------------------------------------------------------------------------
// CompareMethods
// ---------------------------------------------------------------------------

// CompareMethodsResult holds results from all retrieval methods.
type CompareMethodsResult struct {
	Standard   []VectorSearchResult
	Hyde       []VectorSearchResult
	MultiQuery []VectorSearchResult
	Decompose  []VectorSearchResult
	Overlap    map[string][]string // text[:80] → methods that found it
	Unique     map[string]int      // method → count of unique results
}

// CompareMethods runs all retrieval methods on question and returns a comparison.
func (ar *AdvancedRetriever) CompareMethods(ctx context.Context, question string, k int) (*CompareMethodsResult, error) {
	standard, err := ar.StandardRetrieve(ctx, question, k)
	if err != nil {
		return nil, err
	}
	hyde, err := ar.HydeRetrieve(ctx, question, k)
	if err != nil {
		return nil, err
	}
	multi, err := ar.MultiQueryRetrieve(ctx, question, 3, k)
	if err != nil {
		return nil, err
	}
	decompose, err := ar.DecomposeAndRetrieve(ctx, question, k)
	if err != nil {
		return nil, err
	}

	methods := map[string][]VectorSearchResult{
		"standard":    standard,
		"hyde":        hyde,
		"multi_query": multi,
		"decompose":   decompose,
	}

	textToMethods := make(map[string][]string)
	for name, results := range methods {
		for _, r := range results {
			key := r.Text
			if len(key) > 80 {
				key = key[:80]
			}
			textToMethods[key] = append(textToMethods[key], name)
		}
	}

	overlap := make(map[string][]string)
	for text, foundIn := range textToMethods {
		if len(foundIn) > 1 {
			overlap[text] = foundIn
		}
	}
	unique := make(map[string]int)
	for name, results := range methods {
		count := 0
		for _, r := range results {
			key := r.Text
			if len(key) > 80 {
				key = key[:80]
			}
			if len(textToMethods[key]) == 1 {
				count++
			}
		}
		unique[name] = count
	}

	return &CompareMethodsResult{
		Standard:   standard,
		Hyde:       hyde,
		MultiQuery: multi,
		Decompose:  decompose,
		Overlap:    overlap,
		Unique:     unique,
	}, nil
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunAdvancedRetriever demonstrates the advanced retrieval methods.
func RunAdvancedRetriever() {
	fmt.Println("ADVANCED RETRIEVER DEMO")
	fmt.Println("Build a knowledge base, then run advanced retrieval strategies.")
	fmt.Println("Set OPENAI_API_KEY to run a live demo.")
}
