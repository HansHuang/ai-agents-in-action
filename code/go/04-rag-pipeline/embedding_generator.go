// EmbeddingGenerator and EmbeddingComparator for the RAG pipeline.
//
// Demonstrates generating text embeddings with the OpenAI API and comparing
// vectors with cosine similarity to find semantically similar texts.
//
// See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
package ragpipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// OpenAI embedding API types (minimal, no external SDK needed for basic use)
// ---------------------------------------------------------------------------

type embeddingRequest struct {
	Model      string `json:"model"`
	Input      any    `json:"input"` // string or []string
	Dimensions *int   `json:"dimensions,omitempty"`
}

type embeddingObject struct {
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type embeddingResponse struct {
	Data []embeddingObject `json:"data"`
}

type openAIErrorBody struct {
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ---------------------------------------------------------------------------
// EmbeddingGenerator
// ---------------------------------------------------------------------------

// EmbeddingGenerator generates embeddings using the OpenAI embeddings API.
//
// The zero value is not usable; create instances with NewEmbeddingGenerator.
type EmbeddingGenerator struct {
	// Model is the OpenAI embedding model name, e.g. "text-embedding-3-small".
	Model string

	// Dimensions optionally reduces the output dimensions.
	// Only supported by text-embedding-3-* models.
	// Zero means use the model default.
	Dimensions int

	apiKey string
	client *http.Client
}

// NewEmbeddingGenerator creates an EmbeddingGenerator that reads the API key
// from the OPENAI_API_KEY environment variable.
func NewEmbeddingGenerator(model string, dimensions int) *EmbeddingGenerator {
	return &EmbeddingGenerator{
		Model:      model,
		Dimensions: dimensions,
		apiKey:     os.Getenv("OPENAI_API_KEY"),
		client:     &http.Client{Timeout: 30 * time.Second},
	}
}

// Embed generates an embedding for a single text string.
func (g *EmbeddingGenerator) Embed(ctx context.Context, text string) ([]float64, error) {
	return g.callAPI(ctx, text)
}

// EmbedBatch generates embeddings for multiple texts in a single API call.
// The returned slice preserves the order of the input texts.
func (g *EmbeddingGenerator) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	return g.callAPIBatch(ctx, texts)
}

// EmbedWithRetry embeds a single text with automatic exponential-backoff
// retry on transient failures.
func (g *EmbeddingGenerator) EmbedWithRetry(ctx context.Context, text string, maxRetries int) ([]float64, error) {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		emb, err := g.Embed(ctx, text)
		if err == nil {
			return emb, nil
		}
		lastErr = err
		if attempt < maxRetries {
			sleep := time.Duration(math.Pow(2, float64(attempt))) * time.Second
			fmt.Fprintf(os.Stderr, "embedding attempt %d/%d failed: %v. Retrying in %v…\n",
				attempt+1, maxRetries, err, sleep)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(sleep):
			}
		}
	}
	return nil, lastErr
}

// ---------------------------------------------------------------------------
// EmbeddingGenerator — internal HTTP helpers
// ---------------------------------------------------------------------------

func (g *EmbeddingGenerator) callAPI(ctx context.Context, input any) ([]float64, error) {
	results, err := g.callAPIBatch(ctx, []string{fmt.Sprintf("%v", input)})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("empty response from embeddings API")
	}
	return results[0], nil
}

func (g *EmbeddingGenerator) callAPIBatch(ctx context.Context, texts []string) ([][]float64, error) {
	req := embeddingRequest{
		Model: g.Model,
		Input: texts,
	}
	if g.Dimensions > 0 {
		d := g.Dimensions
		req.Dimensions = &d
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		"https://api.openai.com/v1/embeddings",
		strings.NewReader(string(body)),
	)
	if err != nil {
		return nil, fmt.Errorf("create HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+g.apiKey)

	resp, err := g.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errBody openAIErrorBody
		_ = json.NewDecoder(resp.Body).Decode(&errBody)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, errBody.Error.Message)
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	// Sort by index to preserve input order.
	sort.Slice(result.Data, func(i, j int) bool {
		return result.Data[i].Index < result.Data[j].Index
	})

	embeddings := make([][]float64, len(result.Data))
	for i, obj := range result.Data {
		embeddings[i] = obj.Embedding
	}
	return embeddings, nil
}

// ---------------------------------------------------------------------------
// EmbeddingComparator
// ---------------------------------------------------------------------------

// EmbeddingComparator compares embeddings and finds semantically similar texts.
type EmbeddingComparator struct{}

// SimilarityResult holds a candidate text and its similarity score.
type SimilarityResult struct {
	Text  string
	Score float64
}

// CosineSimilarity calculates the cosine similarity between two embedding vectors.
// Returns a value in [-1, 1]; closer to 1 means more similar.
func (c *EmbeddingComparator) CosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector length mismatch: %d vs %d", len(a), len(b))
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0, nil
	}
	return dot / denom, nil
}

// EuclideanDistance calculates the Euclidean distance between two embedding vectors.
// Smaller values indicate more similar texts.
func (c *EmbeddingComparator) EuclideanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector length mismatch: %d vs %d", len(a), len(b))
	}
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum), nil
}

// FindMostSimilar ranks candidate texts by semantic similarity to query.
// It embeds the query and all candidates in a single batch call for efficiency.
func (c *EmbeddingComparator) FindMostSimilar(
	ctx context.Context,
	query string,
	candidates []string,
	gen *EmbeddingGenerator,
) ([]SimilarityResult, error) {
	all := append([]string{query}, candidates...)
	embeddings, err := gen.EmbedBatch(ctx, all)
	if err != nil {
		return nil, err
	}
	queryEmb := embeddings[0]
	results := make([]SimilarityResult, len(candidates))
	for i, text := range candidates {
		score, err := c.CosineSimilarity(queryEmb, embeddings[i+1])
		if err != nil {
			return nil, err
		}
		results[i] = SimilarityResult{Text: text, Score: score}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return results, nil
}

// VisualizeSimilarity returns a similarity matrix formatted as ASCII art.
func (c *EmbeddingComparator) VisualizeSimilarity(
	ctx context.Context,
	texts []string,
	gen *EmbeddingGenerator,
) (string, error) {
	embeddings, err := gen.EmbedBatch(ctx, texts)
	if err != nil {
		return "", err
	}
	n := len(texts)
	labels := make([]string, n)
	for i := range texts {
		labels[i] = fmt.Sprintf("Text%d", i+1)
	}
	labelW := 0
	for _, l := range labels {
		if len(l) > labelW {
			labelW = len(l)
		}
	}
	labelW += 2
	colW := 7

	var sb strings.Builder
	sb.WriteString(strings.Repeat(" ", labelW))
	for _, l := range labels {
		sb.WriteString(fmt.Sprintf("%*s", colW, l))
	}
	sb.WriteByte('\n')

	for i := range texts {
		sb.WriteString(fmt.Sprintf("%-*s", labelW, labels[i]))
		for j := range texts {
			score, _ := c.CosineSimilarity(embeddings[i], embeddings[j])
			sb.WriteString(fmt.Sprintf("%*.2f", colW, score))
		}
		sb.WriteByte('\n')
	}
	return sb.String(), nil
}
