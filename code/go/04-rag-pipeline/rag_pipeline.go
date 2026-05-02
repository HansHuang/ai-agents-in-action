// RAGPipeline — Complete Retrieval-Augmented Generation pipeline (Go port).
//
// Four-phase pipeline:
//
//	INGEST   — Load → Chunk → Embed → Store
//	RETRIEVE — Embed query → Search → Filter by threshold
//	AUGMENT  — Build prompt with retrieved context
//	GENERATE — Call LLM with augmented prompt
//
// See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
package ragpipeline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

// ---------------------------------------------------------------------------
// Prompt template
// ---------------------------------------------------------------------------

const ragSystemPromptTemplate = `You are a helpful assistant that answers questions based on the provided documents.

Rules:
1. Answer ONLY using information from the documents below.
2. If the documents don't contain the answer, say exactly: "I don't have information about that in my knowledge base."
3. Cite sources using [Source: filename] format.
4. If multiple documents are relevant, synthesize information from all of them.
5. If documents contain conflicting information, note the conflict and cite both sources.
6. Do not use any knowledge outside the provided documents.

Documents:
%s

When answering, structure your response as:
1. Direct answer to the question
2. Supporting details from the documents
3. Source citations`

const citationSuffix = "\n\nIMPORTANT: You MUST cite every factual claim with [Source: <filename>]. Do not make any statement without an explicit citation."

var supportedExtensions = map[string]bool{
	".txt":  true,
	".md":   true,
	".rst":  true,
	".text": true,
}

// ---------------------------------------------------------------------------
// OpenAI chat types
// ---------------------------------------------------------------------------

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
}

type chatChoice struct {
	Message chatMessage `json:"message"`
}

type chatUsage struct {
	TotalTokens int `json:"total_tokens"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

// ---------------------------------------------------------------------------
// RAGResponse
// ---------------------------------------------------------------------------

// RAGResponse is the structured result returned by RAGPipeline.Query.
type RAGResponse struct {
	Answer           string
	Sources          []string
	RetrievedChunks  []RetrievedChunk
	SimilarityScores []float64
	TokensUsed       int
	PipelineSteps    []string
}

// RetrievedChunk holds a single retrieved document chunk with its score.
type RetrievedChunk struct {
	Text     string
	Score    float64
	Metadata map[string]interface{}
}

// IngestResult is returned by IngestDirectory.
type IngestResult struct {
	DocumentsProcessed int
	ChunksCreated      int
	Errors             []string
}

// ---------------------------------------------------------------------------
// RAGPipeline
// ---------------------------------------------------------------------------

// RAGPipeline is the complete RAG pipeline.
//
// Create with NewRAGPipeline; do not use the zero value.
type RAGPipeline struct {
	VectorStore         *SimpleVectorStore
	Embedder            *EmbeddingGenerator
	Model               string
	ChunkSize           int // approximate token count per chunk (1 token ≈ 4 chars)
	Overlap             int // token overlap
	RetrievalK          int
	SimilarityThreshold float64

	apiKey string
	client *http.Client
}

// NewRAGPipeline creates a RAGPipeline that reads OPENAI_API_KEY from the environment.
func NewRAGPipeline(
	vectorStore *SimpleVectorStore,
	embedder *EmbeddingGenerator,
	model string,
	chunkSize, overlap, retrievalK int,
	similarityThreshold float64,
) *RAGPipeline {
	return &RAGPipeline{
		VectorStore:         vectorStore,
		Embedder:            embedder,
		Model:               model,
		ChunkSize:           chunkSize,
		Overlap:             overlap,
		RetrievalK:          retrievalK,
		SimilarityThreshold: similarityThreshold,
		apiKey:              os.Getenv("OPENAI_API_KEY"),
		client:              &http.Client{Timeout: 60 * time.Second},
	}
}

// ---------------------------------------------------------------------------
// Phase 1 — Ingest
// ---------------------------------------------------------------------------

// IngestDirectory loads, chunks, embeds, and stores all documents in dir.
func (p *RAGPipeline) IngestDirectory(ctx context.Context, dir string) (IngestResult, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return IngestResult{}, fmt.Errorf("read directory %q: %w", dir, err)
	}

	var result IngestResult
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if !supportedExtensions[ext] {
			continue
		}
		filePath := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(filePath)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", entry.Name(), err))
			continue
		}
		n, err := p.IngestText(ctx, string(data), map[string]interface{}{
			"source": entry.Name(),
			"path":   filePath,
		})
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", entry.Name(), err))
			continue
		}
		result.DocumentsProcessed++
		result.ChunksCreated += n
	}
	return result, nil
}

// IngestText chunks, embeds, and stores a single document.
// Returns the number of chunks created.
func (p *RAGPipeline) IngestText(ctx context.Context, text string, metadata map[string]interface{}) (int, error) {
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	chunks := chunkText(text, p.ChunkSize, p.Overlap)
	if len(chunks) == 0 {
		return 0, nil
	}

	embeddings, err := p.Embedder.EmbedBatch(ctx, chunks)
	if err != nil {
		return 0, fmt.Errorf("embed batch: %w", err)
	}

	for i, chunk := range chunks {
		chunkMeta := copyMetadata(metadata)
		chunkMeta["chunk_index"] = i
		chunkMeta["total_chunks"] = len(chunks)
		if _, err := p.VectorStore.Add(chunk, embeddings[i], chunkMeta); err != nil {
			return i, fmt.Errorf("store chunk %d: %w", i, err)
		}
	}
	return len(chunks), nil
}

// ---------------------------------------------------------------------------
// Phase 2 — Retrieve
// ---------------------------------------------------------------------------

func (p *RAGPipeline) retrieve(
	ctx context.Context,
	question string,
	k int,
	threshold float64,
	steps *[]string,
) ([]VectorSearchResult, error) {
	*steps = append(*steps, "RETRIEVE: embedding query")
	queryEmb, err := p.Embedder.Embed(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	*steps = append(*steps, fmt.Sprintf("RETRIEVE: searching vector store (k=%d, threshold=%.2f)", k, threshold))
	results, err := p.VectorStore.SearchWithThreshold(queryEmb, threshold, k)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}
	*steps = append(*steps, fmt.Sprintf("RETRIEVE: found %d chunk(s) above threshold", len(results)))
	return results, nil
}

// ---------------------------------------------------------------------------
// Phase 3 — Augment
// ---------------------------------------------------------------------------

func (p *RAGPipeline) buildMessages(question string, results []VectorSearchResult, extraSuffix string) []chatMessage {
	parts := make([]string, len(results))
	for i, r := range results {
		source := "unknown"
		if s, ok := r.Metadata["source"].(string); ok {
			source = s
		}
		parts[i] = fmt.Sprintf("[Document %d — Source: %s]\n%s", i+1, source, r.Text)
	}
	documentContext := strings.Join(parts, "\n\n---\n\n")
	systemPrompt := fmt.Sprintf(ragSystemPromptTemplate, documentContext) + extraSuffix
	return []chatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: question},
	}
}

// ---------------------------------------------------------------------------
// Phase 4 — Generate
// ---------------------------------------------------------------------------

func (p *RAGPipeline) generate(ctx context.Context, messages []chatMessage, steps *[]string) (string, int, error) {
	*steps = append(*steps, fmt.Sprintf("GENERATE: calling %s", p.Model))

	reqBody := chatRequest{
		Model:       p.Model,
		Messages:    messages,
		Temperature: 0.3,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.openai.com/v1/chat/completions",
		bytes.NewReader(bodyBytes),
	)
	if err != nil {
		return "", 0, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBytes))
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBytes, &chatResp); err != nil {
		return "", 0, fmt.Errorf("unmarshal response: %w", err)
	}
	if len(chatResp.Choices) == 0 {
		return "", 0, fmt.Errorf("no choices in response")
	}

	answer := chatResp.Choices[0].Message.Content
	tokens := chatResp.Usage.TotalTokens
	*steps = append(*steps, fmt.Sprintf("GENERATE: received %d total tokens", tokens))
	return answer, tokens, nil
}

// ---------------------------------------------------------------------------
// Public query interface
// ---------------------------------------------------------------------------

// Query answers a question using the RAG pipeline.
func (p *RAGPipeline) Query(ctx context.Context, question string, k int, threshold float64) (*RAGResponse, error) {
	if k <= 0 {
		k = p.RetrievalK
	}
	if threshold <= 0 {
		threshold = p.SimilarityThreshold
	}
	steps := []string{"INGEST: (already complete)"}

	results, err := p.retrieve(ctx, question, k, threshold, &steps)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		steps = append(steps, "AUGMENT: no results above threshold — short-circuit")
		return &RAGResponse{
			Answer:        "I don't have information about that in my knowledge base.",
			PipelineSteps: steps,
		}, nil
	}

	steps = append(steps, fmt.Sprintf("AUGMENT: building prompt with %d document(s)", len(results)))
	messages := p.buildMessages(question, results, "")
	answer, tokens, err := p.generate(ctx, messages, &steps)
	if err != nil {
		return nil, err
	}

	return p.buildResponse(answer, tokens, results, steps), nil
}

// QueryWithCitations answers a question and enforces citation format.
func (p *RAGPipeline) QueryWithCitations(ctx context.Context, question string, k int, threshold float64) (*RAGResponse, error) {
	if k <= 0 {
		k = p.RetrievalK
	}
	if threshold <= 0 {
		threshold = p.SimilarityThreshold
	}
	steps := []string{"INGEST: (already complete)"}

	results, err := p.retrieve(ctx, question, k, threshold, &steps)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		steps = append(steps, "AUGMENT: no results above threshold — short-circuit")
		return &RAGResponse{
			Answer:        "I don't have information about that in my knowledge base.",
			PipelineSteps: steps,
		}, nil
	}

	steps = append(steps, fmt.Sprintf("AUGMENT: building citation-enforced prompt with %d document(s)", len(results)))
	messages := p.buildMessages(question, results, citationSuffix)
	answer, tokens, err := p.generate(ctx, messages, &steps)
	if err != nil {
		return nil, err
	}

	return p.buildResponse(answer, tokens, results, steps), nil
}

// RemoveDocument removes all chunks whose metadata["source"] matches sourceID.
// Returns the number of chunks removed.
func (p *RAGPipeline) RemoveDocument(sourceID string) int {
	var toRemove []string
	for _, doc := range p.VectorStore.documents {
		if s, ok := doc.Metadata["source"].(string); ok && s == sourceID {
			toRemove = append(toRemove, doc.ID)
		}
	}
	for _, id := range toRemove {
		p.VectorStore.Delete(id)
	}
	return len(toRemove)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func (p *RAGPipeline) buildResponse(
	answer string,
	tokens int,
	results []VectorSearchResult,
	steps []string,
) *RAGResponse {
	// Deduplicated sources in order
	seen := map[string]bool{}
	var sources []string
	chunks := make([]RetrievedChunk, len(results))
	scores := make([]float64, len(results))
	for i, r := range results {
		src := "unknown"
		if s, ok := r.Metadata["source"].(string); ok {
			src = s
		}
		if !seen[src] {
			seen[src] = true
			sources = append(sources, src)
		}
		chunks[i] = RetrievedChunk{Text: r.Text, Score: r.Score, Metadata: r.Metadata}
		scores[i] = r.Score
	}
	return &RAGResponse{
		Answer:           answer,
		Sources:          sources,
		RetrievedChunks:  chunks,
		SimilarityScores: scores,
		TokensUsed:       tokens,
		PipelineSteps:    steps,
	}
}

// chunkText splits text into overlapping chunks of approximately chunkSize tokens.
// 1 token ≈ 4 UTF-8 characters; this is a heuristic only.
func chunkText(text string, chunkSize, overlap int) []string {
	charSize := chunkSize * 4
	charStep := (chunkSize - overlap) * 4
	if charStep < 1 {
		charStep = 1
	}

	var chunks []string
	runes := []rune(text)
	_ = utf8.RuneCountInString(text) // satisfy import

	start := 0
	for start < len(runes) {
		end := start + charSize
		if end > len(runes) {
			end = len(runes)
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		start += charStep
	}
	return chunks
}

func copyMetadata(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{}, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}
