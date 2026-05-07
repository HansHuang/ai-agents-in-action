// Hybrid RAG agent in Go: native document loading, Qdrant for vectors,
// custom code for agent logic.
//
// Architecture:
//
//	NATIVE GO (commodity parts):
//	  • filepath.WalkDir  — load .md / .txt files from a directory
//	  • OpenAI embeddings API — embed documents and queries
//	  • Qdrant Go SDK     — vector storage and ANN search
//	    (or SimpleVectorStore — pure-Go fallback, no external DB needed)
//
//	YOUR CODE (differentiated parts):
//	  • Agent loop        — you control orchestration
//	  • ContextAssembler  — you control prompt quality
//	  • MemoryManager     — you control conversation budget
//	  • TokenTracker      — you control cost visibility
//
// Go framework situation (May 2026):
//   - LangChain Go (github.com/tmc/langchaingo) exists but lags the Python
//     version by 6-12 months and has a smaller community.
//   - Most Go teams build from scratch using:
//   - github.com/sashabaranov/go-openai  — OpenAI SDK
//   - Qdrant Go SDK / Weaviate Go client  — vector search
//   - Standard library                   — document loading
//   - This is an ADVANTAGE: Go agents are typically cleaner and more
//     maintainable than Python teams that over-relied on frameworks.
//
// Run:
//
//	go run hybrid_rag_agent.go [docs-directory]
//
// See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	llmModel   = openai.GPT4oMini
	embedModel = openai.SmallEmbedding3
	maxTurns   = 5
)

// ---------------------------------------------------------------------------
// Inline demo documents
// ---------------------------------------------------------------------------

var inlineDocs = []rawDoc{
	{
		Text:   "RAG has four phases: Ingest (load, chunk, embed, store), Retrieve (embed query, search), Augment (build prompt), Generate (call LLM with augmented prompt).",
		Source: "rag_phases.md",
	},
	{
		Text:   "Vector databases store embeddings and support approximate nearest-neighbour search. Popular choices: Qdrant, Pinecone, Chroma, FAISS, Weaviate.",
		Source: "vector_databases.md",
	},
	{
		Text:   "An agent loop: perceive, think (LLM call), act (tool execution), observe. Repeat until a final answer is produced.",
		Source: "agent_loop.md",
	},
	{
		Text:   "LangChain is an open-source framework with 700+ integrations. The API stabilised significantly since 2024.",
		Source: "langchain.md",
	},
	{
		Text:   "Python was created by Guido van Rossum in 1991. Python 3.0 broke backward compatibility with Python 2.",
		Source: "python_history.md",
	},
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

// rawDoc is a plain document before embedding.
type rawDoc struct {
	Text   string
	Source string
}

// storedDoc is a document with its embedding vector.
type storedDoc struct {
	Text      string
	Source    string
	Embedding []float32
}

// HybridResult is returned by HybridRAGAgent.Query.
type HybridResult struct {
	Answer       string
	Sources      []string
	Tokens       int
	RetrievalMs  float64
	GenerationMs float64
}

// chatMessage mirrors the OpenAI chat message structure.
type chatMessage struct {
	Role    string
	Content string
}

// tokenUsage accumulates prompt and completion tokens.
type tokenUsage struct {
	PromptTokens     int
	CompletionTokens int
}

func (t tokenUsage) Total() int { return t.PromptTokens + t.CompletionTokens }

// ---------------------------------------------------------------------------
// VectorStoreProtocol
//
// In Go, protocols are expressed as interfaces.  Both SimpleVectorStore and
// any Qdrant wrapper satisfy this interface, making the swap transparent.
// ---------------------------------------------------------------------------

// VectorStore is the minimal interface the agent depends on.
// Satisfying it requires only SimilaritySearch and BackendName.
type VectorStore interface {
	SimilaritySearch(ctx context.Context, query string, k int) ([]storedDoc, error)
	BackendName() string
}

// ---------------------------------------------------------------------------
// SimpleVectorStore — pure-Go, brute-force cosine similarity
// ---------------------------------------------------------------------------

// SimpleVectorStore is an in-memory vector store for prototyping.
// No external dependency required — just the OpenAI embeddings API.
type SimpleVectorStore struct {
	client *openai.Client
	docs   []storedDoc
}

// NewSimpleVectorStore creates an empty store.
func NewSimpleVectorStore(client *openai.Client) *SimpleVectorStore {
	return &SimpleVectorStore{client: client}
}

// AddDocuments embeds and stores a batch of raw documents.
func (s *SimpleVectorStore) AddDocuments(ctx context.Context, docs []rawDoc) error {
	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.Text
	}

	resp, err := s.client.CreateEmbeddings(ctx, openai.EmbeddingRequestStrings{
		Input: texts,
		Model: embedModel,
	})
	if err != nil {
		return fmt.Errorf("embedding docs: %w", err)
	}

	for i, d := range docs {
		s.docs = append(s.docs, storedDoc{
			Text:      d.Text,
			Source:    d.Source,
			Embedding: resp.Data[i].Embedding,
		})
	}
	return nil
}

// SimilaritySearch returns the top-k most similar documents to query.
func (s *SimpleVectorStore) SimilaritySearch(ctx context.Context, query string, k int) ([]storedDoc, error) {
	if len(s.docs) == 0 {
		return nil, nil
	}

	resp, err := s.client.CreateEmbeddings(ctx, openai.EmbeddingRequestStrings{
		Input: []string{query},
		Model: embedModel,
	})
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	qEmb := resp.Data[0].Embedding

	type scored struct {
		doc   storedDoc
		score float64
	}
	results := make([]scored, len(s.docs))
	for i, d := range s.docs {
		results[i] = scored{doc: d, score: cosineSimilarity(qEmb, d.Embedding)}
	}

	// Partial sort: bubble top-k to front
	topK := min(k, len(results))
	for i := 0; i < topK; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	out := make([]storedDoc, topK)
	for i := range topK {
		out[i] = results[i].doc
	}
	return out, nil
}

// BackendName satisfies the VectorStore interface.
func (s *SimpleVectorStore) BackendName() string { return "SimpleVectorStore" }

// ---------------------------------------------------------------------------
// QdrantVectorStore stub
//
// A real implementation would use github.com/qdrant/go-client.
// This stub demonstrates the swap pattern — swapping it in requires only
// calling agent.SwapVectorStore(qdrantStore).  Zero logic changes.
// ---------------------------------------------------------------------------

// QdrantVectorStore is a placeholder for a production Qdrant-backed store.
// Uncomment and implement when you need production-scale vector search.
//
//	type QdrantVectorStore struct {
//	    client     *qdrant.Client
//	    collection string
//	    embedClient *openai.Client
//	}
//
//	func (q *QdrantVectorStore) SimilaritySearch(ctx context.Context, query string, k int) ([]storedDoc, error) { ... }
//	func (q *QdrantVectorStore) BackendName() string { return "Qdrant" }

// ---------------------------------------------------------------------------
// ContextAssembler — YOUR logic, no framework dependency
// ---------------------------------------------------------------------------

// ContextAssembler builds system prompts from retrieved documents.
type ContextAssembler struct{}

// Assemble returns a fully-formed system prompt for the given documents.
func (a *ContextAssembler) Assemble(docs []storedDoc) string {
	if len(docs) == 0 {
		return "Answer the question using ONLY the documents below.\n\n(no documents retrieved)"
	}
	var sb strings.Builder
	sb.WriteString("Answer the question using ONLY the documents below.\n")
	sb.WriteString("If the answer is not in the documents, say so explicitly.\n")
	sb.WriteString("Cite sources with [Source: filename].\n\n")
	for _, d := range docs {
		sb.WriteString(fmt.Sprintf("[Source: %s]\n%s\n\n", d.Source, d.Text))
	}
	return strings.TrimRight(sb.String(), "\n")
}

// ---------------------------------------------------------------------------
// MemoryManager — YOUR logic, no framework dependency
// ---------------------------------------------------------------------------

// MemoryManager maintains a sliding window of conversation history.
type MemoryManager struct {
	history  []chatMessage
	maxTurns int
}

// NewMemoryManager creates a manager capped at maxTurns user+assistant pairs.
func NewMemoryManager(maxTurns int) *MemoryManager {
	return &MemoryManager{maxTurns: maxTurns}
}

// GetMessages returns the full message list for the next LLM call.
func (m *MemoryManager) GetMessages(systemPrompt, userMessage string) []chatMessage {
	start := len(m.history) - m.maxTurns*2
	if start < 0 {
		start = 0
	}
	msgs := make([]chatMessage, 0, len(m.history[start:])+2)
	msgs = append(msgs, chatMessage{Role: "system", Content: systemPrompt})
	msgs = append(msgs, m.history[start:]...)
	msgs = append(msgs, chatMessage{Role: "user", Content: userMessage})
	return msgs
}

// Record appends a user+assistant turn to history.
func (m *MemoryManager) Record(userMsg, assistantMsg string) {
	m.history = append(m.history,
		chatMessage{Role: "user", Content: userMsg},
		chatMessage{Role: "assistant", Content: assistantMsg},
	)
}

// ---------------------------------------------------------------------------
// TokenTracker — YOUR logic, no framework dependency
// ---------------------------------------------------------------------------

// TokenTracker accumulates token usage across all LLM calls in a session.
type TokenTracker struct {
	usage tokenUsage
	calls int
}

// Record adds usage from one LLM call.
func (t *TokenTracker) Record(prompt, completion int) {
	t.usage.PromptTokens += prompt
	t.usage.CompletionTokens += completion
	t.calls++
}

// Totals returns accumulated usage.
func (t *TokenTracker) Totals() tokenUsage { return t.usage }

// CallCount returns the number of LLM calls recorded.
func (t *TokenTracker) CallCount() int { return t.calls }

// ---------------------------------------------------------------------------
// HybridRAGAgent
// ---------------------------------------------------------------------------

// HybridRAGAgent uses native Go document loading, OpenAI embeddings, and a
// swappable vector store backend.  The agent loop, context assembly, memory
// management, and token tracking are all custom Go code.
type HybridRAGAgent struct {
	client *openai.Client

	// Your code: the important parts
	contextAssembler *ContextAssembler
	memoryManager    *MemoryManager
	tokenTracker     *TokenTracker

	// Swappable backend (satisfies VectorStore)
	vectorStore   VectorStore
	ingestedCount int
}

// NewHybridRAGAgent creates an agent that uses SimpleVectorStore by default.
func NewHybridRAGAgent(client *openai.Client) *HybridRAGAgent {
	return &HybridRAGAgent{
		client:           client,
		contextAssembler: &ContextAssembler{},
		memoryManager:    NewMemoryManager(maxTurns),
		tokenTracker:     &TokenTracker{},
	}
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

// Ingest embeds and stores the supplied documents.
// Pass nil to use the inline demo documents.
func (a *HybridRAGAgent) Ingest(ctx context.Context, docs []rawDoc) (int, error) {
	if docs == nil {
		docs = inlineDocs
	}
	store := NewSimpleVectorStore(a.client)
	if err := store.AddDocuments(ctx, docs); err != nil {
		return 0, fmt.Errorf("ingest: %w", err)
	}
	a.vectorStore = store
	a.ingestedCount = len(docs)
	return a.ingestedCount, nil
}

// IngestDirectory loads all .md and .txt files from dir and ingests them.
func (a *HybridRAGAgent) IngestDirectory(ctx context.Context, dir string) (int, error) {
	var docs []rawDoc
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		ext := strings.ToLower(filepath.Ext(path))
		if d.IsDir() || (ext != ".md" && ext != ".txt") {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		docs = append(docs, rawDoc{
			Text:   string(data),
			Source: filepath.Base(path),
		})
		return nil
	})
	if err != nil {
		return 0, fmt.Errorf("walk dir: %w", err)
	}
	if len(docs) == 0 {
		return a.Ingest(ctx, nil) // fall back to inline docs
	}
	return a.Ingest(ctx, docs)
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

// Query runs the full RAG pipeline for question.
// Your code controls every step.  The vector store is just a tool.
func (a *HybridRAGAgent) Query(ctx context.Context, question string) (*HybridResult, error) {
	if a.vectorStore == nil {
		return nil, fmt.Errorf("call Ingest before Query")
	}

	// 1. Retrieve
	t0Ret := time.Now()
	retrieved, err := a.vectorStore.SimilaritySearch(ctx, question, 5)
	if err != nil {
		return nil, fmt.Errorf("retrieval: %w", err)
	}
	retrievalMs := float64(time.Since(t0Ret).Microseconds()) / 1000

	// 2. Your context assembly logic
	systemPrompt := a.contextAssembler.Assemble(retrieved)

	// 3. Your memory management
	msgs := a.memoryManager.GetMessages(systemPrompt, question)

	// 4. Your LLM call
	apiMsgs := make([]openai.ChatCompletionMessage, len(msgs))
	for i, m := range msgs {
		apiMsgs[i] = openai.ChatCompletionMessage{Role: m.Role, Content: m.Content}
	}
	t0Gen := time.Now()
	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    llmModel,
		Messages: apiMsgs,
	})
	if err != nil {
		return nil, fmt.Errorf("LLM call: %w", err)
	}
	generationMs := float64(time.Since(t0Gen).Microseconds()) / 1000

	answer := resp.Choices[0].Message.Content

	// 5. Your token tracking
	a.tokenTracker.Record(resp.Usage.PromptTokens, resp.Usage.CompletionTokens)

	// Record this turn in memory
	a.memoryManager.Record(question, answer)

	// Collect unique sources
	sourceSet := map[string]struct{}{}
	for _, d := range retrieved {
		sourceSet[d.Source] = struct{}{}
	}
	sources := make([]string, 0, len(sourceSet))
	for s := range sourceSet {
		sources = append(sources, s)
	}

	return &HybridResult{
		Answer:       answer,
		Sources:      sources,
		Tokens:       resp.Usage.TotalTokens,
		RetrievalMs:  retrievalMs,
		GenerationMs: generationMs,
	}, nil
}

// ---------------------------------------------------------------------------
// SwapVectorStore
// ---------------------------------------------------------------------------

// SwapVectorStore replaces the backend with zero logic changes to the agent.
func (a *HybridRAGAgent) SwapVectorStore(vs VectorStore) {
	a.vectorStore = vs
}

// VectorBackend returns the name of the current vector store backend.
func (a *HybridRAGAgent) VectorBackend() string {
	if a.vectorStore == nil {
		return "none"
	}
	return a.vectorStore.BackendName()
}

// TokenSummary returns accumulated token usage.
func (a *HybridRAGAgent) TokenSummary() tokenUsage { return a.tokenTracker.Totals() }

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ---------------------------------------------------------------------------
// main (demo)
// ---------------------------------------------------------------------------

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY not set")
		os.Exit(1)
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()

	fmt.Println()
	fmt.Println(strings.Repeat("=", 65))
	fmt.Println("  HYBRID RAG AGENT DEMO (Go)")
	fmt.Println(strings.Repeat("=", 65))

	// ---- Phase 1: Ingest -----------------------------------------------
	fmt.Printf("\n[1] Ingesting %d inline documents (SimpleVectorStore)...\n", len(inlineDocs))
	agent := NewHybridRAGAgent(client)
	n, err := agent.Ingest(ctx, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ingest error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("    Ingested %d documents. Backend: %s\n", n, agent.VectorBackend())

	// ---- Phase 2: Query ------------------------------------------------
	question := "What are the four phases of a RAG pipeline?"
	fmt.Printf("\n[2] Query: '%s'\n", question)
	result, err := agent.Query(ctx, question)
	if err != nil {
		fmt.Fprintf(os.Stderr, "query error: %v\n", err)
		os.Exit(1)
	}
	answerPreview := result.Answer
	if len(answerPreview) > 120 {
		answerPreview = answerPreview[:120] + "..."
	}
	fmt.Printf("    Answer: %s\n", answerPreview)
	fmt.Printf("    Sources: %s\n", strings.Join(result.Sources, ", "))
	fmt.Printf("    Tokens: %d  |  Retrieval: %.0fms  |  Generation: %.0fms\n",
		result.Tokens, result.RetrievalMs, result.GenerationMs)

	// ---- Phase 3: Swap vector store (simulating Qdrant) ----------------
	fmt.Println("\n[3] Swapping vector store: SimpleVectorStore → (another) SimpleVectorStore")
	fmt.Println("    (In production, swap for a QdrantVectorStore — same interface)")
	newStore := NewSimpleVectorStore(client)
	if err := newStore.AddDocuments(ctx, inlineDocs); err != nil {
		fmt.Fprintf(os.Stderr, "re-ingest error: %v\n", err)
		os.Exit(1)
	}
	agent.SwapVectorStore(newStore)
	fmt.Printf("    Backend is now: %s\n", agent.VectorBackend())

	fmt.Printf("\n[4] Same query after swap: '%s'\n", question)
	result2, err := agent.Query(ctx, question)
	if err != nil {
		fmt.Fprintf(os.Stderr, "query error: %v\n", err)
		os.Exit(1)
	}
	preview2 := result2.Answer
	if len(preview2) > 120 {
		preview2 = preview2[:120] + "..."
	}
	fmt.Printf("    Answer: %s\n", preview2)
	fmt.Println("    Agent logic unchanged — only storage backend changed.")

	// ---- Phase 4: Where Go developers turn for framework-like help -----
	fmt.Println("\n[5] Go framework landscape (May 2026):")
	fmt.Println("    • github.com/sashabaranov/go-openai  — OpenAI SDK (official-quality)")
	fmt.Println("    • github.com/qdrant/go-client         — Qdrant vector DB")
	fmt.Println("    • github.com/weaviate/weaviate-go-client — Weaviate vector DB")
	fmt.Println("    • github.com/tmc/langchaingo           — LangChain Go (limited, use carefully)")
	fmt.Println("    • Standard library (os, filepath, bufio) — document loading")
	fmt.Println("    Go agents are typically cleaner than over-framework'd Python agents.")

	fmt.Printf("\nTotal tokens this session: %d\n", agent.TokenSummary().Total())
	fmt.Println(strings.Repeat("=", 65))
}
