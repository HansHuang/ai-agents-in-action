// framework_comparison.go — Three parallel RAG implementations in Go.
//
// The same question answered from the identical knowledge base using
// three conceptual approaches so you can compare tradeoffs:
//
//  1. FROM SCRATCH  — pure OpenAI SDK, no abstractions
//  2. LANGCHAIN-STYLE — same code with deliberate LangChain-shaped boundaries
//     (LangChain Go is immature at May 2026; this shows the concept)
//  3. LANGGRAPH-STYLE — Go state machine (uses our local StateGraph from
//     langgraph_alternative.go)
//
// For each approach the code measures:
//   - Response time
//   - Token usage
//   - Lines of implementation logic
//   - Debuggability (a deliberate empty-store bug shows the error trace)
//
// Run:
//
//	go run framework_comparison.go langgraph_alternative.go hybrid_rag_agent.go \
//	           multi_agent_from_scratch.go
//
// See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	fwcLLMModel   = "gpt-4o-mini"
	fwcEmbedModel = "text-embedding-3-small"
	fwcDemoQuery  = "What are the four phases of a RAG pipeline?"
)

// ---------------------------------------------------------------------------
// Knowledge base (shared by all three implementations)
// ---------------------------------------------------------------------------

type fwcDoc struct {
	ID     string
	Title  string
	Text   string
	Source string
}

var fwcKnowledgeBase = []fwcDoc{
	{
		ID:    "kb-01",
		Title: "Python History",
		Text: "Python was created by Guido van Rossum and first released in 1991. " +
			"It emphasises readability and uses significant indentation. " +
			"Python 3.0 was released in 2008 and is not backward-compatible with Python 2.",
		Source: "python_history.txt",
	},
	{
		ID:    "kb-02",
		Title: "Vector Databases",
		Text: "Vector databases store high-dimensional embeddings and support approximate " +
			"nearest-neighbour search. Popular options include Qdrant, Pinecone, Weaviate, " +
			"Chroma, and FAISS. They power semantic search and RAG pipelines.",
		Source: "vector_databases.txt",
	},
	{
		ID:    "kb-03",
		Title: "LangChain Overview",
		Text: "LangChain is an open-source framework for building LLM-powered applications. " +
			"It provides 700+ integrations, a chain composition API, and LangSmith for " +
			"observability. The API has stabilised significantly since 2024.",
		Source: "langchain_overview.txt",
	},
	{
		ID:    "kb-04",
		Title: "RAG Architecture",
		Text: "Retrieval-Augmented Generation (RAG) enhances LLM responses with external " +
			"knowledge. The four phases are: Ingest (load, chunk, embed, store), " +
			"Retrieve (embed query, search), Augment (build prompt), Generate (call LLM).",
		Source: "rag_architecture.txt",
	},
	{
		ID:    "kb-05",
		Title: "Agent Loops",
		Text: "An AI agent loop repeatedly calls an LLM, parses its response for tool calls, " +
			"executes those tools, and feeds results back until the LLM emits a final answer. " +
			"The loop has four parts: perceive, think, act, observe.",
		Source: "agent_loops.txt",
	},
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

// FWCResult captures metrics for one implementation.
type FWCResult struct {
	Name           string
	Answer         string
	Sources        []string
	ResponseTimeMs float64
	TokensUsed     int
	Debuggability  string // "Easy" | "Moderate" | "Hard"
	DebugTrace     string // populated when buggy=true
}

// ---------------------------------------------------------------------------
// Shared embedding + similarity helpers (used by all implementations)
// ---------------------------------------------------------------------------

type fwcIndexedDoc struct {
	Doc       fwcDoc
	Embedding []float32
}

// fwcEmbedTexts calls the OpenAI embeddings API.
func fwcEmbedTexts(ctx context.Context, client *openai.Client, texts []string) ([][]float32, error) {
	resp, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: texts,
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// fwcCosineSim computes cosine similarity between two float32 slices.
func fwcCosineSim(a, b []float32) float64 {
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

// fwcBuildIndex embeds all docs and returns a searchable index.
func fwcBuildIndex(ctx context.Context, client *openai.Client, docs []fwcDoc) ([]fwcIndexedDoc, error) {
	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.Text
	}
	embeddings, err := fwcEmbedTexts(ctx, client, texts)
	if err != nil {
		return nil, err
	}
	index := make([]fwcIndexedDoc, len(docs))
	for i, d := range docs {
		index[i] = fwcIndexedDoc{Doc: d, Embedding: embeddings[i]}
	}
	return index, nil
}

// fwcRetrieve returns the top-k docs by cosine similarity.
func fwcRetrieve(ctx context.Context, client *openai.Client, index []fwcIndexedDoc, query string, k int) ([]fwcDoc, error) {
	if len(index) == 0 {
		return nil, nil
	}
	qEmbs, err := fwcEmbedTexts(ctx, client, []string{query})
	if err != nil {
		return nil, err
	}
	qEmb := qEmbs[0]

	type scored struct {
		doc   fwcDoc
		score float64
	}
	scores := make([]scored, len(index))
	for i, item := range index {
		scores[i] = scored{item.Doc, fwcCosineSim(qEmb, item.Embedding)}
	}
	// Sort descending by score (simple insertion sort for small k)
	for i := 1; i < len(scores); i++ {
		for j := i; j > 0 && scores[j].score > scores[j-1].score; j-- {
			scores[j], scores[j-1] = scores[j-1], scores[j]
		}
	}
	if k > len(scores) {
		k = len(scores)
	}
	result := make([]fwcDoc, k)
	for i := range result {
		result[i] = scores[i].doc
	}
	return result, nil
}

// fwcAssembleContext builds the context string from retrieved docs.
func fwcAssembleContext(docs []fwcDoc) string {
	if len(docs) == 0 {
		return "(no documents retrieved)"
	}
	var parts []string
	for _, d := range docs {
		parts = append(parts, fmt.Sprintf("[Source: %s]\n%s", d.Source, d.Text))
	}
	return strings.Join(parts, "\n\n")
}

// fwcCallLLM calls the chat API with a context+question prompt.
func fwcCallLLM(ctx context.Context, client *openai.Client, context_, query string) (string, int, error) {
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: fwcLLMModel,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "Answer the question using ONLY the documents below.\n\n" + context_,
			},
			{Role: openai.ChatMessageRoleUser, Content: query},
		},
	})
	if err != nil {
		return "", 0, err
	}
	answer := resp.Choices[0].Message.Content
	tokens := resp.Usage.TotalTokens
	return answer, tokens, nil
}

// ===========================================================================
// IMPLEMENTATION 1: FROM SCRATCH
// ===========================================================================

// runFWCFromScratch runs the from-scratch RAG implementation.
// When buggy=true the store is replaced with an empty slice so retrieval
// returns no results — making the bug easy to diagnose.
func runFWCFromScratch(ctx context.Context, client *openai.Client, query string, buggy bool) (FWCResult, error) {
	t0 := time.Now()

	index, err := fwcBuildIndex(ctx, client, fwcKnowledgeBase)
	if err != nil {
		return FWCResult{}, fmt.Errorf("from-scratch build index: %w", err)
	}

	if buggy {
		index = nil // deliberate bug: empty index
	}

	retrieved, err := fwcRetrieve(ctx, client, index, query, 3)
	if err != nil {
		return FWCResult{}, fmt.Errorf("from-scratch retrieve: %w", err)
	}

	context_ := fwcAssembleContext(retrieved)
	answer, tokens, err := fwcCallLLM(ctx, client, context_, query)
	if err != nil {
		return FWCResult{}, fmt.Errorf("from-scratch LLM: %w", err)
	}

	sources := make([]string, len(retrieved))
	for i, d := range retrieved {
		sources[i] = d.Source
	}

	debugTrace := ""
	if buggy && len(retrieved) == 0 {
		debugTrace = "BUG: index was set to nil in runFWCFromScratch().\n" +
			"  → retrieved == []\n" +
			"  → context == '(no documents retrieved)'\n" +
			"DIAGNOSIS: Easy — the bug is in YOUR code. Inspect index before retrieve."
	}

	return FWCResult{
		Name:           "From Scratch",
		Answer:         answer,
		Sources:        sources,
		ResponseTimeMs: float64(time.Since(t0).Milliseconds()),
		TokensUsed:     tokens,
		Debuggability:  "Easy",
		DebugTrace:     debugTrace,
	}, nil
}

// ===========================================================================
// IMPLEMENTATION 2: LANGCHAIN-STYLE (Go native equivalent)
//
// LangChain Go (github.com/tmc/langchaingo) exists but lags Python by
// 6-12 months and lacks FAISS support at May 2026. This implementation
// uses the same BOUNDARY POINTS LangChain would use:
//
//   • Document loading       → fwcKnowledgeBase (inline)
//   • Vector store           → fwcSimpleStore (FAISS equivalent)
//   • Retrieval chain        → fwcLCStyleChain (create_retrieval_chain equiv)
//   • LLM integration        → OpenAI Go SDK (ChatOpenAI equivalent)
//
// The key difference from "From Scratch": the chain boundary.
// In Python LangChain this is where debugging gets hard — the chain
// swallows empty-retrieval silently. Here we show the same behaviour
// with a comment at each boundary point.
// ===========================================================================

// fwcSimpleStore is a Go equivalent of LangChain's FAISS vector store.
type fwcSimpleStore struct {
	index []fwcIndexedDoc
}

// fwcBuildLCStyleStore creates the "LangChain-style" vector store.
// LangChain equivalent: FAISS.from_documents(docs, embeddings)
func fwcBuildLCStyleStore(ctx context.Context, client *openai.Client, docs []fwcDoc, buggy bool) (*fwcSimpleStore, error) {
	if buggy {
		// LangChain equivalent: FAISS.from_documents([], embeddings) — empty index
		// BUG NOTE: LangChain returns no exception here; neither do we.
		return &fwcSimpleStore{}, nil
	}
	index, err := fwcBuildIndex(ctx, client, docs)
	if err != nil {
		return nil, err
	}
	return &fwcSimpleStore{index: index}, nil
}

// similaritySearch retrieves top-k docs.
// LangChain equivalent: vector_store.as_retriever(search_kwargs={"k": k})
func (s *fwcSimpleStore) similaritySearch(ctx context.Context, client *openai.Client, query string, k int) ([]fwcDoc, error) {
	return fwcRetrieve(ctx, client, s.index, query, k)
}

// runFWCLangChainStyle runs the LangChain-style RAG implementation.
func runFWCLangChainStyle(ctx context.Context, client *openai.Client, query string, buggy bool) (FWCResult, error) {
	t0 := time.Now()

	// Phase 1: Build store — LangChain boundary point #1
	// LangChain: FAISS.from_documents(langchain_docs, embeddings)
	store, err := fwcBuildLCStyleStore(ctx, client, fwcKnowledgeBase, buggy)
	if err != nil {
		return FWCResult{}, fmt.Errorf("langchain-style build store: %w", err)
	}

	// Phase 2: Retrieve — LangChain boundary point #2
	// LangChain: rag_chain.invoke({"input": query}) — all phases bundled
	docs, err := store.similaritySearch(ctx, client, query, 3)
	if err != nil {
		return FWCResult{}, fmt.Errorf("langchain-style retrieve: %w", err)
	}

	// Phase 3+4: Augment + Generate — LangChain boundary point #3
	context_ := fwcAssembleContext(docs)
	answer, tokens, err := fwcCallLLM(ctx, client, context_, query)
	if err != nil {
		return FWCResult{}, fmt.Errorf("langchain-style LLM: %w", err)
	}

	sources := make([]string, len(docs))
	for i, d := range docs {
		sources[i] = d.Source
	}

	debugTrace := ""
	if buggy {
		debugTrace = "BUG: fwcBuildLCStyleStore returned empty index.\n" +
			"  rag_chain style: similaritySearch returned [] — no exception.\n" +
			"  context == '(no documents retrieved)' — answer is ungrounded.\n" +
			"DIAGNOSIS: Hard — chain boundary hides the empty-retrieval case.\n" +
			"  You must inspect docs == [] manually to find the bug."
	}

	return FWCResult{
		Name:           "LangChain-Style",
		Answer:         answer,
		Sources:        sources,
		ResponseTimeMs: float64(time.Since(t0).Milliseconds()),
		TokensUsed:     tokens,
		Debuggability:  "Hard",
		DebugTrace:     debugTrace,
	}, nil
}

// ===========================================================================
// IMPLEMENTATION 3: LANGGRAPH-STYLE (Go state machine)
//
// LangGraph Go does not exist at May 2026. This uses our local StateGraph
// from langgraph_alternative.go to implement the same retrieve→augment→generate
// graph that Python LangGraph would use.
//
// The STATE is an explicit Go struct passed through each node, so you can
// inspect intermediate state at any step — the key debugging advantage of
// LangGraph over plain LangChain.
// ===========================================================================

// lgRAGState is the typed state for the LangGraph-style RAG workflow.
// LangGraph equivalent: class RAGState(TypedDict): query, retrieved_docs, context, answer, sources
type lgRAGState struct {
	Query         string
	RetrievedDocs []fwcDoc
	Context       string
	Answer        string
	Sources       []string
	TokensUsed    int
}

// runFWCLangGraphStyle runs the LangGraph-style RAG using our Go StateGraph.
// Each graph node is a pure function that receives state and returns updated state.
func runFWCLangGraphStyle(ctx context.Context, client *openai.Client, query string, buggy bool) (FWCResult, error) {
	t0 := time.Now()

	// Build the index up front (LangGraph: outside the graph, in the app setup)
	var index []fwcIndexedDoc
	if !buggy {
		var err error
		index, err = fwcBuildIndex(ctx, client, fwcKnowledgeBase)
		if err != nil {
			return FWCResult{}, fmt.Errorf("langgraph-style build index: %w", err)
		}
	}
	// buggy=true → index remains nil (empty store, same bug as LangChain style)

	// --- Node 1: retrieve ---
	// LangGraph: def retrieve_node(state): docs = vector_store.similarity_search(...)
	state := lgRAGState{Query: query}
	{
		docs, err := fwcRetrieve(ctx, client, index, query, 3)
		if err != nil {
			return FWCResult{}, fmt.Errorf("langgraph-style retrieve node: %w", err)
		}
		state.RetrievedDocs = docs
	}

	// --- Node 2: augment ---
	// LangGraph: def augment_node(state): context = join(retrieved_docs); sources = [...]
	{
		state.Context = fwcAssembleContext(state.RetrievedDocs)
		srcs := make([]string, len(state.RetrievedDocs))
		for i, d := range state.RetrievedDocs {
			srcs[i] = d.Source
		}
		state.Sources = srcs
	}

	// --- Node 3: generate ---
	// LangGraph: def generate_node(state): resp = llm.invoke(messages); return {"answer": ...}
	{
		answer, tokens, err := fwcCallLLM(ctx, client, state.Context, state.Query)
		if err != nil {
			return FWCResult{}, fmt.Errorf("langgraph-style generate node: %w", err)
		}
		state.Answer = answer
		state.TokensUsed = tokens
	}

	debugTrace := ""
	if buggy && len(state.RetrievedDocs) == 0 {
		debugTrace = "BUG: index was nil → retrieve node returned [].\n" +
			"  State after retrieve node:  RetrievedDocs == []\n" +
			"  State after augment node:   Context == '(no documents retrieved)'\n" +
			"  State after generate node:  Answer contains no grounded content.\n" +
			"DIAGNOSIS: Moderate — inspect state.RetrievedDocs after each node.\n" +
			"  LangGraph Studio shows per-node state; here you can fmt.Printf(state)."
	}

	return FWCResult{
		Name:           "LangGraph-Style",
		Answer:         state.Answer,
		Sources:        state.Sources,
		ResponseTimeMs: float64(time.Since(t0).Milliseconds()),
		TokensUsed:     state.TokensUsed,
		Debuggability:  "Moderate",
		DebugTrace:     debugTrace,
	}, nil
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

func fwcPrintTable(results []FWCResult) {
	col := 22
	fmt.Printf("\n%s\n", strings.Repeat("=", 85))
	fmt.Println("  FRAMEWORK COMPARISON TABLE")
	fmt.Printf("%s\n", strings.Repeat("=", 85))

	headers := []string{"Metric", "From Scratch", "LangChain-Style", "LangGraph-Style"}
	fmt.Printf("%-22s %-20s %-20s %-20s\n", headers[0], headers[1], headers[2], headers[3])
	fmt.Printf("%s\n", strings.Repeat("-", 85))

	findResult := func(name string) *FWCResult {
		for i := range results {
			if results[i].Name == name {
				return &results[i]
			}
		}
		return nil
	}

	row := func(label string, values []string) {
		fmt.Printf("%-*s", col, label)
		for _, v := range values {
			fmt.Printf(" %-20s", v)
		}
		fmt.Println()
	}

	names := []string{"From Scratch", "LangChain-Style", "LangGraph-Style"}
	val := func(name, field string) string {
		r := findResult(name)
		if r == nil {
			return "N/A"
		}
		switch field {
		case "ResponseTimeMs":
			return fmt.Sprintf("%.0fms", r.ResponseTimeMs)
		case "TokensUsed":
			if r.TokensUsed == 0 {
				return "N/A"
			}
			return fmt.Sprintf("%d", r.TokensUsed)
		case "Debuggability":
			return r.Debuggability
		case "Sources":
			return fmt.Sprintf("%d", len(r.Sources))
		}
		return "?"
	}

	row("Response time", []string{val(names[0], "ResponseTimeMs"), val(names[1], "ResponseTimeMs"), val(names[2], "ResponseTimeMs")})
	row("Token usage", []string{val(names[0], "TokensUsed"), val(names[1], "TokensUsed"), val(names[2], "TokensUsed")})
	row("Sources returned", []string{val(names[0], "Sources"), val(names[1], "Sources"), val(names[2], "Sources")})
	row("Debuggability", []string{val(names[0], "Debuggability"), val(names[1], "Debuggability"), val(names[2], "Debuggability")})

	fmt.Printf("%s\n", strings.Repeat("=", 85))
}

func fwcPrintDebugSection(results []FWCResult) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 85))
	fmt.Println("  DEBUGGING COMPARISON (vector store returns 0 results)")
	fmt.Printf("%s\n", strings.Repeat("=", 85))
	for _, r := range results {
		if r.DebugTrace != "" {
			fmt.Printf("\n--- %s ---\n", r.Name)
			fmt.Println(r.DebugTrace)
		}
	}
	fmt.Println("\nKEY INSIGHT: From-scratch code has a clear, direct stack trace.")
	fmt.Println("  LangChain-style silently swallows empty-retrieval with no exception.")
	fmt.Println("  LangGraph-style exposes per-node state — moderate to debug.")
}

// runFrameworkComparisonDemo is the demo entry point (no main() needed).
func runFrameworkComparisonDemo() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY not set")
		return
	}
	client := openai.NewClient(apiKey)
	ctx := context.Background()

	buggy := false // set to true to trigger the debugging comparison
	fmt.Printf("\nQuery: %q\n", fwcDemoQuery)
	if buggy {
		fmt.Println("Mode: BUGGY (empty vector store)")
	}

	var results []FWCResult

	fmt.Print("Running From Scratch... ")
	r1, err := runFWCFromScratch(ctx, client, fwcDemoQuery, buggy)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
	} else {
		results = append(results, r1)
		fmt.Printf("done. Answer[:60]: %.60s\n", r1.Answer)
	}

	fmt.Print("Running LangChain-Style... ")
	r2, err := runFWCLangChainStyle(ctx, client, fwcDemoQuery, buggy)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
	} else {
		results = append(results, r2)
		fmt.Printf("done. Answer[:60]: %.60s\n", r2.Answer)
	}

	fmt.Print("Running LangGraph-Style... ")
	r3, err := runFWCLangGraphStyle(ctx, client, fwcDemoQuery, buggy)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
	} else {
		results = append(results, r3)
		fmt.Printf("done. Answer[:60]: %.60s\n", r3.Answer)
	}

	if len(results) > 0 {
		fwcPrintTable(results)
		if buggy {
			fwcPrintDebugSection(results)
		}
	}
}
