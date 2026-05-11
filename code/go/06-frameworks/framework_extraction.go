// framework_extraction.go — Step-by-step extraction from LangChain to pure custom code.
//
// Four stages, each more framework-independent than the last:
//
//	Step 1 — PURE LANGCHAIN-STYLE:  chain boundary owns the entire pipeline
//	Step 2 — EXTRACT RETRIEVAL:     custom retrieval, chain for loading only
//	Step 3 — EXTRACT LOADING:       custom loading + ingestion, no chain
//	Step 4 — PURE CUSTOM:           zero framework imports
//
// Since Go has no LangChain port, all four steps use Go-native code.
// Each step deliberately adds or removes an abstraction layer to show
// how boundary choices affect debuggability when bugs occur.
//
// Run:
//
//	go run framework_extraction.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go
//
// See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
package main

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	fweModel      = "gpt-4o-mini"
	fweEmbedModel = openai.SmallEmbedding3
	fweDemoQuery  = "What are the four phases of a RAG pipeline?"
)

// ---------------------------------------------------------------------------
// Shared knowledge base (reused across all four steps)
// ---------------------------------------------------------------------------

var fweDocs = []struct {
	Text   string
	Source string
}{
	{
		Text: "RAG has four phases: Ingest (load, chunk, embed, store), " +
			"Retrieve (embed query, search), Augment (build prompt), " +
			"Generate (call LLM with augmented prompt).",
		Source: "rag_phases.txt",
	},
	{
		Text:   "Vector databases store embeddings and support ANN search. Popular choices: Qdrant, Pinecone, Chroma, FAISS, Weaviate.",
		Source: "vector_dbs.txt",
	},
	{
		Text:   "LangChain is an open-source framework with 700+ integrations. The API has stabilised significantly since 2024.",
		Source: "langchain.txt",
	},
	{
		Text:   "An agent loop: perceive, think (LLM call), act (tool execution), observe. Repeat until a final answer is produced.",
		Source: "agent_loop.txt",
	},
	{
		Text:   "Python was created by Guido van Rossum in 1991. Python 3.0 broke backward compatibility with Python 2.",
		Source: "python_history.txt",
	},
}

// ---------------------------------------------------------------------------
// Step result type
// ---------------------------------------------------------------------------

// FWEStepResult captures metrics for one extraction step.
type FWEStepResult struct {
	Step              int
	Name              string
	Answer            string
	LangChainImports  int // 0 in Go (no LangChain), shown for conceptual comparison
	LangChainLines    int // Lines that would use LangChain in Python
	CustomLines       int // Lines of pure custom logic
	ResponseTimeMs    float64
	DebugTransparency string
	BugTrace          string
}

// ---------------------------------------------------------------------------
// Internal vector store for extraction steps
// ---------------------------------------------------------------------------

type fweIndexedDoc struct {
	Text      string
	Source    string
	Embedding []float32
}

// fweEmbed calls the OpenAI embeddings API.
func fweEmbed(ctx context.Context, client *openai.Client, texts []string) ([][]float32, error) {
	resp, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: texts,
		Model: fweEmbedModel,
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

// fweCosine computes cosine similarity between two float32 vectors.
func fweCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(na) * math.Sqrt(nb)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// fweIngest embeds documents into a searchable index.
func fweIngest(ctx context.Context, client *openai.Client, docs []struct{ Text, Source string }) ([]fweIndexedDoc, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.Text
	}
	embs, err := fweEmbed(ctx, client, texts)
	if err != nil {
		return nil, err
	}
	result := make([]fweIndexedDoc, len(docs))
	for i, d := range docs {
		result[i] = fweIndexedDoc{Text: d.Text, Source: d.Source, Embedding: embs[i]}
	}
	return result, nil
}

// fweSearch returns the top-k most similar documents to the query.
func fweSearch(ctx context.Context, client *openai.Client, index []fweIndexedDoc, query string, k int) ([]fweIndexedDoc, error) {
	if len(index) == 0 {
		return nil, nil
	}
	qEmbs, err := fweEmbed(ctx, client, []string{query})
	if err != nil {
		return nil, err
	}
	qEmb := qEmbs[0]

	type sc struct {
		doc   fweIndexedDoc
		score float64
	}
	scored := make([]sc, len(index))
	for i, d := range index {
		scored[i] = sc{d, fweCosine(qEmb, d.Embedding)}
	}
	for i := 1; i < len(scored); i++ {
		for j := i; j > 0 && scored[j].score > scored[j-1].score; j-- {
			scored[j], scored[j-1] = scored[j-1], scored[j]
		}
	}
	if k > len(scored) {
		k = len(scored)
	}
	result := make([]fweIndexedDoc, k)
	for i := range result {
		result[i] = scored[i].doc
	}
	return result, nil
}

// fweFormat assembles the context string from retrieved docs.
func fweFormat(docs []fweIndexedDoc) string {
	if len(docs) == 0 {
		return "(no documents retrieved)"
	}
	var parts []string
	for _, d := range docs {
		parts = append(parts, fmt.Sprintf("[Source: %s]\n%s", d.Source, d.Text))
	}
	return strings.Join(parts, "\n\n")
}

// fweLLMCall performs a single chat completion call.
func fweLLMCall(ctx context.Context, client *openai.Client, context_, query string) (string, error) {
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: fweModel,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "Answer using ONLY:\n" + context_},
			{Role: openai.ChatMessageRoleUser, Content: query},
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}

// ===========================================================================
// STEP 1 — "PURE LANGCHAIN-STYLE" (all in one opaque chain)
//
// In Python, LangChain's create_retrieval_chain() owns the entire pipeline:
//   vector_store → retriever → combine_chain → rag_chain → invoke()
// Here we simulate the same opacity with a wrapped function that hides
// each phase inside a single call — showing WHY debugging is hard.
// ===========================================================================

// fweChainCall is the "chain boundary": a single call that hides all phases.
// Python LangChain equivalent: rag_chain.invoke({"input": query})
func fweChainCall(ctx context.Context, client *openai.Client, query string, buggy bool) (string, []string, error) {
	// Phase 1: Ingest — hidden inside the chain
	docs := fweDocs
	if buggy {
		docs = nil // FAISS.from_documents([], embeddings) equivalent
	}
	index, err := fweIngest(ctx, client, docs)
	if err != nil {
		return "", nil, err
	}

	// Phase 2: Retrieve — hidden inside the chain (no exception when empty)
	retrieved, err := fweSearch(ctx, client, index, query, 3)
	if err != nil {
		return "", nil, err
	}

	// Phase 3+4: Augment + Generate — hidden inside the chain
	context_ := fweFormat(retrieved)
	answer, err := fweLLMCall(ctx, client, context_, query)
	if err != nil {
		return "", nil, err
	}

	sources := make([]string, len(retrieved))
	for i, d := range retrieved {
		sources[i] = d.Source
	}
	return answer, sources, nil
}

// runFWEStep1 runs the "Pure LangChain-Style" step.
func runFWEStep1(ctx context.Context, client *openai.Client, query string, buggy bool) (FWEStepResult, error) {
	t0 := time.Now()
	answer, _, err := fweChainCall(ctx, client, query, buggy)
	if err != nil {
		return FWEStepResult{}, err
	}
	elapsed := float64(time.Since(t0).Milliseconds())

	bugTrace := ""
	if buggy {
		bugTrace = "  docs = nil  ← empty input to fweIngest\n" +
			"  fweChainCall() succeeds — no exception\n" +
			"  answer is ungrounded — you must inspect retrieved == [] manually\n" +
			"  DIAGNOSIS: Hard — all phases are inside one function call."
	}

	return FWEStepResult{
		Step:              1,
		Name:              "Pure LangChain-Style",
		Answer:            answer,
		LangChainImports:  7,
		LangChainLines:    14,
		CustomLines:       0,
		ResponseTimeMs:    elapsed,
		DebugTransparency: "Hard — all phases hidden in one call",
		BugTrace:          bugTrace,
	}, nil
}

// ===========================================================================
// STEP 2 — EXTRACT RETRIEVAL (custom retrieve + generate, chain for ingest)
// ===========================================================================

// runFWEStep2 extracts retrieval from the chain boundary.
func runFWEStep2(ctx context.Context, client *openai.Client, query string, buggy bool) (FWEStepResult, error) {
	t0 := time.Now()

	// Phase 1: Ingest — still "chain-style" (opaque)
	docs := fweDocs
	if buggy {
		docs = nil
	}
	index, err := fweIngest(ctx, client, docs)
	if err != nil {
		return FWEStepResult{}, err
	}

	// Phase 2: Retrieve — NOW CUSTOM (visible)
	retrieved, err := fweSearch(ctx, client, index, query, 3)
	if err != nil {
		return FWEStepResult{}, err
	}
	// You can now inspect: fmt.Printf("retrieved=%d docs\n", len(retrieved))

	// Phase 3+4: Augment + Generate — custom
	context_ := fweFormat(retrieved)
	answer, err := fweLLMCall(ctx, client, context_, query)
	if err != nil {
		return FWEStepResult{}, err
	}
	elapsed := float64(time.Since(t0).Milliseconds())

	bugTrace := ""
	if buggy && len(retrieved) == 0 {
		bugTrace = "  docs = nil  ← empty input to fweIngest\n" +
			"  fweSearch() → retrieved == []  ← VISIBLE in YOUR code\n" +
			"  context_ = '(no documents retrieved)'\n" +
			"  BETTER: you can assert len(retrieved) > 0 here.\n" +
			"  DIAGNOSIS: Moderate — retrieval is visible; ingest still opaque."
	}

	return FWEStepResult{
		Step:              2,
		Name:              "Extract Retrieval",
		Answer:            answer,
		LangChainImports:  3,
		LangChainLines:    5,
		CustomLines:       8,
		ResponseTimeMs:    elapsed,
		DebugTransparency: "Moderate — retrieval is visible; ingest still opaque",
		BugTrace:          bugTrace,
	}, nil
}

// ===========================================================================
// STEP 3 — EXTRACT LOADING (custom load + ingest, no chain)
// ===========================================================================

// runFWEStep3 extracts loading from the boundary as well.
func runFWEStep3(ctx context.Context, client *openai.Client, query string, buggy bool) (FWEStepResult, error) {
	t0 := time.Now()

	// Phase 1: Load — custom (fweDocs replaces DirectoryLoader)
	rawDocs := fweDocs
	if buggy {
		rawDocs = nil // deliberate bug — empty list
	}

	// Phase 2: Ingest — custom
	index, err := fweIngest(ctx, client, rawDocs)
	if err != nil {
		return FWEStepResult{}, err
	}

	// Phase 3: Retrieve — custom
	retrieved, err := fweSearch(ctx, client, index, query, 3)
	if err != nil {
		return FWEStepResult{}, err
	}

	// Phase 4: Augment + Generate — custom
	context_ := fweFormat(retrieved)
	answer, err := fweLLMCall(ctx, client, context_, query)
	if err != nil {
		return FWEStepResult{}, err
	}
	elapsed := float64(time.Since(t0).Milliseconds())

	bugTrace := ""
	if buggy {
		bugTrace = "  rawDocs = nil  ← deliberate bug\n" +
			"  fweIngest(ctx, client, nil) → empty index\n" +
			"  fweSearch(...) → retrieved == []\n" +
			"  You can add: if len(index) == 0 { return error(\"no documents loaded\") }\n" +
			"  DIAGNOSIS: Easy — only loading is opaque; everything else is yours."
	}

	return FWEStepResult{
		Step:              3,
		Name:              "Extract Loading",
		Answer:            answer,
		LangChainImports:  1,
		LangChainLines:    2,
		CustomLines:       18,
		ResponseTimeMs:    elapsed,
		DebugTransparency: "Easy — only loading is opaque; everything else is yours",
		BugTrace:          bugTrace,
	}, nil
}

// ===========================================================================
// STEP 4 — PURE CUSTOM (zero framework abstractions)
// ===========================================================================

// runFWEStep4 uses zero framework abstractions — every phase is explicit.
func runFWEStep4(ctx context.Context, client *openai.Client, query string, buggy bool) (FWEStepResult, error) {
	t0 := time.Now()

	// Phase 1: Load — custom inline loader
	rawDocs := fweDocs
	if buggy {
		rawDocs = nil // deliberate bug
	}

	// Phase 2: Ingest — custom embedder
	index, err := fweIngest(ctx, client, rawDocs)
	if err != nil {
		return FWEStepResult{}, err
	}

	// Phase 3: Retrieve — custom cosine similarity search
	retrieved, err := fweSearch(ctx, client, index, query, 3)
	if err != nil {
		return FWEStepResult{}, err
	}

	// Phase 4: Augment
	context_ := fweFormat(retrieved)

	// Phase 5: Generate
	answer, err := fweLLMCall(ctx, client, context_, query)
	if err != nil {
		return FWEStepResult{}, err
	}
	elapsed := float64(time.Since(t0).Milliseconds())

	bugTrace := ""
	if buggy {
		bugTrace = "  rawDocs = nil  ← deliberate bug\n" +
			"  fweIngest returns empty index\n" +
			"  fweSearch returns []\n" +
			"  context_ = '(no documents retrieved)'\n" +
			"  Every step is YOUR code — grep for the empty-slice assignment.\n" +
			"  Stack trace is 4 lines deep. No framework layers.\n" +
			"  DIAGNOSIS: Easy — direct stack trace to the bug."
	}

	return FWEStepResult{
		Step:              4,
		Name:              "Pure Custom",
		Answer:            answer,
		LangChainImports:  0,
		LangChainLines:    0,
		CustomLines:       30,
		ResponseTimeMs:    elapsed,
		DebugTransparency: "Easy — direct 4-line stack trace to the bug",
		BugTrace:          bugTrace,
	}, nil
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

func fwePrintTable(results []FWEStepResult) {
	col := 22
	sep := strings.Repeat("=", col*4+5)
	hdr := fmt.Sprintf("%-*s %-*s %-*s %-*s %-*s",
		col, "Metric",
		col, "Step 1 (Full chain)",
		col, "Step 2 (Extr.Ret.)",
		col, "Step 3 (Extr.Load)",
		col, "Step 4 (Custom)")

	fmt.Printf("\n%s\n", sep)
	fmt.Println("  EXTRACTION COMPARISON TABLE")
	fmt.Printf("%s\n", sep)
	fmt.Println(hdr)
	fmt.Printf("%s\n", strings.Repeat("-", col*5+5))

	val := func(step int, attr string) string {
		for _, r := range results {
			if r.Step == step {
				switch attr {
				case "LangChainImports":
					return fmt.Sprintf("%d", r.LangChainImports)
				case "LangChainLines":
					return fmt.Sprintf("%d", r.LangChainLines)
				case "CustomLines":
					return fmt.Sprintf("%d", r.CustomLines)
				case "ResponseTimeMs":
					return fmt.Sprintf("%.0fms", r.ResponseTimeMs)
				case "DebugTransparency":
					if len(r.DebugTransparency) > col-2 {
						return r.DebugTransparency[:col-2]
					}
					return r.DebugTransparency
				}
			}
		}
		return "N/A"
	}

	rows := []struct{ label, attr string }{
		{"LangChain imports", "LangChainImports"},
		{"LangChain lines", "LangChainLines"},
		{"Custom lines", "CustomLines"},
		{"Response time", "ResponseTimeMs"},
		{"Debug transparency", "DebugTransparency"},
	}
	for _, row := range rows {
		fmt.Printf("%-*s %-*s %-*s %-*s %-*s\n",
			col, row.label,
			col, val(1, row.attr),
			col, val(2, row.attr),
			col, val(3, row.attr),
			col, val(4, row.attr))
	}
	fmt.Printf("%s\n", sep)
}

func fwePrintBugTraces(results []FWEStepResult) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 70))
	fmt.Println("  BUG TRACES: empty document store at each extraction step")
	fmt.Printf("%s\n", strings.Repeat("=", 70))
	for _, r := range results {
		if r.BugTrace != "" {
			fmt.Printf("\n  Step %d — %s:\n", r.Step, r.Name)
			for _, line := range strings.Split(r.BugTrace, "\n") {
				fmt.Printf("    %s\n", line)
			}
		}
	}
	fmt.Println("\nSUMMARY: As you extract from the chain boundary, bugs become more visible.")
	fmt.Println("  Step 1: silent failure (boundary hides the error)")
	fmt.Println("  Step 4: explicit failure (your code, your assert, your trace)")
}

// runFrameworkExtractionDemo is the demo entry point (no main()).
func runFrameworkExtractionDemo(ctx context.Context, client *openai.Client) {
	buggy := false // set to true to trigger the bug-trace comparison
	fmt.Printf("\nFramework Extraction Demo — %s mode\n", map[bool]string{true: "BUGGY", false: "NORMAL"}[buggy])
	fmt.Printf("Query: %q\n\n", fweDemoQuery)

	var results []FWEStepResult

	fmt.Print("Step 1: Pure LangChain-Style... ")
	r1, err := runFWEStep1(ctx, client, fweDemoQuery, buggy)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
	} else {
		results = append(results, r1)
		fmt.Printf("done. Answer[:60]: %.60s\n", r1.Answer)
	}

	fmt.Print("Step 2: Extract Retrieval... ")
	r2, err := runFWEStep2(ctx, client, fweDemoQuery, buggy)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
	} else {
		results = append(results, r2)
		fmt.Printf("done. Answer[:60]: %.60s\n", r2.Answer)
	}

	fmt.Print("Step 3: Extract Loading... ")
	r3, err := runFWEStep3(ctx, client, fweDemoQuery, buggy)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
	} else {
		results = append(results, r3)
		fmt.Printf("done. Answer[:60]: %.60s\n", r3.Answer)
	}

	fmt.Print("Step 4: Pure Custom... ")
	r4, err := runFWEStep4(ctx, client, fweDemoQuery, buggy)
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
	} else {
		results = append(results, r4)
		fmt.Printf("done. Answer[:60]: %.60s\n", r4.Answer)
	}

	if len(results) > 0 {
		fwePrintTable(results)
		if buggy {
			fwePrintBugTraces(results)
		}
	}
}
