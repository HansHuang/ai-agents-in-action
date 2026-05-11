// langchain_rag_pipeline.go — LangChain RAG pipeline equivalent in Go.
//
// LangChain has no official Go port. This file demonstrates the same CONCEPTS:
//   - Ingest: load documents → chunk → embed → store
//   - Retrieve: embed query → ANN search → top-k docs
//   - Augment: build a grounded prompt
//   - Generate: call the LLM with the augmented prompt
//
// It also shows the 4-step extraction path from a LangChain-style monolithic
// pipeline to pure custom Go code, and compares the two side-by-side.
//
// Run:
//
//	go run langchain_rag_pipeline.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go over_engineering_detector.go langsmith_tracer.go \
//	           autogen_design_team.go crewai_research_crew.go langgraph_multi_agent.go \
//	           langgraph_react_agent.go
//
// See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
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
// Sample knowledge base
// ---------------------------------------------------------------------------

var lcSampleDocs = []struct {
	Text   string
	Source string
}{
	{
		Text:   "LangChain is an open-source framework for building LLM applications. It provides chains, agents, and memory abstractions with 700+ integrations.",
		Source: "langchain_overview.txt",
	},
	{
		Text:   "A RAG pipeline has four phases: Ingest (load, chunk, embed, store), Retrieve (embed query, search), Augment (build prompt), Generate (LLM call).",
		Source: "rag_architecture.txt",
	},
	{
		Text:   "LangChain's LCEL (LangChain Expression Language) uses the pipe operator to compose chains: loader | splitter | embedder | store | retriever | prompt | llm | parser.",
		Source: "lcel_guide.txt",
	},
	{
		Text:   "The main argument for using LangChain: fast prototyping with ready-made components. The main argument against: debugging is hard because the chain boundary hides data flow.",
		Source: "framework_tradeoffs.txt",
	},
	{
		Text:   "Go has no official LangChain port. Teams using Go for AI pipelines typically build custom retrieval logic using the OpenAI Go SDK and a vector database client.",
		Source: "go_ai_ecosystem.txt",
	},
	{
		Text:   "Context window management is critical for RAG. If retrieved documents exceed the context limit, use compression (summarise chunks) or reranking (keep top-N by relevance).",
		Source: "context_management.txt",
	},
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

// LCRAGResult holds the output of a single RAG query.
type LCRAGResult struct {
	Answer         string
	Sources        []string
	RetrievedCount int
	TokensUsed     int
	ElapsedMs      float64
}

// LCPipelineComparison holds the comparison between LangChain-style and custom.
type LCPipelineComparison struct {
	Query         string
	LangChainLike LCRAGResult
	Custom        LCRAGResult
}

// ---------------------------------------------------------------------------
// Internal vector store
// ---------------------------------------------------------------------------

type lcIndexedDoc struct {
	Text      string
	Source    string
	Embedding []float32
}

// lcCosineSim computes cosine similarity between two float32 vectors.
func lcCosineSim(a, b []float32) float64 {
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

// ---------------------------------------------------------------------------
// LCPipeline — the "LangChain-style" RAG pipeline
// ---------------------------------------------------------------------------

// LCPipeline wraps the knowledge base and provides RAG operations.
type LCPipeline struct {
	client *openai.Client
	index  []lcIndexedDoc
}

// NewLCPipeline creates a new pipeline (not yet ingested).
func NewLCPipeline(client *openai.Client) *LCPipeline {
	return &LCPipeline{client: client}
}

// lcIngest embeds all sample documents and stores them in the index.
func (p *LCPipeline) lcIngest(ctx context.Context) error {
	texts := make([]string, len(lcSampleDocs))
	for i, d := range lcSampleDocs {
		texts[i] = d.Text
	}
	resp, err := p.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: texts,
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		return err
	}
	p.index = make([]lcIndexedDoc, len(lcSampleDocs))
	for i, d := range lcSampleDocs {
		p.index[i] = lcIndexedDoc{
			Text:      d.Text,
			Source:    d.Source,
			Embedding: resp.Data[i].Embedding,
		}
	}
	return nil
}

// lcQuery retrieves relevant documents and generates an answer.
func (p *LCPipeline) lcQuery(ctx context.Context, question string) (LCRAGResult, error) {
	t0 := time.Now()
	result := LCRAGResult{}

	// Step 1: Embed the query
	qResp, err := p.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{question},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		return result, fmt.Errorf("embed query: %w", err)
	}
	qEmb := qResp.Data[0].Embedding

	// Step 2: Retrieve top-3 most similar documents
	type scored struct {
		doc   lcIndexedDoc
		score float64
	}
	scored_ := make([]scored, len(p.index))
	for i, d := range p.index {
		scored_[i] = scored{d, lcCosineSim(qEmb, d.Embedding)}
	}
	for i := 1; i < len(scored_); i++ {
		for j := i; j > 0 && scored_[j].score > scored_[j-1].score; j-- {
			scored_[j], scored_[j-1] = scored_[j-1], scored_[j]
		}
	}
	k := 3
	if k > len(scored_) {
		k = len(scored_)
	}
	var contextParts []string
	for i := 0; i < k; i++ {
		contextParts = append(contextParts, fmt.Sprintf("[%s]\n%s", scored_[i].doc.Source, scored_[i].doc.Text))
		result.Sources = append(result.Sources, scored_[i].doc.Source)
	}
	result.RetrievedCount = k

	// Step 3: Augment + Generate
	contextStr := strings.Join(contextParts, "\n\n")
	chatResp, err := p.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "Answer the question using ONLY the context below. If the context does not contain the answer, say 'I don't know'.\n\nContext:\n" + contextStr,
			},
			{Role: openai.ChatMessageRoleUser, Content: question},
		},
		MaxTokens: 300,
	})
	if err != nil {
		return result, fmt.Errorf("generate answer: %w", err)
	}
	result.Answer = chatResp.Choices[0].Message.Content
	result.TokensUsed = chatResp.Usage.TotalTokens
	result.ElapsedMs = float64(time.Since(t0).Milliseconds())
	return result, nil
}

// extractToCustom returns a summary of the 4-step extraction path.
func (p *LCPipeline) extractToCustom() string {
	return `4-step extraction from LangChain-style to pure custom Go:

Step 1 (Full chain):   One function call owns all 4 phases.
                       Debugging: Hard — the chain boundary hides data flow.

Step 2 (Extract Ret.): Custom retrieval, chain owns loading/ingestion.
                       Debugging: Moderate — retrieval is visible.

Step 3 (Extract Load): Custom loading + ingestion, no chain.
                       Debugging: Easy — only loading phase is opaque.

Step 4 (Pure Custom):  Zero abstractions. Every phase is your code.
                       Debugging: Easy — 4-line stack trace to any bug.

Recommendation: Start with Step 4 unless you need 700+ integrations.
`
}

// showPDFExtension shows how to extend the pipeline to ingest PDFs.
func showPDFExtension() {
	fmt.Println(`
PDF Extension Pattern (Go):
  // 1. Extract text from PDF (e.g. github.com/ledongthuc/pdf or pdfcpu)
  text, err := extractPDFText("report.pdf")
  // 2. Chunk by paragraph
  chunks := chunkByParagraph(text, 400)
  // 3. Add to lcSampleDocs and re-ingest
  for _, chunk := range chunks {
      lcSampleDocs = append(lcSampleDocs, struct{ Text, Source string }{chunk, "report.pdf"})
  }
  pipeline.lcIngest(ctx)
`)
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// runLangChainRAGPipelineDemo is the demo entry point (no main()).
func runLangChainRAGPipelineDemo(ctx context.Context, client *openai.Client) {
	fmt.Printf("\n%s\n  LANGCHAIN RAG PIPELINE (Go-native equivalent)\n%s\n", strings.Repeat("═", 60), strings.Repeat("═", 60))

	pipeline := NewLCPipeline(client)

	fmt.Printf("\nIngesting %d documents...\n", len(lcSampleDocs))
	if err := pipeline.lcIngest(ctx); err != nil {
		fmt.Printf("Ingest error: %v\n", err)
		return
	}
	fmt.Printf("Indexed %d documents.\n\n", len(lcSampleDocs))

	questions := []string{
		"What are the four phases of a RAG pipeline?",
		"Why might a developer choose LangChain over custom code?",
		"How does Go handle the lack of a LangChain port?",
	}

	for _, q := range questions {
		fmt.Printf("Q: %s\n", q)
		result, err := pipeline.lcQuery(ctx, q)
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}
		fmt.Printf("A: %s\n", result.Answer)
		fmt.Printf("   [sources: %s | tokens: %d | %.0fms]\n\n",
			strings.Join(result.Sources, ", "), result.TokensUsed, result.ElapsedMs)
	}

	fmt.Printf("\n%s\n  EXTRACTION PATH\n%s\n", strings.Repeat("─", 60), strings.Repeat("─", 60))
	fmt.Println(pipeline.extractToCustom())

	showPDFExtension()
}
