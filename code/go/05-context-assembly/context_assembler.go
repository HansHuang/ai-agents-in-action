// ContextAssembler — dynamic context assembly from multiple sources.
//
// Assembles the LLM context string from RAG documents, tool results, user
// profiles, conversation summaries, and template variables.  Integrates with
// ContextBudget for token budget enforcement.
//
// See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// ContextConfig is a declarative configuration for context assembly.
type ContextConfig struct {
	Template           string
	TemplateVars       map[string]string
	IncludeSources     []string // "rag", "tools", "profile", "summary"
	MaxTokensPerSource map[string]int
	Format             string // "markdown" | "plain" | "json"
	PriorityOrder      []string
}

// ---------------------------------------------------------------------------
// Document types
// ---------------------------------------------------------------------------

// Document is a single retrieved document with text and metadata.
type Document struct {
	Text     string
	Metadata map[string]any
}

// ToolResult is the output of a single tool invocation.
type ToolResult struct {
	Tool   string
	Result any
}

// ---------------------------------------------------------------------------
// Assembly result
// ---------------------------------------------------------------------------

// AssemblyResult holds the assembled context and per-source token accounting.
type AssemblyResult struct {
	Context         string
	TokenBreakdown  map[string]int
	SourcesIncluded []string
	SourcesExcluded []string
}

// TotalTokens returns the sum of all section token counts.
func (r *AssemblyResult) TotalTokens() int {
	total := 0
	for _, t := range r.TokenBreakdown {
		total += t
	}
	return total
}

// ---------------------------------------------------------------------------
// Section headers
// ---------------------------------------------------------------------------

var sectionHeaders = map[string]string{
	"rag":     "## Retrieved Documents\n",
	"tools":   "## Tool Results\n",
	"profile": "## User Profile\n",
	"summary": "## Conversation Summary\n",
}

// ---------------------------------------------------------------------------
// ContextAssembler
// ---------------------------------------------------------------------------

// ContextAssembler assembles context for LLM calls from multiple sources.
//
// Usage:
//
//	budget    := NewContextBudget(128_000, "gpt-4o")
//	assembler := NewContextAssembler(budget, "gpt-4o")
//	result, err := assembler.Assemble(AssembleOptions{...})
type ContextAssembler struct {
	budget *ContextBudget
	model  string
}

// NewContextAssembler creates a new ContextAssembler.
func NewContextAssembler(budget *ContextBudget, model string) *ContextAssembler {
	if model == "" {
		model = "gpt-4o"
	}
	return &ContextAssembler{budget: budget, model: model}
}

// AssembleOptions holds all optional inputs for Assemble.
type AssembleOptions struct {
	Template            string
	Variables           map[string]string
	RetrievedDocs       []Document
	ToolResults         []ToolResult
	UserProfile         map[string]string
	ConversationSummary string
	Query               string
	Optimize            bool
}

// Assemble assembles the dynamic context for an LLM call.
//
// Sources are assembled in priority order (rag → tools → profile → summary).
// Each section is added while budget remains; oversized sections are truncated;
// sections with no remaining budget are excluded.
func (a *ContextAssembler) Assemble(opts AssembleOptions) (*AssemblyResult, error) {
	if opts.Variables == nil {
		opts.Variables = map[string]string{}
	}

	renderedTemplate := renderTemplate(opts.Template, opts.Variables)

	// Build source content
	sections := map[string]string{}

	if len(opts.RetrievedDocs) > 0 {
		if opts.Optimize && opts.Query != "" {
			sections["rag"] = structureDocuments(opts.RetrievedDocs)
		} else {
			var parts []string
			for _, d := range opts.RetrievedDocs {
				parts = append(parts, d.Text)
			}
			sections["rag"] = strings.Join(parts, "\n\n")
		}
	}

	if len(opts.ToolResults) > 0 {
		var lines []string
		for _, tr := range opts.ToolResults {
			var resultStr string
			switch v := tr.Result.(type) {
			case string:
				resultStr = v
			default:
				b, _ := json.MarshalIndent(tr.Result, "", "  ")
				resultStr = string(b)
			}
			lines = append(lines, fmt.Sprintf("**%s**:\n%s", tr.Tool, resultStr))
		}
		sections["tools"] = strings.Join(lines, "\n\n")
	}

	if len(opts.UserProfile) > 0 {
		var lines []string
		for k, v := range opts.UserProfile {
			lines = append(lines, fmt.Sprintf("- **%s**: %s", k, v))
		}
		sections["profile"] = strings.Join(lines, "\n")
	}

	if opts.ConversationSummary != "" {
		sections["summary"] = opts.ConversationSummary
	}

	dcBudget, err := a.budget.GetTokenBudget("dynamic_context")
	if err != nil {
		return nil, err
	}

	contextParts := []string{renderedTemplate}
	tokenBreakdown := map[string]int{}
	templateTok, _ := CountTokens(renderedTemplate, a.model)
	tokenBreakdown["template"] = templateTok

	sourcesIncluded := []string{}
	sourcesExcluded := []string{}

	for _, source := range []string{"rag", "tools", "profile", "summary"} {
		content, ok := sections[source]
		if !ok {
			continue
		}

		header, ok2 := sectionHeaders[source]
		if !ok2 {
			header = fmt.Sprintf("## %s\n", strings.Title(source))
		}

		sectionText := header + content
		sectionTok, _ := CountTokens(sectionText, a.model)

		used := 0
		for _, t := range tokenBreakdown {
			used += t
		}

		if used+sectionTok <= dcBudget {
			contextParts = append(contextParts, sectionText)
			tokenBreakdown[source] = sectionTok
			sourcesIncluded = append(sourcesIncluded, source)
		} else {
			headerTok, _ := CountTokens(header, a.model)
			available := dcBudget - used - headerTok
			if available > 100 {
				truncated, _ := a.budget.compressText(content, available)
				truncText := header + truncated
				truncTok, _ := CountTokens(truncText, a.model)
				contextParts = append(contextParts, truncText)
				tokenBreakdown[source] = truncTok
				sourcesIncluded = append(sourcesIncluded, source)
			} else {
				sourcesExcluded = append(sourcesExcluded, source)
			}
		}
	}

	return &AssemblyResult{
		Context:         strings.Join(contextParts, "\n\n"),
		TokenBreakdown:  tokenBreakdown,
		SourcesIncluded: sourcesIncluded,
		SourcesExcluded: sourcesExcluded,
	}, nil
}

// AssembleFromConfig assembles context from a ContextConfig.
func (a *ContextAssembler) AssembleFromConfig(
	config ContextConfig,
	docs []Document,
	toolResults []ToolResult,
	profile map[string]string,
	summary string,
	query string,
) (*AssemblyResult, error) {
	included := map[string]bool{}
	srcs := config.IncludeSources
	if len(srcs) == 0 {
		srcs = []string{"rag", "tools", "profile", "summary"}
	}
	for _, s := range srcs {
		included[s] = true
	}

	var filteredDocs []Document
	if included["rag"] {
		filteredDocs = docs
	}

	var filteredTools []ToolResult
	if included["tools"] {
		filteredTools = toolResults
	}

	var filteredProfile map[string]string
	if included["profile"] {
		filteredProfile = profile
	}

	filteredSummary := ""
	if included["summary"] {
		filteredSummary = summary
	}

	// Apply per-source token limits
	if ragCap, ok := config.MaxTokensPerSource["rag"]; ok && len(filteredDocs) > 0 {
		filteredDocs = a.clipDocsToBudget(filteredDocs, ragCap)
	}

	optimize := config.Format == "markdown" || config.Format == ""

	return a.Assemble(AssembleOptions{
		Template:            config.Template,
		Variables:           config.TemplateVars,
		RetrievedDocs:       filteredDocs,
		ToolResults:         filteredTools,
		UserProfile:         filteredProfile,
		ConversationSummary: filteredSummary,
		Query:               query,
		Optimize:            optimize,
	})
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func (a *ContextAssembler) clipDocsToBudget(docs []Document, maxTokens int) []Document {
	kept := []Document{}
	used := 0
	for _, doc := range docs {
		t, _ := CountTokens(doc.Text, a.model)
		if used+t <= maxTokens {
			kept = append(kept, doc)
			used += t
		} else {
			break
		}
	}
	return kept
}

// renderTemplate replaces $var and ${var} placeholders in template.
func renderTemplate(template string, vars map[string]string) string {
	result := template
	for k, v := range vars {
		result = strings.ReplaceAll(result, "${"+k+"}", v)
		result = strings.ReplaceAll(result, "$"+k, v)
	}
	return result
}

// structureDocuments builds a table-of-contents + section-marker context string.
func structureDocuments(docs []Document) string {
	var sb strings.Builder
	sb.WriteString("## Context Overview\n\n")
	for i, doc := range docs {
		source := fmt.Sprintf("Document %d", i+1)
		if s, ok := doc.Metadata["source"]; ok {
			source = fmt.Sprintf("%v", s)
		}
		sb.WriteString(fmt.Sprintf("- Section %d: %s\n", i+1, source))
	}
	sb.WriteString("\n---\n\n")

	for i, doc := range docs {
		source := fmt.Sprintf("Document %d", i+1)
		if s, ok := doc.Metadata["source"]; ok {
			source = fmt.Sprintf("%v", s)
		}
		sb.WriteString(fmt.Sprintf("## [%d] %s\n\n", i+1, source))
		sb.WriteString(doc.Text)
		sb.WriteString(fmt.Sprintf("\n\n[End Section %d]\n\n---\n\n", i+1))
	}

	return sb.String()
}
