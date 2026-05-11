// Context Optimizer — structure and prioritize context for LLM attention.
//
// Go port of code/python/05-context-assembly/context_optimizer.py
//
// Implements research-backed techniques for maximising the model's ability to
// find and use information within a context window:
//
//   - Structured document layout with table-of-contents and section markers.
//   - Position-aware re-ordering so critical facts land in the 20-60% "golden
//     middle" zone where models recall information best.
//   - Chunk deduplication to remove near-identical passages (character 3-gram
//     Jaccard similarity).
//   - Attention-score estimation based on position and structural salience.
//
// See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
package main

import (
	"fmt"
	"math"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// Attention zone boundaries (fractions of total context character length)
// ---------------------------------------------------------------------------

const (
	optimizerPrimacyEnd   = 0.10 // 0 – 10%:  first impression
	optimizerOptimalStart = 0.20 // 20 – 60%: best recall zone
	optimizerOptimalEnd   = 0.60
	optimizerRecencyStart = 0.80 // 80 – 100%: recency bias
)

// ---------------------------------------------------------------------------
// AttentionZones
// ---------------------------------------------------------------------------

// AttentionZones splits a context string into expected-attention quality regions.
type AttentionZones struct {
	PrimacyZone  string // 0–10%
	Transition12 string // 10–20%
	OptimalZone  string // 20–60% — best recall; place critical facts here
	Transition23 string // 60–80%
	RecencyZone  string // 80–100%
}

// ---------------------------------------------------------------------------
// ContextOptimizer
// ---------------------------------------------------------------------------

var (
	optimizerSectionRe     = regexp.MustCompile(`(?s)## \[(\d+)\] (.+?)(?=\n## \[\d+\]|\z)`)
	optimizerHeadingLineRe = regexp.MustCompile(`(?m)^## \[\d+\].*\n`)
	optimizerEndSectionRe  = regexp.MustCompile(`\[End Section \d+\]\s*`)
	optimizerStructureRe   = regexp.MustCompile(`(?m)(^|\n)(#{1,3}\s|\*\s|-\s|\d+\.\s)`)
	optimizerNonWordRe     = regexp.MustCompile(`[^\w\s]`)
)

// ContextOptimizer optimises context structure and content for LLM comprehension.
type ContextOptimizer struct {
	Model string
}

// NewContextOptimizer creates a new ContextOptimizer.
func NewContextOptimizer(model string) *ContextOptimizer {
	if model == "" {
		model = "gpt-4o"
	}
	return &ContextOptimizer{Model: model}
}

// StructureForRetrieval builds a context string optimised for needle-in-a-haystack retrieval.
//
// Applies five techniques:
//  1. Table of contents at the top.
//  2. Numbered section markers (## [N] Title).
//  3. [End Section N] footers.
//  4. [IMPORTANT] tags on documents whose metadata["important"] is true.
//  5. Visual separator lines between sections.
func (co *ContextOptimizer) StructureForRetrieval(documents []map[string]any) string {
	var parts []string

	// Table of contents
	parts = append(parts, "## Context Overview\n")
	for i, doc := range documents {
		meta, _ := doc["metadata"].(map[string]any)
		source := fmt.Sprintf("Document %d", i+1)
		if meta != nil {
			if s, ok := meta["source"].(string); ok && s != "" {
				source = s
			}
		}
		important := false
		if meta != nil {
			if imp, ok := meta["important"].(bool); ok {
				important = imp
			}
		}
		tag := ""
		if important {
			tag = " [IMPORTANT]"
		}
		parts = append(parts, fmt.Sprintf("- Section %d: %s%s", i+1, source, tag))
	}
	parts = append(parts, "\n---\n")

	// Document sections
	for i, doc := range documents {
		meta, _ := doc["metadata"].(map[string]any)
		source := fmt.Sprintf("Document %d", i+1)
		if meta != nil {
			if s, ok := meta["source"].(string); ok && s != "" {
				source = s
			}
		}
		important := false
		if meta != nil {
			if imp, ok := meta["important"].(bool); ok {
				important = imp
			}
		}
		importanceTag := ""
		if important {
			importanceTag = "\n> **[IMPORTANT — Key information]**"
		}
		text, _ := doc["text"].(string)
		parts = append(parts, fmt.Sprintf("## [%d] %s%s\n", i+1, source, importanceTag))
		parts = append(parts, text)
		parts = append(parts, fmt.Sprintf("\n[End Section %d]\n", i+1))
		parts = append(parts, "---\n")
	}

	return strings.Join(parts, "\n")
}

// PrioritizeInformation re-orders context so the most query-relevant section
// lands in the optimal attention zone (20–60% of the total context).
func (co *ContextOptimizer) PrioritizeInformation(context, query string) string {
	sections := co.splitIntoSections(context)
	if len(sections) <= 2 {
		return context
	}

	queryTerms := map[string]bool{}
	for _, t := range strings.Fields(optimizerNonWordRe.ReplaceAllString(strings.ToLower(query), "")) {
		queryTerms[t] = true
	}

	type scoredSection struct {
		score float64
		sec   map[string]any
	}
	var scored []scoredSection
	for _, sec := range sections {
		text, _ := sec["text"].(string)
		textLower := strings.ToLower(text)
		hits := 0
		for t := range queryTerms {
			if strings.Contains(textLower, t) {
				hits++
			}
		}
		score := float64(hits) / math.Max(float64(len(queryTerms)), 1)
		scored = append(scored, scoredSection{score, sec})
	}

	// Sort descending
	for i := 1; i < len(scored); i++ {
		for j := i; j > 0 && scored[j].score > scored[j-1].score; j-- {
			scored[j], scored[j-1] = scored[j-1], scored[j]
		}
	}

	best := scored[0].sec
	var others []map[string]any
	for _, s := range scored[1:] {
		others = append(others, s.sec)
	}

	// Place the best section ~35% of the way through the remaining list
	nBefore := int(math.Ceil(float64(len(others)) * 0.35))
	var reordered []map[string]any
	reordered = append(reordered, others[:nBefore]...)
	reordered = append(reordered, best)
	reordered = append(reordered, others[nBefore:]...)

	return co.reassembleSections(reordered)
}

// ChunkByAttentionZones splits context into attention zones by character position.
func (co *ContextOptimizer) ChunkByAttentionZones(context string) AttentionZones {
	n := len(context)
	if n == 0 {
		return AttentionZones{}
	}
	return AttentionZones{
		PrimacyZone:  context[:int(float64(n)*optimizerPrimacyEnd)],
		Transition12: context[int(float64(n)*optimizerPrimacyEnd):int(float64(n)*optimizerOptimalStart)],
		OptimalZone:  context[int(float64(n)*optimizerOptimalStart):int(float64(n)*optimizerOptimalEnd)],
		Transition23: context[int(float64(n)*optimizerOptimalEnd):int(float64(n)*optimizerRecencyStart)],
		RecencyZone:  context[int(float64(n)*optimizerRecencyStart):],
	}
}

// DeduplicateContext removes near-duplicate documents.
// Two documents are duplicates when their character 3-gram Jaccard similarity > 0.90.
func (co *ContextOptimizer) DeduplicateContext(documents []map[string]any) []map[string]any {
	var unique []map[string]any
	for _, doc := range documents {
		text, _ := doc["text"].(string)
		dupIdx := -1
		for j, u := range unique {
			uText, _ := u["text"].(string)
			if optimizerJaccardSimilarity(text, uText) > 0.90 {
				dupIdx = j
				break
			}
		}
		if dupIdx == -1 {
			unique = append(unique, doc)
		} else {
			uText, _ := unique[dupIdx]["text"].(string)
			if len(text) > len(uText) {
				unique[dupIdx] = doc
			}
		}
	}
	return unique
}

// EstimateAttentionScore estimates how likely the model is to find and use fact in context.
func (co *ContextOptimizer) EstimateAttentionScore(context, fact string) float64 {
	if fact == "" || !strings.Contains(context, fact) {
		return 0.0
	}
	n := len(context)
	if n == 0 {
		return 0.0
	}

	// Find all occurrences
	var occurrences []int
	search := context
	offset := 0
	for {
		idx := strings.Index(search, fact)
		if idx == -1 {
			break
		}
		occurrences = append(occurrences, offset+idx)
		offset += idx + 1
		search = context[offset:]
	}
	if len(occurrences) == 0 {
		return 0.0
	}

	pos := float64(occurrences[0]) / float64(n)
	positionScore := co.positionScore(pos)

	repBonus := math.Min(0.15, float64(len(occurrences)-1)*0.05)

	structureBonus := 0.0
	for _, occ := range occurrences {
		start := occ - 120
		if start < 0 {
			start = 0
		}
		snippet := context[start:occ]
		if optimizerStructureRe.MatchString(snippet) {
			structureBonus = 0.10
			break
		}
	}

	return math.Min(1.0, positionScore+repBonus+structureBonus)
}

// OptimizeForQuery runs the full optimisation pipeline.
func (co *ContextOptimizer) OptimizeForQuery(context, query string) string {
	sections := co.splitIntoSections(context)
	if len(sections) == 0 {
		sections = []map[string]any{{"text": context, "metadata": map[string]any{"source": "context"}}}
	}

	queryTerms := map[string]bool{}
	for _, t := range strings.Fields(optimizerNonWordRe.ReplaceAllString(strings.ToLower(query), "")) {
		queryTerms[t] = true
	}

	type scoredSection struct {
		score float64
		sec   map[string]any
	}
	var scored []scoredSection
	for _, sec := range sections {
		text, _ := sec["text"].(string)
		textLower := strings.ToLower(text)
		hits := 0
		for t := range queryTerms {
			if strings.Contains(textLower, t) {
				hits++
			}
		}
		score := float64(hits) / math.Max(float64(len(queryTerms)), 1)
		scored = append(scored, scoredSection{score, sec})
	}

	var relevant []map[string]any
	for _, s := range scored {
		if s.score > 0 {
			relevant = append(relevant, s.sec)
		}
	}
	if len(relevant) == 0 {
		relevant = sections
	}

	deduplicated := co.DeduplicateContext(relevant)
	structured := co.StructureForRetrieval(deduplicated)
	return co.PrioritizeInformation(structured, query)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (co *ContextOptimizer) positionScore(relativePos float64) float64 {
	centre := 0.40
	width := 0.30
	raw := 1.0 - math.Pow((relativePos-centre)/width, 2)
	return math.Max(0.40, math.Min(1.0, raw))
}

func optimizerJaccardSimilarity(a, b string) float64 {
	trigramsA := stringTrigrams(a)
	trigramsB := stringTrigrams(b)
	if len(trigramsA) == 0 && len(trigramsB) == 0 {
		return 1.0
	}
	if len(trigramsA) == 0 || len(trigramsB) == 0 {
		return 0.0
	}
	intersection := 0
	for t := range trigramsA {
		if trigramsB[t] {
			intersection++
		}
	}
	union := len(trigramsA) + len(trigramsB) - intersection
	return float64(intersection) / float64(union)
}

func stringTrigrams(s string) map[string]bool {
	tg := map[string]bool{}
	runes := []rune(s)
	for i := 0; i+3 <= len(runes); i++ {
		tg[string(runes[i:i+3])] = true
	}
	return tg
}

// splitIntoSections splits a structured context back into component document dicts.
func (co *ContextOptimizer) splitIntoSections(context string) []map[string]any {
	var sections []map[string]any
	for _, m := range optimizerSectionRe.FindAllStringSubmatch(context, -1) {
		idx := m[1]
		title := strings.SplitN(m[2], "\n", 2)[0]
		title = strings.TrimSpace(title)

		text := optimizerHeadingLineRe.ReplaceAllString(m[0], "")
		text = optimizerEndSectionRe.ReplaceAllString(text, "")
		text = strings.TrimSpace(text)

		sectionIdx := 0
		fmt.Sscanf(idx, "%d", &sectionIdx)

		sections = append(sections, map[string]any{
			"text": text,
			"metadata": map[string]any{
				"source":        title,
				"section_index": sectionIdx,
			},
		})
	}
	return sections
}

// reassembleSections reassembles section dicts into a structured context string.
func (co *ContextOptimizer) reassembleSections(sections []map[string]any) string {
	var parts []string
	parts = append(parts, "## Context Overview\n")
	for i, sec := range sections {
		meta, _ := sec["metadata"].(map[string]any)
		source := fmt.Sprintf("Section %d", i+1)
		if meta != nil {
			if s, ok := meta["source"].(string); ok && s != "" {
				source = s
			}
		}
		parts = append(parts, fmt.Sprintf("- Section %d: %s", i+1, source))
	}
	parts = append(parts, "\n---\n")
	for i, sec := range sections {
		meta, _ := sec["metadata"].(map[string]any)
		source := fmt.Sprintf("Section %d", i+1)
		if meta != nil {
			if s, ok := meta["source"].(string); ok && s != "" {
				source = s
			}
		}
		text, _ := sec["text"].(string)
		parts = append(parts, fmt.Sprintf("## [%d] %s\n", i+1, source))
		parts = append(parts, text)
		parts = append(parts, fmt.Sprintf("\n[End Section %d]\n", i+1))
		parts = append(parts, "---\n")
	}
	return strings.Join(parts, "\n")
}
