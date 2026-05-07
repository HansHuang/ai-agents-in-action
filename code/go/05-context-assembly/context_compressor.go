// Package main provides a five-stage context compression pipeline for LLM context.
//
// Pipeline stages:
//  1. Relevance filter   — remove low-similarity documents (adaptive threshold)
//  2. Quality filter     — remove near-duplicates and low-density chunks
//  3. Rerank             — reorder by embedding cosine similarity
//  4. Extractive compress — keep query-relevant sentences verbatim
//  5. Budget enforcement  — drop lowest-scoring documents until within budget
//
// See: docs/04-context-engineering/03-context-compression-and-filtering.md

package main

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

func cosineSim(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var dot, magA, magB float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}
	magA = math.Sqrt(magA)
	magB = math.Sqrt(magB)
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (magA * magB)
}

// ---------------------------------------------------------------------------
// Sentence splitter
// ---------------------------------------------------------------------------

var (
	abbrevRe   = regexp.MustCompile(`(?i)\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|etc|vs|e\.g|i\.e|approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|U\.S|U\.K|E\.U|Corp|Inc|Ltd|LLC)\.`)
	decimalRe  = regexp.MustCompile(`(\d+)\.(\d)`)
	boundaryRe = regexp.MustCompile(`[.!?]\s+[A-Z]`)
)

const sentenceMask = "\x00"

func splitSentences(text string) []string {
	masked := abbrevRe.ReplaceAllStringFunc(text, func(m string) string {
		return strings.ReplaceAll(m, ".", sentenceMask)
	})
	masked = decimalRe.ReplaceAllStringFunc(masked, func(m string) string {
		parts := decimalRe.FindStringSubmatch(m)
		if len(parts) < 3 {
			return m
		}
		return parts[1] + sentenceMask + parts[2]
	})

	// Split on boundary pattern but keep the trailing capital as the start of the next sentence
	var sentences []string
	locs := boundaryRe.FindAllStringIndex(masked, -1)
	prev := 0
	for _, loc := range locs {
		// loc[1]-1 is where the space before the capital is
		end := loc[1] - 1 // include up to (but not including) the space+capital
		sentence := strings.TrimSpace(masked[prev:end])
		if sentence != "" {
			sentence = strings.ReplaceAll(sentence, sentenceMask, ".")
			sentences = append(sentences, sentence)
		}
		prev = loc[1] - 1 // the capital starts the next sentence
	}
	if prev < len(masked) {
		sentence := strings.TrimSpace(masked[prev:])
		if sentence != "" {
			sentence = strings.ReplaceAll(sentence, sentenceMask, ".")
			sentences = append(sentences, sentence)
		}
	}
	return sentences
}

// ---------------------------------------------------------------------------
// Keyword embedder — deterministic mock for tests and demos
// ---------------------------------------------------------------------------

const embedDim = 32

var embedVocab = map[string]int{
	"weather": 0, "temperature": 0, "rain": 0, "sunny": 0, "forecast": 0,
	"stock": 1, "price": 1, "market": 1, "shares": 1, "dividend": 1,
	"sport": 2, "game": 2, "team": 2, "score": 2, "player": 2,
	"return": 3, "refund": 3, "policy": 3, "damaged": 3, "receipt": 3,
	"billing": 4, "invoice": 4, "payment": 4, "charge": 4, "cost": 4,
	"shipping": 5, "delivery": 5, "package": 5, "tracking": 5, "order": 5,
	"security": 6, "password": 6, "authentication": 6, "access": 6,
	"performance": 7, "speed": 7, "latency": 7, "benchmark": 7,
}

// Embedder is the interface required by ContextCompressor.
type Embedder interface {
	Embed(text string) []float64
}

// KeywordEmbedder provides a deterministic keyword-vector embedding.
type KeywordEmbedder struct{}

func (e KeywordEmbedder) Embed(text string) []float64 {
	vec := make([]float64, embedDim)
	for _, word := range strings.Fields(strings.ToLower(text)) {
		word = strings.TrimFunc(word, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
		if dim, ok := embedVocab[word]; ok {
			vec[dim]++
		}
	}
	mag := 0.0
	for _, v := range vec {
		mag += v * v
	}
	mag = math.Sqrt(mag)
	if mag > 0 {
		for i := range vec {
			vec[i] /= mag
		}
	} else {
		val := 1.0 / math.Sqrt(float64(embedDim))
		for i := range vec {
			vec[i] = val
		}
	}
	return vec
}

// ---------------------------------------------------------------------------
// Density scoring (lightweight inline version)
// ---------------------------------------------------------------------------

var (
	numberRe  = regexp.MustCompile(`\b\d[\d,./\-]*%?\b`)
	stopWords = map[string]bool{
		"a": true, "an": true, "the": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
		"with": true, "by": true, "from": true, "as": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true, "did": true,
		"will": true, "would": true, "could": true, "should": true, "may": true,
		"might": true, "must": true, "can": true, "it": true, "its": true,
		"this": true, "that": true, "these": true, "those": true, "i": true,
		"we": true, "you": true, "he": true, "she": true, "they": true,
		"me": true, "us": true, "him": true, "her": true, "them": true,
		"my": true, "our": true, "your": true, "his": true, "their": true,
		"what": true, "which": true, "who": true, "when": true, "where": true,
		"why": true, "how": true, "all": true, "any": true, "some": true,
		"no": true, "not": true, "only": true, "so": true, "than": true,
		"too": true, "very": true, "just": true, "also": true, "if": true,
		"then": true, "there": true, "here": true, "about": true, "after": true,
		"before": true, "into": true, "over": true, "under": true, "while": true,
		"because": true, "since": true, "until": true, "however": true,
		"therefore": true, "yet": true, "still": true, "already": true,
	}
)

func textDensity(text string) float64 {
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}
	numbers := float64(len(numberRe.FindAllString(text, -1)))
	stopCount := 0.0
	for _, w := range words {
		lw := strings.ToLower(strings.TrimFunc(w, func(r rune) bool { return !unicode.IsLetter(r) }))
		if stopWords[lw] {
			stopCount++
		}
	}
	n := float64(len(words))
	factDensity := math.Min(1, numbers/n)
	fillerRatio := stopCount / n
	overall := factDensity*0.35 + (1-fillerRatio)*0.30 + factDensity*0.20 + (1-fillerRatio)*0.15
	return math.Max(0, math.Min(1, overall))
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// CompressionConfig controls all five pipeline stages.
type CompressionConfig struct {
	TargetTokens      int
	MinResults        int
	MaxResults        int
	DedupThreshold    float64
	MinDensity        float64
	RerankMethod      string // "embedding" | "none"
	CompressMethod    string // "extractive" | "none"
	AdaptiveThreshold bool
	FixedThreshold    float64
}

// DefaultCompressionConfig returns sensible defaults.
func DefaultCompressionConfig() CompressionConfig {
	return CompressionConfig{
		TargetTokens:      12_000,
		MinResults:        3,
		MaxResults:        15,
		DedupThreshold:    0.9,
		MinDensity:        0.1,
		RerankMethod:      "none",
		CompressMethod:    "extractive",
		AdaptiveThreshold: true,
		FixedThreshold:    0.6,
	}
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

// StageRecord holds per-stage document and token counts.
type StageRecord struct {
	Name       string
	DocCount   int
	TokenCount int
}

// CompressionAudit holds the full pipeline trace.
type CompressionAudit struct {
	Stages        []StageRecord
	InitialDocs   int
	InitialTokens int
}

// Report returns a formatted ASCII table of the pipeline stages.
func (a *CompressionAudit) Report() string {
	var b strings.Builder
	col := 22
	header := fmt.Sprintf("%-*s  %5s  %9s  %12s", col, "Stage", "Docs", "Tokens", "% Remaining")
	b.WriteString(header + "\n")
	b.WriteString(strings.Repeat("─", col+34) + "\n")
	for _, rec := range a.Stages {
		pct := 0.0
		if a.InitialTokens > 0 {
			pct = float64(rec.TokenCount) / float64(a.InitialTokens) * 100
		}
		b.WriteString(fmt.Sprintf("%-*s  %5d  %9d  %11.0f%%\n",
			col, rec.Name, rec.DocCount, rec.TokenCount, pct))
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// CompressionResult
// ---------------------------------------------------------------------------

// CompressionResult is the output of a Compress call.
type CompressionResult struct {
	Documents []map[string]any
	Stats     map[string]map[string]int
	Audit     *CompressionAudit
}

// ---------------------------------------------------------------------------
// ContextCompressor
// ---------------------------------------------------------------------------

// ContextCompressor runs the five-stage filtering pipeline.
//
// Example:
//
//	compressor := NewContextCompressor(KeywordEmbedder{})
//	result := compressor.Compress("damaged return", docs, 8_000, nil)
//	fmt.Print(result.Audit.Report())
type ContextCompressor struct {
	embedder Embedder
}

// NewContextCompressor creates a ContextCompressor with the given embedder.
// Pass nil to use KeywordEmbedder.
func NewContextCompressor(embedder Embedder) *ContextCompressor {
	if embedder == nil {
		embedder = KeywordEmbedder{}
	}
	return &ContextCompressor{embedder: embedder}
}

// ---------------------------------------------------------------------------
// Stage 1 — Relevance filter
// ---------------------------------------------------------------------------

// FilterByRelevance removes documents below the relevance threshold.
// If threshold < 0, it is determined adaptively.
func (c *ContextCompressor) FilterByRelevance(
	documents []map[string]any,
	threshold float64,
	minResults, maxResults int,
) []map[string]any {
	if len(documents) == 0 {
		return documents
	}

	t := threshold
	if t < 0 {
		t = c.adaptiveThreshold(documents, minResults)
	}

	var filtered []map[string]any
	for _, d := range documents {
		if score(d) >= t {
			filtered = append(filtered, d)
		}
	}

	if len(filtered) < minResults {
		sorted := append([]map[string]any{}, documents...)
		sort.Slice(sorted, func(i, j int) bool { return score(sorted[i]) > score(sorted[j]) })
		if minResults > len(sorted) {
			minResults = len(sorted)
		}
		filtered = sorted[:minResults]
	}

	if len(filtered) > maxResults {
		filtered = filtered[:maxResults]
	}
	return filtered
}

// ---------------------------------------------------------------------------
// Stage 2 — Quality filter
// ---------------------------------------------------------------------------

// FilterByQuality removes near-duplicate and low-density documents.
func (c *ContextCompressor) FilterByQuality(
	documents []map[string]any,
	dedupThreshold, minDensity float64,
) []map[string]any {
	deduped := c.deduplicate(documents, dedupThreshold)
	var kept []map[string]any
	for _, d := range deduped {
		text, _ := d["text"].(string)
		if textDensity(text) >= minDensity {
			kept = append(kept, d)
		}
	}
	return kept
}

// ---------------------------------------------------------------------------
// Stage 3 — Rerank
// ---------------------------------------------------------------------------

// Rerank reorders documents by cosine similarity to query.
func (c *ContextCompressor) Rerank(
	query string,
	documents []map[string]any,
	topK int,
) []map[string]any {
	qEmb := c.embedder.Embed(query)
	type scored struct {
		doc   map[string]any
		score float64
	}
	items := make([]scored, len(documents))
	for i, d := range documents {
		text, _ := d["text"].(string)
		s := cosineSim(c.embedder.Embed(text), qEmb)
		items[i] = scored{doc: d, score: s}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].score > items[j].score })
	result := make([]map[string]any, 0, topK)
	for i, item := range items {
		if i >= topK {
			break
		}
		d := copyDoc(item.doc)
		d["rerankScore"] = item.score
		result = append(result, d)
	}
	return result
}

// ---------------------------------------------------------------------------
// Stage 4 — Compress documents
// ---------------------------------------------------------------------------

// CompressDocuments compresses each document to maxTokensPerDoc tokens.
// method: "extractive" or "none".
func (c *ContextCompressor) CompressDocuments(
	query string,
	documents []map[string]any,
	maxTokensPerDoc int,
	method string,
) []map[string]any {
	if method == "none" {
		return documents
	}
	result := make([]map[string]any, len(documents))
	for i, d := range documents {
		result[i] = c.compressOne(d, query, maxTokensPerDoc)
	}
	return result
}

// ---------------------------------------------------------------------------
// Stage 5 — Budget enforcement
// ---------------------------------------------------------------------------

// EnforceBudget removes the lowest-scoring documents until total fits in targetTokens.
// Always retains at least one document.
func (c *ContextCompressor) EnforceBudget(
	documents []map[string]any,
	targetTokens int,
) []map[string]any {
	docs := append([]map[string]any{}, documents...)
	for {
		total := 0
		for _, d := range docs {
			text, _ := d["text"].(string)
			n, _ := CountTokens(text, "gpt-4o")
			total += n
		}
		if total <= targetTokens || len(docs) <= 1 {
			break
		}
		worst := 0
		for i, d := range docs {
			if rerankOrScore(d) < rerankOrScore(docs[worst]) {
				worst = i
			}
		}
		docs = append(docs[:worst], docs[worst+1:]...)
	}
	return docs
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

// Compress runs the full five-stage pipeline.
// Pass a nil config to use DefaultCompressionConfig.
func (c *ContextCompressor) Compress(
	query string,
	documents []map[string]any,
	targetTokens int,
	config *CompressionConfig,
) *CompressionResult {
	cfg := DefaultCompressionConfig()
	if config != nil {
		cfg = *config
	}
	cfg.TargetTokens = targetTokens

	docs := append([]map[string]any{}, documents...)
	var stages []StageRecord

	snap := func(name string) {
		total := 0
		for _, d := range docs {
			text, _ := d["text"].(string)
			n, _ := CountTokens(text, "gpt-4o")
			total += n
		}
		stages = append(stages, StageRecord{Name: name, DocCount: len(docs), TokenCount: total})
	}

	snap("Raw Input")

	// Stage 1
	t := -1.0
	if !cfg.AdaptiveThreshold {
		t = cfg.FixedThreshold
	}
	docs = c.FilterByRelevance(docs, t, cfg.MinResults, cfg.MaxResults)
	snap("Relevance Filter")

	// Stage 2
	docs = c.FilterByQuality(docs, cfg.DedupThreshold, cfg.MinDensity)
	snap("Quality Filter")

	// Stage 3
	if cfg.RerankMethod != "none" {
		docs = c.Rerank(query, docs, cfg.MaxResults)
	}
	snap(fmt.Sprintf("Rerank (%s)", cfg.RerankMethod))

	// Stage 4
	if cfg.CompressMethod != "none" && len(docs) > 0 {
		tokensPerDoc := cfg.TargetTokens / len(docs)
		if tokensPerDoc < 50 {
			tokensPerDoc = 50
		}
		docs = c.CompressDocuments(query, docs, tokensPerDoc, cfg.CompressMethod)
	}
	snap("Extract/Compress")

	// Stage 5
	docs = c.EnforceBudget(docs, cfg.TargetTokens)
	snap("Budget Enforcement")

	audit := &CompressionAudit{
		Stages:        stages,
		InitialDocs:   stages[0].DocCount,
		InitialTokens: stages[0].TokenCount,
	}
	stats := make(map[string]map[string]int, len(stages))
	for _, r := range stages {
		stats[r.Name] = map[string]int{"docs": r.DocCount, "tokens": r.TokenCount}
	}
	return &CompressionResult{Documents: docs, Stats: stats, Audit: audit}
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

func score(d map[string]any) float64 {
	if v, ok := d["score"].(float64); ok {
		return v
	}
	return 0
}

func rerankOrScore(d map[string]any) float64 {
	if v, ok := d["rerankScore"].(float64); ok {
		return v
	}
	return score(d)
}

func copyDoc(d map[string]any) map[string]any {
	out := make(map[string]any, len(d))
	for k, v := range d {
		out[k] = v
	}
	return out
}

func (c *ContextCompressor) adaptiveThreshold(documents []map[string]any, minResults int) float64 {
	scores := make([]float64, len(documents))
	for i, d := range documents {
		scores[i] = score(d)
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(scores)))
	for _, t := range []float64{0.8, 0.7, 0.6, 0.5, 0.4, 0.3} {
		count := 0
		for _, s := range scores {
			if s >= t {
				count++
			}
		}
		if count >= minResults {
			return t
		}
	}
	return 0.3
}

func (c *ContextCompressor) deduplicate(documents []map[string]any, threshold float64) []map[string]any {
	ngrams := func(text string, n int) map[string]bool {
		t := strings.ToLower(text)
		set := make(map[string]bool)
		for i := 0; i <= len(t)-n; i++ {
			set[t[i:i+n]] = true
		}
		return set
	}
	jaccard := func(a, b string) float64 {
		sa, sb := ngrams(a, 3), ngrams(b, 3)
		if len(sa) == 0 || len(sb) == 0 {
			return 0
		}
		intersection := 0
		for k := range sa {
			if sb[k] {
				intersection++
			}
		}
		union := len(sa) + len(sb) - intersection
		return float64(intersection) / float64(union)
	}

	var kept []map[string]any
	for _, doc := range documents {
		text, _ := doc["text"].(string)
		isDup := false
		for _, k := range kept {
			kt, _ := k["text"].(string)
			if jaccard(text, kt) >= threshold {
				isDup = true
				break
			}
		}
		if !isDup {
			kept = append(kept, doc)
		}
	}
	return kept
}

func (c *ContextCompressor) compressOne(doc map[string]any, query string, maxTokens int) map[string]any {
	text, _ := doc["text"].(string)
	origTokens, _ := CountTokens(text, "gpt-4o")
	if origTokens <= maxTokens {
		return doc
	}

	sentences := splitSentences(text)
	var longEnough []string
	for _, s := range sentences {
		if len(s) >= 20 {
			longEnough = append(longEnough, s)
		}
	}
	if len(longEnough) == 0 {
		return doc
	}

	qEmb := c.embedder.Embed(query)
	maxSentences := maxTokens / 30
	if maxSentences < 1 {
		maxSentences = 1
	}

	type si struct {
		idx   int
		score float64
	}
	ranked := make([]si, len(longEnough))
	for i, s := range longEnough {
		ranked[i] = si{i, cosineSim(c.embedder.Embed(s), qEmb)}
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].score > ranked[b].score })

	keep := make(map[int]bool)
	if len(longEnough) > maxSentences {
		keep[0] = true
		keep[len(longEnough)-1] = true
	}
	for _, r := range ranked {
		if len(keep) >= maxSentences {
			break
		}
		keep[r.idx] = true
	}

	var indices []int
	for i := range keep {
		indices = append(indices, i)
	}
	sort.Ints(indices)

	var parts []string
	for _, i := range indices {
		parts = append(parts, longEnough[i])
	}
	compressed := strings.Join(parts, " ")
	compTokens, _ := CountTokens(compressed, "gpt-4o")

	out := copyDoc(doc)
	out["text"] = compressed
	out["compressed"] = true
	out["originalTokens"] = origTokens
	out["compressedTokens"] = compTokens
	return out
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

func runContextCompressorDemo() {
	compressor := NewContextCompressor(KeywordEmbedder{})

	docs := []map[string]any{
		{"text": "Damaged items must be reported within 48 hours with photos for a full refund.", "score": 0.92},
		{"text": "To return a damaged product visit returns.acme.com and select Damaged Item.", "score": 0.89},
		{"text": "Standard returns are accepted within 30 days of purchase.", "score": 0.65},
		{"text": "Refunds are processed in 5 to 7 business days after we receive your return.", "score": 0.63},
		{"text": "We offer free shipping on orders over $50.", "score": 0.30},
		{"text": "Our headquarters is located in Austin Texas.", "score": 0.08},
		{"text": "Thank you for choosing us. We appreciate your patience and understanding.", "score": 0.12},
		{"text": "Thank you for choosing us. We appreciate your patience and understanding. (dup)", "score": 0.11},
		{"text": "Quarterly earnings exceeded analyst expectations by 12 percent.", "score": 0.05},
	}

	result := compressor.Compress("How do I return a damaged item?", docs, 500, nil)
	fmt.Print(result.Audit.Report())
	fmt.Printf("\nFinal document count: %d\n", len(result.Documents))
}
