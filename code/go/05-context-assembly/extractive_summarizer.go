// Extractive Summarizer — pull the most query-relevant sentences verbatim.
//
// Go port of code/python/05-context-assembly/extractive_summarizer.py
//
// Unlike abstractive summarization (which rewrites), extractive selection
// preserves original wording, eliminating hallucination risk. Every sentence
// in the output appears verbatim in the input.
//
// See: docs/04-context-engineering/03-context-compression-and-filtering.md
package main

import (
	"math"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Sentence splitter (reuses patterns from context_compressor.go)
// ---------------------------------------------------------------------------

// ExtractiveAbbrevRe masks abbreviation periods.
var extractiveAbbrevRe = regexp.MustCompile(
	`(?i)\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|etc|vs|e\.g|i\.e|approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|U\.S|U\.K|E\.U|Corp|Inc|Ltd|LLC)\.`)

// extractiveDecimalRe masks decimal points in numbers.
var extractiveDecimalRe = regexp.MustCompile(`(\d+)\.(\d)`)

// extractiveBoundaryRe finds sentence boundaries: .!? followed by space + capital.
var extractiveBoundaryRe = regexp.MustCompile(`(?:[.!?])\s+(?:[A-Z])`)

const extractiveMask = "\x00"

// splitSentencesExtractive splits text into sentences handling abbreviations and decimals.
func splitSentencesExtractive(text string) []string {
	masked := extractiveAbbrevRe.ReplaceAllStringFunc(text, func(m string) string {
		return strings.ReplaceAll(m, ".", extractiveMask)
	})
	masked = extractiveDecimalRe.ReplaceAllStringFunc(masked, func(m string) string {
		sub := extractiveDecimalRe.FindStringSubmatch(m)
		if len(sub) < 3 {
			return m
		}
		return sub[1] + extractiveMask + sub[2]
	})

	// Split on boundary but keep leading capital with next sentence
	locs := extractiveBoundaryRe.FindAllStringIndex(masked, -1)
	var sentences []string
	prev := 0
	for _, loc := range locs {
		// The match includes the capital; keep the capital for the next sentence
		end := loc[1] - 1
		s := strings.TrimSpace(masked[prev:end])
		if s != "" {
			s = strings.ReplaceAll(s, extractiveMask, ".")
			sentences = append(sentences, s)
		}
		prev = loc[1] - 1
	}
	if prev < len(masked) {
		s := strings.TrimSpace(masked[prev:])
		if s != "" {
			s = strings.ReplaceAll(s, extractiveMask, ".")
			sentences = append(sentences, s)
		}
	}
	return sentences
}

// Embedder interface is defined in context_compressor.go.
// KeywordEmbedder is defined in context_compressor.go.

// NewKeywordEmbedder creates a new KeywordEmbedder (defined in context_compressor.go).
func NewKeywordEmbedder() KeywordEmbedder {
	return KeywordEmbedder{}
}

// cosineSim is defined in context_compressor.go.

// ---------------------------------------------------------------------------
// ExtractiveSummarizer
// ---------------------------------------------------------------------------

// ExtractiveSummarizer summarises text by extracting the most query-relevant sentences.
// All output sentences appear verbatim in the input → zero hallucination risk.
type ExtractiveSummarizer struct {
	embedder Embedder
}

// NewExtractiveSummarizer creates a new ExtractiveSummarizer.
func NewExtractiveSummarizer(embedder Embedder) *ExtractiveSummarizer {
	return &ExtractiveSummarizer{embedder: embedder}
}

// Summarize extracts up to maxSentences query-relevant sentences from text.
//
// Sentences shorter than minSentenceLength characters are skipped.
// Always retains the first and last sentence to preserve context.
func (es *ExtractiveSummarizer) Summarize(text, query string, maxSentences, minSentenceLength int) string {
	all := splitSentencesExtractive(text)
	var sentences []string
	for _, s := range all {
		if len(s) >= minSentenceLength {
			sentences = append(sentences, s)
		}
	}
	if len(sentences) == 0 {
		return text
	}
	if len(sentences) <= maxSentences {
		return strings.Join(sentences, " ")
	}

	qEmb := es.embedder.Embed(query)
	scores := es.ScoreSentences(sentences, qEmb)

	// Anchor first and last sentences for readability
	keep := map[int]bool{0: true, len(sentences) - 1: true}
	remaining := maxSentences - len(keep)

	if remaining > 0 {
		type idxScore struct {
			idx   int
			score float64
		}
		var ranked []idxScore
		for i, s := range scores {
			if !keep[i] {
				ranked = append(ranked, idxScore{i, s})
			}
		}
		sort.Slice(ranked, func(i, j int) bool {
			return ranked[i].score > ranked[j].score
		})
		for _, r := range ranked[:remaining] {
			keep[r.idx] = true
		}
	}

	// Collect in order
	var indices []int
	for i := range keep {
		indices = append(indices, i)
	}
	sort.Ints(indices)

	var result []string
	for _, i := range indices {
		result = append(result, sentences[i])
	}
	return strings.Join(result, " ")
}

// SummarizeDocumentBatch compresses each document individually.
func (es *ExtractiveSummarizer) SummarizeDocumentBatch(documents []map[string]any, query string, maxTokensPerDoc int) []map[string]any {
	var result []map[string]any
	for _, doc := range documents {
		text, _ := doc["text"].(string)
		origTokens, _ := CountTokens(text, "gpt-4o")
		maxSentences := maxTokensPerDoc / 30
		if maxSentences < 1 {
			maxSentences = 1
		}
		compressed := es.Summarize(text, query, maxSentences, 20)
		compTokens, _ := CountTokens(compressed, "gpt-4o")

		newDoc := make(map[string]any)
		for k, v := range doc {
			newDoc[k] = v
		}
		newDoc["text"] = compressed
		newDoc["compressed"] = true
		newDoc["original_tokens"] = origTokens
		newDoc["compressed_tokens"] = compTokens
		result = append(result, newDoc)
	}
	return result
}

// ExtractWithContext extracts key sentences and includes contextWindow neighbours.
func (es *ExtractiveSummarizer) ExtractWithContext(text, query string, contextWindow int) string {
	sentences := splitSentencesExtractive(text)
	if len(sentences) == 0 {
		return text
	}

	qEmb := es.embedder.Embed(query)
	scores := es.ScoreSentences(sentences, qEmb)

	topN := len(sentences) / 3
	if topN < 1 {
		topN = 1
	}

	type idxScore struct {
		idx   int
		score float64
	}
	var ranked []idxScore
	for i, s := range scores {
		ranked = append(ranked, idxScore{i, s})
	}
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].score > ranked[j].score
	})

	keyIndices := map[int]bool{}
	for _, r := range ranked[:topN] {
		keyIndices[r.idx] = true
	}

	include := map[int]bool{}
	for ki := range keyIndices {
		for offset := -contextWindow; offset <= contextWindow; offset++ {
			idx := ki + offset
			if idx >= 0 && idx < len(sentences) {
				include[idx] = true
			}
		}
	}

	var indices []int
	for i := range include {
		indices = append(indices, i)
	}
	sort.Ints(indices)

	var result []string
	for _, i := range indices {
		result = append(result, sentences[i])
	}
	return strings.Join(result, " ")
}

// ScoreSentences returns cosine similarity of each sentence to queryEmbedding.
func (es *ExtractiveSummarizer) ScoreSentences(sentences []string, queryEmbedding []float64) []float64 {
	scores := make([]float64, len(sentences))
	for i, s := range sentences {
		scores[i] = cosineSim(es.embedder.Embed(s), queryEmbedding)
	}
	return scores
}

// CompressWithRatio keeps compressionRatio fraction of the original sentences.
func (es *ExtractiveSummarizer) CompressWithRatio(text, query string, compressionRatio float64) string {
	sentences := splitSentencesExtractive(text)
	keepN := int(math.Round(float64(len(sentences)) * compressionRatio))
	if keepN < 1 {
		keepN = 1
	}
	return es.Summarize(text, query, keepN, 20)
}

// ---------------------------------------------------------------------------
// (KeywordEmbedder and Embedder are defined in context_compressor.go)
// ---------------------------------------------------------------------------
