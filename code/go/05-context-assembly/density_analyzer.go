// Information Density Analyzer — score text for information density.
//
// Go port of code/python/05-context-assembly/density_analyzer.py
//
// Higher density means more facts per token, which makes for better context.
// A low-density chunk (boilerplate, filler, transitional prose) wastes the
// context budget without contributing to answer quality.
//
// Uses only the standard library and regex — no external NLP dependencies.
//
// See: docs/04-context-engineering/03-context-compression-and-filtering.md
package main

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Stop words
// ---------------------------------------------------------------------------

var densityStopWords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true, "but": true,
	"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
	"with": true, "by": true, "from": true, "as": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "must": true, "can": true, "it": true,
	"its": true, "this": true, "that": true, "these": true, "those": true,
	"i": true, "we": true, "you": true, "he": true, "she": true, "they": true,
	"me": true, "us": true, "him": true, "her": true, "them": true,
	"my": true, "our": true, "your": true, "his": true, "their": true,
	"what": true, "which": true, "who": true, "whom": true, "when": true,
	"where": true, "why": true, "how": true, "all": true, "any": true,
	"both": true, "each": true, "few": true, "more": true, "most": true,
	"other": true, "some": true, "such": true, "no": true, "not": true,
	"only": true, "own": true, "same": true, "so": true, "than": true,
	"too": true, "very": true, "just": true, "also": true, "if": true,
	"then": true, "there": true, "here": true, "about": true, "above": true,
	"after": true, "before": true, "between": true, "into": true,
	"through": true, "during": true, "over": true, "under": true,
	"again": true, "further": true, "once": true, "up": true, "down": true,
	"out": true, "off": true, "while": true, "although": true,
	"because": true, "since": true, "unless": true, "until": true,
	"however": true, "therefore": true, "thus": true, "hence": true,
	"yet": true, "still": true, "already": true,
}

// ---------------------------------------------------------------------------
// Compiled patterns
// ---------------------------------------------------------------------------

var (
	densityNumberRe   = regexp.MustCompile(`\b\d[\d,./\-]*%?(?:\s*(?:USD|EUR|GB|MB|KB|ms|px))?\b`)
	densityBulletRe   = regexp.MustCompile(`(?m)^\s*[-*•]\s+`)
	densityNumberedRe = regexp.MustCompile(`(?m)^\s*\d+[.)]\s+`)
	densityHeaderRe   = regexp.MustCompile(`(?m)^#{1,4}\s+`)
	densityTableRe    = regexp.MustCompile(`(?m)^\s*\|.+\|`)
	densityNamedRe    = regexp.MustCompile(`(?:[^.!?\s])\s+([A-Z][a-z]{2,})\b`)
	densityMidCapRe   = regexp.MustCompile(`(?:\w )[A-Z][a-z]{1,}`)
	densitySentRe     = regexp.MustCompile(`[.!?]+`)
)

// ---------------------------------------------------------------------------
// DensityScore
// ---------------------------------------------------------------------------

// DensityScore is a multi-dimensional information density score for a text passage.
type DensityScore struct {
	Overall        float64
	FactDensity    float64
	StructureScore float64
	FillerRatio    float64 // lower is better
	Specificity    float64
}

// IsHighQuality returns true if this text is worth including in LLM context.
func (d DensityScore) IsHighQuality(threshold float64) bool {
	return d.Overall >= threshold
}

// Explain returns a human-readable explanation of each sub-score.
func (d DensityScore) Explain() string {
	verdict := "[PASS — include]"
	if !d.IsHighQuality(0.4) {
		verdict = "[FAIL — drop]"
	}
	return strings.Join([]string{
		fmt.Sprintf("Overall density:     %.2f  %s", d.Overall, verdict),
		fmt.Sprintf("Fact density:        %.2f  (entities + numbers / words)", d.FactDensity),
		fmt.Sprintf("Structure score:     %.2f  (bullets, tables, headers)", d.StructureScore),
		fmt.Sprintf("Filler ratio:        %.2f  (stop words; lower = better)", d.FillerRatio),
		fmt.Sprintf("Specificity:         %.2f  (capitalised terms + numbers)", d.Specificity),
	}, "\n")
}

// ---------------------------------------------------------------------------
// InformationDensityAnalyzer
// ---------------------------------------------------------------------------

// InformationDensityAnalyzer scores text on five density dimensions.
// No external NLP libraries required.
type InformationDensityAnalyzer struct{}

// NewInformationDensityAnalyzer creates a new InformationDensityAnalyzer.
func NewInformationDensityAnalyzer() *InformationDensityAnalyzer {
	return &InformationDensityAnalyzer{}
}

// Score returns a DensityScore for text.
func (a *InformationDensityAnalyzer) Score(text string) DensityScore {
	words := strings.Fields(text)
	if len(words) == 0 {
		return DensityScore{}
	}

	factDensity := a.factDensity(text, words)
	structure := a.structureScore(text)
	fillerRatio := a.fillerRatio(words)
	specificity := a.specificity(text, words)

	// Weighted composite (weights tuned empirically on 50-document set)
	overall := factDensity*0.35 +
		structure*0.15 +
		(1.0-fillerRatio)*0.30 +
		specificity*0.20

	overall = math.Max(0.0, math.Min(1.0, overall))

	round4 := func(f float64) float64 {
		return math.Round(f*10000) / 10000
	}

	return DensityScore{
		Overall:        round4(overall),
		FactDensity:    round4(factDensity),
		StructureScore: round4(structure),
		FillerRatio:    round4(fillerRatio),
		Specificity:    round4(specificity),
	}
}

// ScoredText pairs text with its density score for sorting.
type ScoredText struct {
	Text  string
	Score DensityScore
}

// Compare scores texts and returns them sorted from most to least dense.
func (a *InformationDensityAnalyzer) Compare(texts []string) []ScoredText {
	scored := make([]ScoredText, len(texts))
	for i, t := range texts {
		scored[i] = ScoredText{Text: t, Score: a.Score(t)}
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score.Overall > scored[j].Score.Overall
	})
	return scored
}

// FindLowDensitySections returns paragraphs whose density falls below minDensity.
func (a *InformationDensityAnalyzer) FindLowDensitySections(text string, minDensity float64) []string {
	var low []string
	for _, para := range regexp.MustCompile(`\n\s*\n`).Split(text, -1) {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		if a.Score(para).Overall < minDensity {
			low = append(low, para)
		}
	}
	return low
}

// EstimateReadability returns a 0–1 estimate of how easily an LLM can extract facts.
func (a *InformationDensityAnalyzer) EstimateReadability(text string) float64 {
	if strings.TrimSpace(text) == "" {
		return 0.0
	}

	structure := a.structureScore(text)

	var sentences []string
	for _, s := range densitySentRe.Split(text, -1) {
		s = strings.TrimSpace(s)
		if s != "" {
			sentences = append(sentences, s)
		}
	}
	totalWords := 0
	for _, s := range sentences {
		totalWords += len(strings.Fields(s))
	}
	avgLen := float64(totalWords) / math.Max(float64(len(sentences)), 1)
	lengthScore := math.Max(0.0, 1.0-(avgLen-10)/40)

	result := structure*0.4 + lengthScore*0.6
	return math.Round(result*10000) / 10000
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

func (a *InformationDensityAnalyzer) factDensity(text string, words []string) float64 {
	numbers := len(densityNumberRe.FindAllString(text, -1))
	// Named entities: capitalised words not at sentence start
	named := len(densityNamedRe.FindAllString(text, -1))
	return math.Min(1.0, float64(numbers+named)/math.Max(float64(len(words)), 1))
}

func (a *InformationDensityAnalyzer) structureScore(text string) float64 {
	bullets := len(densityBulletRe.FindAllString(text, -1))
	numbered := len(densityNumberedRe.FindAllString(text, -1))
	headers := len(densityHeaderRe.FindAllString(text, -1))
	tables := len(densityTableRe.FindAllString(text, -1))
	total := bullets + numbered + headers + tables
	lines := strings.Count(text, "\n") + 1
	return math.Min(1.0, float64(total)/math.Max(float64(lines), 1))
}

func (a *InformationDensityAnalyzer) fillerRatio(words []string) float64 {
	trimChars := ".,;:!?\"'()[]{}"
	count := 0
	for _, w := range words {
		lower := strings.ToLower(strings.Trim(w, trimChars))
		if densityStopWords[lower] {
			count++
		}
	}
	return float64(count) / math.Max(float64(len(words)), 1)
}

func (a *InformationDensityAnalyzer) specificity(text string, words []string) float64 {
	numbers := len(densityNumberRe.FindAllString(text, -1))
	midCaps := len(densityMidCapRe.FindAllString(text, -1))
	return math.Min(1.0, float64(numbers+midCaps)/math.Max(float64(len(words)), 1))
}
