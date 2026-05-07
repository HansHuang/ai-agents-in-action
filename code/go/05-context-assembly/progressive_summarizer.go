// Progressive Summarizer — incremental, layered conversation summarization.
//
// Go port of code/python/05-context-assembly/progressive_summarizer.py
//
// Older turns are compressed more aggressively than recent turns.
//
// Layers:
//   - Layer 0 (verbatim):   Last verbatimTurns turns, exact wording
//   - Layer 1 (detailed):   Moderate-detail summary of older turns
//   - Layer 2 (compressed): Key facts only
//   - Layer 3 (archival):   Highly compressed essence
//
// LLM calls are made when OPENAI_API_KEY is set; a deterministic
// keyword-extraction fallback is used otherwise.
//
// See: docs/04-context-engineering/04-multi-turn-context-management.md
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// LLM helper
// ---------------------------------------------------------------------------

func llmAvailable() bool {
	return os.Getenv("OPENAI_API_KEY") != ""
}

// llmSummarize calls the OpenAI API with a prompt and returns the response.
func llmSummarize(prompt, model string) (string, error) {
	type oaiMsg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type oaiReq struct {
		Model       string   `json:"model"`
		Messages    []oaiMsg `json:"messages"`
		Temperature float64  `json:"temperature"`
	}
	type oaiChoice struct {
		Message oaiMsg `json:"message"`
	}
	type oaiResp struct {
		Choices []oaiChoice `json:"choices"`
	}

	reqBody, _ := json.Marshal(oaiReq{
		Model:       model,
		Messages:    []oaiMsg{{Role: "user", Content: prompt}},
		Temperature: 0.3,
	})

	req, err := http.NewRequest("POST",
		"https://api.openai.com/v1/chat/completions",
		bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	var parsed oaiResp
	if err := json.Unmarshal(body, &parsed); err != nil || len(parsed.Choices) == 0 {
		return "", fmt.Errorf("parse error or no choices")
	}
	return strings.TrimSpace(parsed.Choices[0].Message.Content), nil
}

// ---------------------------------------------------------------------------
// Deterministic fallback
// ---------------------------------------------------------------------------

var summarizerStopWords = map[string]bool{
	"that": true, "this": true, "with": true, "from": true, "have": true,
	"will": true, "been": true, "were": true, "they": true, "them": true,
	"user": true, "assistant": true, "just": true, "very": true, "also": true,
}

func extractKeySentences(text string, maxSentences int) string {
	reWord := regexp.MustCompile(`\b\w{4,}\b`)
	reSplit := regexp.MustCompile(`(?:[.!?])\s+`)
	sentences := reSplit.Split(strings.TrimSpace(text), -1)
	if len(sentences) == 0 {
		return text
	}

	type scored struct {
		idx int
		sc  float64
	}
	var list []scored
	for i, s := range sentences {
		words := reWord.FindAllString(strings.ToLower(s), -1)
		content := 0
		for _, w := range words {
			if !summarizerStopWords[w] {
				content++
			}
		}
		sc := 0.0
		if len(words) > 0 {
			sc = float64(content) / float64(len(words))
		}
		list = append(list, scored{i, sc})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].sc > list[j].sc })

	keep := make([]int, 0, maxSentences)
	for i := 0; i < len(list) && i < maxSentences; i++ {
		keep = append(keep, list[i].idx)
	}
	sort.Ints(keep)

	var parts []string
	for _, idx := range keep {
		parts = append(parts, sentences[idx])
	}
	return strings.Join(parts, " ")
}

func fallbackUpdateSummary(existing, newTurn string) string {
	combined := strings.TrimSpace(existing + "\n" + newTurn)
	return extractKeySentences(combined, 8)
}

func fallbackCompress(content string) string {
	return extractKeySentences(content, 5)
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const layerUpdatePrompt = `Update this conversation summary with new information.
Preserve: goals, decisions made, specific data (numbers, dates, names),
user preferences, agent recommendations, pending tasks.

Discard: small talk, repeated information, exact wording of resolved questions.

Existing summary: %s
New information: %s

Updated summary (keep approximately the same length):`

const layerCompressionPrompt = `Compress this detailed conversation summary into a shorter version.
Keep only: the main goal, key decisions, critical facts, and unresolved items.
The compressed version should be about half the length.

Detailed summary: %s

Compressed summary:`

// ---------------------------------------------------------------------------
// ProgressiveSummarizer
// ---------------------------------------------------------------------------

const numLayers = 3

// Turn is a single conversation exchange.
type Turn struct {
	UserMsg      string `json:"user_msg"`
	AssistantMsg string `json:"assistant_msg"`
}

// SummarizerStats holds statistics about the summarizer's state.
type SummarizerStats struct {
	TotalTurnsProcessed int `json:"total_turns_processed"`
	VerbatimTurns       int `json:"verbatim_turns"`
	Layer1Tokens        int `json:"layer_1_tokens"`
	Layer2Tokens        int `json:"layer_2_tokens"`
	Layer3Tokens        int `json:"layer_3_tokens"`
	TotalContextTokens  int `json:"total_context_tokens"`
}

// SummarizerData is the serialisable form of ProgressiveSummarizer.
type SummarizerData struct {
	Verbatim        []Turn   `json:"verbatim"`
	Layers          []string `json:"layers"`
	TotalTurns      int      `json:"total_turns"`
	VerbatimTurns   int      `json:"verbatim_turns"`
	LayerSize       int      `json:"layer_size"`
	LayerTokenLimit int      `json:"layer_token_limit"`
}

// ProgressiveSummarizer summarises conversations incrementally across multiple layers.
type ProgressiveSummarizer struct {
	VerbatimTurns   int
	LayerSize       int
	Model           string
	LayerTokenLimit int

	Verbatim   []Turn
	Layers     []string
	totalTurns int
}

// NewProgressiveSummarizer returns a ProgressiveSummarizer with the given settings.
func NewProgressiveSummarizer(verbatimTurns, layerSize int, model string, layerTokenLimit int) *ProgressiveSummarizer {
	return &ProgressiveSummarizer{
		VerbatimTurns:   verbatimTurns,
		LayerSize:       layerSize,
		Model:           model,
		LayerTokenLimit: layerTokenLimit,
		Verbatim:        []Turn{},
		Layers:          make([]string, numLayers),
	}
}

// AddTurn adds a turn and rebalances layers if needed.
func (p *ProgressiveSummarizer) AddTurn(userMsg, assistantMsg string) {
	p.totalTurns++
	p.Verbatim = append(p.Verbatim, Turn{userMsg, assistantMsg})

	if len(p.Verbatim) > p.VerbatimTurns {
		oldest := p.Verbatim[0]
		p.Verbatim = p.Verbatim[1:]
		turnText := "User: " + oldest.UserMsg + "\nAssistant: " + oldest.AssistantMsg
		p.incorporateIntoLayer(0, turnText)
	}
}

// GetContext returns the full context string for prompt injection.
func (p *ProgressiveSummarizer) GetContext() string {
	var parts []string
	labels := []string{"Early conversation", "Earlier conversation", "Recent conversation"}

	for i := numLayers - 1; i >= 0; i-- {
		content := strings.TrimSpace(p.Layers[i])
		if content != "" {
			parts = append(parts, fmt.Sprintf("[%s:\n%s]", labels[i], content))
		}
	}

	if len(p.Verbatim) > 0 {
		var lines []string
		for _, t := range p.Verbatim {
			lines = append(lines, "User: "+t.UserMsg)
			lines = append(lines, "Assistant: "+t.AssistantMsg)
		}
		parts = append(parts, "[Most recent turns:\n"+strings.Join(lines, "\n")+"]")
	}

	return strings.Join(parts, "\n\n")
}

// GetStats returns statistics about the summarizer's state.
func (p *ProgressiveSummarizer) GetStats() SummarizerStats {
	var layerTokens [numLayers]int
	for i, l := range p.Layers {
		toks, _ := CountTokens(l, "gpt-4o")
		layerTokens[i] = toks
	}
	var sb strings.Builder
	for _, t := range p.Verbatim {
		sb.WriteString(t.UserMsg + " " + t.AssistantMsg + " ")
	}
	vToks, _ := CountTokens(sb.String(), "gpt-4o")
	total := vToks
	for _, t := range layerTokens {
		total += t
	}
	return SummarizerStats{
		TotalTurnsProcessed: p.totalTurns,
		VerbatimTurns:       len(p.Verbatim),
		Layer1Tokens:        layerTokens[0],
		Layer2Tokens:        layerTokens[1],
		Layer3Tokens:        layerTokens[2],
		TotalContextTokens:  total,
	}
}

// ToDict serialises the summarizer for persistence.
func (p *ProgressiveSummarizer) ToDict() SummarizerData {
	return SummarizerData{
		Verbatim:        append([]Turn{}, p.Verbatim...),
		Layers:          append([]string{}, p.Layers...),
		TotalTurns:      p.totalTurns,
		VerbatimTurns:   p.VerbatimTurns,
		LayerSize:       p.LayerSize,
		LayerTokenLimit: p.LayerTokenLimit,
	}
}

// ProgressiveSummarizerFromDict restores a summarizer from serialised data.
func ProgressiveSummarizerFromDict(data SummarizerData) *ProgressiveSummarizer {
	p := NewProgressiveSummarizer(
		data.VerbatimTurns, data.LayerSize, "gpt-4o-mini", data.LayerTokenLimit,
	)
	if data.Verbatim != nil {
		p.Verbatim = append([]Turn{}, data.Verbatim...)
	}
	if data.Layers != nil {
		p.Layers = append([]string{}, data.Layers...)
	}
	// Pad to numLayers if needed
	for len(p.Layers) < numLayers {
		p.Layers = append(p.Layers, "")
	}
	p.totalTurns = data.TotalTurns
	return p
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (p *ProgressiveSummarizer) incorporateIntoLayer(layerIndex int, turnText string) {
	if layerIndex >= numLayers {
		p.Layers[numLayers-1] = p.compressContent(p.Layers[numLayers-1], turnText, numLayers)
		return
	}
	p.Layers[layerIndex] = p.updateSummary(p.Layers[layerIndex], turnText)

	toks, _ := CountTokens(p.Layers[layerIndex], "gpt-4o")
	if toks > p.LayerTokenLimit {
		p.cascadeOverflow(layerIndex)
	}
}

func (p *ProgressiveSummarizer) cascadeOverflow(fromLayer int) {
	toLayer := fromLayer + 1
	if toLayer >= numLayers {
		p.Layers[fromLayer] = p.compressContent(p.Layers[fromLayer], "", fromLayer+1)
		return
	}
	p.Layers[toLayer] = p.compressContent(p.Layers[toLayer], p.Layers[fromLayer], toLayer+1)
	p.Layers[fromLayer] = ""

	toks, _ := CountTokens(p.Layers[toLayer], "gpt-4o")
	if toks > p.LayerTokenLimit {
		p.cascadeOverflow(toLayer)
	}
}

func (p *ProgressiveSummarizer) updateSummary(existing, newTurn string) string {
	if existing == "" {
		return newTurn
	}
	if llmAvailable() {
		prompt := fmt.Sprintf(layerUpdatePrompt,
			orDefault(existing, "(none)"), newTurn)
		if result, err := llmSummarize(prompt, p.Model); err == nil {
			return result
		}
	}
	return fallbackUpdateSummary(existing, newTurn)
}

func (p *ProgressiveSummarizer) compressContent(older, newer string, depth int) string {
	combined := strings.TrimSpace(older + "\n" + newer)
	if combined == "" {
		return ""
	}
	if llmAvailable() {
		prompt := fmt.Sprintf(layerCompressionPrompt, combined)
		if result, err := llmSummarize(prompt, p.Model); err == nil {
			return result
		}
	}
	max := 8 - depth*2
	if max < 3 {
		max = 3
	}
	return extractKeySentences(combined, max)
}

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}
