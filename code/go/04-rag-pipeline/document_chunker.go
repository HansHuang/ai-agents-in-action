// document_chunker.go — Document chunking strategies for embedding pipelines.
//
// Splits documents into smaller pieces suitable for embedding and vector search.
// Supports fixed-size, semantic, hierarchical, and sentence-boundary chunking.
//
// Token counting uses a character-based approximation (~4 chars/token).
//
// See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
package ragpipeline

import (
	"fmt"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

// Chunk is a single embeddable piece of a document.
type Chunk struct {
	Text        string
	Index       int
	TokenCount  int
	Metadata    map[string]interface{}
	ParentChunk *Chunk
	Children    []*Chunk
}

func (c *Chunk) String() string {
	snippet := c.Text
	if len(snippet) > 60 {
		snippet = snippet[:60]
	}
	snippet = strings.ReplaceAll(snippet, "\n", "↵")
	return fmt.Sprintf("Chunk(index=%d, tokens=%d, text=%q)", c.Index, c.TokenCount, snippet)
}

// ---------------------------------------------------------------------------
// DocumentChunker
// ---------------------------------------------------------------------------

// DocumentChunker splits documents into chunks suitable for embedding.
type DocumentChunker struct {
	ChunkSize int    // target chunk size in tokens (approx 4 chars/token)
	Overlap   int    // token overlap between adjacent chunks
	Strategy  string // "fixed", "semantic", "hierarchical", or "sentence"
}

var sentenceEndRe = regexp.MustCompile(`(?:[.!?])\s+`)
var sectionBreakRe = regexp.MustCompile(`(?m)(?:^#{1,3}\s)|(?:\n\n)`)

// NewDocumentChunker creates a DocumentChunker with the given parameters.
// strategy defaults to "semantic" if empty.
func NewDocumentChunker(chunkSize, overlap int, strategy string) *DocumentChunker {
	if strategy == "" {
		strategy = "semantic"
	}
	return &DocumentChunker{
		ChunkSize: chunkSize,
		Overlap:   overlap,
		Strategy:  strategy,
	}
}

// countTok returns an approximate token count for text (~4 chars/token).
func countTok(text string) int {
	return approxTokens(text)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// ChunkText splits text into chunks using the configured strategy.
func (dc *DocumentChunker) ChunkText(text string) ([]*Chunk, error) {
	switch dc.Strategy {
	case "fixed":
		return dc.ChunkFixed(text), nil
	case "semantic":
		return dc.ChunkSemantic(text), nil
	case "hierarchical":
		return dc.ChunkHierarchical(text), nil
	case "sentence":
		return dc.ChunkSentence(text), nil
	default:
		return nil, fmt.Errorf("unknown strategy %q; choose from: fixed, semantic, hierarchical, sentence", dc.Strategy)
	}
}

// ChunkFixed creates fixed-size chunks with token overlap.
func (dc *DocumentChunker) ChunkFixed(text string) []*Chunk {
	// Work at character level approximating tokens (4 chars ≈ 1 token).
	charsPerChunk := dc.ChunkSize * 4
	overlapChars := dc.Overlap * 4
	step := charsPerChunk - overlapChars
	if step < 1 {
		step = 1
	}

	var chunks []*Chunk
	idx := 0
	for start := 0; start < len(text); start += step {
		end := start + charsPerChunk
		if end > len(text) {
			end = len(text)
		}
		seg := text[start:end]
		chunks = append(chunks, &Chunk{
			Text:       seg,
			Index:      idx,
			TokenCount: countTok(seg),
		})
		idx++
		if end >= len(text) {
			break
		}
	}
	return chunks
}

// ChunkSemantic splits at natural boundaries: paragraphs, lines, sentences.
func (dc *DocumentChunker) ChunkSemantic(text string) []*Chunk {
	segs := dc.splitSemantic(text)
	return dc.segmentsToChunks(segs)
}

// ChunkHierarchical creates parent–child chunk relationships.
func (dc *DocumentChunker) ChunkHierarchical(text string) []*Chunk {
	// Split at Markdown headings or double-newlines.
	rawSections := sectionBreakRe.Split(text, -1)
	var allChunks []*Chunk
	parentIdx := 0

	for _, section := range rawSections {
		section = strings.TrimSpace(section)
		if section == "" {
			continue
		}
		parent := &Chunk{
			Text:       section,
			Index:      parentIdx,
			TokenCount: countTok(section),
		}
		parentIdx++

		// Build child chunks from this section using semantic strategy.
		childChunker := NewDocumentChunker(dc.ChunkSize, dc.Overlap, "semantic")
		children, _ := childChunker.ChunkText(section)
		for _, child := range children {
			child.Index = parentIdx
			child.ParentChunk = parent
			parent.Children = append(parent.Children, child)
			parentIdx++
		}

		allChunks = append(allChunks, parent)
		allChunks = append(allChunks, parent.Children...)
	}
	return allChunks
}

// ChunkSentence splits at sentence boundaries and groups into chunk-size buckets.
func (dc *DocumentChunker) ChunkSentence(text string) []*Chunk {
	sentences := sentenceEndRe.Split(text, -1)
	var trimmed []string
	for _, s := range sentences {
		s = strings.TrimSpace(s)
		if s != "" {
			trimmed = append(trimmed, s)
		}
	}
	return dc.segmentsToChunks(trimmed)
}

// ChunkWithMetadata chunks text and attaches metadata to every resulting chunk.
func (dc *DocumentChunker) ChunkWithMetadata(text string, metadata map[string]interface{}) ([]*Chunk, error) {
	chunks, err := dc.ChunkText(text)
	if err != nil {
		return nil, err
	}
	for _, c := range chunks {
		cp := make(map[string]interface{}, len(metadata))
		for k, v := range metadata {
			cp[k] = v
		}
		c.Metadata = cp
	}
	return chunks, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (dc *DocumentChunker) splitSemantic(text string) []string {
	var segments []string
	for _, para := range strings.Split(text, "\n\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		if countTok(para) <= dc.ChunkSize {
			segments = append(segments, para)
		} else {
			for _, line := range strings.Split(para, "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				if countTok(line) <= dc.ChunkSize {
					segments = append(segments, line)
				} else {
					for _, sent := range sentenceEndRe.Split(line, -1) {
						sent = strings.TrimSpace(sent)
						if sent != "" {
							segments = append(segments, sent)
						}
					}
				}
			}
		}
	}
	return segments
}

func (dc *DocumentChunker) segmentsToChunks(segments []string) []*Chunk {
	var chunks []*Chunk
	current := ""
	currentToks := 0
	idx := 0

	flush := func() {
		if current != "" {
			chunks = append(chunks, &Chunk{
				Text:       strings.TrimSpace(current),
				Index:      idx,
				TokenCount: currentToks,
			})
			idx++
		}
	}

	for _, seg := range segments {
		segToks := countTok(seg)
		if currentToks+segToks+1 > dc.ChunkSize {
			flush()
			// Seed next chunk with overlap from tail of previous.
			if currentToks > 0 && dc.Overlap > 0 {
				overlapChars := dc.Overlap * 4
				if overlapChars > len(current) {
					overlapChars = len(current)
				}
				overlap := current[len(current)-overlapChars:]
				current = overlap + " " + seg
				currentToks = countTok(current)
			} else {
				current = seg
				currentToks = segToks
			}
		} else {
			if current == "" {
				current = seg
			} else {
				current = current + " " + seg
			}
			currentToks = countTok(current)
		}
	}
	flush()
	return chunks
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

const sampleDocument = `# Return Policy

## Damaged Items

If you receive a damaged or defective item, please contact our support team
within 30 days of delivery. We will arrange a free return and send a
replacement or issue a full refund — your choice.

## Refund Timeline

Once we receive your returned item, refunds are processed within 3–5 business
days. You will receive a confirmation email when the refund is initiated. Funds
may take an additional 2–3 days to appear in your account depending on your bank.

## Exchanges

We do not currently support direct exchanges. To exchange an item, return the
original and place a new order. You will receive the refund before the new
charge appears.

## Contact Support

Our support team is available Monday through Friday, 9 AM – 6 PM Eastern Time.
You can reach us via the in-app chat, email at support@example.com, or by
calling 1-800-555-0199.
`

// RunDocumentChunker demonstrates the document chunking strategies.
func RunDocumentChunker() {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Sample document word count: %d\n", len(strings.Fields(sampleDocument)))
	fmt.Println(strings.Repeat("=", 60))

	for _, strategy := range []string{"fixed", "semantic", "hierarchical"} {
		chunker := NewDocumentChunker(100, 20, strategy)
		chunks, err := chunker.ChunkText(sampleDocument)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		var leafChunks []*Chunk
		for _, c := range chunks {
			if len(c.Children) == 0 {
				leafChunks = append(leafChunks, c)
			}
		}
		totalToks := 0
		for _, c := range leafChunks {
			totalToks += c.TokenCount
		}
		avg := 0.0
		if len(leafChunks) > 0 {
			avg = float64(totalToks) / float64(len(leafChunks))
		}
		fmt.Printf("\n--- Strategy: %s ---\n", strings.ToUpper(strategy))
		fmt.Printf("  Total chunks: %d  (leaf: %d)\n", len(chunks), len(leafChunks))
		fmt.Printf("  Avg leaf tokens: %.1f\n", avg)
		for i, c := range chunks {
			if i >= 3 {
				break
			}
			label := ""
			if len(c.Children) > 0 {
				label = "(parent)"
			}
			fmt.Printf("  [%d]%s %s\n", i, label, c)
		}
		if len(chunks) > 3 {
			fmt.Printf("  … and %d more\n", len(chunks)-3)
		}
	}
}
