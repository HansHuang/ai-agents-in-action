// hybrid_search.go — Hybrid search combining vector similarity with BM25.
//
// Pure vector search fails when exact terms matter. Hybrid search blends
// ranked lists from a vector index and a keyword (BM25) index.
//
// Two fusion strategies:
//   - Weighted sum: normalise each list to [0,1] then blend with configurable weights.
//   - Reciprocal Rank Fusion (RRF): position-based, robust to score scale differences.
//
// BM25 is implemented from scratch — no external dependencies required.
//
// See: docs/05-the-tool-ecosystem/02-vector-databases.md
package ragpipeline

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// HybridSearchResult
// ---------------------------------------------------------------------------

// HybridSearchResult is a single result from hybrid search.
type HybridSearchResult struct {
	ID            string
	Text          string
	CombinedScore float64
	VectorScore   float64
	KeywordScore  float64
	VectorRank    int // 1-based rank in the vector list (0 = not in list)
	KeywordRank   int // 1-based rank in the keyword list (0 = not in list)
	Metadata      map[string]interface{}
}

// ---------------------------------------------------------------------------
// BM25 implementation
// ---------------------------------------------------------------------------

var nonAlphaNum = regexp.MustCompile(`[^a-z0-9]+`)

func tokenize(text string) []string {
	lower := strings.ToLower(text)
	tokens := nonAlphaNum.Split(lower, -1)
	var result []string
	for _, t := range tokens {
		if t != "" {
			result = append(result, t)
		}
	}
	return result
}

type bm25Index struct {
	docs      []VecDocument
	corpus    [][]string
	avgDocLen float64
	idf       map[string]float64   // IDF per term
	tf        []map[string]float64 // term freq per doc
	k1        float64
	b         float64
}

func newBM25Index(docs []VecDocument) *bm25Index {
	const k1 = 1.5
	const b = 0.75

	corpus := make([][]string, len(docs))
	tf := make([]map[string]float64, len(docs))
	totalLen := 0

	for i, doc := range docs {
		toks := tokenize(doc.Text)
		corpus[i] = toks
		totalLen += len(toks)
		freq := make(map[string]float64)
		for _, t := range toks {
			freq[t]++
		}
		tf[i] = freq
	}

	avgDocLen := 0.0
	if len(docs) > 0 {
		avgDocLen = float64(totalLen) / float64(len(docs))
	}

	// IDF: log((N - df + 0.5) / (df + 0.5) + 1)
	N := float64(len(docs))
	dfMap := make(map[string]int)
	for _, freq := range tf {
		for term := range freq {
			dfMap[term]++
		}
	}
	idf := make(map[string]float64, len(dfMap))
	for term, df := range dfMap {
		idf[term] = math.Log((N-float64(df)+0.5)/(float64(df)+0.5) + 1)
	}

	return &bm25Index{
		docs:      docs,
		corpus:    corpus,
		avgDocLen: avgDocLen,
		idf:       idf,
		tf:        tf,
		k1:        k1,
		b:         b,
	}
}

func (idx *bm25Index) score(queryTokens []string, docIdx int) float64 {
	docLen := float64(len(idx.corpus[docIdx]))
	freq := idx.tf[docIdx]
	score := 0.0
	for _, t := range queryTokens {
		idf, ok := idx.idf[t]
		if !ok {
			continue
		}
		f := freq[t]
		num := f * (idx.k1 + 1)
		denom := f + idx.k1*(1-idx.b+idx.b*docLen/idx.avgDocLen)
		score += idf * num / denom
	}
	return score
}

type bm25Hit struct {
	doc   VecDocument
	score float64
}

func (idx *bm25Index) search(query string, k int) []bm25Hit {
	tokens := tokenize(query)
	results := make([]bm25Hit, len(idx.docs))
	for i, doc := range idx.docs {
		results[i] = bm25Hit{doc: doc, score: idx.score(tokens, i)}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	if k < len(results) {
		results = results[:k]
	}
	return results
}

// ---------------------------------------------------------------------------
// HybridSearch
// ---------------------------------------------------------------------------

// HybridSearch combines vector search and BM25 keyword search.
type HybridSearch struct {
	VectorDB      VectorDatabase
	VectorWeight  float64
	KeywordWeight float64

	keywordIdx *bm25Index
}

// NewHybridSearch creates a HybridSearch instance.
// vectorWeight is in [0,1]; keyword weight = 1 - vectorWeight.
func NewHybridSearch(vectorDB VectorDatabase, vectorWeight float64) (*HybridSearch, error) {
	if vectorWeight < 0 || vectorWeight > 1 {
		return nil, fmt.Errorf("vectorWeight must be in [0, 1]")
	}
	return &HybridSearch{
		VectorDB:      vectorDB,
		VectorWeight:  vectorWeight,
		KeywordWeight: 1 - vectorWeight,
	}, nil
}

// BuildKeywordIndex builds a BM25 index over the provided documents.
// Must be called before Search or SearchWithRRF.
func (hs *HybridSearch) BuildKeywordIndex(documents []VecDocument) {
	hs.keywordIdx = newBM25Index(documents)
}

func (hs *HybridSearch) requireIndex() error {
	if hs.keywordIdx == nil {
		return fmt.Errorf("call BuildKeywordIndex() before searching")
	}
	return nil
}

func normalizeScores(scores []float64) []float64 {
	if len(scores) == 0 {
		return nil
	}
	lo, hi := scores[0], scores[0]
	for _, s := range scores[1:] {
		if s < lo {
			lo = s
		}
		if s > hi {
			hi = s
		}
	}
	if hi == lo {
		out := make([]float64, len(scores))
		return out
	}
	out := make([]float64, len(scores))
	for i, s := range scores {
		out[i] = (s - lo) / (hi - lo)
	}
	return out
}

// Search performs hybrid search using a weighted sum of normalised scores.
func (hs *HybridSearch) Search(query string, queryEmbedding []float64, k int, filterMetadata map[string]interface{}) ([]HybridSearchResult, error) {
	if err := hs.requireIndex(); err != nil {
		return nil, err
	}
	fetchK := k * 2

	// Vector search.
	vecResults, err := hs.VectorDB.Search(queryEmbedding, fetchK, filterMetadata)
	if err != nil {
		return nil, err
	}
	vScoreRaw := make([]float64, len(vecResults))
	for i, r := range vecResults {
		vScoreRaw[i] = r.Score
	}
	vNorm := normalizeScores(vScoreRaw)
	vMap := make(map[string][2]float64) // id → {norm_score, rank}
	for rank, r := range vecResults {
		vMap[r.ID] = [2]float64{vNorm[rank], float64(rank + 1)}
	}

	// Keyword search.
	kwResults := hs.keywordIdx.search(query, fetchK)
	kScoreRaw := make([]float64, len(kwResults))
	for i, r := range kwResults {
		kScoreRaw[i] = r.score
	}
	kNorm := normalizeScores(kScoreRaw)
	kMap := make(map[string][2]float64)
	for rank, r := range kwResults {
		kMap[r.doc.ID] = [2]float64{kNorm[rank], float64(rank + 1)}
	}

	// Merge.
	allIDs := make(map[string]bool)
	for _, r := range vecResults {
		allIDs[r.ID] = true
	}
	for _, r := range kwResults {
		allIDs[r.doc.ID] = true //nolint
	}

	merged := make([]HybridSearchResult, 0, len(allIDs))
	for docID := range allIDs {
		vEntry := vMap[docID]
		kEntry := kMap[docID]
		combined := hs.VectorWeight*vEntry[0] + hs.KeywordWeight*kEntry[0]
		text, meta := hs.lookupTextMeta(docID, vecResults)
		merged = append(merged, HybridSearchResult{
			ID:            docID,
			Text:          text,
			CombinedScore: combined,
			VectorScore:   vEntry[0],
			KeywordScore:  kEntry[0],
			VectorRank:    int(vEntry[1]),
			KeywordRank:   int(kEntry[1]),
			Metadata:      meta,
		})
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].CombinedScore > merged[j].CombinedScore })
	if k < len(merged) {
		merged = merged[:k]
	}
	return merged, nil
}

// SearchWithRRF performs hybrid search using Reciprocal Rank Fusion.
func (hs *HybridSearch) SearchWithRRF(query string, queryEmbedding []float64, k, rrfK int, filterMetadata map[string]interface{}) ([]HybridSearchResult, error) {
	if err := hs.requireIndex(); err != nil {
		return nil, err
	}
	fetchK := k * 2

	vecResults, err := hs.VectorDB.Search(queryEmbedding, fetchK, filterMetadata)
	if err != nil {
		return nil, err
	}
	vRank := make(map[string]int)
	for rank, r := range vecResults {
		vRank[r.ID] = rank + 1
	}
	vScoreMap := make(map[string]float64)
	for _, r := range vecResults {
		vScoreMap[r.ID] = r.Score
	}

	kwResults := hs.keywordIdx.search(query, fetchK)
	kRank := make(map[string]int)
	for rank, r := range kwResults {
		kRank[r.doc.ID] = rank + 1
	}
	kScoreMap := make(map[string]float64)
	for _, r := range kwResults {
		kScoreMap[r.doc.ID] = r.score
	}

	// Normalize for reporting.
	vScores := make([]float64, len(vecResults))
	for i, r := range vecResults {
		vScores[i] = r.Score
	}
	kScores := make([]float64, len(kwResults))
	for i, r := range kwResults {
		kScores[i] = r.score
	}
	vNorm := normalizeScores(vScores)
	kNorm := normalizeScores(kScores)
	vNormMap := make(map[string]float64)
	for i, r := range vecResults {
		vNormMap[r.ID] = vNorm[i]
	}
	kNormMap := make(map[string]float64)
	for i, r := range kwResults {
		kNormMap[r.doc.ID] = kNorm[i]
	}

	allIDs := make(map[string]bool)
	for _, r := range vecResults {
		allIDs[r.ID] = true
	}
	for _, r := range kwResults {
		allIDs[r.doc.ID] = true
	}

	results := make([]HybridSearchResult, 0, len(allIDs))
	for docID := range allIDs {
		rrf := 0.0
		if rank, ok := vRank[docID]; ok {
			rrf += 1.0 / float64(rrfK+rank)
		}
		if rank, ok := kRank[docID]; ok {
			rrf += 1.0 / float64(rrfK+rank)
		}
		text, meta := hs.lookupTextMeta(docID, vecResults)
		results = append(results, HybridSearchResult{
			ID:            docID,
			Text:          text,
			CombinedScore: rrf,
			VectorScore:   vNormMap[docID],
			KeywordScore:  kNormMap[docID],
			VectorRank:    vRank[docID],
			KeywordRank:   kRank[docID],
			Metadata:      meta,
		})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].CombinedScore > results[j].CombinedScore })
	if k < len(results) {
		results = results[:k]
	}
	return results, nil
}

// ExplainResult returns a human-readable explanation for a hybrid search result.
func (hs *HybridSearch) ExplainResult(query string, r HybridSearchResult) string {
	vPart := fmt.Sprintf("not in vector results (score 0, weight %.0f%%)", hs.VectorWeight*100)
	if r.VectorRank > 0 {
		vPart = fmt.Sprintf("vector rank #%d (score %.4f, weight %.0f%%)", r.VectorRank, r.VectorScore, hs.VectorWeight*100)
	}
	kPart := fmt.Sprintf("not in keyword results (score 0, weight %.0f%%)", hs.KeywordWeight*100)
	if r.KeywordRank > 0 {
		kPart = fmt.Sprintf("keyword rank #%d (score %.4f, weight %.0f%%)", r.KeywordRank, r.KeywordScore, hs.KeywordWeight*100)
	}
	return fmt.Sprintf("[id=%q] combined=%.4f | %s + %s | query: %q",
		r.ID, r.CombinedScore, vPart, kPart, query)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (hs *HybridSearch) lookupTextMeta(docID string, vecResults []VecSearchResult) (string, map[string]interface{}) {
	for _, r := range vecResults {
		if r.ID == docID {
			return r.Text, r.Metadata
		}
	}
	if hs.keywordIdx != nil {
		for _, doc := range hs.keywordIdx.docs {
			if doc.ID == docID {
				return doc.Text, doc.Metadata
			}
		}
	}
	return "", nil
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunHybridSearch demonstrates that hybrid search ranks the exact-match document
// higher than pure vector search when the query contains a specific keyword.
func RunHybridSearch() {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("Hybrid Search Demo — 'Error 503'")
	fmt.Println(strings.Repeat("=", 60))

	// Build a fake vector space: doc_A and doc_B have very similar embeddings,
	// but only doc_A mentions "503" in its text.
	dim := 8
	base := make([]float64, dim)
	for i := range base {
		base[i] = 1.0 / math.Sqrt(float64(dim))
	}
	similar := make([]float64, dim)
	copy(similar, base)
	similar[0] += 0.05
	norm := 0.0
	for _, v := range similar {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	for i := range similar {
		similar[i] /= norm
	}

	docs := []VecDocument{
		{ID: "doc_A", Text: "Server returns error 503 when traffic exceeds limits", Embedding: base},
		{ID: "doc_B", Text: "Server errors occur under heavy load", Embedding: similar},
	}
	for i := 0; i < 8; i++ {
		v := make([]float64, dim)
		v[i%dim] = 1.0
		docs = append(docs, VecDocument{
			ID:        fmt.Sprintf("noise_%d", i),
			Text:      fmt.Sprintf("Unrelated content about topic %d", i),
			Embedding: v,
		})
	}

	db := NewInMemoryVecStore()
	db.Insert(docs)

	hs, _ := NewHybridSearch(db, 0.7)
	hs.BuildKeywordIndex(docs)

	queryText := "Error 503"
	queryEmb := base

	fmt.Println("\n[Pure vector search — top 3]")
	pure, _ := db.Search(queryEmb, 3, nil)
	for i, r := range pure {
		fmt.Printf("  #%d id=%q score=%.4f  text=%q\n", i+1, r.ID, r.Score, r.Text)
	}

	fmt.Println("\n[Weighted hybrid (70% vector / 30% keyword) — top 3]")
	hybrid, _ := hs.Search(queryText, queryEmb, 3, nil)
	for i, r := range hybrid {
		fmt.Printf("  #%d id=%q combined=%.4f  text=%q\n", i+1, r.ID, r.CombinedScore, r.Text)
		fmt.Printf("       %s\n", hs.ExplainResult(queryText, r))
	}

	fmt.Println("\n[RRF hybrid — top 3]")
	rrf, _ := hs.SearchWithRRF(queryText, queryEmb, 3, 60, nil)
	for i, r := range rrf {
		fmt.Printf("  #%d id=%q combined=%.6f  text=%q\n", i+1, r.ID, r.CombinedScore, r.Text)
	}

	findRank := func(results []HybridSearchResult, id string) int {
		for i, r := range results {
			if r.ID == id {
				return i + 1
			}
		}
		return -1
	}
	vecRankDocA := -1
	for i, r := range pure {
		if r.ID == "doc_A" {
			vecRankDocA = i + 1
			break
		}
	}
	fmt.Printf("\ndoc_A ('error 503') rank: vector=#%d  hybrid=#%d  rrf=#%d\n",
		vecRankDocA, findRank(hybrid, "doc_A"), findRank(rrf, "doc_A"))
}
