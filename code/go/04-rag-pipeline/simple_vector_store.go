// SimpleVectorStore — in-memory vector store with cosine similarity search.
//
// Suitable for prototyping and datasets up to ~10,000 documents.
// For production, switch to a dedicated vector database (Qdrant, Pinecone, etc.).
//
// See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
package ragpipeline

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"crypto/rand"
	"encoding/hex"
)

// ---------------------------------------------------------------------------
// StoredDocument
// ---------------------------------------------------------------------------

// StoredDocument represents a single document in the vector store.
type StoredDocument struct {
	ID        string                 `json:"id"`
	Text      string                 `json:"text"`
	Embedding []float64              `json:"embedding"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// VectorSearchResult is returned by search methods.
type VectorSearchResult struct {
	ID       string
	Text     string
	Score    float64
	Metadata map[string]interface{}
}

// ---------------------------------------------------------------------------
// SimpleVectorStore
// ---------------------------------------------------------------------------

// SimpleVectorStore is an in-memory vector store for prototyping.
//
// All searches are O(n) brute-force cosine similarity. Acceptable for up to
// ~10,000 documents; use a dedicated vector database beyond that.
type SimpleVectorStore struct {
	documents []StoredDocument
}

// NewSimpleVectorStore creates an empty SimpleVectorStore.
func NewSimpleVectorStore() *SimpleVectorStore {
	return &SimpleVectorStore{}
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

// Add stores a single document and returns its generated UUID.
func (s *SimpleVectorStore) Add(text string, embedding []float64, metadata map[string]interface{}) (string, error) {
	id, err := newUUID()
	if err != nil {
		return "", fmt.Errorf("generate UUID: %w", err)
	}
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	s.documents = append(s.documents, StoredDocument{
		ID:        id,
		Text:      text,
		Embedding: embedding,
		Metadata:  metadata,
	})
	return id, nil
}

// AddBatch stores multiple documents at once.
// Each item must have "text" and "embedding" keys; "metadata" is optional.
func (s *SimpleVectorStore) AddBatch(items []map[string]interface{}) ([]string, error) {
	ids := make([]string, 0, len(items))
	for _, item := range items {
		text, _ := item["text"].(string)
		emb, _ := item["embedding"].([]float64)
		meta, _ := item["metadata"].(map[string]interface{})
		id, err := s.Add(text, emb, meta)
		if err != nil {
			return ids, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// Delete removes a document by its ID. Returns true if a document was removed.
func (s *SimpleVectorStore) Delete(docID string) bool {
	before := len(s.documents)
	filtered := s.documents[:0]
	for _, d := range s.documents {
		if d.ID != docID {
			filtered = append(filtered, d)
		}
	}
	s.documents = filtered
	return len(s.documents) < before
}

// Clear removes all documents from the store.
func (s *SimpleVectorStore) Clear() {
	s.documents = s.documents[:0]
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

// Search returns the k most similar documents to queryEmbedding.
// If filterMetadata is non-nil, only documents whose metadata contains all
// the specified key-value pairs are considered.
func (s *SimpleVectorStore) Search(
	queryEmbedding []float64,
	k int,
	filterMetadata map[string]interface{},
) ([]VectorSearchResult, error) {
	candidates := s.applyFilter(filterMetadata)
	if len(candidates) == 0 {
		return nil, nil
	}

	type scored struct {
		doc   StoredDocument
		score float64
	}
	results := make([]scored, 0, len(candidates))
	for _, doc := range candidates {
		score, err := vectorCosineSimilarity(queryEmbedding, doc.Embedding)
		if err != nil {
			return nil, err
		}
		results = append(results, scored{doc: doc, score: score})
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if k > len(results) {
		k = len(results)
	}
	out := make([]VectorSearchResult, k)
	for i := range out {
		out[i] = VectorSearchResult{
			ID:       results[i].doc.ID,
			Text:     results[i].doc.Text,
			Score:    results[i].score,
			Metadata: results[i].doc.Metadata,
		}
	}
	return out, nil
}

// SearchWithThreshold returns only results at or above the similarity threshold.
func (s *SimpleVectorStore) SearchWithThreshold(
	queryEmbedding []float64,
	threshold float64,
	k int,
) ([]VectorSearchResult, error) {
	results, err := s.Search(queryEmbedding, k, nil)
	if err != nil {
		return nil, err
	}
	filtered := results[:0]
	for _, r := range results {
		if r.Score >= threshold {
			filtered = append(filtered, r)
		}
	}
	return filtered, nil
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

// Count returns the number of documents currently stored.
func (s *SimpleVectorStore) Count() int {
	return len(s.documents)
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

// Save persists the store to a JSON file.
func (s *SimpleVectorStore) Save(filepath string) error {
	data, err := json.Marshal(s.documents)
	if err != nil {
		return fmt.Errorf("marshal documents: %w", err)
	}
	if err := os.WriteFile(filepath, data, 0o600); err != nil {
		return fmt.Errorf("write file: %w", err)
	}
	return nil
}

// Load replaces the store contents by loading from filepath.
func (s *SimpleVectorStore) Load(filepath string) error {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}
	var docs []StoredDocument
	if err := json.Unmarshal(data, &docs); err != nil {
		return fmt.Errorf("unmarshal documents: %w", err)
	}
	s.documents = docs
	return nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (s *SimpleVectorStore) applyFilter(filter map[string]interface{}) []StoredDocument {
	if len(filter) == 0 {
		return s.documents
	}
	out := make([]StoredDocument, 0, len(s.documents))
	for _, doc := range s.documents {
		match := true
		for k, v := range filter {
			if doc.Metadata[k] != v {
				match = false
				break
			}
		}
		if match {
			out = append(out, doc)
		}
	}
	return out
}

// vectorCosineSimilarity is the package-level cosine similarity used by the store.
// Separate from EmbeddingComparator.CosineSimilarity to avoid a receiver dependency.
func vectorCosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector length mismatch: %d vs %d", len(a), len(b))
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0, nil
	}
	return dot / denom, nil
}

func newUUID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%s-%s-%s-%s-%s",
		hex.EncodeToString(b[0:4]),
		hex.EncodeToString(b[4:6]),
		hex.EncodeToString(b[6:8]),
		hex.EncodeToString(b[8:10]),
		hex.EncodeToString(b[10:]),
	), nil
}
