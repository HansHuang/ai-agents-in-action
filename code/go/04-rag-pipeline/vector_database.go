// vector_database.go — Vector database abstraction layer (Go port).
//
// Provides a unified VectorDatabase interface with multiple backends:
//   - SimpleVectorStore: in-memory, brute-force cosine similarity (no deps)
//   - ChromaVectorDB: Chroma REST API
//   - QdrantVectorDB: Qdrant REST API
//   - PineconeVectorDB: Pinecone REST API
//
// All backends implement the same VectorDatabase interface so callers can swap
// one for another without any changes to the calling code.
//
// Usage:
//
//	db := NewSimpleVectorStore()
//	db.Insert([]VectorDocument{{ID: "1", Text: "hello", Embedding: []float64{0.1, 0.2}}})
//	results, _ := db.Search([]float64{0.1, 0.2}, 5, nil)
//
// See: docs/05-the-tool-ecosystem/02-vector-databases.md
package ragpipeline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

// VecDocument is a document ready to be stored in a vector database.
type VecDocument struct {
	ID        string                 `json:"id"`
	Text      string                 `json:"text"`
	Embedding []float64              `json:"embedding"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// VecSearchResult is a single result returned from a vector search.
type VecSearchResult struct {
	ID       string                 `json:"id"`
	Text     string                 `json:"text"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata"`
}

// ---------------------------------------------------------------------------
// VectorDatabase interface
// ---------------------------------------------------------------------------

// VectorDatabase is the unified interface all backends must implement.
type VectorDatabase interface {
	// Insert upserts documents and returns the count inserted.
	Insert(documents []VecDocument) (int, error)

	// Search returns the k nearest documents to queryEmbedding.
	Search(queryEmbedding []float64, k int, filterMetadata map[string]interface{}) ([]VecSearchResult, error)

	// Delete removes documents by ID and returns the count deleted.
	Delete(ids []string) (int, error)

	// Count returns the total number of documents stored.
	Count() (int, error)

	// Clear removes all documents.
	Clear() error

	// BatchInsert inserts documents in batches of batchSize.
	BatchInsert(documents []VecDocument, batchSize int) (int, error)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func cosineSim(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func defaultBatchInsert(db VectorDatabase, documents []VecDocument, batchSize int) (int, error) {
	total := 0
	for i := 0; i < len(documents); i += batchSize {
		end := i + batchSize
		if end > len(documents) {
			end = len(documents)
		}
		n, err := db.Insert(documents[i:end])
		if err != nil {
			return total, err
		}
		total += n
	}
	return total, nil
}

// ---------------------------------------------------------------------------
// Backend: SimpleVectorStore
// ---------------------------------------------------------------------------

type simpleDoc struct {
	id        string
	text      string
	embedding []float64
	metadata  map[string]interface{}
}

// SimpleVectorStore is an in-memory vector store — no external dependencies.
//
// Uses brute-force O(n) cosine similarity.  Suitable for tests, prototyping,
// and datasets up to ~10,000 documents.
type SimpleVectorStore struct {
	mu   sync.RWMutex
	docs []simpleDoc
}

// NewSimpleVectorStore creates an empty SimpleVectorStore.
func NewSimpleVectorStore() *SimpleVectorStore {
	return &SimpleVectorStore{}
}

func (s *SimpleVectorStore) Insert(documents []VecDocument) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, doc := range documents {
		// Upsert: remove existing document with same ID
		filtered := s.docs[:0]
		for _, d := range s.docs {
			if d.id != doc.ID {
				filtered = append(filtered, d)
			}
		}
		s.docs = append(filtered, simpleDoc{
			id:        doc.ID,
			text:      doc.Text,
			embedding: doc.Embedding,
			metadata:  copyMeta(doc.Metadata),
		})
	}
	return len(documents), nil
}

func (s *SimpleVectorStore) Search(queryEmbedding []float64, k int, filterMetadata map[string]interface{}) ([]VecSearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var candidates []simpleDoc
	for _, d := range s.docs {
		if matchesFilter(d.metadata, filterMetadata) {
			candidates = append(candidates, d)
		}
	}

	results := make([]VecSearchResult, 0, len(candidates))
	for _, d := range candidates {
		results = append(results, VecSearchResult{
			ID:       d.id,
			Text:     d.text,
			Score:    cosineSim(queryEmbedding, d.embedding),
			Metadata: d.metadata,
		})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if k < len(results) {
		results = results[:k]
	}
	return results, nil
}

func (s *SimpleVectorStore) Delete(ids []string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	idSet := make(map[string]bool, len(ids))
	for _, id := range ids {
		idSet[id] = true
	}
	before := len(s.docs)
	filtered := s.docs[:0]
	for _, d := range s.docs {
		if !idSet[d.id] {
			filtered = append(filtered, d)
		}
	}
	s.docs = filtered
	return before - len(s.docs), nil
}

func (s *SimpleVectorStore) Count() (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.docs), nil
}

func (s *SimpleVectorStore) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docs = s.docs[:0]
	return nil
}

func (s *SimpleVectorStore) BatchInsert(documents []VecDocument, batchSize int) (int, error) {
	return defaultBatchInsert(s, documents, batchSize)
}

// ---------------------------------------------------------------------------
// Backend: ChromaVectorDB
// ---------------------------------------------------------------------------

// ChromaVectorDB is a vector database backend for Chroma's HTTP API.
//
// Requires a running Chroma server (default: http://localhost:8000).
// Chroma returns cosine distances in [0,2]; we convert to similarity as
// 1 - distance/2.
type ChromaVectorDB struct {
	baseURL        string
	collectionName string
	collectionID   string // resolved on first use
	httpClient     *http.Client
}

// NewChromaVectorDB creates a ChromaVectorDB that connects to a Chroma server.
//
//	host: e.g. "localhost"
//	port: e.g. 8000
//	collectionName: name of the Chroma collection
func NewChromaVectorDB(host string, port int, collectionName string) (*ChromaVectorDB, error) {
	db := &ChromaVectorDB{
		baseURL:        fmt.Sprintf("http://%s:%d", host, port),
		collectionName: collectionName,
		httpClient:     &http.Client{Timeout: 30 * time.Second},
	}
	if err := db.ensureCollection(); err != nil {
		return nil, fmt.Errorf("chroma: ensure collection: %w", err)
	}
	return db, nil
}

func (c *ChromaVectorDB) doJSON(method, path string, body interface{}) ([]byte, int, error) {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		reqBody = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, c.baseURL+path, reqBody)
	if err != nil {
		return nil, 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	return data, resp.StatusCode, err
}

func (c *ChromaVectorDB) ensureCollection() error {
	// Try to get existing collection
	data, status, err := c.doJSON("GET", "/api/v1/collections/"+c.collectionName, nil)
	if err != nil {
		return err
	}
	if status == 200 {
		var col struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(data, &col); err != nil {
			return err
		}
		c.collectionID = col.ID
		return nil
	}
	// Create collection
	payload := map[string]interface{}{
		"name":     c.collectionName,
		"metadata": map[string]string{"hnsw:space": "cosine"},
	}
	data, _, err = c.doJSON("POST", "/api/v1/collections", payload)
	if err != nil {
		return err
	}
	var col struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(data, &col); err != nil {
		return err
	}
	c.collectionID = col.ID
	return nil
}

func (c *ChromaVectorDB) Insert(documents []VecDocument) (int, error) {
	if len(documents) == 0 {
		return 0, nil
	}
	ids := make([]string, len(documents))
	embeddings := make([][]float64, len(documents))
	texts := make([]string, len(documents))
	metas := make([]map[string]interface{}, len(documents))
	for i, d := range documents {
		ids[i] = d.ID
		embeddings[i] = d.Embedding
		texts[i] = d.Text
		metas[i] = d.Metadata
		if metas[i] == nil {
			metas[i] = map[string]interface{}{}
		}
	}
	payload := map[string]interface{}{
		"ids":        ids,
		"embeddings": embeddings,
		"documents":  texts,
		"metadatas":  metas,
	}
	_, _, err := c.doJSON("POST", "/api/v1/collections/"+c.collectionID+"/upsert", payload)
	if err != nil {
		return 0, err
	}
	return len(documents), nil
}

func (c *ChromaVectorDB) Search(queryEmbedding []float64, k int, filterMetadata map[string]interface{}) ([]VecSearchResult, error) {
	payload := map[string]interface{}{
		"query_embeddings": [][]float64{queryEmbedding},
		"n_results":        k,
		"include":          []string{"documents", "distances", "metadatas"},
	}
	if filterMetadata != nil {
		payload["where"] = filterMetadata
	}
	data, _, err := c.doJSON("POST", "/api/v1/collections/"+c.collectionID+"/query", payload)
	if err != nil {
		return nil, err
	}
	var resp struct {
		IDs       [][]string                 `json:"ids"`
		Documents [][]string                 `json:"documents"`
		Distances [][]float64                `json:"distances"`
		Metadatas [][]map[string]interface{} `json:"metadatas"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	if len(resp.IDs) == 0 {
		return nil, nil
	}
	results := make([]VecSearchResult, len(resp.IDs[0]))
	for i := range resp.IDs[0] {
		results[i] = VecSearchResult{
			ID:       resp.IDs[0][i],
			Text:     resp.Documents[0][i],
			Score:    1.0 - resp.Distances[0][i]/2.0,
			Metadata: resp.Metadatas[0][i],
		}
	}
	return results, nil
}

func (c *ChromaVectorDB) Delete(ids []string) (int, error) {
	payload := map[string]interface{}{"ids": ids}
	_, _, err := c.doJSON("POST", "/api/v1/collections/"+c.collectionID+"/delete", payload)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

func (c *ChromaVectorDB) Count() (int, error) {
	data, _, err := c.doJSON("GET", "/api/v1/collections/"+c.collectionID+"/count", nil)
	if err != nil {
		return 0, err
	}
	var n int
	if err := json.Unmarshal(data, &n); err != nil {
		return 0, err
	}
	return n, nil
}

func (c *ChromaVectorDB) Clear() error {
	_, _, err := c.doJSON("DELETE", "/api/v1/collections/"+c.collectionName, nil)
	if err != nil {
		return err
	}
	return c.ensureCollection()
}

func (c *ChromaVectorDB) BatchInsert(documents []VecDocument, batchSize int) (int, error) {
	return defaultBatchInsert(c, documents, batchSize)
}

// ---------------------------------------------------------------------------
// Backend: QdrantVectorDB
// ---------------------------------------------------------------------------

// QdrantVectorDB is a vector database backend for Qdrant's REST API.
//
// Requires a running Qdrant server (default: http://localhost:6333).
type QdrantVectorDB struct {
	baseURL        string
	collectionName string
	dimension      int
	httpClient     *http.Client
}

// NewQdrantVectorDB creates a QdrantVectorDB.
//
//	host:           e.g. "localhost"
//	port:           e.g. 6333
//	collectionName: name of the Qdrant collection
//	dimension:      embedding vector dimension (must match your model)
func NewQdrantVectorDB(host string, port int, collectionName string, dimension int) (*QdrantVectorDB, error) {
	db := &QdrantVectorDB{
		baseURL:        fmt.Sprintf("http://%s:%d", host, port),
		collectionName: collectionName,
		dimension:      dimension,
		httpClient:     &http.Client{Timeout: 30 * time.Second},
	}
	if err := db.ensureCollection(); err != nil {
		return nil, fmt.Errorf("qdrant: ensure collection: %w", err)
	}
	return db, nil
}

func (q *QdrantVectorDB) doJSON(method, path string, body interface{}) ([]byte, int, error) {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		reqBody = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, q.baseURL+path, reqBody)
	if err != nil {
		return nil, 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := q.httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	return data, resp.StatusCode, err
}

func (q *QdrantVectorDB) ensureCollection() error {
	_, status, err := q.doJSON("GET", "/collections/"+q.collectionName, nil)
	if err != nil {
		return err
	}
	if status == 200 {
		return nil
	}
	payload := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     q.dimension,
			"distance": "Cosine",
		},
	}
	_, _, err = q.doJSON("PUT", "/collections/"+q.collectionName, payload)
	return err
}

func (q *QdrantVectorDB) Insert(documents []VecDocument) (int, error) {
	if len(documents) == 0 {
		return 0, nil
	}
	type point struct {
		ID      string                 `json:"id"`
		Vector  []float64              `json:"vector"`
		Payload map[string]interface{} `json:"payload"`
	}
	points := make([]point, len(documents))
	for i, d := range documents {
		payload := map[string]interface{}{"text": d.Text}
		for k, v := range d.Metadata {
			payload[k] = v
		}
		points[i] = point{ID: d.ID, Vector: d.Embedding, Payload: payload}
	}
	body := map[string]interface{}{"points": points}
	_, _, err := q.doJSON("PUT", "/collections/"+q.collectionName+"/points?wait=true", body)
	if err != nil {
		return 0, err
	}
	return len(documents), nil
}

func (q *QdrantVectorDB) Search(queryEmbedding []float64, k int, filterMetadata map[string]interface{}) ([]VecSearchResult, error) {
	payload := map[string]interface{}{
		"vector":       queryEmbedding,
		"limit":        k,
		"with_payload": true,
	}
	if filterMetadata != nil {
		conditions := make([]map[string]interface{}, 0, len(filterMetadata))
		for key, value := range filterMetadata {
			conditions = append(conditions, map[string]interface{}{
				"key":   key,
				"match": map[string]interface{}{"value": value},
			})
		}
		payload["filter"] = map[string]interface{}{"must": conditions}
	}
	data, _, err := q.doJSON("POST", "/collections/"+q.collectionName+"/points/search", payload)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Result []struct {
			ID      string                 `json:"id"`
			Score   float64                `json:"score"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	results := make([]VecSearchResult, len(resp.Result))
	for i, r := range resp.Result {
		text, _ := r.Payload["text"].(string)
		meta := make(map[string]interface{})
		for k, v := range r.Payload {
			if k != "text" {
				meta[k] = v
			}
		}
		results[i] = VecSearchResult{ID: r.ID, Text: text, Score: r.Score, Metadata: meta}
	}
	return results, nil
}

func (q *QdrantVectorDB) Delete(ids []string) (int, error) {
	body := map[string]interface{}{"points": ids}
	_, _, err := q.doJSON("POST", "/collections/"+q.collectionName+"/points/delete?wait=true", body)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

func (q *QdrantVectorDB) Count() (int, error) {
	data, _, err := q.doJSON("POST", "/collections/"+q.collectionName+"/points/count", map[string]bool{"exact": true})
	if err != nil {
		return 0, err
	}
	var resp struct {
		Result struct {
			Count int `json:"count"`
		} `json:"result"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return 0, err
	}
	return resp.Result.Count, nil
}

func (q *QdrantVectorDB) Clear() error {
	q.doJSON("DELETE", "/collections/"+q.collectionName, nil) //nolint:errcheck
	return q.ensureCollection()
}

func (q *QdrantVectorDB) BatchInsert(documents []VecDocument, batchSize int) (int, error) {
	return defaultBatchInsert(q, documents, batchSize)
}

// ---------------------------------------------------------------------------
// Backend: PineconeVectorDB
// ---------------------------------------------------------------------------

// PineconeVectorDB is a vector database backend for the Pinecone REST API.
//
// Requires a valid Pinecone API key and an existing or auto-created index.
type PineconeVectorDB struct {
	apiKey     string
	indexName  string
	indexHost  string // resolved after index creation/lookup
	dimension  int
	httpClient *http.Client
}

// NewPineconeVectorDB creates a PineconeVectorDB.
//
//	apiKey:    Pinecone API key
//	indexName: name of the Pinecone index (created if absent)
//	dimension: embedding vector dimension
func NewPineconeVectorDB(apiKey, indexName string, dimension int) (*PineconeVectorDB, error) {
	db := &PineconeVectorDB{
		apiKey:     apiKey,
		indexName:  indexName,
		dimension:  dimension,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
	if err := db.ensureIndex(); err != nil {
		return nil, fmt.Errorf("pinecone: ensure index: %w", err)
	}
	return db, nil
}

func (p *PineconeVectorDB) controlReq(method, path string, body interface{}) ([]byte, int, error) {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		reqBody = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, "https://api.pinecone.io"+path, reqBody)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Api-Key", p.apiKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	return data, resp.StatusCode, err
}

func (p *PineconeVectorDB) dataReq(method, path string, body interface{}) ([]byte, int, error) {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		reqBody = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, "https://"+p.indexHost+path, reqBody)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Api-Key", p.apiKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	return data, resp.StatusCode, err
}

func (p *PineconeVectorDB) ensureIndex() error {
	data, status, err := p.controlReq("GET", "/indexes/"+p.indexName, nil)
	if err != nil {
		return err
	}
	if status == 200 {
		var idx struct {
			Host string `json:"host"`
		}
		if err := json.Unmarshal(data, &idx); err != nil {
			return err
		}
		p.indexHost = idx.Host
		return nil
	}
	// Create serverless index
	payload := map[string]interface{}{
		"name":      p.indexName,
		"dimension": p.dimension,
		"metric":    "cosine",
		"spec": map[string]interface{}{
			"serverless": map[string]string{
				"cloud":  "aws",
				"region": "us-east-1",
			},
		},
	}
	createData, _, err := p.controlReq("POST", "/indexes", payload)
	if err != nil {
		return err
	}
	var idx struct {
		Host string `json:"host"`
	}
	if err := json.Unmarshal(createData, &idx); err != nil {
		return err
	}
	p.indexHost = idx.Host
	return nil
}

func (p *PineconeVectorDB) Insert(documents []VecDocument) (int, error) {
	if len(documents) == 0 {
		return 0, nil
	}
	type vector struct {
		ID       string                 `json:"id"`
		Values   []float64              `json:"values"`
		Metadata map[string]interface{} `json:"metadata"`
	}
	vectors := make([]vector, len(documents))
	for i, d := range documents {
		meta := map[string]interface{}{"text": d.Text}
		for k, v := range d.Metadata {
			meta[k] = v
		}
		vectors[i] = vector{ID: d.ID, Values: d.Embedding, Metadata: meta}
	}
	_, _, err := p.dataReq("POST", "/vectors/upsert", map[string]interface{}{"vectors": vectors})
	if err != nil {
		return 0, err
	}
	return len(documents), nil
}

func (p *PineconeVectorDB) Search(queryEmbedding []float64, k int, filterMetadata map[string]interface{}) ([]VecSearchResult, error) {
	payload := map[string]interface{}{
		"vector":          queryEmbedding,
		"topK":            k,
		"includeMetadata": true,
	}
	if filterMetadata != nil {
		payload["filter"] = filterMetadata
	}
	data, _, err := p.dataReq("POST", "/query", payload)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Matches []struct {
			ID       string                 `json:"id"`
			Score    float64                `json:"score"`
			Metadata map[string]interface{} `json:"metadata"`
		} `json:"matches"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	results := make([]VecSearchResult, len(resp.Matches))
	for i, m := range resp.Matches {
		text, _ := m.Metadata["text"].(string)
		meta := make(map[string]interface{})
		for k, v := range m.Metadata {
			if k != "text" {
				meta[k] = v
			}
		}
		results[i] = VecSearchResult{ID: m.ID, Text: text, Score: m.Score, Metadata: meta}
	}
	return results, nil
}

func (p *PineconeVectorDB) Delete(ids []string) (int, error) {
	_, _, err := p.dataReq("POST", "/vectors/delete", map[string]interface{}{"ids": ids})
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

func (p *PineconeVectorDB) Count() (int, error) {
	data, _, err := p.dataReq("GET", "/describe_index_stats", nil)
	if err != nil {
		return 0, err
	}
	var resp struct {
		TotalVectorCount int `json:"totalVectorCount"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return 0, err
	}
	return resp.TotalVectorCount, nil
}

func (p *PineconeVectorDB) Clear() error {
	_, _, err := p.dataReq("POST", "/vectors/delete", map[string]interface{}{"deleteAll": true})
	return err
}

func (p *PineconeVectorDB) BatchInsert(documents []VecDocument, batchSize int) (int, error) {
	return defaultBatchInsert(p, documents, batchSize)
}

// ---------------------------------------------------------------------------
// VectorDBFactory
// ---------------------------------------------------------------------------

// VectorDBConfig holds configuration for creating a VectorDatabase.
type VectorDBConfig struct {
	Type string // "simple", "chroma", "qdrant", "pinecone"

	// Chroma / Qdrant
	Host           string
	Port           int
	CollectionName string
	Dimension      int

	// Pinecone
	APIKey    string
	IndexName string
}

// VectorDBFactory creates VectorDatabase instances from configuration.
type VectorDBFactory struct{}

// Create instantiates a VectorDatabase from a VectorDBConfig.
func (VectorDBFactory) Create(cfg VectorDBConfig) (VectorDatabase, error) {
	switch strings.ToLower(cfg.Type) {
	case "simple":
		return NewSimpleVectorStore(), nil
	case "chroma":
		host := cfg.Host
		if host == "" {
			host = "localhost"
		}
		port := cfg.Port
		if port == 0 {
			port = 8000
		}
		name := cfg.CollectionName
		if name == "" {
			name = "documents"
		}
		return NewChromaVectorDB(host, port, name)
	case "qdrant":
		host := cfg.Host
		if host == "" {
			host = "localhost"
		}
		port := cfg.Port
		if port == 0 {
			port = 6333
		}
		name := cfg.CollectionName
		if name == "" {
			name = "documents"
		}
		dim := cfg.Dimension
		if dim == 0 {
			dim = 1536
		}
		return NewQdrantVectorDB(host, port, name, dim)
	case "pinecone":
		name := cfg.IndexName
		if name == "" {
			name = "documents"
		}
		dim := cfg.Dimension
		if dim == 0 {
			dim = 1536
		}
		return NewPineconeVectorDB(cfg.APIKey, name, dim)
	default:
		return nil, fmt.Errorf("unknown database type: %q (available: simple, chroma, qdrant, pinecone)", cfg.Type)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func copyMeta(m map[string]interface{}) map[string]interface{} {
	if m == nil {
		return map[string]interface{}{}
	}
	out := make(map[string]interface{}, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func matchesFilter(meta, filter map[string]interface{}) bool {
	if filter == nil {
		return true
	}
	for k, v := range filter {
		if meta[k] != v {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Demo (called from main.go or tests)
// ---------------------------------------------------------------------------

// RunVectorDatabaseDemo demonstrates the SimpleVectorStore backend.
func RunVectorDatabaseDemo() {
	const dim = 64
	const nDocs = 100

	rng := rand.New(rand.NewSource(42))
	randEmbed := func() []float64 {
		v := make([]float64, dim)
		var norm float64
		for i := range v {
			v[i] = rng.NormFloat64()
			norm += v[i] * v[i]
		}
		norm = math.Sqrt(norm)
		for i := range v {
			v[i] /= norm
		}
		return v
	}

	embeddings := make([][]float64, nDocs)
	for i := range embeddings {
		embeddings[i] = randEmbed()
	}

	docs := make([]VecDocument, nDocs)
	for i := range docs {
		docs[i] = VecDocument{
			ID:        fmt.Sprintf("%d", i),
			Text:      fmt.Sprintf("Document %d about topic %d", i, i%5),
			Embedding: embeddings[i],
			Metadata:  map[string]interface{}{"category": fmt.Sprintf("cat_%d", i%3)},
		}
	}

	db := NewSimpleVectorStore()

	t0 := time.Now()
	inserted, _ := db.BatchInsert(docs, 50)
	insertMs := float64(time.Since(t0).Microseconds()) / 1000.0

	t1 := time.Now()
	results, _ := db.Search(embeddings[0], 5, nil)
	searchMs := float64(time.Since(t1).Microseconds()) / 1000.0

	count, _ := db.Count()
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("Vector Database Abstraction Demo (Go)")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("\n[SimpleVectorStore]\n")
	fmt.Printf("  Inserted %d docs in %.1f ms\n", inserted, insertMs)
	fmt.Printf("  Top-5 search in %.2f ms\n", searchMs)
	fmt.Printf("  Count: %d\n", count)
	if len(results) > 0 {
		fmt.Printf("  Top result: id=%q, score=%.4f\n", results[0].ID, results[0].Score)
	}

	filtered, _ := db.Search(embeddings[0], 5, map[string]interface{}{"category": "cat_0"})
	ids := make([]string, len(filtered))
	for i, r := range filtered {
		ids[i] = r.ID
	}
	fmt.Printf("  Filtered (category=cat_0): [%s]\n", strings.Join(ids, ", "))
}
