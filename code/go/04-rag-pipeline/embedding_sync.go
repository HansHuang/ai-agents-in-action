// embedding_sync.go — Keep a vector index in sync with source documents.
//
// Provides full and incremental sync strategies plus a health verification check.
// Uses the VectorDatabase interface so it works with any backend.
//
// See: docs/05-the-tool-ecosystem/02-vector-databases.md
package ragpipeline

import (
	"crypto/sha256"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// SourceDocument
// ---------------------------------------------------------------------------

// SourceDocument is a document in the source document store.
type SourceDocument struct {
	ID        string
	Text      string
	Metadata  map[string]interface{}
	UpdatedAt float64 // Unix timestamp
}

// ContentHash returns the SHA-256 hex digest of the document text.
func (d *SourceDocument) ContentHash() string {
	sum := sha256.Sum256([]byte(d.Text))
	return fmt.Sprintf("%x", sum)
}

// ---------------------------------------------------------------------------
// SyncReport
// ---------------------------------------------------------------------------

// SyncReport is the result of a full or incremental sync operation.
type SyncReport struct {
	Strategy        string
	Created         int
	Updated         int
	Deleted         int
	Errors          []string
	DurationSeconds float64
	Timestamp       time.Time
}

func (r *SyncReport) String() string {
	status := "OK"
	if len(r.Errors) > 0 {
		status = fmt.Sprintf("%d errors", len(r.Errors))
	}
	return fmt.Sprintf("SyncReport[%s] created=%d updated=%d deleted=%d duration=%.2fs status=%s at=%s",
		r.Strategy, r.Created, r.Updated, r.Deleted,
		r.DurationSeconds, status, r.Timestamp.Format(time.RFC3339))
}

// ---------------------------------------------------------------------------
// SyncHealth
// ---------------------------------------------------------------------------

// SyncHealth is the result of a sync health verification.
type SyncHealth struct {
	IsHealthy       bool
	TotalDocuments  int
	TotalVectors    int
	MismatchCount   int
	OrphanCount     int
	Recommendations []string
}

func (h *SyncHealth) String() string {
	status := "HEALTHY"
	if !h.IsHealthy {
		status = "DEGRADED"
	}
	return fmt.Sprintf("SyncHealth[%s] docs=%d vectors=%d missing=%d orphans=%d recommendations=%v",
		status, h.TotalDocuments, h.TotalVectors, h.MismatchCount, h.OrphanCount, h.Recommendations)
}

// ---------------------------------------------------------------------------
// InMemoryDocumentStore
// ---------------------------------------------------------------------------

// InMemoryDocumentStore is a lightweight document store for demos and tests.
type InMemoryDocumentStore struct {
	mu   sync.RWMutex
	docs map[string]SourceDocument
}

// NewInMemoryDocumentStore creates an empty InMemoryDocumentStore.
func NewInMemoryDocumentStore() *InMemoryDocumentStore {
	return &InMemoryDocumentStore{docs: make(map[string]SourceDocument)}
}

// Add adds or replaces a document.
func (s *InMemoryDocumentStore) Add(doc SourceDocument) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if doc.UpdatedAt == 0 {
		doc.UpdatedAt = float64(time.Now().UnixNano()) / 1e9
	}
	s.docs[doc.ID] = doc
}

// AddMany adds multiple documents.
func (s *InMemoryDocumentStore) AddMany(docs []SourceDocument) {
	for _, d := range docs {
		s.Add(d)
	}
}

// Update updates the text (and optionally metadata) of an existing document.
func (s *InMemoryDocumentStore) Update(docID, newText string, metadata map[string]interface{}) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	existing, ok := s.docs[docID]
	if !ok {
		return fmt.Errorf("document %q not found", docID)
	}
	if metadata == nil {
		metadata = existing.Metadata
	}
	s.docs[docID] = SourceDocument{
		ID:        docID,
		Text:      newText,
		Metadata:  metadata,
		UpdatedAt: float64(time.Now().UnixNano()) / 1e9,
	}
	return nil
}

// Remove removes a document by ID.
func (s *InMemoryDocumentStore) Remove(docID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.docs, docID)
}

// GetAll returns all documents.
func (s *InMemoryDocumentStore) GetAll() []SourceDocument {
	s.mu.RLock()
	defer s.mu.RUnlock()
	docs := make([]SourceDocument, 0, len(s.docs))
	for _, d := range s.docs {
		docs = append(docs, d)
	}
	return docs
}

// GetIDs returns the set of all current document IDs.
func (s *InMemoryDocumentStore) GetIDs() map[string]bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := make(map[string]bool, len(s.docs))
	for id := range s.docs {
		ids[id] = true
	}
	return ids
}

// GetChanges returns documents created or updated after since (epoch seconds).
// since=0 returns all documents.
type StoreChanges struct {
	CreatedOrUpdated []SourceDocument
	AllIDs           map[string]bool
}

func (s *InMemoryDocumentStore) GetChanges(since float64) StoreChanges {
	s.mu.RLock()
	defer s.mu.RUnlock()
	allIDs := make(map[string]bool, len(s.docs))
	var changed []SourceDocument
	for id, doc := range s.docs {
		allIDs[id] = true
		if since == 0 || doc.UpdatedAt > since {
			changed = append(changed, doc)
		}
	}
	return StoreChanges{CreatedOrUpdated: changed, AllIDs: allIDs}
}

// Count returns the number of documents.
func (s *InMemoryDocumentStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.docs)
}

// ---------------------------------------------------------------------------
// EmbeddingSyncManager
// ---------------------------------------------------------------------------

// DocEmbedder is any function that converts text to an embedding vector.
type DocEmbedder func(text string) ([]float64, error)

// EmbeddingSyncManager synchronises a VectorDatabase with a document store.
type EmbeddingSyncManager struct {
	VectorDB   VectorDatabase
	Embedder   DocEmbedder
	Documents  *InMemoryDocumentStore
	lastSyncTS float64

	stopCh   chan struct{}
	stopOnce sync.Once
}

// NewEmbeddingSyncManager creates an EmbeddingSyncManager.
func NewEmbeddingSyncManager(db VectorDatabase, embedder DocEmbedder, store *InMemoryDocumentStore) *EmbeddingSyncManager {
	return &EmbeddingSyncManager{
		VectorDB:  db,
		Embedder:  embedder,
		Documents: store,
	}
}

// FullSync re-embeds all documents and rebuilds the entire vector index.
func (m *EmbeddingSyncManager) FullSync() (*SyncReport, error) {
	t0 := time.Now()
	var errors []string
	created := 0

	if err := m.VectorDB.Clear(); err != nil {
		return nil, fmt.Errorf("clear: %w", err)
	}

	for _, doc := range m.Documents.GetAll() {
		vdoc, err := m.toVecDocument(doc)
		if err != nil {
			errors = append(errors, fmt.Sprintf("[%s] %v", doc.ID, err))
			continue
		}
		if _, err := m.VectorDB.Insert([]VecDocument{vdoc}); err != nil {
			errors = append(errors, fmt.Sprintf("[%s] %v", doc.ID, err))
			continue
		}
		created++
	}

	m.lastSyncTS = float64(time.Now().UnixNano()) / 1e9
	return &SyncReport{
		Strategy:        "full",
		Created:         created,
		Errors:          errors,
		DurationSeconds: time.Since(t0).Seconds(),
		Timestamp:       time.Now().UTC(),
	}, nil
}

// IncrementalSync only syncs documents that have changed since the last sync.
func (m *EmbeddingSyncManager) IncrementalSync(since float64) (*SyncReport, error) {
	if since == 0 {
		since = m.lastSyncTS
	}
	t0 := time.Now()
	var errors []string
	created, updated, deleted := 0, 0, 0

	changes := m.Documents.GetChanges(since)
	currentIDs := changes.AllIDs

	// Detect deletions.
	vectorIDs := m.getVectorIDs()
	for id := range vectorIDs {
		if !currentIDs[id] {
			if _, err := m.VectorDB.Delete([]string{id}); err != nil {
				errors = append(errors, fmt.Sprintf("delete %s: %v", id, err))
			} else {
				deleted++
			}
		}
	}

	// Handle additions and updates.
	for _, doc := range changes.CreatedOrUpdated {
		vdoc, err := m.toVecDocument(doc)
		if err != nil {
			errors = append(errors, fmt.Sprintf("[%s] %v", doc.ID, err))
			continue
		}
		if vectorIDs[doc.ID] {
			m.VectorDB.Delete([]string{doc.ID})
			m.VectorDB.Insert([]VecDocument{vdoc})
			updated++
		} else {
			m.VectorDB.Insert([]VecDocument{vdoc})
			created++
		}
	}

	m.lastSyncTS = float64(time.Now().UnixNano()) / 1e9
	return &SyncReport{
		Strategy:        "incremental",
		Created:         created,
		Updated:         updated,
		Deleted:         deleted,
		Errors:          errors,
		DurationSeconds: time.Since(t0).Seconds(),
		Timestamp:       time.Now().UTC(),
	}, nil
}

// VerifySync checks whether the vector index is consistent with the document store.
func (m *EmbeddingSyncManager) VerifySync(sampleSize int) (*SyncHealth, error) {
	allDocs := m.Documents.GetAll()
	totalDocs := len(allDocs)
	totalVectors, err := m.VectorDB.Count()
	if err != nil {
		return nil, err
	}

	if sampleSize > totalDocs {
		sampleSize = totalDocs
	}
	sample := make([]SourceDocument, len(allDocs))
	copy(sample, allDocs)
	rand.Shuffle(len(sample), func(i, j int) { sample[i], sample[j] = sample[j], sample[i] })
	sample = sample[:sampleSize]

	vectorIDs := m.getVectorIDs()
	mismatchCount := 0
	for _, doc := range sample {
		if !vectorIDs[doc.ID] {
			mismatchCount++
		}
	}

	storeIDs := m.Documents.GetIDs()
	orphanCount := 0
	for id := range vectorIDs {
		if !storeIDs[id] {
			orphanCount++
		}
	}

	var recommendations []string
	if mismatchCount > 0 {
		recommendations = append(recommendations, fmt.Sprintf(
			"%d sampled documents missing from vector DB — run IncrementalSync() or FullSync().", mismatchCount))
	}
	if orphanCount > 0 {
		recommendations = append(recommendations, fmt.Sprintf(
			"%d orphaned vectors found — run IncrementalSync() to clean up.", orphanCount))
	}
	if totalDocs != totalVectors {
		recommendations = append(recommendations, fmt.Sprintf(
			"Document count (%d) ≠ vector count (%d) — index may be stale.", totalDocs, totalVectors))
	}

	return &SyncHealth{
		IsHealthy:       mismatchCount == 0 && orphanCount == 0 && totalDocs == totalVectors,
		TotalDocuments:  totalDocs,
		TotalVectors:    totalVectors,
		MismatchCount:   mismatchCount,
		OrphanCount:     orphanCount,
		Recommendations: recommendations,
	}, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (m *EmbeddingSyncManager) toVecDocument(doc SourceDocument) (VecDocument, error) {
	emb, err := m.Embedder(doc.Text)
	if err != nil {
		return VecDocument{}, err
	}
	meta := make(map[string]interface{}, len(doc.Metadata)+2)
	for k, v := range doc.Metadata {
		meta[k] = v
	}
	meta["_content_hash"] = doc.ContentHash()
	meta["_updated_at"] = doc.UpdatedAt
	return VecDocument{
		ID:        doc.ID,
		Text:      doc.Text,
		Embedding: emb,
		Metadata:  meta,
	}, nil
}

func (m *EmbeddingSyncManager) getVectorIDs() map[string]bool {
	// InMemoryVecStore (implements VectorDatabase) stores docs internally;
	// we access it via a type assertion.
	if store, ok := m.VectorDB.(*InMemoryVecStore); ok {
		store.mu.RLock()
		defer store.mu.RUnlock()
		ids := make(map[string]bool, len(store.docs))
		for _, d := range store.docs {
			ids[d.id] = true
		}
		return ids
	}
	return map[string]bool{}
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunEmbeddingSync demonstrates full and incremental sync operations.
func RunEmbeddingSync() {
	fmt.Println("EMBEDDING SYNC DEMO")
	fmt.Println("Set OPENAI_API_KEY to run a live embedding demo.")
	fmt.Println("Using deterministic fake embedder for illustration.")

	fakeEmbedder := func(text string) ([]float64, error) {
		v := make([]float64, 8)
		for i, c := range text {
			v[i%8] += float64(c)
		}
		// Normalize
		norm := 0.0
		for _, x := range v {
			norm += x * x
		}
		if norm > 0 {
			norm = 1.0 / (norm * norm)
			for i := range v {
				v[i] *= norm
			}
		}
		return v, nil
	}

	store := NewInMemoryDocumentStore()
	for i := 0; i < 10; i++ {
		store.Add(SourceDocument{
			ID:   fmt.Sprintf("doc_%03d", i),
			Text: fmt.Sprintf("Document %d: information about topic %d.", i, i%3),
		})
	}

	db := NewInMemoryVecStore()
	manager := NewEmbeddingSyncManager(db, fakeEmbedder, store)

	report, _ := manager.FullSync()
	fmt.Printf("Full sync: %s\n", report)

	store.Add(SourceDocument{ID: "doc_new", Text: "A brand new document."})
	store.Remove("doc_000")

	report2, _ := manager.IncrementalSync(0)
	fmt.Printf("Incremental sync: %s\n", report2)

	health, _ := manager.VerifySync(100)
	fmt.Printf("Health: %s\n", health)
}
