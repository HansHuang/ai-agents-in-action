// knowledge_base_manager.go — Incremental knowledge base management.
//
// Handles document additions, updates, and deletions without rebuilding the
// entire index. Uses content hashing to detect changes and avoid redundant
// re-embedding.
//
// See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
package ragpipeline

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

var kbSupportedExtensions = map[string]bool{
	".txt":  true,
	".md":   true,
	".rst":  true,
	".text": true,
}

// ---------------------------------------------------------------------------
// DocumentRecord
// ---------------------------------------------------------------------------

// DocumentRecord holds metadata about a single document tracked by the KB.
type DocumentRecord struct {
	SourceID    string
	ChunkIDs    []string
	ChunkCount  int
	ContentHash string
	LastUpdated float64 // Unix timestamp
}

// ---------------------------------------------------------------------------
// KnowledgeBaseManager
// ---------------------------------------------------------------------------

// KnowledgeBaseManager manages a knowledge base that changes over time.
// Wraps a RAGPipeline with an index layer for incremental lifecycle operations.
type KnowledgeBaseManager struct {
	Pipeline      *RAGPipeline
	VectorStore   *SimpleVectorStore
	DocumentIndex map[string]*DocumentRecord
}

// NewKnowledgeBaseManager creates a KnowledgeBaseManager.
func NewKnowledgeBaseManager(pipeline *RAGPipeline, vectorStore *SimpleVectorStore) *KnowledgeBaseManager {
	return &KnowledgeBaseManager{
		Pipeline:      pipeline,
		VectorStore:   vectorStore,
		DocumentIndex: make(map[string]*DocumentRecord),
	}
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func kbHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return fmt.Sprintf("%x", sum)
}

func (m *KnowledgeBaseManager) chunkIDsForSource(sourceID string) []string {
	var ids []string
	for _, doc := range m.VectorStore.documents {
		if s, ok := doc.Metadata["source"].(string); ok && s == sourceID {
			ids = append(ids, doc.ID)
		}
	}
	return ids
}

func (m *KnowledgeBaseManager) ingestAndRecord(ctx context.Context, sourceID, text string) (int, error) {
	n, err := m.Pipeline.IngestText(ctx, text, map[string]interface{}{"source": sourceID})
	if err != nil {
		return 0, err
	}
	chunkIDs := m.chunkIDsForSource(sourceID)
	m.DocumentIndex[sourceID] = &DocumentRecord{
		SourceID:    sourceID,
		ChunkIDs:    chunkIDs,
		ChunkCount:  n,
		ContentHash: kbHash(text),
		LastUpdated: float64(time.Now().UnixNano()) / 1e9,
	}
	return n, nil
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// AddDocument adds a new document to the knowledge base.
// Returns an error if sourceID already exists (use UpdateDocument instead).
func (m *KnowledgeBaseManager) AddDocument(ctx context.Context, sourceID, text string) (int, error) {
	if _, exists := m.DocumentIndex[sourceID]; exists {
		return 0, fmt.Errorf("document %q already exists; use UpdateDocument() to replace it", sourceID)
	}
	return m.ingestAndRecord(ctx, sourceID, text)
}

// KBUpdateResult holds the result of an UpdateDocument call.
type KBUpdateResult struct {
	Removed   int
	Added     int
	NetChange int
	Unchanged bool
}

// UpdateDocument replaces an existing document with new content.
// Removes old chunks, ingests the new content, updates the index.
func (m *KnowledgeBaseManager) UpdateDocument(ctx context.Context, sourceID, newText string) (*KBUpdateResult, error) {
	record, exists := m.DocumentIndex[sourceID]
	if !exists {
		return nil, fmt.Errorf("document %q not found; use AddDocument() first", sourceID)
	}

	oldHash := record.ContentHash
	newHash := kbHash(newText)
	if oldHash == newHash {
		return &KBUpdateResult{Unchanged: true}, nil
	}

	removed := m.Pipeline.RemoveDocument(sourceID)
	delete(m.DocumentIndex, sourceID)

	added, err := m.ingestAndRecord(ctx, sourceID, newText)
	if err != nil {
		return nil, err
	}
	return &KBUpdateResult{
		Removed:   removed,
		Added:     added,
		NetChange: added - removed,
	}, nil
}

// RemoveDocument removes all chunks for a document and deregisters it.
func (m *KnowledgeBaseManager) RemoveDocument(sourceID string) (int, error) {
	if _, exists := m.DocumentIndex[sourceID]; !exists {
		return 0, fmt.Errorf("document %q not found in index", sourceID)
	}
	removed := m.Pipeline.RemoveDocument(sourceID)
	delete(m.DocumentIndex, sourceID)
	return removed, nil
}

// KBSyncResult holds the result of a SyncDirectory call.
type KBSyncResult struct {
	Added   []string
	Updated []string
	Removed []string
	Errors  []string
}

// SyncDirectory syncs the knowledge base with the current state of a directory.
func (m *KnowledgeBaseManager) SyncDirectory(ctx context.Context, directory string) (*KBSyncResult, error) {
	info, err := os.Stat(directory)
	if err != nil || !info.IsDir() {
		return nil, fmt.Errorf("not a directory: %q", directory)
	}

	result := &KBSyncResult{}

	// Collect current files.
	diskFiles := make(map[string]string) // sourceID → text
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if !kbSupportedExtensions[ext] {
			continue
		}
		data, err := os.ReadFile(filepath.Join(directory, entry.Name()))
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", entry.Name(), err))
			continue
		}
		diskFiles[entry.Name()] = string(data)
	}

	// Add / update.
	for sourceID, text := range diskFiles {
		if _, exists := m.DocumentIndex[sourceID]; !exists {
			if _, err := m.ingestAndRecord(ctx, sourceID, text); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", sourceID, err))
			} else {
				result.Added = append(result.Added, sourceID)
			}
		} else {
			r, err := m.UpdateDocument(ctx, sourceID, text)
			if err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", sourceID, err))
			} else if !r.Unchanged {
				result.Updated = append(result.Updated, sourceID)
			}
		}
	}

	// Remove documents no longer on disk.
	for sourceID := range m.DocumentIndex {
		if _, found := diskFiles[sourceID]; !found {
			if _, err := m.RemoveDocument(sourceID); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", sourceID, err))
			} else {
				result.Removed = append(result.Removed, sourceID)
			}
		}
	}
	return result, nil
}

// KBStats holds knowledge base statistics.
type KBStats struct {
	TotalDocuments int
	TotalChunks    int
	Documents      []KBDocStat
}

// KBDocStat holds stats for a single document.
type KBDocStat struct {
	ID          string
	Chunks      int
	LastUpdated float64
}

// GetStats returns knowledge base statistics.
func (m *KnowledgeBaseManager) GetStats() KBStats {
	docs := make([]KBDocStat, 0, len(m.DocumentIndex))
	for _, rec := range m.DocumentIndex {
		docs = append(docs, KBDocStat{
			ID:          rec.SourceID,
			Chunks:      rec.ChunkCount,
			LastUpdated: rec.LastUpdated,
		})
	}
	return KBStats{
		TotalDocuments: len(m.DocumentIndex),
		TotalChunks:    m.VectorStore.Count(),
		Documents:      docs,
	}
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunKnowledgeBaseManager demonstrates incremental knowledge base management.
func RunKnowledgeBaseManager() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("KNOWLEDGE BASE MANAGER DEMO")
	fmt.Println(strings.Repeat("=", 70))

	embedder := NewEmbeddingGenerator("text-embedding-3-small", 0)
	vectorStore := NewSimpleVectorStore()
	pipeline := NewRAGPipeline(vectorStore, embedder, "gpt-4o", 200, 30, 5, 0.4)
	manager := NewKnowledgeBaseManager(pipeline, vectorStore)

	ctx := context.Background()
	docs := map[string]string{
		"policy-v1.md": "Return policy: 30 days return window. No restocking fee.",
		"shipping.md":  "Standard shipping: 3-5 days at $4.99.",
		"faq.md":       "We accept Visa, Mastercard, and PayPal.",
	}
	fmt.Println("\n--- Step 1: Add 3 documents ---")
	for sourceID, text := range docs {
		n, err := manager.AddDocument(ctx, sourceID, text)
		if err != nil {
			fmt.Printf("  Error adding %s: %v\n", sourceID, err)
			continue
		}
		fmt.Printf("  Added %s: %d chunks\n", sourceID, n)
	}

	stats := manager.GetStats()
	fmt.Printf("\n--- Stats: %d documents, %d total chunks ---\n", stats.TotalDocuments, stats.TotalChunks)
}
