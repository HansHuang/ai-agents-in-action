// embedding_visualizer.go — Embedding visualization utilities.
//
// Provides text similarity matrices, outlier detection, and K-means clustering
// over embedding vectors — all without requiring external plotting libraries.
//
// See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
package ragpipeline

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// EmbeddingVisualizer
// ---------------------------------------------------------------------------

// EmbeddingVisualizer generates text representations of embedding relationships.
type EmbeddingVisualizer struct {
	Generator  *EmbeddingGenerator
	Comparator *EmbeddingComparator
}

// NewEmbeddingVisualizer creates an EmbeddingVisualizer.
func NewEmbeddingVisualizer(generator *EmbeddingGenerator) *EmbeddingVisualizer {
	return &EmbeddingVisualizer{
		Generator:  generator,
		Comparator: &EmbeddingComparator{},
	}
}

// ---------------------------------------------------------------------------
// Similarity matrix
// ---------------------------------------------------------------------------

// CompareTexts generates a similarity matrix as a formatted Unicode table.
func (ev *EmbeddingVisualizer) CompareTexts(ctx context.Context, texts []string, labels []string) (string, error) {
	if labels == nil {
		labels = make([]string, len(texts))
		for i := range texts {
			labels[i] = fmt.Sprintf("Text %d", i+1)
		}
	}
	if len(labels) != len(texts) {
		return "", fmt.Errorf("len(labels) must equal len(texts)")
	}

	embeddings, err := ev.Generator.EmbedBatch(ctx, texts)
	if err != nil {
		return "", err
	}
	n := len(texts)

	// Compute pairwise similarities.
	scores := make([][]float64, n)
	for i := range scores {
		scores[i] = make([]float64, n)
		for j := range scores[i] {
			sim, _ := ev.Comparator.CosineSimilarity(embeddings[i], embeddings[j])
			scores[i][j] = sim
		}
	}

	// Format as table.
	colW := 8
	labelW := 2
	for _, lbl := range labels {
		if len(lbl)+2 > labelW {
			labelW = len(lbl) + 2
		}
	}
	hLine := strings.Repeat("─", labelW)
	colLines := make([]string, n)
	for i := range colLines {
		colLines[i] = strings.Repeat("─", colW)
	}

	colsStr := strings.Join(colLines, "┬")
	top := "┌" + hLine + "┬" + colsStr + "┐"
	var headerCells strings.Builder
	headerCells.WriteString("│" + strings.Repeat(" ", labelW))
	for _, lbl := range labels {
		cell := fmt.Sprintf(" %-*s", colW-1, lbl)
		if len(cell) > colW {
			cell = cell[:colW]
		}
		headerCells.WriteString("│" + cell)
	}
	headerCells.WriteString("│")
	divider := "├" + hLine + "┼" + strings.Join(colLines, "┼") + "┤"
	bottom := "└" + hLine + "┴" + strings.Join(colLines, "┴") + "┘"

	var rows []string
	for i := 0; i < n; i++ {
		lbl := fmt.Sprintf(" %-*s", labelW-1, labels[i])
		var row strings.Builder
		row.WriteString("│" + lbl)
		for j := 0; j < n; j++ {
			cell := fmt.Sprintf(" %*.2f", colW-1, scores[i][j])
			row.WriteString("│" + cell)
		}
		row.WriteString("│")
		rows = append(rows, row.String())
	}

	parts := []string{top, headerCells.String(), divider}
	parts = append(parts, rows...)
	parts = append(parts, bottom)
	return strings.Join(parts, "\n"), nil
}

// ---------------------------------------------------------------------------
// Outlier detection
// ---------------------------------------------------------------------------

// FindOutliers finds texts that are semantically different from all others.
// A text is an outlier if its average similarity to all others is below
// the threshold_percentile of the pairwise distribution.
func (ev *EmbeddingVisualizer) FindOutliers(ctx context.Context, texts []string, labels []string, thresholdPercentile float64) ([]string, error) {
	if labels == nil {
		labels = texts
	}

	embeddings, err := ev.Generator.EmbedBatch(ctx, texts)
	if err != nil {
		return nil, err
	}
	n := len(texts)
	avgSims := make([]float64, n)
	for i := range texts {
		total := 0.0
		for j := range texts {
			if i != j {
				sim, _ := ev.Comparator.CosineSimilarity(embeddings[i], embeddings[j])
				total += sim
			}
		}
		if n > 1 {
			avgSims[i] = total / float64(n-1)
		} else {
			avgSims[i] = 1.0
		}
	}

	// Compute percentile cutoff.
	sorted := make([]float64, n)
	copy(sorted, avgSims)
	sort.Float64s(sorted)
	cutoffIdx := int(math.Floor(thresholdPercentile / 100.0 * float64(n)))
	if cutoffIdx >= n {
		cutoffIdx = n - 1
	}
	cutoff := sorted[cutoffIdx]

	var outliers []string
	for i, sim := range avgSims {
		if sim < cutoff {
			outliers = append(outliers, labels[i])
		}
	}
	return outliers, nil
}

// ---------------------------------------------------------------------------
// K-means clustering
// ---------------------------------------------------------------------------

// Cluster groups texts into semantic clusters using K-means on their embeddings.
// Returns a map of cluster index → list of texts in that cluster.
func (ev *EmbeddingVisualizer) Cluster(ctx context.Context, texts []string, nClusters, maxIterations int, randomSeed int64) (map[int][]string, error) {
	if nClusters > len(texts) {
		nClusters = len(texts)
	}
	embeddings, err := ev.Generator.EmbedBatch(ctx, texts)
	if err != nil {
		return nil, err
	}
	n := len(texts)
	dim := len(embeddings[0])

	rng := rand.New(rand.NewSource(randomSeed))

	// Choose k random initial centroids.
	centroidIdxs := rng.Perm(n)[:nClusters]
	centroids := make([][]float64, nClusters)
	for i, idx := range centroidIdxs {
		centroids[i] = make([]float64, dim)
		copy(centroids[i], embeddings[idx])
	}

	assignments := make([]int, n)

	for iter := 0; iter < maxIterations; iter++ {
		// Assign each text to the nearest centroid (by cosine similarity).
		changed := false
		for i := range texts {
			best := 0
			bestSim := cosineSim(embeddings[i], centroids[0])
			for k := 1; k < nClusters; k++ {
				sim := cosineSim(embeddings[i], centroids[k])
				if sim > bestSim {
					bestSim = sim
					best = k
				}
			}
			if assignments[i] != best {
				assignments[i] = best
				changed = true
			}
		}
		if !changed {
			break
		}

		// Recompute centroids.
		for k := 0; k < nClusters; k++ {
			newCentroid := make([]float64, dim)
			count := 0
			for i, a := range assignments {
				if a == k {
					for d := range newCentroid {
						newCentroid[d] += embeddings[i][d]
					}
					count++
				}
			}
			if count > 0 {
				for d := range newCentroid {
					newCentroid[d] /= float64(count)
				}
				centroids[k] = newCentroid
			}
		}
	}

	result := make(map[int][]string, nClusters)
	for k := 0; k < nClusters; k++ {
		result[k] = nil
	}
	for i, text := range texts {
		result[assignments[i]] = append(result[assignments[i]], text)
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// DemoSemanticCategories shows clear semantic clustering across categories.
func (ev *EmbeddingVisualizer) DemoSemanticCategories(ctx context.Context) {
	categories := map[string][]string{
		"weather": {
			"It's raining outside",
			"Sunny with a high of 75",
			"Snow expected tomorrow",
		},
		"technology": {
			"Python 3.12 released",
			"New GPU architecture announced",
			"API deprecation notice",
		},
		"food": {
			"Best pizza in New York",
			"How to make sourdough bread",
			"Restaurant review: French cuisine",
		},
	}

	var allTexts []string
	var shortLabels []string
	catNames := []string{"weather", "technology", "food"}
	for _, cat := range catNames {
		texts := categories[cat]
		for i, t := range texts {
			allTexts = append(allTexts, t)
			shortLabels = append(shortLabels, fmt.Sprintf("%s#%d", cat[:4], i+1))
		}
	}

	fmt.Println("=== Similarity matrix (9 texts × 9 texts) ===")
	table, err := ev.CompareTexts(ctx, allTexts, shortLabels)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Println(table)

	fmt.Println("\n=== K-means clustering (k=3) ===")
	clusters, err := ev.Cluster(ctx, allTexts, 3, 100, 42)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	for clusterID, members := range clusters {
		fmt.Printf("  Cluster %d:\n", clusterID)
		for _, text := range members {
			fmt.Printf("    - %s\n", text)
		}
	}

	fmt.Println("\n=== Outlier detection ===")
	weatherPlusOutlier := []string{
		"The weather is sunny today.",
		"It will rain tomorrow afternoon.",
		"Clear skies expected all week.",
		"The quarterly earnings beat expectations.",
	}
	outliers, err := ev.FindOutliers(ctx, weatherPlusOutlier, nil, 25.0)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("  Outliers: %v\n", outliers)
}

// RunEmbeddingVisualizer demonstrates the embedding visualizer.
func RunEmbeddingVisualizer() {
	gen := NewEmbeddingGenerator("text-embedding-3-small", 0)
	viz := NewEmbeddingVisualizer(gen)
	viz.DemoSemanticCategories(context.Background())
}
