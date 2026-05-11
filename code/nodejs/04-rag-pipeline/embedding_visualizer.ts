/**
 * Embedding visualization utilities for building semantic intuition.
 *
 * Provides similarity matrices, outlier detection, and K-means clustering
 * over embedding vectors — all without requiring a display library.
 * See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
 */

export type Vector = number[];

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

export function cosineSimilarity(a: Vector, b: Vector): number {
  if (a.length !== b.length) throw new Error("Vector length mismatch");
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ---------------------------------------------------------------------------
// Similarity matrix
// ---------------------------------------------------------------------------

export interface SimilarityMatrix {
  labels: string[];
  matrix: number[][];
}

/** Build an N×N cosine similarity matrix. */
export function buildSimilarityMatrix(
  embeddings: { label: string; vector: Vector }[]
): SimilarityMatrix {
  const n = embeddings.length;
  const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const sim = cosineSimilarity(embeddings[i].vector, embeddings[j].vector);
      matrix[i][j] = sim;
      matrix[j][i] = sim;
    }
  }
  return { labels: embeddings.map((e) => e.label), matrix };
}

/** Print the similarity matrix as a text table. */
export function printSimilarityMatrix(m: SimilarityMatrix): void {
  const pad = 8;
  const header = "".padEnd(pad) + m.labels.map((l) => l.slice(0, pad).padStart(pad)).join("");
  console.log(header);
  for (let i = 0; i < m.matrix.length; i++) {
    const row = m.labels[i].slice(0, pad - 1).padEnd(pad - 1) +
      m.matrix[i].map((v) => v.toFixed(2).padStart(pad)).join("");
    console.log(row);
  }
}

// ---------------------------------------------------------------------------
// Outlier detection
// ---------------------------------------------------------------------------

export interface OutlierResult {
  label: string;
  avgSimilarity: number;
  isOutlier: boolean;
}

/** Detect outlier embeddings (low average similarity to the rest). */
export function detectOutliers(
  embeddings: { label: string; vector: Vector }[],
  zThreshold = 1.5
): OutlierResult[] {
  const n = embeddings.length;
  const avgSims = embeddings.map((e, i) => {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      if (i !== j) sum += cosineSimilarity(e.vector, embeddings[j].vector);
    }
    return sum / Math.max(n - 1, 1);
  });

  const mean = avgSims.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(avgSims.reduce((a, b) => a + (b - mean) ** 2, 0) / n);

  return embeddings.map((e, i) => ({
    label: e.label,
    avgSimilarity: avgSims[i],
    isOutlier: std > 0 && (mean - avgSims[i]) / std > zThreshold,
  }));
}

// ---------------------------------------------------------------------------
// K-means clustering
// ---------------------------------------------------------------------------

export interface ClusterResult {
  clusterIndex: number;
  label: string;
  vector: Vector;
}

/** Simple k-means clustering over embedding vectors. */
export function kMeansClustering(
  embeddings: { label: string; vector: Vector }[],
  k: number,
  maxIter = 50
): ClusterResult[] {
  if (embeddings.length === 0 || k <= 0) return [];
  k = Math.min(k, embeddings.length);

  // Initialize centroids from first k embeddings
  let centroids: Vector[] = embeddings.slice(0, k).map((e) => [...e.vector]);
  let assignments = new Array<number>(embeddings.length).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign
    const newAssignments = embeddings.map((e) => {
      let best = 0, bestSim = -Infinity;
      for (let c = 0; c < k; c++) {
        const sim = cosineSimilarity(e.vector, centroids[c]);
        if (sim > bestSim) { bestSim = sim; best = c; }
      }
      return best;
    });

    if (newAssignments.every((v, i) => v === assignments[i])) break;
    assignments = newAssignments;

    // Update centroids
    centroids = Array.from({ length: k }, (_, c) => {
      const members = embeddings.filter((_, i) => assignments[i] === c);
      if (members.length === 0) return centroids[c];
      const sum = members.reduce<Vector>(
        (acc, e) => acc.map((v, i) => v + e.vector[i]),
        new Array(embeddings[0].vector.length).fill(0)
      );
      return sum.map((v) => v / members.length);
    });
  }

  return embeddings.map((e, i) => ({
    clusterIndex: assignments[i],
    label: e.label,
    vector: e.vector,
  }));
}

// Demo
function main(): void {
  const items = [
    { label: "cat", vector: [0.9, 0.1, 0.1] },
    { label: "dog", vector: [0.85, 0.15, 0.05] },
    { label: "car", vector: [0.1, 0.9, 0.1] },
    { label: "truck", vector: [0.05, 0.85, 0.15] },
    { label: "apple", vector: [0.1, 0.1, 0.9] },
  ];

  console.log("Similarity matrix:");
  printSimilarityMatrix(buildSimilarityMatrix(items));

  console.log("\nOutlier detection:");
  detectOutliers(items).forEach((r) => {
    console.log(`  ${r.label}: avgSim=${r.avgSimilarity.toFixed(3)} outlier=${r.isOutlier}`);
  });

  console.log("\nK-means (k=2):");
  kMeansClustering(items, 2).forEach((r) => {
    console.log(`  cluster ${r.clusterIndex}: ${r.label}`);
  });
}

main();
