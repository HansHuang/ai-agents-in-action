/**
 * Hybrid search: combine vector similarity with keyword (BM25) ranking.
 *
 * Pure vector search fails when exact terms matter. Hybrid search blends
 * vector and keyword ranked lists using weighted sum or RRF.
 * See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
 */

import { VectorDocument } from "./vector_database.js";

export interface HybridSearchConfig {
  vectorWeight: number;  // 0–1; keyword weight = 1 - vectorWeight
  k?: number;            // RRF k parameter (default 60)
  topK?: number;
  fusionMethod?: "weighted" | "rrf";
}

export interface ScoredDoc {
  document: VectorDocument;
  score: number;
  vectorScore: number;
  keywordScore: number;
}

// ---------------------------------------------------------------------------
// Simple BM25-like keyword scorer
// ---------------------------------------------------------------------------

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean);
}

function bm25Score(
  queryTokens: string[],
  docTokens: string[],
  avgDocLen: number,
  k1 = 1.5,
  b = 0.75
): number {
  const tf: Record<string, number> = {};
  for (const t of docTokens) tf[t] = (tf[t] ?? 0) + 1;
  let score = 0;
  const docLen = docTokens.length;
  for (const qt of queryTokens) {
    const f = tf[qt] ?? 0;
    if (f === 0) continue;
    score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * (docLen / avgDocLen)));
  }
  return score;
}

// ---------------------------------------------------------------------------
// HybridSearch
// ---------------------------------------------------------------------------

export class HybridSearch {
  constructor(private config: HybridSearchConfig) {}

  /**
   * Fuse vector results and keyword results into a single ranked list.
   *
   * @param vectorResults  Pre-ranked vector search results (index 0 = best)
   * @param query          Original query string for keyword scoring
   * @param allDocs        All documents in the corpus (for BM25 IDF)
   */
  fuse(
    vectorResults: VectorDocument[],
    query: string,
    allDocs: VectorDocument[]
  ): ScoredDoc[] {
    const queryTokens = tokenize(query);
    const avgDocLen =
      allDocs.reduce((s, d) => s + tokenize(d.text).length, 0) / (allDocs.length || 1);

    // Compute keyword scores
    const keywordScores = new Map<string, number>();
    for (const doc of allDocs) {
      keywordScores.set(doc.id, bm25Score(queryTokens, tokenize(doc.text), avgDocLen));
    }

    const topK = this.config.topK ?? 10;
    const method = this.config.fusionMethod ?? "rrf";

    if (method === "rrf") {
      return this.rrf(vectorResults, keywordScores, allDocs, topK);
    }
    return this.weighted(vectorResults, keywordScores, allDocs, topK);
  }

  private rrf(
    vectorResults: VectorDocument[],
    keywordScores: Map<string, number>,
    allDocs: VectorDocument[],
    topK: number
  ): ScoredDoc[] {
    const k = this.config.k ?? 60;
    const rrfScores = new Map<string, number>();

    // Vector rank contribution
    vectorResults.forEach((doc, rank) => {
      rrfScores.set(doc.id, (rrfScores.get(doc.id) ?? 0) + 1 / (k + rank + 1));
    });

    // Keyword rank contribution (sort by keyword score)
    const sortedByKeyword = [...allDocs].sort(
      (a, b) => (keywordScores.get(b.id) ?? 0) - (keywordScores.get(a.id) ?? 0)
    );
    sortedByKeyword.forEach((doc, rank) => {
      rrfScores.set(doc.id, (rrfScores.get(doc.id) ?? 0) + 1 / (k + rank + 1));
    });

    const docById = new Map(allDocs.map((d) => [d.id, d]));
    return Array.from(rrfScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([id, score]) => ({
        document: docById.get(id)!,
        score,
        vectorScore: 0,
        keywordScore: keywordScores.get(id) ?? 0,
      }))
      .filter((r) => r.document !== undefined);
  }

  private weighted(
    vectorResults: VectorDocument[],
    keywordScores: Map<string, number>,
    allDocs: VectorDocument[],
    topK: number
  ): ScoredDoc[] {
    const maxKeyword = Math.max(...Array.from(keywordScores.values()), 1);
    const vw = this.config.vectorWeight;
    const kw = 1 - vw;
    const n = vectorResults.length || 1;

    const scores = new Map<string, ScoredDoc>();
    vectorResults.forEach((doc, rank) => {
      const vs = 1 - rank / n;
      const ks = (keywordScores.get(doc.id) ?? 0) / maxKeyword;
      scores.set(doc.id, {
        document: doc,
        score: vw * vs + kw * ks,
        vectorScore: vs,
        keywordScore: ks,
      });
    });

    return Array.from(scores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}
