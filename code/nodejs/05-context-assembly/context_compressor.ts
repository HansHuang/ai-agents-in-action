/**
 * Context Compressor — five-stage filtering pipeline for LLM context.
 *
 * TypeScript port of code/python/05-context-assembly/context_compressor.py
 *
 * Pipeline stages:
 *   1. Relevance filter   — remove low-similarity documents (adaptive threshold)
 *   2. Quality filter     — remove near-duplicates and low-density chunks
 *   3. Rerank             — reorder by embedding similarity
 *   4. Extractive compress — keep query-relevant sentences verbatim
 *   5. Budget enforcement  — drop lowest-scoring documents until within budget
 *
 * See: docs/04-context-engineering/03-context-compression-and-filtering.md
 */

import { countTokens } from "./context_budget.js";

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  return magA === 0 || magB === 0 ? 0 : dot / (magA * magB);
}

// ---------------------------------------------------------------------------
// Sentence splitter
// ---------------------------------------------------------------------------

const ABBREV_RE = /\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|etc|vs|e\.g|i\.e|approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|U\.S|U\.K|E\.U|Corp|Inc|Ltd|LLC)\./gi;
const DECIMAL_RE = /(\d+)\.(\d)/g;
const BOUNDARY_RE = /(?<=[.!?])\s+(?=[A-Z])/;

class SentenceSplitter {
  split(text: string): string[] {
    const MASK = "\x00";
    let masked = text.replace(ABBREV_RE, (m) => m.replace(".", MASK));
    masked = masked.replace(DECIMAL_RE, `$1${MASK}$2`);
    const parts = masked.split(BOUNDARY_RE);
    return parts
      .map((s) => s.replace(/\x00/g, ".").trim())
      .filter(Boolean);
  }
}

// ---------------------------------------------------------------------------
// Keyword embedder — deterministic mock for offline use / tests
// ---------------------------------------------------------------------------

const DIM = 32;
const VOCAB: Record<string, number> = {
  weather: 0, temperature: 0, rain: 0, sunny: 0, forecast: 0,
  stock: 1, price: 1, market: 1, shares: 1, dividend: 1,
  sport: 2, game: 2, team: 2, score: 2, player: 2,
  return: 3, refund: 3, policy: 3, damaged: 3, receipt: 3,
  billing: 4, invoice: 4, payment: 4, charge: 4, cost: 4,
  shipping: 5, delivery: 5, package: 5, tracking: 5, order: 5,
  security: 6, password: 6, authentication: 6, access: 6,
  performance: 7, speed: 7, latency: 7, benchmark: 7,
};

export class KeywordEmbedder {
  embed(text: string): number[] {
    const vec = new Array<number>(DIM).fill(0);
    const words = text.toLowerCase().match(/\b\w+\b/g) ?? [];
    for (const word of words) {
      if (word in VOCAB) vec[VOCAB[word]] += 1;
    }
    const mag = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
    if (mag > 0) return vec.map((x) => x / mag);
    const val = 1 / Math.sqrt(DIM);
    return vec.map(() => val);
  }
}

export interface Embedder {
  embed(text: string): number[];
}

// ---------------------------------------------------------------------------
// Density analyzer (inline lightweight version)
// ---------------------------------------------------------------------------

const STOP_WORDS = new Set([
  "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
  "from","as","is","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","could","should","may","might","must","can",
  "it","its","this","that","these","those","i","we","you","he","she","they",
  "me","us","him","her","them","my","our","your","his","their","what","which",
  "who","when","where","why","how","all","any","both","each","few","more",
  "most","other","some","such","no","not","only","so","than","too","very",
  "just","also","if","then","there","here","about","after","before","into",
  "through","over","under","again","while","because","since","until","however",
  "therefore","yet","still","already",
]);

function densityScore(text: string): number {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return 0;

  const numberMatches = text.match(/\b\d[\d,./\-]*%?\b/g) ?? [];
  const namedMatches  = text.match(/(?<!\.\s)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b/g) ?? [];
  const factDensity   = Math.min(1, (numberMatches.length + namedMatches.length) / words.length);

  const stopCount     = words.filter((w) => STOP_WORDS.has(w.toLowerCase().replace(/[.,;:!?"'()[\]]/g, ""))).length;
  const fillerRatio   = stopCount / words.length;

  return Math.max(0, Math.min(1,
    factDensity * 0.35 + (1 - fillerRatio) * 0.30 + 0.35 * 0   // no structure/specificity for speed
  + factDensity * 0.20 + (1 - fillerRatio) * 0.15
  ));
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface CompressionConfig {
  targetTokens?:      number;
  minResults?:        number;
  maxResults?:        number;
  dedupThreshold?:    number;
  minDensity?:        number;
  rerankMethod?:      "embedding" | "none";
  compressMethod?:    "extractive" | "none";
  adaptiveThreshold?: boolean;
  fixedThreshold?:    number;
}

function resolveConfig(cfg: CompressionConfig, targetTokens: number): Required<CompressionConfig> {
  return {
    targetTokens:      cfg.targetTokens      ?? targetTokens,
    minResults:        cfg.minResults        ?? 3,
    maxResults:        cfg.maxResults        ?? 15,
    dedupThreshold:    cfg.dedupThreshold    ?? 0.9,
    minDensity:        cfg.minDensity        ?? 0.1,
    rerankMethod:      cfg.rerankMethod      ?? "none",
    compressMethod:    cfg.compressMethod    ?? "extractive",
    adaptiveThreshold: cfg.adaptiveThreshold ?? true,
    fixedThreshold:    cfg.fixedThreshold    ?? 0.6,
  };
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

export interface StageRecord {
  name:       string;
  docCount:   number;
  tokenCount: number;
}

export class CompressionAudit {
  constructor(
    public readonly stages:        StageRecord[],
    public readonly initialDocs:   number,
    public readonly initialTokens: number,
  ) {}

  report(): string {
    const col = 22;
    const lines = [
      `${"Stage".padEnd(col)}  ${"Docs".padStart(5)}  ${"Tokens".padStart(9)}  ${"% Remaining".padStart(12)}`,
      "─".repeat(col + 34),
    ];
    for (const rec of this.stages) {
      const pct = ((rec.tokenCount / Math.max(this.initialTokens, 1)) * 100).toFixed(0);
      lines.push(
        `${rec.name.padEnd(col)}  ${String(rec.docCount).padStart(5)}  ` +
        `${rec.tokenCount.toLocaleString().padStart(9)}  ${pct.padStart(11)}%`
      );
    }
    return lines.join("\n");
  }

  toDict(): object {
    return {
      initialDocs:   this.initialDocs,
      initialTokens: this.initialTokens,
      stages:        this.stages.map((r) => ({
        name:          r.name,
        docs:          r.docCount,
        tokens:        r.tokenCount,
        pctRemaining:  +((r.tokenCount / Math.max(this.initialTokens, 1)) * 100).toFixed(1),
      })),
    };
  }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

export interface CompressionResult {
  documents: Record<string, unknown>[];
  stats:     Record<string, { docs: number; tokens: number }>;
  audit:     CompressionAudit;
}

// ---------------------------------------------------------------------------
// ContextCompressor
// ---------------------------------------------------------------------------

export class ContextCompressor {
  private readonly splitter = new SentenceSplitter();

  constructor(
    private readonly embedder: Embedder = new KeywordEmbedder(),
  ) {}

  // ------------------------------------------------------------------
  // Stage 1 — Relevance filter
  // ------------------------------------------------------------------

  filterByRelevance(
    documents: Record<string, unknown>[],
    threshold?: number,
    minResults = 3,
    maxResults = 15,
  ): Record<string, unknown>[] {
    if (documents.length === 0) return documents;

    const t = threshold ?? this._adaptiveThreshold(documents, minResults);
    let filtered = documents.filter((d) => ((d["score"] as number) ?? 0) >= t);

    if (filtered.length < minResults) {
      filtered = [...documents]
        .sort((a, b) => ((b["score"] as number) ?? 0) - ((a["score"] as number) ?? 0))
        .slice(0, minResults);
    }

    return filtered.slice(0, maxResults);
  }

  // ------------------------------------------------------------------
  // Stage 2 — Quality filter
  // ------------------------------------------------------------------

  filterByQuality(
    documents: Record<string, unknown>[],
    dedupThreshold = 0.9,
    minDensity = 0.1,
  ): Record<string, unknown>[] {
    let docs = this._deduplicate(documents, dedupThreshold);
    docs = docs.filter((d) => {
      const text = (d["text"] as string) ?? "";
      return densityScore(text) >= minDensity;
    });
    return docs;
  }

  // ------------------------------------------------------------------
  // Stage 3 — Rerank
  // ------------------------------------------------------------------

  rerank(
    query: string,
    documents: Record<string, unknown>[],
    topK = 10,
  ): Record<string, unknown>[] {
    const qEmb = this.embedder.embed(query);
    const scored = documents.map((doc) => ({
      ...doc,
      rerankScore: cosineSimilarity(this.embedder.embed((doc["text"] as string) ?? ""), qEmb),
    }));
    scored.sort((a, b) => (b["rerankScore"] as number) - (a["rerankScore"] as number));
    return scored.slice(0, topK);
  }

  // ------------------------------------------------------------------
  // Stage 4 — Compress
  // ------------------------------------------------------------------

  compressDocuments(
    query: string,
    documents: Record<string, unknown>[],
    maxTokensPerDoc = 300,
    method: "extractive" | "none" = "extractive",
  ): Record<string, unknown>[] {
    if (method === "none") return documents;
    return documents.map((doc) => this._compressOne(doc, query, maxTokensPerDoc));
  }

  // ------------------------------------------------------------------
  // Stage 5 — Budget enforcement
  // ------------------------------------------------------------------

  enforceBudget(
    documents: Record<string, unknown>[],
    targetTokens: number,
  ): Record<string, unknown>[] {
    const docs = [...documents];
    while (true) {
      const total = docs.reduce((s, d) => s + countTokens((d["text"] as string) ?? ""), 0);
      if (total <= targetTokens || docs.length <= 1) break;
      const worst = docs.reduce(
        (minIdx, d, i) =>
          (((d["rerankScore"] ?? d["score"]) as number) ?? 0) <
          (((docs[minIdx]["rerankScore"] ?? docs[minIdx]["score"]) as number) ?? 0)
            ? i
            : minIdx,
        0,
      );
      docs.splice(worst, 1);
    }
    return docs;
  }

  // ------------------------------------------------------------------
  // Full pipeline
  // ------------------------------------------------------------------

  compress(
    query: string,
    documents: Record<string, unknown>[],
    targetTokens = 12_000,
    config: CompressionConfig = {},
  ): CompressionResult {
    const cfg  = resolveConfig(config, targetTokens);
    let   docs = [...documents];
    const stages: StageRecord[] = [];

    const snap = (name: string) => {
      stages.push({
        name,
        docCount:   docs.length,
        tokenCount: docs.reduce((s, d) => s + countTokens((d["text"] as string) ?? ""), 0),
      });
    };

    snap("Raw Input");

    // Stage 1
    docs = this.filterByRelevance(
      docs,
      cfg.adaptiveThreshold ? undefined : cfg.fixedThreshold,
      cfg.minResults,
      cfg.maxResults,
    );
    snap("Relevance Filter");

    // Stage 2
    docs = this.filterByQuality(docs, cfg.dedupThreshold, cfg.minDensity);
    snap("Quality Filter");

    // Stage 3
    if (cfg.rerankMethod !== "none") {
      docs = this.rerank(query, docs, cfg.maxResults);
    }
    snap(`Rerank (${cfg.rerankMethod})`);

    // Stage 4
    if (cfg.compressMethod !== "none") {
      const tokensPerDoc = Math.max(50, Math.floor(cfg.targetTokens / Math.max(docs.length, 1)));
      docs = this.compressDocuments(query, docs, tokensPerDoc, cfg.compressMethod);
    }
    snap("Extract/Compress");

    // Stage 5
    docs = this.enforceBudget(docs, cfg.targetTokens);
    snap("Budget Enforcement");

    const audit = new CompressionAudit(stages, stages[0].docCount, stages[0].tokenCount);
    const stats = Object.fromEntries(
      stages.map((r) => [r.name, { docs: r.docCount, tokens: r.tokenCount }]),
    );
    return { documents: docs, stats, audit };
  }

  // ------------------------------------------------------------------
  // Private helpers
  // ------------------------------------------------------------------

  private _adaptiveThreshold(documents: Record<string, unknown>[], minResults: number): number {
    const scores = documents.map((d) => ((d["score"] as number) ?? 0)).sort((a, b) => b - a);
    for (const t of [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]) {
      if (scores.filter((s) => s >= t).length >= minResults) return t;
    }
    return 0.3;
  }

  private _deduplicate(
    documents: Record<string, unknown>[],
    threshold: number,
  ): Record<string, unknown>[] {
    const ngrams = (text: string, n = 3): Set<string> => {
      const t = text.toLowerCase();
      const s = new Set<string>();
      for (let i = 0; i <= t.length - n; i++) s.add(t.slice(i, i + n));
      return s;
    };
    const jaccard = (a: string, b: string): number => {
      const sa = ngrams(a), sb = ngrams(b);
      if (sa.size === 0 || sb.size === 0) return 0;
      const intersection = [...sa].filter((x) => sb.has(x)).length;
      return intersection / (sa.size + sb.size - intersection);
    };
    const kept: Record<string, unknown>[] = [];
    for (const doc of documents) {
      const text = (doc["text"] as string) ?? "";
      const isDup = kept.some((k) => jaccard(text, (k["text"] as string) ?? "") >= threshold);
      if (!isDup) kept.push(doc);
    }
    return kept;
  }

  private _compressOne(
    doc: Record<string, unknown>,
    query: string,
    maxTokens: number,
  ): Record<string, unknown> {
    const text = (doc["text"] as string) ?? "";
    const origTokens = countTokens(text);
    if (origTokens <= maxTokens) return doc;

    const qEmb      = this.embedder.embed(query);
    const sentences = this.splitter.split(text).filter((s) => s.length >= 20);
    if (sentences.length === 0) return doc;

    const maxSentences = Math.max(1, Math.floor(maxTokens / 30));
    const scores = sentences.map((s) => cosineSimilarity(this.embedder.embed(s), qEmb));

    const keep = new Set<number>();
    if (sentences.length > maxSentences) {
      keep.add(0);
      keep.add(sentences.length - 1);
    }
    const ranked = scores
      .map((s, i) => ({ i, s }))
      .filter(({ i }) => !keep.has(i))
      .sort((a, b) => b.s - a.s);
    for (let j = 0; j < Math.max(0, maxSentences - keep.size); j++) {
      if (j < ranked.length) keep.add(ranked[j].i);
    }

    const compressed = [...keep].sort((a, b) => a - b).map((i) => sentences[i]).join(" ");
    return {
      ...doc,
      text:              compressed,
      compressed:        true,
      originalTokens:    origTokens,
      compressedTokens:  countTokens(compressed),
    };
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function demo(): void {
  const compressor = new ContextCompressor(new KeywordEmbedder());

  const docs: Record<string, unknown>[] = [
    { text: "Damaged items must be reported within 48 hours with photos for a full refund.", score: 0.92 },
    { text: "To return a damaged product, visit returns.acme.com and select 'Damaged Item'.", score: 0.89 },
    { text: "Standard returns are accepted within 30 days of purchase.", score: 0.65 },
    { text: "Refunds are processed in 5–7 business days after we receive the return.", score: 0.63 },
    { text: "We offer free shipping on orders over $50.", score: 0.30 },
    { text: "Our headquarters is located in Austin, Texas.", score: 0.08 },
    { text: "Thank you for choosing us. We appreciate your patience.", score: 0.12 },
    { text: "Thank you for choosing us. We appreciate your patience. (dup)", score: 0.11 },
    { text: "Quarterly earnings exceeded analyst expectations by 12%.", score: 0.05 },
  ];

  const result = compressor.compress("How do I return a damaged item?", docs, 500);
  console.log(result.audit.report());
  console.log(`\nFinal docs: ${result.documents.length}`);
}

demo();
