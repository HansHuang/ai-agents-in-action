/**
 * Context Optimizer — structure and prioritize context for LLM attention.
 *
 * - Position-aware reordering (critical info in the "golden middle")
 * - Chunk deduplication via Jaccard similarity
 * - Structured layout with section markers
 * See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
 */

export interface ContextChunk {
  id: string;
  text: string;
  priority: number;   // 0-1 (higher = more important)
  source?: string;
}

export interface OptimizerConfig {
  deduplicationThreshold?: number;  // Jaccard threshold (default 0.8)
  addTableOfContents?: boolean;
  goldenZoneStart?: number;         // fraction (default 0.2)
  goldenZoneEnd?: number;           // fraction (default 0.6)
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

function charNgrams(text: string, n = 3): Set<string> {
  const ngrams = new Set<string>();
  const normalized = text.toLowerCase().replace(/\s+/g, " ");
  for (let i = 0; i <= normalized.length - n; i++) {
    ngrams.add(normalized.slice(i, i + n));
  }
  return ngrams;
}

function jaccardSimilarity(a: string, b: string, n = 3): number {
  const setA = charNgrams(a, n);
  const setB = charNgrams(b, n);
  let intersection = 0;
  for (const g of setA) if (setB.has(g)) intersection++;
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 1 : intersection / union;
}

/** Remove near-duplicate chunks based on Jaccard similarity. */
export function deduplicate(chunks: ContextChunk[], threshold = 0.8): ContextChunk[] {
  const kept: ContextChunk[] = [];
  for (const chunk of chunks) {
    const isDuplicate = kept.some((k) => jaccardSimilarity(k.text, chunk.text) >= threshold);
    if (!isDuplicate) kept.push(chunk);
  }
  return kept;
}

// ---------------------------------------------------------------------------
// Position-aware reordering
// ---------------------------------------------------------------------------

/**
 * Reorder chunks so high-priority items land in the golden middle zone
 * (20-60% of position), where recall is best.
 */
export function reorderForAttention(
  chunks: ContextChunk[],
  config: OptimizerConfig = {}
): ContextChunk[] {
  const n = chunks.length;
  if (n === 0) return [];

  const start = Math.floor((config.goldenZoneStart ?? 0.2) * n);
  const end = Math.floor((config.goldenZoneEnd ?? 0.6) * n);

  const sorted = [...chunks].sort((a, b) => b.priority - a.priority);
  const highPriority = sorted.slice(0, end - start);
  const lowPriority = sorted.slice(end - start);

  // Fill: low priority first, high priority in middle, rest at end
  const result: ContextChunk[] = [];
  const midCount = Math.min(Math.floor(n * 0.2), lowPriority.length);
  result.push(...lowPriority.slice(0, midCount));
  result.push(...highPriority);
  result.push(...lowPriority.slice(midCount));
  return result;
}

// ---------------------------------------------------------------------------
// Structured layout
// ---------------------------------------------------------------------------

/** Build a structured context string with optional table of contents. */
export function buildStructuredContext(
  chunks: ContextChunk[],
  config: OptimizerConfig = {}
): string {
  const deduped = deduplicate(chunks, config.deduplicationThreshold);
  const ordered = reorderForAttention(deduped, config);

  const parts: string[] = [];

  if (config.addTableOfContents && ordered.length > 2) {
    const toc = ordered.map((c, i) => `  ${i + 1}. ${c.source ?? c.id}`).join("\n");
    parts.push(`## Context Overview\n${toc}`);
  }

  for (let i = 0; i < ordered.length; i++) {
    const c = ordered[i];
    const header = c.source ? `### [${i + 1}] ${c.source}` : `### [${i + 1}]`;
    parts.push(`${header}\n${c.text}`);
  }

  return parts.join("\n\n");
}

// Demo
function main(): void {
  const chunks: ContextChunk[] = [
    { id: "c1", text: "TypeScript adds static typing to JavaScript.", priority: 0.5, source: "intro" },
    { id: "c2", text: "RAG reduces hallucinations by grounding answers.", priority: 0.9, source: "rag-paper" },
    { id: "c3", text: "TypeScript adds static types to JavaScript.", priority: 0.4, source: "duplicate" },
    { id: "c4", text: "Vector databases store embedding vectors.", priority: 0.7, source: "db-overview" },
    { id: "c5", text: "The context window is a finite resource.", priority: 0.8, source: "context-guide" },
  ];

  const result = buildStructuredContext(chunks, { addTableOfContents: true, deduplicationThreshold: 0.7 });
  console.log(result);
}

main();
