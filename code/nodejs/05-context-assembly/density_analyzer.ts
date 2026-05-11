/**
 * Information Density Analyzer — score text for information density.
 *
 * Higher density means more facts per token. Low-density chunks
 * (boilerplate, filler) waste the context budget.
 * See: docs/04-context-engineering/03-context-compression-and-filtering.md
 */

const STOP_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
  "been", "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "must", "can", "it", "its",
  "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
  "not", "no", "so", "if", "then", "than", "very", "just", "also",
]);

export interface DensityScore {
  text: string;
  score: number;          // 0-1 overall density score
  uniqueTermRatio: number;
  contentWordRatio: number;
  entityDensity: number;
  avgWordLength: number;
}

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
}

function countEntities(text: string): number {
  const capitalizedWords = text.match(/\b[A-Z][a-zA-Z]{1,}\b/g) ?? [];
  const numbers = text.match(/\b\d+(?:\.\d+)?(?:[KMBkmb%$€£])?\b/g) ?? [];
  return new Set([...capitalizedWords, ...numbers]).size;
}

/** Compute an information density score for a piece of text. */
export function analyzeDensity(text: string): DensityScore {
  const words = tokenize(text);
  if (words.length === 0) {
    return { text, score: 0, uniqueTermRatio: 0, contentWordRatio: 0, entityDensity: 0, avgWordLength: 0 };
  }

  const unique = new Set(words);
  const contentWords = words.filter((w) => !STOP_WORDS.has(w));
  const entities = countEntities(text);
  const avgLen = words.reduce((s, w) => s + w.length, 0) / words.length;

  const uniqueTermRatio = unique.size / words.length;
  const contentWordRatio = contentWords.length / words.length;
  const entityDensity = Math.min(entities / words.length, 1.0);

  // Weighted combination
  const score = Math.min(
    0.4 * contentWordRatio +
    0.3 * uniqueTermRatio +
    0.2 * entityDensity +
    0.1 * Math.min(avgLen / 8, 1.0),
    1.0
  );

  return { text, score, uniqueTermRatio, contentWordRatio, entityDensity, avgWordLength: avgLen };
}

/** Rank chunks by density and optionally filter below a threshold. */
export function rankByDensity(chunks: string[], threshold = 0): DensityScore[] {
  return chunks
    .map(analyzeDensity)
    .filter((s) => s.score >= threshold)
    .sort((a, b) => b.score - a.score);
}

// Demo
function main(): void {
  const samples = [
    "The quick brown fox jumps over the lazy dog.",
    "OpenAI released GPT-4 in March 2023 with 1.8T parameters and 128K context.",
    "Actually, you know, it is very important to note that this is indeed quite relevant.",
    "BERT, RoBERTa, and DeBERTa are transformer encoder models achieving SOTA on GLUE.",
  ];

  console.log("Density Analysis:");
  const ranked = rankByDensity(samples);
  ranked.forEach((r, i) => {
    console.log(`\n  [${i + 1}] score=${r.score.toFixed(3)}`);
    console.log(`      content_ratio=${r.contentWordRatio.toFixed(2)} entity_density=${r.entityDensity.toFixed(2)}`);
    console.log(`      "${r.text.slice(0, 70)}"`);
  });
}

main();
