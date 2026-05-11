/**
 * Document chunking strategies for embedding pipelines.
 *
 * Splits documents into smaller pieces suitable for embedding and search.
 * Supports fixed-size, sentence-boundary, and overlap chunking.
 *
 * Token counting uses a fast approximation (4 chars ≈ 1 token for English).
 * See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
 */

export interface Chunk {
  id: string;
  text: string;
  index: number;
  startChar: number;
  endChar: number;
  tokenCount: number;
  metadata: Record<string, unknown>;
}

export interface ChunkerConfig {
  chunkSize: number;    // approximate tokens
  chunkOverlap: number; // approximate tokens
}

/** Approximate token count: ~4 characters per token for English text. */
export function approxTokenCount(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Split text into overlapping fixed-size chunks (token approximation).
 */
export function chunkByTokens(
  text: string,
  sourceId: string,
  config: ChunkerConfig = { chunkSize: 512, chunkOverlap: 64 }
): Chunk[] {
  const charsPerToken = 4;
  const chunkChars = config.chunkSize * charsPerToken;
  const overlapChars = config.chunkOverlap * charsPerToken;
  const step = chunkChars - overlapChars;

  const chunks: Chunk[] = [];
  let index = 0;

  for (let start = 0; start < text.length; start += step) {
    const end = Math.min(start + chunkChars, text.length);
    const chunkText = text.slice(start, end);
    chunks.push({
      id: `${sourceId}-chunk-${index}`,
      text: chunkText,
      index,
      startChar: start,
      endChar: end,
      tokenCount: approxTokenCount(chunkText),
      metadata: { sourceId },
    });
    index++;
    if (end >= text.length) break;
  }
  return chunks;
}

/**
 * Split text by sentence boundaries, merging sentences up to maxTokens.
 */
export function chunkBySentence(
  text: string,
  sourceId: string,
  maxTokens = 512
): Chunk[] {
  const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text];
  const chunks: Chunk[] = [];
  let current = "";
  let index = 0;

  for (const sentence of sentences) {
    const candidate = current ? `${current} ${sentence.trim()}` : sentence.trim();
    if (approxTokenCount(candidate) > maxTokens && current) {
      chunks.push({
        id: `${sourceId}-chunk-${index}`,
        text: current.trim(),
        index,
        startChar: 0,
        endChar: current.length,
        tokenCount: approxTokenCount(current),
        metadata: { sourceId },
      });
      current = sentence.trim();
      index++;
    } else {
      current = candidate;
    }
  }
  if (current.trim()) {
    chunks.push({
      id: `${sourceId}-chunk-${index}`,
      text: current.trim(),
      index,
      startChar: 0,
      endChar: current.length,
      tokenCount: approxTokenCount(current),
      metadata: { sourceId },
    });
  }
  return chunks;
}
