/**
 * Extractive Summarizer — pull the most query-relevant sentences verbatim.
 *
 * Unlike abstractive summarization, extractive selection preserves original
 * wording, eliminating hallucination risk.
 * See: docs/04-context-engineering/03-context-compression-and-filtering.md
 */

export interface ExtractiveConfig {
  maxSentences?: number;
  minSentenceLength?: number;
}

/** Sentence-tokenize text. */
function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/** Compute term frequency for a token list. */
function termFrequency(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}

/** Tokenize for scoring: lowercase, strip punctuation. */
function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
}

/** Simple keyword overlap score between query and sentence. */
function keywordScore(queryTokens: string[], sentence: string): number {
  const sTokens = new Set(tokenize(sentence));
  const overlap = queryTokens.filter((t) => sTokens.has(t)).length;
  return overlap / Math.max(queryTokens.length, 1);
}

/**
 * Select the top-K most query-relevant sentences from text verbatim.
 */
export function extractiveSummarize(
  text: string,
  query: string,
  config: ExtractiveConfig = {}
): string[] {
  const maxSentences = config.maxSentences ?? 5;
  const minLen = config.minSentenceLength ?? 20;
  const queryTokens = tokenize(query);
  const sentences = splitSentences(text).filter((s) => s.length >= minLen);

  if (sentences.length === 0) return [];

  // Score each sentence
  const scored = sentences.map((s, i) => ({
    sentence: s,
    index: i,
    score: keywordScore(queryTokens, s) + tfIdfBonus(s, sentences),
  }));

  // Take top N by score, then re-order by original position
  const top = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSentences);

  return top
    .sort((a, b) => a.index - b.index)
    .map((s) => s.sentence);
}

/** Bonus score based on TF across all sentences (rare terms = more info). */
function tfIdfBonus(sentence: string, allSentences: string[]): number {
  const allText = allSentences.join(" ");
  const allTokens = tokenize(allText);
  const globalTf = termFrequency(allTokens);
  const sTokens = tokenize(sentence);
  const uniqueCount = new Set(sTokens).size;
  let bonus = 0;
  for (const t of new Set(sTokens)) {
    const freq = globalTf.get(t) ?? 1;
    bonus += 1 / freq; // inverse frequency bonus
  }
  return bonus / Math.max(uniqueCount, 1) * 0.1;
}

// Demo
function main(): void {
  const text = `
    Large language models are powerful AI systems trained on vast text corpora.
    They can generate human-like text, answer questions, and summarize documents.
    However, they sometimes produce hallucinations, which are factually incorrect statements.
    Retrieval-augmented generation (RAG) reduces hallucinations by grounding answers in retrieved documents.
    The context window limits how much information a model can process at once.
    Chunking strategies affect the quality of retrieved content significantly.
    Embeddings represent semantic meaning as dense numerical vectors.
    Vector databases enable fast nearest-neighbour search over embeddings.
    Fine-tuning adapts a pre-trained model to a specific domain.
    Prompt engineering guides model behavior through carefully crafted instructions.
  `.trim();

  const query = "How does RAG reduce hallucinations?";
  const result = extractiveSummarize(text, query, { maxSentences: 3 });

  console.log(`Query: "${query}"\n`);
  console.log("Extracted sentences:");
  result.forEach((s, i) => console.log(`  [${i + 1}] ${s}`));
}

main();
