/**
 * Progressive Summarizer — incremental, layered conversation summarization.
 *
 * TypeScript port of code/python/05-context-assembly/progressive_summarizer.py
 *
 * Older turns are compressed more aggressively than recent turns.
 *
 * Layers:
 *   Layer 0 (verbatim):  Last verbatimTurns turns, exact wording
 *   Layer 1 (detailed):  Moderate-detail summary of older turns
 *   Layer 2 (compressed): Key facts only
 *   Layer 3 (archival):  Highly compressed essence
 *
 * See: docs/04-context-engineering/04-multi-turn-context-management.md
 */

import { countTokens } from "./context_budget.js";

// ---------------------------------------------------------------------------
// LLM call helper (optional)
// ---------------------------------------------------------------------------

async function llmSummarize(
  prompt: string,
  model: string = "gpt-4o-mini",
): Promise<string> {
  // Dynamic import so the module works without openai installed
  const { default: OpenAI } = await import("openai" as string);
  const client = new OpenAI();
  const response = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: prompt }],
    temperature: 0.3,
  });
  return (response.choices[0].message.content ?? "").trim();
}

function llmAvailable(): boolean {
  return !!process.env["OPENAI_API_KEY"];
}

// ---------------------------------------------------------------------------
// Deterministic fallback summarizer
// ---------------------------------------------------------------------------

const STOP_WORDS = new Set(
  "i me my we our you your he she it its they them the a an and or " +
  "but in on at to for of with is was are were be been have has had " +
  "do does did will would could should may might shall this that these " +
  "those not no so just very really also then when where how what who " +
  "user assistant".split(" "),
);

function extractKeySentences(text: string, maxSentences: number = 6): string {
  const sentences = text.trim().split(/(?<=[.!?])\s+/).filter(Boolean);
  if (sentences.length === 0) return text;

  function score(s: string): number {
    const words = (s.toLowerCase().match(/\b\w{4,}\b/g) ?? []);
    const content = words.filter(w => !STOP_WORDS.has(w));
    return content.length / Math.max(words.length, 1);
  }

  const indexed = sentences.map((s, i) => ({ i, s, sc: score(s) }));
  indexed.sort((a, b) => b.sc - a.sc);
  const keepIndices = indexed.slice(0, maxSentences).map(x => x.i).sort((a, b) => a - b);
  return keepIndices.map(i => sentences[i]).join(" ");
}

function fallbackUpdateSummary(existing: string, newTurn: string): string {
  const combined = [existing, newTurn].filter(Boolean).join("\n");
  return extractKeySentences(combined, 8);
}

function fallbackCompress(content: string): string {
  return extractKeySentences(content, 5);
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const LAYER_UPDATE_PROMPT = `\
Update this conversation summary with new information.
Preserve: goals, decisions made, specific data (numbers, dates, names),
user preferences, agent recommendations, pending tasks.

Discard: small talk, repeated information, exact wording of resolved questions.

Existing summary: {existing}
New information: {new_turn}

Updated summary (keep approximately the same length):`;

const LAYER_COMPRESSION_PROMPT = `\
Compress this detailed conversation summary into a shorter version.
Keep only: the main goal, key decisions, critical facts, and unresolved items.
The compressed version should be about half the length.

Detailed summary: {layer_content}

Compressed summary:`;

// ---------------------------------------------------------------------------
// ProgressiveSummarizer
// ---------------------------------------------------------------------------

export interface SummarizerStats {
  totalTurnsProcessed: number;
  verbatimTurns:       number;
  layer1Tokens:        number;
  layer2Tokens:        number;
  layer3Tokens:        number;
  totalContextTokens:  number;
}

export interface SummarizerData {
  verbatim:        [string, string][];
  layers:          string[];
  totalTurns:      number;
  verbatimTurns:   number;
  layerSize:       number;
  layerTokenLimit: number;
}

export class ProgressiveSummarizer {
  static readonly NUM_LAYERS = 3;

  verbatimTurns:   number;
  layerSize:       number;
  model:           string;
  layerTokenLimit: number;

  verbatim: [string, string][] = [];
  layers:   string[];
  private _totalTurns: number = 0;

  constructor(
    verbatimTurns:   number = 5,
    layerSize:       number = 10,
    model:           string = "gpt-4o-mini",
    layerTokenLimit: number = 1500,
  ) {
    this.verbatimTurns   = verbatimTurns;
    this.layerSize       = layerSize;
    this.model           = model;
    this.layerTokenLimit = layerTokenLimit;
    this.layers          = Array(ProgressiveSummarizer.NUM_LAYERS).fill("");
  }

  // ------------------------------------------------------------------
  // Public interface
  // ------------------------------------------------------------------

  async addTurn(userMsg: string, assistantMsg: string): Promise<void> {
    this._totalTurns++;
    this.verbatim.push([userMsg, assistantMsg]);

    if (this.verbatim.length > this.verbatimTurns) {
      const oldest = this.verbatim.shift()!;
      const turnText = `User: ${oldest[0]}\nAssistant: ${oldest[1]}`;
      await this._incorporateIntoLayer(0, turnText);
    }
  }

  getContext(): string {
    const parts: string[] = [];
    const labels = ["Early conversation", "Earlier conversation", "Recent conversation"];

    for (let i = ProgressiveSummarizer.NUM_LAYERS - 1; i >= 0; i--) {
      const content = this.layers[i].trim();
      if (content)
        parts.push(`[${labels[i]}:\n${content}]`);
    }

    if (this.verbatim.length > 0) {
      const lines: string[] = [];
      for (const [u, a] of this.verbatim) {
        lines.push(`User: ${u}`);
        lines.push(`Assistant: ${a}`);
      }
      parts.push("[Most recent turns:\n" + lines.join("\n") + "]");
    }

    return parts.join("\n\n");
  }

  getStats(): SummarizerStats {
    const layerTokens = this.layers.map(l => countTokens(l));
    const verbatimText = this.verbatim.map(([u, a]) => `${u} ${a}`).join(" ");
    const verbatimTokens = countTokens(verbatimText);
    return {
      totalTurnsProcessed: this._totalTurns,
      verbatimTurns:       this.verbatim.length,
      layer1Tokens:        layerTokens[0],
      layer2Tokens:        layerTokens[1],
      layer3Tokens:        layerTokens[2],
      totalContextTokens:  layerTokens.reduce((a, b) => a + b, 0) + verbatimTokens,
    };
  }

  // ------------------------------------------------------------------
  // Persistence
  // ------------------------------------------------------------------

  toDict(): SummarizerData {
    return {
      verbatim:        [...this.verbatim],
      layers:          [...this.layers],
      totalTurns:      this._totalTurns,
      verbatimTurns:   this.verbatimTurns,
      layerSize:       this.layerSize,
      layerTokenLimit: this.layerTokenLimit,
    };
  }

  static fromDict(data: SummarizerData): ProgressiveSummarizer {
    const s = new ProgressiveSummarizer(
      data.verbatimTurns,
      data.layerSize,
      "gpt-4o-mini",
      data.layerTokenLimit,
    );
    s.verbatim     = data.verbatim ?? [];
    s.layers       = data.layers   ?? Array(ProgressiveSummarizer.NUM_LAYERS).fill("");
    s._totalTurns  = data.totalTurns ?? 0;
    return s;
  }

  // ------------------------------------------------------------------
  // Internal layer management
  // ------------------------------------------------------------------

  private async _incorporateIntoLayer(
    layerIndex: number,
    turnText: string,
  ): Promise<void> {
    if (layerIndex >= ProgressiveSummarizer.NUM_LAYERS) {
      this.layers[ProgressiveSummarizer.NUM_LAYERS - 1] =
        await this._compressContent(
          this.layers[ProgressiveSummarizer.NUM_LAYERS - 1],
          turnText,
          ProgressiveSummarizer.NUM_LAYERS,
        );
      return;
    }

    const existing = this.layers[layerIndex];
    this.layers[layerIndex] = await this._updateSummary(existing, turnText);

    if (countTokens(this.layers[layerIndex]) > this.layerTokenLimit)
      await this._cascadeOverflow(layerIndex);
  }

  private async _cascadeOverflow(fromLayer: number): Promise<void> {
    const toLayer = fromLayer + 1;
    if (toLayer >= ProgressiveSummarizer.NUM_LAYERS) {
      this.layers[fromLayer] = await this._compressContent(
        this.layers[fromLayer], "", fromLayer + 1,
      );
      return;
    }
    this.layers[toLayer] = await this._compressContent(
      this.layers[toLayer],
      this.layers[fromLayer],
      toLayer + 1,
    );
    this.layers[fromLayer] = "";

    if (countTokens(this.layers[toLayer]) > this.layerTokenLimit)
      await this._cascadeOverflow(toLayer);
  }

  private async _updateSummary(existing: string, newTurn: string): Promise<string> {
    if (!existing) return newTurn;

    if (llmAvailable()) {
      try {
        const prompt = LAYER_UPDATE_PROMPT
          .replace("{existing}", existing || "(none)")
          .replace("{new_turn}", newTurn);
        return await llmSummarize(prompt, this.model);
      } catch { /* fall through */ }
    }

    return fallbackUpdateSummary(existing, newTurn);
  }

  private async _compressContent(
    older: string,
    newer: string,
    depth: number,
  ): Promise<string> {
    const combined = [older, newer].filter(Boolean).join("\n");
    if (!combined) return "";

    if (llmAvailable()) {
      try {
        const prompt = LAYER_COMPRESSION_PROMPT
          .replace("{layer_content}", combined);
        return await llmSummarize(prompt, this.model);
      } catch { /* fall through */ }
    }

    const maxSentences = Math.max(3, 8 - depth * 2);
    return extractKeySentences(combined, maxSentences);
  }
}
