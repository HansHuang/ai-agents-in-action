/**
 * Conversation summarizer — compresses message history for memory management.
 *
 * Uses a cheap model to produce dense, information-preserving summaries of
 * conversation history.
 * See: docs/03-memory-and-retrieval/01-short-term-memory.md
 */

import OpenAI from "openai";

const DEFAULT_MODEL = "gpt-4o-mini";

export interface SummarizationConfig {
  model?: string;
  maxSummaryTokens?: number;
  summaryStyle?: "bullet" | "narrative";
}

/**
 * Compress a list of messages into a concise summary.
 */
export async function summarizeMessages(
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
  client: OpenAI,
  config: SummarizationConfig = {}
): Promise<string> {
  const model = config.model ?? DEFAULT_MODEL;
  const maxTokens = config.maxSummaryTokens ?? 256;
  const style = config.summaryStyle ?? "bullet";

  const conversation = messages
    .filter((m) => m.role !== "system")
    .map((m) => `${m.role.toUpperCase()}: ${typeof m.content === "string" ? m.content : ""}`)
    .join("\n");

  const styleInstruction =
    style === "bullet"
      ? "Use bullet points for each key fact or decision."
      : "Write a concise narrative paragraph.";

  const response = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "system",
        content:
          "You compress conversation history into dense, information-preserving summaries. " +
          styleInstruction +
          " Preserve all key facts, decisions, tool results, and action items.",
      },
      {
        role: "user",
        content: `Summarize this conversation:\n\n${conversation}`,
      },
    ],
    max_tokens: maxTokens,
    temperature: 0,
  });

  return response.choices[0].message.content?.trim() ?? "";
}

/**
 * ConversationSummarizer manages rolling summaries of long conversations.
 */
export class ConversationSummarizer {
  private summaries: string[] = [];

  constructor(
    private client: OpenAI,
    private config: SummarizationConfig = {}
  ) {}

  /** Summarize a window of messages and store the summary. */
  async summarize(
    messages: OpenAI.Chat.ChatCompletionMessageParam[]
  ): Promise<string> {
    const summary = await summarizeMessages(messages, this.client, this.config);
    this.summaries.push(summary);
    return summary;
  }

  /** Return a combined summary of all stored summaries. */
  async getCombinedSummary(): Promise<string> {
    if (this.summaries.length === 0) return "";
    if (this.summaries.length === 1) return this.summaries[0];
    return this.summaries.join("\n\n---\n\n");
  }

  /** Return a system prompt fragment containing the stored summaries. */
  async buildContextMessage(): Promise<OpenAI.Chat.ChatCompletionSystemMessageParam> {
    const combined = await this.getCombinedSummary();
    return {
      role: "system",
      content: `Previous conversation summary:\n${combined}`,
    };
  }

  reset(): void {
    this.summaries = [];
  }
}
