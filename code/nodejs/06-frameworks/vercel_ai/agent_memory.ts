/**
 * agent_memory.ts — Persistent memory for a Vercel AI SDK agent.
 *
 * Three memory tiers:
 *   1. Short-term  — Managed by useChat / streamText (message array).
 *                    This file adds token counting + automatic truncation.
 *   2. Long-term   — Per-session conversation summaries (key-value store).
 *   3. User memory — Cross-session facts extracted from conversations.
 *
 * The MemoryManager class wires all three tiers together and produces
 * an enriched system prompt that the agent reads at the start of each turn.
 *
 * Usage:
 *   npx tsx agent_memory.ts
 */

import { generateText, streamText, CoreMessage } from "ai";
import { openai } from "@ai-sdk/openai";
import { searchKnowledgeBase, lookupOrder } from "./tools.js";

// ──── Types ───────────────────────────────────────────────────────────────────

interface ConversationRecord {
  sessionId: string;
  messages: CoreMessage[];
  summary?: string;
  createdAt: string;
  updatedAt: string;
}

interface UserProfile {
  userId: string;
  facts: Record<string, string>; // e.g. { name: "Alice", city: "Tokyo" }
  updatedAt: string;
}

interface MemoryStats {
  messageCount: number;
  estimatedTokens: number;
  truncated: boolean;
  truncatedMessages: number;
}

// ──── Mock in-memory store (replace with Redis / DB in production) ────────────

const conversationStore = new Map<string, ConversationRecord>();
const userStore = new Map<string, UserProfile>();

// ──── Token estimation ────────────────────────────────────────────────────────

/**
 * Rough token estimate: ~4 chars per token for English.
 * Use tiktoken for precise counts in production.
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function estimateMessageTokens(messages: CoreMessage[]): number {
  return messages.reduce((sum, m) => {
    const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    return sum + estimateTokens(content) + 4; // 4 overhead per message
  }, 0);
}

// ──── MemoryManager ───────────────────────────────────────────────────────────

/**
 * Manages short-term, long-term, and user memory for a Vercel AI SDK agent.
 */
export class MemoryManager {
  /** Maximum tokens to keep in the active message window. */
  private readonly contextLimitTokens: number;

  constructor({ contextLimitTokens = 6_000 }: { contextLimitTokens?: number } = {}) {
    this.contextLimitTokens = contextLimitTokens;
  }

  // ── Short-term memory: truncation ──────────────────────────────────────────

  /**
   * Trim the message history so it fits within the context window.
   *
   * Strategy:
   *   1. Always keep the system message (injected separately).
   *   2. Always keep the last user message.
   *   3. Drop the oldest messages from the middle until it fits.
   *
   * @returns The trimmed messages and stats about what was dropped.
   */
  trimMessages(messages: CoreMessage[]): {
    trimmed: CoreMessage[];
    stats: MemoryStats;
  } {
    const estimated = estimateMessageTokens(messages);

    if (estimated <= this.contextLimitTokens) {
      return {
        trimmed: messages,
        stats: {
          messageCount: messages.length,
          estimatedTokens: estimated,
          truncated: false,
          truncatedMessages: 0,
        },
      };
    }

    // Keep the last N messages that fit
    let kept: CoreMessage[] = [];
    let tokenCount = 0;
    for (let i = messages.length - 1; i >= 0; i--) {
      const content =
        typeof messages[i].content === "string"
          ? (messages[i].content as string)
          : JSON.stringify(messages[i].content);
      const tokens = estimateTokens(content) + 4;
      if (tokenCount + tokens > this.contextLimitTokens) break;
      kept.unshift(messages[i]);
      tokenCount += tokens;
    }

    const truncatedCount = messages.length - kept.length;

    return {
      trimmed: kept,
      stats: {
        messageCount: kept.length,
        estimatedTokens: tokenCount,
        truncated: true,
        truncatedMessages: truncatedCount,
      },
    };
  }

  // ── Long-term memory: conversation summaries ───────────────────────────────

  /**
   * Save a conversation to the store.
   */
  async saveConversation(
    sessionId: string,
    messages: CoreMessage[]
  ): Promise<void> {
    const existing = conversationStore.get(sessionId);
    conversationStore.set(sessionId, {
      sessionId,
      messages,
      summary: existing?.summary,
      createdAt: existing?.createdAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });
  }

  /**
   * Retrieve a stored conversation by session ID.
   */
  async getConversation(sessionId: string): Promise<CoreMessage[]> {
    return conversationStore.get(sessionId)?.messages ?? [];
  }

  /**
   * Summarize the current conversation and archive it.
   * The summary is stored and injected into future conversations.
   */
  async summarizeAndArchive(sessionId: string): Promise<string> {
    const record = conversationStore.get(sessionId);
    if (!record || record.messages.length === 0) {
      return "";
    }

    const transcript = record.messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => {
        const content =
          typeof m.content === "string" ? m.content : JSON.stringify(m.content);
        return `${m.role.toUpperCase()}: ${content}`;
      })
      .join("\n");

    const { text: summary } = await generateText({
      model: openai("gpt-4o-mini"),
      system:
        "You are a summarization assistant. Summarize the following customer support conversation in 2–3 sentences. " +
        "Focus on: what was the issue, what was resolved, and what (if anything) remains open.",
      prompt: transcript,
      maxTokens: 200,
    });

    record.summary = summary;
    record.updatedAt = new Date().toISOString();
    conversationStore.set(sessionId, record);

    console.log(`[memory] archived session ${sessionId}: "${summary.substring(0, 80)}…"`);
    return summary;
  }

  // ── User memory: cross-session facts ──────────────────────────────────────

  /**
   * Retrieve all stored facts for a user.
   */
  async getUserFacts(userId: string): Promise<Record<string, string>> {
    return userStore.get(userId)?.facts ?? {};
  }

  /**
   * Store a single fact about a user.
   */
  async saveUserFact(
    userId: string,
    key: string,
    value: string
  ): Promise<void> {
    const existing = userStore.get(userId) ?? {
      userId,
      facts: {},
      updatedAt: "",
    };
    existing.facts[key] = value;
    existing.updatedAt = new Date().toISOString();
    userStore.set(userId, existing);
    console.log(`[memory] saved fact for ${userId}: ${key}="${value}"`);
  }

  /**
   * Extract facts from a conversation using an LLM and store them.
   *
   * Looks for facts like:
   *   - Name: "My name is Alice"
   *   - Location: "I live in Tokyo"
   *   - Preference: "I prefer email contact"
   *   - Order: "My order is ORD-12345"
   */
  async extractAndSaveFacts(
    userId: string,
    messages: CoreMessage[]
  ): Promise<Record<string, string>> {
    const userMessages = messages
      .filter((m) => m.role === "user")
      .map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content)))
      .join("\n");

    if (userMessages.trim().length < 20) return {};

    const { text } = await generateText({
      model: openai("gpt-4o-mini"),
      system:
        "Extract factual claims the user made about themselves. " +
        "Return ONLY a JSON object like: { \"name\": \"Alice\", \"city\": \"Tokyo\" }. " +
        "Use snake_case keys. Include only explicitly stated facts. " +
        "Return {} if no facts are stated.",
      prompt: userMessages,
      maxTokens: 200,
    });

    let facts: Record<string, string> = {};
    try {
      const cleaned = text.trim().replace(/^```json\n?/, "").replace(/\n?```$/, "");
      facts = JSON.parse(cleaned) as Record<string, string>;
    } catch {
      // LLM returned non-JSON — skip
      return {};
    }

    for (const [key, value] of Object.entries(facts)) {
      if (typeof value === "string" && value.length > 0) {
        await this.saveUserFact(userId, key, value);
      }
    }

    return facts;
  }

  // ── System prompt assembly ──────────────────────────────────────────────────

  /**
   * Build an enriched system prompt by injecting:
   *   - Recent conversation summary (if any)
   *   - Known user facts (if any)
   *
   * This is the core of the memory system: the agent "remembers" across sessions
   * because the memory is injected at the top of the system prompt.
   */
  async buildSystemPrompt(
    userId: string,
    sessionId: string,
    basePrompt: string
  ): Promise<string> {
    const [facts, previousRecord] = await Promise.all([
      this.getUserFacts(userId),
      Promise.resolve(conversationStore.get(sessionId)),
    ]);

    const sections: string[] = [basePrompt];

    // Inject known user facts
    const factEntries = Object.entries(facts);
    if (factEntries.length > 0) {
      const factLines = factEntries
        .map(([k, v]) => `  - ${k.replace(/_/g, " ")}: ${v}`)
        .join("\n");
      sections.push(
        `\n═══ WHAT YOU KNOW ABOUT THIS USER ═══\n${factLines}`
      );
    }

    // Inject previous session summary
    const prevSummaries = [...conversationStore.values()]
      .filter((r) => r.sessionId !== sessionId && r.summary)
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
      .slice(0, 3); // Last 3 sessions

    if (prevSummaries.length > 0) {
      const summaryLines = prevSummaries
        .map((r) => `  [${r.updatedAt.substring(0, 10)}] ${r.summary}`)
        .join("\n");
      sections.push(`\n═══ PREVIOUS CONVERSATIONS ═══\n${summaryLines}`);
    }

    return sections.join("\n");
  }
}

// ──── Demo ────────────────────────────────────────────────────────────────────

const BASE_SYSTEM_PROMPT =
  "You are a helpful customer support agent for Acme Corp. " +
  "Use the tools to look up orders and search the knowledge base. " +
  "Be empathetic and specific.";

async function runMemoryDemo(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    console.error("OPENAI_API_KEY is required.");
    process.exit(1);
  }

  const memory = new MemoryManager({ contextLimitTokens: 4_000 });
  const userId = "user-alice";
  const session1Id = "session-001";
  const session2Id = "session-002";

  // ── Session 1: Alice introduces herself and reports an issue ─────────────
  console.log("═══ SESSION 1 ═══════════════════════════════════════════════\n");

  const session1Messages: CoreMessage[] = [
    {
      role: "user",
      content:
        "Hi, my name is Alice and I'm having trouble with order ORD-12345. It says delivered but I never received it.",
    },
  ];

  const systemPrompt1 = await memory.buildSystemPrompt(
    userId,
    session1Id,
    BASE_SYSTEM_PROMPT
  );
  console.log("System prompt (session 1):\n" + systemPrompt1 + "\n");

  const result1 = streamText({
    model: openai("gpt-4o"),
    system: systemPrompt1,
    messages: session1Messages,
    tools: { searchKnowledgeBase, lookupOrder },
    maxSteps: 5,
  });

  let response1 = "";
  for await (const chunk of result1.textStream) {
    response1 += chunk;
  }

  console.log(`Agent: ${response1}\n`);

  // Save conversation and extract facts
  session1Messages.push({ role: "assistant", content: response1 });
  await memory.saveConversation(session1Id, session1Messages);
  const extractedFacts = await memory.extractAndSaveFacts(userId, session1Messages);
  console.log("Extracted facts:", extractedFacts);

  // Archive the summary
  const summary1 = await memory.summarizeAndArchive(session1Id);
  console.log(`Summary: ${summary1}\n`);

  // ── Session 2 (next day): Alice asks about her order again ───────────────
  console.log("═══ SESSION 2 (next day) ════════════════════════════════════\n");

  const session2Messages: CoreMessage[] = [
    {
      role: "user",
      content: "Hi, what's the status of my order?",
    },
  ];

  const systemPrompt2 = await memory.buildSystemPrompt(
    userId,
    session2Id,
    BASE_SYSTEM_PROMPT
  );
  console.log("System prompt (session 2, with memory injected):\n" + systemPrompt2 + "\n");

  const result2 = streamText({
    model: openai("gpt-4o"),
    system: systemPrompt2,
    messages: session2Messages,
    tools: { searchKnowledgeBase, lookupOrder },
    maxSteps: 5,
  });

  let response2 = "";
  for await (const chunk of result2.textStream) {
    response2 += chunk;
  }

  console.log(`Agent: ${response2}\n`);

  // ── Context window management demo ──────────────────────────────────────
  console.log("═══ CONTEXT WINDOW MANAGEMENT ═══════════════════════════════\n");

  const longHistory: CoreMessage[] = Array.from({ length: 40 }, (_, i) => ({
    role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
    content: `This is message ${i + 1} in a very long conversation. `.repeat(10),
  }));

  const { trimmed, stats } = memory.trimMessages(longHistory);
  console.log(`Original messages: ${longHistory.length}`);
  console.log(`After trimming: ${trimmed.length} messages`);
  console.log(`Estimated tokens: ${stats.estimatedTokens}`);
  console.log(`Truncated: ${stats.truncated} (dropped ${stats.truncatedMessages} messages)\n`);
}

runMemoryDemo().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
