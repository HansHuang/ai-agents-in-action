/**
 * Branch manager — parallel conversation contexts.
 *
 * Allows an agent to explore hypothetical paths without polluting the main
 * conversation history. Each branch is an independent message list.
 * See: docs/03-memory-and-retrieval/01-short-term-memory.md
 */

import OpenAI from "openai";
import { ConversationSummarizer } from "./conversation_summarizer.js";

export interface BranchOptions {
  inheritHistory?: boolean;
  label?: string;
}

export interface Branch {
  id: string;
  label: string;
  messages: OpenAI.Chat.ChatCompletionMessageParam[];
  createdAt: number;
  parentId?: string;
}

/**
 * BranchManager manages multiple parallel conversation branches.
 */
export class BranchManager {
  private branches = new Map<string, Branch>();
  private counter = 0;

  constructor(
    private client: OpenAI,
    private mainMessages: OpenAI.Chat.ChatCompletionMessageParam[] = []
  ) {}

  /** Create a new branch, optionally inheriting current main history. */
  createBranch(options: BranchOptions = {}): Branch {
    const id = `branch-${++this.counter}`;
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = options.inheritHistory
      ? [...this.mainMessages]
      : [];

    const branch: Branch = {
      id,
      label: options.label ?? id,
      messages,
      createdAt: Date.now(),
    };
    this.branches.set(id, branch);
    return branch;
  }

  /** Get a branch by ID. */
  getBranch(id: string): Branch {
    const branch = this.branches.get(id);
    if (!branch) throw new Error(`Branch ${id} not found`);
    return branch;
  }

  /** Append a message to a branch. */
  addMessage(branchId: string, message: OpenAI.Chat.ChatCompletionMessageParam): void {
    this.getBranch(branchId).messages.push(message);
  }

  /** Summarize a branch and inject its summary into main history. */
  async mergeBranchSummary(
    branchId: string,
    summarizerConfig: ConstructorParameters<typeof ConversationSummarizer>[1] = {}
  ): Promise<string> {
    const branch = this.getBranch(branchId);
    const summarizer = new ConversationSummarizer(this.client, summarizerConfig);
    const summary = await summarizer.summarize(branch.messages);
    this.mainMessages.push({
      role: "system",
      content: `[Branch "${branch.label}" summary]: ${summary}`,
    });
    return summary;
  }

  /** Delete a branch. */
  deleteBranch(id: string): void {
    this.branches.delete(id);
  }

  /** List all branch IDs and labels. */
  list(): { id: string; label: string; messageCount: number }[] {
    return Array.from(this.branches.values()).map((b) => ({
      id: b.id,
      label: b.label,
      messageCount: b.messages.length,
    }));
  }

  get mainHistory(): OpenAI.Chat.ChatCompletionMessageParam[] {
    return this.mainMessages;
  }
}

// Demo
function main(): void {
  const manager = new BranchManager(
    new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "demo" }),
    [{ role: "user", content: "How should I invest $10,000?" }]
  );

  const branch = manager.createBranch({ label: "stocks-scenario", inheritHistory: true });
  manager.addMessage(branch.id, { role: "user", content: "What if I put it all in stocks?" });
  manager.addMessage(branch.id, { role: "assistant", content: "Stocks carry higher risk but higher returns..." });

  console.log("Branches:", manager.list());
  console.log("Main history length:", manager.mainHistory.length);
}

main();
