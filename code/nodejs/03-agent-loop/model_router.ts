/**
 * Task-based model router.
 *
 * Routes LLM tasks to the most appropriate provider based on task type,
 * priority, and capability tags.
 * See: docs/05-the-tool-ecosystem/01-model-providers.md
 */

import OpenAI from "openai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TaskType = "chat" | "reasoning" | "classification" | "summarization" | "code";
export type Priority = "cost" | "latency" | "quality";

export interface RoutingTask {
  messages: OpenAI.Chat.ChatCompletionMessageParam[];
  taskType: TaskType;
  estimatedInputTokens: number;
  estimatedOutputTokens: number;
  priority: Priority;
  requiresTools?: boolean;
  requiresStructuredOutput?: boolean;
}

export interface RoutingRule {
  /** A capability tag that triggers this rule */
  capability: string;
  /** Provider name to force */
  provider: string;
}

export interface RouterConfig {
  rules?: RoutingRule[];
  maxCostPer1kTokens?: number;
  maxLatencyMs?: number;
}

export interface ProviderEntry {
  client: OpenAI;
  model: string;
  capabilities: string[];
  costPer1kInput: number;
  costPer1kOutput: number;
  typicalLatencyMs: number;
}

export interface UsageStats {
  totalCalls: number;
  successfulCalls: number;
  totalTokens: number;
  totalLatencyMs: number;
}

// ---------------------------------------------------------------------------
// ModelRouter
// ---------------------------------------------------------------------------

export class ModelRouter {
  private providers = new Map<string, ProviderEntry>();
  private stats = new Map<string, UsageStats>();

  constructor(private config: RouterConfig = {}) {}

  /** Register a provider with its capability tags. */
  registerProvider(
    name: string,
    entry: Omit<ProviderEntry, "capabilities"> & { capabilities: string[] }
  ): void {
    this.providers.set(name, entry);
    this.stats.set(name, { totalCalls: 0, successfulCalls: 0, totalTokens: 0, totalLatencyMs: 0 });
  }

  /** Select the best provider name for a task. */
  route(task: RoutingTask): string {
    // Check declarative rules first
    if (this.config.rules) {
      for (const rule of this.config.rules) {
        const entry = this.providers.get(rule.provider);
        if (entry?.capabilities.includes(rule.capability)) {
          return rule.provider;
        }
      }
    }

    const candidates = Array.from(this.providers.entries()).filter(([, e]) => {
      if (task.requiresTools && !e.capabilities.includes("function_calling")) return false;
      if (task.requiresStructuredOutput && !e.capabilities.includes("structured_output")) return false;
      if (this.config.maxCostPer1kTokens) {
        const cost = (e.costPer1kInput + e.costPer1kOutput) / 2;
        if (cost > this.config.maxCostPer1kTokens) return false;
      }
      if (this.config.maxLatencyMs && e.typicalLatencyMs > this.config.maxLatencyMs) return false;
      return true;
    });

    if (candidates.length === 0) {
      const fallback = Array.from(this.providers.keys())[0];
      if (!fallback) throw new Error("No providers registered");
      return fallback;
    }

    // Score candidates
    const scored = candidates.map(([name, e]) => {
      let score = 0;
      if (task.priority === "quality") {
        if (e.capabilities.includes("smart")) score += 2;
        if (e.capabilities.includes("cheap")) score -= 1;
      } else if (task.priority === "cost") {
        if (e.capabilities.includes("cheap")) score += 2;
        score += 1 / (e.costPer1kInput + e.costPer1kOutput + 0.001);
      } else if (task.priority === "latency") {
        if (e.capabilities.includes("fast")) score += 2;
        score += 10000 / (e.typicalLatencyMs + 1);
      }
      return { name, score };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored[0].name;
  }

  /** Get the OpenAI client and model for a provider name. */
  getProvider(name: string): ProviderEntry {
    const entry = this.providers.get(name);
    if (!entry) throw new Error(`Provider ${name} not registered`);
    return entry;
  }

  /** Record stats after a call. */
  recordCall(name: string, tokens: number, latencyMs: number): void {
    const s = this.stats.get(name);
    if (!s) return;
    s.totalCalls++;
    s.successfulCalls++;
    s.totalTokens += tokens;
    s.totalLatencyMs += latencyMs;
  }

  /** Print usage statistics. */
  printStats(): void {
    console.log("\nProvider Usage Statistics:");
    for (const [name, s] of this.stats) {
      console.log(`  ${name}: ${s.totalCalls} calls, ${s.totalTokens} tokens`);
    }
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY ?? "";
  const client = new OpenAI({ apiKey });
  const router = new ModelRouter();

  router.registerProvider("gpt-4o", {
    client, model: "gpt-4o",
    capabilities: ["smart", "function_calling", "structured_output"],
    costPer1kInput: 0.005, costPer1kOutput: 0.015, typicalLatencyMs: 2000,
  });
  router.registerProvider("gpt-4o-mini", {
    client, model: "gpt-4o-mini",
    capabilities: ["cheap", "fast", "function_calling"],
    costPer1kInput: 0.00015, costPer1kOutput: 0.0006, typicalLatencyMs: 800,
  });

  const tasks: RoutingTask[] = [
    { messages: [], taskType: "chat", estimatedInputTokens: 100, estimatedOutputTokens: 50, priority: "cost" },
    { messages: [], taskType: "reasoning", estimatedInputTokens: 500, estimatedOutputTokens: 500, priority: "quality" },
    { messages: [], taskType: "classification", estimatedInputTokens: 200, estimatedOutputTokens: 10, priority: "latency" },
  ];

  for (const task of tasks) {
    const name = router.route(task);
    const entry = router.getProvider(name);
    console.log(`Task [${task.taskType}/${task.priority}] → ${name} (${entry.model})`);
  }
}

main().catch(console.error);
