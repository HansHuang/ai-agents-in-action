/**
 * agent.ts — Customer support agent using the Vercel AI SDK agent() function.
 *
 * This is the most polished agent configuration in the repo. It combines
 * all six tools with a comprehensive system prompt and full observability.
 */

import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";
import {
  searchKnowledgeBase,
  lookupOrder,
  lookupCustomer,
  createTicket,
  escalateToHuman,
  checkReturnEligibility,
} from "./tools.js";

// ──── Environment validation ──────────────────────────────────────────────────

if (!process.env.OPENAI_API_KEY) {
  throw new Error(
    "OPENAI_API_KEY is required. Copy .env.example to .env and set your key."
  );
}

// ──── Pricing table (USD per 1M tokens, May 2026) ────────────────────────────

const PRICING = {
  "gpt-4o": { input: 2.5, output: 10.0 },
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
} as const;

type SupportedModel = keyof typeof PRICING;

/** Calculate cost in USD from token counts. */
function estimateCost(
  model: SupportedModel,
  inputTokens: number,
  outputTokens: number
): string {
  const prices = PRICING[model];
  const cost =
    (inputTokens / 1_000_000) * prices.input +
    (outputTokens / 1_000_000) * prices.output;
  return `$${cost.toFixed(6)}`;
}

// ──── System prompt ───────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `
You are a knowledgeable, empathetic customer support agent for Acme Corp.

═══ YOUR PROCESS ════════════════════════════════════════════════════════

Step 1 — UNDERSTAND
  • Read the customer's message carefully before taking any action.
  • Identify: What is the core issue? Is there an order number? An email?

Step 2 — LOOK UP (if needed)
  • If an order number is mentioned, call lookupOrder FIRST.
  • If the customer gives an email, call lookupCustomer to see all orders.
  • Never guess order details. Always look them up.

Step 3 — SEARCH THE KNOWLEDGE BASE
  • Before creating a ticket, search the knowledge base.
  • Most common issues (returns, tracking, billing) have self-service answers.
  • If you find a relevant article, cite its title in your response.

Step 4 — CHECK ELIGIBILITY (for returns/refunds)
  • If the customer wants to return something, call checkReturnEligibility first.
  • Set accurate expectations: tell them how many days they have remaining.

Step 5 — RESOLVE OR ESCALATE
  • If the knowledge base answers the question → answer it, cite the article.
  • If the issue needs human intervention → escalateToHuman (very upset, fraud, VIP).
  • If no self-service answer exists → createTicket (last resort).

═══ TONE GUIDELINES ══════════════════════════════════════════════════════

• Empathetic first: Acknowledge frustration BEFORE trying to solve the problem.
  ✓ "I'm sorry to hear your package hasn't arrived — let me look into this right away."
  ✗ "Please provide your order number." (cold and transactional)

• Be specific: Use the actual data from tools. Don't speak in generalities.
  ✓ "Your order ORD-12345 was delivered on May 1st via UPS."
  ✗ "Your order should arrive soon."

• Be concise: Customers are frustrated. Get to the point quickly.

═══ RULES ════════════════════════════════════════════════════════════════

• NEVER make up order details, tracking numbers, or delivery dates.
• NEVER promise a specific agent name when escalating.
• ALWAYS tell the customer what to expect next (ticket ID, wait time, etc.).
• If you're unsure about something, say so and create a ticket.
• For fraud or safety concerns, escalate immediately — do not delay.
`.trim();

// ──── Agent configuration ─────────────────────────────────────────────────────

/** All tools available to the customer support agent. */
export const SUPPORT_TOOLS = {
  searchKnowledgeBase,
  lookupOrder,
  lookupCustomer,
  createTicket,
  escalateToHuman,
  checkReturnEligibility,
} as const;

// ──── Exported run function ───────────────────────────────────────────────────

export interface AgentMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface AgentRunOptions {
  messages: AgentMessage[];
  /** Override the model. Defaults to gpt-4o. */
  model?: SupportedModel;
  /** Maximum number of agent loop iterations. Defaults to 10. */
  maxSteps?: number;
}

export interface AgentRunResult {
  text: string;
  totalSteps: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  estimatedCost: string;
}

/**
 * Run the customer support agent with full streaming and observability.
 *
 * Uses streamText with maxSteps to implement the agent loop. Each step
 * is logged with token counts and tool call names.
 *
 * @example
 * const result = await runSupportAgent({
 *   messages: [{ role: 'user', content: 'Where is my order ORD-12345?' }],
 * });
 * console.log(result.text);
 */
export async function runSupportAgent(
  options: AgentRunOptions
): Promise<AgentRunResult> {
  const model = options.model ?? "gpt-4o";
  const maxSteps = options.maxSteps ?? 10;

  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalSteps = 0;

  const result = streamText({
    model: openai(model),
    system: SYSTEM_PROMPT,
    messages: options.messages,
    tools: SUPPORT_TOOLS,
    maxSteps,

    onStepFinish: (step) => {
      totalSteps += 1;
      const inputTokens = step.usage?.promptTokens ?? 0;
      const outputTokens = step.usage?.completionTokens ?? 0;
      totalInputTokens += inputTokens;
      totalOutputTokens += outputTokens;

      console.log(`[agent] step ${totalSteps}`, {
        finishReason: step.finishReason,
        toolCalls: step.toolCalls?.map((tc) => tc.toolName) ?? [],
        text: step.text?.substring(0, 120) ?? "",
        tokens: { input: inputTokens, output: outputTokens },
      });
    },
  });

  // Collect the complete streamed text
  let text = "";
  for await (const chunk of result.textStream) {
    text += chunk;
  }

  const finalUsage = await result.usage;
  if (finalUsage) {
    // Use the final accumulated usage if available
    totalInputTokens = finalUsage.promptTokens;
    totalOutputTokens = finalUsage.completionTokens;
  }

  console.log(`[agent] finished`, {
    totalSteps,
    totalInputTokens,
    totalOutputTokens,
    estimatedCost: estimateCost(model, totalInputTokens, totalOutputTokens),
  });

  return {
    text,
    totalSteps,
    totalInputTokens,
    totalOutputTokens,
    estimatedCost: estimateCost(model, totalInputTokens, totalOutputTokens),
  };
}

// ──── CLI demo ────────────────────────────────────────────────────────────────

async function main() {
  const scenarios: { label: string; message: string }[] = [
    {
      label: "Order lookup",
      message: "Hi, where is my order ORD-12345? It says delivered but I never got it.",
    },
    {
      label: "Return eligibility",
      message: "I want to return the headphones from order ORD-12345. Can I still do that?",
    },
    {
      label: "Knowledge base",
      message: "What's your return policy?",
    },
  ];

  for (const scenario of scenarios) {
    console.log(`\n${"─".repeat(60)}`);
    console.log(`Scenario: ${scenario.label}`);
    console.log(`User: ${scenario.message}`);
    console.log("─".repeat(60));

    const result = await runSupportAgent({
      messages: [{ role: "user", content: scenario.message }],
    });

    console.log(`\nAgent: ${result.text}`);
    console.log(`\n[stats] steps=${result.totalSteps} tokens=${result.totalInputTokens + result.totalOutputTokens} cost=${result.estimatedCost}`);
  }
}

// Run demo when executed directly
const isMain =
  process.argv[1] !== undefined &&
  (process.argv[1].endsWith("agent.ts") || process.argv[1].endsWith("agent.js"));

if (isMain) {
  main().catch((err) => {
    console.error("Fatal:", err);
    process.exit(1);
  });
}
