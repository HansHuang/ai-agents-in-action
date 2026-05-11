/**
 * Specialized request handlers for each routing intent.
 *
 * Each handler accepts (userInput, conversationHistory, config) and returns
 * a HandlerResponse. Handlers are intentionally independent.
 * See: docs/07-harness-engineering/03-routing-and-intent-classification.md
 */

import OpenAI from "openai";
import type { HandlerConfig, HandlerResponse } from "./hybrid_router.js";

const MODEL_COSTS: Record<string, [number, number]> = {
  "gpt-4o-mini": [0.00015, 0.00060],
  "gpt-4o":      [0.00250, 0.01000],
};

function estimateCost(model: string, inputTokens: number, outputTokens: number): number {
  const [inp, out] = MODEL_COSTS[model] ?? [0.002, 0.002];
  return (inputTokens * inp + outputTokens * out) / 1000;
}

type Message = { role: string; content: string };

// ---------------------------------------------------------------------------
// Handler factory
// ---------------------------------------------------------------------------

/** Create a handler that calls the OpenAI API with the given system prompt. */
function makeHandler(systemPrompt: string, maxHistory = 6) {
  return async (
    userInput: string,
    history: Message[],
    config: HandlerConfig,
    client: OpenAI
  ): Promise<HandlerResponse> => {
    const start = Date.now();
    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: systemPrompt },
      ...(history.slice(-maxHistory) as OpenAI.ChatCompletionMessageParam[]),
      { role: "user", content: userInput },
    ];

    const resp = await client.chat.completions.create({
      model: config.model,
      messages,
      max_tokens: config.maxTokens,
      temperature: config.temperature,
    });

    const content = resp.choices[0].message.content ?? "";
    const usage = resp.usage;
    const inputTokens = usage?.prompt_tokens ?? 0;
    const outputTokens = usage?.completion_tokens ?? 0;

    return {
      content,
      handlerUsed: "llm",
      tokensUsed: inputTokens + outputTokens,
      cost: estimateCost(config.model, inputTokens, outputTokens),
      metadata: { durationMs: Date.now() - start, model: config.model },
    };
  };
}

// ---------------------------------------------------------------------------
// Specific handlers
// ---------------------------------------------------------------------------

/** Handle casual conversation — fast, cheap, no tools. */
export async function simpleChatHandler(
  userInput: string,
  history: Message[],
  config: HandlerConfig,
  client: OpenAI
): Promise<HandlerResponse> {
  return makeHandler("You are a friendly, concise assistant. Keep responses brief and warm.")(
    userInput, history, config, client
  );
}

/** Handle knowledge/factual questions with a helpful researcher persona. */
export async function knowledgeHandler(
  userInput: string,
  history: Message[],
  config: HandlerConfig,
  client: OpenAI
): Promise<HandlerResponse> {
  return makeHandler(
    "You are a knowledgeable assistant. Provide accurate, well-structured answers. " +
    "Cite reasoning clearly. If uncertain, say so.",
    10
  )(userInput, history, config, client);
}

/** Handle code-related questions with a senior engineer persona. */
export async function codeHandler(
  userInput: string,
  history: Message[],
  config: HandlerConfig,
  client: OpenAI
): Promise<HandlerResponse> {
  return makeHandler(
    "You are a senior software engineer. Write clean, idiomatic code with brief explanations. " +
    "Prefer modern TypeScript/Python. Include error handling. Show complete, runnable examples.",
    8
  )(userInput, history, config, client);
}

/** Handle analytical tasks requiring structured reasoning. */
export async function analysisHandler(
  userInput: string,
  history: Message[],
  config: HandlerConfig,
  client: OpenAI
): Promise<HandlerResponse> {
  return makeHandler(
    "You are an expert analyst. Break down problems systematically. " +
    "Use numbered steps, weigh trade-offs, and provide a clear recommendation.",
    6
  )(userInput, history, config, client);
}

/** Handle creative writing tasks. */
export async function creativeHandler(
  userInput: string,
  history: Message[],
  config: HandlerConfig,
  client: OpenAI
): Promise<HandlerResponse> {
  const start = Date.now();
  const resp = await client.chat.completions.create({
    model: config.model,
    messages: [
      { role: "system", content: "You are a creative writer. Be imaginative and engaging." },
      { role: "user", content: userInput },
    ],
    max_tokens: config.maxTokens,
    temperature: 0.9, // Higher temperature for creativity
  });

  const usage = resp.usage;
  return {
    content: resp.choices[0].message.content ?? "",
    handlerUsed: "creative",
    tokensUsed: (usage?.prompt_tokens ?? 0) + (usage?.completion_tokens ?? 0),
    cost: estimateCost(config.model, usage?.prompt_tokens ?? 0, usage?.completion_tokens ?? 0),
    metadata: { durationMs: Date.now() - start },
  };
}

/** Escalation handler — used when all other handlers fail. */
export async function escalationHandler(
  userInput: string,
  _history: Message[],
  _config: HandlerConfig,
  _client: OpenAI
): Promise<HandlerResponse> {
  return {
    content: "I'm having trouble processing your request right now. Please try again shortly or contact support.",
    handlerUsed: "escalation",
    tokensUsed: 0,
    cost: 0,
    metadata: { escalated: true, originalQuery: userInput },
  };
}
