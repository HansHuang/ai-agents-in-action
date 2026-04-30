/**
 * Prompt template example: variable substitution, token counting, and LLM call.
 *
 * Demonstrates the pattern described in:
 * docs/01-foundations/02-prompt-engineering.md — "Prompt Templates"
 *
 * Functions are exported so they can be reused and tested independently.
 */

import OpenAI from "openai";
import { encodingForModel } from "js-tiktoken";

const MODEL = "gpt-4o";

const SYSTEM_PROMPT_TEMPLATE = `You are a technical summarizer.
Summarize the following article in 3 bullet points.
Focus on: {focusArea}`;

const USER_PROMPT_TEMPLATE = `Article:
{articleText}`;

/**
 * Return the filled system prompt for the given focus area.
 * @param {string} focusArea
 * @returns {string}
 */
export function buildSystemPrompt(focusArea) {
  if (!focusArea.trim()) throw new Error("focusArea must not be empty");
  return SYSTEM_PROMPT_TEMPLATE.replace("{focusArea}", focusArea);
}

/**
 * Return the filled user prompt for the given article text.
 * @param {string} articleText
 * @returns {string}
 */
export function buildUserPrompt(articleText) {
  if (!articleText.trim()) throw new Error("articleText must not be empty");
  return USER_PROMPT_TEMPLATE.replace("{articleText}", articleText);
}

/**
 * Return a messages array ready to send to chat.completions.create.
 * @param {string} focusArea
 * @param {string} articleText
 * @returns {Array<{role: string, content: string}>}
 */
export function buildMessages(focusArea, articleText) {
  return [
    { role: "system", content: buildSystemPrompt(focusArea) },
    { role: "user", content: buildUserPrompt(articleText) },
  ];
}

/**
 * Return the token cost of a messages array (includes API overhead).
 *
 * Accounts for the per-message overhead (3 tokens) and reply primer
 * (3 tokens) that the API adds automatically.
 *
 * @param {Array<{role: string, content: string}>} messages
 * @param {string} [model]
 * @returns {number}
 */
export function countTokens(messages, model = MODEL) {
  const enc = encodingForModel(model);
  const tokensPerMessage = 3;
  let total = 0;
  for (const message of messages) {
    total += tokensPerMessage;
    for (const value of Object.values(message)) {
      total += enc.encode(value).length;
    }
  }
  total += 3; // reply is primed with <|start|>assistant<|message|>
  return total;
}

export async function main() {
  const focusArea = "practical implementation details";
  const articleText =
    "AI agents are software systems that use large language models as their " +
    "reasoning engine. Unlike chatbots, agents can take actions: call APIs, " +
    "search the web, write code, and orchestrate other agents. The key " +
    "architectural pattern is the agent loop: perceive, think, act, observe. " +
    "Production agents require harness engineering — input validation, retry " +
    "logic, output guardrails, and human-in-the-loop checkpoints.";

  const messages = buildMessages(focusArea, articleText);
  const tokens = countTokens(messages);
  console.log(`Token count before sending: ${tokens}`);

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const response = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0.3,
  });

  console.log("\nResponse:");
  console.log(response.choices[0].message.content);
  const { prompt_tokens, completion_tokens } = response.usage;
  console.log(
    `\nActual tokens used — prompt: ${prompt_tokens}, completion: ${completion_tokens}`
  );
}
