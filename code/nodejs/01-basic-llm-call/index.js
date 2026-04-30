/**
 * Token counting for OpenAI-compatible models using js-tiktoken.
 *
 * Shows how to count tokens for a plain string and for a messages array
 * (the format used by chat.completions.create). The messages array count
 * mirrors what the API actually charges you for.
 */

import { encodingForModel } from "js-tiktoken";

const MODEL = "gpt-4o";

/**
 * Return the number of tokens in `text` for the given model.
 * @param {string} text
 * @param {string} model
 * @returns {number}
 */
function countTokens(text, model = MODEL) {
  const enc = encodingForModel(model);
  return enc.encode(text).length;
}

/**
 * Return the token cost of a messages array for chat completions.
 *
 * Accounts for the per-message overhead (3 tokens) and reply primer (3
 * tokens) that the API adds automatically.
 * See: https://platform.openai.com/docs/guides/chat/managing-tokens
 *
 * @param {Array<{role: string, content: string, name?: string}>} messages
 * @param {string} model
 * @returns {number}
 */
function countMessagesTokens(messages, model = MODEL) {
  const enc = encodingForModel(model);
  const tokensPerMessage = 3;
  const tokensPerName = 1;
  let total = 0;
  for (const message of messages) {
    total += tokensPerMessage;
    for (const [key, value] of Object.entries(message)) {
      total += enc.encode(value).length;
      if (key === "name") total += tokensPerName;
    }
  }
  total += 3; // reply is primed with <|start|>assistant<|message|>
  return total;
}

const text = "The quick brown fox jumps over the lazy dog.";
console.log(`Text  : ${JSON.stringify(text)}`);
console.log(`Tokens: ${countTokens(text)}`);

const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is the capital of France?" },
];
console.log(`\nMessages array token count: ${countMessagesTokens(messages)}`);
