/**
 * Chain-of-thought prompting demonstration.
 *
 * Sends the same multi-step math problem with and without CoT instructions.
 * See: docs/01-foundations/02-prompt-engineering.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o";

const PROBLEM =
  "A store sells apples for $1.20 each and bananas for $0.40 each. " +
  "Alice buys 5 apples and 8 bananas. She pays with a $20 bill. " +
  "How much change does she receive?";

const WITHOUT_COT: OpenAI.Chat.ChatCompletionMessageParam[] = [
  { role: "system", content: "You are a math assistant. Answer concisely." },
  { role: "user", content: PROBLEM },
];

const WITH_COT: OpenAI.Chat.ChatCompletionMessageParam[] = [
  { role: "system", content: "You are a math assistant." },
  {
    role: "user",
    content: PROBLEM + "\n\nThink step by step before giving the final answer.",
  },
];

async function call(
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
  client: OpenAI
): Promise<string> {
  const response = await client.chat.completions.create({
    model: MODEL,
    messages,
    temperature: 0,
  });
  return response.choices[0].message.content?.trim() ?? "";
}

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  console.log("Problem:", PROBLEM);
  console.log("=".repeat(60));

  console.log("\n[WITHOUT chain-of-thought]");
  const without = await call(WITHOUT_COT, client);
  console.log(without);

  console.log("\n[WITH chain-of-thought]");
  const withCot = await call(WITH_COT, client);
  console.log(withCot);

  console.log("\n" + "=".repeat(60));
  console.log("Observation: the CoT response shows every arithmetic step.");
  console.log("If the answer is wrong, you can see exactly which step failed.");
}

main().catch(console.error);
