/**
 * ReAct agent orchestration loop.
 *
 * Implements the Reason → Act → Observe cycle.
 * See: docs/02-the-agent-loop/01-anatomy-of-an-agent.md
 */

import OpenAI from "openai";
import { ToolRegistry } from "./tool_registry.js";
import { createDefaultRegistry } from "./tools.js";

const MAX_ITERATIONS = 10;
const MODEL = "gpt-4o";

const SYSTEM_PROMPT = `You are an AI assistant with access to tools.

## Your Process
1. When the user asks a question, determine if you need a tool to answer it.
2. If yes, call the appropriate tool with the correct parameters.
3. Wait for the tool result, then determine if you need more tools or can answer.
4. Never guess tool results — always wait for the actual result.
5. If a tool fails, explain the failure and suggest alternatives.

## Tool Usage Rules
- Call only one tool at a time unless they are independent.
- If you don't have enough information, ask the user.
- Never make up parameters. If unsure, ask for clarification.`;

/**
 * Run the ReAct loop until a final answer is reached.
 */
export async function runAgent(
  userInput: string,
  options: {
    messages?: OpenAI.Chat.ChatCompletionMessageParam[];
    registry?: ToolRegistry;
    client?: OpenAI;
  } = {}
): Promise<string> {
  if (!userInput.trim()) throw new Error("userInput must not be empty");

  const client = options.client ?? new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const registry = options.registry ?? createDefaultRegistry();
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = options.messages ?? [
    { role: "system", content: SYSTEM_PROMPT },
  ];
  messages.push({ role: "user", content: userInput });

  const tools = registry.getOpenAISchemas();

  for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
    const response = await client.chat.completions.create({
      model: MODEL,
      messages,
      tools,
      tool_choice: "auto",
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      return msg.content?.trim() ?? "";
    }

    // Execute all tool calls in this turn
    for (const toolCall of msg.tool_calls) {
      const result = await registry.executeTool(toolCall);
      messages.push(result);
    }
  }

  return "Agent reached maximum iterations without a final answer.";
}

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const registry = createDefaultRegistry();

  const queries = [
    "What is the weather in Tokyo?",
    "What is 15% of 847 plus 42?",
    "What is today's date?",
  ];

  for (const query of queries) {
    console.log(`\nQ: ${query}`);
    const answer = await runAgent(query, { client, registry });
    console.log(`A: ${answer}`);
  }
}

main().catch(console.error);
