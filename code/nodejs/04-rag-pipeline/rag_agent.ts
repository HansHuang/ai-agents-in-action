/**
 * RAG integrated as a tool in an agent loop.
 *
 * The agent decides when to search the knowledge base versus answering
 * directly from general knowledge.
 * See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
 */

import OpenAI from "openai";
import { RAGPipeline } from "./rag_pipeline.js";

const MAX_ITERATIONS = 10;

export interface RagAgentConfig {
  model?: string;
  systemPrompt?: string;
  maxIterations?: number;
}

export interface RagAgentResult {
  answer: string;
  iterations: number;
  ragCalled: boolean;
}

/**
 * Agent loop with RAG as a tool. The model chooses when to retrieve context.
 */
export async function runRagAgent(
  question: string,
  pipeline: RAGPipeline,
  client: OpenAI,
  config: RagAgentConfig = {}
): Promise<RagAgentResult> {
  const model = config.model ?? "gpt-4o-mini";
  const maxIter = config.maxIterations ?? MAX_ITERATIONS;
  const systemPrompt =
    config.systemPrompt ??
    "You are a helpful assistant. Use the search_knowledge_base tool when you need information from the knowledge base.";

  const tools: OpenAI.Chat.ChatCompletionTool[] = [
    {
      type: "function",
      function: {
        name: "search_knowledge_base",
        description: "Search the knowledge base for relevant information to answer the question.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string", description: "Search query" },
            top_k: { type: "integer", description: "Number of results", default: 5 },
          },
          required: ["query"],
        },
      },
    },
  ];

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: question },
  ];

  let iterations = 0;
  let ragCalled = false;

  for (let i = 0; i < maxIter; i++) {
    iterations++;
    const response = await client.chat.completions.create({ model, messages, tools });
    const choice = response.choices[0];

    if (choice.finish_reason === "stop" || !choice.message.tool_calls?.length) {
      return { answer: choice.message.content ?? "", iterations, ragCalled };
    }

    messages.push(choice.message);

    for (const toolCall of choice.message.tool_calls) {
      if (toolCall.function.name === "search_knowledge_base") {
        ragCalled = true;
        const args = JSON.parse(toolCall.function.arguments) as { query: string; top_k?: number };
        const ragResp = await pipeline.query(args.query, args.top_k ?? 5);
        const context = ragResp.retrievedChunks.map((r, i) => `[${i + 1}] ${r.text}`).join("\n\n");
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: context || "No relevant documents found.",
        });
      }
    }
  }

  return { answer: "Max iterations reached", iterations, ragCalled };
}
