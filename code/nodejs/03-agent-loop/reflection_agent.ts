/**
 * Reflection agent — agent that critiques and improves its own outputs.
 * See: docs/02-the-agent-loop/03-planning-strategies.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o";

export interface ReflectionResult {
  initialResponse: string;
  reflection: string;
  improvedResponse: string;
  iterations: number;
}

/**
 * Run a reflection loop: generate → critique → improve.
 */
export async function reflect(
  userQuery: string,
  client: OpenAI,
  maxIterations = 2
): Promise<ReflectionResult> {
  // Step 1: Initial response
  const initialResp = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "You are a helpful assistant. Answer the user's question thoroughly." },
      { role: "user", content: userQuery },
    ],
    temperature: 0.7,
  });
  const initialResponse = initialResp.choices[0].message.content?.trim() ?? "";

  let currentResponse = initialResponse;
  let reflection = "";

  for (let i = 0; i < maxIterations; i++) {
    // Step 2: Critique
    const critiqueResp = await client.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content:
            "You are a critical reviewer. Identify weaknesses, errors, or gaps in the following response. Be specific and constructive.",
        },
        {
          role: "user",
          content: `Original question: ${userQuery}\n\nResponse to critique:\n${currentResponse}`,
        },
      ],
      temperature: 0,
    });
    reflection = critiqueResp.choices[0].message.content?.trim() ?? "";

    // Step 3: Improve
    const improveResp = await client.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant. Improve the response based on the critique provided.",
        },
        { role: "user", content: userQuery },
        { role: "assistant", content: currentResponse },
        {
          role: "user",
          content: `Please improve your response based on this critique:\n${reflection}`,
        },
      ],
      temperature: 0.5,
    });
    currentResponse = improveResp.choices[0].message.content?.trim() ?? "";
  }

  return {
    initialResponse,
    reflection,
    improvedResponse: currentResponse,
    iterations: maxIterations,
  };
}

async function main(): Promise<void> {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const query = "Explain the trade-offs between synchronous and asynchronous programming.";

  console.log(`Query: ${query}\n`);
  const result = await reflect(query, client, 1);

  console.log("=== Initial Response ===");
  console.log(result.initialResponse);
  console.log("\n=== Reflection/Critique ===");
  console.log(result.reflection);
  console.log("\n=== Improved Response ===");
  console.log(result.improvedResponse);
}

main().catch(console.error);
