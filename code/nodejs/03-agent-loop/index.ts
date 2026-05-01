/**
 * ReAct agent: Reason → Act → Observe loop in TypeScript.
 *
 * Structurally identical to code/python/03-agent-loop/agent.py.
 * Run:  npx tsx index.ts
 *
 * See docs/02-the-agent-loop/01-anatomy-of-an-agent.md
 */

import OpenAI from "openai";
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionMessageToolCall,
} from "openai/resources/chat/completions";
import { dispatchTool, type ToolRegistry } from "./tool_dispatcher.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const MAX_ITERATIONS = 10;

export const SYSTEM_PROMPT = `You are an AI assistant with access to tools.

## Your Process
1. When the user asks a question, determine if you need a tool to answer it.
2. If yes, call the appropriate tool with the correct parameters.
3. Wait for the tool result, then determine if you need more tools or can answer.
4. Never guess tool results. Always wait for the actual result.
5. If a tool fails, explain the failure to the user and suggest alternatives.

## Tool Usage Rules
- Call only one tool at a time unless they are independent.
- If you don't have enough information to call a tool, ask the user.
- Never make up parameters. If unsure, ask for clarification.

## Answer Format
- Use the tool results to answer the user's question directly.
- Cite specific data from tool results.
- If multiple tools were used, synthesize the information.`;

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

interface WeatherResult {
  city: string;
  temperature_c: number;
  condition: string;
  humidity_percent: number;
  wind_kph: number;
}

const _weatherMock: Record<string, Omit<WeatherResult, "city">> = {
  Shanghai:    { temperature_c: 22, condition: "light rain",   humidity_percent: 85, wind_kph: 15 },
  London:   { temperature_c: 14, condition: "overcast",     humidity_percent: 78, wind_kph: 20 },
  "New York": { temperature_c: 18, condition: "partly cloudy", humidity_percent: 60, wind_kph: 25 },
  Paris:    { temperature_c: 16, condition: "sunny",        humidity_percent: 55, wind_kph: 12 },
  Sydney:   { temperature_c: 28, condition: "clear",        humidity_percent: 45, wind_kph: 18 },
};

export function getWeather(city: string): WeatherResult {
  const cityKey = city.split(",")[0].trim();
  const data = _weatherMock[cityKey] ?? {
    temperature_c: 20,
    condition: "clear",
    humidity_percent: 55,
    wind_kph: 10,
  };
  return { city, ...data };
}

interface StockResult {
  ticker: string;
  price_usd: number;
  change_percent: number;
  currency: string;
  market_status: string;
}

const _stockMock: Record<string, Omit<StockResult, "ticker" | "market_status">> = {
  AAPL:  { price_usd: 192.35, change_percent:  1.2, currency: "USD" },
  GOOGL: { price_usd: 171.80, change_percent: -0.5, currency: "USD" },
  MSFT:  { price_usd: 415.10, change_percent:  0.8, currency: "USD" },
  TSLA:  { price_usd: 175.20, change_percent: -2.3, currency: "USD" },
  AMZN:  { price_usd: 188.40, change_percent:  0.3, currency: "USD" },
};

export function getStockPrice(ticker: string): StockResult {
  const tickerUpper = ticker.toUpperCase();
  const data = _stockMock[tickerUpper] ?? {
    price_usd: 100.0,
    change_percent: 0.0,
    currency: "USD",
  };
  return { ticker: tickerUpper, market_status: "open", ...data };
}

// ---------------------------------------------------------------------------
// Tool definitions (OpenAI function-calling format)
// ---------------------------------------------------------------------------

export const TOOLS: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description:
        "Get current weather conditions for a city. " +
        "Use this when the user asks about weather, temperature, rain, " +
        "humidity, wind, or whether to bring an umbrella or coat. " +
        "Always call this tool rather than guessing — weather is dynamic.",
      parameters: {
        type: "object",
        properties: {
          city: {
            type: "string",
            description:
              "City name with optional ISO country code, " +
              "e.g. 'Shanghai, CN', 'London, UK', 'New York, US'. " +
              "Include the country code when the city name is ambiguous.",
          },
        },
        required: ["city"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_stock_price",
      description:
        "Get the current stock price and daily percentage change for a publicly " +
        "traded company. Use this when the user asks about stock price, share " +
        "value, investment potential, or financial performance of a company. " +
        "Always call this tool rather than using stale training data.",
      parameters: {
        type: "object",
        properties: {
          ticker: {
            type: "string",
            description:
              "Stock ticker symbol in uppercase, " +
              "e.g. 'AAPL' for Apple, 'GOOGL' for Google, " +
              "'MSFT' for Microsoft, 'TSLA' for Tesla.",
          },
        },
        required: ["ticker"],
        additionalProperties: false,
      },
    },
  },
];

// ---------------------------------------------------------------------------
// Default registry
// ---------------------------------------------------------------------------

export const DEFAULT_REGISTRY: ToolRegistry = {
  get_weather: (args) => getWeather(args["city"] as string),
  get_stock_price: (args) => getStockPrice(args["ticker"] as string),
};

// ---------------------------------------------------------------------------
// runAgent
// ---------------------------------------------------------------------------

/**
 * Run the ReAct loop until a final answer is reached or MAX_ITERATIONS is hit.
 *
 * @param userInput - The user's question. Must be non-empty.
 * @param messages  - Message history. A fresh history is created when undefined.
 *                    Extended in-place on each iteration.
 * @param tools     - Tool definitions. Defaults to TOOLS.
 * @param registry  - Tool execution registry. Defaults to DEFAULT_REGISTRY.
 * @returns The agent's final answer string.
 */
export async function runAgent(
  userInput: string,
  messages: ChatCompletionMessageParam[] = [{ role: "system", content: SYSTEM_PROMPT }],
  tools: ChatCompletionTool[] = TOOLS,
  registry: ToolRegistry = DEFAULT_REGISTRY
): Promise<string> {
  if (!userInput.trim()) {
    throw new Error("userInput must not be empty");
  }

  const client = new OpenAI({ apiKey: process.env["OPENAI_API_KEY"] });
  messages.push({ role: "user", content: userInput });

  for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages,
      tools,
      tool_choice: "auto",
    });

    const msg = response.choices[0].message;

    // Always append the assistant turn BEFORE processing tool calls.
    const assistantMsg: ChatCompletionMessageParam = {
      role: "assistant",
      content: msg.content,
    };
    if (msg.tool_calls && msg.tool_calls.length > 0) {
      (assistantMsg as ChatCompletionMessageParam & { tool_calls: ChatCompletionMessageToolCall[] }).tool_calls =
        msg.tool_calls;
    }
    messages.push(assistantMsg);

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      return msg.content ?? "";
    }

    // Execute every tool call and append results.
    for (const toolCall of msg.tool_calls) {
      const toolMsg = dispatchTool(toolCall, registry);
      messages.push(toolMsg);
    }
  }

  return (
    "I was unable to complete your request within the allowed number of " +
    "steps. Please try rephrasing your question or breaking it into smaller parts."
  );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const queries = [
    "What's the weather in Shanghai?",
    "Should I invest in Apple stock right now?",
  ];

  for (const query of queries) {
    const messages: ChatCompletionMessageParam[] = [
      { role: "system", content: SYSTEM_PROMPT },
    ];

    console.log(`\n${"=".repeat(60)}`);
    console.log(`Query: ${query}`);
    console.log("=".repeat(60));

    const answer = await runAgent(query, messages, TOOLS, DEFAULT_REGISTRY);

    console.log(`\nFinal Answer:\n${answer}`);
    console.log("\n--- Full Conversation History ---");

    for (const msg of messages) {
      const role = msg.role.toUpperCase();
      if ("tool_calls" in msg && msg.tool_calls) {
        for (const tc of msg.tool_calls as ChatCompletionMessageToolCall[]) {
          console.log(`  [${role}] → tool call: ${tc.function.name}(${tc.function.arguments})`);
        }
      } else if (msg.role === "tool") {
        const content = JSON.parse((msg as { content: string }).content ?? "{}");
        console.log(`  [${role}] ← result:`, content);
      } else {
        const content = (msg as { content?: string }).content ?? "";
        const preview = content.slice(0, 80).replace(/\n/g, " ") + (content.length > 80 ? "…" : "");
        console.log(`  [${role}] ${preview}`);
      }
    }
  }
}

main().catch(console.error);
