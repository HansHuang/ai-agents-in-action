/**
 * LangGraph.js ReAct agent — the same loop from Chapter 02, now as a StateGraph.
 *
 * TypeScript port of code/python/06-frameworks/langgraph_react_agent.py
 * Uses @langchain/langgraph with strict TypeScript types.
 *
 * Implements the identical ReAct (Reason → Act → Observe) loop using
 * LangGraph.js's StateGraph instead of an imperative for-loop.
 *
 * Also includes a comparison with the from-scratch TypeScript agent
 * from code/nodejs/03-agent-loop/index.ts.
 *
 * Run:
 *   npx tsx langgraph_react_agent.ts
 *
 * See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
 */

// ---------------------------------------------------------------------------
// Optional LangGraph.js imports (graceful fallback)
// ---------------------------------------------------------------------------

let StateGraph: any;
let END: any;
let MessagesAnnotation: any;
let ChatOpenAI: any;
let tool: any;
let ToolMessage: any;
let HumanMessage: any;
let LANGGRAPH_AVAILABLE = false;

try {
  const langgraph = await import("@langchain/langgraph");
  const openaiPkg = await import("@langchain/openai");
  const corePkg = await import("@langchain/core/messages");
  const toolsPkg = await import("@langchain/core/tools");

  StateGraph = langgraph.StateGraph;
  END = langgraph.END;
  MessagesAnnotation = langgraph.MessagesAnnotation;
  ChatOpenAI = openaiPkg.ChatOpenAI;
  tool = toolsPkg.tool;
  HumanMessage = corePkg.HumanMessage;
  ToolMessage = corePkg.ToolMessage;
  LANGGRAPH_AVAILABLE = true;
} catch {
  // LangGraph.js not installed
}

// ---------------------------------------------------------------------------
// Mock data (mirrors Python langgraph_react_agent.py)
// ---------------------------------------------------------------------------

const WEATHER_MOCK: Record<string, {
  temperature_c: number;
  condition: string;
  humidity_percent: number;
  wind_kph: number;
}> = {
  Tokyo:     { temperature_c: 18, condition: "partly cloudy", humidity_percent: 65, wind_kph: 14 },
  Shanghai:  { temperature_c: 22, condition: "light rain",    humidity_percent: 85, wind_kph: 15 },
  London:    { temperature_c: 14, condition: "overcast",      humidity_percent: 78, wind_kph: 20 },
  "New York":{ temperature_c: 18, condition: "partly cloudy", humidity_percent: 60, wind_kph: 25 },
  Paris:     { temperature_c: 16, condition: "sunny",         humidity_percent: 55, wind_kph: 12 },
};

const STOCK_MOCK: Record<string, {
  price_usd: number;
  change_percent: number;
  currency: string;
}> = {
  AAPL:  { price_usd: 192.35, change_percent:  1.2, currency: "USD" },
  GOOGL: { price_usd: 171.80, change_percent: -0.5, currency: "USD" },
  MSFT:  { price_usd: 415.10, change_percent:  0.8, currency: "USD" },
  TSLA:  { price_usd: 175.20, change_percent: -2.3, currency: "USD" },
  AMZN:  { price_usd: 188.40, change_percent:  0.3, currency: "USD" },
};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

export interface AgentResult {
  answer: string;
  iterations: number;
  toolCallsMade: string[];
  elapsedMs: number;
}

// ---------------------------------------------------------------------------
// LangGraphReActAgent
// ---------------------------------------------------------------------------

export class LangGraphReActAgent {
  /**
   * ReAct agent built with LangGraph.js's StateGraph.
   *
   * Functionally identical to the from-scratch agent in
   * code/nodejs/03-agent-loop/index.ts.
   *
   * @param model          OpenAI model name.
   * @param maxIterations  Hard cap on agent iterations.
   */

  private readonly maxIterations: number;
  private readonly toolList: any[];
  private readonly llmWithTools: any;
  private readonly graph: any;

  constructor(model = "gpt-4o", maxIterations = 10) {
    if (!LANGGRAPH_AVAILABLE) {
      throw new Error(
        "LangGraph.js is required. Install with:\n" +
        "  npm install @langchain/langgraph @langchain/openai @langchain/core"
      );
    }
    this.maxIterations = maxIterations;
    this.toolList = this.loadTools();
    const llm = new ChatOpenAI({ model });
    this.llmWithTools = llm.bindTools(this.toolList);
    this.graph = this.buildGraph();
  }

  // ------------------------------------------------------------------
  // Tool definitions
  // ------------------------------------------------------------------

  private loadTools(): any[] {
    const getWeatherTool = tool(
      async ({ city }: { city: string }) => {
        const cityKey = city.split(",")[0].trim();
        const data = WEATHER_MOCK[cityKey] ?? {
          temperature_c: 20, condition: "clear",
          humidity_percent: 55, wind_kph: 10,
        };
        return JSON.stringify({ city, ...data });
      },
      {
        name: "get_weather",
        description:
          "Get current weather for a city. City must include the city name, e.g. 'Tokyo'.",
        schema: {
          type: "object" as const,
          properties: { city: { type: "string" } },
          required: ["city"],
        },
      }
    );

    const getStockPriceTool = tool(
      async ({ ticker }: { ticker: string }) => {
        const t = ticker.toUpperCase();
        const data = STOCK_MOCK[t] ?? {
          price_usd: 100.0, change_percent: 0.0, currency: "USD",
        };
        return JSON.stringify({ ticker: t, market_status: "open", ...data });
      },
      {
        name: "get_stock_price",
        description:
          "Get current stock price and daily change for a ticker symbol, e.g. 'AAPL'.",
        schema: {
          type: "object" as const,
          properties: { ticker: { type: "string" } },
          required: ["ticker"],
        },
      }
    );

    const calculatorTool = tool(
      async ({ expression }: { expression: string }) => {
        try {
          // Safe numeric-only eval — no arbitrary code
          if (!/^[\d\s+\-*/().%]+$/.test(expression)) {
            return "Error: only arithmetic expressions are supported";
          }
          // eslint-disable-next-line no-eval
          const result = eval(expression);
          return String(result);
        } catch (e) {
          return `Error: ${e}`;
        }
      },
      {
        name: "calculator",
        description:
          "Evaluate a simple arithmetic expression, e.g. '2 + 2' or '192.35 * 1.1'.",
        schema: {
          type: "object" as const,
          properties: { expression: { type: "string" } },
          required: ["expression"],
        },
      }
    );

    return [getWeatherTool, getStockPriceTool, calculatorTool];
  }

  // ------------------------------------------------------------------
  // Graph construction
  // ------------------------------------------------------------------

  private buildGraph(): any {
    const toolMap = new Map(this.toolList.map((t: any) => [t.name, t]));
    const llmWithTools = this.llmWithTools;
    const maxIter = this.maxIterations;

    // State type — LangGraph.js uses MessagesAnnotation for message accumulation
    const graphState = {
      messages: MessagesAnnotation.spec,
      iterationCount: { value: (a: number, b: number) => a + b, default: () => 0 },
      toolCallsMade: { value: (a: string[], b: string[]) => [...a, ...b], default: () => [] },
    };

    // Agent node — LLM decides
    async function agentNode(state: any): Promise<any> {
      const response = await llmWithTools.invoke(state.messages);
      return {
        messages: [response],
        iterationCount: 1,
        toolCallsMade: [],
      };
    }

    // Tool node — execute tool calls
    async function toolNode(state: any): Promise<any> {
      const lastMessage = state.messages[state.messages.length - 1];
      const toolResults = [];
      const newCalls: string[] = [];

      for (const tc of (lastMessage.tool_calls ?? [])) {
        const { name, args, id } = tc;
        newCalls.push(name);
        const toolFn = toolMap.get(name);
        const output = toolFn
          ? await toolFn.invoke(args)
          : `Unknown tool: ${name}`;
        toolResults.push(new ToolMessage({ content: String(output), tool_call_id: id }));
      }

      return {
        messages: toolResults,
        iterationCount: 0,
        toolCallsMade: newCalls,
      };
    }

    // Routing function
    function shouldContinue(state: any): "tools" | "end" {
      if (state.iterationCount >= maxIter) return "end";
      const last = state.messages[state.messages.length - 1];
      if (last?.tool_calls?.length) return "tools";
      return "end";
    }

    const workflow = new StateGraph({ channels: graphState })
      .addNode("agent", agentNode)
      .addNode("tools", toolNode)
      .addEdge("__start__", "agent")
      .addConditionalEdges("agent", shouldContinue, {
        tools: "tools",
        end: END,
      })
      .addEdge("tools", "agent");

    return workflow.compile();
  }

  // ------------------------------------------------------------------
  // Public interface
  // ------------------------------------------------------------------

  /**
   * Run the agent synchronously and return a structured result.
   */
  async run(userInput: string): Promise<AgentResult> {
    const start = performance.now();

    const initialState = {
      messages: [new HumanMessage(userInput)],
      iterationCount: 0,
      toolCallsMade: [],
    };

    const finalState = await this.graph.invoke(initialState);
    const elapsed = performance.now() - start;

    const lastMsg = finalState.messages[finalState.messages.length - 1];
    const answer = lastMsg?.content ?? "";

    return {
      answer: typeof answer === "string" ? answer : JSON.stringify(answer),
      iterations: finalState.iterationCount,
      toolCallsMade: finalState.toolCallsMade,
      elapsedMs: elapsed,
    };
  }

  /**
   * Stream step-by-step execution.
   * Yields each graph step as { nodeName: state } objects.
   */
  async *stream(userInput: string): AsyncGenerator<Record<string, any>> {
    const initialState = {
      messages: [new HumanMessage(userInput)],
      iterationCount: 0,
      toolCallsMade: [],
    };
    for await (const step of await this.graph.stream(initialState)) {
      yield step;
    }
  }

  static visualize(): string {
    return `
LangGraph.js ReAct Agent
─────────────────────────

┌─────────┐
│  START  │
└────┬────┘
     ▼
┌─────────┐     has tool_calls     ┌─────────┐
│  agent  │──────────────────────▶ │  tools  │
└─────────┘                        └────┬────┘
     │                                  │
     │  no tool_calls / maxIterations   │ (always)
     ▼                                  │
┌─────────┐ ◀────────────────────────── ┘
│   END   │
└─────────┘

Nodes:
  agent — LLM (ChatOpenAI with bound tools)
  tools — executes tool calls, appends ToolMessage results

Edges:
  START → agent          (entry point)
  agent → tools          (conditional: tool_calls present)
  agent → END            (conditional: no tool_calls OR maxIterations)
  tools → agent          (always: loop back after execution)
`.trim();
  }
}

// ---------------------------------------------------------------------------
// Comparison with from-scratch TypeScript agent
// ---------------------------------------------------------------------------

export interface ComparisonResult {
  query: string;
  langGraphAnswer: string;
  scratchAnswer: string;
  langGraphElapsedMs: number;
  scratchElapsedMs: number;
  langGraphToolCalls: string[];
  scratchToolCalls: string[];
  tracesMatch: boolean;
}

/**
 * Compare LangGraph agent vs from-scratch agent on the same query.
 *
 * The from-scratch agent is imported from code/nodejs/03-agent-loop/index.ts.
 * Falls back gracefully if that module is not resolvable.
 */
export async function compareAgents(query: string): Promise<ComparisonResult> {
  // LangGraph
  const lgAgent = new LangGraphReActAgent();
  const t0 = performance.now();
  const lgResult = await lgAgent.run(query);
  const lgElapsed = performance.now() - t0;

  // From-scratch
  let scratchAnswer = "(from-scratch agent not available)";
  let scratchElapsed = 0;
  let scratchToolCalls: string[] = [];

  try {
    const scratchModule = await import("../03-agent-loop/index.js");
    const OpenAI = (await import("openai")).default;
    const client = new OpenAI();

    const messages: any[] = [
      { role: "system", content: scratchModule.SYSTEM_PROMPT },
    ];

    const t1 = performance.now();
    scratchAnswer = await scratchModule.runAgent(client, query, messages);
    scratchElapsed = performance.now() - t1;

    scratchToolCalls = messages
      .filter((m: any) => m.role === "assistant" && m.tool_calls)
      .flatMap((m: any) => m.tool_calls.map((tc: any) => tc.function.name));
  } catch {
    // Module not resolvable in this execution context
  }

  return {
    query,
    langGraphAnswer: lgResult.answer,
    scratchAnswer,
    langGraphElapsedMs: lgElapsed,
    scratchElapsedMs: scratchElapsed,
    langGraphToolCalls: lgResult.toolCallsMade,
    scratchToolCalls,
    tracesMatch:
      new Set(lgResult.toolCallsMade).size === new Set(scratchToolCalls).size &&
      lgResult.toolCallsMade.every((t) => scratchToolCalls.includes(t)),
  };
}

function printComparison(result: ComparisonResult): void {
  const W = 40;
  const sep = "─".repeat(W * 2 + 5);

  console.log(`\n${"AGENT COMPARISON".padStart(W + 8)}`);
  console.log(sep);
  console.log(`Query: ${result.query}`);
  console.log(sep);

  const row = (label: string, a: string | number, b: string | number) => {
    console.log(
      `${String(label).padEnd(28)}${String(a).padStart(W - 28)}${String(b).padStart(W - 10)}`
    );
  };

  row("Elapsed (ms)", result.langGraphElapsedMs.toFixed(0), result.scratchElapsedMs.toFixed(0));
  row(
    "Tool calls",
    result.langGraphToolCalls.join(", ") || "none",
    result.scratchToolCalls.join(", ") || "none"
  );
  row("Traces match?", result.tracesMatch ? "yes" : "no", "—");
  console.log(sep);
  console.log(`\nLangGraph:    ${result.langGraphAnswer.slice(0, 150)}`);
  console.log(`From-scratch: ${result.scratchAnswer.slice(0, 150)}`);
}

// ---------------------------------------------------------------------------
// Demo entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log(LangGraphReActAgent.visualize());

  if (!LANGGRAPH_AVAILABLE) {
    console.log(
      "LangGraph.js is not installed. Install with:\n" +
      "  npm install @langchain/langgraph @langchain/openai @langchain/core"
    );
    return;
  }

  const query = "What's the weather in Tokyo and should I invest in AAPL?";
  console.log(`\nDemo query: ${JSON.stringify(query)}\n`);

  // Stream step-by-step
  const agent = new LangGraphReActAgent();
  console.log("Streaming execution steps:");
  for await (const step of agent.stream(query)) {
    for (const [node, state] of Object.entries(step)) {
      const msgs: any[] = (state as any).messages ?? [];
      const last = msgs[msgs.length - 1];
      if (last?.tool_calls?.length) {
        const calls = last.tool_calls.map((tc: any) => tc.name);
        console.log(`  [${node}] → tool calls: ${calls.join(", ")}`);
      } else if (last?.content) {
        const preview = String(last.content).slice(0, 80);
        console.log(`  [${node}] → ${preview}`);
      }
    }
  }
  console.log();

  // Side-by-side comparison
  const comparison = await compareAgents(query);
  printComparison(comparison);
}

// Run when executed directly
if (import.meta.url === new URL(process.argv[1], "file://").href) {
  main().catch(console.error);
}
