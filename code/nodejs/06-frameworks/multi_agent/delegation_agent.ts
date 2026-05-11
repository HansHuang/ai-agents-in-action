/**
 * Delegation multi-agent system in TypeScript.
 *
 * The coordinator runs a ReAct loop where its "tools" are other agents.
 * Each specialist agent is a focused LLM with its own tools.
 *
 * Structurally equivalent to code/python/06-multi-agent/delegation_agent.py.
 *
 * Run:  npx tsx delegation_agent.ts
 * See:  docs/02-the-agent-loop/04-multi-agent-patterns.md
 */

import OpenAI from "openai";
import { z } from "zod";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "" });

const MAX_SPECIALIST_ITERATIONS = 5;
const MAX_DELEGATIONS_PER_AGENT = 3;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export const HandoffSchema = z.object({
  from_agent: z.string().min(1),
  to_agent: z.string().min(1),
  task: z.string().min(1),
  context: z.record(z.unknown()).default({}),
});

export const HandoffResultSchema = z.object({
  from_agent: z.string(),
  status: z.enum(["complete", "failed", "need_clarification"]),
  result: z.string(),
  error: z.string().nullable().default(null),
});

export type Handoff = z.infer<typeof HandoffSchema>;
export type HandoffResult = z.infer<typeof HandoffResultSchema>;

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const STOCK_MOCK: Record<string, Record<string, unknown>> = {
  AAPL:  { price_usd: 192.35, change_percent: 1.2,  weekly_change_percent: 3.1 },
  MSFT:  { price_usd: 415.10, change_percent: 0.8,  weekly_change_percent: 2.4 },
  GOOGL: { price_usd: 171.80, change_percent: -0.5, weekly_change_percent: 1.5 },
  TSLA:  { price_usd: 175.20, change_percent: -2.3, weekly_change_percent: -4.1 },
  AMZN:  { price_usd: 188.40, change_percent: 0.3,  weekly_change_percent: 0.8 },
};

const FINANCIALS_MOCK: Record<string, Record<string, unknown>> = {
  Apple:     { revenue_ttm_b: 391.0, net_income_b: 97.0, gross_margin_pct: 45.0, yoy_revenue_growth_pct: 5.1 },
  Microsoft: { revenue_ttm_b: 245.0, net_income_b: 88.0, gross_margin_pct: 70.0, yoy_revenue_growth_pct: 15.7 },
  Google:    { revenue_ttm_b: 350.0, net_income_b: 76.0, gross_margin_pct: 58.0, yoy_revenue_growth_pct: 14.3 },
};

const NEWS_MOCK: Record<string, string[]> = {
  Apple: [
    "Apple reports record services revenue of $25B in Q3.",
    "Apple Vision Pro sales expected to reach 1M units by year-end.",
    "iPhone 17 pre-orders exceed 15M in opening weekend.",
  ],
  Microsoft: [
    "Microsoft Copilot adds 5M enterprise users in Q3.",
    "Azure cloud revenue surpasses $30B quarterly run rate.",
    "Microsoft acquires AI research lab for $2.1B.",
  ],
};

// ---------------------------------------------------------------------------
// Mock tool implementations
// ---------------------------------------------------------------------------

function getStockPrice(ticker: string): Record<string, unknown> {
  const upper = ticker.toUpperCase();
  const data = STOCK_MOCK[upper] ?? { price_usd: 100.0, change_percent: 0.0 };
  return { ticker: upper, ...data };
}

function getCompanyFinancials(company: string): Record<string, unknown> {
  const key = Object.keys(FINANCIALS_MOCK).find((k) =>
    company.toLowerCase().includes(k.toLowerCase())
  );
  if (key) return { company: key, ...FINANCIALS_MOCK[key] };
  return { company, error: "No financials found" };
}

function webSearch(query: string): Record<string, unknown> {
  const key = Object.keys(NEWS_MOCK).find((k) =>
    query.toLowerCase().includes(k.toLowerCase())
  );
  if (key) return { query, results: NEWS_MOCK[key] };
  return { query, results: [`No mock results found for '${query}'`] };
}

function fetchArticle(url: string): Record<string, unknown> {
  return { url, title: "Mock article", content: "Mock article body for testing." };
}

const TOOL_DISPATCH: Record<string, (args: Record<string, unknown>) => unknown> = {
  get_stock_price:        (a) => getStockPrice(a["ticker"] as string),
  get_company_financials: (a) => getCompanyFinancials(a["company"] as string),
  web_search:             (a) => webSearch(a["query"] as string),
  fetch_article:          (a) => fetchArticle(a["url"] as string),
};

// ---------------------------------------------------------------------------
// OpenAI tool schemas
// ---------------------------------------------------------------------------

const FINANCE_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_stock_price",
      description: "Get the current stock price and weekly change for a ticker symbol.",
      parameters: {
        type: "object",
        properties: { ticker: { type: "string", description: "Uppercase ticker, e.g. 'AAPL'." } },
        required: ["ticker"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_company_financials",
      description: "Get TTM financials: revenue, net income, gross margin, YoY growth.",
      parameters: {
        type: "object",
        properties: { company: { type: "string", description: "Company name, e.g. 'Apple'." } },
        required: ["company"],
      },
    },
  },
];

const RESEARCH_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web for recent news and information about a topic.",
      parameters: {
        type: "object",
        properties: { query: { type: "string", description: "Search query string." } },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fetch_article",
      description: "Fetch the full text of a web article by URL.",
      parameters: {
        type: "object",
        properties: { url: { type: "string", description: "Full article URL." } },
        required: ["url"],
      },
    },
  },
];

// ---------------------------------------------------------------------------
// SpecialistAgent
// ---------------------------------------------------------------------------

export class SpecialistAgent {
  constructor(
    public readonly name: string,
    public readonly role: string,
    public readonly tools: OpenAI.Chat.ChatCompletionTool[],
    public readonly systemPrompt: string,
    public readonly taskGuidance: string = "",
  ) {}

  async run(handoff: Handoff): Promise<HandoffResult> {
    const userContent =
      Object.keys(handoff.context).length > 0
        ? `${handoff.task}\n\nContext provided:\n${JSON.stringify(handoff.context, null, 2)}`
        : handoff.task;

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      { role: "system", content: this.systemPrompt },
      { role: "user", content: userContent },
    ];

    try {
      for (let i = 0; i < MAX_SPECIALIST_ITERATIONS; i++) {
        const response = await client.chat.completions.create({
          model: "gpt-4o",
          messages,
          ...(this.tools.length > 0 ? { tools: this.tools, tool_choice: "auto" } : {}),
        });

        const msg = response.choices[0].message;
        messages.push(msg as OpenAI.Chat.ChatCompletionMessageParam);

        if (!msg.tool_calls || msg.tool_calls.length === 0) {
          return { from_agent: this.name, status: "complete", result: msg.content ?? "", error: null };
        }

        for (const tc of msg.tool_calls) {
          const args = JSON.parse(tc.function.arguments) as Record<string, unknown>;
          const fn = TOOL_DISPATCH[tc.function.name];
          const content = fn ? JSON.stringify(fn(args)) : JSON.stringify({ error: `Unknown tool: ${tc.function.name}` });
          messages.push({ role: "tool", tool_call_id: tc.id, content });
        }
      }

      const last = messages[messages.length - 1];
      return {
        from_agent: this.name,
        status: "complete",
        result: "content" in last ? String(last.content ?? "") : "",
        error: null,
      };
    } catch (err) {
      return { from_agent: this.name, status: "failed", result: "", error: String(err) };
    }
  }
}

// ---------------------------------------------------------------------------
// CoordinatorAgent
// ---------------------------------------------------------------------------

const COORDINATOR_SYSTEM = `\
You are a coordinator agent. Your ONLY job is to delegate tasks to specialist agents.
Do NOT answer questions directly. Do NOT perform analysis yourself.

Your specialists:
- delegate_to_finance_agent:   financial data, stock prices, earnings, revenue figures
- delegate_to_research_agent:  web search, news, recent events, background research
- delegate_to_writing_agent:   synthesis, summaries, reports, polished prose

Rules:
1. Always delegate immediately — never answer from your own knowledge.
2. If a request spans multiple domains, delegate to multiple specialists.
3. Give each specialist a complete, self-contained task with all the context they need.
4. After all specialists have responded, synthesize their results into a final answer.
5. If a specialist reports failure, acknowledge it in your final answer and continue.`;

export class CoordinatorAgent {
  private readonly delegationTools: OpenAI.Chat.ChatCompletionTool[];
  private delegationCounts: Record<string, number> = {};

  constructor(private readonly specialists: Record<string, SpecialistAgent>) {
    for (const name of Object.keys(specialists)) {
      this.delegationCounts[name] = 0;
    }
    this.delegationTools = this.buildDelegationTools();
  }

  async run(userInput: string): Promise<{ answer: string; delegations: object[] }> {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      { role: "system", content: COORDINATOR_SYSTEM },
      { role: "user", content: userInput },
    ];
    const delegations: object[] = [];

    for (let i = 0; i < 10; i++) {
      const response = await client.chat.completions.create({
        model: "gpt-4o",
        messages,
        tools: this.delegationTools,
        tool_choice: "auto",
      });

      const msg = response.choices[0].message;
      messages.push(msg as OpenAI.Chat.ChatCompletionMessageParam);

      if (!msg.tool_calls || msg.tool_calls.length === 0) {
        return { answer: msg.content ?? "", delegations };
      }

      for (const tc of msg.tool_calls) {
        const args = JSON.parse(tc.function.arguments) as { task: string; context?: Record<string, unknown> };
        const specialistName = tc.function.name.replace(/^delegate_to_/, "").replace(/_agent$/, "");
        const specialist = this.specialists[specialistName];

        if (!specialist) {
          messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify({ error: `Unknown specialist: ${specialistName}` }) });
          continue;
        }

        const count = (this.delegationCounts[specialistName] ?? 0) + 1;
        if (count > MAX_DELEGATIONS_PER_AGENT) {
          messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify({ error: "Delegation limit reached" }) });
          continue;
        }
        this.delegationCounts[specialistName] = count;

        const handoff: Handoff = {
          from_agent: "coordinator",
          to_agent: specialistName,
          task: args.task,
          context: args.context ?? {},
        };
        const result = await specialist.run(handoff);
        delegations.push({ specialist: specialistName, task: args.task, status: result.status });
        messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify({ result: result.result, status: result.status }) });
      }
    }

    return { answer: "Maximum iterations reached.", delegations };
  }

  private buildDelegationTools(): OpenAI.Chat.ChatCompletionTool[] {
    return Object.entries(this.specialists).map(([name, agent]) => ({
      type: "function" as const,
      function: {
        name: `delegate_to_${name}_agent`,
        description: `Delegate a task to the ${name} specialist. ${agent.taskGuidance}`,
        parameters: {
          type: "object",
          properties: {
            task: { type: "string", description: "Complete self-contained task description." },
            context: { type: "object", description: "Relevant context for the specialist.", additionalProperties: true },
          },
          required: ["task"],
        },
      },
    }));
  }
}

// ---------------------------------------------------------------------------
// Default specialists
// ---------------------------------------------------------------------------

export function buildSpecialists(): Record<string, SpecialistAgent> {
  return {
    finance: new SpecialistAgent(
      "finance",
      "Financial analysis expert",
      FINANCE_TOOLS,
      "You are a financial analyst. Use your tools to retrieve accurate financial data. Present numbers clearly with units.",
      "Use for stock prices, earnings reports, revenue figures, and financial comparisons.",
    ),
    research: new SpecialistAgent(
      "research",
      "Research and news expert",
      RESEARCH_TOOLS,
      "You are a research specialist. Search for relevant information and synthesise key findings with source references.",
      "Use for web searches, news, recent events, and background research.",
    ),
    writing: new SpecialistAgent(
      "writing",
      "Professional writer",
      [],
      "You are a professional writer. Turn data and findings into polished, well-structured prose suitable for business audiences.",
      "Use to create summaries, reports, and polished final deliverables.",
    ),
  };
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const specialists = buildSpecialists();
  const coordinator = new CoordinatorAgent(specialists);

  const task = "Research Apple's financial performance and write a summary";
  console.log(`\n${"=".repeat(60)}`);
  console.log(`Task: ${task}`);
  console.log("=".repeat(60));

  const result = await coordinator.run(task);

  console.log("\n--- Delegations ---");
  for (const d of result.delegations) {
    const delegation = d as { specialist: string; task: string; status: string };
    console.log(`  [${delegation.specialist}] ${delegation.status}: ${delegation.task.slice(0, 60)}…`);
  }

  console.log(`\n${"=".repeat(60)}`);
  console.log("FINAL ANSWER:");
  console.log("=".repeat(60));
  console.log(result.answer);
}

main().catch(console.error);
