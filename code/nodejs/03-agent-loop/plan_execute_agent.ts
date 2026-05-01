/**
 * Plan-and-Execute agent in TypeScript.
 *
 * Three-phase pattern:
 *   Phase 1 PLAN:      LLM generates a structured JSON plan.
 *   Phase 2 EXECUTE:   Steps run in dependency order; independent steps in parallel.
 *   Phase 3 SYNTHESIZE: LLM combines results into a final answer.
 *
 * Structurally equivalent to code/python/03-agent-loop/plan_execute_agent.py.
 *
 * Run:  npx tsx plan_execute_agent.ts
 * See:  docs/02-the-agent-loop/03-planning-strategies.md
 */

import OpenAI from "openai";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Zod schemas (equivalent to plan_schema.py)
// ---------------------------------------------------------------------------

export const PlanStepSchema = z.object({
  step_number: z.number().int().min(1),
  description: z.string().min(10).max(500),
  tool_name: z.enum(["get_weather", "get_stock_price", "search_news"]).nullable(),
  tool_params: z.record(z.unknown()).nullable(),
  depends_on: z.array(z.number().int()).default([]),
  expected_output: z.string().min(5),
});

export const AgentPlanSchema = z.object({
  user_question: z.string(),
  steps: z.array(PlanStepSchema).min(1),
  estimated_tool_calls: z.number().int().default(0),
});

export const StepResultSchema = z.object({
  step_number: z.number().int().min(1),
  success: z.boolean(),
  data: z.record(z.unknown()).nullable().default(null),
  error: z.string().nullable().default(null),
  duration_ms: z.number().int().min(0).default(0),
});

export type PlanStep = z.infer<typeof PlanStepSchema>;
export type AgentPlan = z.infer<typeof AgentPlanSchema>;
export type StepResult = z.infer<typeof StepResultSchema>;

// ---------------------------------------------------------------------------
// Mock tool implementations
// ---------------------------------------------------------------------------

const WEATHER_MOCK: Record<string, Record<string, unknown>> = {
  Tokyo:     { temperature_c: 22, condition: "light rain",    humidity_percent: 85, wind_kph: 15 },
  London:    { temperature_c: 14, condition: "overcast",      humidity_percent: 78, wind_kph: 20 },
  "New York":{ temperature_c: 18, condition: "partly cloudy", humidity_percent: 60, wind_kph: 25 },
  Paris:     { temperature_c: 16, condition: "sunny",         humidity_percent: 55, wind_kph: 12 },
  Sydney:    { temperature_c: 28, condition: "clear",         humidity_percent: 45, wind_kph: 18 },
};

const STOCK_MOCK: Record<string, Record<string, unknown>> = {
  AAPL:  { price_usd: 192.35, change_percent:  1.2, currency: "USD", weekly_change_percent:  3.1 },
  GOOGL: { price_usd: 171.80, change_percent: -0.5, currency: "USD", weekly_change_percent:  1.5 },
  MSFT:  { price_usd: 415.10, change_percent:  0.8, currency: "USD", weekly_change_percent:  2.4 },
  TSLA:  { price_usd: 175.20, change_percent: -2.3, currency: "USD", weekly_change_percent: -4.1 },
  AMZN:  { price_usd: 188.40, change_percent:  0.3, currency: "USD", weekly_change_percent:  0.8 },
};

const NEWS_MOCK: Record<string, Array<{ headline: string; sentiment: string }>> = {
  AAPL: [
    { headline: "Apple unveils M4 Ultra chip with record AI performance", sentiment: "positive" },
    { headline: "iPhone 17 pre-orders exceed expectations",                sentiment: "positive" },
  ],
  MSFT: [
    { headline: "Microsoft Azure revenue grows 33% year-over-year",       sentiment: "positive" },
    { headline: "Copilot+ PC sales drive record Surface quarter",          sentiment: "positive" },
  ],
};

function getWeather(city: string): Record<string, unknown> {
  const key = city.split(",")[0].trim();
  return { city, ...(WEATHER_MOCK[key] ?? { temperature_c: 20, condition: "clear" }) };
}

function getStockPrice(ticker: string): Record<string, unknown> {
  const upper = ticker.toUpperCase();
  return { ticker: upper, market_status: "open", ...(STOCK_MOCK[upper] ?? { price_usd: 100, change_percent: 0 }) };
}

function searchNews(query: string): Record<string, unknown> {
  const ticker = query.toUpperCase().split(" ")[0];
  const articles = NEWS_MOCK[ticker] ?? [{ headline: `No major news for '${query}'`, sentiment: "neutral" }];
  return { query, articles, count: articles.length };
}

type ToolFn = (params: Record<string, unknown>) => Record<string, unknown>;

const TOOL_DISPATCH: Record<string, ToolFn> = {
  get_weather:     (p) => getWeather(p["city"] as string),
  get_stock_price: (p) => getStockPrice(p["ticker"] as string),
  search_news:     (p) => searchNews(p["query"] as string),
};

// ---------------------------------------------------------------------------
// Planner / synthesizer prompts
// ---------------------------------------------------------------------------

const PLANNER_SYSTEM = `You are a planning assistant. Break the user's request into concrete sequential steps.

Rules:
- step_number must start at 1 and increment by 1.
- tool_name must be one of: get_weather, get_stock_price, search_news — or null for reasoning/synthesis steps.
- tool_params must contain the exact arguments (or null for reasoning steps).
- depends_on lists the step_numbers this step must wait for.
- expected_output describes the result (at least 15 chars).
- Keep the plan to at most 10 steps.

Output ONLY valid JSON:
{
  "user_question": "<question>",
  "steps": [{"step_number":1,"description":"...","tool_name":"get_stock_price","tool_params":{"ticker":"AAPL"},"depends_on":[],"expected_output":"AAPL price and weekly change"}],
  "estimated_tool_calls": 1
}`;

const SYNTHESIZER_SYSTEM = `You are a synthesizer. Given the original question and all execution results,
write a complete, well-structured answer. Cite specific numbers and data from the results.
Use markdown formatting where it improves readability.`;

// ---------------------------------------------------------------------------
// PlanAndExecuteAgent
// ---------------------------------------------------------------------------

export class PlanAndExecuteAgent {
  private readonly model: string;
  private readonly maxSteps: number;
  private readonly client: OpenAI;

  constructor(
    model = "gpt-4o",
    maxSteps = 20,
    apiKey = process.env["OPENAI_API_KEY"] ?? "",
  ) {
    this.model = model;
    this.maxSteps = maxSteps;
    this.client = new OpenAI({ apiKey });
  }

  /** Run the full Plan → Execute → Synthesize cycle. */
  async run(userInput: string): Promise<{
    plan: PlanStep[];
    results: StepResult[];
    answer: string;
  }> {
    const plan = await this.generatePlan(userInput);
    const results = await this.executePlan(plan);
    const answer = await this.synthesize(userInput, plan, results);
    return { plan, results, answer };
  }

  // ------------------------------------------------------------------
  // Phase 1: Plan
  // ------------------------------------------------------------------

  private async generatePlan(userInput: string): Promise<PlanStep[]> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: [
        { role: "system", content: PLANNER_SYSTEM },
        { role: "user",   content: userInput },
      ],
      response_format: { type: "json_object" },
      temperature: 0,
    });

    const raw = response.choices[0]?.message.content ?? "{}";
    try {
      const parsed = JSON.parse(raw) as unknown;
      const planObj = AgentPlanSchema.parse(parsed);
      return planObj.steps.slice(0, this.maxSteps);
    } catch {
      // Fallback to single reasoning step
      return [
        {
          step_number:     1,
          description:     "Answer the user's question using available tools.",
          tool_name:       null,
          tool_params:     null,
          depends_on:      [],
          expected_output: "A complete answer to the user's question",
        },
      ];
    }
  }

  // ------------------------------------------------------------------
  // Phase 2: Execute
  // ------------------------------------------------------------------

  private async executePlan(steps: PlanStep[]): Promise<StepResult[]> {
    const results = new Map<number, StepResult>();
    let remaining = [...steps];
    const maxRounds = steps.length + 1;

    for (let round = 0; round < maxRounds && remaining.length > 0; round++) {
      const ready = remaining.filter((s) =>
        s.depends_on.every((dep) => results.has(dep)),
      );
      if (ready.length === 0) {
        // Unresolvable dependencies
        for (const s of remaining) {
          results.set(s.step_number, {
            step_number: s.step_number,
            success:     false,
            data:        null,
            error:       "Unresolved dependencies",
            duration_ms: 0,
          });
        }
        break;
      }

      // Execute ready steps in parallel
      const settled = await Promise.allSettled(
        ready.map((step) => this.executeStep(step, results)),
      );

      for (let i = 0; i < ready.length; i++) {
        const step = ready[i]!;
        const outcome = settled[i]!;
        if (outcome.status === "fulfilled") {
          results.set(step.step_number, outcome.value);
        } else {
          results.set(step.step_number, {
            step_number: step.step_number,
            success:     false,
            data:        null,
            error:       String(outcome.reason),
            duration_ms: 0,
          });
        }
        remaining = remaining.filter((s) => s.step_number !== step.step_number);
      }
    }

    return steps.map((s) => results.get(s.step_number)!);
  }

  private async executeStep(
    step: PlanStep,
    priorResults: Map<number, StepResult>,
  ): Promise<StepResult> {
    const start = Date.now();
    try {
      let data: Record<string, unknown>;
      if (step.tool_name !== null) {
        data = this.callTool(step.tool_name, (step.tool_params as Record<string, unknown>) ?? {});
      } else {
        data = await this.reason(step, priorResults);
      }
      return {
        step_number: step.step_number,
        success:     true,
        data,
        error:       null,
        duration_ms: Date.now() - start,
      };
    } catch (err) {
      return {
        step_number: step.step_number,
        success:     false,
        data:        null,
        error:       String(err),
        duration_ms: Date.now() - start,
      };
    }
  }

  private callTool(
    toolName: string,
    params: Record<string, unknown>,
  ): Record<string, unknown> {
    const fn = TOOL_DISPATCH[toolName];
    if (!fn) throw new Error(`Unknown tool: '${toolName}'`);
    return fn(params);
  }

  private async reason(
    step: PlanStep,
    priorResults: Map<number, StepResult>,
  ): Promise<Record<string, unknown>> {
    const contextParts: string[] = [];
    for (const depNum of step.depends_on) {
      const dep = priorResults.get(depNum);
      if (dep?.success && dep.data) {
        contextParts.push(`Step ${depNum} result:\n${JSON.stringify(dep.data, null, 2)}`);
      }
    }

    const userMessage =
      step.description +
      (contextParts.length > 0
        ? "\n\nContext from previous steps:\n" + contextParts.join("\n\n")
        : "");

    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: [
        { role: "system", content: "You are an executor. Complete the given reasoning step concisely." },
        { role: "user",   content: userMessage },
      ],
      temperature: 0.3,
    });
    return { reasoning: response.choices[0]?.message.content ?? "" };
  }

  // ------------------------------------------------------------------
  // Phase 3: Synthesize
  // ------------------------------------------------------------------

  private async synthesize(
    question: string,
    steps: PlanStep[],
    results: StepResult[],
  ): Promise<string> {
    const lines = [`Original question: ${question}\n`];
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i]!;
      const result = results[i]!;
      const status = result.success ? "✓" : "✗";
      lines.push(`Step ${step.step_number} [${status}]: ${step.description}`);
      if (result.success && result.data) {
        lines.push(`Result: ${JSON.stringify(result.data)}`);
      } else if (result.error) {
        lines.push(`Error: ${result.error}`);
      }
      lines.push("");
    }

    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: [
        { role: "system", content: SYNTHESIZER_SYSTEM },
        { role: "user",   content: lines.join("\n") },
      ],
      temperature: 0.5,
    });
    return response.choices[0]?.message.content ?? "";
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const agent = new PlanAndExecuteAgent();
  const query = "Compare Apple and Microsoft stock performance this week";

  console.log("\n" + "=".repeat(60));
  console.log(`Query: ${query}`);
  console.log("=".repeat(60));

  const output = await agent.run(query);

  console.log("\n--- Plan ---");
  for (const step of output.plan) {
    const deps = step.depends_on.length > 0 ? ` (depends on ${step.depends_on})` : "";
    const tool = step.tool_name ? ` [${step.tool_name}]` : " [reasoning]";
    console.log(`  ${step.step_number}. ${step.description}${tool}${deps}`);
  }

  console.log("\n--- Results ---");
  for (const result of output.results) {
    const status = result.success ? "✓" : "✗";
    console.log(`  Step ${result.step_number} [${status}] (${result.duration_ms}ms)`);
    if (result.success && result.data) {
      console.log(`    ${JSON.stringify(result.data)}`);
    } else if (result.error) {
      console.log(`    ERROR: ${result.error}`);
    }
  }

  console.log("\n--- Final Answer ---");
  console.log(output.answer);
}

main().catch(console.error);
