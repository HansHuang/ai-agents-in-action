/**
 * Hybrid Routing System — TypeScript
 * ====================================
 * Two-stage router: deterministic regex patterns first, LLM fallback
 * for ambiguous requests.
 *
 * Classes (mirroring the Python implementation):
 *   DeterministicRouter  — regex patterns, ~0 ms, no cost
 *   LLMRouter            — gpt-4o-mini, ~300 ms, tiny cost
 *   HybridRouter         — combines both stages
 *   RouterMetrics        — latency, cost, and intent distribution tracking
 *   HandlerRegistry      — maps intents to handler functions + config
 *   EscalatingRouter     — automatic re-routing on handler failure
 *   RoutingEvaluator     — accuracy measurement against labelled test cases
 *
 * See: docs/07-harness-engineering/03-routing-and-intent-classification.md
 */

import OpenAI from "openai";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface RouteResult {
  intent: string;
  confidence: number;
  /** "deterministic" | "llm" | "deterministic_fallback" */
  method: string;
  reasoning?: string;
  matchedPattern?: string;
  extractedParams?: Record<string, string | null>;
}

export interface HandlerConfig {
  model: string;
  maxTokens: number;
  temperature: number;
  timeoutSeconds: number;
  requiresTools: boolean;
  requiresRag: boolean;
  requiresApproval: boolean;
  costBudget: number;
}

export function defaultHandlerConfig(overrides: Partial<HandlerConfig> = {}): HandlerConfig {
  return {
    model: "gpt-4o-mini",
    maxTokens: 1024,
    temperature: 0.7,
    timeoutSeconds: 30,
    requiresTools: false,
    requiresRag: false,
    requiresApproval: false,
    costBudget: 0.01,
    ...overrides,
  };
}

export interface HandlerResponse {
  content: string;
  handlerUsed: string;
  tokensUsed: number;
  cost: number;
  metadata: Record<string, unknown>;
}

export type HandlerFn = (
  userInput: string,
  conversationHistory: Array<{ role: string; content: string }>,
  config: HandlerConfig
) => Promise<HandlerResponse>;

export interface RouteHandler {
  handler: HandlerFn;
  config: HandlerConfig;
}

export interface RoutingTestCase {
  userInput: string;
  expectedIntent: string;
  description?: string;
}

export interface EvaluationResult {
  input: string;
  expected: string;
  predicted: string;
  correct: boolean;
  method: string;
  confidence: number;
}

export interface RoutingReport {
  overallAccuracy: number;
  totalCases: number;
  byIntent: Record<string, number>;
  topMisclassifications: Array<[string, number]>;
  deterministicRate: number;
  avgConfidenceCorrect: number;
  avgConfidenceIncorrect: number;
}

// ---------------------------------------------------------------------------
// Zod schema for LLM routing response
// ---------------------------------------------------------------------------

const LLMRouteResponseSchema = z.object({
  intent: z.string(),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().optional(),
  extracted_params: z
    .object({
      order_number: z.string().nullable().optional(),
      product_name: z.string().nullable().optional(),
      issue_type: z.string().nullable().optional(),
    })
    .optional(),
});

type LLMRouteResponse = z.infer<typeof LLMRouteResponseSchema>;

// ---------------------------------------------------------------------------
// DeterministicRouter
// ---------------------------------------------------------------------------

const DEFAULT_PATTERNS: Record<string, string[]> = {
  greeting: [
    "^(hi|hello|hey|good morning|good evening|good afternoon|yo|sup)\\b",
    "^(how are you|how's it going|what's up|howdy)\\b",
    "^(nice to meet you|pleased to meet you)\\b",
  ],
  goodbye: [
    "\\b(bye|goodbye|see you|talk later|farewell|ciao|later|ttyl)\\b",
    "\\b(take care|have a good (day|night|one))\\b",
  ],
  thanks: [
    "\\b(thanks|thank you|thx|ty|appreciate it|grateful|cheers)\\b",
    "\\b(many thanks|much appreciated|that's helpful)\\b",
  ],
  reset: [
    "\\b(start over|start fresh|reset|clear|new conversation|forget everything|fresh start)\\b",
    "\\b(wipe (the slate|history)|begin again|restart)\\b",
  ],
  help: [
    "\\b(what can you do|help me|capabilities|features|how do (I|you)|what do you (do|know))\\b",
    "^help$",
    "\\b(show me (what you|your) (can do|capabilities))\\b",
  ],
  weather: [
    "\\b(weather|temperature|forecast|humidity|rain(ing)?|sunny|cloudy|snow(ing)?|wind)\\b",
    "\\b(what's it like outside|will it (rain|snow))\\b",
  ],
  stock: [
    "\\b(stock|market|price|ticker|nasdaq|dow jones|s&p|invest(ment)?|share price|equity)\\b",
    "\\b(\\baapl\\b|\\bgoog\\b|\\bmsft\\b|\\btsla\\b|\\bamzn\\b)\\b",
  ],
  order_lookup: [
    "\\b(order|tracking|shipment|delivery|where is my|status of)\\b.*\\b(order|package|item|number|parcel)\\b",
    "\\border\\s*#?\\d+\\b",
    "\\b(track|locate) my (package|order|shipment)\\b",
  ],
  return_request: [
    "\\b(return|refund|exchange|money back|send back|cancel order|send it back)\\b",
    "\\b(initiate a return|process a refund|want my money back)\\b",
  ],
  billing: [
    "\\b(bill|invoice|charge|payment|subscription|receipt|pricing|cost|fee)\\b",
    "\\b(overcharged|unauthorized charge|billing issue|payment failed)\\b",
  ],
  technical_support: [
    "\\b(not working|broken|error|bug|crash|down|failed|issue|problem with)\\b",
    "\\b(won't (load|open|start)|keeps (crashing|freezing)|can't (connect|access))\\b",
  ],
  account: [
    "\\b(account|login|password|profile|settings|email change|update.*info|sign in)\\b",
    "\\b(forgot (my )?password|reset (my )?password|locked out|can't log in)\\b",
  ],
};

export class DeterministicRouter {
  private compiled: Map<string, RegExp[]>;

  constructor(patterns: Record<string, string[]> = DEFAULT_PATTERNS) {
    this.compiled = new Map(
      Object.entries(patterns).map(([intent, pats]) => [
        intent,
        pats.map((p) => new RegExp(p, "i")),
      ])
    );
  }

  classify(userInput: string): RouteResult | null {
    const matches: Array<{ intent: string; pattern: RegExp }> = [];

    for (const [intent, patterns] of this.compiled) {
      for (const pattern of patterns) {
        if (pattern.test(userInput)) {
          matches.push({ intent, pattern });
        }
      }
    }

    if (matches.length === 0) return null;

    // Longest pattern string = most specific
    const best = matches.reduce((a, b) =>
      a.pattern.source.length >= b.pattern.source.length ? a : b
    );

    const confidence = matches.length === 1 ? 0.85 : 0.65;

    return {
      intent: best.intent,
      confidence,
      method: "deterministic",
      matchedPattern: best.pattern.source,
    };
  }
}

// ---------------------------------------------------------------------------
// LLMRouter
// ---------------------------------------------------------------------------

const ROUTING_PROMPT = `You are a request classifier for a customer-facing AI assistant.
Analyze the user's message and determine its primary intent.

Available routes:
- simple_chat: Casual conversation, greetings, general questions not requiring tools
- knowledge_question: Questions answerable from a knowledge base (policies, docs, FAQs)
- agent_task: Requests requiring tool use (lookups, calculations, multi-step tasks)
- human_escalation: User explicitly asks for a human, or the request is too complex/sensitive
- support_request: Customer support issues, complaints, problems with products or services
- out_of_scope: Requests that cannot or should not be handled (harmful, impossible, off-topic)

Classification rules:
- "hi" or small talk → simple_chat
- Questions about policies, procedures, documentation → knowledge_question
- Requests to DO something (look up, calculate, book, create, send) → agent_task
- Frustrated user demanding a person → human_escalation
- Reports a problem with a product or service → support_request
- Inappropriate, impossible, or clearly unrelated requests → out_of_scope

Output ONLY a JSON object — no markdown, no extra text:
{
    "intent": "<intent_name>",
    "confidence": 0.0,
    "reasoning": "<one sentence>",
    "extracted_params": {
        "order_number": null,
        "product_name": null,
        "issue_type": null
    }
}`;

export class LLMRouter {
  private client: OpenAI;
  readonly model: string;

  constructor(model = "gpt-4o-mini") {
    this.model = model;
    this.client = new OpenAI({ apiKey: process.env["OPENAI_API_KEY"] ?? "" });
  }

  async classify(
    userInput: string,
    conversationHistory: Array<{ role: string; content: string }> = []
  ): Promise<RouteResult> {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      { role: "system", content: ROUTING_PROMPT },
    ];

    if (conversationHistory.length > 0) {
      const recent = conversationHistory.slice(-4);
      const summary = recent
        .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.slice(0, 200)}`)
        .join("\n");
      messages.push({
        role: "user",
        content: `Recent conversation:\n${summary}\n\nClassify this message: ${userInput}`,
      });
    } else {
      messages.push({ role: "user", content: userInput });
    }

    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages,
        response_format: { type: "json_object" },
        temperature: 0.1,
      });

      const raw = response.choices[0]?.message?.content ?? "{}";
      const parsed = JSON.parse(raw) as unknown;
      const result = LLMRouteResponseSchema.parse(parsed);

      return {
        intent: result.intent,
        confidence: result.confidence,
        method: "llm",
        reasoning: result.reasoning,
        extractedParams: result.extracted_params as Record<string, string | null> | undefined,
      };
    } catch (err) {
      return {
        intent: "simple_chat",
        confidence: 0.3,
        method: "llm",
        reasoning: `Classification failed (${err}); defaulting to simple_chat`,
      };
    }
  }
}

// ---------------------------------------------------------------------------
// RouterMetrics
// ---------------------------------------------------------------------------

export class RouterMetrics {
  total = 0;
  private byMethod = new Map<string, number>();
  private byIntent = new Map<string, number>();
  private latenciesMs: number[] = [];
  private llmCosts: number[] = [];

  record(method: string, intent: string, latencyMs = 0, cost = 0): void {
    this.total++;
    this.byMethod.set(method, (this.byMethod.get(method) ?? 0) + 1);
    this.byIntent.set(intent, (this.byIntent.get(intent) ?? 0) + 1);
    this.latenciesMs.push(latencyMs);
    this.llmCosts.push(cost);
  }

  summary(): Record<string, unknown> {
    const n = Math.max(this.total, 1);
    const sorted = [...this.latenciesMs].sort((a, b) => a - b);
    const avg = sorted.reduce((s, v) => s + v, 0) / Math.max(sorted.length, 1);
    const p50 = sorted[Math.floor(sorted.length / 2)] ?? 0;
    const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? 0;

    return {
      totalRouted: this.total,
      deterministicRate: (this.byMethod.get("deterministic") ?? 0) / n,
      llmFallbackRate: (this.byMethod.get("llm") ?? 0) / n,
      intentDistribution: Object.fromEntries(this.byIntent),
      avgLatencyMs: avg,
      p50LatencyMs: p50,
      p95LatencyMs: p95,
      totalLlmCostUsd: this.llmCosts.reduce((s, c) => s + c, 0),
    };
  }
}

// ---------------------------------------------------------------------------
// HybridRouter
// ---------------------------------------------------------------------------

export class HybridRouter {
  readonly deterministic: DeterministicRouter;
  readonly llm: LLMRouter;
  readonly metrics: RouterMetrics;

  constructor() {
    this.deterministic = new DeterministicRouter();
    this.llm = new LLMRouter();
    this.metrics = new RouterMetrics();
  }

  async route(
    userInput: string,
    conversationHistory: Array<{ role: string; content: string }> = []
  ): Promise<RouteResult> {
    const t0 = Date.now();

    // Stage 1: deterministic
    const detResult = this.deterministic.classify(userInput);

    if (detResult && detResult.confidence > 0.8) {
      const latency = Date.now() - t0;
      this.metrics.record("deterministic", detResult.intent, latency);
      return detResult;
    }

    // Stage 2: LLM
    const llmResult = await this.llm.classify(userInput, conversationHistory);
    const latency = Date.now() - t0;

    if (detResult && detResult.confidence > llmResult.confidence) {
      this.metrics.record("deterministic_fallback", detResult.intent, latency);
      return { ...detResult, method: "deterministic_fallback" };
    }

    this.metrics.record("llm", llmResult.intent, latency);
    return llmResult;
  }

  getMetrics(): Record<string, unknown> {
    return this.metrics.summary();
  }
}

// ---------------------------------------------------------------------------
// HandlerRegistry
// ---------------------------------------------------------------------------

export class HandlerRegistry {
  private handlers = new Map<string, RouteHandler>();
  private defaultIntent = "simple_chat";

  register(intent: string, handler: HandlerFn, config: HandlerConfig): void {
    this.handlers.set(intent, { handler, config });
  }

  getHandler(intent: string): RouteHandler {
    return (
      this.handlers.get(intent) ??
      this.handlers.get(this.defaultIntent) ?? {
        handler: async () => ({
          content: "No handler configured.",
          handlerUsed: "fallback",
          tokensUsed: 0,
          cost: 0,
          metadata: {},
        }),
        config: defaultHandlerConfig(),
      }
    );
  }

  setDefaultIntent(intent: string): void {
    this.defaultIntent = intent;
  }
}

// ---------------------------------------------------------------------------
// EscalatingRouter
// ---------------------------------------------------------------------------

const ESCALATION_PATHS: Record<string, string[]> = {
  simple_chat: ["knowledge_question"],
  knowledge_question: ["agent_task"],
  agent_task: ["human_escalation"],
  support_request: ["human_escalation"],
  human_escalation: [],
};

const UNCERTAINTY_PHRASES = [
  "i'm not sure",
  "i don't know",
  "i cannot",
  "i'm unable",
  "i don't have enough information",
  "i have no information",
  "i couldn't find",
];

export class EscalatingRouter {
  constructor(
    private readonly router: HybridRouter,
    private readonly registry: HandlerRegistry
  ) {}

  async handle(
    userInput: string,
    conversationHistory: Array<{ role: string; content: string }> = []
  ): Promise<HandlerResponse> {
    const intent = await this.router.route(userInput, conversationHistory);
    const { handler, config } = this.registry.getHandler(intent.intent);

    try {
      const response = await handler(userInput, conversationHistory, config);

      if (this.shouldEscalate(response)) {
        return this.escalate(intent.intent, userInput, conversationHistory, response);
      }

      return response;
    } catch (err) {
      return this.escalate(intent.intent, userInput, conversationHistory, null, String(err));
    }
  }

  private shouldEscalate(response: HandlerResponse): boolean {
    const meta = response.metadata;
    if ((meta["documents_found"] as number) === 0) return true;
    if ((meta["iterations"] as number) >= 10) return true;

    const lower = response.content.toLowerCase();
    return UNCERTAINTY_PHRASES.some((p) => lower.includes(p));
  }

  private async escalate(
    originalIntent: string,
    userInput: string,
    conversationHistory: Array<{ role: string; content: string }>,
    previousResponse: HandlerResponse | null,
    error?: string
  ): Promise<HandlerResponse> {
    const path = ESCALATION_PATHS[originalIntent] ?? ["human_escalation"];

    for (const nextIntent of path) {
      const { handler, config } = this.registry.getHandler(nextIntent);

      const augmented = previousResponse
        ? `[Previous attempt via '${originalIntent}' was insufficient. ` +
          `Response was: '${previousResponse.content.slice(0, 200)}...']\n\nOriginal request: ${userInput}`
        : userInput;

      try {
        const response = await handler(augmented, conversationHistory, config);
        response.metadata["escalated_from"] = originalIntent;
        response.metadata["escalation_reason"] = error ?? "low_confidence";
        return response;
      } catch {
        continue;
      }
    }

    return {
      content:
        "I apologize — I'm having trouble processing your request. " +
        "A human team member will follow up with you shortly.",
      handlerUsed: "escalation_fallback",
      tokensUsed: 0,
      cost: 0,
      metadata: { escalation_chain_exhausted: true },
    };
  }
}

// ---------------------------------------------------------------------------
// RoutingEvaluator
// ---------------------------------------------------------------------------

export class RoutingEvaluator {
  constructor(private readonly router: HybridRouter) {}

  async evaluate(testCases: RoutingTestCase[]): Promise<RoutingReport> {
    const results: EvaluationResult[] = [];

    for (const tc of testCases) {
      const route = await this.router.route(tc.userInput);
      const correct = route.intent === tc.expectedIntent;
      results.push({
        input: tc.userInput,
        expected: tc.expectedIntent,
        predicted: route.intent,
        correct,
        method: route.method,
        confidence: route.confidence,
      });
    }

    return this.generateReport(results);
  }

  private generateReport(results: EvaluationResult[]): RoutingReport {
    const total = results.length;
    const correctCount = results.filter((r) => r.correct).length;

    const byIntentRaw = new Map<string, { correct: number; total: number }>();
    for (const r of results) {
      const existing = byIntentRaw.get(r.expected) ?? { correct: 0, total: 0 };
      existing.total++;
      if (r.correct) existing.correct++;
      byIntentRaw.set(r.expected, existing);
    }

    const byIntent = Object.fromEntries(
      [...byIntentRaw.entries()].map(([intent, stats]) => [
        intent,
        stats.correct / Math.max(stats.total, 1),
      ])
    );

    const misclassRaw = new Map<string, number>();
    for (const r of results) {
      if (!r.correct) {
        const key = `${r.expected} → ${r.predicted}`;
        misclassRaw.set(key, (misclassRaw.get(key) ?? 0) + 1);
      }
    }

    const topMisclassifications = [...misclassRaw.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10) as Array<[string, number]>;

    const correctConfs = results.filter((r) => r.correct).map((r) => r.confidence);
    const incorrectConfs = results.filter((r) => !r.correct).map((r) => r.confidence);
    const avg = (arr: number[]) =>
      arr.length === 0 ? 0 : arr.reduce((s, v) => s + v, 0) / arr.length;

    return {
      overallAccuracy: correctCount / Math.max(total, 1),
      totalCases: total,
      byIntent,
      topMisclassifications,
      deterministicRate:
        results.filter((r) => r.method === "deterministic").length / Math.max(total, 1),
      avgConfidenceCorrect: avg(correctConfs),
      avgConfidenceIncorrect: avg(incorrectConfs),
    };
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function demo(): Promise<void> {
  // Stub LLM so demo runs without an API key
  const stubAnswers: Record<string, string> = {
    "tell me about your return policy": "knowledge_question",
    "where is my order #12345": "agent_task",
    "compare iphone 15 and samsung s24": "agent_task",
    "do you offer international shipping?": "knowledge_question",
  };

  const router = new HybridRouter();
  const originalClassify = router.llm.classify.bind(router.llm);
  (router.llm as unknown as { classify: typeof router.llm.classify }).classify = async (
    input: string
  ) => {
    const key = input.toLowerCase().replace(/[?!.]+$/, "");
    const intent = Object.entries(stubAnswers).find(([k]) => key.includes(k))?.[1] ?? "simple_chat";
    return { intent, confidence: 0.88, method: "llm" };
  };

  const testCases: Array<[string, string]> = [
    ["hi", "greeting"],
    ["What's the weather in Tokyo?", "weather"],
    ["Tell me about your return policy", "knowledge_question"],
    ["Where is my order #12345", "agent_task"],
    ["I want to talk to a real person", "human_escalation"],
    ["bye", "goodbye"],
    ["thanks!", "thanks"],
    ["AAPL stock price today", "stock"],
    ["Do you offer international shipping?", "knowledge_question"],
    ["start over", "reset"],
    ["I need to change my password", "account"],
    ["the app won't load", "technical_support"],
    ["what can you do?", "help"],
  ];

  console.log("\n" + "=".repeat(70));
  console.log("HYBRID ROUTING DEMO (TypeScript)");
  console.log("=".repeat(70));

  let correct = 0;
  for (const [input, expected] of testCases) {
    const result = await router.route(input);
    const ok = result.intent === expected;
    if (ok) correct++;
    const mark = ok ? "✓" : "✗";
    console.log(
      `${mark} ${input.padEnd(38)} → ${result.intent.padEnd(22)} [${result.method}] ${result.confidence.toFixed(2)}`
    );
  }

  const accuracy = (correct / testCases.length) * 100;
  console.log(`\nAccuracy: ${correct}/${testCases.length} (${accuracy.toFixed(0)}%)`);

  const metrics = router.getMetrics();
  console.log("\nMetrics:", JSON.stringify(metrics, null, 2));
}

// Run demo when executed directly
if (require.main === module) {
  demo().catch(console.error);
}
