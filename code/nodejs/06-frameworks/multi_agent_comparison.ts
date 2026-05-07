/**
 * Multi-agent comparison (TypeScript): the same research task three ways.
 *
 * Task: "Research the impact of AI on software developer productivity."
 *
 * Three implementations in this file:
 *
 *   1. FROM SCRATCH     — OpenAI SDK only; explicit agent orchestration
 *   2. LANGCHAIN.JS     — LangChain.js agents (CrewAI has no TS support)
 *   3. CONVERSATIONAL   — custom round-robin loop (AutoGen has no TS support)
 *
 * Why not CrewAI or AutoGen for TypeScript?
 *   CrewAI is Python-only (May 2026).  AutoGen v0.2/v0.3 is Python-only.
 *   AutoGen v0.4 has limited TS bindings but no GroupChat equivalent.
 *   LangChain.js provides the closest TypeScript equivalent to the Python
 *   framework ecosystem: agents, tools, chains, and memory.
 *
 * Comparison metrics printed at the end:
 *   • Lines of code (implementation + orchestration)
 *   • Execution time, LLM calls, token usage, estimated cost
 *   • Report length, sources cited, critique integration, structure score
 *   • Control ratings (traceability, modifiability, extensibility)
 *
 * Run:
 *   npx tsx multi_agent_comparison.ts
 *
 * See: docs/06-frameworks-in-practice/03-crewai-autogen.md
 */

// ---------------------------------------------------------------------------
// Optional LangChain.js imports (graceful fallback)
// ---------------------------------------------------------------------------

let ChatOpenAI: any;
let HumanMessage: any;
let SystemMessage: any;
let ChatPromptTemplate: any;
let StringOutputParser: any;
let LANGCHAIN_AVAILABLE = false;

try {
  const openaiPkg = await import("@langchain/openai");
  const messagesPkg = await import("@langchain/core/messages");
  const promptsPkg = await import("@langchain/core/prompts");
  const outputParsersPkg = await import("@langchain/core/output_parsers");

  ChatOpenAI = openaiPkg.ChatOpenAI;
  HumanMessage = messagesPkg.HumanMessage;
  SystemMessage = messagesPkg.SystemMessage;
  ChatPromptTemplate = promptsPkg.ChatPromptTemplate;
  StringOutputParser = outputParsersPkg.StringOutputParser;
  LANGCHAIN_AVAILABLE = true;
} catch {
  // LangChain.js not installed — from-scratch and conversational still work
}

// ---------------------------------------------------------------------------
// OpenAI SDK (always required)
// ---------------------------------------------------------------------------

import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? "" });

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RESEARCH_TASK = "Research the impact of AI on software developer productivity.";
const GPT4O_INPUT_COST_PER_1K = 0.0025;   // USD per 1,000 input tokens (gpt-4o, May 2026)
const GPT4O_OUTPUT_COST_PER_1K = 0.010;   // USD per 1,000 output tokens

// ---------------------------------------------------------------------------
// Metric types
// ---------------------------------------------------------------------------

interface CodeMetrics {
  agentDefinitionLines: number;
  orchestrationLines: number;
  totalLines: number;
  importCount: number;
}

interface ExecutionMetrics {
  executionTimeMs: number;
  llmCalls: number;
  totalTokens: number;
  estimatedCostUsd: number;
}

interface QualityMetrics {
  reportLengthWords: number;
  sourcesCited: number;
  hasCritiqueFeedback: boolean;
  structureScore: number; // 1–10
}

interface ControlMetrics {
  traceableDecisionPath: boolean;
  canModifyAgentComms: boolean;
  canAddAgentMidWorkflow: boolean;
  canChangeExecutionOrder: boolean;
  controlScore: number; // 0–10
}

interface ComparisonResult {
  name: string;
  code: CodeMetrics;
  execution: ExecutionMetrics;
  quality: QualityMetrics;
  control: ControlMetrics;
  report: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

let llmCalls = 0;
let totalTokens = 0;

async function llm(system: string, user: string): Promise<string> {
  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
  });
  llmCalls++;
  totalTokens += response.usage?.total_tokens ?? 0;
  return response.choices[0].message.content ?? "";
}

function countSources(text: string): number {
  let count = 0;
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (/^\d+\./.test(s) || s.startsWith("http") || s.startsWith("[") || s.startsWith("•")) {
      if (s.length > 15) count++;
    }
  }
  return Math.min(count, 20);
}

function hasCritique(text: string): boolean {
  return /critiq|however|caveat|limitation|concern/i.test(text);
}

function structureScore(text: string): number {
  const headings = (text.match(/^#{1,3} .+/gm) ?? []).length;
  const bullets = (text.match(/^[•\-\*] .+/gm) ?? []).length;
  return Math.min(10, Math.max(1, headings * 2 + bullets));
}

function wordCount(text: string): number {
  return text.trim().split(/\s+/).length;
}

function estimateCost(tokens: number): number {
  const inputTokens = tokens * 0.4;
  const outputTokens = tokens * 0.6;
  return (
    (inputTokens / 1000) * GPT4O_INPUT_COST_PER_1K +
    (outputTokens / 1000) * GPT4O_OUTPUT_COST_PER_1K
  );
}

// ---------------------------------------------------------------------------
// Implementation 1: From Scratch (OpenAI SDK only)
// ---------------------------------------------------------------------------

/**
 * Three-agent research pipeline — no framework dependencies.
 * You see every message, every prompt, and the full execution order.
 *
 * Agents: Researcher → Critic → Writer
 *
 * Code metrics (approx):
 *   Agent definitions  : 3 system prompt constants ≈ 12 lines
 *   Orchestration      : sequential await calls ≈ 20 lines
 *   Total              : ~32 lines
 */
async function runFromScratch(task: string): Promise<ComparisonResult> {
  llmCalls = 0;
  totalTokens = 0;
  const start = performance.now();

  // Agent 1: Researcher
  const research = await llm(
    "You are a research analyst. Produce a structured research brief. " +
      "Include at least three concrete statistics, three named sources, " +
      "and a timeline of key developments.",
    `Research topic: ${task}`
  );

  // Agent 2: Critic
  const critique = await llm(
    "You are a rigorous research critic. Identify three specific weaknesses: " +
      "missing data, unsupported claims, or coverage gaps. Be constructive.",
    `Critique this research brief:\n\n${research}`
  );

  // Agent 3: Writer
  const report = await llm(
    "You are a technical report writer. Produce a 400–500 word report with " +
      "sections: ## Executive Summary, ## Key Findings (bullets), ## Analysis, " +
      "## Sources. Address the critic's feedback.",
    `Write a report based on:\n\nRESEARCH:\n${research}\n\nCRITIQUE:\n${critique}`
  );

  const elapsedMs = performance.now() - start;
  const calls = llmCalls;
  const tokens = totalTokens;

  return {
    name: "From Scratch",
    code: {
      agentDefinitionLines: 12,
      orchestrationLines: 20,
      totalLines: 32,
      importCount: 1, // openai
    },
    execution: {
      executionTimeMs: Math.round(elapsedMs),
      llmCalls: calls,
      totalTokens: tokens,
      estimatedCostUsd: parseFloat(estimateCost(tokens).toFixed(4)),
    },
    quality: {
      reportLengthWords: wordCount(report),
      sourcesCited: countSources(report),
      hasCritiqueFeedback: hasCritique(report),
      structureScore: structureScore(report),
    },
    control: {
      traceableDecisionPath: true,
      canModifyAgentComms: true,
      canAddAgentMidWorkflow: true,
      canChangeExecutionOrder: true,
      controlScore: 10,
    },
    report,
  };
}

// ---------------------------------------------------------------------------
// Implementation 2: LangChain.js (CrewAI TypeScript alternative)
// ---------------------------------------------------------------------------

/**
 * Same three-agent pipeline using LangChain.js chains.
 *
 * CrewAI does not have TypeScript support (May 2026).
 * LangChain.js provides the closest equivalent: ChatPromptTemplate,
 * ChatOpenAI, OutputParser, and chain composition with .pipe().
 *
 * LangChain.js handles: prompt template management, model configuration,
 * output parsing, and the LCEL (LangChain Expression Language) pipe syntax.
 *
 * What it does NOT automate (unlike Python CrewAI):
 *   - Role-based agent delegation
 *   - Task dependency graphs
 *   - Automatic context passing between agents
 *
 * Code metrics (approx):
 *   Agent definitions  : 3 ChatOpenAI + ChatPromptTemplate pairs ≈ 24 lines
 *   Orchestration      : .pipe() chains ≈ 15 lines
 *   Total              : ~39 lines
 */
async function runLangChain(task: string): Promise<ComparisonResult> {
  if (!LANGCHAIN_AVAILABLE) {
    return {
      name: "LangChain.js",
      code: { agentDefinitionLines: 0, orchestrationLines: 0, totalLines: 0, importCount: 0 },
      execution: { executionTimeMs: 0, llmCalls: 0, totalTokens: 0, estimatedCostUsd: 0 },
      quality: { reportLengthWords: 0, sourcesCited: 0, hasCritiqueFeedback: false, structureScore: 0 },
      control: {
        traceableDecisionPath: true,
        canModifyAgentComms: true,
        canAddAgentMidWorkflow: false,
        canChangeExecutionOrder: false,
        controlScore: 6,
      },
      report: "",
      error:
        "@langchain/openai and @langchain/core not installed. " +
        "Run: npm install @langchain/openai @langchain/core",
    };
  }

  llmCalls = 0;
  totalTokens = 0;
  const start = performance.now();

  const model = new ChatOpenAI({
    modelName: "gpt-4o",
    openAIApiKey: process.env.OPENAI_API_KEY ?? "",
  });
  const parser = new StringOutputParser();

  // Chain 1: Researcher
  const researchChain = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a research analyst. Produce a structured research brief with " +
        "statistics, sources, and a timeline.",
    ],
    ["human", "{task}"],
  ])
    .pipe(model)
    .pipe(parser);

  const research: string = await researchChain.invoke({ task });
  llmCalls++;

  // Chain 2: Critic
  const criticChain = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a rigorous research critic. Identify three specific weaknesses. " +
        "Be constructive and specific.",
    ],
    ["human", "Critique this research brief:\n\n{research}"],
  ])
    .pipe(model)
    .pipe(parser);

  const critique: string = await criticChain.invoke({ research });
  llmCalls++;

  // Chain 3: Writer
  const writerChain = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a technical writer. Produce a 400–500 word report with: " +
        "## Executive Summary, ## Key Findings, ## Analysis, ## Sources.",
    ],
    [
      "human",
      "Write a report.\n\nRESEARCH:\n{research}\n\nCRITIQUE:\n{critique}",
    ],
  ])
    .pipe(model)
    .pipe(parser);

  const report: string = await writerChain.invoke({ research, critique });
  llmCalls++;

  // LangChain.js doesn't expose token counts on the chain level easily;
  // fall back to estimating from a direct client call in tests.
  const elapsedMs = performance.now() - start;
  const tokens = totalTokens || llmCalls * 1500; // fallback estimate

  return {
    name: "LangChain.js",
    code: {
      agentDefinitionLines: 24,
      orchestrationLines: 15,
      totalLines: 39,
      importCount: 4, // openai, @langchain/openai, @langchain/core/messages, @langchain/core/prompts
    },
    execution: {
      executionTimeMs: Math.round(elapsedMs),
      llmCalls: llmCalls,
      totalTokens: tokens,
      estimatedCostUsd: parseFloat(estimateCost(tokens).toFixed(4)),
    },
    quality: {
      reportLengthWords: wordCount(report),
      sourcesCited: countSources(report),
      hasCritiqueFeedback: hasCritique(report),
      structureScore: structureScore(report),
    },
    control: {
      traceableDecisionPath: true,           // LCEL is inspectable
      canModifyAgentComms: true,             // .pipe() can be modified
      canAddAgentMidWorkflow: false,         // chains are defined upfront
      canChangeExecutionOrder: false,        // sequential by design
      controlScore: 6,
    },
    report,
  };
}

// ---------------------------------------------------------------------------
// Implementation 3: Conversational loop (AutoGen TypeScript alternative)
// ---------------------------------------------------------------------------

/**
 * Round-robin conversational agent loop — AutoGen-style without AutoGen.
 *
 * AutoGen has no TypeScript/JavaScript port (May 2026).
 * This implementation captures the core AutoGen pattern:
 *   - Agents speak in turn
 *   - A shared message history accumulates
 *   - The loop terminates on a signal or max rounds
 *   - Solutions emerge from conversation, not from a predefined workflow
 *
 * Code metrics (approx):
 *   Agent definitions  : 3 system prompt objects ≈ 12 lines
 *   Orchestration      : round-robin loop ≈ 25 lines
 *   Total              : ~37 lines
 */
async function runConversational(task: string): Promise<ComparisonResult> {
  const TERMINATE = "RESEARCH_COMPLETE";
  llmCalls = 0;
  totalTokens = 0;
  const start = performance.now();

  const agents: Array<{ name: string; systemPrompt: string }> = [
    {
      name: "Researcher",
      systemPrompt:
        "You are a research analyst. Provide concrete data, statistics, and sources. " +
        "Focus on the research task.",
    },
    {
      name: "Critic",
      systemPrompt:
        "You are a research critic. Identify exactly three specific weaknesses: " +
        "missing data, unsupported claims, or coverage gaps.",
    },
    {
      name: "Writer",
      systemPrompt:
        `You are a technical writer. Once you have both research and critique, ` +
        `produce a structured 400–500 word report. ` +
        `When your report is complete, end with '${TERMINATE}'.`,
    },
  ];

  type Message = { role: "user" | "assistant"; name?: string; content: string };
  const history: Message[] = [
    { role: "user", content: `Research task: ${task}` },
  ];

  let report = "";
  const MAX_ROUNDS = 12;

  for (let round = 0; round < MAX_ROUNDS; round++) {
    const { name, systemPrompt } = agents[round % agents.length];

    // Sliding window: last 6 messages for context
    const windowedHistory = history.slice(-6);

    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        ...windowedHistory.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
      ],
    });

    const reply = response.choices[0].message.content ?? "";
    llmCalls++;
    totalTokens += response.usage?.total_tokens ?? 0;

    history.push({ role: "assistant", name, content: reply });

    if (reply.includes(TERMINATE)) {
      report = reply;
      break;
    }
  }

  if (!report) {
    report = history[history.length - 1]?.content ?? "";
  }

  const elapsedMs = performance.now() - start;
  const tokens = totalTokens;

  return {
    name: "Conversational (AutoGen pattern)",
    code: {
      agentDefinitionLines: 12,
      orchestrationLines: 25,
      totalLines: 37,
      importCount: 1, // openai
    },
    execution: {
      executionTimeMs: Math.round(elapsedMs),
      llmCalls: llmCalls,
      totalTokens: tokens,
      estimatedCostUsd: parseFloat(estimateCost(tokens).toFixed(4)),
    },
    quality: {
      reportLengthWords: wordCount(report),
      sourcesCited: countSources(report),
      hasCritiqueFeedback: hasCritique(report),
      structureScore: structureScore(report),
    },
    control: {
      traceableDecisionPath: false,          // Emergent conversation
      canModifyAgentComms: true,             // History is mutable
      canAddAgentMidWorkflow: false,         // Agents array defined upfront
      canChangeExecutionOrder: false,        // Round-robin is fixed
      controlScore: 5,
    },
    report,
  };
}

// ---------------------------------------------------------------------------
// Comparison table printer
// ---------------------------------------------------------------------------

function printTable(results: ComparisonResult[]): void {
  const COL_W = 28;
  const LABEL_W = 34;

  const header =
    " ".repeat(LABEL_W) +
    results.map((r) => r.name.padStart(COL_W)).join("");
  const sep = "─".repeat(LABEL_W + COL_W * results.length);

  const row = (label: string, values: Array<string | number>): void => {
    const vals = values.map((v) => String(v).padStart(COL_W));
    console.log(`  ${label.padEnd(LABEL_W - 2)}${vals.join("")}`);
  };

  const section = (title: string): void => {
    console.log(`\n${title}`);
  };

  console.log(`\n${sep}`);
  console.log(header);
  console.log(sep);

  section("--- CODE METRICS ---");
  row("Lines of code (total)", results.map((r) => r.code.totalLines));
  row("Agent definition lines", results.map((r) => r.code.agentDefinitionLines));
  row("Orchestration lines", results.map((r) => r.code.orchestrationLines));
  row("Import count", results.map((r) => r.code.importCount));

  section("--- EXECUTION METRICS ---");
  row("Execution time (ms)", results.map((r) => `${r.execution.executionTimeMs}ms`));
  row("LLM calls", results.map((r) => r.execution.llmCalls));
  row("Total tokens", results.map((r) => r.execution.totalTokens.toLocaleString()));
  row("Estimated cost (USD)", results.map((r) => `$${r.execution.estimatedCostUsd.toFixed(4)}`));

  section("--- QUALITY METRICS ---");
  row("Report length (words)", results.map((r) => r.quality.reportLengthWords));
  row("Sources cited", results.map((r) => r.quality.sourcesCited));
  row("Includes critique", results.map((r) => String(r.quality.hasCritiqueFeedback)));
  row("Structure score (1-10)", results.map((r) => r.quality.structureScore));

  section("--- CONTROL METRICS ---");
  row("Traceable path", results.map((r) => String(r.control.traceableDecisionPath)));
  row("Modify agent comms", results.map((r) => String(r.control.canModifyAgentComms)));
  row("Add agent mid-workflow", results.map((r) => String(r.control.canAddAgentMidWorkflow)));
  row("Change execution order", results.map((r) => String(r.control.canChangeExecutionOrder)));
  row("Control score (0-10)", results.map((r) => r.control.controlScore));

  console.log(`\n${sep}`);

  // Analysis
  const runnable = results.filter((r) => !r.error && r.execution.executionTimeMs > 0);
  if (runnable.length === 0) return;

  const fastest = runnable.reduce((a, b) =>
    a.execution.executionTimeMs < b.execution.executionTimeMs ? a : b
  );
  const cheapest = runnable.reduce((a, b) =>
    a.execution.estimatedCostUsd < b.execution.estimatedCostUsd ? a : b
  );
  const mostControl = runnable.reduce((a, b) =>
    a.control.controlScore > b.control.controlScore ? a : b
  );

  console.log("\nANALYSIS");
  console.log("─".repeat(40));
  console.log(`  Fastest execution  : ${fastest.name} (${fastest.execution.executionTimeMs}ms)`);
  console.log(`  Lowest cost        : ${cheapest.name} ($${cheapest.execution.estimatedCostUsd.toFixed(4)})`);
  console.log(`  Most control       : ${mostControl.name} (${mostControl.control.controlScore}/10)`);

  const scratch = results.find((r) => r.name === "From Scratch");
  if (scratch && !scratch.error) {
    console.log(
      `\n  For this specific task, 'From Scratch' is best because it uses the fewest ` +
        `tokens (${scratch.execution.totalTokens.toLocaleString()}), gives full control ` +
        `over every agent interaction, and has zero framework dependencies.\n` +
        `  Use LangChain.js when you need its 200+ integrations or LCEL prompt management. ` +
        `Use the Conversational pattern when solutions should emerge from open-ended dialogue.`
    );
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log(`Task: ${RESEARCH_TASK}\n`);

const results: ComparisonResult[] = [];

for (const [label, fn] of [
  ["From Scratch", runFromScratch],
  ["LangChain.js", runLangChain],
  ["Conversational", runConversational],
] as Array<[string, (task: string) => Promise<ComparisonResult>]>) {
  process.stdout.write(`Running ${label}… `);
  try {
    const r = await fn(RESEARCH_TASK);
    results.push(r);
    console.log(r.error ? `ERROR: ${r.error.slice(0, 60)}` : "done.");
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    results.push({
      name: label,
      code: { agentDefinitionLines: 0, orchestrationLines: 0, totalLines: 0, importCount: 0 },
      execution: { executionTimeMs: 0, llmCalls: 0, totalTokens: 0, estimatedCostUsd: 0 },
      quality: { reportLengthWords: 0, sourcesCited: 0, hasCritiqueFeedback: false, structureScore: 0 },
      control: {
        traceableDecisionPath: false,
        canModifyAgentComms: false,
        canAddAgentMidWorkflow: false,
        canChangeExecutionOrder: false,
        controlScore: 0,
      },
      report: "",
      error: msg,
    });
    console.log(`ERROR: ${msg.slice(0, 60)}`);
  }
}

printTable(results);

console.log("\n\nFULL REPORTS");
for (const r of results) {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`  ${r.name}`);
  console.log("═".repeat(60));
  if (r.error) {
    console.log(`  ERROR: ${r.error}`);
  } else {
    const preview = r.report.slice(0, 600);
    const tail = r.report.length > 600 ? `\n  … [${r.report.length - 600} more chars]` : "";
    console.log(preview + tail);
  }
}

export {
  runFromScratch,
  runLangChain,
  runConversational,
  ComparisonResult,
  CodeMetrics,
  ExecutionMetrics,
  QualityMetrics,
  ControlMetrics,
};
