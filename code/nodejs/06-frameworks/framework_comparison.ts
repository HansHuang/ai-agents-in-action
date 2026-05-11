/**
 * Framework comparison: the same RAG agent built three ways.
 *
 * Compares:
 *   1. FROM SCRATCH — pure OpenAI SDK
 *   2. LANGCHAIN-STYLE — pattern-based chain abstraction
 *   3. LANGGRAPH-STYLE — explicit state machine
 *
 * Measures lines-of-code approximation, dependency count, and approach tradeoffs.
 * See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o-mini";

export interface ComparisonResult {
  approach: string;
  answer: string;
  approachLines: number;
  dependencies: string[];
  tradeoffs: { pros: string[]; cons: string[] };
  durationMs: number;
}

// ---------------------------------------------------------------------------
// 1. From Scratch
// ---------------------------------------------------------------------------

async function fromScratch(question: string, context: string, client: OpenAI): Promise<string> {
  const resp = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: "Answer questions using only the provided context." },
      { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` },
    ],
    temperature: 0,
    max_tokens: 200,
  });
  return resp.choices[0].message.content?.trim() ?? "";
}

// ---------------------------------------------------------------------------
// 2. LangChain-Style (chain abstraction)
// ---------------------------------------------------------------------------

type ChainFn = (input: Record<string, string>) => Promise<string>;

function createRetrievalChain(client: OpenAI): ChainFn {
  return async (input: Record<string, string>) => {
    const resp = await client.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant. Answer using the provided context.",
        },
        {
          role: "user",
          content: `Context: ${input.context}\nQuestion: ${input.question}`,
        },
      ],
      temperature: 0,
      max_tokens: 200,
    });
    return resp.choices[0].message.content?.trim() ?? "";
  };
}

async function langchainStyle(question: string, context: string, client: OpenAI): Promise<string> {
  const chain = createRetrievalChain(client);
  return chain({ question, context });
}

// ---------------------------------------------------------------------------
// 3. LangGraph-Style (explicit state machine)
// ---------------------------------------------------------------------------

interface AgentState {
  question: string;
  context: string;
  retrievedDocs: string[];
  answer: string;
  step: "retrieve" | "generate" | "done";
}

async function langgraphStyle(question: string, context: string, client: OpenAI): Promise<string> {
  // Simulate a LangGraph state machine with explicit transitions
  let state: AgentState = {
    question,
    context,
    retrievedDocs: [],
    answer: "",
    step: "retrieve",
  };

  // Node: retrieve
  if (state.step === "retrieve") {
    state.retrievedDocs = context.split("\n").filter(Boolean).slice(0, 3);
    state.step = "generate";
  }

  // Node: generate
  if (state.step === "generate") {
    const retrievedContext = state.retrievedDocs.join("\n");
    const resp = await client.chat.completions.create({
      model: MODEL,
      messages: [
        { role: "system", content: "Answer based on retrieved documents." },
        { role: "user", content: `Documents:\n${retrievedContext}\n\nQuestion: ${state.question}` },
      ],
      temperature: 0,
      max_tokens: 200,
    });
    state.answer = resp.choices[0].message.content?.trim() ?? "";
    state.step = "done";
  }

  return state.answer;
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/**
 * Run all three approaches on the same question and return comparison results.
 */
export async function compareApproaches(
  question: string,
  context: string,
  client: OpenAI
): Promise<ComparisonResult[]> {
  const approaches: Array<{
    name: string;
    fn: (q: string, c: string, client: OpenAI) => Promise<string>;
    lines: number;
    deps: string[];
    pros: string[];
    cons: string[];
  }> = [
    {
      name: "From Scratch",
      fn: fromScratch,
      lines: 8,
      deps: ["openai"],
      pros: ["Zero overhead", "Full control", "Easy to debug", "Minimal dependencies"],
      cons: ["No reusable patterns", "Manual boilerplate for complex flows"],
    },
    {
      name: "LangChain-Style",
      fn: langchainStyle,
      lines: 15,
      deps: ["openai", "langchain-pattern"],
      pros: ["Composable chains", "Reusable pipeline components"],
      cons: ["Abstraction overhead", "Debugging requires understanding internals"],
    },
    {
      name: "LangGraph-Style",
      fn: langgraphStyle,
      lines: 30,
      deps: ["openai", "langgraph-pattern"],
      pros: ["Explicit state transitions", "Easy to add branches/loops", "Visualizable"],
      cons: ["More verbose for simple flows", "State management overhead"],
    },
  ];

  const results: ComparisonResult[] = [];

  for (const approach of approaches) {
    const start = Date.now();
    const answer = await approach.fn(question, context, client);
    const durationMs = Date.now() - start;
    results.push({
      approach: approach.name,
      answer,
      approachLines: approach.lines,
      dependencies: approach.deps,
      tradeoffs: { pros: approach.pros, cons: approach.cons },
      durationMs,
    });
  }

  return results;
}

/** Print comparison table. */
export function printComparison(results: ComparisonResult[]): void {
  console.log("\n=== Framework Comparison ===\n");
  for (const r of results) {
    console.log(`[${r.approach}] ${r.durationMs}ms | ~${r.approachLines} lines | deps: ${r.dependencies.join(", ")}`);
    console.log(`  Answer: ${r.answer.slice(0, 100)}${r.answer.length > 100 ? "..." : ""}`);
    console.log(`  Pros: ${r.tradeoffs.pros.slice(0, 2).join("; ")}`);
    console.log(`  Cons: ${r.tradeoffs.cons.slice(0, 1).join("; ")}`);
  }
}
