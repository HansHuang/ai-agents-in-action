/**
 * Interactive framework advisor: answer questions, get a tailored recommendation.
 *
 * Asks questions about your project and generates:
 *   - Primary framework recommendation with rationale
 *   - Frameworks to avoid and why
 *   - Migration path as the project evolves
 * See: docs/06-frameworks-in-practice/01-when-to-use-frameworks.md
 */

import OpenAI from "openai";

export interface ProjectProfile {
  teamSize: "solo" | "small" | "large";
  projectSize: "prototype" | "internal" | "production";
  needsStreaming: boolean;
  needsMultiAgent: boolean;
  needsRAG: boolean;
  needsObservability: boolean;
  primaryLanguage: "typescript" | "python" | "other";
  latencyRequirement: "realtime" | "interactive" | "batch";
}

export interface FrameworkRecommendation {
  primary: string;
  primaryRationale: string;
  secondary: string[];
  avoid: Array<{ framework: string; reason: string }>;
  migrationPath: string;
  architectureSummary: string;
}

// ---------------------------------------------------------------------------
// Decision rules
// ---------------------------------------------------------------------------

export function recommendFramework(profile: ProjectProfile): FrameworkRecommendation {
  let primary: string;
  let primaryRationale: string;
  const secondary: string[] = [];
  const avoid: Array<{ framework: string; reason: string }> = [];

  if (profile.primaryLanguage === "typescript") {
    if (profile.needsStreaming) {
      primary = "Vercel AI SDK";
      primaryRationale = "First-class streaming support with useChat/useCompletion hooks, ideal for TypeScript/Next.js.";
    } else if (profile.needsMultiAgent) {
      primary = "LangGraph.js";
      primaryRationale = "Explicit state machine for agent loops, handles complex multi-agent coordination with type safety.";
    } else if (profile.needsRAG) {
      primary = "LangChain.js";
      primaryRationale = "Rich ecosystem of loaders, splitters, and vector store integrations for RAG pipelines.";
    } else {
      primary = "OpenAI SDK (from scratch)";
      primaryRationale = "Simple projects don't need framework overhead. Start with the raw SDK and extract patterns.";
    }

    if (profile.needsObservability) secondary.push("LangSmith");
    if (profile.needsRAG && primary !== "LangChain.js") secondary.push("LangChain.js (for document loading)");
  } else {
    if (profile.needsMultiAgent) {
      primary = "LangGraph";
      primaryRationale = "Battle-tested Python framework for complex multi-agent workflows with graph-based state management.";
      secondary.push("CrewAI (for role-based pipelines)");
    } else if (profile.needsRAG) {
      primary = "LangChain";
      primaryRationale = "Extensive Python ecosystem with hundreds of integrations.";
    } else {
      primary = "OpenAI SDK (from scratch)";
      primaryRationale = "Start simple. Add frameworks only when you hit concrete limitations.";
    }
    if (profile.needsObservability) secondary.push("LangSmith");
  }

  if (profile.projectSize === "prototype") {
    avoid.push({ framework: "AutoGen", reason: "High setup cost not worth it for prototypes" });
  }
  if (!profile.needsMultiAgent) {
    avoid.push({ framework: "CrewAI/AutoGen", reason: "Multi-agent overhead without multi-agent need" });
  }

  const migrationPath =
    profile.projectSize === "prototype"
      ? "Prototype → Extract reusable patterns → Add LangChain selectively → LangGraph for complex flows"
      : "Production → Add LangSmith tracing → Gradually adopt framework primitives → Full migration only if needed";

  return {
    primary,
    primaryRationale,
    secondary,
    avoid,
    migrationPath,
    architectureSummary: `[${primary}] ${secondary.length ? `+ ${secondary.join(", ")}` : "(standalone)"}`,
  };
}

/**
 * Generate a detailed recommendation using an LLM.
 */
export async function advisorWithLLM(
  profile: ProjectProfile,
  userDescription: string,
  client: OpenAI
): Promise<string> {
  const ruleRec = recommendFramework(profile);

  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content:
          "You are an expert AI framework advisor. Given a project profile and rule-based recommendation, " +
          "provide a concise, actionable framework recommendation (3-4 sentences).",
      },
      {
        role: "user",
        content:
          `Project: ${userDescription}\n` +
          `Profile: ${JSON.stringify(profile)}\n` +
          `Rule-based recommendation: ${ruleRec.primary} — ${ruleRec.primaryRationale}`,
      },
    ],
    temperature: 0,
    max_tokens: 200,
  });

  return resp.choices[0].message.content?.trim() ?? ruleRec.primaryRationale;
}

/** Print a recommendation. */
export function printRecommendation(rec: FrameworkRecommendation): void {
  console.log("\nFramework Recommendation:");
  console.log(`  Primary: ${rec.primary}`);
  console.log(`  Rationale: ${rec.primaryRationale}`);
  if (rec.secondary.length) console.log(`  Secondary: ${rec.secondary.join(", ")}`);
  if (rec.avoid.length) {
    console.log("  Avoid:");
    rec.avoid.forEach((a) => console.log(`    - ${a.framework}: ${a.reason}`));
  }
  console.log(`  Migration Path: ${rec.migrationPath}`);
}

// Demo
function main(): void {
  const profiles: Array<{ desc: string; profile: ProjectProfile }> = [
    {
      desc: "Next.js chatbot with streaming",
      profile: {
        teamSize: "solo", projectSize: "production", needsStreaming: true,
        needsMultiAgent: false, needsRAG: false, needsObservability: true,
        primaryLanguage: "typescript", latencyRequirement: "realtime",
      },
    },
    {
      desc: "Python research pipeline with multiple agents",
      profile: {
        teamSize: "small", projectSize: "production", needsStreaming: false,
        needsMultiAgent: true, needsRAG: true, needsObservability: true,
        primaryLanguage: "python", latencyRequirement: "batch",
      },
    },
  ];

  for (const { desc, profile } of profiles) {
    console.log(`\n=== ${desc} ===`);
    printRecommendation(recommendFramework(profile));
  }
}

main();
