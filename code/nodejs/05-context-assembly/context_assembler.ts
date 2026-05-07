/**
 * Context Assembler — dynamic context assembly from multiple sources.
 *
 * Assembles the LLM context string from:
 *   1. RAG retrieved documents
 *   2. Tool execution results
 *   3. User profile / preferences
 *   4. Conversation summary
 *   5. Template variables
 *
 * Integrates with ContextBudget for enforcement and a lightweight
 * structure helper for attention-aware formatting.
 *
 * See: docs/04-context-engineering/01-the-context-window-as-a-resource.md
 */

import { ContextBudget, countTokens, type ContentInput } from "./context_budget.js";
import { get_encoding } from "tiktoken";

// ---------------------------------------------------------------------------
// Simple template renderer (${var} syntax)
// ---------------------------------------------------------------------------

function renderTemplate(template: string, vars: Record<string, string>): string {
  return template.replace(/\$\{(\w+)\}|\$(\w+)/g, (_, braced, bare) => {
    const key = braced ?? bare;
    return key in vars ? vars[key] : `$${key}`;
  });
}

// ---------------------------------------------------------------------------
// Lightweight structure helper (mirrors ContextOptimizer.structureForRetrieval)
// ---------------------------------------------------------------------------

function structureDocuments(documents: Array<{ text: string; metadata?: Record<string, unknown> }>): string {
  const parts: string[] = ["## Context Overview\n"];
  for (let i = 0; i < documents.length; i++) {
    const source = (documents[i].metadata?.["source"] as string) ?? `Document ${i + 1}`;
    parts.push(`- Section ${i + 1}: ${source}`);
  }
  parts.push("\n---\n");
  for (let i = 0; i < documents.length; i++) {
    const source = (documents[i].metadata?.["source"] as string) ?? `Document ${i + 1}`;
    parts.push(`## [${i + 1}] ${source}\n`);
    parts.push(documents[i].text);
    parts.push(`\n[End Section ${i + 1}]\n`);
    parts.push("---\n");
  }
  return parts.join("\n");
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface ContextConfig {
  /** Python-style $variable or ${variable} template. */
  template: string;
  /** Variable substitutions for the template. */
  templateVars?: Record<string, string>;
  /** Which sources to include. Defaults to all four. */
  includeSources?: Array<"rag" | "tools" | "profile" | "summary">;
  /** Per-source token caps. */
  maxTokensPerSource?: Partial<Record<"rag" | "tools" | "profile" | "summary", number>>;
  /** Output format hint. Only "markdown" triggers structural optimisation. */
  format?: "markdown" | "plain" | "json";
  /** Preferred source order. */
  priorityOrder?: Array<"rag" | "tools" | "profile" | "summary">;
}

// ---------------------------------------------------------------------------
// Assembly result
// ---------------------------------------------------------------------------

export interface AssemblyResult {
  /** Final assembled context string. */
  context: string;
  /** Token count per source section. */
  tokenBreakdown: Record<string, number>;
  /** Sources that made it into the context. */
  sourcesIncluded: string[];
  /** Sources dropped due to budget exhaustion. */
  sourcesExcluded: string[];
  /** Sum of all included section tokens. */
  readonly totalTokens: number;
}

function makeAssemblyResult(
  context: string,
  tokenBreakdown: Record<string, number>,
  sourcesIncluded: string[],
  sourcesExcluded: string[]
): AssemblyResult {
  return {
    context,
    tokenBreakdown,
    sourcesIncluded,
    sourcesExcluded,
    get totalTokens() {
      return Object.values(this.tokenBreakdown).reduce((s, v) => s + v, 0);
    },
  };
}

// ---------------------------------------------------------------------------
// ContextAssembler
// ---------------------------------------------------------------------------

const SECTION_HEADERS: Record<string, string> = {
  rag:     "## Retrieved Documents\n",
  tools:   "## Tool Results\n",
  profile: "## User Profile\n",
  summary: "## Conversation Summary\n",
};

/**
 * Assemble context for LLM calls from multiple sources.
 *
 * @example
 * ```ts
 * const budget    = new ContextBudget(128_000);
 * const assembler = new ContextAssembler(budget);
 *
 * const result = await assembler.assemble({
 *   template: "You are a ${role} support agent.",
 *   variables: { role: "customer" },
 *   retrievedDocs: ragDocs,
 *   userProfile: { name: "Alice", plan: "Pro" },
 *   query: "How do I cancel?",
 * });
 * ```
 */
export class ContextAssembler {
  private readonly budget: ContextBudget;
  private readonly model: string;

  constructor(budget: ContextBudget, model: string = "gpt-4o") {
    this.budget = budget;
    this.model  = model;
  }

  /**
   * Assemble the dynamic context for an LLM call.
   *
   * Sources are assembled in priority order (rag → tools → profile → summary).
   * Each section is added while budget remains; oversized sections are truncated;
   * sections with no remaining budget are excluded.
   */
  assemble(options: {
    template: string;
    variables?: Record<string, string>;
    retrievedDocs?: Array<{ text: string; metadata?: Record<string, unknown> }>;
    toolResults?: Array<{ tool: string; result: unknown }>;
    userProfile?: Record<string, string>;
    conversationSummary?: string;
    query?: string;
    optimize?: boolean;
  }): AssemblyResult {
    const {
      template,
      variables = {},
      retrievedDocs,
      toolResults,
      userProfile,
      conversationSummary,
      query = "",
      optimize = true,
    } = options;

    // Render template
    const renderedTemplate = renderTemplate(template, variables);

    // Build source sections
    const sections: Record<string, string> = {};

    if (retrievedDocs && retrievedDocs.length > 0) {
      sections["rag"] =
        optimize && query
          ? structureDocuments(retrievedDocs)
          : retrievedDocs.map((d) => d.text).join("\n\n");
    }

    if (toolResults && toolResults.length > 0) {
      sections["tools"] = toolResults
        .map(({ tool, result }) => {
          const resultStr =
            typeof result === "object"
              ? JSON.stringify(result, null, 2)
              : String(result);
          return `**${tool}**:\n${resultStr}`;
        })
        .join("\n\n");
    }

    if (userProfile && Object.keys(userProfile).length > 0) {
      sections["profile"] = Object.entries(userProfile)
        .map(([k, v]) => `- **${k}**: ${v}`)
        .join("\n");
    }

    if (conversationSummary) {
      sections["summary"] = conversationSummary;
    }

    // Assemble within budget
    const dcBudget = this.budget.getTokenBudget("dynamic_context");
    const contextParts: string[] = [renderedTemplate];
    const tokenBreakdown: Record<string, number> = {
      template: countTokens(renderedTemplate, this.model),
    };
    const sourcesIncluded: string[] = [];
    const sourcesExcluded: string[] = [];

    for (const source of ["rag", "tools", "profile", "summary"] as const) {
      if (!(source in sections)) continue;

      const content    = sections[source];
      const header     = SECTION_HEADERS[source] ?? `## ${source}\n`;
      const sectionText  = header + content;
      const sectionTok   = countTokens(sectionText, this.model);
      const usedSoFar    = Object.values(tokenBreakdown).reduce((s, v) => s + v, 0);

      if (usedSoFar + sectionTok <= dcBudget) {
        contextParts.push(sectionText);
        tokenBreakdown[source] = sectionTok;
        sourcesIncluded.push(source);
      } else {
        const available = dcBudget - usedSoFar - countTokens(header, this.model);
        if (available > 100) {
          const truncated  = this.truncateToTokens(content, available);
          const truncText  = header + truncated;
          const truncTok   = countTokens(truncText, this.model);
          contextParts.push(truncText);
          tokenBreakdown[source] = truncTok;
          sourcesIncluded.push(source);
        } else {
          sourcesExcluded.push(source);
        }
      }
    }

    return makeAssemblyResult(
      contextParts.join("\n\n"),
      tokenBreakdown,
      sourcesIncluded,
      sourcesExcluded
    );
  }

  /**
   * Assemble context from a declarative {@link ContextConfig}.
   */
  assembleFromConfig(
    config: ContextConfig,
    data: {
      retrievedDocs?: Array<{ text: string; metadata?: Record<string, unknown> }>;
      toolResults?: Array<{ tool: string; result: unknown }>;
      userProfile?: Record<string, string>;
      conversationSummary?: string;
      query?: string;
    } = {}
  ): AssemblyResult {
    const included = new Set(config.includeSources ?? ["rag", "tools", "profile", "summary"]);

    let docs    = included.has("rag")     ? data.retrievedDocs      : undefined;
    const tools   = included.has("tools")   ? data.toolResults         : undefined;
    const profile = included.has("profile") ? data.userProfile         : undefined;
    const summary = included.has("summary") ? data.conversationSummary : undefined;

    // Apply per-source token caps
    const ragCap = config.maxTokensPerSource?.["rag"];
    if (ragCap != null && docs) {
      docs = this.clipDocsToBudget(docs, ragCap);
    }

    return this.assemble({
      template:             config.template,
      variables:            config.templateVars ?? {},
      retrievedDocs:        docs,
      toolResults:          tools,
      userProfile:          profile,
      conversationSummary:  summary,
      query:                data.query ?? "",
      optimize:             (config.format ?? "markdown") === "markdown",
    });
  }

  // ----------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------

  private truncateToTokens(text: string, maxTokens: number): string {
    let enc;
    try {
      enc = get_encoding("cl100k_base");
    } catch {
      return text.slice(0, maxTokens * 4); // rough fallback
    }
    const tokens = enc.encode(text);
    if (tokens.length <= maxTokens) {
      enc.free();
      return text;
    }
    const result = new TextDecoder().decode(enc.decode(tokens.slice(0, maxTokens)));
    enc.free();
    return result;
  }

  private clipDocsToBudget(
    docs: Array<{ text: string; metadata?: Record<string, unknown> }>,
    maxTokens: number
  ): Array<{ text: string; metadata?: Record<string, unknown> }> {
    const kept: typeof docs = [];
    let used = 0;
    for (const doc of docs) {
      const t = countTokens(doc.text, this.model);
      if (used + t <= maxTokens) {
        kept.push(doc);
        used += t;
      } else {
        break;
      }
    }
    return kept;
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function runDemo(): void {
  console.log("=== Context Assembler (TypeScript) Demo ===\n");

  const budget    = new ContextBudget(16_000);
  const assembler = new ContextAssembler(budget);

  const ragDocs = [
    {
      text: "## Cancellation Policy\nYou may cancel at any time. Takes effect at end of billing period.",
      metadata: { source: "help-center/cancellation.md", score: 0.92 },
    },
    {
      text: "## Billing\nInvoices available at Settings → Billing → Invoice History.",
      metadata: { source: "help-center/billing.md", score: 0.89 },
    },
  ];

  const userProfile = {
    name:         "Alice Johnson",
    plan:         "Pro (annual)",
    member_since: "2022-03-15",
  };

  const result = assembler.assemble({
    template:            "You are a ${role} support agent. Answer using only the documents.\n",
    variables:           { role: "customer" },
    retrievedDocs:       ragDocs,
    userProfile,
    conversationSummary: "Alice asked about changing her password (resolved).",
    query:               "How do I cancel my subscription?",
  });

  console.log("Sources included :", result.sourcesIncluded);
  console.log("Sources excluded :", result.sourcesExcluded);
  console.log("Total tokens     :", result.totalTokens.toLocaleString());
  console.log("\nToken breakdown:");
  for (const [src, tok] of Object.entries(result.tokenBreakdown)) {
    console.log(`  ${src.padEnd(10)}: ${tok.toLocaleString()} tokens`);
  }
  console.log("\n--- Context (first 500 chars) ---");
  console.log(result.context.slice(0, 500));
}

// Run demo when executed directly
if (process.argv[1] && process.argv[1].endsWith("context_assembler.ts")) {
  runDemo();
}
