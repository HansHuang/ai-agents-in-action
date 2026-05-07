/**
 * Prompt Assembler — dynamic prompt construction from templates, conditional
 * sections, and multi-source context injection.
 *
 * TypeScript port of code/python/05-context-assembly/prompt_assembler.py
 *
 * Same class names and method signatures (camelCase), same {variable} syntax,
 * same conditional section evaluation, same YAML template format.
 *
 * See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
 */

import { countTokens } from "./context_budget.js";

// ---------------------------------------------------------------------------
// Token helper
// ---------------------------------------------------------------------------

function truncateToTokens(text: string, maxTokens: number): string {
  const words = text.split(/\s+/);
  let result = text;
  // Binary-search style: drop words from the end until under budget
  if (countTokens(result) <= maxTokens) return result;
  let lo = 0;
  let hi = words.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const candidate = words.slice(0, mid).join(" ");
    if (countTokens(candidate) <= maxTokens) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return words.slice(0, lo - 1).join(" ") + "…";
}

// ---------------------------------------------------------------------------
// Variable regex
// ---------------------------------------------------------------------------

const VAR_RE = /\{(\w+)\}/g;

function fillTemplate(template: string, vars: Record<string, string>): [string, string | null] {
  let missingKey: string | null = null;
  const result = template.replace(/\{(\w+)\}/g, (match, key: string) => {
    if (key in vars) return String(vars[key]);
    missingKey = key;
    return match; // leave as-is on missing
  });
  return [result, missingKey];
}

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------

export class MissingVariableError extends Error {
  constructor(public readonly key: string) {
    super(`Template variable '{${key}}' not provided`);
    this.name = "MissingVariableError";
  }
}

// ---------------------------------------------------------------------------
// Condition evaluator (inline — no separate module needed)
// ---------------------------------------------------------------------------

function getNestedValue(variables: Record<string, unknown>, key: string): unknown {
  const parts = key.split(".");
  let obj: unknown = variables;
  for (const part of parts) {
    if (obj === null || obj === undefined) return undefined;
    if (typeof obj === "object") {
      obj = (obj as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }
  return obj;
}

function parseRhs(raw: string | null): unknown {
  if (raw === null || raw === undefined) return null;
  raw = raw.trim();
  try {
    return JSON.parse(raw);
  } catch {
    // Strip surrounding quotes for plain string values
    if ((raw.startsWith("'") && raw.endsWith("'")) ||
        (raw.startsWith('"') && raw.endsWith('"'))) {
      return raw.slice(1, -1);
    }
    const n = Number(raw);
    return isNaN(n) ? raw : n;
  }
}

type OpFn = (a: unknown, b: unknown) => boolean;

const OP_TABLE: Record<string, OpFn> = {
  eq:       (a, b) => a === b,
  neq:      (a, b) => a !== b,
  in:       (a, b) => Array.isArray(b) ? b.includes(a) : String(b).includes(String(a)),
  not_in:   (a, b) => Array.isArray(b) ? !b.includes(a) : !String(b).includes(String(a)),
  gt:       (a, b) => (a as number) > (b as number),
  lt:       (a, b) => (a as number) < (b as number),
  gte:      (a, b) => (a as number) >= (b as number),
  lte:      (a, b) => (a as number) <= (b as number),
  contains: (a, b) => typeof a === "string" && typeof b === "string" && a.includes(b),
  exists:   (a, _) => a !== null && a !== undefined,
};

interface ParsedAtom {
  key: string;
  op:  string;
  rhs: unknown;
}

function parseAtom(atomStr: string): ParsedAtom {
  atomStr = atomStr.trim();

  const patterns: Array<[RegExp, string]> = [
    [/^([\w.]+)\s+not\s+in\s+(.+)$/i,   "not_in"],
    [/^([\w.]+)\s+not_in\s+(.+)$/i,     "not_in"],
    [/^([\w.]+)\s+>=(.*)/,              "gte"],
    [/^([\w.]+)\s+<=(.*)/,              "lte"],
    [/^([\w.]+)\s+==(.*)/,              "eq"],
    [/^([\w.]+)\s+!=(.*)/,              "neq"],
    [/^([\w.]+)\s+>(.*)/,               "gt"],
    [/^([\w.]+)\s+<(.*)/,               "lt"],
    [/^([\w.]+)\s+contains\s+(.*)/i,    "contains"],
    [/^([\w.]+)\s+exists$/i,            "exists"],
    [/^([\w.]+)\s+in\s+(.+)$/i,         "in"],
  ];

  for (const [re, opName] of patterns) {
    const m = atomStr.match(re);
    if (m) {
      return {
        key: m[1],
        op:  opName,
        rhs: opName === "exists" ? null : parseRhs(m[2] ?? null),
      };
    }
  }
  throw new Error(`Cannot parse condition atom: ${JSON.stringify(atomStr)}`);
}

function evaluateAtom(atom: ParsedAtom, variables: Record<string, unknown>): boolean {
  const left = getNestedValue(variables, atom.key);
  const fn   = OP_TABLE[atom.op];
  if (!fn) throw new Error(`Unknown operator: ${atom.op}`);
  return fn(left, atom.rhs);
}

export function evaluateCondition(condition: string, variables: Record<string, unknown>): boolean {
  const orBranches = condition.split(/\bOR\b/i);
  for (const branch of orBranches) {
    const andAtoms = branch.split(/\bAND\b/i);
    try {
      const allTrue = andAtoms.every(a => evaluateAtom(parseAtom(a.trim()), variables));
      if (allTrue) return true;
    } catch {
      // Malformed atom — treat as false
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

export interface PromptSection {
  name:      string;
  content:   string;
  condition: (variables: Record<string, unknown>) => boolean;
  shouldInclude(variables: Record<string, unknown>): boolean;
}

function makeSection(
  name:      string,
  content:   string,
  condition: (v: Record<string, unknown>) => boolean,
): PromptSection {
  return {
    name,
    content,
    condition,
    shouldInclude(variables) {
      try { return this.condition(variables); } catch { return false; }
    },
  };
}

export interface ContextSource {
  name:      string;
  formatter: (data: unknown) => string;
  priority:  number;
  maxTokens: number | null;
}

// ---------------------------------------------------------------------------
// Built-in formatters
// ---------------------------------------------------------------------------

export interface RagDocument {
  text:     string;
  score?:   number;
  metadata?: Record<string, string>;
}

export function formatRagResults(documents: RagDocument[]): string {
  if (!documents.length) return "(no documents retrieved)";
  return documents.map((doc, i) => {
    const source   = doc.metadata?.["source"] ?? "unknown";
    const score    = (doc.score ?? 0) * 100;
    return `[${i + 1}] Source: ${source} (relevance: ${score.toFixed(0)}%)\n${doc.text}`;
  }).join("\n\n---\n\n");
}

export interface UserProfile {
  name?:          string;
  plan?:          string;
  location?:      string;
  preferences?:   string;
  memberSince?:   string;
  recentOrders?:  number;
  openTickets?:   number;
}

export function formatUserProfile(profile: UserProfile): string {
  const lines: string[] = [];
  const map: Array<[keyof UserProfile, string]> = [
    ["name",         "Name"],
    ["plan",         "Plan"],
    ["location",     "Location"],
    ["preferences",  "Preferences"],
    ["memberSince",  "Member since"],
    ["recentOrders", "Recent orders"],
    ["openTickets",  "Open tickets"],
  ];
  for (const [key, label] of map) {
    if (profile[key] !== undefined) lines.push(`${label}: ${profile[key]}`);
  }
  return lines.length ? lines.join("\n") : "(no profile data)";
}

export interface ToolResult {
  toolName: string;
  success:  boolean;
  summary:  string;
}

export function formatToolResults(results: ToolResult[]): string {
  if (!results.length) return "(no tool results)";
  return results.map(r => `${r.success ? "✓" : "✗"} ${r.toolName}: ${r.summary}`).join("\n");
}

export function formatConversationSummary(summary: string): string {
  return summary ? `Previous conversation summary:\n${summary}` : "(no conversation history)";
}

export function formatBusinessRules(rules: string[]): string {
  return rules.length ? rules.map(r => `- ${r}`).join("\n") : "(no business rules)";
}

// ---------------------------------------------------------------------------
// PromptAssembler
// ---------------------------------------------------------------------------

export class PromptAssembler {
  readonly baseTemplates: Map<string, string> = new Map();
  readonly sections:      Map<string, PromptSection> = new Map();
  readonly sourceFormatters: Map<string, ContextSource> = new Map();

  /** Register a base template with {placeholder} variables. */
  registerTemplate(name: string, template: string): void {
    this.baseTemplates.set(name, template);
  }

  /**
   * Register a conditional section.
   * @param condition - Function receiving the variables object; returns true
   *                   when the section should be included.
   */
  registerSection(
    name:      string,
    content:   string,
    condition: (variables: Record<string, unknown>) => boolean,
  ): void {
    this.sections.set(name, makeSection(name, content, condition));
  }

  /**
   * Register a context source formatter.
   * @param priority  - Higher = included first and survives budget cuts longest.
   * @param maxTokens - Truncate this source to this many tokens before injection.
   */
  registerSourceFormatter(
    name:      string,
    formatter: (data: unknown) => string,
    priority:  number = 0,
    maxTokens: number | null = null,
  ): void {
    this.sourceFormatters.set(name, { name, formatter, priority, maxTokens });
  }

  /**
   * Assemble a complete prompt.
   *
   * Steps:
   * 1. Look up the base template.
   * 2. Evaluate conditional sections.
   * 3. Format context sources.
   * 4. Sort by priority (highest first).
   * 5. Build {context} and {sections} blocks.
   * 6. Fill the template.
   * 7. Log metadata.
   */
  assemble(
    templateName:   string,
    variables:      Record<string, unknown>,
    contextSources: Record<string, unknown> = {},
  ): string {
    const template = this.baseTemplates.get(templateName);
    if (template === undefined) {
      throw new Error(`Template '${templateName}' not registered`);
    }

    // 2. Evaluate conditional sections
    const activeSections: PromptSection[] = [];
    for (const section of this.sections.values()) {
      if (section.shouldInclude(variables)) activeSections.push(section);
    }

    // 3+4. Format and sort context sources
    const formatted: Array<{ priority: number; name: string; text: string }> = [];
    for (const [srcName, data] of Object.entries(contextSources)) {
      const source = this.sourceFormatters.get(srcName);
      if (!source) continue;
      let text = source.formatter(data);
      if (source.maxTokens !== null && countTokens(text) > source.maxTokens) {
        text = truncateToTokens(text, source.maxTokens);
      }
      formatted.push({ priority: source.priority, name: srcName, text });
    }
    formatted.sort((a, b) => b.priority - a.priority);

    // 5. Build blocks
    const contextBlock  = formatted.map(f => `## ${f.name}\n${f.text}`).join("\n\n");
    const sectionsBlock = activeSections.map(s => s.content).join("\n\n");

    // 6. Fill template
    const fillVars: Record<string, string> = {};
    for (const [k, v] of Object.entries(variables)) fillVars[k] = String(v);
    fillVars["context"]  = contextBlock;
    fillVars["sections"] = sectionsBlock;

    let result = template.replace(/\{(\w+)\}/g, (match, key: string) => {
      if (key in fillVars) return fillVars[key];
      throw new MissingVariableError(key);
    });

    // Append sections / context if template has no placeholder for them
    if (!template.includes("{sections}") && sectionsBlock) {
      result = result.trimEnd() + "\n\n" + sectionsBlock;
    }
    if (!template.includes("{context}") && contextBlock) {
      result = result.trimEnd() + "\n\n" + contextBlock;
    }

    // 7. Log
    const tokenCount = countTokens(result);
    console.debug(
      `Assembled prompt | template=${templateName} | ` +
      `sections=[${activeSections.map(s => s.name).join(", ")}] | ` +
      `sources=[${formatted.map(f => f.name).join(", ")}] | ` +
      `tokens=${tokenCount}`,
    );

    return result;
  }

  /**
   * Assemble a prompt while enforcing a total token budget.
   * Low-priority context sources are dropped first.
   */
  assembleWithBudget(
    templateName:   string,
    variables:      Record<string, unknown>,
    contextSources: Record<string, unknown> = {},
    maxTokens:      number = 100_000,
  ): string {
    let result = this.assemble(templateName, variables, contextSources);
    if (countTokens(result) <= maxTokens) return result;

    const getPriority = (name: string): number =>
      this.sourceFormatters.get(name)?.priority ?? 0;

    const dropOrder = Object.keys(contextSources).sort(
      (a, b) => getPriority(a) - getPriority(b),   // ascending — drop lowest first
    );

    const remaining = { ...contextSources };
    for (const name of dropOrder) {
      delete remaining[name];
      result = this.assemble(templateName, variables, remaining);
      if (countTokens(result) <= maxTokens) return result;
    }
    return result;
  }

  /** Return all {variable} names used in the template, in order of appearance. */
  getAvailableVariables(templateName: string): string[] {
    const template = this.baseTemplates.get(templateName);
    if (template === undefined) throw new Error(`Template '${templateName}' not registered`);
    const seen = new Set<string>();
    const result: string[] = [];
    for (const m of template.matchAll(VAR_RE)) {
      const name = m[1];
      if (!seen.has(name)) { seen.add(name); result.push(name); }
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function demo(): void {
  const assembler = new PromptAssembler();

  assembler.registerTemplate("billing", [
    "You are a billing support specialist for {company}.",
    "",
    "Responsibilities:",
    "- Handle payment inquiries and invoice questions.",
    "- Process refunds according to policy.",
    "{sections}",
    "",
    "{context}",
  ].join("\n"));

  assembler.registerSection(
    "premium_user",
    "Additional instructions (premium customer):\n- Provide priority service.",
    v => v["plan"] === "premium",
  );
  assembler.registerSection(
    "gdpr_required",
    "Additional instructions (EU customer — GDPR):\n- Include data processing notice.",
    v => ["DE", "FR", "ES", "IT"].includes(v["country"] as string),
  );

  assembler.registerSourceFormatter("rag", formatRagResults as (d: unknown) => string, 3, 4000);
  assembler.registerSourceFormatter("user_profile", formatUserProfile as (d: unknown) => string, 2, 500);
  assembler.registerSourceFormatter("conversation_summary", formatConversationSummary as (d: unknown) => string, 1, 1000);

  const ragDocs: RagDocument[] = [
    { text: "Refund policy: full refund within 30 days.", score: 0.92, metadata: { source: "refund-policy.md" } },
  ];

  const sep = "\n" + "─".repeat(60);

  // Scenario 1
  console.log(`${sep}\nScenario 1: Free US user`);
  const p1 = assembler.assemble("billing", { company: "Acme Corp", plan: "free", country: "US" }, { rag: ragDocs });
  console.log(p1);
  console.log(`\n[${countTokens(p1)} tokens]`);

  // Scenario 2
  console.log(`${sep}\nScenario 2: Premium US user`);
  const p2 = assembler.assemble(
    "billing",
    { company: "Acme Corp", plan: "premium", country: "US" },
    { rag: ragDocs, user_profile: { name: "Alice", plan: "Premium", location: "San Francisco" } },
  );
  console.log(p2);
  console.log(`\n[${countTokens(p2)} tokens]`);

  // Variables
  console.log(`\nTemplate variables in 'billing':`, assembler.getAvailableVariables("billing"));
}

// Uncomment to run: demo();
