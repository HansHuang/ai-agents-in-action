/**
 * Condition Engine — evaluate DSL conditions for dynamic prompt sections.
 *
 * Supports a simple, safe condition DSL for including or excluding prompt
 * sections based on runtime variables. No eval() is used.
 *
 * Simple conditions:
 *   "plan == 'premium'"
 *   "sentiment_score > 0.7"
 *   "conversation_history exists"
 *
 * Compound conditions:
 *   "plan == 'premium' AND country == 'US'"
 *   "country == 'DE' OR country == 'FR'"
 *
 * See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
 */

export type ContextVars = Record<string, unknown>;

// ---------------------------------------------------------------------------
// Token types
// ---------------------------------------------------------------------------

type Operator = "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "contains" | "exists" | "not_exists";

interface SimpleCondition {
  kind: "simple";
  variable: string;
  operator: Operator;
  value?: unknown;
}

interface CompoundCondition {
  kind: "compound";
  left: Condition;
  logic: "AND" | "OR";
  right: Condition;
}

type Condition = SimpleCondition | CompoundCondition;

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

function parseValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (/^'.*'$/.test(trimmed)) return trimmed.slice(1, -1);
  if (/^".*"$/.test(trimmed)) return trimmed.slice(1, -1);
  if (/^\[.*\]$/.test(trimmed)) {
    return trimmed
      .slice(1, -1)
      .split(",")
      .map((s) => s.trim().replace(/^['"]|['"]$/g, ""));
  }
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  const num = Number(trimmed);
  if (!isNaN(num)) return num;
  return trimmed;
}

function parseSimple(expr: string): SimpleCondition {
  const existsMatch = expr.match(/^(\S+)\s+(exists|not_exists)$/i);
  if (existsMatch) {
    return {
      kind: "simple",
      variable: existsMatch[1],
      operator: existsMatch[2].toLowerCase() as Operator,
    };
  }

  const inMatch = expr.match(/^(\S+)\s+in\s+(\[.+\])/i);
  if (inMatch) {
    return {
      kind: "simple",
      variable: inMatch[1],
      operator: "in",
      value: parseValue(inMatch[2]),
    };
  }

  const containsMatch = expr.match(/^(\S+)\s+contains\s+(.+)$/i);
  if (containsMatch) {
    return {
      kind: "simple",
      variable: containsMatch[1],
      operator: "contains",
      value: parseValue(containsMatch[2]),
    };
  }

  const opMatch = expr.match(/^(\S+)\s*(==|!=|>=|<=|>|<)\s*(.+)$/);
  if (opMatch) {
    return {
      kind: "simple",
      variable: opMatch[1],
      operator: opMatch[2] as Operator,
      value: parseValue(opMatch[3]),
    };
  }

  throw new Error(`Cannot parse condition: "${expr}"`);
}

function parseCondition(expr: string): Condition {
  const andIdx = expr.toUpperCase().indexOf(" AND ");
  const orIdx = expr.toUpperCase().indexOf(" OR ");

  if (andIdx !== -1) {
    return {
      kind: "compound",
      left: parseCondition(expr.slice(0, andIdx).trim()),
      logic: "AND",
      right: parseCondition(expr.slice(andIdx + 5).trim()),
    };
  }
  if (orIdx !== -1) {
    return {
      kind: "compound",
      left: parseCondition(expr.slice(0, orIdx).trim()),
      logic: "OR",
      right: parseCondition(expr.slice(orIdx + 4).trim()),
    };
  }
  return parseSimple(expr.trim());
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

function resolve(variable: string, vars: ContextVars): unknown {
  const parts = variable.split(".");
  let cur: unknown = vars;
  for (const p of parts) {
    if (cur === undefined || cur === null) return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

function evalSimple(cond: SimpleCondition, vars: ContextVars): boolean {
  const val = resolve(cond.variable, vars);
  switch (cond.operator) {
    case "exists": return val !== undefined && val !== null;
    case "not_exists": return val === undefined || val === null;
    case "==": return val == cond.value;
    case "!=": return val != cond.value;
    case ">": return typeof val === "number" && val > (cond.value as number);
    case ">=": return typeof val === "number" && val >= (cond.value as number);
    case "<": return typeof val === "number" && val < (cond.value as number);
    case "<=": return typeof val === "number" && val <= (cond.value as number);
    case "in": return Array.isArray(cond.value) && cond.value.includes(val);
    case "contains":
      return typeof val === "string" && val.includes(String(cond.value));
    default: return false;
  }
}

function evalCondition(cond: Condition, vars: ContextVars): boolean {
  if (cond.kind === "simple") return evalSimple(cond, vars);
  const left = evalCondition(cond.left, vars);
  if (cond.logic === "AND") return left && evalCondition(cond.right, vars);
  return left || evalCondition(cond.right, vars);
}

/**
 * Evaluate a condition string against runtime context variables.
 * Returns true if the condition is satisfied.
 */
export function evaluateCondition(expr: string, vars: ContextVars): boolean {
  const cond = parseCondition(expr.trim());
  return evalCondition(cond, vars);
}

// Demo
function main(): void {
  const vars: ContextVars = {
    plan: "premium",
    country: "US",
    sentiment_score: 0.85,
    user: { email: "alice@enterprise.com" },
    conversation_history: ["Hello", "Hi there"],
  };

  const tests = [
    "plan == 'premium'",
    "plan == 'free'",
    "sentiment_score > 0.7",
    "country in ['US', 'CA']",
    "user.email contains '@enterprise'",
    "conversation_history exists",
    "plan == 'premium' AND country == 'US'",
    "plan == 'free' OR country == 'US'",
  ];

  console.log("Condition Engine Demo:");
  for (const t of tests) {
    console.log(`  "${t}" → ${evaluateCondition(t, vars)}`);
  }
}

main();
