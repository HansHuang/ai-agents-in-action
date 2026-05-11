/**
 * Tool argument validator — validates tool arguments against JSON-schema-like
 * parameter definitions before execution.
 * See: docs/02-the-agent-loop/02-tool-design-patterns.md
 */

import { ToolParameter, InvalidArgsError } from "./tool_registry.js";

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  coerced: Record<string, unknown>;
}

/** Validate and coerce arguments against parameter definitions. */
export function validateArgs(
  args: Record<string, unknown>,
  params: Record<string, ToolParameter>
): ValidationResult {
  const errors: string[] = [];
  const coerced: Record<string, unknown> = { ...args };

  // Check required parameters
  for (const [key, param] of Object.entries(params)) {
    if (param.required !== false && !(key in args)) {
      if (param.default !== undefined) {
        coerced[key] = param.default;
      } else {
        errors.push(`Missing required parameter: ${key}`);
      }
    }
  }

  // Type checks and enum validation
  for (const [key, value] of Object.entries(coerced)) {
    const param = params[key];
    if (!param) continue;

    if (param.enum && !param.enum.includes(String(value))) {
      errors.push(
        `Parameter '${key}' must be one of: ${param.enum.join(", ")}. Got: ${String(value)}`
      );
    }

    if (param.type === "number" || param.type === "integer") {
      const n = Number(value);
      if (isNaN(n)) {
        errors.push(`Parameter '${key}' must be a number. Got: ${String(value)}`);
      } else {
        coerced[key] = param.type === "integer" ? Math.trunc(n) : n;
      }
    }

    if (param.type === "boolean" && typeof value !== "boolean") {
      coerced[key] = value === "true" || value === 1;
    }
  }

  return { valid: errors.length === 0, errors, coerced };
}

/** Validate and throw InvalidArgsError if validation fails. */
export function assertValidArgs(
  args: Record<string, unknown>,
  params: Record<string, ToolParameter>
): Record<string, unknown> {
  const result = validateArgs(args, params);
  if (!result.valid) {
    throw new InvalidArgsError(result.errors.join("; "));
  }
  return result.coerced;
}
