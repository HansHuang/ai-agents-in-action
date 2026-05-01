/**
 * Programmatic tool definition builder with Zod-powered runtime validation.
 *
 * Provides `ToolDef` and `Param` classes for constructing OpenAI-compatible
 * function-calling tool definitions and validating arguments at runtime.
 *
 * Structurally identical to code/python/03-agent-loop/tool_builder.py.
 *
 * Usage:
 *
 *   const weatherTool = new ToolDef(
 *     "get_weather",
 *     "Get current weather for a city. Returns temperature (C/F), humidity, and conditions.",
 *     [
 *       new Param("city", "string", {
 *         required: true,
 *         description: "City name with country code. Format: 'City, CC'. Example: 'Shanghai, SH'",
 *       }),
 *       new Param("units", "string", {
 *         required: false,
 *         enum: ["celsius", "fahrenheit"],
 *         description: "Temperature unit. Defaults to celsius. Example: 'celsius'",
 *       }),
 *     ],
 *   );
 *
 *   const schema = weatherTool.toOpenAISchema();
 *   weatherTool.validateArgs({ city: "Shanghai, SH" });   // OK
 *   weatherTool.validateArgs({ city: 123 });              // throws Error
 *
 * See docs/02-the-agent-loop/02-tool-design-patterns.md
 */

import { z, ZodTypeAny } from "zod";

// ---------------------------------------------------------------------------
// Valid JSON Schema primitive types
// ---------------------------------------------------------------------------

const VALID_TYPES = new Set([
  "string",
  "integer",
  "number",
  "boolean",
  "array",
  "object",
] as const);

type ParamType = (typeof VALID_TYPES extends Set<infer T> ? T : never);

// ---------------------------------------------------------------------------
// Param options
// ---------------------------------------------------------------------------

export interface ParamOptions {
  required?: boolean;
  description?: string;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  default?: unknown;
}

// ---------------------------------------------------------------------------
// Param
// ---------------------------------------------------------------------------

/**
 * A single parameter in a tool definition.
 *
 * @param name        - Parameter name (must match the function argument).
 * @param type        - JSON Schema primitive type.
 * @param options     - Optional constraints and metadata.
 */
export class Param {
  readonly name: string;
  readonly type: string;
  readonly required: boolean;
  readonly description: string;
  readonly enum?: unknown[];
  readonly minimum?: number;
  readonly maximum?: number;
  readonly default?: unknown;

  constructor(name: string, type: string, options: ParamOptions = {}) {
    if (!VALID_TYPES.has(type as ParamType)) {
      throw new Error(
        `Param '${name}': type must be one of [${[...VALID_TYPES].join(", ")}], got '${type}'`,
      );
    }
    this.name = name;
    this.type = type;
    this.required = options.required ?? true;
    this.description = options.description ?? "";
    this.enum = options.enum;
    this.minimum = options.minimum;
    this.maximum = options.maximum;
    this.default = options.default;
  }

  /** Return the JSON Schema fragment for this parameter. */
  toSchema(): Record<string, unknown> {
    const schema: Record<string, unknown> = {
      type: this.type,
      description: this.description,
    };
    if (this.enum !== undefined) schema.enum = this.enum;
    if (this.minimum !== undefined) schema.minimum = this.minimum;
    if (this.maximum !== undefined) schema.maximum = this.maximum;
    if (this.default !== undefined) schema.default = this.default;
    return schema;
  }

  /**
   * Build a Zod schema for this parameter.
   * Used internally by `ToolDef.validateArgs()` for runtime validation.
   */
  _toZod(): ZodTypeAny {
    let schema: ZodTypeAny;

    // Enum schema (union of literals)
    if (this.enum !== undefined && this.enum.length > 0) {
      if (this.enum.length === 1) {
        schema = z.literal(this.enum[0] as string | number | boolean);
      } else {
        const [first, second, ...rest] = this.enum as (string | number | boolean)[];
        schema = z.union([z.literal(first), z.literal(second), ...rest.map((v) => z.literal(v))]);
      }
    } else {
      switch (this.type) {
        case "string":
          schema = z.string();
          break;
        case "integer":
          schema = z.number().int();
          break;
        case "number":
          schema = z.number();
          break;
        case "boolean":
          schema = z.boolean();
          break;
        case "array":
          schema = z.array(z.unknown());
          break;
        case "object":
          schema = z.record(z.unknown());
          break;
        default:
          schema = z.unknown();
      }
    }

    if (this.minimum !== undefined && schema instanceof z.ZodNumber) {
      schema = schema.min(this.minimum);
    }
    if (this.maximum !== undefined && schema instanceof z.ZodNumber) {
      schema = schema.max(this.maximum);
    }

    return schema;
  }
}

// ---------------------------------------------------------------------------
// ToolDef
// ---------------------------------------------------------------------------

/**
 * An OpenAI-compatible function-calling tool definition.
 *
 * @param name        - Snake-case tool name shown to the LLM.
 * @param description - Full description including what the tool returns.
 * @param parameters  - Array of `Param` objects.
 * @param strict      - When true, the model is constrained to the exact schema.
 */
export class ToolDef {
  readonly name: string;
  readonly description: string;
  readonly parameters: Param[];
  readonly strict: boolean;

  constructor(
    name: string,
    description: string,
    parameters: Param[] = [],
    strict = false,
  ) {
    this.name = name;
    this.description = description;
    this.parameters = parameters;
    this.strict = strict;
  }

  // ------------------------------------------------------------------
  // Schema generation
  // ------------------------------------------------------------------

  /**
   * Generate the exact object expected by the OpenAI `tools` parameter.
   *
   * Returns a dict of the form:
   * ```json
   * {
   *   "type": "function",
   *   "function": {
   *     "name": "...",
   *     "description": "...",
   *     "strict": false,
   *     "parameters": {
   *       "type": "object",
   *       "properties": { ... },
   *       "required": [...],
   *       "additionalProperties": false
   *     }
   *   }
   * }
   * ```
   */
  toOpenAISchema(): Record<string, unknown> {
    const properties: Record<string, unknown> = {};
    for (const p of this.parameters) {
      properties[p.name] = p.toSchema();
    }

    const required = this.strict
      ? this.parameters.map((p) => p.name)
      : this.parameters.filter((p) => p.required).map((p) => p.name);

    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        strict: this.strict,
        parameters: {
          type: "object",
          properties,
          required,
          additionalProperties: false,
        },
      },
    };
  }

  // ------------------------------------------------------------------
  // Argument validation
  // ------------------------------------------------------------------

  /**
   * Validate `args` against this tool's parameter definitions.
   *
   * Checks required presence, types (via Zod), enums, and numeric ranges.
   *
   * @throws {Error} On the first violation, with a message of the form:
   *   "Parameter 'city' must be a string, got number (123)"
   */
  validateArgs(args: Record<string, unknown>): void {
    // --- Required presence ---
    for (const param of this.parameters) {
      if (param.required && !(param.name in args)) {
        throw new Error(`Missing required parameter: '${param.name}'`);
      }
    }

    // --- Type, enum, and range checks via Zod ---
    for (const param of this.parameters) {
      if (!(param.name in args)) continue;
      const value = args[param.name];
      const zodSchema = param._toZod();
      const result = zodSchema.safeParse(value);
      if (!result.success) {
        const actual = typeof value;
        const repr = JSON.stringify(value);
        throw new Error(
          `Parameter '${param.name}' must be a ${param.type}, got ${actual} (${repr})`,
        );
      }
    }
  }

  // ------------------------------------------------------------------
  // Deserialisation
  // ------------------------------------------------------------------

  /**
   * Create a `ToolDef` from a plain object (e.g. parsed from YAML/JSON).
   *
   * Expected format:
   * ```json
   * {
   *   "name": "get_weather",
   *   "description": "...",
   *   "strict": false,
   *   "parameters": [
   *     { "name": "city", "type": "string", "required": true, "description": "..." }
   *   ]
   * }
   * ```
   */
  static fromDict(data: Record<string, unknown>): ToolDef {
    const rawParams = (data.parameters as Record<string, unknown>[] | undefined) ?? [];
    const params: Param[] = rawParams.map((p) =>
      new Param(p.name as string, p.type as string, {
        required: (p.required as boolean | undefined) ?? true,
        description: (p.description as string | undefined) ?? "",
        enum: p.enum as unknown[] | undefined,
        minimum: p.minimum as number | undefined,
        maximum: p.maximum as number | undefined,
        default: p.default,
      }),
    );
    return new ToolDef(
      data.name as string,
      data.description as string,
      params,
      (data.strict as boolean | undefined) ?? false,
    );
  }
}
