/**
 * Skill base class and registry for composable agent capabilities.
 *
 * TypeScript port of code/python/09-skills/skill_base.py
 *
 * A Skill bundles a tool with:
 *   - Input validation   (runs before the tool)
 *   - Output normalisation  (runs after the tool)
 *   - A fallback  (runs when the tool throws)
 *   - A prompt fragment  (injected into the agent system prompt)
 *   - Test cases  (runnable without an LLM or API key)
 *
 * See: docs/02-the-agent-loop/05-skills-composing-capabilities.md
 */

import { z } from "zod";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export type Params = Record<string, unknown>;
export type ToolOutput = Record<string, unknown>;

export type ToolFn = (params: Params) => ToolOutput | Promise<ToolOutput>;
export type ValidatorFn = (params: Params) => Params | Promise<Params>;
export type NormalizerFn = (raw: ToolOutput) => ToolOutput | Promise<ToolOutput>;
export type FallbackFn = (params: Params, error: Error) => string;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class SkillInputError extends Error {
  readonly suggestion?: string;
  readonly fixAction?: string;

  constructor(message: string, suggestion?: string, fixAction?: string) {
    super(message);
    this.name = "SkillInputError";
    this.suggestion = suggestion;
    this.fixAction = fixAction;
  }
}

export class CircularDependencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CircularDependencyError";
  }
}

export class MissingDependencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MissingDependencyError";
  }
}

// ---------------------------------------------------------------------------
// Runtime-validated result types (Zod)
// ---------------------------------------------------------------------------

export const SkillResultSchema = z.object({
  success: z.boolean(),
  data: z.record(z.unknown()).nullable().optional(),
  error: z.string().nullable().optional(),
  /** "invalid_input" | "unavailable" | "internal" */
  errorType: z.string().nullable().optional(),
  suggestion: z.string().nullable().optional(),
  executionTimeMs: z.number().int().default(0),
});

export type SkillResult = z.infer<typeof SkillResultSchema>;

export const SkillTestSchema = z.object({
  input: z.record(z.unknown()),
  expectSuccess: z.boolean().default(true),
  expectOutputContains: z.array(z.string()).nullable().optional(),
  expectFallback: z.boolean().default(false),
});

export type SkillTest = z.infer<typeof SkillTestSchema>;

export interface TestResult {
  testInput: Params;
  passed: boolean;
  reason: string;
  result?: SkillResult;
}

// ---------------------------------------------------------------------------
// OpenAI schema helper
// ---------------------------------------------------------------------------

export interface OpenAISchema {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

export interface SkillOptions {
  name: string;
  description: string;
  tool: ToolFn;
  parameters: Record<string, unknown>;
  version?: string;
  tags?: string[];
  promptFragment?: string;
  inputValidator?: ValidatorFn;
  outputNormalizer?: NormalizerFn;
  fallback?: FallbackFn;
  dependencies?: string[];
  testCases?: SkillTest[];
}

export class Skill {
  readonly name: string;
  readonly description: string;
  readonly tool: ToolFn;
  readonly parameters: Record<string, unknown>;
  readonly version: string;
  readonly tags: string[];
  readonly promptFragment?: string;
  readonly inputValidator?: ValidatorFn;
  readonly outputNormalizer?: NormalizerFn;
  readonly fallback?: FallbackFn;
  readonly dependencies: string[];
  readonly testCases: SkillTest[];

  constructor(opts: SkillOptions) {
    this.name = opts.name;
    this.description = opts.description;
    this.tool = opts.tool;
    this.parameters = opts.parameters;
    this.version = opts.version ?? "1.0.0";
    this.tags = opts.tags ?? [];
    this.promptFragment = opts.promptFragment;
    this.inputValidator = opts.inputValidator;
    this.outputNormalizer = opts.outputNormalizer;
    this.fallback = opts.fallback;
    this.dependencies = opts.dependencies ?? [];
    this.testCases = opts.testCases ?? [];
  }

  // ------------------------------------------------------------------
  // Core execution pipeline
  // ------------------------------------------------------------------

  async execute(params: Params): Promise<SkillResult> {
    const start = Date.now();

    try {
      // 1. Validate input
      let p = params;
      if (this.inputValidator) {
        p = await this.inputValidator(p);
      }

      // 2. Run tool
      let raw = await this.tool(p);

      // 3. Normalise output
      if (this.outputNormalizer) {
        raw = await this.outputNormalizer(raw);
      }

      return {
        success: true,
        data: raw,
        executionTimeMs: Date.now() - start,
      };
    } catch (err) {
      const elapsed = Date.now() - start;

      if (err instanceof SkillInputError) {
        return {
          success: false,
          error: err.message,
          errorType: "invalid_input",
          suggestion: err.suggestion,
          executionTimeMs: elapsed,
        };
      }

      const error = err instanceof Error ? err : new Error(String(err));
      if (this.fallback) {
        return {
          success: false,
          error: this.fallback(params, error),
          errorType: "unavailable",
          executionTimeMs: elapsed,
        };
      }

      throw err;
    }
  }

  // ------------------------------------------------------------------
  // Schema / prompt helpers
  // ------------------------------------------------------------------

  getOpenAISchema(): OpenAISchema {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: this.parameters,
      },
    };
  }

  getPromptFragment(): string {
    return this.promptFragment?.trim() ?? `Use ${this.name} when: ${this.description}`;
  }

  // ------------------------------------------------------------------
  // Testing
  // ------------------------------------------------------------------

  async runTests(): Promise<TestResult[]> {
    const results: TestResult[] = [];

    for (const test of this.testCases) {
      const result = await this.execute(test.input as Params);
      let passed = true;
      let reason = "";

      if (test.expectFallback) {
        if (result.success || result.errorType === "invalid_input") {
          passed = false;
          reason = `Expected fallback but got success=${result.success}, errorType=${result.errorType}`;
        }
      } else if (!test.expectSuccess) {
        if (result.success) {
          passed = false;
          reason = "Expected failure but skill reported success";
        } else if (test.expectOutputContains) {
          const combined = (result.error ?? "") + " " + (result.suggestion ?? "");
          for (const kw of test.expectOutputContains) {
            if (!combined.includes(kw)) {
              passed = false;
              reason = `Expected '${kw}' in error/suggestion`;
              break;
            }
          }
        }
      } else {
        if (!result.success) {
          passed = false;
          reason = `Expected success but got error: ${result.error}`;
        } else if (test.expectOutputContains) {
          const dataStr = JSON.stringify(result.data ?? {});
          for (const kw of test.expectOutputContains) {
            if (!dataStr.includes(kw)) {
              passed = false;
              reason = `Expected '${kw}' in output data`;
              break;
            }
          }
        }
      }

      results.push({ testInput: test.input as Params, passed, reason, result });
    }

    return results;
  }

  // ------------------------------------------------------------------
  // Validation
  // ------------------------------------------------------------------

  validate(): string[] {
    const warnings: string[] = [];

    if (!this.description) {
      warnings.push(`[${this.name}] has no description`);
    }

    const props = (this.parameters as Record<string, unknown>)["properties"] as
      | Record<string, { description?: string }>
      | undefined;

    if (!props || Object.keys(props).length === 0) {
      warnings.push(`[${this.name}] parameters has no properties defined`);
    } else {
      for (const [paramName, schema] of Object.entries(props)) {
        if (!schema.description) {
          warnings.push(`[${this.name}] parameter '${paramName}' has no description`);
        }
      }
    }

    if (!this.fallback) {
      warnings.push(
        `[${this.name}] no fallback defined — tool failures will propagate`
      );
    }

    if (!this.promptFragment) {
      warnings.push(
        `[${this.name}] no promptFragment — agent will use default wording`
      );
    }

    return warnings;
  }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export class SkillRegistry {
  private readonly skills = new Map<string, Skill>();

  // ------------------------------------------------------------------
  // Registration
  // ------------------------------------------------------------------

  register(skill: Skill): void {
    if (this.skills.has(skill.name)) {
      throw new Error(`Skill '${skill.name}' is already registered`);
    }
    this.skills.set(skill.name, skill);
    try {
      this._checkDependencies(skill);
    } catch (err) {
      this.skills.delete(skill.name);
      throw err;
    }
  }

  registerMany(skills: Skill[]): void {
    for (const skill of skills) {
      this.register(skill);
    }
  }

  // ------------------------------------------------------------------
  // Lookup
  // ------------------------------------------------------------------

  get(name: string): Skill {
    const skill = this.skills.get(name);
    if (!skill) {
      throw new Error(`Skill '${name}' is not registered`);
    }
    return skill;
  }

  findByTags(tags: string[]): Skill[] {
    const tagSet = new Set(tags);
    return [...this.skills.values()].filter(
      (s) => s.tags.some((t) => tagSet.has(t))
    );
  }

  getAllSchemas(): OpenAISchema[] {
    return [...this.skills.values()].map((s) => s.getOpenAISchema());
  }

  getCombinedPrompt(skillNames: string[]): string {
    return skillNames
      .map((name) => {
        const skill = this.get(name);
        return `### ${skill.name}\n${skill.getPromptFragment()}`;
      })
      .join("\n\n");
  }

  // ------------------------------------------------------------------
  // Execution
  // ------------------------------------------------------------------

  async execute(name: string, params: Params): Promise<SkillResult> {
    return this.get(name).execute(params);
  }

  // ------------------------------------------------------------------
  // Dependency resolution
  // ------------------------------------------------------------------

  resolveDependencies(skill: Skill): Skill[] {
    const order: string[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();

    const visit = (name: string): void => {
      if (visiting.has(name)) {
        throw new CircularDependencyError(
          `Circular dependency detected involving '${name}'`
        );
      }
      if (visited.has(name)) return;

      visiting.add(name);
      const s = this.skills.get(name);
      if (!s) {
        throw new MissingDependencyError(`Dependency '${name}' is not registered`);
      }
      for (const dep of s.dependencies) {
        visit(dep);
      }
      visiting.delete(name);
      visited.add(name);
      order.push(name);
    };

    visit(skill.name);
    return order.map((n) => this.skills.get(n)!);
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  private _checkDependencies(skill: Skill): void {
    for (const dep of skill.dependencies) {
      if (!this.skills.has(dep)) {
        throw new MissingDependencyError(
          `Skill '${skill.name}' depends on '${dep}' which is not registered`
        );
      }
    }

    const visiting = new Set<string>();
    const visited = new Set<string>();

    const visit = (name: string): void => {
      if (visiting.has(name)) {
        throw new CircularDependencyError(
          `Circular dependency detected involving '${name}'`
        );
      }
      if (visited.has(name)) return;
      visiting.add(name);
      for (const dep of (this.skills.get(name)?.dependencies ?? [])) {
        if (this.skills.has(dep)) visit(dep);
      }
      visiting.delete(name);
      visited.add(name);
    };

    visit(skill.name);
  }
}
