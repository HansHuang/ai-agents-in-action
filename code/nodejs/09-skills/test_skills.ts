/**
 * Tests for nodejs/09-skills — Skill, SkillRegistry, SkilledAgent
 *
 * No LLM calls — all external dependencies are mocked.
 * Run: node --import tsx/esm --test test_skills.ts
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  Skill,
  SkillRegistry,
  SkillInputError,
  CircularDependencyError,
  MissingDependencyError,
} from "./skill_base.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeEchoSkill(name = "echo", deps: string[] = []): Skill {
  return new Skill({
    name,
    description: "Echoes input",
    tool: async (p) => ({ echo: p.text }),
    parameters: {
      type: "object",
      properties: { text: { type: "string" } },
      required: ["text"],
    },
    promptFragment: `Use ${name} to echo text`,
    fallback: (_p, e) => `echo failed: ${e.message}`,
    dependencies: deps,
  });
}

function makeFailingSkill(name = "failing"): Skill {
  return new Skill({
    name,
    description: "Always fails",
    tool: async () => { throw new Error("tool error"); },
    parameters: { type: "object", properties: {}, required: [] },
    fallback: (_p, e) => `fallback: ${e.message}`,
  });
}

function makeValidatedSkill(): Skill {
  return new Skill({
    name: "validated",
    description: "Input validated skill",
    tool: async (p) => ({ result: String(p.value) }),
    parameters: { type: "object", properties: { value: { type: "string" } }, required: ["value"] },
    inputValidator: async (p) => {
      if (!p.value) throw new SkillInputError("value is required", "provide a non-empty value");
      return p;
    },
  });
}

// ---------------------------------------------------------------------------
// Skill Base Tests
// ---------------------------------------------------------------------------

describe("Skill execute", () => {
  it("happy path returns success result", async () => {
    const skill = makeEchoSkill();
    const result = await skill.execute({ text: "hello" });
    assert.equal(result.success, true);
    assert.deepEqual(result.data, { echo: "hello" });
    assert.ok(typeof result.executionTimeMs === "number");
  });

  it("uses fallback on tool failure", async () => {
    const skill = makeFailingSkill();
    const result = await skill.execute({});
    assert.equal(result.success, false);
    assert.ok(result.error?.includes("tool error"));
    assert.equal(result.errorType, "unavailable");
  });

  it("raises without fallback when tool fails", async () => {
    const skill = new Skill({
      name: "no-fallback",
      description: "No fallback",
      tool: async () => { throw new Error("boom"); },
      parameters: { type: "object", properties: {}, required: [] },
    });
    await assert.rejects(() => skill.execute({}), /boom/);
  });

  it("input validator blocks bad input", async () => {
    const skill = makeValidatedSkill();
    const result = await skill.execute({ value: "" });
    assert.equal(result.success, false);
    assert.equal(result.errorType, "invalid_input");
    assert.ok(result.suggestion?.includes("non-empty"));
  });

  it("output normalizer transforms result", async () => {
    const skill = new Skill({
      name: "normalizer-skill",
      description: "Normalizes output",
      tool: async () => ({ raw: "value" }),
      parameters: { type: "object", properties: {}, required: [] },
      outputNormalizer: async (data) => ({ ...data as object, normalized: true }),
    });
    const result = await skill.execute({});
    assert.equal(result.success, true);
    assert.ok((result.data as { normalized: boolean }).normalized);
  });
});

// ---------------------------------------------------------------------------
// SkillRegistry Tests
// ---------------------------------------------------------------------------

describe("SkillRegistry", () => {
  it("registers and retrieves a skill", () => {
    const reg = new SkillRegistry();
    reg.register(makeEchoSkill());
    const skill = reg.get("echo");
    assert.equal(skill.name, "echo");
  });

  it("rejects duplicate skill names", () => {
    const reg = new SkillRegistry();
    reg.register(makeEchoSkill());
    assert.throws(() => reg.register(makeEchoSkill()), /already registered/);
  });

  it("resolves dependencies in order", () => {
    const reg = new SkillRegistry();
    const base = makeEchoSkill("base");
    const dep = makeEchoSkill("dep", ["base"]);
    reg.register(base);
    reg.register(dep);
    const resolved = reg.resolveDependencies(dep);
    const names = resolved.map((s) => s.name);
    assert.ok(names.indexOf("base") < names.indexOf("dep"));
  });

  it("throws on missing dependency during registration", () => {
    const reg = new SkillRegistry();
    const a = new Skill({
      name: "a-dep-missing",
      description: "a",
      tool: async () => ({}),
      parameters: { type: "object", properties: {}, required: [] },
      dependencies: ["nonexistent"],
    });
    assert.throws(() => reg.register(a), /not registered|missing/i);
  });

  it("finds skills by tags", () => {
    const reg = new SkillRegistry();
    const tagged = new Skill({
      name: "tagged",
      description: "Tagged skill",
      tool: async () => ({}),
      parameters: { type: "object", properties: {}, required: [] },
      tags: ["weather", "tools"],
    });
    reg.register(tagged);
    const found = reg.findByTags(["weather"]);
    assert.equal(found.length, 1);
    assert.equal(found[0].name, "tagged");
  });
});

// ---------------------------------------------------------------------------
// OpenAI Schema generation
// ---------------------------------------------------------------------------

describe("Skill.getOpenAISchema", () => {
  it("produces correct function schema structure", () => {
    const skill = makeEchoSkill();
    const schema = skill.getOpenAISchema();
    assert.equal(schema.type, "function");
    assert.equal(schema.function.name, "echo");
    assert.ok(schema.function.description.length > 0);
    assert.ok("properties" in schema.function.parameters);
  });
});

// ---------------------------------------------------------------------------
// SkillAgent prompt building (no LLM required)
// ---------------------------------------------------------------------------

describe("SkilledAgent.buildSystemPrompt", async () => {
  const { SkilledAgent } = await import("./skilled_agent.js");
  // Provide dummy API key so OpenAI client init doesn't throw
  process.env.OPENAI_API_KEY ??= "test-key-not-used";

  it("includes skill fragments in system prompt", () => {
    const reg = new SkillRegistry();
    reg.register(makeEchoSkill());
    const agent = new SkilledAgent(reg, "gpt-4o-mini");
    agent.loadSkills(["echo"]);
    const prompt = agent.buildSystemPrompt();
    assert.ok(prompt.includes("echo"));
  });

  it("returns base prompt with no skills loaded", () => {
    const reg = new SkillRegistry();
    const agent = new SkilledAgent(reg, "gpt-4o-mini");
    const prompt = agent.buildSystemPrompt();
    assert.ok(prompt.includes("helpful assistant"));
  });
});
