import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ToolRegistry } from "./tool_registry.js";
import { ToolDef, Param } from "./tool_builder.js";

// ---------------------------------------------------------------------------
// Tool Registry
// ---------------------------------------------------------------------------

describe("ToolRegistry", () => {
  it("registers a tool and includes in schemas", () => {
    const reg = new ToolRegistry();
    reg.register({
      name: "greet",
      description: "Greet someone",
      parameters: { name: { type: "string", description: "Name", required: true } },
      handler: async (args) => `Hello, ${args.name}!`,
    });
    const schemas = reg.getOpenAISchemas();
    assert.ok(schemas.some((s) => s.function.name === "greet"));
  });

  it("returns tool_not_found error for unknown tool", async () => {
    const reg = new ToolRegistry();
    const result = await reg.executeTool({ id: "1", type: "function", function: { name: "nonexistent", arguments: "{}" } });
    const parsed = JSON.parse(result.content);
    assert.equal(parsed.error, "tool_not_found");
  });

  it("executes a tool successfully", async () => {
    const reg = new ToolRegistry();
    reg.register({
      name: "add",
      description: "Add numbers",
      parameters: {
        a: { type: "integer", description: "First", required: true },
        b: { type: "integer", description: "Second", required: true },
      },
      handler: async (args) => ({ result: Number(args.a) + Number(args.b) }),
    });
    const result = await reg.executeTool({ id: "1", type: "function", function: { name: "add", arguments: JSON.stringify({ a: 3, b: 4 }) } });
    const parsed = JSON.parse(result.content);
    assert.deepEqual(parsed, { result: 7 });
  });

  it("generates OpenAI-compatible tool schemas", () => {
    const reg = new ToolRegistry();
    reg.register({
      name: "echo",
      description: "Echo input",
      parameters: { text: { type: "string", description: "Text", required: true } },
      handler: async (args) => args.text,
    });
    const schemas = reg.getOpenAISchemas();
    assert.ok(Array.isArray(schemas));
    assert.equal(schemas[0].type, "function");
    assert.equal(schemas[0].function.name, "echo");
  });

  it("lists registered tool names", () => {
    const reg = new ToolRegistry();
    reg.register({ name: "a", description: "A", parameters: {}, handler: async () => "a" });
    reg.register({ name: "b", description: "B", parameters: {}, handler: async () => "b" });
    const names = reg.list();
    assert.ok(names.includes("a"));
    assert.ok(names.includes("b"));
  });
});

// ---------------------------------------------------------------------------
// Plan Schema
// ---------------------------------------------------------------------------

describe("PlanSchema", async () => {
  const planSchema = await import("./plan_schema.js");

  it("PlanSchema module exports expected types", () => {
    // Basic smoke test — confirm module imports without error
    assert.ok(typeof planSchema === "object");
  });
});

// ---------------------------------------------------------------------------
// Tool Builder (ToolDef/Param)
// ---------------------------------------------------------------------------

describe("ToolBuilder", () => {
  it("creates a Param with correct properties", () => {
    const param = new Param("query", "string", { description: "Search query", required: true });
    assert.equal(param.name, "query");
    assert.equal(param.type, "string");
    assert.equal(param.required, true);
  });

  it("ToolDef can be constructed and generates schema", () => {
    const def = new ToolDef("search", "Search the web", [
      new Param("query", "string", { description: "Query", required: true }),
    ]);
    assert.equal(def.name, "search");
    const schema = def.toOpenAISchema() as { type: string; function: { name: string } };
    assert.equal(schema.type, "function");
    assert.equal(schema.function.name, "search");
  });
});
