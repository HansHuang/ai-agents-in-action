/**
 * Tests for nodejs/10-mcp-server — MCPMarketplace, SimpleMCPServer
 *
 * No network calls — all tests are pure logic.
 * Run: node --import tsx/esm --test test_mcp.ts
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { MCPMarketplace, NPM_CATALOG, ServerInfo } from "./mcp_marketplace.js";

// ---------------------------------------------------------------------------
// MCPMarketplace
// ---------------------------------------------------------------------------

describe("MCPMarketplace", () => {
  const market = new MCPMarketplace();

  it("catalog has at least 5 servers", () => {
    assert.ok(NPM_CATALOG.length >= 5, `Expected >= 5 servers, got ${NPM_CATALOG.length}`);
  });

  it("search finds servers by keyword", () => {
    const results = market.search("github");
    assert.ok(results.length >= 1, "Expected at least one result for 'github'");
    assert.ok(results.some((s) => s.name === "github"), "Expected 'github' server in results");
  });

  it("search finds servers by category", () => {
    const results = market.search("database");
    assert.ok(results.length >= 1);
  });

  it("listCategories returns sorted unique categories", () => {
    const cats = market.listCategories();
    assert.ok(Array.isArray(cats));
    assert.ok(cats.length > 0);
    // Check sorted
    for (let i = 1; i < cats.length; i++) {
      assert.ok(cats[i - 1] <= cats[i], "Categories should be sorted");
    }
    // Check unique
    const unique = new Set(cats);
    assert.equal(unique.size, cats.length, "Categories should be unique");
  });

  it("byCategory returns only servers in that category", () => {
    const servers = market.byCategory("files");
    for (const s of servers) {
      assert.ok(s.categories.includes("files"), `Server ${s.name} should have 'files' category`);
    }
  });

  it("getDetails returns server by name", () => {
    const fs = market.getDetails("filesystem");
    assert.ok(fs !== undefined, "Expected 'filesystem' server");
    assert.equal(fs!.name, "filesystem");
    assert.ok(fs!.installCommand.length > 0);
    assert.ok(fs!.runCommand.length > 0);
  });

  it("getDetails returns undefined for unknown server", () => {
    const result = market.getDetails("nonexistent-server-xyz");
    assert.equal(result, undefined);
  });

  it("topRated returns N servers sorted by rating desc", () => {
    const top = market.topRated(3);
    assert.ok(top.length <= 3);
    for (let i = 1; i < top.length; i++) {
      assert.ok((top[i - 1].rating ?? 0) >= (top[i].rating ?? 0), "Should be sorted desc");
    }
  });

  it("search returns empty array for no matches", () => {
    const results = market.search("xyzabc_no_match_zyx");
    assert.deepEqual(results, []);
  });
});

// ---------------------------------------------------------------------------
// SimpleMCPServer (tool/resource registration only — no stdio startup)
// ---------------------------------------------------------------------------

describe("SimpleMCPServer", async () => {
  const { SimpleMCPServer } = await import("./simple_mcp_server.js");

  it("can register a tool and resource without errors", () => {
    const server = new SimpleMCPServer("test-server");
    assert.doesNotThrow(() => {
      server.tool(
        "ping",
        "Return pong",
        { type: "object", properties: { message: { type: "string" } }, required: ["message"] },
        async (p) => `pong: ${p.message}`
      );
    });
  });

  it("summary includes registered tool names", () => {
    const server = new SimpleMCPServer("test-summary");
    server.tool("my_tool", "A test tool", { type: "object", properties: {}, required: [] }, async () => "ok");
    server.resource("info://test", "Test resource", async () => "data");
    const summary = server.summary();
    assert.ok(summary.includes("my_tool"), `Expected summary to include 'my_tool':\n${summary}`);
    assert.ok(summary.includes("info://test"), `Expected summary to include resource URI:\n${summary}`);
  });

  it("supports string, number, boolean parameter types", () => {
    const server = new SimpleMCPServer("type-test");
    assert.doesNotThrow(() => {
      server.tool(
        "typed",
        "Typed params",
        {
          type: "object",
          properties: {
            name: { type: "string" },
            count: { type: "integer" },
            ratio: { type: "number" },
            flag: { type: "boolean" },
          },
          required: ["name"],
        },
        async (p) => JSON.stringify(p)
      );
    });
  });
});
