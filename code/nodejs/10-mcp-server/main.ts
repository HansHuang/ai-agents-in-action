/**
 * Entry point for nodejs/10-mcp-server demos.
 *
 * Demonstrates:
 *   1. MCPMarketplace — catalog discovery and search
 *   2. SimpleMCPServer — creating an MCP server (shows config, does NOT start stdio)
 *
 * See: docs/05-the-tool-ecosystem/04-mcp-protocol.md
 */

import { MCPMarketplace } from "./mcp_marketplace.js";
import { SimpleMCPServer } from "./simple_mcp_server.js";

// ---------------------------------------------------------------------------
// Demo 1 — Marketplace
// ---------------------------------------------------------------------------

function demoMarketplace(): void {
  console.log("=== MCP Marketplace Demo ===\n");

  const market = new MCPMarketplace();

  market.printCatalog();

  console.log("  Categories available:", market.listCategories().join(", "));

  console.log("\n  Search results for 'database':");
  for (const s of market.search("database")) {
    console.log(`    - ${s.name}: ${s.description}`);
  }

  console.log("\n  Top-rated servers:");
  for (const s of market.topRated(3)) {
    console.log(`    ${s.rating?.toFixed(1)} — ${s.name}: ${s.description}`);
  }

  const fs = market.getDetails("filesystem");
  if (fs) {
    console.log(`\n  Details for '${fs.name}':`);
    console.log(`    Install: ${fs.installCommand}`);
    console.log(`    Run: ${fs.runCommand.join(" ")}`);
    console.log(`    Tools: ${fs.toolCount}`);
  }
}

// ---------------------------------------------------------------------------
// Demo 2 — SimpleMCPServer (configuration only, no stdio startup)
// ---------------------------------------------------------------------------

function demoServerConfig(): void {
  console.log("\n=== SimpleMCPServer Configuration Demo ===\n");

  const server = new SimpleMCPServer("demo-tools", "1.0.0");

  server.tool(
    "add",
    "Add two numbers together",
    {
      type: "object",
      properties: {
        a: { type: "number", description: "First number" },
        b: { type: "number", description: "Second number" },
      },
      required: ["a", "b"],
    },
    async (params) => {
      const a = Number(params.a ?? 0);
      const b = Number(params.b ?? 0);
      return String(a + b);
    }
  );

  server.tool(
    "echo",
    "Echo a message back",
    {
      type: "object",
      properties: {
        message: { type: "string", description: "Message to echo" },
      },
      required: ["message"],
    },
    async (params) => String(params.message ?? "")
  );

  server.resource("info://server", "Server information", async () =>
    JSON.stringify({ name: "demo-tools", version: "1.0.0", toolCount: 2 })
  );

  console.log("  Registered server components:");
  console.log(server.summary());
  console.log("\n  Run with: import.meta.url check or spawn as subprocess.");
  console.log("  Start stdio server: server.run()");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main(): void {
  demoMarketplace();
  demoServerConfig();
  console.log("\n[Demo complete — no LLM or network calls required]");
}

main();
