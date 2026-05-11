/**
 * SimpleMCPServer — build MCP servers with minimal boilerplate.
 *
 * Usage:
 *   const server = new SimpleMCPServer("my-tools");
 *
 *   server.tool("add", "Add two numbers", {
 *     type: "object",
 *     properties: { a: { type: "number" }, b: { type: "number" } },
 *     required: ["a", "b"]
 *   }, async (params) => {
 *     return String(params.a + params.b);
 *   });
 *
 *   server.resource("config://settings", "Server settings", async () => {
 *     return JSON.stringify({ version: "1.0" });
 *   });
 *
 *   await server.run();
 *
 * See: docs/05-the-tool-ecosystem/04-mcp-protocol.md
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ToolHandler = (params: Record<string, unknown>) => Promise<string>;
type ResourceHandler = () => Promise<string>;

interface RegisteredTool {
  name: string;
  description: string;
  schema: Record<string, unknown>;
  handler: ToolHandler;
}

interface RegisteredResource {
  uri: string;
  description: string;
  handler: ResourceHandler;
}

// ---------------------------------------------------------------------------
// SimpleMCPServer
// ---------------------------------------------------------------------------

export class SimpleMCPServer {
  private server: McpServer;
  private tools: RegisteredTool[] = [];
  private resources: RegisteredResource[] = [];

  constructor(name: string, version = "1.0.0") {
    this.server = new McpServer({ name, version }, { capabilities: { tools: {}, resources: {} } });
  }

  /**
   * Register a tool. The schema must be a valid JSON Schema object.
   */
  tool(name: string, description: string, schema: Record<string, unknown>, handler: ToolHandler): void {
    this.tools.push({ name, description, schema, handler });

    // Build Zod shape from JSON Schema properties for McpServer API
    const properties = (schema.properties ?? {}) as Record<string, { type: string; description?: string }>;
    const required = (schema.required ?? []) as string[];
    const zodShape: Record<string, z.ZodTypeAny> = {};

    for (const [key, def] of Object.entries(properties)) {
      let field: z.ZodTypeAny;
      switch (def.type) {
        case "integer":
          field = z.number().int().describe(def.description ?? key);
          break;
        case "number":
          field = z.number().describe(def.description ?? key);
          break;
        case "boolean":
          field = z.boolean().describe(def.description ?? key);
          break;
        case "array":
          field = z.array(z.unknown()).describe(def.description ?? key);
          break;
        default:
          field = z.string().describe(def.description ?? key);
      }
      zodShape[key] = required.includes(key) ? field : field.optional();
    }

    this.server.tool(name, description, zodShape, async (params) => {
      const result = await handler(params as Record<string, unknown>);
      return { content: [{ type: "text" as const, text: result }] };
    });
  }

  /**
   * Register a resource at the given URI.
   */
  resource(uri: string, description: string, handler: ResourceHandler): void {
    this.resources.push({ uri, description, handler });

    this.server.resource(uri, uri, { description }, async () => {
      const text = await handler();
      return { contents: [{ uri, mimeType: "text/plain", text }] };
    });
  }

  /** Start the server on stdio. Blocks until the process exits. */
  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error(`MCP server started — ${this.tools.length} tools, ${this.resources.length} resources`);
  }

  /** Summary for inspection. */
  summary(): string {
    const t = this.tools.map((t) => `  tool: ${t.name} — ${t.description}`).join("\n");
    const r = this.resources.map((r) => `  resource: ${r.uri} — ${r.description}`).join("\n");
    return [t, r].filter(Boolean).join("\n");
  }
}
