/**
 * MCP-Enabled Agent — TypeScript
 *
 * Discovers and calls tools from MCP servers using the same
 * {serverName}__{toolName} namespace pattern as the Python version.
 *
 * Usage:
 *   OPENAI_API_KEY=sk-... npx tsx mcp_agent.ts
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import OpenAI from "openai";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AgentResult {
  answer: string;
  messages: OpenAI.Chat.ChatCompletionMessageParam[];
  toolsCalled: string[];
}

export interface ToolResult {
  serverName: string;
  toolName: string;
  output: string;
  isError: boolean;
}

// Raw MCP tool as returned by client.listTools()
interface McpTool {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// LLM Provider
// ---------------------------------------------------------------------------

export interface LLMProvider {
  chat(
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
    tools?: OpenAI.Chat.ChatCompletionTool[],
  ): Promise<OpenAI.Chat.ChatCompletionMessage>;
}

export class OpenAIProvider implements LLMProvider {
  private client: OpenAI;

  constructor(
    private model: string = "gpt-4o",
    apiKey?: string,
  ) {
    this.client = new OpenAI({ apiKey: apiKey ?? process.env["OPENAI_API_KEY"] });
  }

  async chat(
    messages: OpenAI.Chat.ChatCompletionMessageParam[],
    tools?: OpenAI.Chat.ChatCompletionTool[],
  ): Promise<OpenAI.Chat.ChatCompletionMessage> {
    const params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
      model: this.model,
      messages,
      ...(tools && tools.length > 0 ? { tools } : {}),
    };
    const response = await this.client.chat.completions.create(params);
    return response.choices[0]!.message;
  }
}

// ---------------------------------------------------------------------------
// ServerConnection — one long-lived MCP session
// ---------------------------------------------------------------------------

class ServerConnection {
  private client: Client;
  private transport: StdioClientTransport;
  tools: McpTool[] = [];
  connected = false;

  constructor(public readonly name: string) {
    this.client = new Client(
      { name: `agent-client-${name}`, version: "1.0.0" },
      { capabilities: {} },
    );
    // Transport is created during connect()
    this.transport = null as unknown as StdioClientTransport;
  }

  async connect(command: string, args: string[]): Promise<void> {
    this.transport = new StdioClientTransport({ command, args });
    await this.client.connect(this.transport);

    const result = await this.client.listTools();
    this.tools = result.tools as McpTool[];
    this.connected = true;

    process.stderr.write(
      `[registry] Connected to '${this.name}': ${this.tools.length} tools discovered\n`,
    );
  }

  async callTool(toolName: string, toolArgs: Record<string, unknown>): Promise<ToolResult> {
    if (!this.connected) {
      return {
        serverName: this.name,
        toolName,
        output: JSON.stringify({ error: `Server '${this.name}' is not connected.` }),
        isError: true,
      };
    }
    try {
      const result = await this.client.callTool({ name: toolName, arguments: toolArgs });
      const content = result.content as Array<{ type: string; text?: string }>;
      const output = content
        .filter(c => c.type === "text" && c.text)
        .map(c => c.text!)
        .join("\n") || "(empty response)";
      return {
        serverName: this.name,
        toolName,
        output,
        isError: Boolean(result.isError),
      };
    } catch (err) {
      this.connected = false;
      return {
        serverName: this.name,
        toolName,
        output: JSON.stringify({ error: `Tool execution failed: ${err}` }),
        isError: true,
      };
    }
  }

  async close(): Promise<void> {
    this.connected = false;
    await this.transport.close();
  }
}

// ---------------------------------------------------------------------------
// ServerRegistry — manages multiple connections
// ---------------------------------------------------------------------------

interface ConnectParams {
  command: string;
  args: string[];
}

export class ServerRegistry {
  readonly servers: Map<string, ServerConnection> = new Map();
  private params: Map<string, ConnectParams> = new Map();

  async connect(
    name: string,
    command: string,
    args: string[],
  ): Promise<OpenAI.Chat.ChatCompletionTool[]> {
    const conn = new ServerConnection(name);
    await conn.connect(command, args);
    this.servers.set(name, conn);
    this.params.set(name, { command, args });
    return this.formatTools(conn, "openai") as OpenAI.Chat.ChatCompletionTool[];
  }

  private toOpenAI(
    tool: McpTool,
    serverName: string,
  ): OpenAI.Chat.ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: `${serverName}__${tool.name}`,
        description: tool.description ?? "",
        parameters: tool.inputSchema as OpenAI.FunctionParameters,
      },
    };
  }

  private toAnthropic(tool: McpTool, serverName: string): Record<string, unknown> {
    return {
      name: `${serverName}__${tool.name}`,
      description: tool.description ?? "",
      input_schema: tool.inputSchema,
    };
  }

  private formatTools(conn: ServerConnection, fmt: "openai" | "anthropic") {
    return conn.tools.map(t =>
      fmt === "openai" ? this.toOpenAI(t, conn.name) : this.toAnthropic(t, conn.name),
    );
  }

  getAllTools(fmt: "openai" | "anthropic" = "openai") {
    const tools: unknown[] = [];
    for (const conn of this.servers.values()) {
      if (conn.connected) tools.push(...this.formatTools(conn, fmt));
    }
    return tools;
  }

  async callTool(
    serverName: string,
    toolName: string,
    args: Record<string, unknown>,
  ): Promise<string> {
    const conn = this.servers.get(serverName);
    if (!conn) return JSON.stringify({ error: `Unknown server: '${serverName}'` });
    const result = await conn.callTool(toolName, args);
    return result.output;
  }

  async reconnect(serverName: string): Promise<boolean> {
    const p = this.params.get(serverName);
    if (!p) return false;
    const old = this.servers.get(serverName);
    if (old) await old.close();
    try {
      const conn = new ServerConnection(serverName);
      await conn.connect(p.command, p.args);
      this.servers.set(serverName, conn);
      process.stderr.write(`[registry] Reconnected to '${serverName}'\n`);
      return true;
    } catch (err) {
      process.stderr.write(`[registry] Reconnect failed for '${serverName}': ${err}\n`);
      return false;
    }
  }

  async disconnectAll(): Promise<void> {
    for (const conn of this.servers.values()) await conn.close();
    this.servers.clear();
  }

  healthCheck(): Record<string, { connected: boolean; toolCount: number; tools: string[] }> {
    const result: Record<string, { connected: boolean; toolCount: number; tools: string[] }> = {};
    for (const [name, conn] of this.servers) {
      result[name] = {
        connected: conn.connected,
        toolCount: conn.tools.length,
        tools: conn.tools.map(t => t.name),
      };
    }
    return result;
  }
}

// ---------------------------------------------------------------------------
// MCPAgent
// ---------------------------------------------------------------------------

export class MCPAgent {
  readonly serverRegistry: ServerRegistry;
  tools: OpenAI.Chat.ChatCompletionTool[] = [];

  private static readonly MAX_ITERATIONS = 10;

  constructor(private llm: LLMProvider) {
    this.serverRegistry = new ServerRegistry();
  }

  async connectServer(name: string, command: string, args: string[]): Promise<number> {
    const newTools = await this.serverRegistry.connect(name, command, args);
    this.tools.push(...newTools);
    process.stderr.write(
      `[agent] +${newTools.length} tools from '${name}'. Total: ${this.tools.length}\n`,
    );
    return newTools.length;
  }

  async disconnectServer(name: string): Promise<void> {
    const conn = this.serverRegistry.servers.get(name);
    if (conn) {
      await conn.close();
      this.serverRegistry.servers.delete(name);
    }
    this.tools = this.serverRegistry.getAllTools("openai") as OpenAI.Chat.ChatCompletionTool[];
    process.stderr.write(`[agent] Disconnected from '${name}'\n`);
  }

  convertMcpToolToOpenai(
    mcpTool: McpTool,
    serverName: string,
  ): OpenAI.Chat.ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: `${serverName}__${mcpTool.name}`,
        description: mcpTool.description ?? "",
        parameters: mcpTool.inputSchema as OpenAI.FunctionParameters,
      },
    };
  }

  convertMcpToolToAnthropic(
    mcpTool: McpTool,
    serverName: string,
  ): Record<string, unknown> {
    return {
      name: `${serverName}__${mcpTool.name}`,
      description: mcpTool.description ?? "",
      input_schema: mcpTool.inputSchema,
    };
  }

  async run(userInput: string): Promise<AgentResult> {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: "system",
        content:
          "You are a helpful assistant. Use the available tools whenever needed to answer accurately.",
      },
      { role: "user", content: userInput },
    ];
    const toolsCalled: string[] = [];

    for (let i = 0; i < MCPAgent.MAX_ITERATIONS; i++) {
      const response = await this.llm.chat(
        messages,
        this.tools.length > 0 ? this.tools : undefined,
      );
      messages.push(response as OpenAI.Chat.ChatCompletionMessageParam);

      const toolCalls = response.tool_calls ?? [];
      if (toolCalls.length === 0) {
        return { answer: response.content ?? "", messages, toolsCalled };
      }

      for (const tc of toolCalls) {
        const fullName = tc.function.name;
        const result = await this.executeMcpTool(
          fullName,
          JSON.parse(tc.function.arguments) as Record<string, unknown>,
        );
        toolsCalled.push(fullName);
        messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: result.output,
        });
      }
    }

    return {
      answer: "Maximum iterations reached without a final answer.",
      messages,
      toolsCalled,
    };
  }

  async executeMcpTool(
    fullToolName: string,
    args: Record<string, unknown>,
  ): Promise<ToolResult> {
    if (!fullToolName.includes("__")) {
      return {
        serverName: "unknown",
        toolName: fullToolName,
        output: JSON.stringify({
          error: `Invalid tool name: '${fullToolName}'. Expected '{server}__{tool}'.`,
        }),
        isError: true,
      };
    }
    const sep = fullToolName.indexOf("__");
    const serverName = fullToolName.slice(0, sep);
    const toolName   = fullToolName.slice(sep + 2);

    process.stderr.write(
      `[agent] → ${serverName}/${toolName} args=${JSON.stringify(args)}\n`,
    );
    const output = await this.serverRegistry.callTool(serverName, toolName, args);
    return { serverName, toolName, output, isError: false };
  }

  async close(): Promise<void> {
    await this.serverRegistry.disconnectAll();
    process.stderr.write("[agent] All servers disconnected.\n");
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function runDemo(): Promise<void> {
  const weatherServer = path.join(
    __dirname,
    "weather_mcp_server",
    "server.ts",
  );

  process.stderr.write("=".repeat(60) + "\n");
  process.stderr.write("MCP Agent Demo (TypeScript)\n");
  process.stderr.write("=".repeat(60) + "\n");

  const agent = new MCPAgent(new OpenAIProvider("gpt-4o"));

  // Connect to weather server (started via tsx)
  await agent.connectServer("weather", "npx", ["tsx", weatherServer]);

  // Show discovered tools
  process.stderr.write("\n--- Discovered Tools ---\n");
  for (const tool of agent.tools) {
    process.stderr.write(`  ${tool.function.name}\n`);
  }

  process.stderr.write("\n--- Tool Schemas ---\n");
  for (const tool of agent.tools) {
    process.stderr.write(
      `\n${tool.function.name}:\n${JSON.stringify(tool.function.parameters, null, 2)}\n`,
    );
  }

  const queries = [
    "What's the weather in Tokyo?",
    "What's the weather in London and Dubai?",
    "Give me a 3-day forecast for Paris.",
  ];

  for (const query of queries) {
    process.stderr.write("\n" + "─".repeat(50) + "\n");
    process.stderr.write(`Query: ${query}\n`);
    const result = await agent.run(query);
    process.stderr.write(`Answer: ${result.answer}\n`);
    process.stderr.write(`Tools called: ${result.toolsCalled.join(", ")}\n`);
  }

  process.stderr.write(
    "\n--- Health Check ---\n" +
    JSON.stringify(agent.serverRegistry.healthCheck(), null, 2) + "\n",
  );

  await agent.close();
}

runDemo().catch(err => {
  process.stderr.write(`[demo] Fatal: ${err}\n`);
  process.exit(1);
});
