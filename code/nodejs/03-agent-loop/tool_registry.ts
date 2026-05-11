/**
 * Tool registry with OpenAI schema generation and structured error handling.
 * See: docs/02-the-agent-loop/02-tool-design-patterns.md
 */

import OpenAI from "openai";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class ToolNotFoundError extends Error {
  constructor(name: string) {
    super(`Tool not found: ${name}`);
  }
}

export class InvalidArgsError extends Error {
  constructor(
    message: string,
    public readonly allowedValues?: unknown[]
  ) {
    super(message);
  }
}

export class NotFoundError extends Error {
  constructor(
    message: string,
    public readonly suggestion?: string
  ) {
    super(message);
  }
}

// ---------------------------------------------------------------------------
// Registry types
// ---------------------------------------------------------------------------

export interface ToolParameter {
  type: string;
  description: string;
  required?: boolean;
  enum?: string[];
  default?: unknown;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, ToolParameter>;
  handler: (args: Record<string, unknown>) => Promise<unknown> | unknown;
}

export interface ToolCallResult {
  role: "tool";
  tool_call_id: string;
  content: string;
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

export class ToolRegistry {
  private tools = new Map<string, ToolDefinition>();

  /** Register a tool. */
  register(definition: ToolDefinition): void {
    this.tools.set(definition.name, definition);
  }

  /** Return all tools as OpenAI tool schemas. */
  getOpenAISchemas(): OpenAI.Chat.ChatCompletionTool[] {
    return Array.from(this.tools.values()).map((def) => {
      const required = Object.entries(def.parameters)
        .filter(([, p]) => p.required !== false)
        .map(([k]) => k);
      const properties: Record<string, unknown> = {};
      for (const [k, p] of Object.entries(def.parameters)) {
        const prop: Record<string, unknown> = { type: p.type, description: p.description };
        if (p.enum) prop.enum = p.enum;
        properties[k] = prop;
      }
      return {
        type: "function" as const,
        function: {
          name: def.name,
          description: def.description,
          parameters: { type: "object", properties, required },
        },
      };
    });
  }

  /** Execute a tool call and return the tool-role message. */
  async executeTool(toolCall: OpenAI.Chat.ChatCompletionMessageToolCall): Promise<ToolCallResult> {
    const def = this.tools.get(toolCall.function.name);
    if (!def) {
      return {
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify({ error: "tool_not_found", tool: toolCall.function.name }),
      };
    }
    let args: Record<string, unknown>;
    try {
      args = JSON.parse(toolCall.function.arguments) as Record<string, unknown>;
    } catch {
      return {
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify({ error: "invalid_json_args" }),
      };
    }
    try {
      const result = await def.handler(args);
      return { role: "tool", tool_call_id: toolCall.id, content: JSON.stringify(result) };
    } catch (err) {
      if (err instanceof NotFoundError) {
        return {
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify({
            error: "not_found",
            message: (err as Error).message,
            suggestion: (err as NotFoundError).suggestion,
          }),
        };
      }
      if (err instanceof InvalidArgsError) {
        return {
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify({
            error: "invalid_args",
            message: (err as Error).message,
            allowed_values: (err as InvalidArgsError).allowedValues,
          }),
        };
      }
      return {
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify({ error: "execution_error", message: String(err) }),
      };
    }
  }

  /** List all registered tool names. */
  list(): string[] {
    return Array.from(this.tools.keys());
  }
}
