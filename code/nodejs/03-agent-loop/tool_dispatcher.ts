/**
 * Tool execution dispatcher for the ReAct agent loop.
 *
 * dispatchTool() is the bridge between the LLM's tool_call decision and your
 * TypeScript functions. It:
 *   - Looks up the tool by name in a registry
 *   - Parses the JSON arguments from the tool_call
 *   - Calls the function and returns a formatted tool message
 *   - Returns a descriptive error message (not a thrown exception) if anything
 *     fails — the LLM receives the error and can explain it to the user
 *
 * See docs/02-the-agent-loop/01-anatomy-of-an-agent.md — "The Hands (Tools)"
 */

import type {
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
} from "openai/resources/chat/completions";

// Registry maps tool names to functions that accept parsed args.
export type ToolFunction = (args: Record<string, unknown>) => unknown;
export type ToolRegistry = Record<string, ToolFunction>;

/**
 * Execute a tool call and return a formatted tool message.
 *
 * @param toolCall - An OpenAI tool call object with .id, .function.name,
 *                   and .function.arguments (JSON string).
 * @param registry - Mapping of tool names to callables.
 * @returns A tool message ready to push onto the messages array:
 *          `{ role: "tool", content: "<json>", tool_call_id: "<id>" }`
 */
export function dispatchTool(
  toolCall: ChatCompletionMessageToolCall,
  registry: ToolRegistry
): ChatCompletionMessageParam {
  const { id: toolCallId, function: fn } = toolCall;
  const name = fn.name;

  // Parse arguments.
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(fn.arguments) as Record<string, unknown>;
  } catch (err) {
    console.error(`[dispatchTool] Tool ${name}: invalid argument JSON:`, err);
    return _errorMessage(toolCallId, `Invalid arguments JSON: ${String(err)}`);
  }

  // Look up the tool.
  const toolFn = registry[name];
  if (!toolFn) {
    const available = Object.keys(registry).join(", ");
    console.warn(`[dispatchTool] Tool '${name}' not found. Available: ${available}`);
    return _errorMessage(
      toolCallId,
      `Tool '${name}' is not available. Available tools: ${available}`
    );
  }

  // Execute with timing.
  const start = performance.now();
  let result: unknown;
  try {
    result = toolFn(args);
  } catch (err) {
    const elapsed = (performance.now() - start).toFixed(1);
    console.error(`[dispatchTool] Tool '${name}' failed (${elapsed} ms):`, err);
    return _errorMessage(toolCallId, `Tool '${name}' failed: ${String(err)}`);
  }

  const elapsed = (performance.now() - start).toFixed(1);
  const content = JSON.stringify(result);
  const preview = content.length > 200 ? content.slice(0, 200) + "…" : content;
  console.debug(
    `[dispatchTool] ${name}(${JSON.stringify(args)}) → ${preview}  [${elapsed} ms]`
  );

  return {
    role: "tool",
    content,
    tool_call_id: toolCallId,
  };
}

function _errorMessage(toolCallId: string, error: string): ChatCompletionMessageParam {
  return {
    role: "tool",
    content: JSON.stringify({ error }),
    tool_call_id: toolCallId,
  };
}
