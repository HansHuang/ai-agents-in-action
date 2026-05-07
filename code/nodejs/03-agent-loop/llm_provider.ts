/**
 * Provider-agnostic LLM abstraction layer — TypeScript port.
 *
 * Implements a uniform interface over multiple LLM providers so that agent
 * code never depends on a specific SDK or API format.
 *
 * Supported providers:
 *   OpenAIProvider, AnthropicProvider, GoogleProvider,
 *   OllamaProvider, TogetherProvider, FallbackProvider
 *
 * Usage:
 *   const provider = LLMFactory.create("openai", {
 *     apiKey: process.env.OPENAI_API_KEY!,
 *     model: "gpt-4o",
 *   });
 *   const response = await provider.chat([{ role: "user", content: "Hello" }]);
 *   console.log(response.content);
 *
 * See: docs/05-the-tool-ecosystem/01-model-providers.md
 */

import { performance } from "perf_hooks";

// ---------------------------------------------------------------------------
// Shared data types
// ---------------------------------------------------------------------------

export interface LLMResponse {
  content: string | null;
  toolCalls: ToolCall[] | null;
  tokenUsage: { promptTokens: number; completionTokens: number; totalTokens: number };
  model: string;
  finishReason: string;
  latencyMs: number;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  toolCallId?: string;   // for role === "tool"
  toolCalls?: ToolCall[]; // for role === "assistant"
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

/** Rough token estimate: ~4 chars per token. */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.floor(text.length / 4));
}

// ---------------------------------------------------------------------------
// Abstract interface
// ---------------------------------------------------------------------------

export abstract class LLMProvider {
  abstract chat(
    messages: Message[],
    options?: {
      tools?: ToolDefinition[];
      temperature?: number;
      maxTokens?: number;
      responseFormat?: Record<string, unknown>;
    }
  ): Promise<LLMResponse>;

  abstract supportsFunctionCalling(): boolean;
  abstract supportsStructuredOutput(): boolean;
  abstract getContextWindow(): number;
  abstract getModelName(): string;

  /** Count tokens. Override with a real tokenizer when available. */
  countTokens(text: string): number {
    return estimateTokens(text);
  }
}

// ---------------------------------------------------------------------------
// OpenAI provider
// ---------------------------------------------------------------------------

const OPENAI_CONTEXT_WINDOWS: Record<string, number> = {
  "gpt-4o":       128_000,
  "gpt-4o-mini":  128_000,
  "gpt-3.5-turbo": 16_385,
};

export class OpenAIProvider extends LLMProvider {
  private client: import("openai").default | null = null;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl?: string;

  constructor(opts: { apiKey: string; model?: string; baseUrl?: string }) {
    super();
    this.apiKey = opts.apiKey;
    this.model = opts.model ?? "gpt-4o";
    this.baseUrl = opts.baseUrl;
  }

  private async getClient(): Promise<import("openai").default> {
    if (!this.client) {
      const { default: OpenAI } = await import("openai" as string) as { default: typeof import("openai").default };
      this.client = new OpenAI({
        apiKey: this.apiKey,
        ...(this.baseUrl ? { baseURL: this.baseUrl } : {}),
      });
    }
    return this.client;
  }

  async chat(
    messages: Message[],
    options: {
      tools?: ToolDefinition[];
      temperature?: number;
      maxTokens?: number;
      responseFormat?: Record<string, unknown>;
    } = {}
  ): Promise<LLMResponse> {
    const openai = await this.getClient();
    const t0 = performance.now();

    const params: Record<string, unknown> = {
      model: this.model,
      messages: messages as unknown[],
      temperature: options.temperature ?? 0.7,
      max_tokens: options.maxTokens ?? 4096,
    };
    if (options.tools?.length) params["tools"] = options.tools;
    if (options.responseFormat) params["response_format"] = options.responseFormat;

    const resp = await (openai.chat.completions.create as (p: Record<string, unknown>) => Promise<import("openai").ChatCompletion>)(params);
    const latencyMs = Math.round(performance.now() - t0);
    const msg = resp.choices[0].message;

    const toolCalls: ToolCall[] | null = msg.tool_calls?.map((tc) => ({
      id: tc.id,
      type: "function" as const,
      function: { name: tc.function.name, arguments: tc.function.arguments },
    })) ?? null;

    return {
      content: msg.content,
      toolCalls,
      tokenUsage: {
        promptTokens: resp.usage?.prompt_tokens ?? 0,
        completionTokens: resp.usage?.completion_tokens ?? 0,
        totalTokens: resp.usage?.total_tokens ?? 0,
      },
      model: this.model,
      finishReason: resp.choices[0].finish_reason,
      latencyMs,
    };
  }

  supportsFunctionCalling(): boolean { return true; }
  supportsStructuredOutput(): boolean { return true; }
  getContextWindow(): number { return OPENAI_CONTEXT_WINDOWS[this.model] ?? 128_000; }
  getModelName(): string { return this.model; }
}

// ---------------------------------------------------------------------------
// Anthropic provider
// ---------------------------------------------------------------------------

type AnthropicMessage = { role: "user" | "assistant"; content: unknown };

export class AnthropicProvider extends LLMProvider {
  private client: import("@anthropic-ai/sdk").default | null = null;
  private readonly apiKey: string;
  private readonly model: string;

  constructor(opts: { apiKey: string; model?: string }) {
    super();
    this.apiKey = opts.apiKey;
    this.model = opts.model ?? "claude-3-5-sonnet-20241022";
  }

  private async getClient(): Promise<import("@anthropic-ai/sdk").default> {
    if (!this.client) {
      const { default: Anthropic } = await import("@anthropic-ai/sdk" as string) as { default: typeof import("@anthropic-ai/sdk").default };
      this.client = new Anthropic({ apiKey: this.apiKey });
    }
    return this.client;
  }

  /** Split system message; convert the rest to Anthropic format. */
  private toAnthropicMessages(messages: Message[]): { system?: string; messages: AnthropicMessage[] } {
    let system: string | undefined;
    const converted: AnthropicMessage[] = [];
    for (const msg of messages) {
      if (msg.role === "system") { system = msg.content; continue; }
      if (msg.role === "tool") {
        converted.push({
          role: "user",
          content: [{ type: "tool_result", tool_use_id: msg.toolCallId ?? "", content: msg.content }],
        });
      } else {
        converted.push({ role: msg.role as "user" | "assistant", content: msg.content });
      }
    }
    return { system, messages: converted };
  }

  /** Convert OpenAI tool definitions to Anthropic format. */
  private toAnthropicTools(tools: ToolDefinition[]): unknown[] {
    return tools.map((t) => ({
      name: t.function.name,
      description: t.function.description ?? "",
      input_schema: t.function.parameters ?? { type: "object", properties: {} },
    }));
  }

  async chat(
    messages: Message[],
    options: {
      tools?: ToolDefinition[];
      temperature?: number;
      maxTokens?: number;
      responseFormat?: Record<string, unknown>;
    } = {}
  ): Promise<LLMResponse> {
    const anthropic = await this.getClient();
    const { system, messages: converted } = this.toAnthropicMessages(messages);

    const params: Record<string, unknown> = {
      model: this.model,
      messages: converted,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.maxTokens ?? 4096,
    };
    if (system) params["system"] = system;
    if (options.tools?.length) params["tools"] = this.toAnthropicTools(options.tools);

    const t0 = performance.now();
    type AnthropicResponse = {
      content: Array<{ type: string; text?: string; id?: string; name?: string; input?: Record<string, unknown> }>;
      stop_reason: string;
      usage: { input_tokens: number; output_tokens: number };
    };
    const resp = await (anthropic.messages.create as (p: Record<string, unknown>) => Promise<AnthropicResponse>)(params);
    const latencyMs = Math.round(performance.now() - t0);

    let content: string | null = null;
    const toolCalls: ToolCall[] = [];
    for (const block of resp.content) {
      if (block.type === "text" && block.text) content = (content ?? "") + block.text;
      if (block.type === "tool_use" && block.id && block.name) {
        toolCalls.push({
          id: block.id,
          type: "function",
          function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
        });
      }
    }

    return {
      content,
      toolCalls: toolCalls.length ? toolCalls : null,
      tokenUsage: {
        promptTokens: resp.usage.input_tokens,
        completionTokens: resp.usage.output_tokens,
        totalTokens: resp.usage.input_tokens + resp.usage.output_tokens,
      },
      model: this.model,
      finishReason: resp.stop_reason ?? "stop",
      latencyMs,
    };
  }

  supportsFunctionCalling(): boolean { return true; }
  supportsStructuredOutput(): boolean { return false; }
  getContextWindow(): number { return 200_000; }
  getModelName(): string { return this.model; }
}

// ---------------------------------------------------------------------------
// Ollama provider  (OpenAI-compatible local endpoint)
// ---------------------------------------------------------------------------

const OLLAMA_TOOL_MODELS = new Set([
  "llama3.1", "llama3.1:8b", "llama3.1:70b",
  "llama3.2", "llama3.2:3b",
  "mistral", "mistral-nemo",
  "qwen2.5", "qwen2.5:7b",
  "command-r",
]);

export class OllamaProvider extends LLMProvider {
  private readonly inner: OpenAIProvider;
  private readonly model: string;

  constructor(opts: { model?: string; baseUrl?: string }) {
    super();
    this.model = opts.model ?? "llama3.1:8b";
    this.inner = new OpenAIProvider({
      apiKey: "ollama",
      model: this.model,
      baseUrl: opts.baseUrl ?? "http://localhost:11434/v1",
    });
  }

  async chat(messages: Message[], options: {
    tools?: ToolDefinition[];
    temperature?: number;
    maxTokens?: number;
    responseFormat?: Record<string, unknown>;
  } = {}): Promise<LLMResponse> {
    const effectiveTools = this.supportsFunctionCalling() ? options.tools : undefined;
    return this.inner.chat(messages, { ...options, tools: effectiveTools });
  }

  supportsFunctionCalling(): boolean {
    const base = this.model.split(":")[0];
    return OLLAMA_TOOL_MODELS.has(base) || OLLAMA_TOOL_MODELS.has(this.model);
  }
  supportsStructuredOutput(): boolean { return false; }
  getContextWindow(): number { return 128_000; }
  getModelName(): string { return this.model; }
}

// ---------------------------------------------------------------------------
// Together AI provider  (OpenAI-compatible cloud endpoint)
// ---------------------------------------------------------------------------

export class TogetherProvider extends LLMProvider {
  private readonly inner: OpenAIProvider;
  private readonly model: string;

  constructor(opts: { apiKey: string; model?: string }) {
    super();
    this.model = opts.model ?? "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo";
    this.inner = new OpenAIProvider({
      apiKey: opts.apiKey,
      model: this.model,
      baseUrl: "https://api.together.xyz/v1",
    });
  }

  async chat(messages: Message[], options = {}): Promise<LLMResponse> {
    return this.inner.chat(messages, options);
  }

  supportsFunctionCalling(): boolean { return false; }
  supportsStructuredOutput(): boolean { return false; }
  getContextWindow(): number { return 128_000; }
  getModelName(): string { return this.model; }
}

// ---------------------------------------------------------------------------
// FallbackProvider
// ---------------------------------------------------------------------------

export class FallbackProvider extends LLMProvider {
  private readonly primary: LLMProvider;
  private readonly fallbacks: LLMProvider[];

  constructor(primary: LLMProvider, fallbacks: LLMProvider[]) {
    super();
    this.primary = primary;
    this.fallbacks = fallbacks;
  }

  async chat(messages: Message[], options = {}): Promise<LLMResponse> {
    const candidates = [this.primary, ...this.fallbacks];
    let lastError: Error | null = null;
    for (const provider of candidates) {
      try {
        return await provider.chat(messages, options);
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        console.warn(
          `[FallbackProvider] ${provider.getModelName()} failed: ${lastError.message} — trying next`
        );
      }
    }
    throw new Error(`All providers failed. Last error: ${lastError?.message}`);
  }

  supportsFunctionCalling(): boolean { return this.primary.supportsFunctionCalling(); }
  supportsStructuredOutput(): boolean { return this.primary.supportsStructuredOutput(); }
  getContextWindow(): number { return this.primary.getContextWindow(); }
  getModelName(): string {
    return [this.primary, ...this.fallbacks].map((p) => p.getModelName()).join(" → ");
  }
}

// ---------------------------------------------------------------------------
// LLMFactory
// ---------------------------------------------------------------------------

export type ProviderConfig =
  | { provider: "openai";    apiKey?: string; apiKeyEnv?: string; model?: string; baseUrl?: string }
  | { provider: "anthropic"; apiKey?: string; apiKeyEnv?: string; model?: string }
  | { provider: "ollama";    model?: string; baseUrl?: string }
  | { provider: "together";  apiKey?: string; apiKeyEnv?: string; model?: string }
  | { provider: string;      [key: string]: unknown };

export type ProviderConfigWithFallback = ProviderConfig & {
  fallback?: ProviderConfigWithFallback;
};

export class LLMFactory {
  static create(config: ProviderConfig): LLMProvider {
    const apiKey = "apiKeyEnv" in config && config.apiKeyEnv
      ? (process.env[config.apiKeyEnv] ?? "")
      : ("apiKey" in config ? (config.apiKey ?? "") : "");

    switch (config.provider) {
      case "openai":
        return new OpenAIProvider({
          apiKey,
          model: (config as { model?: string }).model,
          baseUrl: (config as { baseUrl?: string }).baseUrl,
        });
      case "anthropic":
        return new AnthropicProvider({ apiKey, model: (config as { model?: string }).model });
      case "ollama":
        return new OllamaProvider({
          model: (config as { model?: string }).model,
          baseUrl: (config as { baseUrl?: string }).baseUrl,
        });
      case "together":
        return new TogetherProvider({ apiKey, model: (config as { model?: string }).model });
      default:
        throw new Error(
          `Unknown provider: "${config.provider}". Available: openai, anthropic, ollama, together`
        );
    }
  }

  static createFromConfig(config: ProviderConfigWithFallback): LLMProvider {
    const primary = LLMFactory.create(config);
    if (config.fallback) {
      const fallback = LLMFactory.createFromConfig(config.fallback);
      return new FallbackProvider(primary, [fallback]);
    }
    return primary;
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

async function _demo(): Promise<void> {
  const question = "What is 17 × 23? Answer with just the number.";
  const messages: Message[] = [{ role: "user", content: question }];

  const providers: Array<[string, LLMProvider]> = [];

  const oaiKey = process.env["OPENAI_API_KEY"];
  const antKey = process.env["ANTHROPIC_API_KEY"];

  if (oaiKey) {
    providers.push(["OpenAI gpt-4o-mini", LLMFactory.create({ provider: "openai", apiKey: oaiKey, model: "gpt-4o-mini" })]);
  }
  if (antKey) {
    providers.push(["Anthropic claude-3-haiku", LLMFactory.create({ provider: "anthropic", apiKey: antKey, model: "claude-3-haiku-20240307" })]);
  }
  providers.push(["Ollama llama3.1:8b", LLMFactory.create({ provider: "ollama", model: "llama3.1:8b" })]);

  if (!providers.length) {
    console.log("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run the demo.");
    return;
  }

  console.log("\n=== Provider Comparison ===");
  console.log(`Question: ${question}\n`);
  console.log("Provider".padEnd(30), "Answer".padEnd(20), "Tokens".padEnd(10), "Latency (ms)");
  console.log("-".repeat(75));
  for (const [name, provider] of providers) {
    try {
      const resp = await provider.chat(messages, { temperature: 0, maxTokens: 20 });
      console.log(
        name.padEnd(30),
        (resp.content ?? "").trim().padEnd(20),
        String(resp.tokenUsage.totalTokens).padEnd(10),
        resp.latencyMs
      );
    } catch (err) {
      console.log(name.padEnd(30), `ERROR: ${err}`);
    }
  }

  // Fallback demo
  console.log("\n=== Fallback Demonstration ===");
  class AlwaysFail extends LLMProvider {
    async chat(): Promise<LLMResponse> { throw new Error("Simulated outage"); }
    supportsFunctionCalling() { return false; }
    supportsStructuredOutput() { return false; }
    getContextWindow() { return 0; }
    getModelName() { return "always-fail"; }
  }
  const real = providers[0][1];
  const fallback = new FallbackProvider(new AlwaysFail(), [real]);
  const resp = await fallback.chat(messages, { temperature: 0, maxTokens: 20 });
  console.log(`Fallback result: ${JSON.stringify(resp.content)} (from ${real.getModelName()})`);
}

// Run demo when executed directly
const isMain = process.argv[1]?.endsWith("llm_provider.ts") ||
               process.argv[1]?.endsWith("llm_provider.js");
if (isMain) {
  _demo().catch(console.error);
}
