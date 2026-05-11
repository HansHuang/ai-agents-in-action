/**
 * Express-based Agent Service for production deployment.
 *
 * Endpoints:
 *   POST /agent/chat        — synchronous chat
 *   POST /agent/chat/stream — Server-Sent Events streaming
 *   GET  /health            — comprehensive health status
 *   GET  /metrics           — Prometheus-format metrics
 *
 * Reference: docs/09-from-dev-to-production/01-deployment-strategies.md
 */

import http from "http";
import OpenAI from "openai";

const PORT = Number(process.env.PORT ?? 3000);
const MODEL = process.env.DEFAULT_MODEL ?? "gpt-4o-mini";
const API_KEY = process.env.OPENAI_API_KEY ?? "";

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

let totalRequests = 0;
let totalErrors = 0;
let totalTokens = 0;
const latencies: number[] = [];
const startTime = Date.now();

function recordRequest(durationMs: number, tokens: number, error = false): void {
  totalRequests++;
  if (error) totalErrors++;
  totalTokens += tokens;
  latencies.push(durationMs);
  if (latencies.length > 1000) latencies.shift();
}

function avgLatency(): number {
  return latencies.length ? latencies.reduce((s, v) => s + v, 0) / latencies.length : 0;
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

interface ChatRequest {
  message: string;
  conversationId?: string;
  systemPrompt?: string;
}

async function handleChat(body: ChatRequest, client: OpenAI): Promise<{ answer: string; tokensUsed: number; durationMs: number }> {
  const start = Date.now();
  const resp = await client.chat.completions.create({
    model: MODEL,
    messages: [
      { role: "system", content: body.systemPrompt ?? "You are a helpful assistant." },
      { role: "user", content: body.message },
    ],
    max_tokens: 512,
    temperature: 0.7,
  });
  const durationMs = Date.now() - start;
  const tokensUsed = resp.usage?.total_tokens ?? 0;
  return { answer: resp.choices[0].message.content ?? "", tokensUsed, durationMs };
}

function healthCheck(): Record<string, unknown> {
  return {
    status: "healthy",
    version: "1.0.0",
    uptime_ms: Date.now() - startTime,
    model: MODEL,
    llm_configured: Boolean(API_KEY),
    metrics: {
      total_requests: totalRequests,
      error_rate: totalRequests > 0 ? (totalErrors / totalRequests).toFixed(3) : "0.000",
      avg_latency_ms: avgLatency().toFixed(0),
    },
  };
}

function prometheusMetrics(): string {
  return [
    `# HELP agent_requests_total Total requests processed`,
    `# TYPE agent_requests_total counter`,
    `agent_requests_total ${totalRequests}`,
    `# HELP agent_errors_total Total errors`,
    `# TYPE agent_errors_total counter`,
    `agent_errors_total ${totalErrors}`,
    `# HELP agent_tokens_total Total LLM tokens used`,
    `# TYPE agent_tokens_total counter`,
    `agent_tokens_total ${totalTokens}`,
    `# HELP agent_latency_avg_ms Average latency`,
    `# TYPE agent_latency_avg_ms gauge`,
    `agent_latency_avg_ms ${avgLatency().toFixed(1)}`,
    "",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

const client = new OpenAI({ apiKey: API_KEY });

const server = http.createServer(async (req, res) => {
  const url = req.url ?? "/";
  const method = req.method ?? "GET";

  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  try {
    if (method === "GET" && url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(healthCheck()));
      return;
    }

    if (method === "GET" && url === "/metrics") {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end(prometheusMetrics());
      return;
    }

    if (method === "POST" && url === "/agent/chat") {
      const body = await readBody(req);
      const parsed: ChatRequest = JSON.parse(body);
      if (!parsed.message) throw new Error("message is required");

      const result = await handleChat(parsed, client);
      recordRequest(result.durationMs, result.tokensUsed);

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ answer: result.answer, tokens_used: result.tokensUsed, duration_ms: result.durationMs }));
      return;
    }

    if (method === "POST" && url === "/agent/chat/stream") {
      const body = await readBody(req);
      const parsed: ChatRequest = JSON.parse(body);
      if (!parsed.message) throw new Error("message is required");

      res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" });

      const stream = await client.chat.completions.create({
        model: MODEL,
        messages: [
          { role: "system", content: parsed.systemPrompt ?? "You are a helpful assistant." },
          { role: "user", content: parsed.message },
        ],
        stream: true,
        max_tokens: 512,
      });

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content ?? "";
        if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
      }
      res.write("data: [DONE]\n\n");
      res.end();
      recordRequest(0, 0);
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not found" }));
  } catch (err) {
    totalErrors++;
    const message = err instanceof Error ? err.message : String(err);
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: message }));
  }
});

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => (data += chunk));
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

export function startServer(port = PORT): void {
  server.listen(port, () => {
    console.log(`Agent Service running on http://localhost:${port}`);
    console.log(`  POST /agent/chat        — sync chat`);
    console.log(`  POST /agent/chat/stream — streaming (SSE)`);
    console.log(`  GET  /health            — health check`);
    console.log(`  GET  /metrics           — Prometheus metrics`);
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  startServer();
}
