/**
 * route.ts — Next.js App Router API route for the streaming support agent.
 *
 * POST /api/chat
 *   Body: { messages: Message[] }
 *   Response: AI data stream (application/octet-stream)
 *
 * The route streams back tokens as they are generated so the frontend
 * receives real-time updates via the useChat hook.
 */

import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";
import { SUPPORT_TOOLS } from "./agent.js";

// ──── Types ───────────────────────────────────────────────────────────────────

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface RequestBody {
  messages: ChatMessage[];
}

// ──── Constants ───────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `
You are a knowledgeable, empathetic customer support agent for Acme Corp.

PROCESS:
1. Understand the customer's issue fully before acting.
2. Look up any mentioned order numbers or customer emails immediately.
3. Search the knowledge base for self-service answers.
4. Check return eligibility before helping with returns.
5. Only create a ticket if no self-service resolution is possible.
6. Escalate to a human for fraud, VIP customers, or very distressed customers.

TONE: Empathetic first, specific with data, concise.
RULES: Never fabricate order details. Always tell customers what to expect next.
`.trim();

const MAX_STEPS = 10;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN ?? "*";

// ──── Rate limiting stub ──────────────────────────────────────────────────────

/** In-memory rate limit tracker. Replace with Redis in production. */
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();

/**
 * Check whether the given IP has exceeded the rate limit.
 * Returns true if the request should be allowed.
 *
 * Production note: Replace this stub with a proper rate limiter
 * (e.g., @upstash/ratelimit with Redis) before deploying.
 */
function checkRateLimit(ip: string, limit = 20, windowMs = 60_000): boolean {
  const now = Date.now();
  const record = rateLimitMap.get(ip);

  if (!record || record.resetAt < now) {
    rateLimitMap.set(ip, { count: 1, resetAt: now + windowMs });
    return true;
  }

  if (record.count >= limit) {
    return false;
  }

  record.count += 1;
  return true;
}

// ──── CORS headers ────────────────────────────────────────────────────────────

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

// ──── Request validation ──────────────────────────────────────────────────────

function isValidMessages(messages: unknown): messages is ChatMessage[] {
  if (!Array.isArray(messages)) return false;
  if (messages.length === 0) return false;
  if (messages.length > 100) return false; // sanity cap

  return messages.every(
    (m) =>
      typeof m === "object" &&
      m !== null &&
      ["user", "assistant", "system"].includes((m as ChatMessage).role) &&
      typeof (m as ChatMessage).content === "string" &&
      (m as ChatMessage).content.length <= 32_000
  );
}

// ──── Route handlers ──────────────────────────────────────────────────────────

/**
 * Handle CORS preflight.
 */
export function OPTIONS(): Response {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

/**
 * Handle chat POST requests.
 *
 * Validates the request body, applies rate limiting, then streams the
 * agent's response back to the client as an AI data stream.
 */
export async function POST(req: Request): Promise<Response> {
  // ── Rate limiting ──────────────────────────────────────────────────────────
  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";

  if (!checkRateLimit(ip)) {
    return new Response(
      JSON.stringify({ error: "Too many requests. Please wait a minute." }),
      {
        status: 429,
        headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
      }
    );
  }

  // ── Parse body ─────────────────────────────────────────────────────────────
  let body: RequestBody;
  try {
    body = (await req.json()) as RequestBody;
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON in request body." }), {
      status: 400,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    });
  }

  // ── Validate messages ──────────────────────────────────────────────────────
  if (!isValidMessages(body.messages)) {
    return new Response(
      JSON.stringify({
        error:
          "Invalid messages. Must be a non-empty array of {role, content} objects.",
      }),
      {
        status: 422,
        headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
      }
    );
  }

  // ── Stream the agent response ──────────────────────────────────────────────
  try {
    const result = streamText({
      model: openai("gpt-4o"),
      system: SYSTEM_PROMPT,
      messages: body.messages,
      tools: SUPPORT_TOOLS,
      maxSteps: MAX_STEPS,

      onStepFinish: (step) => {
        // Structured logging for observability
        console.log(
          JSON.stringify({
            event: "agent_step",
            ip,
            stepNumber: step.stepNumber,
            finishReason: step.finishReason,
            toolCalls: step.toolCalls?.map((tc) => tc.toolName) ?? [],
            tokens: {
              input: step.usage?.promptTokens ?? 0,
              output: step.usage?.completionTokens ?? 0,
            },
          })
        );
      },

      onFinish: (finalResult) => {
        console.log(
          JSON.stringify({
            event: "agent_finish",
            ip,
            totalSteps: finalResult.steps.length,
            totalTokens: finalResult.usage?.totalTokens ?? 0,
          })
        );
      },
    });

    return result.toDataStreamResponse({
      headers: CORS_HEADERS,
    });
  } catch (err) {
    // Log the error server-side but don't leak internal details to the client
    console.error("Agent error:", err);
    return new Response(
      JSON.stringify({ error: "An internal error occurred. Please try again." }),
      {
        status: 500,
        headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
      }
    );
  }
}
