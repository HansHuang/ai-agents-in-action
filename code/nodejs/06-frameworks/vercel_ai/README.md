# Vercel AI SDK — Streaming Agent

> **Key insight:** The Vercel AI SDK brings AI to the frontend. Streaming is first-class, not an afterthought.

This folder demonstrates the Vercel AI SDK's approach to building streaming AI agents — from tool definitions through to a live React chat interface.

→ Full background: [docs/06-frameworks-in-practice/04-vercel-ai-sdk.md](../../../../docs/06-frameworks-in-practice/04-vercel-ai-sdk.md)

---

## The Three Core Primitives

| Primitive | What it does |
|:---|:---|
| `generateText` | One-shot prompt → text. No streaming. |
| `streamText` | Streams tokens as they're generated. Powers the agent loop via `maxSteps`. |
| `tool` | Zod-validated tool that the model can call. You write the `execute` function. |

---

## Files

### Core Implementation

| File | Purpose |
|:---|:---|
| `tools.ts` | Six production-ready tools: `searchKnowledgeBase`, `lookupOrder`, `lookupCustomer`, `createTicket`, `escalateToHuman`, `checkReturnEligibility` |
| `agent.ts` | Customer support agent — `streamText` with all tools, comprehensive system prompt, step logging |
| `route.ts` | Next.js App Router API route with streaming response, CORS, rate limiting |
| `chat-component.tsx` | React `useChat` chat UI: streaming, tool call cards, localStorage persistence |
| `page.tsx` | Next.js page that renders the chat component |

### Comparisons

| File | Purpose |
|:---|:---|
| `provider_comparison.ts` | Same agent across OpenAI, Anthropic, Google, Groq — time, tokens, cost table |
| `custom_vs_sdk.ts` | Custom `generateText` loop vs. `streamText maxSteps` — code complexity + capability table |

### Memory

| File | Purpose |
|:---|:---|
| `agent_memory.ts` | Three-tier memory: short-term truncation, long-term summaries, cross-session user facts |

### Tests

| File | Purpose |
|:---|:---|
| `test_agent.test.ts` | Vitest unit tests for all tools and the `MemoryManager`; integration tests for streaming |

---

## Quick Start

```bash
# 1. Install dependencies (from the 06-frameworks/ directory)
npm install

# 2. Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Run the support agent demo
npx tsx vercel_ai/agent.ts

# 4. Compare providers (set any subset of keys)
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... npx tsx vercel_ai/provider_comparison.ts

# 5. Custom loop vs. SDK comparison
npx tsx vercel_ai/custom_vs_sdk.ts

# 6. Run unit tests
npx vitest run vercel_ai/test_agent.test.ts
```

To use the Next.js chat interface, copy the files into a Next.js 14+ project:

```
app/api/chat/route.ts      ← vercel_ai/route.ts
app/chat/page.tsx          ← vercel_ai/page.tsx
components/ChatContainer.tsx ← vercel_ai/chat-component.tsx
```

---

## Architecture

```
Tools (Zod schemas + mock execute)
  └── agent.ts (streamText + maxSteps)
        └── route.ts (Next.js POST /api/chat)
              └── useChat hook (chat-component.tsx)
                    └── page.tsx
```

---

## Provider Support

OpenAI · Anthropic · Google · Groq · Mistral · and more — one unified API:

```typescript
// Switch providers by changing one import
import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";

const result = await generateText({ model: openai("gpt-4o"), prompt });
const result = await generateText({ model: anthropic("claude-3-5-sonnet-20241022"), prompt });
```

---

## When to Use / Avoid

**Use it when:** building full-stack TypeScript apps, streaming UX is critical, you want to switch providers easily, deploying on serverless/edge.

**Avoid it when:** your backend is Python, you need LangChain's 700+ integrations, you need LangGraph-style workflow graphs.
