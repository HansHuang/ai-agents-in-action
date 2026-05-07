# Vercel AI SDK

## What You'll Learn
- Why the Vercel AI SDK takes a fundamentally different approach from LangChain
- The core primitives: `generateText`, `streamText`, and `tool`
- Building a streaming agent using `streamText` with `maxSteps`
- Provider-agnostic design: one API for OpenAI, Anthropic, Google, and more
- The `useChat` hook: bringing AI to the frontend with React
- How `streamText + maxSteps` compares to a fully custom agent loop

## Prerequisites
- [When to Use Frameworks](01-when-to-use-frameworks.md) — the build vs. buy decision
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — you understand the agent loop
- [LangChain and LangGraph](02-langchain-langgraph.md) — the other major framework approach

---

## A Different Philosophy

The Vercel AI SDK starts from a different place than LangChain. LangChain began as a Python framework for composing LLM calls. The Vercel AI SDK began as a TypeScript library for streaming AI responses to the frontend.

This difference in origin shapes everything:

| | LangChain | Vercel AI SDK |
|:---|:---|:---|
| **Origin** | Python, backend chains | TypeScript, frontend streaming |
| **Primary concern** | Composing LLM operations | Delivering AI experiences to users |
| **Streaming** | Added later, good | First-class, excellent |
| **Frontend** | Not a concern | React hooks, Svelte, Vue, Solid |
| **Backend** | Python-first | TypeScript/Node.js (edge-ready) |
| **Provider model** | Adapts to each provider | Unified provider interface |
| **Learning curve** | Steep | Gentle, especially for TS developers |

---

## Core Primitives

The Vercel AI SDK provides a small set of powerful primitives:

### `generateText`

The simplest entry point. Send a prompt, get text back:

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const { text } = await generateText({
  model: openai('gpt-4o'),
  prompt: 'Explain quantum computing in one sentence.',
});

console.log(text);
// "Quantum computing uses quantum bits (qubits) that can exist in multiple
//  states simultaneously, enabling certain calculations to be performed
//  exponentially faster than classical computers."
```

### `streamText`

Stream the response token by token. This is where the SDK shines:

```typescript
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

const result = streamText({
  model: openai('gpt-4o'),
  prompt: 'Write a haiku about programming.',
});

// Stream to the client
for await (const textPart of result.textStream) {
  process.stdout.write(textPart);
  // Each token appears in real-time
}
```

The `textStream` is an async iterable. Each chunk is a piece of the response as it's generated. This is the foundation for the real-time AI experiences that the SDK enables.

### `tool`

Define tools that the model can call. This is the equivalent of OpenAI function calling, but provider-agnostic:

```typescript
import { tool } from 'ai';
import { z } from 'zod';

const weatherTool = tool({
  description: 'Get the current weather for a city. City must include country code.',
  parameters: z.object({
    city: z.string().describe('City name with country code, e.g. "Tokyo, JP"'),
    units: z.enum(['celsius', 'fahrenheit']).optional()
      .describe('Temperature unit. Defaults to celsius.'),
  }),
  execute: async ({ city, units = 'celsius' }) => {
    // In production: call a real weather API
    const temp = units === 'celsius' ? 22 : 72;
    return {
      city,
      temperature: temp,
      units,
      condition: 'partly cloudy',
      humidity: 65,
    };
  },
});
```

The `tool` function uses Zod for schema validation. The `parameters` define the input schema. The `execute` function runs when the model calls the tool. The model never sees the implementation — only the description and parameter schema.

### The Agent Loop: `streamText` with `maxSteps`

The Vercel AI SDK does not export a separate `agent()` function. The agent loop is `streamText` (or `generateText`) with `maxSteps` set. When `maxSteps > 1`, the SDK automatically executes tool calls and feeds the results back to the model, repeating until the model produces a final answer or `maxSteps` is reached:

```typescript
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

const result = streamText({
  model: openai('gpt-4o'),
  tools: {
    getWeather: weatherTool,
    searchKnowledgeBase: knowledgeBaseTool,
    createTicket: ticketTool,
  },
  system: `
    You are a customer support agent.
    
    Process:
    1. Understand the customer's issue
    2. Search the knowledge base for relevant information
    3. If the answer is in the knowledge base, provide it
    4. If not, create a support ticket
    
    Always be empathetic and professional.
  `,
  messages: [
    { role: 'user', content: 'My package says delivered but I never received it.' }
  ],
  maxSteps: 10, // Maximum agent loop iterations
  onStepFinish: (step) => {
    console.log(`Step ${step.stepNumber}:`, step.toolCalls?.map(tc => tc.toolName));
  },
});

let text = '';
for await (const chunk of result.textStream) {
  text += chunk;
}

const allSteps = await result.steps;
console.log(`Completed in ${allSteps.length} step(s)`);
```

Each element in `steps` contains the text, tool calls, tool results, token usage, and finish reason for that iteration — giving you the full trace of the agent's decision-making.

---

## Building a Complete Agent

Let's build a complete customer support agent with the Vercel AI SDK:

```typescript
// agent.ts
import { streamText, tool } from 'ai';
import { openai } from '@ai-sdk/openai';
import { z } from 'zod';

// ──── Tools ────────────────────────────────────────────

const searchKnowledgeBase = tool({
  description: 'Search the support knowledge base for articles related to the query.',
  parameters: z.object({
    query: z.string().describe('The search query. Be specific.'),
    category: z.enum(['orders', 'returns', 'billing', 'technical', 'general'])
      .optional().describe('Filter by category.'),
  }),
  execute: async ({ query, category }) => {
    // In production: search your vector database
    const results = await vectorDb.search(query, { 
      filter: category ? { category } : undefined,
      limit: 5 
    });
    
    return results.map(r => ({
      title: r.title,
      content: r.content,
      relevance: r.score,
      url: r.url,
    }));
  },
});

const lookupOrder = tool({
  description: 'Look up a customer order by order number or email address.',
  parameters: z.object({
    orderNumber: z.string().optional()
      .describe('The order number, e.g. "ORD-12345"'),
    email: z.string().email().optional()
      .describe('Customer email address'),
  }),
  execute: async ({ orderNumber, email }) => {
    if (!orderNumber && !email) {
      return { error: 'Either order number or email is required.' };
    }
    
    // In production: query your order database
    const orders = await db.orders.find({
      ...(orderNumber && { orderNumber }),
      ...(email && { email }),
    });
    
    if (orders.length === 0) {
      return { found: false, message: 'No orders found matching your criteria.' };
    }
    
    return {
      found: true,
      orders: orders.map(o => ({
        orderNumber: o.id,
        date: o.date,
        status: o.status,
        items: o.items.length,
        total: o.total,
        tracking: o.tracking,
      })),
    };
  },
});

const createTicket = tool({
  description: 'Create a support ticket for issues that cannot be resolved immediately.',
  parameters: z.object({
    subject: z.string().describe('Brief summary of the issue'),
    description: z.string().describe('Detailed description of the problem'),
    priority: z.enum(['low', 'medium', 'high', 'urgent'])
      .describe('Issue priority'),
    customerEmail: z.string().email(),
    relatedOrderNumber: z.string().optional(),
  }),
  execute: async (params) => {
    // In production: create ticket in your support system
    const ticket = await supportSystem.createTicket(params);
    return {
      ticketId: ticket.id,
      status: 'created',
      estimatedResponseTime: '2-4 hours',
      confirmationSent: true,
    };
  },
});

// ──── Agent factory ──────────────────────────────────────

/**
 * Create a streaming support agent response.
 * The SDK drives the tool-call loop automatically via maxSteps.
 */
export function runSupportAgent(userMessage: string) {
  return streamText({
    model: openai('gpt-4o'),
    tools: {
      searchKnowledgeBase,
      lookupOrder,
      createTicket,
    },
    system: `
      You are a helpful customer support agent for Acme Corp.
      
      YOUR PROCESS:
      1. Understand the customer's issue completely before taking action
      2. If they mention an order, look it up FIRST
      3. Search the knowledge base for relevant help articles
      4. Answer using the knowledge base if possible
      5. Only create a ticket if the issue cannot be resolved immediately
      
      TONE:
      - Empathetic: Acknowledge frustration before solving problems
      - Professional: Use clear, concise language
      - Helpful: Offer additional relevant information proactively
      
      IMPORTANT RULES:
      - Never make up order details. Always look them up.
      - If the knowledge base has the answer, cite the article title.
      - If you create a ticket, tell the customer when to expect a response.
      - If you're unsure, say so and escalate.
    `,
    messages: [{ role: 'user', content: userMessage }],
    maxSteps: 10,
    onStepFinish: (step) => {
      // Observability hook: log each step
      console.log(`Step ${step.stepNumber}:`, {
        text: step.text?.substring(0, 100),
        toolCalls: step.toolCalls?.map(tc => tc.toolName),
        // usage has promptTokens, completionTokens, totalTokens
        tokens: step.usage?.totalTokens,
      });
    },
  });
}

// ──── Usage ─────────────────────────────────────────────

// With streaming to the frontend
export async function handleSupportRequest(userMessage: string) {
  // The streamText result can be piped directly to a Response
  return runSupportAgent(userMessage);
}
```

---

## Streaming to the Frontend with `useChat`

The Vercel AI SDK's killer feature is the bridge between backend AI and frontend UI. The `useChat` hook handles the entire streaming lifecycle:

### Backend: API Route

```typescript
// app/api/chat/route.ts (Next.js App Router)
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

export async function POST(req: Request) {
  const { messages } = await req.json();
  
  const result = streamText({
    model: openai('gpt-4o'),
    system: 'You are a helpful assistant.',
    messages,
    tools: {
      getWeather: weatherTool,
      searchKnowledgeBase: knowledgeBaseTool,
    },
  });
  
  // Return the stream as a response
  return result.toDataStreamResponse();
}
```

### Frontend: React Component

```typescript
// app/chat/page.tsx (Next.js App Router)
'use client';

import { useChat } from '@ai-sdk/react';

export default function ChatPage() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
    // The hook handles:
    // - Sending messages to the API
    // - Receiving streaming responses
    // - Updating the UI in real-time
    // - Optimistic updates
    // - Error handling
    // - Loading states
  });
  
  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="message-content">
              {message.content}
            </div>
            {/* Tool calls are rendered automatically */}
            {message.toolInvocations?.map(toolInvocation => (
              <div key={toolInvocation.toolCallId} className="tool-call">
                {toolInvocation.toolName}: {toolInvocation.state}
                {toolInvocation.state === 'result' && (
                  <pre>{JSON.stringify(toolInvocation.result, null, 2)}</pre>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
      
      {isLoading && <div className="thinking">Agent is thinking...</div>}
      
      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Ask me anything..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </div>
  );
}
```

This is the magic of the Vercel AI SDK. With ~50 lines of frontend code, you get:
- Real-time streaming of AI responses
- Tool call visualization
- Loading states
- Error handling
- Message history management
- Optimistic UI updates

### What `useChat` Handles Automatically

| Feature | How It Works |
|:---|:---|
| **Streaming** | Tokens appear in the UI as they're generated |
| **Tool calls** | Tool invocations are rendered with their results |
| **Message history** | All messages are stored and sent with each request |
| **Loading states** | `isLoading` is true while the AI is responding |
| **Error recovery** | Failed requests can be retried |
| **Optimistic updates** | User messages appear instantly before the API responds |
| **Abort** | Users can stop generation mid-stream |

---

## Provider-Agnostic Design

The Vercel AI SDK's unified provider interface is one of its strongest features. Switch models by changing one import:

```typescript
import { generateText } from 'ai';

// OpenAI
import { openai } from '@ai-sdk/openai';
const result = await generateText({ model: openai('gpt-4o'), prompt });

// Anthropic
import { anthropic } from '@ai-sdk/anthropic';
const result = await generateText({ model: anthropic('claude-3-5-sonnet-20241022'), prompt });

// Google
import { google } from '@ai-sdk/google';
const result = await generateText({ model: google('gemini-1.5-pro'), prompt });

// Groq (fast inference)
import { groq } from '@ai-sdk/groq';
const result = await generateText({ model: groq('llama-3.1-70b-versatile'), prompt });

// Mistral
import { mistral } from '@ai-sdk/mistral';
const result = await generateText({ model: mistral('mistral-large-latest'), prompt });

// All use the SAME API
```

The `generateText` and `streamText` functions work identically regardless of provider. Tools, system prompts, and messages are all handled consistently.

### Provider-Specific Configuration

When you need provider-specific features, the SDK supports them:

```typescript
const result = await generateText({
  model: openai('gpt-4o'),
  prompt: 'Analyze this data...',
  // OpenAI-specific options
  providerOptions: {
    openai: {
      reasoningEffort: 'high',  // For o1 models
      responseFormat: { type: 'json_object' },
    },
  },
});

const result2 = await generateText({
  model: anthropic('claude-3-5-sonnet-20241022'),
  prompt: 'Analyze this data...',
  // Anthropic-specific options
  providerOptions: {
    anthropic: {
      maxTokens: 4096,
      thinking: { type: 'enabled', budgetTokens: 2000 },
    },
  },
});
```

---

## The `maxSteps` Pattern: Agent Loop with Escape Hatch

The Vercel AI SDK's `agent` function uses `maxSteps` to prevent infinite loops. But sometimes you need more control:

```typescript
// Custom agent loop with the Vercel AI SDK
async function customAgentLoop(userInput: string) {
  const messages: Message[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: userInput },
  ];
  
  let stepCount = 0;
  const maxSteps = 15;
  let finalAnswer = '';
  
  while (stepCount < maxSteps) {
    stepCount++;
    
    const result = await generateText({
      model: openai('gpt-4o'),
      messages,
      tools: {
        searchKnowledgeBase,
        lookupOrder,
        createTicket,
      },
      maxSteps: 1, // Drive one tool round per loop iteration
    });
    
    // Check for tool calls
    if (result.toolCalls && result.toolCalls.length > 0) {
      // Add the assistant turn that contains the tool call requests
      messages.push({
        role: 'assistant',
        content: result.text || '',
      });
      
      // Execute tools and add results in the required format
      const toolResultContent: Array<{
        type: 'tool-result';
        toolCallId: string;
        toolName: string;
        result: unknown;
      }> = [];
      for (const toolCall of result.toolCalls) {
        const toolResult = await executeToolCall(toolCall);
        toolResultContent.push({
          type: 'tool-result',
          toolCallId: toolCall.toolCallId, // note: toolCallId, not id
          toolName: toolCall.toolName,
          result: toolResult,
        });
      }
      messages.push({ role: 'tool', content: toolResultContent });
      
      continue; // Loop back for next iteration
    }
    
    // No tool calls — this is the final answer
    finalAnswer = result.text;
    break;
  }
  
  if (!finalAnswer) {
    finalAnswer = 'I was unable to complete the task within the time limit.';
  }
  
  return {
    answer: finalAnswer,
    steps: stepCount,
    messages,
  };
}
```

This gives you the same control as a from-scratch agent loop, with the Vercel AI SDK handling provider abstraction and tool execution.

---

## Streaming Agent Responses

The Vercel AI SDK can stream the entire agent process, not just the final answer:

```typescript
import { streamText } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();
  
  const result = streamText({
    model: openai('gpt-4o'),
    system: SUPPORT_AGENT_SYSTEM_PROMPT,
    messages,
    tools: {
      searchKnowledgeBase,
      lookupOrder,
      createTicket,
    },
    maxSteps: 10,
    
    // Callbacks for each step
    onStepFinish: async (step) => {
      // Log to observability
      await logAgentStep({
        stepNumber: step.stepNumber,
        text: step.text?.substring(0, 200),
        toolCalls: step.toolCalls?.map(tc => tc.toolName),
        tokens: step.usage?.totalTokens,
        finishReason: step.finishReason,
      });
    },
    
    // Called when the full response is complete
    onFinish: async (result) => {
      await logAgentCompletion({
        totalSteps: result.steps.length,
        // onFinish receives result.usage (not result.totalUsage)
        totalTokens: result.usage?.totalTokens,
        totalCost: calculateCost(result.usage),
      });
    },
  });
  
  return result.toDataStreamResponse();
}
```

The frontend receives updates as each step completes. The user sees the agent "thinking" in real-time.

---

## Comparing the Vercel AI SDK to From-Scratch

### What the Vercel AI SDK Gives You

1. **Provider abstraction**: One API for 10+ providers. Switch models with one line.
2. **Streaming infrastructure**: `textStream`, `toDataStreamResponse()`, `useChat` hook
3. **Tool system**: Zod-based schema validation, automatic type inference
4. **Agent loop**: `streamText` with `maxSteps` and `onStepFinish` callbacks
5. **Frontend integration**: React, Svelte, Vue, and Solid hooks
6. **Edge runtime**: Optimized for serverless and edge deployment
7. **TypeScript-native**: Full type safety from provider to frontend

### What You Still Control

1. **Tool implementation**: You write the `execute` function
2. **System prompt**: You define the agent's behavior
3. **Error handling**: You decide how to handle failures
4. **Observability**: You implement `onStepFinish` and `onFinish`
5. **Custom logic**: You can bypass the `agent()` function and write your own loop

### What You Give Up

1. **Python ecosystem**: The Vercel AI SDK is TypeScript-only. If your team is Python, look elsewhere.
2. **LangChain's integrations**: LangChain has 700+ integrations. The Vercel AI SDK has a focused set of high-quality providers.
3. **Framework flexibility**: The SDK is optimized for Next.js and the Vercel platform. It works elsewhere but shines on Vercel.

---

## When the Vercel AI SDK Is the Right Choice

| Scenario | Why |
|:---|:---|
| **You're building a full-stack TypeScript app** | Native integration with React, Next.js, Svelte |
| **Streaming UX is critical** | Best-in-class streaming from model to UI |
| **You want to switch providers easily** | Unified API across 10+ providers |
| **You're deploying on serverless/edge** | Optimized for edge runtime |
| **Your team prefers TypeScript** | First-class TypeScript support |
| **You want a lightweight framework** | Small API surface, less to learn |

## When It Isn't

| Scenario | Why Not |
|:---|:---|
| **Your backend is Python** | The SDK is TypeScript-only |
| **You need LangChain's integrations** | The SDK doesn't have 700+ connectors |
| **You need complex workflow orchestration** | LangGraph is better for graph-based workflows |
| **You want Python's ML ecosystem** | NumPy, Pandas, scikit-learn are Python |
| **Your team doesn't know TypeScript** | Learning TypeScript + AI is a lot |

---

## Common Pitfalls

- **"I use the Vercel AI SDK for everything, including non-streaming tasks"**: The SDK excels at streaming and agent workflows. For simple one-shot completions, the OpenAI SDK is simpler and has less overhead.
- **"I ignore `maxSteps` and my agent loops forever"**: Always set `maxSteps`. Even well-designed agents can get stuck. 10 is a good default for most use cases.
- **"I don't implement `onStepFinish` for observability"**: Without step-level logging, debugging a multi-step agent is nearly impossible. Always implement at least basic step logging.
- **"I mix the Vercel AI SDK with LangChain"**: These frameworks have different philosophies. Mixing them creates confusion about which abstraction owns what. Pick one for your agent's core loop.
- **"I deploy without testing edge runtime compatibility"**: Not all Node.js packages work on edge runtimes. Test your tool implementations in the edge environment before deploying.
- **"I forget that `useChat` manages state on the client"**: Message history lives in the browser. If the user refreshes, it's gone (unless you persist it). Plan for state persistence.

## What's Next

You've covered all four major framework approaches. Next: harness engineering — the control systems that make agents reliable in production. Guardrails, routing, retries, and human-in-the-loop.
→ [The Harness Mindset](../07-harness-engineering/01-the-harness-mindset.md)