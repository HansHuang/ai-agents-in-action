"use client";

/**
 * page.tsx — Next.js App Router chat page (bonus file).
 *
 * A minimal but complete chat page that wires the /api/chat route
 * to the full-featured ChatContainer component.
 *
 * Render tree:
 *   page.tsx → ChatContainer (chat-component.tsx)
 *
 * To use in your Next.js project:
 *   1. Copy this file to app/chat/page.tsx
 *   2. Copy chat-component.tsx to components/chat/ChatContainer.tsx
 *   3. Copy route.ts to app/api/chat/route.ts
 *   4. Run: npm run dev
 */

import ChatContainer from "./chat-component";

export const metadata = {
  title: "Acme Support — AI Agent",
  description: "Get instant help from our AI support agent.",
};

export default function ChatPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-2xl">
        <h1 className="mb-4 text-center text-2xl font-semibold text-gray-800">
          Acme Customer Support
        </h1>
        <ChatContainer apiUrl="/api/chat" />
      </div>
    </main>
  );
}
