"use client";

/**
 * chat-component.tsx — Full-featured streaming chat UI using useChat.
 *
 * Features:
 *   - Real-time token streaming with blinking cursor
 *   - Tool call cards (collapsible) showing arguments + results
 *   - Thinking indicator while the agent is running
 *   - Error banner with retry
 *   - "Stop generating" button during streaming
 *   - Message persistence in localStorage
 *   - "Clear conversation" button
 *   - Enter to send, Shift+Enter for newline
 *   - Auto-scroll to bottom
 *   - Responsive Tailwind CSS layout
 *   - ARIA labels and screen-reader announcements
 */

import {
  useChat,
  type Message,
  type ToolInvocation,
} from "@ai-sdk/react";
import { useEffect, useRef, useState, useCallback, KeyboardEvent } from "react";

// ──── Types ───────────────────────────────────────────────────────────────────

interface ChatContainerProps {
  /** The API endpoint URL. Defaults to /api/chat. */
  apiUrl?: string;
}

interface ToolCallCardProps {
  invocation: ToolInvocation;
}

interface MessageBubbleProps {
  message: Message;
  isStreaming: boolean;
}

// ──── Constants ───────────────────────────────────────────────────────────────

const STORAGE_KEY = "acme_chat_messages";
const EMPTY_STATE_TEXT =
  "Ask me anything about your orders, returns, or our products.";

// ──── ToolCallCard ────────────────────────────────────────────────────────────

function ToolCallCard({ invocation }: ToolCallCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isPending = invocation.state === "call";

  return (
    <div className="mt-2 rounded-lg border border-gray-200 bg-white text-xs shadow-sm">
      <button
        type="button"
        aria-expanded={isExpanded}
        aria-label={`${invocation.toolName} tool call details`}
        onClick={() => setIsExpanded((v) => !v)}
        className="flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left hover:bg-gray-50 transition-colors"
      >
        <span className="flex items-center gap-2 font-medium text-gray-700">
          {isPending ? (
            <span
              aria-hidden="true"
              className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"
            />
          ) : (
            <span aria-hidden="true" className="text-green-500">✓</span>
          )}
          <span className="font-mono">{invocation.toolName}</span>
          <span className="text-gray-400">{isPending ? "running…" : "done"}</span>
        </span>
        <span aria-hidden="true" className="text-gray-400">
          {isExpanded ? "▲" : "▼"}
        </span>
      </button>

      {isExpanded && (
        <div className="border-t border-gray-100 px-3 py-2 space-y-2">
          <div>
            <p className="font-semibold text-gray-500 uppercase tracking-wide mb-1">
              Arguments
            </p>
            <pre className="overflow-x-auto rounded bg-gray-50 p-2 text-gray-700">
              {JSON.stringify(invocation.args, null, 2)}
            </pre>
          </div>

          {invocation.state === "result" && (
            <div>
              <p className="font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Result
              </p>
              <pre className="overflow-x-auto rounded bg-green-50 p-2 text-green-800">
                {JSON.stringify(invocation.result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ──── MessageBubble ───────────────────────────────────────────────────────────

function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
      role="listitem"
    >
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "rounded-br-sm bg-blue-600 text-white"
            : "rounded-bl-sm bg-white text-gray-800 shadow-sm border border-gray-100"
        }`}
      >
        {/* Text content */}
        <p className="whitespace-pre-wrap leading-relaxed">
          {message.content}
          {/* Streaming cursor: only on the last assistant message while streaming */}
          {!isUser && isStreaming && (
            <span
              aria-hidden="true"
              className="ml-0.5 inline-block h-4 w-0.5 animate-[blink_1s_step-end_infinite] bg-current align-middle"
            />
          )}
        </p>

        {/* Tool call cards */}
        {message.toolInvocations && message.toolInvocations.length > 0 && (
          <div className="mt-2 space-y-1">
            {message.toolInvocations.map((inv) => (
              <ToolCallCard key={inv.toolCallId} invocation={inv} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ──── ThinkingIndicator ───────────────────────────────────────────────────────

function ThinkingIndicator() {
  return (
    <div
      className="flex justify-start"
      role="status"
      aria-live="polite"
      aria-label="Agent is thinking"
    >
      <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-white px-4 py-3 shadow-sm border border-gray-100">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            aria-hidden="true"
            className="h-2 w-2 animate-bounce rounded-full bg-gray-400"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

// ──── ChatContainer ───────────────────────────────────────────────────────────

/**
 * Full-featured chat component powered by the Vercel AI SDK useChat hook.
 */
export default function ChatContainer({
  apiUrl = "/api/chat",
}: ChatContainerProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [announcerText, setAnnouncerText] = useState("");

  // ── Load persisted messages ──────────────────────────────────────────────
  const getInitialMessages = (): Message[] => {
    if (typeof window === "undefined") return [];
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? (JSON.parse(stored) as Message[]) : [];
    } catch {
      return [];
    }
  };

  // ── useChat hook ─────────────────────────────────────────────────────────
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    error,
    stop,
    reload,
    setMessages,
  } = useChat({
    api: apiUrl,
    initialMessages: getInitialMessages(),
    onFinish: (message) => {
      // Announce new assistant message for screen readers
      setAnnouncerText(
        `Agent replied: ${message.content.substring(0, 80)}${
          message.content.length > 80 ? "…" : ""
        }`
      );
    },
  });

  // ── Persist messages ─────────────────────────────────────────────────────
  useEffect(() => {
    if (messages.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
      } catch {
        // Storage quota exceeded — ignore silently
      }
    }
  }, [messages]);

  // ── Auto-scroll to bottom ────────────────────────────────────────────────
  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isLoading]);

  // ── Clear conversation ───────────────────────────────────────────────────
  const clearConversation = useCallback(() => {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
    inputRef.current?.focus();
  }, [setMessages]);

  // ── Keyboard: Enter sends, Shift+Enter inserts newline ───────────────────
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && input.trim()) {
        handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
      }
    }
  };

  // ── Derived state ─────────────────────────────────────────────────────────
  const lastMessageIsAssistant =
    messages.length > 0 && messages[messages.length - 1].role === "assistant";
  const showThinking = isLoading && !lastMessageIsAssistant;

  return (
    <div className="flex h-[600px] flex-col rounded-2xl border border-gray-200 bg-gray-50 shadow-lg overflow-hidden">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-green-400" aria-hidden="true" />
          <h2 className="font-semibold text-gray-800">Support Agent</h2>
        </div>
        <button
          type="button"
          onClick={clearConversation}
          className="rounded-lg px-3 py-1 text-sm text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
          aria-label="Clear conversation history"
        >
          Clear
        </button>
      </div>

      {/* ── Error banner ───────────────────────────────────────────────── */}
      {error && (
        <div
          role="alert"
          className="flex items-center justify-between gap-2 bg-red-50 px-4 py-2 text-sm text-red-700 border-b border-red-100"
        >
          <span>Something went wrong. Please try again.</span>
          <button
            type="button"
            onClick={() => reload()}
            className="rounded px-2 py-0.5 font-medium underline hover:no-underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* ── Message list ───────────────────────────────────────────────── */}
      <div
        ref={listRef}
        role="list"
        aria-label="Chat messages"
        aria-live="polite"
        aria-atomic="false"
        className="flex-1 overflow-y-auto p-4 space-y-3"
      >
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-center text-gray-400 text-sm max-w-xs">
              {EMPTY_STATE_TEXT}
            </p>
          </div>
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              isStreaming={
                isLoading &&
                index === messages.length - 1 &&
                message.role === "assistant"
              }
            />
          ))
        )}

        {showThinking && <ThinkingIndicator />}
      </div>

      {/* Screen-reader live region */}
      <span className="sr-only" aria-live="assertive" aria-atomic="true">
        {announcerText}
      </span>

      {/* ── Input area ─────────────────────────────────────────────────── */}
      <div className="border-t border-gray-200 bg-white p-3">
        <form
          onSubmit={handleSubmit}
          className="flex items-end gap-2"
          aria-label="Send a message"
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={onKeyDown}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={isLoading}
            aria-label="Message input"
            className="flex-1 resize-none rounded-xl border border-gray-200 bg-gray-50 px-4 py-2.5 text-sm text-gray-800 placeholder-gray-400 focus:border-blue-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-100 disabled:opacity-50 transition-colors max-h-32"
            style={{ overflowY: "auto" }}
          />

          {isLoading ? (
            <button
              type="button"
              onClick={stop}
              aria-label="Stop generating"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-red-500 text-white hover:bg-red-600 transition-colors"
            >
              <svg
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-4 w-4"
              >
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              aria-label="Send message"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <svg
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-4 w-4 translate-x-0.5"
              >
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
