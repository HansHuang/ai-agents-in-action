/**
 * Session Manager — multi-session conversation management.
 *
 * TypeScript port of code/python/05-context-assembly/session_manager.py
 *
 * Handles session creation, expiry, branching, reset, and JSON persistence.
 *
 * See: docs/04-context-engineering/04-multi-turn-context-management.md
 */

import { randomUUID } from "crypto";
import { countTokens } from "./context_budget.js";
import { ProgressiveSummarizer, SummarizerData } from "./progressive_summarizer.js";
import { ConversationState, StateManager } from "./state_manager.js";

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export interface SessionData {
  sessionId:        string;
  userId:           string;
  createdAt:        number;
  lastActivity:     number;
  state:            ReturnType<ConversationState["toDict"]>;
  messages:         Array<{ role: string; content: string; tool_calls?: unknown[] }>;
  inheritedContext: string | null;
  isActive:         boolean;
  summarizer:       SummarizerData;
}

export class Session {
  readonly sessionId:  string;
  readonly userId:     string;
  readonly createdAt:  number;
  lastActivity:        number;
  readonly stateManager: StateManager;
  messages: Array<{ role: string; content: string; tool_calls?: unknown[] }> = [];
  readonly summarizer: ProgressiveSummarizer;
  inheritedContext: string | null = null;
  isActive: boolean = true;

  constructor(userId: string) {
    this.sessionId    = randomUUID();
    this.userId       = userId;
    this.createdAt    = Date.now() / 1000;
    this.lastActivity = Date.now() / 1000;
    this.stateManager = new StateManager();
    this.summarizer   = new ProgressiveSummarizer();
  }

  /** Convenience accessor for the session's ConversationState. */
  get state(): ConversationState {
    return this.stateManager.state;
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  isExpired(ttlMinutes: number): boolean {
    return (Date.now() / 1000 - this.lastActivity) > ttlMinutes * 60;
  }

  touch(): void {
    (this as { lastActivity: number }).lastActivity = Date.now() / 1000;
  }

  // ------------------------------------------------------------------
  // Message management
  // ------------------------------------------------------------------

  addUserMessage(message: string): void {
    this.touch();
    this.messages.push({ role: "user", content: message });
    this.stateManager.processUserTurn(message);
  }

  async addAgentMessage(
    message: string,
    toolCalls?: Array<{ name?: string; function?: { name: string } }>,
  ): Promise<void> {
    this.touch();
    const msg: { role: string; content: string; tool_calls?: unknown[] } = {
      role: "assistant",
      content: message,
    };
    if (toolCalls) msg.tool_calls = toolCalls;
    this.messages.push(msg);
    this.stateManager.processAgentTurn(message, toolCalls);

    // Feed turn to summarizer — find the preceding user message
    let userContent = "";
    for (let i = this.messages.length - 2; i >= 0; i--) {
      if (this.messages[i].role === "user") {
        userContent = this.messages[i].content;
        break;
      }
    }
    await this.summarizer.addTurn(userContent, message);
  }

  buildMessagesForLlm(
    systemPrompt: string = "",
    maxTokens: number = 100_000,
  ): Array<{ role: string; content: string }> {
    let augmented = this.stateManager.buildSystemPromptWithState(systemPrompt);
    const summaryCtx = this.summarizer.getContext();
    if (summaryCtx)
      augmented += `\n\n## Conversation History Summary\n${summaryCtx}`;
    if (this.inheritedContext)
      augmented += `\n\n## Inherited from Previous Session\n${this.inheritedContext}`;

    const systemMsg = { role: "system", content: augmented };
    const systemTokens = countTokens(augmented);
    const budget = maxTokens - systemTokens;

    const kept: Array<{ role: string; content: string }> = [];
    let running = 0;
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const msg = this.messages[i];
      const tokens = countTokens(msg.content ?? "");
      if (running + tokens > budget) break;
      kept.unshift({ role: msg.role, content: msg.content });
      running += tokens;
    }

    return [systemMsg, ...kept];
  }

  // ------------------------------------------------------------------
  // Serialisation
  // ------------------------------------------------------------------

  toDict(): SessionData {
    return {
      sessionId:        this.sessionId,
      userId:           this.userId,
      createdAt:        this.createdAt,
      lastActivity:     this.lastActivity,
      state:            this.state.toDict(),
      messages:         [...this.messages],
      inheritedContext: this.inheritedContext,
      isActive:         this.isActive,
      summarizer:       this.summarizer.toDict(),
    };
  }

  static fromDict(data: SessionData): Session {
    const session = new Session(data.userId);
    (session as { sessionId: string }).sessionId    = data.sessionId;
    (session as { createdAt: number }).createdAt    = data.createdAt;
    session.touch();
    (session as { lastActivity: number }).lastActivity = data.lastActivity;
    session.stateManager.state.currentGoal = ConversationState.fromDict(data.state).currentGoal;
    Object.assign(session.stateManager.state, ConversationState.fromDict(data.state));
    session.messages         = data.messages ?? [];
    session.inheritedContext = data.inheritedContext ?? null;
    (session as { isActive: boolean }).isActive = data.isActive ?? true;
    (session as { summarizer: ProgressiveSummarizer }).summarizer =
      ProgressiveSummarizer.fromDict(data.summarizer ?? {
        verbatim: [], layers: ["", "", ""], totalTurns: 0,
        verbatimTurns: 5, layerSize: 10, layerTokenLimit: 1500,
      });
    return session;
  }
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

export class SessionManager {
  readonly ttlMinutes:  number;
  readonly maxSessions: number;
  private _sessions:    Map<string, Session> = new Map();

  constructor(ttlMinutes: number = 60, maxSessions: number = 10_000) {
    this.ttlMinutes  = ttlMinutes;
    this.maxSessions = maxSessions;
  }

  // ------------------------------------------------------------------
  // Core session access
  // ------------------------------------------------------------------

  getSession(userId: string): Session {
    const existing = this._sessions.get(userId);
    if (existing && !existing.isExpired(this.ttlMinutes))
      return existing;
    return this.createSession(userId);
  }

  createSession(userId: string, inheritFrom?: string): Session {
    this._evictIfNeeded();
    const session = new Session(userId);

    if (inheritFrom) {
      const parent = this._sessions.get(inheritFrom);
      if (parent) {
        session.state.userName        = parent.state.userName;
        session.state.userPreferences = { ...parent.state.userPreferences };
        if (parent.state.currentGoal) {
          session.inheritedContext =
            `Previous conversation goal: ${parent.state.currentGoal}. ` +
            `Completed steps: ${parent.state.subtasksCompleted.join(", ") || "none"}.`;
        }
      }
    }

    this._sessions.set(userId, session);
    return session;
  }

  branchSession(
    userId: string,
    contextKeys: string[] = ["user_profile", "goal_summary"],
  ): Session {
    const parent = this._sessions.get(userId);
    this._evictIfNeeded();
    const newSession = new Session(userId);

    if (parent) {
      if (contextKeys.includes("user_profile") || contextKeys.includes("preferences")) {
        newSession.state.userName        = parent.state.userName;
        newSession.state.userPreferences = { ...parent.state.userPreferences };
      }
      if (contextKeys.includes("user_profile")) {
        newSession.state.userProvidedInfo = { ...parent.state.userProvidedInfo };
      }
      if (contextKeys.includes("goal_summary") && parent.state.currentGoal) {
        const completed = parent.state.subtasksCompleted.join(", ") || "none";
        newSession.inheritedContext =
          `Previous conversation goal: ${parent.state.currentGoal}. ` +
          `Completed: ${completed}.`;
      }
    }

    this._sessions.set(userId, newSession);
    return newSession;
  }

  resetSession(userId: string, keepIdentity: boolean = true): Session {
    const old = this._sessions.get(userId);
    this._evictIfNeeded();
    const session = new Session(userId);

    if (keepIdentity && old) {
      session.state.userName        = old.state.userName;
      session.state.userPreferences = { ...old.state.userPreferences };
    }

    this._sessions.set(userId, session);
    return session;
  }

  endSession(userId: string): void {
    const session = this._sessions.get(userId);
    if (session) {
      (session as { isActive: boolean }).isActive = false;
      this._sessions.delete(userId);
    }
  }

  // ------------------------------------------------------------------
  // Housekeeping
  // ------------------------------------------------------------------

  cleanupExpired(): number {
    let count = 0;
    for (const [uid, session] of this._sessions) {
      if (session.isExpired(this.ttlMinutes)) {
        this._sessions.delete(uid);
        count++;
      }
    }
    return count;
  }

  getActiveSessions(): number {
    let count = 0;
    for (const session of this._sessions.values())
      if (!session.isExpired(this.ttlMinutes)) count++;
    return count;
  }

  private _evictIfNeeded(): void {
    if (this._sessions.size >= this.maxSessions) {
      let oldestUid = "";
      let oldestTime = Infinity;
      for (const [uid, s] of this._sessions) {
        if (s.lastActivity < oldestTime) {
          oldestTime = s.lastActivity;
          oldestUid  = uid;
        }
      }
      if (oldestUid) this._sessions.delete(oldestUid);
    }
  }

  // ------------------------------------------------------------------
  // Persistence
  // ------------------------------------------------------------------

  persistSession(userId: string): string {
    const session = this._sessions.get(userId);
    if (!session) return "{}";
    return JSON.stringify(session.toDict());
  }

  restoreSession(userId: string, data: string): Session {
    const parsed = JSON.parse(data) as SessionData;
    const session = Session.fromDict(parsed);
    session.userId !== userId && ((session as { userId: string }).userId = userId);
    this._sessions.set(userId, session);
    return session;
  }
}
