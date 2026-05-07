/**
 * State Manager — explicit conversation state that survives context truncation.
 *
 * TypeScript port of code/python/05-context-assembly/state_manager.py
 *
 * The message list is temporary.  State is durable.
 *
 * Every turn updates ConversationState, which is then injected back into
 * the system prompt so the agent never loses track of goals, collected
 * user information, or recommendations it has already made.
 *
 * See: docs/04-context-engineering/04-multi-turn-context-management.md
 */

// ---------------------------------------------------------------------------
// ConversationState
// ---------------------------------------------------------------------------

export interface ConversationStateData {
  currentGoal:             string | null;
  subtasksCompleted:       string[];
  subtasksPending:         string[];
  goalSetAtTurn:           number;
  userName:                string | null;
  userPreferences:         Record<string, string>;
  userProvidedInfo:        Record<string, string>;
  agentRecommendations:    string[];
  agentQuestionsAsked:     string[];
  agentMode:               string;
  turnsSinceGoalMentioned: number;
  userFrustrationSignals:  number;
  topicChanges:            string[];
  turnCount:               number;
}

export class ConversationState implements ConversationStateData {
  currentGoal:             string | null = null;
  subtasksCompleted:       string[]      = [];
  subtasksPending:         string[]      = [];
  goalSetAtTurn:           number        = 0;
  userName:                string | null = null;
  userPreferences:         Record<string, string> = {};
  userProvidedInfo:        Record<string, string> = {};
  agentRecommendations:    string[]      = [];
  agentQuestionsAsked:     string[]      = [];
  agentMode:               string        = "general";
  turnsSinceGoalMentioned: number        = 0;
  userFrustrationSignals:  number        = 0;
  topicChanges:            string[]      = [];
  turnCount:               number        = 0;

  /**
   * Generate a compact state summary for prompt injection.
   * Intentionally terse — it consumes part of every system-prompt token budget.
   */
  toPromptContext(): string {
    const parts: string[] = [];

    if (this.userName)
      parts.push(`User: ${this.userName}`);

    if (this.currentGoal)
      parts.push(`Current goal: ${this.currentGoal}`);

    if (this.subtasksCompleted.length)
      parts.push("Completed: " + this.subtasksCompleted.join("; "));

    if (this.subtasksPending.length)
      parts.push("Pending: " + this.subtasksPending.join("; "));

    if (this.agentRecommendations.length) {
      const recs = this.agentRecommendations.slice(-5);
      parts.push("Previous recommendations: " + recs.join("; "));
    }

    if (this.agentQuestionsAsked.length) {
      const qs = this.agentQuestionsAsked.slice(-5);
      parts.push("Already asked about: " + qs.join("; "));
    }

    if (Object.keys(this.userProvidedInfo).length) {
      const infoStr = Object.entries(this.userProvidedInfo)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ");
      parts.push(`User has provided: ${infoStr}`);
    }

    if (Object.keys(this.userPreferences).length) {
      const prefStr = Object.entries(this.userPreferences)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ");
      parts.push(`User preferences: ${prefStr}`);
    }

    if (this.agentMode !== "general")
      parts.push(`Agent mode: ${this.agentMode}`);

    return parts.join("\n");
  }

  /** Serialise state for persistence. */
  toDict(): ConversationStateData {
    return {
      currentGoal:             this.currentGoal,
      subtasksCompleted:       [...this.subtasksCompleted],
      subtasksPending:         [...this.subtasksPending],
      goalSetAtTurn:           this.goalSetAtTurn,
      userName:                this.userName,
      userPreferences:         { ...this.userPreferences },
      userProvidedInfo:        { ...this.userProvidedInfo },
      agentRecommendations:    [...this.agentRecommendations],
      agentQuestionsAsked:     [...this.agentQuestionsAsked],
      agentMode:               this.agentMode,
      turnsSinceGoalMentioned: this.turnsSinceGoalMentioned,
      userFrustrationSignals:  this.userFrustrationSignals,
      topicChanges:            [...this.topicChanges],
      turnCount:               this.turnCount,
    };
  }

  /** Deserialise state from a persistence dictionary. */
  static fromDict(data: Partial<ConversationStateData>): ConversationState {
    const s = new ConversationState();
    s.currentGoal             = data.currentGoal             ?? null;
    s.subtasksCompleted       = data.subtasksCompleted        ?? [];
    s.subtasksPending         = data.subtasksPending          ?? [];
    s.goalSetAtTurn           = data.goalSetAtTurn            ?? 0;
    s.userName                = data.userName                 ?? null;
    s.userPreferences         = data.userPreferences          ?? {};
    s.userProvidedInfo        = data.userProvidedInfo         ?? {};
    s.agentRecommendations    = data.agentRecommendations     ?? [];
    s.agentQuestionsAsked     = data.agentQuestionsAsked      ?? [];
    s.agentMode               = data.agentMode                ?? "general";
    s.turnsSinceGoalMentioned = data.turnsSinceGoalMentioned  ?? 0;
    s.userFrustrationSignals  = data.userFrustrationSignals   ?? 0;
    s.topicChanges            = data.topicChanges             ?? [];
    s.turnCount               = data.turnCount                ?? 0;
    return s;
  }
}

// ---------------------------------------------------------------------------
// Information extraction helpers
// ---------------------------------------------------------------------------

interface InfoPatterns {
  key:     string;
  pattern: RegExp;
}

const INFO_PATTERNS: InfoPatterns[] = [
  { key: "order_number", pattern: /\border\s*(?:number|#|num)?\s*(?:is\s*)?[:#]?\s*([A-Z0-9\-]{4,20})/i },
  { key: "name",         pattern: /\b(?:my\s+name\s+is|i(?:'m| am)\s+called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i },
  { key: "budget",       pattern: /\b(?:budget|spend|cost).*?[\$£€]?\s*([0-9][0-9,]*(?:\.[0-9]{2})?)\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)?/i },
  { key: "preference",   pattern: /\bi\s+prefer\s+([^.!?]{3,60})/i },
  { key: "email",        pattern: /\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b/ },
];

const FRUSTRATION_RE = /\b(?:already\s+told|said\s+(?:this|that)|you\s+asked\s+(?:me\s+)?(?:that|this|about))\b|\b(?:again\??|for\s+the\s+\w+\s+time)\b|\b(?:stop\s+repeating|stop\s+asking|can.t\s+you|why\s+(?:do|are|is|can.t))\b|\b(?:frustrated|annoyed|useless|terrible|awful)\b|!{2,}/i;

const GOAL_KEYWORDS_RE = /\b(?:i\s+(?:need|want|would\s+like|must|have\s+to)|(?:please|can\s+you|could\s+you|help\s+me\s+(?:to|with)?)\s+|(?:my\s+goal|the\s+goal|objective|task)\s+is)\b/i;

function extractUserInfo(message: string): Record<string, string> {
  const found: Record<string, string> = {};
  for (const { key, pattern } of INFO_PATTERNS) {
    const m = message.match(pattern);
    if (m && !(key in found)) {
      found[key] = m[1].trim().replace(/[,.]$/, "");
    }
  }
  return found;
}

function hasFrustration(message: string): boolean {
  return FRUSTRATION_RE.test(message);
}

function goalRelevance(message: string, currentGoal: string | null): number {
  if (!currentGoal) return 1.0;
  const goalWords = new Set((currentGoal.toLowerCase().match(/\b\w{4,}\b/g) ?? []));
  const msgWords  = new Set((message.toLowerCase().match(/\b\w{4,}\b/g) ?? []));
  if (goalWords.size === 0) return 1.0;
  let overlap = 0;
  for (const w of goalWords) if (msgWords.has(w)) overlap++;
  return overlap / goalWords.size;
}

// ---------------------------------------------------------------------------
// StateManager
// ---------------------------------------------------------------------------

export interface UserTurnChanges {
  goalDetected:       boolean;
  infoExtracted:      Record<string, string>;
  frustrationDetected: boolean;
  topicChanged:       boolean;
  turn:               number;
}

export interface AgentTurnChanges {
  recommendationsAdded: number;
  questionsAsked:       number;
  subtasksCompleted:    number;
  goalComplete:         boolean;
}

export class StateManager {
  static readonly DRIFT_TURNS_THRESHOLD = 5;
  static readonly CHECKPOINT_INTERVAL   = 10;
  static readonly MAX_TRACKED_ITEMS     = 20;

  readonly state: ConversationState;

  constructor() {
    this.state = new ConversationState();
  }

  // ------------------------------------------------------------------
  // Goal management
  // ------------------------------------------------------------------

  setGoal(goal: string, subtasks: string[] = []): void {
    this.state.currentGoal              = goal;
    this.state.subtasksPending          = [...subtasks];
    this.state.subtasksCompleted        = [];
    this.state.goalSetAtTurn            = this.state.turnCount;
    this.state.turnsSinceGoalMentioned  = 0;
  }

  markSubtaskComplete(subtask: string): void {
    const idx = this.state.subtasksPending.indexOf(subtask);
    if (idx !== -1) {
      this.state.subtasksPending.splice(idx, 1);
      this.state.subtasksCompleted.push(subtask);
      return;
    }
    // Substring match
    const lower = subtask.toLowerCase();
    for (let i = 0; i < this.state.subtasksPending.length; i++) {
      const item = this.state.subtasksPending[i];
      if (lower.includes(item.toLowerCase()) || item.toLowerCase().includes(lower)) {
        this.state.subtasksPending.splice(i, 1);
        this.state.subtasksCompleted.push(item);
        return;
      }
    }
  }

  checkGoalDrift(): boolean {
    if (!this.state.currentGoal) return false;
    return this.state.turnsSinceGoalMentioned >= StateManager.DRIFT_TURNS_THRESHOLD;
  }

  checkGoalComplete(): boolean {
    if (!this.state.currentGoal) return false;
    if (this.state.subtasksPending.length === 0 && this.state.subtasksCompleted.length > 0)
      return true;
    return false;
  }

  // ------------------------------------------------------------------
  // Per-turn updates
  // ------------------------------------------------------------------

  processUserTurn(userMessage: string): UserTurnChanges {
    this.state.turnCount++;
    const changes: UserTurnChanges = {
      goalDetected:       false,
      infoExtracted:      {},
      frustrationDetected: false,
      topicChanged:       false,
      turn:               this.state.turnCount,
    };

    // Relevance / drift
    const relevance = goalRelevance(userMessage, this.state.currentGoal);
    if (relevance < 0.25 && this.state.currentGoal) {
      this.state.turnsSinceGoalMentioned++;
      if (this.state.turnsSinceGoalMentioned >= StateManager.DRIFT_TURNS_THRESHOLD) {
        const snippet = userMessage.slice(0, 80).trim();
        const last = this.state.topicChanges[this.state.topicChanges.length - 1];
        if (!last || last !== snippet) {
          this.state.topicChanges.push(snippet);
        }
        changes.topicChanged = true;
      }
    } else {
      this.state.turnsSinceGoalMentioned = 0;
    }

    if (GOAL_KEYWORDS_RE.test(userMessage))
      changes.goalDetected = true;

    // Info extraction
    const extracted = extractUserInfo(userMessage);
    if (Object.keys(extracted).length) {
      Object.assign(this.state.userProvidedInfo, extracted);
      if ("name" in extracted && !this.state.userName)
        this.state.userName = extracted["name"];
      changes.infoExtracted = extracted;
    }

    // Frustration
    if (hasFrustration(userMessage)) {
      this.state.userFrustrationSignals++;
      changes.frustrationDetected = true;
    }

    return changes;
  }

  processAgentTurn(
    agentResponse: string,
    toolCalls?: Array<{ name?: string; function?: { name: string } }>,
  ): AgentTurnChanges {
    const changes: AgentTurnChanges = {
      recommendationsAdded: 0,
      questionsAsked:       0,
      subtasksCompleted:    0,
      goalComplete:         false,
    };

    // Recommendations
    const recPattern = /\b(?:I\s+(?:recommend|suggest|advise|propose)|you\s+(?:should|could|might\s+want\s+to))\s+([^.!?]{5,120})/gi;
    let m: RegExpExecArray | null;
    while ((m = recPattern.exec(agentResponse)) !== null) {
      const rec = m[1].trim().replace(/[.,]$/, "");
      if (!this.state.agentRecommendations.includes(rec)) {
        this.state.agentRecommendations.push(rec);
        changes.recommendationsAdded++;
      }
    }
    if (this.state.agentRecommendations.length > StateManager.MAX_TRACKED_ITEMS)
      this.state.agentRecommendations = this.state.agentRecommendations.slice(-StateManager.MAX_TRACKED_ITEMS);

    // Questions asked
    const sentences = agentResponse.split(/(?<=[.!?])\s+/);
    for (const sentence of sentences) {
      if (sentence.includes("?")) {
        const q = sentence.trim().slice(0, 120);
        if (q && !this.state.agentQuestionsAsked.includes(q)) {
          this.state.agentQuestionsAsked.push(q);
          changes.questionsAsked++;
        }
      }
    }
    if (this.state.agentQuestionsAsked.length > StateManager.MAX_TRACKED_ITEMS)
      this.state.agentQuestionsAsked = this.state.agentQuestionsAsked.slice(-StateManager.MAX_TRACKED_ITEMS);

    // Tool calls → subtask completion
    if (toolCalls) {
      for (const call of toolCalls) {
        const fnName = (call.function?.name ?? call.name ?? "").replace(/_/g, " ").toLowerCase();
        if (!fnName) continue;
        for (const pending of [...this.state.subtasksPending]) {
          if (pending.toLowerCase().includes(fnName)) {
            this.markSubtaskComplete(pending);
            changes.subtasksCompleted++;
          }
        }
      }
    }

    if (this.checkGoalComplete()) changes.goalComplete = true;
    return changes;
  }

  // ------------------------------------------------------------------
  // Recovery actions
  // ------------------------------------------------------------------

  getRecoveryAction(): string | null {
    if (this.state.userFrustrationSignals >= 3)
      return "ask_user_to_clarify";

    if (this.checkGoalDrift())
      return "remind_goal";

    if (
      this.state.turnCount > 0 &&
      this.state.turnCount % StateManager.CHECKPOINT_INTERVAL === 0 &&
      this.state.currentGoal
    ) return "inject_checkpoint";

    if (
      this.state.currentGoal &&
      this.state.subtasksPending.length === 0 &&
      this.state.subtasksCompleted.length > 0
    ) return "summarize_progress";

    return null;
  }

  // ------------------------------------------------------------------
  // Prompt injection
  // ------------------------------------------------------------------

  buildSystemPromptWithState(basePrompt: string): string {
    const ctx = this.state.toPromptContext();
    if (!ctx) return basePrompt;
    return (
      `${basePrompt}\n\n` +
      `## Conversation State (maintained across turns)\n` +
      `${ctx}\n\n` +
      "Use this state to maintain continuity. " +
      "Do not repeat questions already asked. " +
      "Do not contradict previous recommendations."
    );
  }
}
