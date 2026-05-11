/**
 * Recovery Manager — goal detection and conversation health recovery.
 *
 * Diagnoses conversation drift, stalls, and repetition loops, then
 * generates targeted interventions.
 * See: docs/04-context-engineering/04-multi-turn-context-management.md
 */

import OpenAI from "openai";

export type ConversationIssueType =
  | "repetition_loop"
  | "topic_drift"
  | "stuck_on_sub_task"
  | "context_overflow"
  | "healthy";

export interface ConversationIssue {
  type: ConversationIssueType;
  description: string;
  severity: "low" | "medium" | "high";
}

export interface GoalResult {
  primaryGoal: string;
  subGoals: string[];
  completedGoals: string[];
  confidence: number;
}

export interface RecoveryPlan {
  issue: ConversationIssue;
  intervention: string;
  injectedMessage?: OpenAI.Chat.ChatCompletionSystemMessageParam;
}

// ---------------------------------------------------------------------------
// Goal Detector
// ---------------------------------------------------------------------------

const MODEL = "gpt-4o-mini";

export class GoalDetector {
  constructor(private client: OpenAI) {}

  /** Extract and track user goals from conversation messages. */
  async detect(
    messages: OpenAI.Chat.ChatCompletionMessageParam[]
  ): Promise<GoalResult> {
    const history = messages
      .filter((m) => m.role !== "system")
      .slice(-10)
      .map((m) => `${m.role}: ${typeof m.content === "string" ? m.content : ""}`)
      .join("\n");

    const resp = await this.client.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content:
            'Analyze the conversation and extract: primaryGoal, subGoals (array), completedGoals (array), confidence (0-1). Return JSON with exactly those keys.',
        },
        { role: "user", content: history },
      ],
      response_format: { type: "json_object" },
      temperature: 0,
    });

    try {
      return JSON.parse(resp.choices[0].message.content ?? "{}") as GoalResult;
    } catch {
      return { primaryGoal: "unknown", subGoals: [], completedGoals: [], confidence: 0 };
    }
  }
}

// ---------------------------------------------------------------------------
// Recovery Manager
// ---------------------------------------------------------------------------

export class RecoveryManager {
  constructor(private client: OpenAI) {}

  /** Diagnose conversation health and return a recovery plan if needed. */
  async diagnose(
    messages: OpenAI.Chat.ChatCompletionMessageParam[]
  ): Promise<RecoveryPlan> {
    const issue = this.heuristicDiagnose(messages);

    if (issue.type === "healthy") {
      return { issue, intervention: "No intervention needed." };
    }

    const intervention = await this.generateIntervention(issue, messages);
    return {
      issue,
      intervention,
      injectedMessage: {
        role: "system",
        content: `[Recovery] ${intervention}`,
      },
    };
  }

  private heuristicDiagnose(
    messages: OpenAI.Chat.ChatCompletionMessageParam[]
  ): ConversationIssue {
    const userMessages = messages
      .filter((m) => m.role === "user")
      .map((m) => (typeof m.content === "string" ? m.content.toLowerCase() : ""));

    // Detect repetition
    if (userMessages.length >= 3) {
      const last3 = userMessages.slice(-3);
      if (last3[0] === last3[1] || last3[1] === last3[2]) {
        return { type: "repetition_loop", description: "User is repeating the same message", severity: "high" };
      }
    }

    // Detect long conversations (potential context overflow)
    if (messages.length > 40) {
      return { type: "context_overflow", description: "Conversation exceeds 40 turns", severity: "medium" };
    }

    return { type: "healthy", description: "Conversation is progressing normally", severity: "low" };
  }

  private async generateIntervention(
    issue: ConversationIssue,
    messages: OpenAI.Chat.ChatCompletionMessageParam[]
  ): Promise<string> {
    const resp = await this.client.chat.completions.create({
      model: MODEL,
      messages: [
        {
          role: "system",
          content:
            "You generate brief conversation recovery interventions. Return a single sentence instruction for the assistant.",
        },
        {
          role: "user",
          content: `Issue: ${issue.type} — ${issue.description}. Last 3 turns: ${messages
            .slice(-3)
            .map((m) => `${m.role}: ${typeof m.content === "string" ? m.content.slice(0, 80) : ""}`)
            .join(" | ")}`,
        },
      ],
      temperature: 0,
      max_tokens: 100,
    });
    return resp.choices[0].message.content?.trim() ?? "Redirect the conversation to the main goal.";
  }
}
