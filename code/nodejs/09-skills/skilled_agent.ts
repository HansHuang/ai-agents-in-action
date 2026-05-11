/**
 * SkilledAgent — an agent that loads capabilities from a SkillRegistry.
 *
 * Instead of raw tools the agent uses Skills. Each Skill provides:
 *   - Its own OpenAI function-calling schema
 *   - Its own prompt fragment (injected into the system prompt)
 *   - Validation, normalisation, and fallback
 *
 * See: docs/02-the-agent-loop/05-skills-composing-capabilities.md
 */

import OpenAI from "openai";
import { Skill, SkillRegistry, SkillResult } from "./skill_base.js";

const MAX_ITERATIONS = 10;

const BASE_PROMPT = `You are a helpful assistant with access to skills.
Answer questions using the skills provided. Always use a skill when the
question falls within its scope — never guess data that a skill can provide.`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SkillAgentResult {
  answer: string;
  toolCalls: Array<{ name: string; args: Record<string, unknown> }>;
  skillResults: Array<{ skill: string; result: SkillResult }>;
  iterations: number;
}

// ---------------------------------------------------------------------------
// SkilledAgent
// ---------------------------------------------------------------------------

export class SkilledAgent {
  private loadedSkills: Skill[] = [];
  private client: OpenAI;

  constructor(
    private registry: SkillRegistry,
    private model = "gpt-4o",
    client?: OpenAI
  ) {
    this.client = client ?? new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }

  /** Load skills by name, resolving dependencies automatically. */
  loadSkills(skillNames: string[]): void {
    const loadedSet = new Set(this.loadedSkills.map((s) => s.name));
    for (const name of skillNames) {
      const skill = this.registry.get(name);
      if (!skill) continue;
      // Resolve dependencies first
      for (const dep of this.registry.resolveDependencies(skill)) {
        if (!loadedSet.has(dep.name)) {
          this.loadedSkills.push(dep);
          loadedSet.add(dep.name);
        }
      }
    }
  }

  /** Build system prompt with loaded skill fragments. */
  buildSystemPrompt(): string {
    if (!this.loadedSkills.length) return BASE_PROMPT;
    const fragments = this.loadedSkills.map(
      (s) => `## Skill: ${s.name}\n${s.getPromptFragment()}`
    );
    return BASE_PROMPT + "\n\n" + fragments.join("\n\n");
  }

  /** Run the agent loop with skill-as-tools. */
  async run(userInput: string): Promise<SkillAgentResult> {
    if (!userInput.trim()) throw new Error("userInput must not be empty");

    const tools = this.loadedSkills.map((s) => s.getOpenAISchema());
    const skillMap = new Map(this.loadedSkills.map((s) => [s.name, s]));

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: this.buildSystemPrompt() },
      { role: "user", content: userInput },
    ];

    const allToolCalls: SkillAgentResult["toolCalls"] = [];
    const allSkillResults: SkillAgentResult["skillResults"] = [];

    for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages,
        tools: tools.length ? tools : undefined,
        tool_choice: tools.length ? "auto" : undefined,
      });

      const choice = response.choices[0];
      const msg = choice.message;
      messages.push(msg as OpenAI.ChatCompletionMessageParam);

      if (choice.finish_reason === "stop" || !msg.tool_calls?.length) {
        return {
          answer: msg.content ?? "",
          toolCalls: allToolCalls,
          skillResults: allSkillResults,
          iterations: iteration + 1,
        };
      }

      // Execute tool calls
      for (const tc of msg.tool_calls) {
        const skillName = tc.function.name;
        let args: Record<string, unknown> = {};
        try { args = JSON.parse(tc.function.arguments); } catch { /* ignore */ }

        allToolCalls.push({ name: skillName, args });

        const skill = skillMap.get(skillName);
        let resultStr: string;
        if (skill) {
          const result = await skill.execute(args);
          allSkillResults.push({ skill: skillName, result });
          resultStr = result.success
            ? JSON.stringify(result.data)
            : `Error: ${result.error}`;
        } else {
          resultStr = `Error: Unknown skill "${skillName}"`;
        }

        messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: resultStr,
        });
      }
    }

    return {
      answer: "Maximum iterations reached without a final answer.",
      toolCalls: allToolCalls,
      skillResults: allSkillResults,
      iterations: MAX_ITERATIONS,
    };
  }
}
