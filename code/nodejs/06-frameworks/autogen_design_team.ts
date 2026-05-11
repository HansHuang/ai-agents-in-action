/**
 * AutoGen-style product design team: conversational multi-agent design.
 *
 * Implements a five-agent group chat pattern (no AutoGen dependency):
 *   ProductManager, Designer, Engineer, Critic, UserProxy → DesignSpec
 *
 * Solutions emerge from conversation rather than a predefined task list.
 * See: docs/06-frameworks-in-practice/03-crewai-autogen.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o-mini";
const MAX_ROUNDS = 4;

export interface AgentPersona {
  name: string;
  role: string;
  systemPrompt: string;
}

export interface GroupChatMessage {
  agent: string;
  content: string;
}

export interface DesignSpec {
  productName: string;
  coreFeatures: string[];
  technicalApproach: string;
  risks: string[];
  nextSteps: string[];
  rawTranscript: GroupChatMessage[];
}

const PERSONAS: AgentPersona[] = [
  {
    name: "ProductManager",
    role: "Product Manager",
    systemPrompt:
      "You are a Product Manager focused on user needs, market fit, and business value. " +
      "Ask clarifying questions, define success metrics, and ensure the product solves real problems. " +
      "Keep responses to 2-3 sentences.",
  },
  {
    name: "Designer",
    role: "UX Designer",
    systemPrompt:
      "You are a UX Designer focused on user experience, interface simplicity, and accessibility. " +
      "Suggest UX patterns, flag usability concerns, and champion the user's perspective. " +
      "Keep responses to 2-3 sentences.",
  },
  {
    name: "Engineer",
    role: "Software Engineer",
    systemPrompt:
      "You are a Software Engineer focused on technical feasibility, architecture, and implementation cost. " +
      "Propose concrete technical approaches and flag complexity risks. " +
      "Keep responses to 2-3 sentences.",
  },
  {
    name: "Critic",
    role: "Devil's Advocate",
    systemPrompt:
      "You are a Devil's Advocate who challenges assumptions and identifies blind spots. " +
      "Ask hard questions, raise edge cases, and push the team to think deeper. " +
      "Keep responses to 2-3 sentences.",
  },
];

/**
 * Run a group chat design session and return a structured DesignSpec.
 */
export async function runDesignTeam(
  productIdea: string,
  client: OpenAI,
  maxRounds = MAX_ROUNDS
): Promise<DesignSpec> {
  const transcript: GroupChatMessage[] = [];
  const history: OpenAI.Chat.ChatCompletionMessageParam[] = [];

  // Seed the conversation
  history.push({ role: "user", content: `Product idea: ${productIdea}` });
  transcript.push({ agent: "UserProxy", content: productIdea });

  for (let round = 0; round < maxRounds; round++) {
    for (const persona of PERSONAS) {
      const resp = await client.chat.completions.create({
        model: MODEL,
        messages: [
          { role: "system", content: persona.systemPrompt },
          ...history,
          {
            role: "user",
            content: `As ${persona.name}, respond to the discussion so far in 2-3 sentences.`,
          },
        ],
        temperature: 0.7,
        max_tokens: 150,
      });

      const content = resp.choices[0].message.content?.trim() ?? "";
      transcript.push({ agent: persona.name, content });
      history.push({ role: "assistant", content: `[${persona.name}]: ${content}` });
    }
  }

  // Synthesize into DesignSpec
  const synthesisResp = await client.chat.completions.create({
    model: MODEL,
    messages: [
      {
        role: "system",
        content:
          'Synthesize the design discussion into a structured spec. ' +
          'Return JSON: {"productName": "...", "coreFeatures": [...], "technicalApproach": "...", "risks": [...], "nextSteps": [...]}',
      },
      {
        role: "user",
        content: transcript.map((m) => `${m.agent}: ${m.content}`).join("\n"),
      },
    ],
    response_format: { type: "json_object" },
    temperature: 0,
  });

  let spec: Omit<DesignSpec, "rawTranscript"> = {
    productName: productIdea.slice(0, 40),
    coreFeatures: [],
    technicalApproach: "",
    risks: [],
    nextSteps: [],
  };
  try {
    spec = JSON.parse(synthesisResp.choices[0].message.content ?? "{}") as typeof spec;
  } catch { /* keep defaults */ }

  return { ...spec, rawTranscript: transcript };
}

/** Print the design session results. */
export function printDesignSpec(spec: DesignSpec): void {
  console.log(`\n=== Design Spec: ${spec.productName} ===`);
  console.log(`\nTranscript (${spec.rawTranscript.length} messages):`);
  spec.rawTranscript.slice(0, 6).forEach((m) => {
    console.log(`  [${m.agent}] ${m.content.slice(0, 80)}...`);
  });
  console.log(`\nCore Features: ${spec.coreFeatures.join(", ")}`);
  console.log(`Technical Approach: ${spec.technicalApproach}`);
  console.log(`Risks: ${spec.risks.join("; ")}`);
  console.log(`Next Steps: ${spec.nextSteps.join("; ")}`);
}
