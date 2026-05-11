/**
 * CrewAI-style research crew: structured multi-agent research pipeline.
 *
 * Implements a four-agent research pipeline (no CrewAI dependency):
 *   Researcher → Analyst → Writer → Reviewer
 *
 * Each agent has a role, goal, and backstory. Outputs chain automatically.
 * See: docs/06-frameworks-in-practice/03-crewai-autogen.md
 */

import OpenAI from "openai";

const MODEL = "gpt-4o-mini";

export interface CrewAgent {
  name: string;
  role: string;
  goal: string;
  backstory: string;
}

export interface TaskResult {
  agent: string;
  task: string;
  output: string;
  tokenCount: number;
}

export interface ResearchReport {
  topic: string;
  stages: TaskResult[];
  finalReport: string;
  totalTokens: number;
}

const RESEARCH_CREW: CrewAgent[] = [
  {
    name: "Researcher",
    role: "Senior Research Analyst",
    goal: "Find comprehensive, accurate information on the topic",
    backstory: "You are an expert researcher with 10 years of experience synthesizing complex topics.",
  },
  {
    name: "Analyst",
    role: "Data Analyst",
    goal: "Identify patterns, insights, and implications from research",
    backstory: "You excel at turning raw information into structured insights and identifying trends.",
  },
  {
    name: "Writer",
    role: "Technical Writer",
    goal: "Produce clear, engaging prose from raw research and analysis",
    backstory: "You write clearly for technical audiences, balancing depth with accessibility.",
  },
  {
    name: "Reviewer",
    role: "Quality Reviewer",
    goal: "Ensure accuracy, completeness, and clarity of the final output",
    backstory: "You are a meticulous reviewer who catches errors and improves structure.",
  },
];

async function runAgent(
  agent: CrewAgent,
  task: string,
  previousOutput: string,
  client: OpenAI
): Promise<{ output: string; tokens: number }> {
  const context = previousOutput
    ? `\n\nPrevious stage output:\n${previousOutput}`
    : "";

  const resp = await client.chat.completions.create({
    model: MODEL,
    messages: [
      {
        role: "system",
        content: `You are ${agent.name}, a ${agent.role}.\nGoal: ${agent.goal}\nBackstory: ${agent.backstory}`,
      },
      { role: "user", content: `${task}${context}\n\nProvide a concise 3-4 paragraph response.` },
    ],
    temperature: 0.5,
    max_tokens: 400,
  });

  return {
    output: resp.choices[0].message.content?.trim() ?? "",
    tokens: resp.usage?.total_tokens ?? 0,
  };
}

/**
 * Run the four-agent research pipeline and return a structured report.
 */
export async function runResearchCrew(
  topic: string,
  client: OpenAI
): Promise<ResearchReport> {
  const tasks = [
    `Research the following topic comprehensively: "${topic}"`,
    `Analyze the research and identify key insights, patterns, and implications for: "${topic}"`,
    `Write a well-structured report based on the research and analysis about: "${topic}"`,
    `Review the report and improve clarity, accuracy, and completeness. Final topic: "${topic}"`,
  ];

  const stages: TaskResult[] = [];
  let lastOutput = "";
  let totalTokens = 0;

  for (let i = 0; i < RESEARCH_CREW.length; i++) {
    const agent = RESEARCH_CREW[i];
    const { output, tokens } = await runAgent(agent, tasks[i], lastOutput, client);
    stages.push({ agent: agent.name, task: tasks[i].slice(0, 60), output, tokenCount: tokens });
    lastOutput = output;
    totalTokens += tokens;
  }

  return {
    topic,
    stages,
    finalReport: lastOutput,
    totalTokens,
  };
}

/** Print the research report. */
export function printResearchReport(report: ResearchReport): void {
  console.log(`\n=== Research Report: ${report.topic} ===`);
  console.log(`Total tokens used: ${report.totalTokens}`);
  report.stages.forEach((s) => {
    console.log(`\n[${s.agent}] ${s.task}:`);
    console.log(s.output.slice(0, 200) + (s.output.length > 200 ? "..." : ""));
  });
}
