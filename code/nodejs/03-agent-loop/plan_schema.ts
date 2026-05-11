/**
 * Zod schemas and validation for agent plans.
 *
 * Defines PlanStep, AgentPlan, and StepResult — the data contracts used
 * by PlanAndExecuteAgent.
 * See: docs/02-the-agent-loop/03-planning-strategies.md
 */

import { z } from "zod";

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

export const PlanStepSchema = z.object({
  stepNumber: z.number().int().min(1),
  description: z.string().min(10).max(500),
  toolName: z.string().optional(),
  toolParams: z.record(z.unknown()).optional(),
  dependsOn: z.array(z.number().int()).default([]),
  expectedOutput: z.string().min(5),
});

export const AgentPlanSchema = z
  .object({
    userQuestion: z.string(),
    steps: z.array(PlanStepSchema).min(1),
    estimatedToolCalls: z.number().int().min(0).default(0),
  })
  .refine(
    (plan) => {
      const nums = plan.steps.map((s) => s.stepNumber).sort((a, b) => a - b);
      return nums.every((n, i) => n === i + 1);
    },
    { message: "Step numbers must be sequential starting at 1" }
  )
  .refine(
    (plan) => {
      const valid = new Set(plan.steps.map((s) => s.stepNumber));
      return plan.steps.every((s) => s.dependsOn.every((d) => valid.has(d)));
    },
    { message: "All dependsOn references must point to existing steps" }
  );

export const StepResultSchema = z.object({
  stepNumber: z.number().int().min(1),
  success: z.boolean(),
  output: z.unknown().optional(),
  error: z.string().optional(),
  toolName: z.string().optional(),
  durationMs: z.number().optional(),
});

export type PlanStep = z.infer<typeof PlanStepSchema>;
export type AgentPlan = z.infer<typeof AgentPlanSchema>;
export type StepResult = z.infer<typeof StepResultSchema>;

// ---------------------------------------------------------------------------
// Quality scorer
// ---------------------------------------------------------------------------

/** Score a plan from 0–100 based on structural quality. */
export function scorePlan(plan: AgentPlan): number {
  let score = 100;
  if (plan.steps.length === 0) return 0;
  for (const step of plan.steps) {
    if (!step.toolName && step.toolParams) score -= 5; // params without tool
    if (step.description.length < 20) score -= 5;
    if (!step.expectedOutput) score -= 5;
  }
  // Penalise very long plans (likely over-engineered)
  if (plan.steps.length > 10) score -= (plan.steps.length - 10) * 3;
  return Math.max(0, Math.min(100, score));
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const rawPlan = {
    userQuestion: "What is the weather in Tokyo and what should I wear?",
    steps: [
      {
        stepNumber: 1,
        description: "Fetch the current weather conditions for Tokyo, Japan",
        toolName: "get_weather",
        toolParams: { city: "Tokyo, JP" },
        dependsOn: [],
        expectedOutput: "Temperature and conditions for Tokyo",
      },
      {
        stepNumber: 2,
        description:
          "Based on the weather data, recommend appropriate clothing for the conditions",
        dependsOn: [1],
        expectedOutput: "Clothing recommendation string",
      },
    ],
    estimatedToolCalls: 1,
  };

  const result = AgentPlanSchema.safeParse(rawPlan);
  if (!result.success) {
    console.error("Plan validation failed:", result.error.issues);
    return;
  }

  const plan = result.data;
  const score = scorePlan(plan);
  console.log("Plan validated successfully.");
  console.log(`Steps: ${plan.steps.length}, Score: ${score}/100`);
  plan.steps.forEach((s) => {
    console.log(`  [${s.stepNumber}] ${s.description.slice(0, 60)}...`);
  });
}

main();
