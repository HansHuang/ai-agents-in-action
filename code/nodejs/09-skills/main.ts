/**
 * Demo entry point for nodejs/09-skills.
 *
 * Shows the SkillRegistry and SkilledAgent in action with example skills.
 * See: docs/02-the-agent-loop/05-skills-composing-capabilities.md
 */

import { Skill, SkillRegistry } from "./skill_base.js";
import { SkilledAgent } from "./skilled_agent.js";

// ---------------------------------------------------------------------------
// Sample skills (no LLM required for skill execution)
// ---------------------------------------------------------------------------

const weatherSkill = new Skill({
  name: "get_weather",
  description: "Get current weather for a city",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string", description: "City name" },
    },
    required: ["city"],
  },
  tool: async (params) => {
    // Mock implementation
    const city = String(params.city ?? "Unknown");
    return { city, temperature: 22, condition: "sunny", humidity: 45 };
  },
  promptFragment: "Use get_weather when the user asks about weather conditions.",
  testCases: [
    { input: { city: "London" }, expectSuccess: true, expectFallback: false, expectOutputContains: ["temperature"] },
  ],
});

const calculatorSkill = new Skill({
  name: "calculate",
  description: "Perform arithmetic calculations",
  parameters: {
    type: "object",
    properties: {
      expression: { type: "string", description: "Math expression to evaluate (e.g. '2 + 2')" },
    },
    required: ["expression"],
  },
  tool: async (params) => {
    const expr = String(params.expression ?? "");
    // Safe arithmetic-only evaluation
    const sanitized = expr.replace(/[^0-9+\-*/().\s]/g, "");
    try {
      // eslint-disable-next-line no-eval
      const result = Function(`"use strict"; return (${sanitized})`)() as number;
      return { expression: expr, result, success: true };
    } catch {
      return { expression: expr, result: null, success: false, error: "Invalid expression" };
    }
  },
  promptFragment: "Use calculate for any arithmetic: addition, subtraction, multiplication, division.",
  testCases: [
    { input: { expression: "2 + 2" }, expectSuccess: true, expectFallback: false, expectOutputContains: ["4"] },
  ],
});

const timeSkill = new Skill({
  name: "get_current_time",
  description: "Get the current date and time",
  parameters: { type: "object", properties: {}, required: [] },
  tool: async () => ({
    iso: new Date().toISOString(),
    unix: Math.floor(Date.now() / 1000),
    readable: new Date().toLocaleString(),
  }),
  promptFragment: "Use get_current_time when the user asks what time or date it is.",
});

// ---------------------------------------------------------------------------
// Demo runner
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("=== SkilledAgent Demo ===\n");

  // Build registry
  const registry = new SkillRegistry();
  registry.register(weatherSkill);
  registry.register(calculatorSkill);
  registry.register(timeSkill);

  const allSkillNames = ["get_weather", "calculate", "get_current_time"];
  console.log(`Registered skills: ${allSkillNames.join(", ")}`);

  // Run skill tests (no LLM needed)
  console.log("\n--- Skill Tests ---");
  for (const skillName of allSkillNames) {
    const skill = registry.get(skillName)!;
    const tests = await skill.runTests();
    const passed = tests.filter((t) => t.passed).length;
    console.log(`  ${skillName}: ${passed}/${tests.length} tests passed`);
  }

  // Demo direct skill execution (no LLM needed)
  console.log("\n--- Direct Skill Execution ---");
  const weather = await weatherSkill.execute({ city: "Berlin" });
  console.log(`  Weather in Berlin: ${JSON.stringify(weather.data)}`);

  const calc = await calculatorSkill.execute({ expression: "15 * 7 + 3" });
  console.log(`  15 * 7 + 3 = ${JSON.stringify(calc.data)}`);

  const time = await timeSkill.execute({});
  console.log(`  Current time: ${(time.data as { readable: string }).readable}`);

  // SkilledAgent requires OPENAI_API_KEY — skip if not set
  if (!process.env.OPENAI_API_KEY) {
    console.log("\n[Skip] SkilledAgent demo requires OPENAI_API_KEY");
    return;
  }

  console.log("\n--- SkilledAgent Demo ---");
  const agent = new SkilledAgent(registry, "gpt-4o-mini");
  agent.loadSkills(["get_weather", "calculate", "get_current_time"]);
  console.log("Loaded skills:", allSkillNames.join(", "));

  const result = await agent.run("What is the weather like in Tokyo, and what is 42 * 7?");
  console.log(`Answer: ${result.answer}`);
  console.log(`Skills called: ${result.toolCalls.map((tc) => tc.name).join(", ")}`);
}

main().catch(console.error);
