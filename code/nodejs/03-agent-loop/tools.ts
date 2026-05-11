/**
 * Concrete tool implementations: weather, calculator, web search, and date.
 * See: docs/02-the-agent-loop/01-anatomy-of-an-agent.md
 */

import { ToolRegistry, NotFoundError, InvalidArgsError } from "./tool_registry.js";

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

const KNOWN_CITIES: Record<string, { temperature: number; condition: string; humidity: number }> = {
  "tokyo, jp": { temperature: 22, condition: "partly cloudy", humidity: 65 },
  "london, uk": { temperature: 14, condition: "overcast", humidity: 80 },
  "new york, us": { temperature: 18, condition: "sunny", humidity: 55 },
  "paris, fr": { temperature: 16, condition: "light rain", humidity: 72 },
  "sydney, au": { temperature: 25, condition: "clear", humidity: 50 },
};

export function getWeather(args: Record<string, unknown>): unknown {
  const city = (args.city as string | undefined)?.toLowerCase().trim() ?? "";
  const weather = KNOWN_CITIES[city];
  if (!weather) {
    const suggestion = Object.keys(KNOWN_CITIES)[0];
    throw new NotFoundError(`City '${city}' not found`, `Try '${suggestion}'`);
  }
  return { city, ...weather };
}

export function calculate(args: Record<string, unknown>): unknown {
  const expr = args.expression as string | undefined;
  if (!expr) throw new InvalidArgsError("expression is required");
  // Safe evaluation: only allow numeric expressions
  const safe = /^[\d\s+\-*/().%]+$/.test(expr);
  if (!safe) throw new InvalidArgsError("Only numeric expressions are allowed");
  try {
    // eslint-disable-next-line no-new-func
    const result = Function(`"use strict"; return (${expr})`)() as number;
    return { expression: expr, result };
  } catch (e) {
    throw new InvalidArgsError(`Could not evaluate: ${String(e)}`);
  }
}

export function webSearch(args: Record<string, unknown>): unknown {
  const query = args.query as string | undefined;
  if (!query) throw new InvalidArgsError("query is required");
  // Mock search results
  return {
    query,
    results: [
      { title: `Result for "${query}"`, url: "https://example.com/1", snippet: `Information about ${query}.` },
      { title: `More about "${query}"`, url: "https://example.com/2", snippet: `Additional details on ${query}.` },
    ],
  };
}

export function getCurrentDate(_args: Record<string, unknown>): unknown {
  const now = new Date();
  return {
    date: now.toISOString().split("T")[0],
    time: now.toTimeString().split(" ")[0],
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    timestamp: now.getTime(),
  };
}

// ---------------------------------------------------------------------------
// Registry builder
// ---------------------------------------------------------------------------

/** Create a ToolRegistry pre-loaded with all demo tools. */
export function createDefaultRegistry(): ToolRegistry {
  const registry = new ToolRegistry();

  registry.register({
    name: "get_weather",
    description: "Get current weather for a city. Returns temperature and conditions.",
    parameters: {
      city: { type: "string", description: "City name with country code, e.g. 'Tokyo, JP'", required: true },
    },
    handler: getWeather,
  });

  registry.register({
    name: "calculate",
    description: "Evaluate a numeric arithmetic expression.",
    parameters: {
      expression: { type: "string", description: "Arithmetic expression, e.g. '2 + 3 * 4'", required: true },
    },
    handler: calculate,
  });

  registry.register({
    name: "web_search",
    description: "Search the web for information about a topic.",
    parameters: {
      query: { type: "string", description: "Search query string", required: true },
    },
    handler: webSearch,
  });

  registry.register({
    name: "get_current_date",
    description: "Get the current date and time.",
    parameters: {},
    handler: getCurrentDate,
  });

  return registry;
}
