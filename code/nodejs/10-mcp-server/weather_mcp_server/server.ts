/**
 * MCP Weather Server — TypeScript
 *
 * Exposes two tools and one resource over the stdio transport:
 *   Tools:     get_weather(city, units)    — current conditions
 *              get_forecast(city, days)    — multi-day forecast
 *   Resources: weather://status            — health and uptime
 *
 * Usage:
 *   npx tsx weather_mcp_server/server.ts
 *   # or after build:
 *   node dist/weather_mcp_server/server.js
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// ---------------------------------------------------------------------------
// Mock weather data
// ---------------------------------------------------------------------------

interface WeatherRecord {
  temp_c: number;
  humidity: number;
  condition: string;
  wind_kph: number;
  country: string;
}

const WEATHER_DB: Record<string, WeatherRecord> = {
  tokyo:     { temp_c: 22, humidity: 68, condition: "partly cloudy",  wind_kph: 14, country: "JP" },
  london:    { temp_c: 12, humidity: 80, condition: "overcast",        wind_kph: 20, country: "UK" },
  "new york":{ temp_c: 18, humidity: 60, condition: "sunny",           wind_kph: 12, country: "US" },
  paris:     { temp_c: 16, humidity: 72, condition: "light rain",      wind_kph:  8, country: "FR" },
  sydney:    { temp_c: 20, humidity: 65, condition: "clear",           wind_kph: 18, country: "AU" },
  berlin:    { temp_c: 10, humidity: 75, condition: "cloudy",          wind_kph: 22, country: "DE" },
  dubai:     { temp_c: 38, humidity: 45, condition: "sunny",           wind_kph: 16, country: "AE" },
  moscow:    { temp_c:  5, humidity: 70, condition: "snow",            wind_kph: 10, country: "RU" },
  singapore: { temp_c: 30, humidity: 85, condition: "thunderstorm",    wind_kph: 24, country: "SG" },
  toronto:   { temp_c:  8, humidity: 62, condition: "clear",           wind_kph: 15, country: "CA" },
};

const FORECAST_CONDITIONS = [
  "sunny", "partly cloudy", "cloudy", "light rain",
  "rain", "thunderstorm", "clear", "overcast",
];

function normalizeCity(city: string): string {
  return city.split(",")[0].trim().toLowerCase();
}

function cToF(c: number): number {
  return Math.round((c * 9 / 5 + 32) * 10) / 10;
}

/** Simple seeded pseudo-random (deterministic per city). */
function seededRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function getCurrentWeather(city: string, units: "celsius" | "fahrenheit") {
  const key = normalizeCity(city);
  const data = WEATHER_DB[key];
  if (!data) return null;
  const temp = units === "celsius" ? data.temp_c : cToF(data.temp_c);
  return {
    city,
    country: data.country,
    temperature: temp,
    units,
    humidity: data.humidity,
    condition: data.condition,
    wind_kph: data.wind_kph,
    timestamp: new Date().toISOString(),
    source: "mock_data",
  };
}

function getForecast(city: string, days: number) {
  const key = normalizeCity(city);
  const data = WEATHER_DB[key];
  if (!data) return null;

  const clampedDays = Math.max(1, Math.min(days, 10));
  // Simple numeric hash of city name for deterministic output
  const seed = [...key].reduce((acc, c) => acc + c.charCodeAt(0), 0);
  const rand = seededRng(seed);

  const forecast = Array.from({ length: clampedDays }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() + i + 1);
    return {
      date: date.toISOString().slice(0, 10),
      day: date.toLocaleDateString("en-US", { weekday: "short" }),
      high_c: +(data.temp_c + rand() * 4 + 1).toFixed(1),
      low_c:  +(data.temp_c - rand() * 4 - 1).toFixed(1),
      condition: FORECAST_CONDITIONS[Math.floor(rand() * FORECAST_CONDITIONS.length)],
      precipitation_chance_pct: Math.floor(rand() * 100),
    };
  });

  return { city, days: clampedDays, forecast, source: "mock_data" };
}

function listSupportedCities(): string[] {
  return Object.keys(WEATHER_DB).map(k =>
    k.split(" ").map(w => w[0].toUpperCase() + w.slice(1)).join(" ")
  ).sort();
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

const SERVER_START = Date.now();
const server = new Server(
  { name: "weather-server", version: "1.0.0" },
  { capabilities: { tools: {}, resources: {} } },
);

// Tool list
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "get_weather",
      description:
        "Get current weather conditions for a city. Returns temperature, humidity, " +
        "wind speed, and conditions. Examples: 'Tokyo, JP', 'London, UK'.",
      inputSchema: {
        type: "object",
        properties: {
          city: {
            type: "string",
            description:
              "City name with optional ISO country code. Examples: 'Tokyo, JP', 'Sydney, AU'.",
          },
          units: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
            description: "Temperature unit. Defaults to 'celsius'.",
            default: "celsius",
          },
        },
        required: ["city"],
        additionalProperties: false,
      },
    },
    {
      name: "get_forecast",
      description:
        "Get a multi-day weather forecast for a city (1–10 days). " +
        "Returns daily high/low, conditions, precipitation chance.",
      inputSchema: {
        type: "object",
        properties: {
          city: {
            type: "string",
            description: "City name with optional ISO country code.",
          },
          days: {
            type: "integer",
            minimum: 1,
            maximum: 10,
            description: "Number of forecast days (1–10). Defaults to 5.",
            default: 5,
          },
        },
        required: ["city"],
        additionalProperties: false,
      },
    },
  ],
}));

// Tool execution
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;
  process.stderr.write(`[weather-server] Tool call: ${name}\n`);

  if (name === "get_weather") {
    const city = (args["city"] as string | undefined);
    if (!city) {
      return { content: [{ type: "text", text: JSON.stringify({ error: "Missing required argument: city" }) }] };
    }
    const units = ((args["units"] as string) || "celsius") as "celsius" | "fahrenheit";
    const data = getCurrentWeather(city, units);
    if (!data) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: `City not found: '${city}'.`,
            supported_cities: listSupportedCities(),
          }),
        }],
      };
    }
    return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
  }

  if (name === "get_forecast") {
    const city = (args["city"] as string | undefined);
    if (!city) {
      return { content: [{ type: "text", text: JSON.stringify({ error: "Missing required argument: city" }) }] };
    }
    const days = Number(args["days"] ?? 5);
    const data = getForecast(city, isNaN(days) ? 5 : days);
    if (!data) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: `City not found: '${city}'.`,
            supported_cities: listSupportedCities(),
          }),
        }],
      };
    }
    return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
  }

  return {
    content: [{
      type: "text",
      text: JSON.stringify({ error: `Unknown tool: '${name}'`, available: ["get_weather", "get_forecast"] }),
    }],
  };
});

// Resource list
server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: [{
    uri: "weather://status",
    name: "Server Status",
    description: "Health, uptime, and capability information for this weather MCP server.",
    mimeType: "application/json",
  }],
}));

// Resource read
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;
  if (uri === "weather://status") {
    const status = {
      status: "healthy",
      server: "weather-server",
      version: "1.0.0",
      uptime_seconds: Math.floor((Date.now() - SERVER_START) / 1000),
      tools: ["get_weather", "get_forecast"],
      supported_cities: listSupportedCities(),
      timestamp: new Date().toISOString(),
      transport: "stdio",
    };
    return {
      contents: [{
        uri,
        mimeType: "application/json",
        text: JSON.stringify(status, null, 2),
      }],
    };
  }
  throw new Error(`Unknown resource URI: ${uri}`);
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  process.stderr.write("[weather-server] Starting on stdio transport ...\n");
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  process.stderr.write(`[weather-server] Fatal: ${err}\n`);
  process.exit(1);
});
