/**
 * Complete configuration system for the ProductionHarness.
 *
 * Provides HarnessConfig with every tuneable parameter, loading from
 * environment variables or a plain object, validation, and named presets.
 * See: docs/07-harness-engineering/01-the-harness-mindset.md
 */

// ---------------------------------------------------------------------------
// HarnessConfig
// ---------------------------------------------------------------------------

export interface HarnessConfig {
  // Identity
  agentId: string;
  systemPrompt: string;

  // Input guardrails
  maxInputLength: number;
  minInputLength: number;
  rateLimitRpm: number;
  rateLimitRph: number;
  rateLimitRpd: number;
  checkInputPii: boolean;
  checkInputContent: boolean;
  checkInputInjection: boolean;
  inputInjectionThreshold: "low" | "medium" | "high";

  // Routing / model config
  chatModel: string;
  chatMaxTokens: number;
  chatTemperature: number;
  chatTimeoutMs: number;
  ragModel: string;
  ragMaxTokens: number;
  ragTemperature: number;
  ragTimeoutMs: number;
  agentModel: string;
  agentMaxTokens: number;
  agentTemperature: number;
  agentTimeoutMs: number;
  agentMaxIterations: number;

  // Resilience
  circuitBreakerThreshold: number;
  circuitBreakerRecoveryMs: number;
  circuitBreakerWindowMs: number;
  llmMaxRetries: number;
  llmBaseDelayMs: number;
  llmMaxDelayMs: number;
  llmTotalDeadlineMs: number;
  toolMaxRetries: number;
  toolBaseDelayMs: number;
  toolTimeoutMs: number;

  // Output guardrails
  validateOutputSchema: boolean;
  checkOutputPii: boolean;
  checkOutputSafety: boolean;
  checkOutputLeakage: boolean;
  checkOutputHallucination: boolean;
  blockOnHallucination: boolean;
  hallucinationConfidenceThreshold: number;
  outputMaxLength: number;

  // Human-in-the-loop
  approvalChannels: string[];
  approvalHighValueThreshold: number;
  approvalExternalCommunication: boolean;
  approvalDatabaseModification: boolean;
  approvalDefaultTimeoutMs: number;
  approvalCriticalTimeoutMs: number;

  // Observability
  logLevel: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  traceExportEnabled: boolean;
  metricsExportIntervalMs: number;
  alertOnCircuitOpen: boolean;
  alertOnFallbackExhaustion: boolean;
  alertOnCostSpike: boolean;
  costSpikeThresholdMultiplier: number;
}

const DEFAULTS: HarnessConfig = {
  agentId: "production-agent-v1",
  systemPrompt: "",
  maxInputLength: 100_000,
  minInputLength: 2,
  rateLimitRpm: 30,
  rateLimitRph: 500,
  rateLimitRpd: 5_000,
  checkInputPii: true,
  checkInputContent: true,
  checkInputInjection: true,
  inputInjectionThreshold: "medium",
  chatModel: "gpt-4o-mini",
  chatMaxTokens: 512,
  chatTemperature: 0.7,
  chatTimeoutMs: 15_000,
  ragModel: "gpt-4o",
  ragMaxTokens: 2_048,
  ragTemperature: 0.3,
  ragTimeoutMs: 45_000,
  agentModel: "gpt-4o",
  agentMaxTokens: 4_096,
  agentTemperature: 0.2,
  agentTimeoutMs: 120_000,
  agentMaxIterations: 10,
  circuitBreakerThreshold: 5,
  circuitBreakerRecoveryMs: 120_000,
  circuitBreakerWindowMs: 60_000,
  llmMaxRetries: 3,
  llmBaseDelayMs: 1_000,
  llmMaxDelayMs: 60_000,
  llmTotalDeadlineMs: 300_000,
  toolMaxRetries: 2,
  toolBaseDelayMs: 500,
  toolTimeoutMs: 30_000,
  validateOutputSchema: true,
  checkOutputPii: true,
  checkOutputSafety: true,
  checkOutputLeakage: true,
  checkOutputHallucination: true,
  blockOnHallucination: false,
  hallucinationConfidenceThreshold: 0.7,
  outputMaxLength: 50_000,
  approvalChannels: ["dashboard"],
  approvalHighValueThreshold: 500.0,
  approvalExternalCommunication: true,
  approvalDatabaseModification: true,
  approvalDefaultTimeoutMs: 300_000,
  approvalCriticalTimeoutMs: 600_000,
  logLevel: "INFO",
  traceExportEnabled: true,
  metricsExportIntervalMs: 60_000,
  alertOnCircuitOpen: true,
  alertOnFallbackExhaustion: true,
  alertOnCostSpike: true,
  costSpikeThresholdMultiplier: 2.0,
};

/**
 * Load HarnessConfig from environment variables, falling back to defaults.
 */
export function fromEnv(overrides: Partial<HarnessConfig> = {}): HarnessConfig {
  const e = process.env;
  const cfg: HarnessConfig = {
    ...DEFAULTS,
    agentId: e.AGENT_ID ?? DEFAULTS.agentId,
    systemPrompt: e.SYSTEM_PROMPT ?? DEFAULTS.systemPrompt,
    maxInputLength: Number(e.MAX_INPUT_LENGTH ?? DEFAULTS.maxInputLength),
    minInputLength: Number(e.MIN_INPUT_LENGTH ?? DEFAULTS.minInputLength),
    rateLimitRpm: Number(e.RATE_LIMIT_RPM ?? DEFAULTS.rateLimitRpm),
    chatModel: e.CHAT_MODEL ?? DEFAULTS.chatModel,
    agentModel: e.AGENT_MODEL ?? DEFAULTS.agentModel,
    ragModel: e.RAG_MODEL ?? DEFAULTS.ragModel,
    logLevel: (e.LOG_LEVEL as HarnessConfig["logLevel"]) ?? DEFAULTS.logLevel,
    ...overrides,
  };
  return cfg;
}

/** Named presets. */
export const PRESETS: Record<string, () => HarnessConfig> = {
  development: () => fromEnv({
    chatModel: "gpt-4o-mini",
    agentModel: "gpt-4o-mini",
    llmMaxRetries: 1,
    rateLimitRpm: 60,
    logLevel: "DEBUG",
    traceExportEnabled: false,
    blockOnHallucination: false,
  }),
  production: () => fromEnv({
    logLevel: "INFO",
    traceExportEnabled: true,
    blockOnHallucination: true,
    alertOnCircuitOpen: true,
  }),
  costOptimized: () => fromEnv({
    chatModel: "gpt-4o-mini",
    agentModel: "gpt-4o-mini",
    ragModel: "gpt-4o-mini",
    chatMaxTokens: 256,
    agentMaxTokens: 1_024,
    llmMaxRetries: 2,
  }),
  highSecurity: () => fromEnv({
    checkInputPii: true,
    checkInputContent: true,
    checkInputInjection: true,
    inputInjectionThreshold: "low",
    checkOutputPii: true,
    checkOutputSafety: true,
    checkOutputLeakage: true,
    blockOnHallucination: true,
    approvalExternalCommunication: true,
    approvalDatabaseModification: true,
  }),
};

/** Validate a config and return warnings. */
export function validateConfig(cfg: HarnessConfig): string[] {
  const warnings: string[] = [];
  if (cfg.rateLimitRpm > 100) warnings.push("rateLimitRpm > 100 is unusually high");
  if (cfg.agentMaxIterations > 20) warnings.push("agentMaxIterations > 20 may cause runaway agents");
  if (!cfg.checkInputInjection) warnings.push("Injection checking is disabled");
  if (!cfg.checkOutputPii) warnings.push("Output PII checking is disabled");
  if (cfg.llmMaxRetries > 5) warnings.push("llmMaxRetries > 5 may cause long waits");
  return warnings;
}

/** Diff two configs, returning only changed keys. */
export function diffConfigs(a: HarnessConfig, b: HarnessConfig): Record<string, { from: unknown; to: unknown }> {
  const diff: Record<string, { from: unknown; to: unknown }> = {};
  for (const key of Object.keys(a) as Array<keyof HarnessConfig>) {
    if (JSON.stringify(a[key]) !== JSON.stringify(b[key])) {
      diff[key] = { from: a[key], to: b[key] };
    }
  }
  return diff;
}
