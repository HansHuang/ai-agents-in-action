/**
 * 12-Factor Agent Self-Assessment Tool (TypeScript)
 * ==================================================
 * Evaluates an agent configuration against all 12 production-readiness factors
 * and generates actionable reports with roadmaps and improvement tracking.
 *
 * Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md
 *
 * @packageDocumentation
 */

import { z } from "zod";

// ---------------------------------------------------------------------------
// Zod schemas for report validation
// ---------------------------------------------------------------------------

export const FactorAssessmentSchema = z.object({
  factorNumber: z.number().int().min(1).max(12),
  factorName: z.string(),
  score: z.number().int().min(1).max(5),
  status: z.enum(["critical", "needs_improvement", "good", "excellent"]),
  evidence: z.array(z.string()),
  gaps: z.array(z.string()),
  recommendations: z.array(z.string()),
});

export const TwelveFactorReportSchema = z.object({
  overallScore: z.number().int().min(12).max(60),
  maturityLevel: z.enum(["Prototype", "Development", "Staging", "Production", "Elite"]),
  maturityLevelNumber: z.number().int().min(1).max(5),
  factors: z.array(FactorAssessmentSchema),
  criticalGaps: z.array(z.string()),
  improvementPriorities: z.array(z.string()),
  assessedAt: z.string(),
});

export const ComparisonReportSchema = z.object({
  baselineScore: z.number().int(),
  currentScore: z.number().int(),
  scoreDelta: z.number().int(),
  baselineLevel: z.string(),
  currentLevel: z.string(),
  improvedFactors: z.array(z.tuple([z.number(), z.string(), z.number(), z.number()])),
  regressedFactors: z.array(z.tuple([z.number(), z.string(), z.number(), z.number()])),
  unchangedFactors: z.array(z.tuple([z.number(), z.string(), z.number()])),
});

// ---------------------------------------------------------------------------
// TypeScript types (inferred from schemas)
// ---------------------------------------------------------------------------

export type AgentConfig = Record<string, unknown>;
export type FactorAssessment = z.infer<typeof FactorAssessmentSchema>;
export type TwelveFactorReport = z.infer<typeof TwelveFactorReportSchema>;
export type ComparisonReport = z.infer<typeof ComparisonReportSchema>;

export interface TrendReport {
  assessments: Array<{ date: string; score: number; level: string }>;
  firstScore: number;
  latestScore: number;
  totalImprovement: number;
  averageImprovementPerAssessment: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FACTOR_NAMES: Record<number, string> = {
  1: "Prompt as Code",
  2: "Explicit State",
  3: "Provider Agnostic",
  4: "Token Budgeting",
  5: "Structured Everything",
  6: "Context Is a Resource",
  7: "Defense in Depth",
  8: "Graceful Degradation",
  9: "Observability First",
  10: "Human in the Loop",
  11: "Continuous Evaluation",
  12: "Dev-Prod Parity",
};

const ROMAN: Record<number, string> = {
  1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
  7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function scoreToStatus(score: number): FactorAssessment["status"] {
  if (score <= 2) return "critical";
  if (score === 3) return "needs_improvement";
  if (score === 4) return "good";
  return "excellent";
}

function checksToScore(passed: number): number {
  return Math.min(5, Math.max(1, passed));
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

// ---------------------------------------------------------------------------
// Factor assessors
// ---------------------------------------------------------------------------

function assessFactor1(cfg: AgentConfig): FactorAssessment {
  const checks = {
    inVcs: Boolean(cfg.prompts_in_version_control),
    semver: Boolean(cfg.prompts_semantically_versioned),
    reviewed: Boolean(cfg.prompts_code_reviewed),
    changeLog: Boolean(cfg.prompts_have_change_log),
    rollback: Boolean(cfg.prompts_independently_rollbackable),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.inVcs) evidence.push("Prompts stored in version control (Git)");
  else { gaps.push("Prompts not tracked in version control"); recs.push("Move prompts to a prompts/ directory under version control"); }

  if (checks.semver) evidence.push("Prompts use semantic versioning");
  else { gaps.push("Prompts lack semantic versioning"); recs.push("Add a version field to each prompt file (YAML frontmatter)"); }

  if (checks.reviewed) evidence.push("Prompt changes go through code review");
  else { gaps.push("Prompt changes bypass code review"); recs.push("Add prompts/ to required CODEOWNERS for PR review"); }

  if (checks.changeLog) evidence.push("Prompts include a change_log field");
  else { gaps.push("No prompt change log"); recs.push("Add change_log field to every prompt YAML"); }

  if (checks.rollback) evidence.push("Prompts can be rolled back independently");
  else { gaps.push("Prompt rollback requires full application rollback"); recs.push("Decouple prompt deployment from application deployment"); }

  return { factorNumber: 1, factorName: FACTOR_NAMES[1], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor2(cfg: AgentConfig): FactorAssessment {
  const checks = {
    stateClass: Boolean(cfg.has_conversation_state_class),
    survivesTruncation: Boolean(cfg.state_survives_truncation),
    serializable: Boolean(cfg.state_is_serializable),
    persisted: Boolean(cfg.state_persisted_across_sessions),
    transitionsLogged: Boolean(cfg.state_transitions_logged),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.stateClass) evidence.push("ConversationState class exists");
  else { gaps.push("No explicit ConversationState class found"); recs.push("Create a ConversationState interface with all session fields"); }

  if (checks.survivesTruncation) evidence.push("State injected into every LLM call, surviving message truncation");
  else { gaps.push("State may be lost when message list is truncated"); recs.push("Inject a state summary into every LLM call prompt"); }

  if (checks.serializable) evidence.push("State is JSON-serializable (toJSON/toDict exists)");
  else { gaps.push("State cannot be serialized for debugging or persistence"); recs.push("Add a toJSON() method to ConversationState"); }

  if (checks.persisted) evidence.push("State is persisted to a durable store");
  else { gaps.push("State is lost when session ends"); recs.push("Persist ConversationState to Redis or a database by session ID"); }

  if (checks.transitionsLogged) evidence.push("State transitions are logged");
  else { gaps.push("State changes are not logged"); recs.push("Log every state field change with timestamp and reason"); }

  return { factorNumber: 2, factorName: FACTOR_NAMES[2], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor3(cfg: AgentConfig): FactorAssessment {
  const numProviders = (cfg.num_providers_supported as number) ?? 0;
  const checks = {
    interface: Boolean(cfg.has_llm_provider_interface),
    twoProviders: numProviders >= 2,
    envConfig: Boolean(cfg.provider_configurable_via_env),
    abstracted: Boolean(cfg.provider_specific_features_abstracted),
    fallbackChain: Boolean(cfg.has_provider_fallback_chain),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.interface) evidence.push("LLM provider interface/abstract class exists");
  else { gaps.push("LLM calls tightly coupled to a specific provider"); recs.push("Create an LLMProvider abstract class with a chat() method"); }

  if (checks.twoProviders) evidence.push(`${numProviders} provider(s) supported`);
  else { gaps.push(`Only ${numProviders} provider(s) supported — need ≥2`); recs.push("Implement at least one fallback provider"); }

  if (checks.envConfig) evidence.push("Provider configured via environment variables");
  else { gaps.push("Provider is hardcoded in source code"); recs.push("Read LLM_PROVIDER and LLM_MODEL from environment variables"); }

  if (checks.abstracted) evidence.push("Provider-specific features abstracted behind interface");
  else { gaps.push("Provider-specific API details leak into application code"); recs.push("Wrap provider-specific features behind a common interface"); }

  if (checks.fallbackChain) evidence.push("Provider fallback chain configured");
  else { gaps.push("No fallback chain — single provider failure causes outage"); recs.push("Implement provider fallback chain with circuit breaker"); }

  return { factorNumber: 3, factorName: FACTOR_NAMES[3], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor4(cfg: AgentConfig): FactorAssessment {
  const checks = {
    tokenBudget: Boolean(cfg.has_per_request_token_budget),
    costBudget: Boolean(cfg.has_per_request_cost_budget),
    usageTracked: Boolean(cfg.token_usage_tracked_and_logged),
    costAlerts: Boolean(cfg.cost_alerts_configured),
    enforced: Boolean(cfg.budget_enforcement_blocks_excess),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.tokenBudget) evidence.push("Per-request token budget defined");
  else { gaps.push("No per-request token budget"); recs.push("Define a TokenBudget with maxTokens per request"); }

  if (checks.costBudget) evidence.push("Per-request cost budget defined");
  else { gaps.push("No per-request cost budget"); recs.push("Add maxCost (USD) to TokenBudget"); }

  if (checks.usageTracked) evidence.push("Token counts tracked and logged per request");
  else { gaps.push("Token usage not tracked"); recs.push("Log promptTokens, completionTokens, and totalCost per request"); }

  if (checks.costAlerts) evidence.push("Cost alerts configured for budget overruns");
  else { gaps.push("No cost alerts — cost spikes go undetected"); recs.push("Alert when request cost exceeds 2x baseline average"); }

  if (checks.enforced) evidence.push("Requests exceeding budget are blocked before LLM call");
  else { gaps.push("Budget tracked but not enforced"); recs.push("Return early with a user-friendly message when budget would be exceeded"); }

  return { factorNumber: 4, factorName: FACTOR_NAMES[4], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor5(cfg: AgentConfig): FactorAssessment {
  const checks = {
    allValidated: Boolean(cfg.all_llm_outputs_schema_validated),
    inVcs: Boolean(cfg.schema_definitions_in_version_control),
    retryPattern: Boolean(cfg.has_parse_validate_retry_pattern),
    consistent: Boolean(cfg.schemas_consistent_across_interactions),
    violationsLogged: Boolean(cfg.schema_violations_logged_and_alerted),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.allValidated) evidence.push("All LLM outputs validated against a schema");
  else { gaps.push("Some LLM outputs parsed as raw text"); recs.push("Use Zod schemas with safeParse() for all LLM calls"); }

  if (checks.inVcs) evidence.push("Schema definitions are version-controlled");
  else { gaps.push("Schemas defined inline or not tracked"); recs.push("Store Zod schemas in a schemas/ module under version control"); }

  if (checks.retryPattern) evidence.push("Parse-validate-retry pattern implemented");
  else { gaps.push("Schema failures raise exceptions without retry"); recs.push("On ZodError, re-prompt LLM with schema error context"); }

  if (checks.consistent) evidence.push("Schemas consistent across all LLM interactions");
  else { gaps.push("Schema definitions are inconsistent or duplicated"); recs.push("Centralise schema definitions and import from a single module"); }

  if (checks.violationsLogged) evidence.push("Schema violations logged with full context");
  else { gaps.push("Schema violations silently swallowed"); recs.push("Log every Zod validation failure with model output and request context"); }

  return { factorNumber: 5, factorName: FACTOR_NAMES[5], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor6(cfg: AgentConfig): FactorAssessment {
  const checks = {
    allocation: Boolean(cfg.has_explicit_token_allocation_per_zone),
    measured: Boolean(cfg.context_consumption_measured_per_request),
    autoCompress: Boolean(cfg.has_automatic_compression_when_over_budget),
    slidingWindow: Boolean(cfg.has_sliding_window_for_history),
    audited: Boolean(cfg.system_prompts_audited_for_efficiency),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.allocation) evidence.push("Token allocation defined per context zone");
  else { gaps.push("No explicit context zone allocation"); recs.push("Define % allocations: system 2%, tools 5%, history 33%, dynamic 45%, buffer 15%"); }

  if (checks.measured) evidence.push("Token consumption measured and logged per request per zone");
  else { gaps.push("Context consumption not measured"); recs.push("Count tokens per zone on each request and emit as metrics"); }

  if (checks.autoCompress) evidence.push("Automatic context compression triggers when over budget");
  else { gaps.push("No compression — context window overflows silently"); recs.push("Summarise conversation history when it exceeds its allocation"); }

  if (checks.slidingWindow) evidence.push("Sliding window keeps conversation history within budget");
  else { gaps.push("All conversation history included, consuming unbounded tokens"); recs.push("Implement a sliding window: summarise old turns, keep last N verbatim"); }

  if (checks.audited) evidence.push("System prompts audited for token efficiency");
  else { gaps.push("System prompts never audited for wasted tokens"); recs.push("Run a quarterly token efficiency audit on system prompts"); }

  return { factorNumber: 6, factorName: FACTOR_NAMES[6], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor7(cfg: AgentConfig): FactorAssessment {
  const inputLayers = (cfg.input_guardrail_layer_count as number) ?? 0;
  const outputLayers = (cfg.output_guardrail_layer_count as number) ?? 0;
  const checks = {
    inputGuardrails: inputLayers >= 3,
    outputGuardrails: outputLayers >= 3,
    safetyBoth: Boolean(cfg.safety_filters_on_both_input_and_output),
    injectionDetect: Boolean(cfg.has_prompt_injection_detection),
    piiDetect: Boolean(cfg.has_pii_detection_and_redaction),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.inputGuardrails) evidence.push(`${inputLayers} independent input guardrail layers`);
  else { gaps.push(`Only ${inputLayers} input guardrail layer(s) — need ≥3`); recs.push("Add: rate limiter → validator → PII detector → content filter → injection detector"); }

  if (checks.outputGuardrails) evidence.push(`${outputLayers} independent output guardrail layers`);
  else { gaps.push(`Only ${outputLayers} output guardrail layer(s) — need ≥3`); recs.push("Add: schema validator → PII check → safety filter → leakage detector"); }

  if (checks.safetyBoth) evidence.push("Safety filters applied to both input and output");
  else { gaps.push("Safety filter not applied symmetrically"); recs.push("Ensure every safety check applies to both input and output"); }

  if (checks.injectionDetect) evidence.push("Prompt injection detection in place");
  else { gaps.push("No prompt injection detection"); recs.push("Add an injection detector that scans for role-override attempts"); }

  if (checks.piiDetect) evidence.push("PII detection and redaction applied");
  else { gaps.push("PII may flow into LLM context or be exposed in responses"); recs.push("Add PII detection on input and output"); }

  return { factorNumber: 7, factorName: FACTOR_NAMES[7], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor8(cfg: AgentConfig): FactorAssessment {
  const checks = {
    llmFallback: Boolean(cfg.has_fallback_llm_provider),
    vectorFallback: Boolean(cfg.has_fallback_for_vector_db),
    staticFallback: Boolean(cfg.has_static_response_for_complete_failure),
    logged: Boolean(cfg.degradation_events_logged),
    chaosTested: Boolean(cfg.degradation_regularly_chaos_tested),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.llmFallback) evidence.push("Fallback LLM provider configured");
  else { gaps.push("No LLM fallback — primary provider outage causes complete failure"); recs.push("Add a fallback provider chain: primary → secondary → cheaper model → static"); }

  if (checks.vectorFallback) evidence.push("Vector database fallback implemented");
  else { gaps.push("No fallback when vector database is unavailable"); recs.push("Continue without RAG when vector DB is unavailable"); }

  if (checks.staticFallback) evidence.push("Static fallback response for complete system failure");
  else { gaps.push("Complete system failure exposes raw errors to users"); recs.push("Add a final catch-all that returns a polite error message"); }

  if (checks.logged) evidence.push("Degradation events logged for postmortem analysis");
  else { gaps.push("Degradation events are silent"); recs.push("Log every degradation event with level, reason, and fallback taken"); }

  if (checks.chaosTested) evidence.push("Degradation paths regularly tested via chaos engineering");
  else { gaps.push("Fallback paths untested"); recs.push("Run monthly chaos tests: kill primary LLM, kill vector DB"); }

  return { factorNumber: 8, factorName: FACTOR_NAMES[8], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor9(cfg: AgentConfig): FactorAssessment {
  const checks = {
    tracing: Boolean(cfg.has_request_tracing_with_trace_ids),
    centralized: Boolean(cfg.traces_exported_to_centralized_system),
    metricsTracked: Boolean(cfg.key_metrics_tracked_latency_tokens_cost_errors),
    dashboards: Boolean(cfg.has_dashboards_for_metrics),
    alerts: Boolean(cfg.has_alerts_for_metric_degradation),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.tracing) evidence.push("Unique trace IDs generated for every request");
  else { gaps.push("No request tracing"); recs.push("Generate a UUID trace_id for every request"); }

  if (checks.centralized) evidence.push("Traces exported to a centralized system");
  else { gaps.push("Traces only in local logs"); recs.push("Export traces via OpenTelemetry to LangSmith, Arize, or similar"); }

  if (checks.metricsTracked) evidence.push("Latency, token usage, cost, and error rate tracked per request");
  else { gaps.push("Key metrics not consistently tracked"); recs.push("Emit structured metrics: p50/p95 latency, tokens, cost, error_rate"); }

  if (checks.dashboards) evidence.push("Operational dashboards show real-time metrics");
  else { gaps.push("No dashboards — metrics not visible to the team"); recs.push("Create a Grafana or DataDog dashboard with the 5 key agent metrics"); }

  if (checks.alerts) evidence.push("Alerts fire when metrics degrade");
  else { gaps.push("No metric alerts — degradation discovered by users, not ops"); recs.push("Set up alerts: error rate >1%, p95 latency >5s, cost >2x baseline"); }

  return { factorNumber: 9, factorName: FACTOR_NAMES[9], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor10(cfg: AgentConfig): FactorAssessment {
  const checks = {
    policy: Boolean(cfg.has_approval_policy_defined),
    flagged: Boolean(cfg.high_stakes_actions_flagged_for_approval),
    interface: Boolean(cfg.has_reviewer_interface),
    timeout: Boolean(cfg.has_timeout_handling_for_approvals),
    logged: Boolean(cfg.approval_decisions_logged_for_audit),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.policy) evidence.push("Explicit approval policy defines which actions require review");
  else { gaps.push("No approval policy"); recs.push("Define an ApprovalPolicy with action types, thresholds, and risk levels"); }

  if (checks.flagged) evidence.push("High-stakes actions flagged for approval");
  else { gaps.push("High-stakes actions execute without human oversight"); recs.push("Categorise actions by risk: auto-execute LOW, review MEDIUM/HIGH/CRITICAL"); }

  if (checks.interface) evidence.push("Reviewer interface exists for approvals");
  else { gaps.push("No reviewer interface"); recs.push("Build a Slack/email/web UI for reviewers with approve/reject/edit options"); }

  if (checks.timeout) evidence.push("Approval requests time out safely");
  else { gaps.push("Approval requests can hang indefinitely"); recs.push("Add timeout: default to rejection after N seconds"); }

  if (checks.logged) evidence.push("Approval decisions logged with reviewer, reason, and timestamp");
  else { gaps.push("No audit trail for approval decisions"); recs.push("Log every approval decision to a tamper-evident audit log"); }

  return { factorNumber: 10, factorName: FACTOR_NAMES[10], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor11(cfg: AgentConfig): FactorAssessment {
  const checks = {
    testSet: Boolean(cfg.test_set_has_50_plus_queries),
    onEveryDeploy: Boolean(cfg.evaluations_run_on_every_deployment),
    regression: Boolean(cfg.has_regression_detection_baseline_comparison),
    redTeam: Boolean(cfg.safety_red_team_run_regularly),
    blocks: Boolean(cfg.evaluation_results_block_deployment_on_regression),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.testSet) evidence.push("Test set of 50+ queries covering all intent categories");
  else { gaps.push("No evaluation test set (or fewer than 50 queries)"); recs.push("Build a test set of 50+ queries: happy path, edge cases, and adversarial"); }

  if (checks.onEveryDeploy) evidence.push("Full evaluation runs on every deployment");
  else { gaps.push("Evaluations are manual and infrequent"); recs.push("Add evaluation step to CI/CD pipeline"); }

  if (checks.regression) evidence.push("Regression detection compares each run to a stored baseline");
  else { gaps.push("No regression detection"); recs.push("Store evaluation baseline and alert when any metric drops >5%"); }

  if (checks.redTeam) evidence.push("Safety red-team evaluations run on every significant change");
  else { gaps.push("No safety red-team evaluations"); recs.push("Run adversarial safety prompts on every significant change or weekly"); }

  if (checks.blocks) evidence.push("Deployment blocked when evaluation detects significant regression");
  else { gaps.push("Regressions reported but don't block deployment"); recs.push("Make evaluation results a deployment gate: fail CI on critical regressions"); }

  return { factorNumber: 11, factorName: FACTOR_NAMES[11], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

function assessFactor12(cfg: AgentConfig): FactorAssessment {
  const checks = {
    sameGuardrails: Boolean(cfg.same_guardrail_config_in_dev_and_prod),
    prodModelTested: Boolean(cfg.production_model_tested_before_deployment),
    stagingExists: Boolean(cfg.has_staging_environment_matching_production),
    kbConsistent: Boolean(cfg.knowledge_base_structures_consistent_across_envs),
    diffsDocumented: Boolean(cfg.environment_differences_documented_and_intentional),
  };
  const passed = Object.values(checks).filter(Boolean).length;
  const score = checksToScore(passed);
  const evidence: string[] = [];
  const gaps: string[] = [];
  const recs: string[] = [];

  if (checks.sameGuardrails) evidence.push("Guardrail configuration identical in dev and production");
  else { gaps.push("Guardrails differ between dev and prod"); recs.push("Use a single guardrail config file loaded in all environments"); }

  if (checks.prodModelTested) evidence.push("Production model tested before every deployment");
  else { gaps.push("Dev uses a cheaper model — production model behaviour not verified until deploy"); recs.push("Add a pre-deployment test step with the prod model"); }

  if (checks.stagingExists) evidence.push("Staging environment mirrors production configuration");
  else { gaps.push("No staging environment"); recs.push("Maintain a staging environment with the same config as prod"); }

  if (checks.kbConsistent) evidence.push("Knowledge base schemas consistent across environments");
  else { gaps.push("Knowledge base structure differs across environments"); recs.push("Keep knowledge base schemas identical across all environments"); }

  if (checks.diffsDocumented) evidence.push("Intentional environment differences documented");
  else { gaps.push("Environment differences undocumented and may be accidental"); recs.push("Document every deliberate difference in an environment parity file"); }

  return { factorNumber: 12, factorName: FACTOR_NAMES[12], score, status: scoreToStatus(score), evidence, gaps, recommendations: recs };
}

const FACTOR_ASSESSORS: Array<(cfg: AgentConfig) => FactorAssessment> = [
  assessFactor1, assessFactor2, assessFactor3, assessFactor4,
  assessFactor5, assessFactor6, assessFactor7, assessFactor8,
  assessFactor9, assessFactor10, assessFactor11, assessFactor12,
];

// ---------------------------------------------------------------------------
// Maturity level calculation
// ---------------------------------------------------------------------------

function computeMaturity(
  factors: FactorAssessment[],
  totalScore: number,
): [TwelveFactorReport["maturityLevel"], number] {
  const scores: Record<number, number> = {};
  for (const f of factors) scores[f.factorNumber] = f.score;

  const allAtLeast = (nums: number[], min: number): boolean =>
    nums.every((n) => (scores[n] ?? 1) >= min);

  if (totalScore === 60 && allAtLeast([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4)) return ["Elite", 5];
  if (totalScore >= 49 && allAtLeast([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3)) return ["Production", 4];
  if (totalScore >= 37 && allAtLeast([1, 2, 3, 4, 5, 6, 8, 9], 3)) return ["Staging", 3];
  if (totalScore >= 25 && allAtLeast([1, 2, 5, 9], 3)) return ["Development", 2];
  return ["Prototype", 1];
}

// ---------------------------------------------------------------------------
// TwelveFactorAssessor class
// ---------------------------------------------------------------------------

/**
 * Evaluates an agent configuration against all 12 production-readiness factors.
 *
 * @example
 * ```typescript
 * const assessor = new TwelveFactorAssessor();
 * const report = assessor.assess(myAgentConfig);
 * console.log(assessor.exportReport(report));
 * ```
 */
export class TwelveFactorAssessor {
  /**
   * Evaluate an agent configuration against all 12 factors.
   * @param agentOrConfig - Agent configuration object with boolean fields per factor
   * @returns Validated TwelveFactorReport
   */
  assess(agentOrConfig: AgentConfig): TwelveFactorReport {
    const factors = FACTOR_ASSESSORS.map((fn) => fn(agentOrConfig));
    const total = factors.reduce((sum, f) => sum + f.score, 0);
    const [levelName, levelNum] = computeMaturity(factors, total);

    const criticalGaps = factors
      .filter((f) => f.score <= 2)
      .map((f) => `Factor ${f.factorNumber} (${f.factorName}) scored ${f.score}/5 — ${f.gaps[0] ?? "see report"}`);

    const sortedByScore = [...factors].sort((a, b) => a.score - b.score || a.factorNumber - b.factorNumber);
    const priorities = sortedByScore
      .filter((f) => f.score < 5 && f.recommendations.length > 0)
      .map((f) => `Improve Factor ${f.factorNumber} (${f.factorName}): ${f.recommendations[0]}`)
      .slice(0, 7);

    return TwelveFactorReportSchema.parse({
      overallScore: total,
      maturityLevel: levelName,
      maturityLevelNumber: levelNum,
      factors,
      criticalGaps,
      improvementPriorities: priorities,
      assessedAt: today(),
    });
  }

  /**
   * Evaluate a single factor by number (1-12).
   */
  assessFactor(factorNumber: number, agentOrConfig: AgentConfig): FactorAssessment {
    if (factorNumber < 1 || factorNumber > 12) throw new RangeError(`Factor number must be 1-12, got ${factorNumber}`);
    return FACTOR_ASSESSORS[factorNumber - 1](agentOrConfig);
  }

  /**
   * Return a flat, ordered list of actionable recommendations.
   */
  generateRecommendations(report: TwelveFactorReport): string[] {
    const recs: string[] = [];
    for (const f of report.factors) {
      if (f.score <= 2) for (const r of f.recommendations) recs.push(`[CRITICAL] Factor ${f.factorNumber} — ${r}`);
    }
    for (const f of report.factors) {
      if (f.score === 3) for (const r of f.recommendations) recs.push(`[IMPROVE] Factor ${f.factorNumber} — ${r}`);
    }
    for (const f of report.factors) {
      if (f.score === 4) for (const r of f.recommendations) recs.push(`[POLISH] Factor ${f.factorNumber} — ${r}`);
    }
    return recs;
  }

  /**
   * Compare a current report to a previous baseline.
   */
  compareToBaseline(current: TwelveFactorReport, baseline: TwelveFactorReport): ComparisonReport {
    const baseScores: Record<number, number> = {};
    for (const f of baseline.factors) baseScores[f.factorNumber] = f.score;
    const currScores: Record<number, number> = {};
    for (const f of current.factors) currScores[f.factorNumber] = f.score;

    const improved: Array<[number, string, number, number]> = [];
    const regressed: Array<[number, string, number, number]> = [];
    const unchanged: Array<[number, string, number]> = [];

    for (let n = 1; n <= 12; n++) {
      const name = FACTOR_NAMES[n];
      const old = baseScores[n] ?? 1;
      const next = currScores[n] ?? 1;
      if (next > old) improved.push([n, name, old, next]);
      else if (next < old) regressed.push([n, name, old, next]);
      else unchanged.push([n, name, old]);
    }

    return ComparisonReportSchema.parse({
      baselineScore: baseline.overallScore,
      currentScore: current.overallScore,
      scoreDelta: current.overallScore - baseline.overallScore,
      baselineLevel: baseline.maturityLevel,
      currentLevel: current.maturityLevel,
      improvedFactors: improved,
      regressedFactors: regressed,
      unchangedFactors: unchanged,
    });
  }

  /**
   * Export report as markdown or HTML.
   */
  exportReport(report: TwelveFactorReport, fmt: "markdown" | "html" = "markdown"): string {
    if (fmt === "markdown") return generateMarkdownReport(report);
    if (fmt === "html") return generateHtmlReport(report);
    throw new Error(`Unknown format: ${fmt}`);
  }
}

// ---------------------------------------------------------------------------
// ImprovementTracker class
// ---------------------------------------------------------------------------

/**
 * Tracks maturity assessments over time and generates improvement roadmaps.
 */
export class ImprovementTracker {
  private history: TwelveFactorReport[] = [];
  private assessor = new TwelveFactorAssessor();

  /** Store a report as the current baseline (appended to history). */
  saveBaseline(report: TwelveFactorReport): void {
    this.history.push(report);
  }

  /** Compare to the most recent saved baseline. */
  compareToBaseline(current: TwelveFactorReport): ComparisonReport | null {
    if (this.history.length === 0) return null;
    return this.assessor.compareToBaseline(current, this.history[this.history.length - 1]);
  }

  /** Summarise score trends across all saved assessments. */
  trackOverTime(): TrendReport {
    if (this.history.length === 0) throw new Error("No assessments recorded yet");
    const assessments = this.history.map((r) => ({ date: r.assessedAt, score: r.overallScore, level: r.maturityLevel }));
    const first = this.history[0].overallScore;
    const latest = this.history[this.history.length - 1].overallScore;
    const delta = latest - first;
    const avg = delta / Math.max(this.history.length - 1, 1);
    return { assessments, firstScore: first, latestScore: latest, totalImprovement: delta, averageImprovementPerAssessment: Math.round(avg * 100) / 100 };
  }

  /** Return ordered steps to reach the target maturity level. */
  generateRoadmap(current: TwelveFactorReport, targetLevel: string): string[] {
    const levelMap: Record<string, number> = {
      prototype: 1, development: 2, staging: 3, production: 4, elite: 5,
    };
    const targetNum = levelMap[targetLevel.toLowerCase()];
    if (targetNum === undefined) throw new Error(`Unknown target level: ${targetLevel}`);
    if (current.maturityLevelNumber >= targetNum) return [`Already at or above ${targetLevel} level.`];

    const required: Record<number, number[]> = {
      2: [1, 2, 5, 9],
      3: [1, 2, 3, 4, 5, 6, 8, 9],
      4: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    };
    const scoreByFactor: Record<number, number> = {};
    for (const f of current.factors) scoreByFactor[f.factorNumber] = f.score;

    const steps: string[] = [];
    for (let level = current.maturityLevelNumber + 1; level <= targetNum; level++) {
      for (const fn of required[level] ?? []) {
        if ((scoreByFactor[fn] ?? 1) < 3) {
          const fa = current.factors.find((f) => f.factorNumber === fn);
          const rec = fa?.recommendations[0] ?? "See assessment gaps";
          steps.push(`[Level ${level}] Bring Factor ${fn} (${FACTOR_NAMES[fn]}) to ≥3: ${rec}`);
        }
      }
    }
    if (targetNum === 5) {
      for (const f of current.factors) {
        if (f.score === 3 && f.recommendations.length > 0) {
          steps.push(`[Elite] Raise Factor ${f.factorNumber} (${f.factorName}) from 3→4: ${f.recommendations[0]}`);
        }
      }
    }
    return steps.length > 0 ? steps : ["No additional steps required — you are on track."];
  }
}

// ---------------------------------------------------------------------------
// Report generators
// ---------------------------------------------------------------------------

const STATUS_EMOJI: Record<string, string> = {
  excellent: "✅ Excellent",
  good: "✅ Good",
  needs_improvement: "⚠️ Needs Improvement",
  critical: "❌ Critical",
};

/**
 * Generate a markdown assessment report.
 */
export function generateMarkdownReport(report: TwelveFactorReport): string {
  const lines: string[] = [
    "# 12-Factor Agent Assessment Report",
    "",
    `**Date:** ${report.assessedAt}`,
    `**Overall Score:** ${report.overallScore}/60`,
    `**Maturity Level:** ${report.maturityLevel} (Level ${report.maturityLevelNumber})`,
    "",
    "## Factor Scores",
    "",
    "| # | Factor | Score | Status |",
    "|---|--------|-------|--------|",
  ];

  for (const f of report.factors) {
    const roman = ROMAN[f.factorNumber];
    const status = STATUS_EMOJI[f.status] ?? f.status;
    lines.push(`| ${roman} | ${f.factorName} | ${f.score}/5 | ${status} |`);
  }

  const critical = report.factors.filter((f) => f.score <= 2);
  if (critical.length > 0) {
    lines.push("", "## Critical Gaps (Must Fix)", "");
    critical.forEach((f, i) => {
      lines.push(`${i + 1}. **Factor ${ROMAN[f.factorNumber]} — ${f.factorName} (Score: ${f.score}/5)**`);
      f.gaps.forEach((g) => lines.push(`   - Missing: ${g}`));
      f.recommendations.forEach((r) => lines.push(`   - Recommendation: ${r}`));
      lines.push("");
    });
  }

  if (report.improvementPriorities.length > 0) {
    lines.push("## Improvement Priorities", "");
    report.improvementPriorities.forEach((p, i) => lines.push(`${i + 1}. ${p}`));
  }

  return lines.join("\n");
}

/**
 * Generate an HTML assessment report.
 */
export function generateHtmlReport(report: TwelveFactorReport): string {
  const pct = Math.round((report.overallScore / 60) * 100);
  const scoreColors: Record<number, string> = {
    5: "#22c55e", 4: "#86efac", 3: "#eab308", 2: "#ef4444", 1: "#b91c1c",
  };

  const rows = report.factors.map((f) => {
    const color = scoreColors[f.score] ?? "#888";
    const barWidth = f.score * 20;
    const roman = ROMAN[f.factorNumber];
    const status = STATUS_EMOJI[f.status] ?? f.status;
    return `<tr>
      <td>${roman}</td>
      <td>${f.factorName}</td>
      <td><div style="background:${color};width:${barWidth}px;height:16px;border-radius:4px;display:inline-block"></div>
          <strong style="color:${color}">${f.score}/5</strong></td>
      <td>${status}</td>
    </tr>`;
  }).join("");

  const gapsHtml = report.criticalGaps.map((g) => `<li>${g}</li>`).join("") || "<li>None — no critical gaps!</li>";
  const prioHtml = report.improvementPriorities.map((p) => `<li>${p}</li>`).join("") || "<li>All factors at maximum score.</li>";

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>12-Factor Agent Assessment</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1e293b; }
  .scorecard { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }
  .score-big { font-size: 2.5rem; font-weight: 700; }
  .progress-bar { background: #e2e8f0; border-radius: 99px; height: 12px; margin: 0.5rem 0; }
  .progress-fill { background: #22c55e; border-radius: 99px; height: 12px; width: ${pct}%; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e2e8f0; }
  th { background: #f1f5f9; font-weight: 600; }
</style>
</head>
<body>
<h1>12-Factor Agent Assessment</h1>
<div class="scorecard">
  <div class="score-big">${report.overallScore}/60</div>
  <div>${report.maturityLevel} (Level ${report.maturityLevelNumber}) — ${report.assessedAt}</div>
  <div class="progress-bar"><div class="progress-fill"></div></div>
  <small>${pct}% toward Elite</small>
</div>
<h2>Factor Scores</h2>
<table>
  <thead><tr><th>#</th><th>Factor</th><th>Score</th><th>Status</th></tr></thead>
  <tbody>${rows}</tbody>
</table>
<h2>Critical Gaps</h2>
<ul>${gapsHtml}</ul>
<h2>Improvement Priorities</h2>
<ol>${prioHtml}</ol>
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function demo(): void {
  const assessor = new TwelveFactorAssessor();
  const tracker = new ImprovementTracker();

  const baselineConfig: AgentConfig = {
    prompts_in_version_control: true,
    prompts_semantically_versioned: true,
    prompts_code_reviewed: false,
    prompts_have_change_log: false,
    prompts_independently_rollbackable: false,
    has_conversation_state_class: true,
    state_survives_truncation: true,
    state_is_serializable: true,
    state_persisted_across_sessions: false,
    state_transitions_logged: false,
    has_llm_provider_interface: false,
    num_providers_supported: 1,
    provider_configurable_via_env: false,
    provider_specific_features_abstracted: false,
    has_provider_fallback_chain: false,
    has_per_request_token_budget: false,
    has_per_request_cost_budget: false,
    token_usage_tracked_and_logged: false,
    cost_alerts_configured: false,
    budget_enforcement_blocks_excess: false,
    all_llm_outputs_schema_validated: true,
    schema_definitions_in_version_control: true,
    has_parse_validate_retry_pattern: true,
    schemas_consistent_across_interactions: true,
    schema_violations_logged_and_alerted: true,
    has_explicit_token_allocation_per_zone: true,
    context_consumption_measured_per_request: true,
    has_automatic_compression_when_over_budget: false,
    has_sliding_window_for_history: false,
    system_prompts_audited_for_efficiency: false,
    input_guardrail_layer_count: 2,
    output_guardrail_layer_count: 2,
    safety_filters_on_both_input_and_output: true,
    has_prompt_injection_detection: false,
    has_pii_detection_and_redaction: false,
    has_fallback_llm_provider: false,
    has_fallback_for_vector_db: false,
    has_static_response_for_complete_failure: true,
    degradation_events_logged: true,
    degradation_regularly_chaos_tested: false,
    has_request_tracing_with_trace_ids: true,
    traces_exported_to_centralized_system: false,
    key_metrics_tracked_latency_tokens_cost_errors: true,
    has_dashboards_for_metrics: false,
    has_alerts_for_metric_degradation: false,
    has_approval_policy_defined: false,
    high_stakes_actions_flagged_for_approval: false,
    has_reviewer_interface: false,
    has_timeout_handling_for_approvals: false,
    approval_decisions_logged_for_audit: false,
    test_set_has_50_plus_queries: true,
    evaluations_run_on_every_deployment: false,
    has_regression_detection_baseline_comparison: false,
    safety_red_team_run_regularly: false,
    evaluation_results_block_deployment_on_regression: false,
    same_guardrail_config_in_dev_and_prod: false,
    production_model_tested_before_deployment: false,
    has_staging_environment_matching_production: true,
    knowledge_base_structures_consistent_across_envs: true,
    environment_differences_documented_and_intentional: false,
  };

  const baselineReport = assessor.assess(baselineConfig);
  console.log("=== BASELINE ASSESSMENT ===");
  console.log(assessor.exportReport(baselineReport));
  console.log(`\nMaturity: ${baselineReport.maturityLevel} (Level ${baselineReport.maturityLevelNumber})`);

  tracker.saveBaseline(baselineReport);

  const improvedConfig: AgentConfig = {
    ...baselineConfig,
    has_llm_provider_interface: true,
    num_providers_supported: 2,
    provider_configurable_via_env: true,
    has_provider_fallback_chain: true,
    has_per_request_token_budget: true,
    has_per_request_cost_budget: true,
    token_usage_tracked_and_logged: true,
    cost_alerts_configured: true,
    budget_enforcement_blocks_excess: true,
    has_approval_policy_defined: true,
    high_stakes_actions_flagged_for_approval: true,
    has_reviewer_interface: true,
    has_timeout_handling_for_approvals: true,
    approval_decisions_logged_for_audit: true,
    evaluations_run_on_every_deployment: true,
    has_regression_detection_baseline_comparison: true,
  };

  const improvedReport = assessor.assess(improvedConfig);
  console.log("\n=== IMPROVED ASSESSMENT ===");
  console.log(`Score: ${improvedReport.overallScore}/60 — ${improvedReport.maturityLevel}`);

  const comparison = assessor.compareToBaseline(improvedReport, baselineReport);
  console.log(`\nScore delta: ${comparison.scoreDelta > 0 ? "+" : ""}${comparison.scoreDelta}`);
  console.log(`Level: ${comparison.baselineLevel} → ${comparison.currentLevel}`);

  tracker.saveBaseline(improvedReport);
  const trend = tracker.trackOverTime();
  console.log(`\nImprovement: +${trend.totalImprovement} points over ${trend.assessments.length} assessments`);

  const roadmap = tracker.generateRoadmap(improvedReport, "elite");
  console.log("\n=== ROADMAP TO ELITE ===");
  roadmap.forEach((step, i) => console.log(`  ${i + 1}. ${step}`));
}

// Run demo when executed directly
if (require.main === module) {
  demo();
}
