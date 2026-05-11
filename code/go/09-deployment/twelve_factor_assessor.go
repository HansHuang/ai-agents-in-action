// Package deployment provides the 12-Factor Agent self-assessment tool.
// It evaluates an agent configuration against all 12 production-readiness
// factors and generates actionable reports with roadmaps and improvement tracking.
//
// Reference: docs/09-from-dev-to-production/02-the-12-factor-agent.md
//
// Usage:
//
//	assessor := deployment.NewTwelveFactorAssessor()
//	report := assessor.Assess(config)
//	fmt.Println(assessor.ExportReport(report, "markdown"))
package main

import (
	"fmt"
	"html/template"
	"os"
	"sort"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

// AgentConfig holds boolean and numeric flags describing the agent's
// implementation against each of the 12 factors.
type AgentConfig map[string]interface{}

// getBool returns a bool from the config, defaulting to false.
func (c AgentConfig) getBool(key string) bool {
	v, ok := c[key]
	if !ok {
		return false
	}
	b, _ := v.(bool)
	return b
}

// getInt returns an int from the config, defaulting to 0.
func (c AgentConfig) getInt(key string) int {
	v, ok := c[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	}
	return 0
}

// FactorAssessment holds the assessment result for a single factor.
type FactorAssessment struct {
	FactorNumber    int
	FactorName      string
	Score           int    // 1-5
	Status          string // "critical" | "needs_improvement" | "good" | "excellent"
	Evidence        []string
	Gaps            []string
	Recommendations []string
}

// TwelveFactorReport is the complete report for all 12 factors.
type TwelveFactorReport struct {
	OverallScore          int
	MaturityLevel         string // "Prototype" through "Elite"
	MaturityLevelNumber   int    // 1-5
	Factors               []FactorAssessment
	CriticalGaps          []string
	ImprovementPriorities []string
	AssessedAt            string
}

// ComparisonReport compares a current report to a baseline.
type ComparisonReport struct {
	BaselineScore    int
	CurrentScore     int
	ScoreDelta       int
	BaselineLevel    string
	CurrentLevel     string
	ImprovedFactors  []FactorDelta
	RegressedFactors []FactorDelta
	UnchangedFactors []FactorUnchanged
}

// FactorDelta describes a factor score that changed.
type FactorDelta struct {
	FactorNumber int
	FactorName   string
	OldScore     int
	NewScore     int
}

// FactorUnchanged describes a factor score that stayed the same.
type FactorUnchanged struct {
	FactorNumber int
	FactorName   string
	Score        int
}

// TrendReport summarises score changes over multiple assessments.
type TrendReport struct {
	Assessments                    []AssessmentSummary
	FirstScore                     int
	LatestScore                    int
	TotalImprovement               int
	AverageImprovementPerAssessment float64
}

// AssessmentSummary is one entry in a trend report.
type AssessmentSummary struct {
	Date  string
	Score int
	Level string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

var factorNames = map[int]string{
	1:  "Prompt as Code",
	2:  "Explicit State",
	3:  "Provider Agnostic",
	4:  "Token Budgeting",
	5:  "Structured Everything",
	6:  "Context Is a Resource",
	7:  "Defense in Depth",
	8:  "Graceful Degradation",
	9:  "Observability First",
	10: "Human in the Loop",
	11: "Continuous Evaluation",
	12: "Dev-Prod Parity",
}

var romanNumerals = map[int]string{
	1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
	7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII",
}

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------

func scoreToStatus(score int) string {
	switch {
	case score <= 2:
		return "critical"
	case score == 3:
		return "needs_improvement"
	case score == 4:
		return "good"
	default:
		return "excellent"
	}
}

func checksToScore(passed, total int) int {
	if passed <= 0 {
		return 1
	}
	if passed > 5 {
		return 5
	}
	return passed
}

func today() string {
	return time.Now().Format("2006-01-02")
}

// ---------------------------------------------------------------------------
// Individual factor assessors
// ---------------------------------------------------------------------------

func assessFactor1(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("prompts_in_version_control"),
		cfg.getBool("prompts_semantically_versioned"),
		cfg.getBool("prompts_code_reviewed"),
		cfg.getBool("prompts_have_change_log"),
		cfg.getBool("prompts_independently_rollbackable"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Prompts stored in version control (Git)")
	} else {
		gaps = append(gaps, "Prompts not tracked in version control")
		recs = append(recs, "Move prompts to a prompts/ directory under version control")
	}
	if checks[1] {
		evidence = append(evidence, "Prompts use semantic versioning")
	} else {
		gaps = append(gaps, "Prompts lack semantic versioning")
		recs = append(recs, "Add a version field to each prompt file (YAML frontmatter)")
	}
	if checks[2] {
		evidence = append(evidence, "Prompt changes go through code review")
	} else {
		gaps = append(gaps, "Prompt changes bypass code review")
		recs = append(recs, "Add prompts/ to required CODEOWNERS for PR review")
	}
	if checks[3] {
		evidence = append(evidence, "Prompts include a change_log field")
	} else {
		gaps = append(gaps, "No prompt change log")
		recs = append(recs, "Add change_log field to every prompt YAML")
	}
	if checks[4] {
		evidence = append(evidence, "Prompts can be rolled back independently of application code")
	} else {
		gaps = append(gaps, "Prompt rollback requires full application rollback")
		recs = append(recs, "Decouple prompt deployment from application deployment")
	}
	return FactorAssessment{FactorNumber: 1, FactorName: factorNames[1], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor2(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("has_conversation_state_class"),
		cfg.getBool("state_survives_truncation"),
		cfg.getBool("state_is_serializable"),
		cfg.getBool("state_persisted_across_sessions"),
		cfg.getBool("state_transitions_logged"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "ConversationState struct exists")
	} else {
		gaps = append(gaps, "No explicit ConversationState struct found")
		recs = append(recs, "Create a ConversationState struct with all session fields")
	}
	if checks[1] {
		evidence = append(evidence, "State injected into every LLM call, surviving message truncation")
	} else {
		gaps = append(gaps, "State may be lost when message list is truncated")
		recs = append(recs, "Inject a state summary into every LLM call prompt")
	}
	if checks[2] {
		evidence = append(evidence, "State is JSON-serializable (MarshalJSON implemented)")
	} else {
		gaps = append(gaps, "State cannot be serialized for debugging or persistence")
		recs = append(recs, "Implement json.Marshaler on ConversationState")
	}
	if checks[3] {
		evidence = append(evidence, "State is persisted to a durable store (DB/Redis)")
	} else {
		gaps = append(gaps, "State is lost when session ends")
		recs = append(recs, "Persist ConversationState to Redis or a database by session ID")
	}
	if checks[4] {
		evidence = append(evidence, "State transitions are logged")
	} else {
		gaps = append(gaps, "State changes are not logged")
		recs = append(recs, "Log every state field change with timestamp and reason")
	}
	return FactorAssessment{FactorNumber: 2, FactorName: factorNames[2], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor3(cfg AgentConfig) FactorAssessment {
	numProviders := cfg.getInt("num_providers_supported")
	checks := []bool{
		cfg.getBool("has_llm_provider_interface"),
		numProviders >= 2,
		cfg.getBool("provider_configurable_via_env"),
		cfg.getBool("provider_specific_features_abstracted"),
		cfg.getBool("has_provider_fallback_chain"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "LLM provider interface exists")
	} else {
		gaps = append(gaps, "LLM calls tightly coupled to a specific provider")
		recs = append(recs, "Define an LLMProvider interface with a Chat() method")
	}
	if checks[1] {
		evidence = append(evidence, fmt.Sprintf("%d provider(s) supported", numProviders))
	} else {
		gaps = append(gaps, fmt.Sprintf("Only %d provider(s) supported — need ≥2", numProviders))
		recs = append(recs, "Implement at least one fallback provider (e.g., Anthropic alongside OpenAI)")
	}
	if checks[2] {
		evidence = append(evidence, "Provider configured via environment variables")
	} else {
		gaps = append(gaps, "Provider is hardcoded in source code")
		recs = append(recs, "Read LLM_PROVIDER and LLM_MODEL from os.Getenv()")
	}
	if checks[3] {
		evidence = append(evidence, "Provider-specific features abstracted behind interface")
	} else {
		gaps = append(gaps, "Provider-specific API details leak into application code")
		recs = append(recs, "Wrap provider-specific features behind a common interface method")
	}
	if checks[4] {
		evidence = append(evidence, "Provider fallback chain configured")
	} else {
		gaps = append(gaps, "No fallback chain — single provider failure causes outage")
		recs = append(recs, "Implement a provider fallback chain with circuit breaker")
	}
	return FactorAssessment{FactorNumber: 3, FactorName: factorNames[3], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor4(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("has_per_request_token_budget"),
		cfg.getBool("has_per_request_cost_budget"),
		cfg.getBool("token_usage_tracked_and_logged"),
		cfg.getBool("cost_alerts_configured"),
		cfg.getBool("budget_enforcement_blocks_excess"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Per-request token budget defined")
	} else {
		gaps = append(gaps, "No per-request token budget")
		recs = append(recs, "Define a TokenBudget struct with MaxTokens per request")
	}
	if checks[1] {
		evidence = append(evidence, "Per-request cost budget defined")
	} else {
		gaps = append(gaps, "No per-request cost budget")
		recs = append(recs, "Add MaxCostUSD to TokenBudget alongside MaxTokens")
	}
	if checks[2] {
		evidence = append(evidence, "Token counts tracked and logged per request")
	} else {
		gaps = append(gaps, "Token usage not tracked")
		recs = append(recs, "Log PromptTokens, CompletionTokens, and TotalCostUSD per request")
	}
	if checks[3] {
		evidence = append(evidence, "Cost alerts configured for budget overruns")
	} else {
		gaps = append(gaps, "No cost alerts — cost spikes go undetected")
		recs = append(recs, "Alert when request cost exceeds 2x baseline average")
	}
	if checks[4] {
		evidence = append(evidence, "Requests exceeding budget are blocked before LLM call")
	} else {
		gaps = append(gaps, "Budget tracked but not enforced")
		recs = append(recs, "Return an error when budget would be exceeded before calling the LLM")
	}
	return FactorAssessment{FactorNumber: 4, FactorName: factorNames[4], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor5(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("all_llm_outputs_schema_validated"),
		cfg.getBool("schema_definitions_in_version_control"),
		cfg.getBool("has_parse_validate_retry_pattern"),
		cfg.getBool("schemas_consistent_across_interactions"),
		cfg.getBool("schema_violations_logged_and_alerted"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "All LLM outputs validated against a schema")
	} else {
		gaps = append(gaps, "Some LLM outputs parsed as raw text")
		recs = append(recs, "Use json.Unmarshal into typed structs for all LLM responses")
	}
	if checks[1] {
		evidence = append(evidence, "Schema definitions are version-controlled")
	} else {
		gaps = append(gaps, "Schemas defined inline or not tracked")
		recs = append(recs, "Store struct definitions in a schemas/ package under version control")
	}
	if checks[2] {
		evidence = append(evidence, "Parse-validate-retry pattern implemented")
	} else {
		gaps = append(gaps, "Schema failures raise panics without retry")
		recs = append(recs, "On json.UnmarshalError, re-prompt LLM with schema error context")
	}
	if checks[3] {
		evidence = append(evidence, "Schemas consistent across all LLM interactions")
	} else {
		gaps = append(gaps, "Schema definitions are inconsistent or duplicated")
		recs = append(recs, "Centralise struct definitions and import from a single package")
	}
	if checks[4] {
		evidence = append(evidence, "Schema violations logged with full context")
	} else {
		gaps = append(gaps, "Schema violations silently swallowed")
		recs = append(recs, "Log every unmarshal failure with model output and request context")
	}
	return FactorAssessment{FactorNumber: 5, FactorName: factorNames[5], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor6(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("has_explicit_token_allocation_per_zone"),
		cfg.getBool("context_consumption_measured_per_request"),
		cfg.getBool("has_automatic_compression_when_over_budget"),
		cfg.getBool("has_sliding_window_for_history"),
		cfg.getBool("system_prompts_audited_for_efficiency"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Token allocation defined per context zone")
	} else {
		gaps = append(gaps, "No explicit context zone allocation")
		recs = append(recs, "Define percentage allocations: system 2%, tools 5%, history 33%, dynamic 45%, buffer 15%")
	}
	if checks[1] {
		evidence = append(evidence, "Token consumption measured and logged per request per zone")
	} else {
		gaps = append(gaps, "Context consumption not measured")
		recs = append(recs, "Count tokens per zone on each request and emit as metrics")
	}
	if checks[2] {
		evidence = append(evidence, "Automatic context compression triggers when over budget")
	} else {
		gaps = append(gaps, "No compression — context window overflows silently")
		recs = append(recs, "Summarise conversation history when it exceeds its allocation")
	}
	if checks[3] {
		evidence = append(evidence, "Sliding window keeps conversation history within budget")
	} else {
		gaps = append(gaps, "All conversation history included, consuming unbounded tokens")
		recs = append(recs, "Implement a sliding window: summarise old turns, keep last N verbatim")
	}
	if checks[4] {
		evidence = append(evidence, "System prompts audited for token efficiency")
	} else {
		gaps = append(gaps, "System prompts never audited for wasted tokens")
		recs = append(recs, "Run a quarterly token efficiency audit on system prompts")
	}
	return FactorAssessment{FactorNumber: 6, FactorName: factorNames[6], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor7(cfg AgentConfig) FactorAssessment {
	inputLayers := cfg.getInt("input_guardrail_layer_count")
	outputLayers := cfg.getInt("output_guardrail_layer_count")
	checks := []bool{
		inputLayers >= 3,
		outputLayers >= 3,
		cfg.getBool("safety_filters_on_both_input_and_output"),
		cfg.getBool("has_prompt_injection_detection"),
		cfg.getBool("has_pii_detection_and_redaction"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, fmt.Sprintf("%d independent input guardrail layers", inputLayers))
	} else {
		gaps = append(gaps, fmt.Sprintf("Only %d input guardrail layer(s) — need ≥3", inputLayers))
		recs = append(recs, "Add: rate limiter → validator → PII detector → content filter → injection detector")
	}
	if checks[1] {
		evidence = append(evidence, fmt.Sprintf("%d independent output guardrail layers", outputLayers))
	} else {
		gaps = append(gaps, fmt.Sprintf("Only %d output guardrail layer(s) — need ≥3", outputLayers))
		recs = append(recs, "Add: schema validator → PII check → safety filter → leakage detector")
	}
	if checks[2] {
		evidence = append(evidence, "Safety filters applied to both input and output")
	} else {
		gaps = append(gaps, "Safety filter not applied symmetrically")
		recs = append(recs, "Apply every safety check to both user input and model response")
	}
	if checks[3] {
		evidence = append(evidence, "Prompt injection detection in place")
	} else {
		gaps = append(gaps, "No prompt injection detection")
		recs = append(recs, "Add an injection detector that scans for role-override attempts")
	}
	if checks[4] {
		evidence = append(evidence, "PII detection and redaction applied")
	} else {
		gaps = append(gaps, "PII may flow into LLM context or be exposed in responses")
		recs = append(recs, "Add PII detection (names, emails, cards) on input and output")
	}
	return FactorAssessment{FactorNumber: 7, FactorName: factorNames[7], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor8(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("has_fallback_llm_provider"),
		cfg.getBool("has_fallback_for_vector_db"),
		cfg.getBool("has_static_response_for_complete_failure"),
		cfg.getBool("degradation_events_logged"),
		cfg.getBool("degradation_regularly_chaos_tested"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Fallback LLM provider configured")
	} else {
		gaps = append(gaps, "No LLM fallback — primary provider outage causes complete failure")
		recs = append(recs, "Add a fallback provider chain: primary → secondary → cheaper model → static")
	}
	if checks[1] {
		evidence = append(evidence, "Vector database fallback implemented")
	} else {
		gaps = append(gaps, "No fallback when vector database is unavailable")
		recs = append(recs, "Continue without RAG when vector DB is unavailable")
	}
	if checks[2] {
		evidence = append(evidence, "Static fallback response for complete system failure")
	} else {
		gaps = append(gaps, "Complete system failure exposes raw errors to users")
		recs = append(recs, "Add a final error handler that returns a polite message")
	}
	if checks[3] {
		evidence = append(evidence, "Degradation events logged for postmortem analysis")
	} else {
		gaps = append(gaps, "Degradation events are silent")
		recs = append(recs, "Log every degradation event with level, reason, and fallback taken")
	}
	if checks[4] {
		evidence = append(evidence, "Degradation paths regularly tested via chaos engineering")
	} else {
		gaps = append(gaps, "Fallback paths untested — may not work when needed")
		recs = append(recs, "Run monthly chaos tests: kill primary LLM, kill vector DB")
	}
	return FactorAssessment{FactorNumber: 8, FactorName: factorNames[8], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor9(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("has_request_tracing_with_trace_ids"),
		cfg.getBool("traces_exported_to_centralized_system"),
		cfg.getBool("key_metrics_tracked_latency_tokens_cost_errors"),
		cfg.getBool("has_dashboards_for_metrics"),
		cfg.getBool("has_alerts_for_metric_degradation"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Unique trace IDs generated for every request")
	} else {
		gaps = append(gaps, "No request tracing — impossible to reconstruct a given request")
		recs = append(recs, "Generate a UUID trace ID at request entry and propagate it everywhere")
	}
	if checks[1] {
		evidence = append(evidence, "Traces exported to a centralized system")
	} else {
		gaps = append(gaps, "Traces only in local logs — no centralized visibility")
		recs = append(recs, "Export traces via OpenTelemetry to a centralized backend")
	}
	if checks[2] {
		evidence = append(evidence, "Latency, token usage, cost, and error rate tracked")
	} else {
		gaps = append(gaps, "Key metrics not consistently tracked")
		recs = append(recs, "Emit structured metrics: p50/p95 latency, tokens, cost, error_rate per request")
	}
	if checks[3] {
		evidence = append(evidence, "Operational dashboards show real-time metrics")
	} else {
		gaps = append(gaps, "No dashboards — metrics not visible to the team")
		recs = append(recs, "Create a Grafana or DataDog dashboard with the 5 key agent metrics")
	}
	if checks[4] {
		evidence = append(evidence, "Alerts fire when metrics degrade")
	} else {
		gaps = append(gaps, "No metric alerts — degradation discovered by users, not ops")
		recs = append(recs, "Set up alerts: error rate >1%, p95 latency >5s, cost >2x baseline")
	}
	return FactorAssessment{FactorNumber: 9, FactorName: factorNames[9], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor10(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("has_approval_policy_defined"),
		cfg.getBool("high_stakes_actions_flagged_for_approval"),
		cfg.getBool("has_reviewer_interface"),
		cfg.getBool("has_timeout_handling_for_approvals"),
		cfg.getBool("approval_decisions_logged_for_audit"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Explicit approval policy defines which actions require review")
	} else {
		gaps = append(gaps, "No approval policy — every action or no action gets reviewed")
		recs = append(recs, "Define an ApprovalPolicy struct with action types, thresholds, and risk levels")
	}
	if checks[1] {
		evidence = append(evidence, "High-stakes actions flagged for approval")
	} else {
		gaps = append(gaps, "High-stakes actions execute without human oversight")
		recs = append(recs, "Categorise actions by risk: auto-execute LOW, review MEDIUM/HIGH/CRITICAL")
	}
	if checks[2] {
		evidence = append(evidence, "Reviewer interface exists")
	} else {
		gaps = append(gaps, "No reviewer interface")
		recs = append(recs, "Build a Slack/email/web UI for reviewers with approve/reject/edit options")
	}
	if checks[3] {
		evidence = append(evidence, "Approval requests time out safely")
	} else {
		gaps = append(gaps, "Approval requests can block indefinitely")
		recs = append(recs, "Use context.WithTimeout for approval requests; default to rejection on timeout")
	}
	if checks[4] {
		evidence = append(evidence, "Approval decisions logged with reviewer, reason, and timestamp")
	} else {
		gaps = append(gaps, "No audit trail for approval decisions")
		recs = append(recs, "Log every approval decision to a tamper-evident audit log")
	}
	return FactorAssessment{FactorNumber: 10, FactorName: factorNames[10], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor11(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("test_set_has_50_plus_queries"),
		cfg.getBool("evaluations_run_on_every_deployment"),
		cfg.getBool("has_regression_detection_baseline_comparison"),
		cfg.getBool("safety_red_team_run_regularly"),
		cfg.getBool("evaluation_results_block_deployment_on_regression"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Test set of 50+ queries covering all intent categories")
	} else {
		gaps = append(gaps, "No evaluation test set (or fewer than 50 queries)")
		recs = append(recs, "Build a test set of 50+ queries: happy path, edge cases, adversarial")
	}
	if checks[1] {
		evidence = append(evidence, "Full evaluation runs on every deployment")
	} else {
		gaps = append(gaps, "Evaluations are manual and infrequent")
		recs = append(recs, "Add evaluation step to CI/CD pipeline")
	}
	if checks[2] {
		evidence = append(evidence, "Regression detection compares each run to a stored baseline")
	} else {
		gaps = append(gaps, "No regression detection")
		recs = append(recs, "Store eval baseline in JSON; alert when any metric drops >5% from baseline")
	}
	if checks[3] {
		evidence = append(evidence, "Safety red-team evaluations run on every significant change")
	} else {
		gaps = append(gaps, "No safety red-team evaluations")
		recs = append(recs, "Run adversarial safety prompts on every significant change or weekly")
	}
	if checks[4] {
		evidence = append(evidence, "Deployment blocked when evaluation detects significant regression")
	} else {
		gaps = append(gaps, "Regressions reported but don't block deployment")
		recs = append(recs, "Make evaluation results a deployment gate: fail CI on critical regressions")
	}
	return FactorAssessment{FactorNumber: 11, FactorName: factorNames[11], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

func assessFactor12(cfg AgentConfig) FactorAssessment {
	checks := []bool{
		cfg.getBool("same_guardrail_config_in_dev_and_prod"),
		cfg.getBool("production_model_tested_before_deployment"),
		cfg.getBool("has_staging_environment_matching_production"),
		cfg.getBool("knowledge_base_structures_consistent_across_envs"),
		cfg.getBool("environment_differences_documented_and_intentional"),
	}
	passed := countTrue(checks)
	score := checksToScore(passed, 5)
	evidence, gaps, recs := []string{}, []string{}, []string{}

	if checks[0] {
		evidence = append(evidence, "Guardrail configuration identical in dev and production")
	} else {
		gaps = append(gaps, "Guardrails differ between dev and prod")
		recs = append(recs, "Load guardrail config from a shared file that all environments use")
	}
	if checks[1] {
		evidence = append(evidence, "Production model tested before every deployment")
	} else {
		gaps = append(gaps, "Dev uses a cheaper model — production model behaviour not verified until deploy")
		recs = append(recs, "Add a pre-deployment test step that runs the eval suite with the prod model")
	}
	if checks[2] {
		evidence = append(evidence, "Staging environment mirrors production configuration")
	} else {
		gaps = append(gaps, "No staging environment")
		recs = append(recs, "Maintain a staging environment with the same config as prod")
	}
	if checks[3] {
		evidence = append(evidence, "Knowledge base schemas consistent across environments")
	} else {
		gaps = append(gaps, "Knowledge base structure differs across environments")
		recs = append(recs, "Keep knowledge base index configs identical across all environments")
	}
	if checks[4] {
		evidence = append(evidence, "Intentional environment differences documented")
	} else {
		gaps = append(gaps, "Environment differences undocumented and may be accidental")
		recs = append(recs, "Document every deliberate difference in an ENVIRONMENTS.md file")
	}
	return FactorAssessment{FactorNumber: 12, FactorName: factorNames[12], Score: score, Status: scoreToStatus(score), Evidence: evidence, Gaps: gaps, Recommendations: recs}
}

// countTrue counts true values in a bool slice.
func countTrue(checks []bool) int {
	n := 0
	for _, c := range checks {
		if c {
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// Maturity level calculation
// ---------------------------------------------------------------------------

func computeMaturity(factors []FactorAssessment, total int) (string, int) {
	scores := make(map[int]int)
	for _, f := range factors {
		scores[f.FactorNumber] = f.Score
	}
	allAtLeast := func(nums []int, min int) bool {
		for _, n := range nums {
			if scores[n] < min {
				return false
			}
		}
		return true
	}
	all12 := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	if total == 60 && allAtLeast(all12, 4) {
		return "Elite", 5
	}
	if total >= 49 && allAtLeast([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3) {
		return "Production", 4
	}
	if total >= 37 && allAtLeast([]int{1, 2, 3, 4, 5, 6, 8, 9}, 3) {
		return "Staging", 3
	}
	if total >= 25 && allAtLeast([]int{1, 2, 5, 9}, 3) {
		return "Development", 2
	}
	return "Prototype", 1
}

// ---------------------------------------------------------------------------
// TwelveFactorAssessor
// ---------------------------------------------------------------------------

// TwelveFactorAssessor evaluates an agent configuration against all 12 factors.
type TwelveFactorAssessor struct{}

// NewTwelveFactorAssessor creates a new TwelveFactorAssessor.
func NewTwelveFactorAssessor() *TwelveFactorAssessor {
	return &TwelveFactorAssessor{}
}

// Assess evaluates all 12 factors and returns a TwelveFactorReport.
func (a *TwelveFactorAssessor) Assess(cfg AgentConfig) TwelveFactorReport {
	assessors := []func(AgentConfig) FactorAssessment{
		assessFactor1, assessFactor2, assessFactor3, assessFactor4,
		assessFactor5, assessFactor6, assessFactor7, assessFactor8,
		assessFactor9, assessFactor10, assessFactor11, assessFactor12,
	}
	factors := make([]FactorAssessment, len(assessors))
	total := 0
	for i, fn := range assessors {
		factors[i] = fn(cfg)
		total += factors[i].Score
	}
	levelName, levelNum := computeMaturity(factors, total)

	var criticalGaps []string
	for _, f := range factors {
		if f.Score <= 2 {
			gap := fmt.Sprintf("Factor %d (%s) scored %d/5", f.FactorNumber, f.FactorName, f.Score)
			if len(f.Gaps) > 0 {
				gap += " — " + f.Gaps[0]
			}
			criticalGaps = append(criticalGaps, gap)
		}
	}

	sorted := make([]FactorAssessment, len(factors))
	copy(sorted, factors)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Score != sorted[j].Score {
			return sorted[i].Score < sorted[j].Score
		}
		return sorted[i].FactorNumber < sorted[j].FactorNumber
	})
	var priorities []string
	for _, f := range sorted {
		if f.Score < 5 && len(f.Recommendations) > 0 {
			priorities = append(priorities, fmt.Sprintf(
				"Improve Factor %d (%s): %s", f.FactorNumber, f.FactorName, f.Recommendations[0],
			))
		}
		if len(priorities) >= 7 {
			break
		}
	}

	return TwelveFactorReport{
		OverallScore:          total,
		MaturityLevel:         levelName,
		MaturityLevelNumber:   levelNum,
		Factors:               factors,
		CriticalGaps:          criticalGaps,
		ImprovementPriorities: priorities,
		AssessedAt:            today(),
	}
}

// AssessFactor evaluates a single factor by number (1-12).
func (a *TwelveFactorAssessor) AssessFactor(factorNumber int, cfg AgentConfig) (FactorAssessment, error) {
	assessors := map[int]func(AgentConfig) FactorAssessment{
		1: assessFactor1, 2: assessFactor2, 3: assessFactor3, 4: assessFactor4,
		5: assessFactor5, 6: assessFactor6, 7: assessFactor7, 8: assessFactor8,
		9: assessFactor9, 10: assessFactor10, 11: assessFactor11, 12: assessFactor12,
	}
	fn, ok := assessors[factorNumber]
	if !ok {
		return FactorAssessment{}, fmt.Errorf("factor number must be 1-12, got %d", factorNumber)
	}
	return fn(cfg), nil
}

// GenerateRecommendations returns a flat, ordered list of actionable recommendations.
func (a *TwelveFactorAssessor) GenerateRecommendations(report TwelveFactorReport) []string {
	var recs []string
	for _, f := range report.Factors {
		if f.Score <= 2 {
			for _, r := range f.Recommendations {
				recs = append(recs, fmt.Sprintf("[CRITICAL] Factor %d — %s", f.FactorNumber, r))
			}
		}
	}
	for _, f := range report.Factors {
		if f.Score == 3 {
			for _, r := range f.Recommendations {
				recs = append(recs, fmt.Sprintf("[IMPROVE] Factor %d — %s", f.FactorNumber, r))
			}
		}
	}
	for _, f := range report.Factors {
		if f.Score == 4 {
			for _, r := range f.Recommendations {
				recs = append(recs, fmt.Sprintf("[POLISH] Factor %d — %s", f.FactorNumber, r))
			}
		}
	}
	return recs
}

// CompareToBaseline compares current report to a baseline.
func (a *TwelveFactorAssessor) CompareToBaseline(current, baseline TwelveFactorReport) ComparisonReport {
	baseScores := make(map[int]int)
	for _, f := range baseline.Factors {
		baseScores[f.FactorNumber] = f.Score
	}
	currScores := make(map[int]int)
	for _, f := range current.Factors {
		currScores[f.FactorNumber] = f.Score
	}

	var improved, regressed []FactorDelta
	var unchanged []FactorUnchanged
	for n := 1; n <= 12; n++ {
		name := factorNames[n]
		old := baseScores[n]
		nxt := currScores[n]
		switch {
		case nxt > old:
			improved = append(improved, FactorDelta{n, name, old, nxt})
		case nxt < old:
			regressed = append(regressed, FactorDelta{n, name, old, nxt})
		default:
			unchanged = append(unchanged, FactorUnchanged{n, name, old})
		}
	}
	return ComparisonReport{
		BaselineScore:    baseline.OverallScore,
		CurrentScore:     current.OverallScore,
		ScoreDelta:       current.OverallScore - baseline.OverallScore,
		BaselineLevel:    baseline.MaturityLevel,
		CurrentLevel:     current.MaturityLevel,
		ImprovedFactors:  improved,
		RegressedFactors: regressed,
		UnchangedFactors: unchanged,
	}
}

// ExportReport exports the report as "markdown" or "html".
func (a *TwelveFactorAssessor) ExportReport(report TwelveFactorReport, format string) (string, error) {
	switch format {
	case "markdown":
		return GenerateMarkdownReport(report), nil
	case "html":
		return GenerateHTMLReport(report)
	default:
		return "", fmt.Errorf("unknown format %q; use 'markdown' or 'html'", format)
	}
}

// ---------------------------------------------------------------------------
// ImprovementTracker
// ---------------------------------------------------------------------------

// ImprovementTracker tracks assessments over time and generates roadmaps.
type ImprovementTracker struct {
	history  []TwelveFactorReport
	assessor *TwelveFactorAssessor
}

// NewImprovementTracker creates a new ImprovementTracker.
func NewImprovementTracker() *ImprovementTracker {
	return &ImprovementTracker{assessor: NewTwelveFactorAssessor()}
}

// SaveBaseline appends a report to the history.
func (t *ImprovementTracker) SaveBaseline(report TwelveFactorReport) {
	t.history = append(t.history, report)
}

// CompareToBaseline compares current to the most recent saved baseline.
func (t *ImprovementTracker) CompareToBaseline(current TwelveFactorReport) (*ComparisonReport, bool) {
	if len(t.history) == 0 {
		return nil, false
	}
	c := t.assessor.CompareToBaseline(current, t.history[len(t.history)-1])
	return &c, true
}

// TrackOverTime returns a trend report across all saved assessments.
func (t *ImprovementTracker) TrackOverTime() (TrendReport, error) {
	if len(t.history) == 0 {
		return TrendReport{}, fmt.Errorf("no assessments recorded yet")
	}
	assessments := make([]AssessmentSummary, len(t.history))
	for i, r := range t.history {
		assessments[i] = AssessmentSummary{r.AssessedAt, r.OverallScore, r.MaturityLevel}
	}
	first := t.history[0].OverallScore
	latest := t.history[len(t.history)-1].OverallScore
	delta := latest - first
	denom := len(t.history) - 1
	if denom < 1 {
		denom = 1
	}
	avg := float64(delta) / float64(denom)
	return TrendReport{assessments, first, latest, delta, avg}, nil
}

// GenerateRoadmap returns ordered steps to reach the target maturity level.
func (t *ImprovementTracker) GenerateRoadmap(current TwelveFactorReport, targetLevel string) ([]string, error) {
	levelMap := map[string]int{
		"prototype": 1, "development": 2, "staging": 3, "production": 4, "elite": 5,
	}
	targetNum, ok := levelMap[strings.ToLower(targetLevel)]
	if !ok {
		return nil, fmt.Errorf("unknown target level: %q", targetLevel)
	}
	if current.MaturityLevelNumber >= targetNum {
		return []string{fmt.Sprintf("Already at or above %s level.", targetLevel)}, nil
	}

	required := map[int][]int{
		2: {1, 2, 5, 9},
		3: {1, 2, 3, 4, 5, 6, 8, 9},
		4: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
		5: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
	}
	scoreByFactor := make(map[int]int)
	for _, f := range current.Factors {
		scoreByFactor[f.FactorNumber] = f.Score
	}
	var steps []string
	for level := current.MaturityLevelNumber + 1; level <= targetNum; level++ {
		for _, fn := range required[level] {
			if scoreByFactor[fn] < 3 {
				fa := findFactor(current.Factors, fn)
				rec := "See assessment gaps"
				if fa != nil && len(fa.Recommendations) > 0 {
					rec = fa.Recommendations[0]
				}
				steps = append(steps, fmt.Sprintf(
					"[Level %d] Bring Factor %d (%s) to ≥3: %s",
					level, fn, factorNames[fn], rec,
				))
			}
		}
	}
	if targetNum == 5 {
		for _, f := range current.Factors {
			if f.Score == 3 && len(f.Recommendations) > 0 {
				steps = append(steps, fmt.Sprintf(
					"[Elite] Raise Factor %d (%s) from 3→4: %s",
					f.FactorNumber, f.FactorName, f.Recommendations[0],
				))
			}
		}
	}
	if len(steps) == 0 {
		return []string{"No additional steps required — you are on track."}, nil
	}
	return steps, nil
}

func findFactor(factors []FactorAssessment, num int) *FactorAssessment {
	for i := range factors {
		if factors[i].FactorNumber == num {
			return &factors[i]
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Report generators
// ---------------------------------------------------------------------------

var statusEmoji = map[string]string{
	"excellent":        "✅ Excellent",
	"good":             "✅ Good",
	"needs_improvement": "⚠️  Needs Improvement",
	"critical":         "❌ Critical",
}

// GenerateMarkdownReport produces a markdown assessment report.
func GenerateMarkdownReport(report TwelveFactorReport) string {
	var sb strings.Builder
	sb.WriteString("# 12-Factor Agent Assessment Report\n\n")
	sb.WriteString(fmt.Sprintf("**Date:** %s\n", report.AssessedAt))
	sb.WriteString(fmt.Sprintf("**Overall Score:** %d/60\n", report.OverallScore))
	sb.WriteString(fmt.Sprintf("**Maturity Level:** %s (Level %d)\n\n", report.MaturityLevel, report.MaturityLevelNumber))
	sb.WriteString("## Factor Scores\n\n")
	sb.WriteString("| # | Factor | Score | Status |\n")
	sb.WriteString("|---|--------|-------|--------|\n")
	for _, f := range report.Factors {
		status := statusEmoji[f.Status]
		if status == "" {
			status = f.Status
		}
		sb.WriteString(fmt.Sprintf("| %s | %s | %d/5 | %s |\n", romanNumerals[f.FactorNumber], f.FactorName, f.Score, status))
	}
	if len(report.CriticalGaps) > 0 {
		sb.WriteString("\n## Critical Gaps (Must Fix)\n\n")
		n := 1
		for _, f := range report.Factors {
			if f.Score > 2 {
				continue
			}
			sb.WriteString(fmt.Sprintf("%d. **Factor %s — %s (Score: %d/5)**\n", n, romanNumerals[f.FactorNumber], f.FactorName, f.Score))
			for _, g := range f.Gaps {
				sb.WriteString(fmt.Sprintf("   - Missing: %s\n", g))
			}
			for _, r := range f.Recommendations {
				sb.WriteString(fmt.Sprintf("   - Recommendation: %s\n", r))
			}
			sb.WriteString("\n")
			n++
		}
	}
	if len(report.ImprovementPriorities) > 0 {
		sb.WriteString("## Improvement Priorities\n\n")
		for i, p := range report.ImprovementPriorities {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, p))
		}
	}
	return sb.String()
}

// htmlTmpl is the Go html/template for the HTML report.
var htmlTmpl = template.Must(template.New("report").Funcs(template.FuncMap{
	"roman": func(n int) string { return romanNumerals[n] },
	"pct":   func(score int) int { return score * 100 / 60 },
	"barWidth": func(score int) int { return score * 20 },
	"scoreColor": func(score int) string {
		switch score {
		case 5:
			return "#22c55e"
		case 4:
			return "#86efac"
		case 3:
			return "#eab308"
		case 2:
			return "#ef4444"
		default:
			return "#b91c1c"
		}
	},
	"statusEmoji": func(status string) string {
		return statusEmoji[status]
	},
}).Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>12-Factor Agent Assessment</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1e293b}
  .scorecard{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:1.5rem;margin-bottom:2rem}
  .score-big{font-size:2.5rem;font-weight:700}
  .progress-bar{background:#e2e8f0;border-radius:99px;height:12px;margin:.5rem 0}
  table{border-collapse:collapse;width:100%}
  th,td{text-align:left;padding:.5rem .75rem;border-bottom:1px solid #e2e8f0}
  th{background:#f1f5f9;font-weight:600}
</style>
</head>
<body>
<h1>12-Factor Agent Assessment Report</h1>
<div class="scorecard">
  <div class="score-big">{{.OverallScore}}/60</div>
  <div>{{.MaturityLevel}} (Level {{.MaturityLevelNumber}}) — {{.AssessedAt}}</div>
  <div class="progress-bar">
    <div style="background:#22c55e;border-radius:99px;height:12px;width:{{pct .OverallScore}}%"></div>
  </div>
  <small>{{pct .OverallScore}}% toward Elite</small>
</div>
<h2>Factor Scores</h2>
<table>
  <thead><tr><th>#</th><th>Factor</th><th>Score</th><th>Status</th></tr></thead>
  <tbody>
  {{range .Factors}}
  <tr>
    <td>{{roman .FactorNumber}}</td>
    <td>{{.FactorName}}</td>
    <td>
      <div style="background:{{scoreColor .Score}};width:{{barWidth .Score}}px;height:16px;border-radius:4px;display:inline-block"></div>
      <strong style="color:{{scoreColor .Score}}">{{.Score}}/5</strong>
    </td>
    <td>{{statusEmoji .Status}}</td>
  </tr>
  {{end}}
  </tbody>
</table>
<h2>Critical Gaps</h2>
<ul>
{{range .CriticalGaps}}<li>{{.}}</li>{{else}}<li>No critical gaps!</li>{{end}}
</ul>
<h2>Improvement Priorities</h2>
<ol>
{{range .ImprovementPriorities}}<li>{{.}}</li>{{else}}<li>All factors at maximum score.</li>{{end}}
</ol>
</body>
</html>`))

// GenerateHTMLReport produces an HTML assessment report using html/template.
func GenerateHTMLReport(report TwelveFactorReport) (string, error) {
	var sb strings.Builder
	if err := htmlTmpl.Execute(&sb, report); err != nil {
		return "", fmt.Errorf("generating HTML report: %w", err)
	}
	return sb.String(), nil
}

// ---------------------------------------------------------------------------
// Demo (main)
// ---------------------------------------------------------------------------

func RunTwelveFactorAssessorDemo() {
	assessor := NewTwelveFactorAssessor()
	tracker := NewImprovementTracker()

	baselineCfg := AgentConfig{
		"prompts_in_version_control":                     true,
		"prompts_semantically_versioned":                 true,
		"prompts_code_reviewed":                          false,
		"prompts_have_change_log":                        false,
		"prompts_independently_rollbackable":             false,
		"has_conversation_state_class":                   true,
		"state_survives_truncation":                      true,
		"state_is_serializable":                          true,
		"state_persisted_across_sessions":                false,
		"state_transitions_logged":                       false,
		"has_llm_provider_interface":                     false,
		"num_providers_supported":                        1,
		"provider_configurable_via_env":                  false,
		"provider_specific_features_abstracted":          false,
		"has_provider_fallback_chain":                    false,
		"has_per_request_token_budget":                   false,
		"has_per_request_cost_budget":                    false,
		"token_usage_tracked_and_logged":                 false,
		"cost_alerts_configured":                         false,
		"budget_enforcement_blocks_excess":               false,
		"all_llm_outputs_schema_validated":               true,
		"schema_definitions_in_version_control":          true,
		"has_parse_validate_retry_pattern":               true,
		"schemas_consistent_across_interactions":         true,
		"schema_violations_logged_and_alerted":           true,
		"has_explicit_token_allocation_per_zone":         true,
		"context_consumption_measured_per_request":       true,
		"has_automatic_compression_when_over_budget":     false,
		"has_sliding_window_for_history":                 false,
		"system_prompts_audited_for_efficiency":          false,
		"input_guardrail_layer_count":                    2,
		"output_guardrail_layer_count":                   2,
		"safety_filters_on_both_input_and_output":        true,
		"has_prompt_injection_detection":                 false,
		"has_pii_detection_and_redaction":                false,
		"has_fallback_llm_provider":                      false,
		"has_fallback_for_vector_db":                     false,
		"has_static_response_for_complete_failure":       true,
		"degradation_events_logged":                      true,
		"degradation_regularly_chaos_tested":             false,
		"has_request_tracing_with_trace_ids":             true,
		"traces_exported_to_centralized_system":          false,
		"key_metrics_tracked_latency_tokens_cost_errors": true,
		"has_dashboards_for_metrics":                     false,
		"has_alerts_for_metric_degradation":              false,
		"has_approval_policy_defined":                    false,
		"high_stakes_actions_flagged_for_approval":       false,
		"has_reviewer_interface":                         false,
		"has_timeout_handling_for_approvals":             false,
		"approval_decisions_logged_for_audit":            false,
		"test_set_has_50_plus_queries":                   true,
		"evaluations_run_on_every_deployment":            false,
		"has_regression_detection_baseline_comparison":   false,
		"safety_red_team_run_regularly":                  false,
		"evaluation_results_block_deployment_on_regression": false,
		"same_guardrail_config_in_dev_and_prod":              false,
		"production_model_tested_before_deployment":          false,
		"has_staging_environment_matching_production":        true,
		"knowledge_base_structures_consistent_across_envs":   true,
		"environment_differences_documented_and_intentional": false,
	}

	baseline := assessor.Assess(baselineCfg)
	fmt.Printf("=== BASELINE: %s (Level %d) Score: %d/60 ===\n",
		baseline.MaturityLevel, baseline.MaturityLevelNumber, baseline.OverallScore)

	md, _ := assessor.ExportReport(baseline, "markdown")
	fmt.Println(md)
	tracker.SaveBaseline(baseline)

	// Improved config
	improvedCfg := make(AgentConfig)
	for k, v := range baselineCfg {
		improvedCfg[k] = v
	}
	improvedCfg["has_llm_provider_interface"] = true
	improvedCfg["num_providers_supported"] = 2
	improvedCfg["provider_configurable_via_env"] = true
	improvedCfg["has_provider_fallback_chain"] = true
	improvedCfg["has_per_request_token_budget"] = true
	improvedCfg["has_per_request_cost_budget"] = true
	improvedCfg["token_usage_tracked_and_logged"] = true
	improvedCfg["cost_alerts_configured"] = true
	improvedCfg["budget_enforcement_blocks_excess"] = true
	improvedCfg["has_approval_policy_defined"] = true
	improvedCfg["high_stakes_actions_flagged_for_approval"] = true
	improvedCfg["has_reviewer_interface"] = true
	improvedCfg["has_timeout_handling_for_approvals"] = true
	improvedCfg["approval_decisions_logged_for_audit"] = true
	improvedCfg["evaluations_run_on_every_deployment"] = true
	improvedCfg["has_regression_detection_baseline_comparison"] = true

	improved := assessor.Assess(improvedCfg)
	fmt.Printf("\n=== IMPROVED: %s (Level %d) Score: %d/60 ===\n",
		improved.MaturityLevel, improved.MaturityLevelNumber, improved.OverallScore)

	cmp := assessor.CompareToBaseline(improved, baseline)
	fmt.Printf("Score delta: %+d | Level: %s → %s\n", cmp.ScoreDelta, cmp.BaselineLevel, cmp.CurrentLevel)
	for _, d := range cmp.ImprovedFactors {
		fmt.Printf("  Factor %d (%s): %d → %d (+%d)\n", d.FactorNumber, d.FactorName, d.OldScore, d.NewScore, d.NewScore-d.OldScore)
	}

	tracker.SaveBaseline(improved)
	trend, _ := tracker.TrackOverTime()
	fmt.Printf("\nTotal improvement: +%d points\n", trend.TotalImprovement)

	roadmap, _ := tracker.GenerateRoadmap(improved, "elite")
	fmt.Println("\n=== ROADMAP TO ELITE ===")
	for i, step := range roadmap {
		fmt.Printf("  %d. %s\n", i+1, step)
	}

	// Export HTML
	html, err := assessor.ExportReport(improved, "html")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	if err := os.WriteFile("/tmp/twelve_factor_report.html", []byte(html), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	fmt.Println("\nHTML report written to /tmp/twelve_factor_report.html")
}
