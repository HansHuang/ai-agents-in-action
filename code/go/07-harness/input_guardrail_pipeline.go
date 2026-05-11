// Package main implements a six-layer input guardrail pipeline for AI agents.
//
// Layers (cheapest → most expensive):
//
//  1. RateLimiter          — per-user sliding-window rate limits
//  2. StructuralValidator  — length, token count, binary, repetition checks
//  3. PIIDetector          — regex + Luhn-validated PII redaction
//  4. ContentPolicyEnforcer— blocked / warned content categories
//  5. InjectionDetector    — prompt injection pattern matching
//  6. InputSanitizer       — Unicode, whitespace, control-char normalisation
//
// See: docs/07-harness-engineering/02-input-guardrails-and-validation.md
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Config holds all configurable parameters for the pipeline.
type Config struct {
	RateLimitRPM        int
	RateLimitRPH        int
	RateLimitRPD        int
	MinInputLength      int
	MaxInputLength      int
	MaxInputTokens      int
	UseLLMContentReview bool
	SanitiserHardCap    int
}

// Option is a functional option for configuring the pipeline.
type Option func(*Config)

// WithRateLimitRPM sets the per-minute rate limit.
func WithRateLimitRPM(n int) Option { return func(c *Config) { c.RateLimitRPM = n } }

// WithRateLimitRPH sets the per-hour rate limit.
func WithRateLimitRPH(n int) Option { return func(c *Config) { c.RateLimitRPH = n } }

// WithRateLimitRPD sets the per-day rate limit.
func WithRateLimitRPD(n int) Option { return func(c *Config) { c.RateLimitRPD = n } }

// WithMaxInputLength overrides the maximum allowed input length in characters.
func WithMaxInputLength(n int) Option { return func(c *Config) { c.MaxInputLength = n } }

func defaultConfig() Config {
	return Config{
		RateLimitRPM:        30,
		RateLimitRPH:        500,
		RateLimitRPD:        5_000,
		MinInputLength:      1,
		MaxInputLength:      100_000,
		MaxInputTokens:      75_000,
		UseLLMContentReview: false,
		SanitiserHardCap:    100_000,
	}
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

// RateLimitResult is returned by RateLimiter.Check.
type RateLimitResult struct {
	Allowed    bool
	Reason     string
	RetryAfter float64 // seconds
}

// ValidationResult is returned by StructuralValidator.Validate.
type ValidationResult struct {
	Passed bool
	Reason string
	Checks []string
}

// PIIDetection describes a single PII occurrence.
type PIIDetection struct {
	Type  string
	Value string
	Start int
	End   int
}

// PIIResult summarises the PII checks on a single input.
type PIIResult struct {
	Detections []PIIDetection
	Redacted   bool
}

// PolicyViolation is a single content-policy match.
type PolicyViolation struct {
	Category       string
	Severity       string // "block" or "warn"
	MatchedPattern string
	Snippet        string
}

// PolicyResult is returned by ContentPolicyEnforcer.Enforce.
type PolicyResult struct {
	Passed     bool
	Violations []PolicyViolation
	Warnings   []PolicyViolation
	Action     string // "allow", "warn", "block"
	Message    string
}

// InjectionDetection is a single injection pattern match.
type InjectionDetection struct {
	Type    string // "injection_pattern", "delimiter_abuse", "structural_anomaly"
	Pattern string
	Snippet string
}

// InjectionResult is returned by InjectionDetector.Detect.
type InjectionResult struct {
	RiskLevel         string // "none", "low", "medium", "high", "critical"
	Detections        []InjectionDetection
	RecommendedAction string // "allow", "warn", "sanitize", "block"
}

// GuardrailResult is the final result of the pipeline.
type GuardrailResult struct {
	OriginalInput   string
	CleanedInput    string
	Passed          bool
	RejectionReason string
	RejectionLayer  string
	Checks          map[string]any
}

// ---------------------------------------------------------------------------
// Layer 1 — Rate Limiter
// ---------------------------------------------------------------------------

// RateLimiter enforces per-user sliding-window rate limits.
// It is safe for concurrent use.
type RateLimiter struct {
	rpm     int
	rph     int
	rpd     int
	mu      sync.Mutex
	buckets map[string][]time.Time
}

// NewRateLimiter creates a RateLimiter with the given per-minute, per-hour,
// and per-day limits.
func NewRateLimiter(rpm, rph, rpd int) *RateLimiter {
	return &RateLimiter{
		rpm:     rpm,
		rph:     rph,
		rpd:     rpd,
		buckets: make(map[string][]time.Time),
	}
}

// Check returns whether userID is within all rate-limit windows.
func (r *RateLimiter) Check(userID string) RateLimitResult {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	r.cleanup(userID, now)

	requests := r.buckets[userID]

	var lastMinute, lastHour int
	for _, t := range requests {
		since := now.Sub(t).Seconds()
		if since < 60 {
			lastMinute++
		}
		if since < 3600 {
			lastHour++
		}
	}
	lastDay := len(requests)

	if lastMinute >= r.rpm {
		return RateLimitResult{
			Allowed:    false,
			Reason:     fmt.Sprintf("Rate limit exceeded: %d requests per minute", r.rpm),
			RetryAfter: 60,
		}
	}
	if lastHour >= r.rph {
		var oldestInHour time.Time
		for _, t := range requests {
			if now.Sub(t).Seconds() < 3600 {
				if oldestInHour.IsZero() || t.Before(oldestInHour) {
					oldestInHour = t
				}
			}
		}
		return RateLimitResult{
			Allowed:    false,
			Reason:     fmt.Sprintf("Rate limit exceeded: %d requests per hour", r.rph),
			RetryAfter: 3600 - now.Sub(oldestInHour).Seconds(),
		}
	}
	if lastDay >= r.rpd {
		oldest := requests[0]
		return RateLimitResult{
			Allowed:    false,
			Reason:     fmt.Sprintf("Daily limit of %d requests reached", r.rpd),
			RetryAfter: 86400 - now.Sub(oldest).Seconds(),
		}
	}

	r.buckets[userID] = append(requests, now)
	return RateLimitResult{Allowed: true}
}

func (r *RateLimiter) cleanup(userID string, now time.Time) {
	requests := r.buckets[userID]
	var kept []time.Time
	for _, t := range requests {
		if now.Sub(t).Seconds() < 86400 {
			kept = append(kept, t)
		}
	}
	r.buckets[userID] = kept
}

// ---------------------------------------------------------------------------
// Layer 2 — Structural Validator
// ---------------------------------------------------------------------------

// StructuralValidator rejects empty, too-long, binary, or repetitive input.
type StructuralValidator struct {
	minLength int
	maxLength int
	maxTokens int
}

// NewStructuralValidator creates a validator with the given limits.
func NewStructuralValidator(minLength, maxLength, maxTokens int) *StructuralValidator {
	return &StructuralValidator{
		minLength: minLength,
		maxLength: maxLength,
		maxTokens: maxTokens,
	}
}

// Validate runs all structural checks and returns a ValidationResult.
func (v *StructuralValidator) Validate(input string) ValidationResult {
	var checks []string

	if strings.TrimSpace(input) == "" {
		return ValidationResult{Passed: false, Reason: "Input is empty or whitespace-only.", Checks: checks}
	}
	checks = append(checks, "not_empty")

	if len(strings.TrimSpace(input)) < v.minLength {
		return ValidationResult{
			Passed: false,
			Reason: fmt.Sprintf("Input too short. Minimum %d character(s).", v.minLength),
			Checks: checks,
		}
	}
	checks = append(checks, "min_length")

	if len(input) > v.maxLength {
		return ValidationResult{
			Passed: false,
			Reason: fmt.Sprintf("Input too long. Maximum %d characters.", v.maxLength),
			Checks: checks,
		}
	}
	checks = append(checks, "max_length")

	estimatedTokens := len(input) / 4
	if estimatedTokens > v.maxTokens {
		return ValidationResult{
			Passed: false,
			Reason: fmt.Sprintf("Input too long. Estimated %d tokens (max %d).", estimatedTokens, v.maxTokens),
			Checks: checks,
		}
	}
	checks = append(checks, "token_count")

	if containsBinary(input) {
		return ValidationResult{Passed: false, Reason: "Input appears to contain binary data.", Checks: checks}
	}
	checks = append(checks, "is_text")

	if isRepetitive(input) {
		return ValidationResult{Passed: false, Reason: "Input contains excessive repetition.", Checks: checks}
	}
	checks = append(checks, "not_repetitive")

	return ValidationResult{Passed: true, Checks: checks}
}

func containsBinary(text string) bool {
	if text == "" {
		return false
	}
	var nonPrintable int
	for _, r := range text {
		if r < 32 && r != '\n' && r != '\r' && r != '\t' {
			nonPrintable++
		}
	}
	return float64(nonPrintable)/float64(len([]rune(text))) > 0.1
}

func isRepetitive(text string) bool {
	runes := []rune(text)
	if len(runes) < 100 {
		return false
	}
	counts := make(map[rune]int)
	for _, r := range runes {
		counts[r]++
	}
	var maxCount int
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	if float64(maxCount)/float64(len(runes)) > 0.9 {
		return true
	}
	half := len(runes) / 2
	if string(runes[:half]) == string(runes[half:half*2]) {
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// Layer 3 — PII Detector
// ---------------------------------------------------------------------------

var igpPiiPatterns = map[string]*regexp.Regexp{
	"credit_card": regexp.MustCompile(`\b(?:\d[ \-]*?){13,16}\b`),
	"ssn":         regexp.MustCompile(`\b\d{3}[ \-]?\d{2}[ \-]?\d{4}\b`),
	"email":       regexp.MustCompile(`\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`),
	"phone":       regexp.MustCompile(`\b\d{3}[.\-]?\d{3}[.\-]?\d{4}\b`),
	"api_key":     regexp.MustCompile(`\b(?:sk-[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z\-_]{35}|AKIA[0-9A-Z]{16})\b`),
	"ip_address":  regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`),
}

// PIIDetector finds and redacts personally identifiable information.
type PIIDetector struct{}

// Detect returns all PII detections found in text.
func (d *PIIDetector) Detect(text string) []PIIDetection {
	var detections []PIIDetection
	for piiType, pattern := range igpPiiPatterns {
		for _, loc := range pattern.FindAllStringIndex(text, -1) {
			value := text[loc[0]:loc[1]]
			if piiType == "credit_card" {
				digits := regexp.MustCompile(`[^0-9]`).ReplaceAllString(value, "")
				if !luhnCheck(digits) {
					continue
				}
			}
			detections = append(detections, PIIDetection{
				Type: piiType, Value: value, Start: loc[0], End: loc[1],
			})
		}
	}
	return detections
}

// Redact replaces PII occurrences with [REDACTED_<TYPE>].
// If detections is nil, Detect is called first.
func (d *PIIDetector) Redact(text string, detections []PIIDetection) (string, []PIIDetection) {
	if detections == nil {
		detections = d.Detect(text)
	}
	// Sort right-to-left to preserve indices
	sorted := make([]PIIDetection, len(detections))
	copy(sorted, detections)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i].Start < sorted[j].Start {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	for _, det := range sorted {
		replacement := "[REDACTED_" + strings.ToUpper(det.Type) + "]"
		text = text[:det.Start] + replacement + text[det.End:]
	}
	return text, detections
}

func luhnCheck(cardNumber string) bool {
	if cardNumber == "" {
		return false
	}
	for _, r := range cardNumber {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	runes := []rune(cardNumber)
	var checksum int
	for i, r := range reverse(runes) {
		d := int(r - '0')
		if i%2 == 1 {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		checksum += d
	}
	return checksum%10 == 0
}

func reverse(r []rune) []rune {
	out := make([]rune, len(r))
	for i, v := range r {
		out[len(r)-1-i] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Layer 4 — Content Policy Enforcer
// ---------------------------------------------------------------------------

var blockedPatterns = map[string][]*regexp.Regexp{
	"self_harm": {
		regexp.MustCompile(`(?i)\b(kill\s+myself|suicide|end\s+my\s+life|want\s+to\s+die)\b`),
	},
	"violence": {
		regexp.MustCompile(`(?i)\b(how\s+to\s+(murder|massacre)|shoot\s+up|bomb\s+(a|the)\s+\w+|terrorist\s+attack)\b`),
	},
	"child_safety": {
		regexp.MustCompile(`(?i)\b(child\s*(porn|abuse|exploitation|sexual))\b`),
		regexp.MustCompile(`(?i)\bcsam\b`),
	},
	"illegal_activity": {
		regexp.MustCompile(`(?i)\b(how\s+to\s+(make|manufacture|synthesize|build)\s+(meth|heroin|fentanyl|bomb|explosive|nerve\s+agent))\b`),
	},
}

var warnPatterns = map[string][]*regexp.Regexp{
	"profanity": {
		regexp.MustCompile(`(?i)\b(damn|hell|shit|fuck|crap|ass|bitch)\b`),
	},
	"aggressive_language": {
		regexp.MustCompile(`(?i)\b(stupid|idiot|useless|worthless|terrible|awful|worst)\b`),
	},
}

// ContentPolicyEnforcer blocks or warns on policy-violating content.
type ContentPolicyEnforcer struct {
	useLLMReview bool
}

// NewContentPolicyEnforcer creates an enforcer.
func NewContentPolicyEnforcer(useLLMReview bool) *ContentPolicyEnforcer {
	return &ContentPolicyEnforcer{useLLMReview: useLLMReview}
}

// Enforce checks content policy and returns a PolicyResult.
func (e *ContentPolicyEnforcer) Enforce(text string) PolicyResult {
	var violations, warnings []PolicyViolation

	for category, patterns := range blockedPatterns {
		for _, pattern := range patterns {
			if pattern.MatchString(text) {
				violations = append(violations, PolicyViolation{
					Category:       category,
					Severity:       "block",
					MatchedPattern: pattern.String(),
					Snippet:        extractContext(text, pattern, 40),
				})
			}
		}
	}

	if len(violations) > 0 {
		return PolicyResult{
			Passed:     false,
			Violations: violations,
			Warnings:   warnings,
			Action:     "block",
			Message:    buildBlockMessage(violations),
		}
	}

	for category, patterns := range warnPatterns {
		for _, pattern := range patterns {
			if pattern.MatchString(text) {
				warnings = append(warnings, PolicyViolation{
					Category:       category,
					Severity:       "warn",
					MatchedPattern: pattern.String(),
					Snippet:        extractContext(text, pattern, 40),
				})
			}
		}
	}

	action := "allow"
	if len(warnings) > 0 {
		action = "warn"
	}
	return PolicyResult{Passed: true, Violations: violations, Warnings: warnings, Action: action}
}

func extractContext(text string, pattern *regexp.Regexp, contextChars int) string {
	loc := pattern.FindStringIndex(text)
	if loc == nil {
		return ""
	}
	start := int(math.Max(0, float64(loc[0]-contextChars)))
	end := int(math.Min(float64(len(text)), float64(loc[1]+contextChars)))
	return "..." + text[start:end] + "..."
}

func buildBlockMessage(violations []PolicyViolation) string {
	seen := make(map[string]bool)
	var categories []string
	for _, v := range violations {
		cat := strings.ReplaceAll(v.Category, "_", " ")
		if !seen[cat] {
			seen[cat] = true
			categories = append(categories, cat)
		}
	}
	return fmt.Sprintf(
		"Your message was blocked because it may contain content related to: %s. "+
			"If you believe this is an error, please rephrase your request.",
		strings.Join(categories, ", "),
	)
}

// ---------------------------------------------------------------------------
// Layer 5 — Injection Detector
// ---------------------------------------------------------------------------

var injectionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|directives?)`),
	regexp.MustCompile(`(?i)\b(you\s+are\s+now|you\s+are|act\s+as|pretend\s+to\s+be|roleplay\s+as)\b`),
	regexp.MustCompile(`(?i)\b(system\s*(prompt|message|instruction|override))\b`),
	regexp.MustCompile(`(?i)\b(forget|disregard|override)\s+(everything|all)\s+(before|above|you\s+know)\b`),
	regexp.MustCompile(`(?i)\[SYSTEM[^\]]*\]`),
	regexp.MustCompile(`(?i)\[INST[^\]]*\]`),
	regexp.MustCompile(`(?is)<\|system\|>.*?<\|/system\|>`),
	regexp.MustCompile(`(?i)\bnew\s+instructions?:`),
}

var delimiterPatterns = []*regexp.Regexp{
	regexp.MustCompile(`={3,}.*?={3,}`),
	regexp.MustCompile(`(?i)---\s*(system|instruction|override)\s*---`),
	regexp.MustCompile(`(?i)\[/\s*(system|instruction)\s*\]`),
}

var structuralPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?is)(respond|answer|reply|output).*?(always|only|must|never).*?(respond|answer|reply|output)`),
	regexp.MustCompile(`(?is)(print|show|reveal|display|output|spit\s+out)\s+.*?(system\s+prompt|instructions|your\s+prompt|your\s+configuration)`),
}

// InjectionDetector detects prompt injection using multiple pattern families.
type InjectionDetector struct{}

// Detect returns an InjectionResult for the given input.
// conversationHistory is accepted for API compatibility and reserved for future use.
func (d *InjectionDetector) Detect(userInput string, _ []map[string]any) InjectionResult {
	var detections []InjectionDetection

	for _, pattern := range injectionPatterns {
		loc := pattern.FindStringIndex(userInput)
		if loc != nil {
			s := int(math.Max(0, float64(loc[0]-20)))
			e := int(math.Min(float64(len(userInput)), float64(loc[1]+20)))
			detections = append(detections, InjectionDetection{
				Type:    "injection_pattern",
				Pattern: pattern.String(),
				Snippet: userInput[s:e],
			})
		}
	}

	for _, pattern := range delimiterPatterns {
		loc := pattern.FindStringIndex(userInput)
		if loc != nil {
			s := int(math.Max(0, float64(loc[0]-20)))
			e := int(math.Min(float64(len(userInput)), float64(loc[1]+20)))
			detections = append(detections, InjectionDetection{
				Type:    "delimiter_abuse",
				Pattern: pattern.String(),
				Snippet: userInput[s:e],
			})
		}
	}

	for _, pattern := range structuralPatterns {
		loc := pattern.FindStringIndex(userInput)
		if loc != nil {
			s := int(math.Max(0, float64(loc[0]-20)))
			e := int(math.Min(float64(len(userInput)), float64(loc[1]+20)))
			detections = append(detections, InjectionDetection{
				Type:    "structural_anomaly",
				Pattern: pattern.String(),
				Snippet: userInput[s:e],
			})
		}
	}

	riskLevel := assessRisk(detections)
	action := determineAction(riskLevel)
	return InjectionResult{
		RiskLevel:         riskLevel,
		Detections:        detections,
		RecommendedAction: action,
	}
}

func assessRisk(detections []InjectionDetection) string {
	if len(detections) == 0 {
		return "none"
	}
	var injectionCount, delimiterCount, structuralCount int
	for _, d := range detections {
		switch d.Type {
		case "injection_pattern":
			injectionCount++
		case "delimiter_abuse":
			delimiterCount++
		case "structural_anomaly":
			structuralCount++
		}
	}
	total := len(detections)
	if total >= 3 || delimiterCount >= 1 {
		return "critical"
	}
	if total >= 2 || injectionCount >= 2 {
		return "high"
	}
	if injectionCount >= 1 || structuralCount >= 2 {
		return "medium"
	}
	return "low"
}

func determineAction(riskLevel string) string {
	switch riskLevel {
	case "critical", "high":
		return "block"
	case "medium":
		return "sanitize"
	case "low":
		return "warn"
	default:
		return "allow"
	}
}

// ---------------------------------------------------------------------------
// Layer 6 — Input Sanitiser
// ---------------------------------------------------------------------------

var zeroWidthChars = []string{"\u200b", "\u200c", "\u200d", "\ufeff"}
var whitespaceRE = regexp.MustCompile(`\s+`)

// InputSanitizer normalises and cleans user input.
type InputSanitizer struct {
	hardCap int
}

// NewInputSanitizer creates a sanitiser with the given hard cap.
func NewInputSanitizer(hardCap int) *InputSanitizer {
	return &InputSanitizer{hardCap: hardCap}
}

// Sanitize applies all normalisation steps to userInput.
func (s *InputSanitizer) Sanitize(userInput string) string {
	// NFKC normalisation (via golang.org/x/text)
	text := norm.NFKC.String(userInput)

	for _, zw := range zeroWidthChars {
		text = strings.ReplaceAll(text, zw, "")
	}

	// Remove control characters except \n, \r, \t
	var sb strings.Builder
	for _, r := range text {
		if r >= 32 || r == '\n' || r == '\r' || r == '\t' {
			sb.WriteRune(r)
		}
	}
	text = sb.String()

	text = whitespaceRE.ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)

	runes := []rune(text)
	if len(runes) > s.hardCap {
		text = string(runes[:s.hardCap])
	}

	return text
}

// Deduplicate returns userInput unchanged, or an error if it is a
// near-duplicate of any entry in recentInputs (Jaccard ≥ threshold).
func (s *InputSanitizer) Deduplicate(
	userInput string,
	recentInputs []string,
	threshold float64,
) (string, error) {
	for _, recent := range recentInputs {
		if textSimilarity(userInput, recent) >= threshold {
			return "", fmt.Errorf("duplicate request detected")
		}
	}
	return userInput, nil
}

func textSimilarity(a, b string) float64 {
	aWords := wordSet(a)
	bWords := wordSet(b)
	if len(aWords) == 0 || len(bWords) == 0 {
		return 0
	}
	var intersection int
	for w := range aWords {
		if bWords[w] {
			intersection++
		}
	}
	union := len(aWords) + len(bWords) - intersection
	return float64(intersection) / float64(union)
}

func wordSet(s string) map[string]bool {
	words := strings.Fields(strings.ToLower(s))
	set := make(map[string]bool, len(words))
	for _, w := range words {
		set[w] = true
	}
	return set
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

// InputGuardrailPipeline processes user input through all six layers.
type InputGuardrailPipeline struct {
	config            Config
	rateLimiter       *RateLimiter
	structural        *StructuralValidator
	piiDetector       *PIIDetector
	contentPolicy     *ContentPolicyEnforcer
	injectionDetector *InjectionDetector
	sanitizer         *InputSanitizer
	logger            *slog.Logger
}

// NewInputGuardrailPipeline creates a pipeline using the given functional options.
func NewInputGuardrailPipeline(opts ...Option) *InputGuardrailPipeline {
	cfg := defaultConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return &InputGuardrailPipeline{
		config:            cfg,
		rateLimiter:       NewRateLimiter(cfg.RateLimitRPM, cfg.RateLimitRPH, cfg.RateLimitRPD),
		structural:        NewStructuralValidator(cfg.MinInputLength, cfg.MaxInputLength, cfg.MaxInputTokens),
		piiDetector:       &PIIDetector{},
		contentPolicy:     NewContentPolicyEnforcer(cfg.UseLLMContentReview),
		injectionDetector: &InjectionDetector{},
		sanitizer:         NewInputSanitizer(cfg.SanitiserHardCap),
		logger:            slog.Default(),
	}
}

// Process runs userInput through all six guardrail layers.
//
// ctx can be used to cancel the operation.
// userID is used for per-user rate limiting.
// conversationHistory is passed to the injection detector for context-aware analysis.
// recentInputs is used for near-duplicate detection.
func (p *InputGuardrailPipeline) Process(
	ctx context.Context,
	userInput string,
	userID string,
	conversationHistory []map[string]any,
	recentInputs []string,
) (GuardrailResult, error) {
	result := GuardrailResult{
		OriginalInput: userInput,
		Checks:        make(map[string]any),
	}

	if err := ctx.Err(); err != nil {
		return result, err
	}

	// Layer 1: Rate limiting
	rateCheck := p.rateLimiter.Check(userID)
	if !rateCheck.Allowed {
		p.log("guardrail.rejected", "layer", "rate_limiter", "user_id", userID, "reason", rateCheck.Reason)
		result.RejectionReason = rateCheck.Reason
		result.RejectionLayer = "rate_limiter"
		return result, nil
	}

	// Layer 2: Structural validation
	structuralCheck := p.structural.Validate(userInput)
	if !structuralCheck.Passed {
		p.log("guardrail.rejected", "layer", "structural", "user_id", userID, "reason", structuralCheck.Reason)
		result.RejectionReason = structuralCheck.Reason
		result.RejectionLayer = "structural"
		return result, nil
	}
	result.Checks["structural"] = structuralCheck

	// Layer 3: PII detection and redaction
	piiDetections := p.piiDetector.Detect(userInput)
	if len(piiDetections) > 0 {
		types := make([]string, len(piiDetections))
		for i, d := range piiDetections {
			types[i] = d.Type
		}
		userInput, _ = p.piiDetector.Redact(userInput, piiDetections)
		p.log("guardrail.pii_redacted", "user_id", userID, "count", len(piiDetections), "types", types)
	}
	result.Checks["pii"] = PIIResult{Detections: piiDetections, Redacted: len(piiDetections) > 0}

	// Layer 4: Content policy
	policyCheck := p.contentPolicy.Enforce(userInput)
	if !policyCheck.Passed {
		cats := make([]string, len(policyCheck.Violations))
		for i, v := range policyCheck.Violations {
			cats[i] = v.Category
		}
		p.log("guardrail.rejected", "layer", "content_policy", "user_id", userID, "violations", cats)
		result.RejectionReason = policyCheck.Message
		result.RejectionLayer = "content_policy"
		return result, nil
	}
	result.Checks["content_policy"] = policyCheck

	// Layer 5: Injection detection
	injectionCheck := p.injectionDetector.Detect(userInput, conversationHistory)
	if injectionCheck.RecommendedAction == "block" {
		p.log("guardrail.rejected", "layer", "injection_detector", "user_id", userID, "risk_level", injectionCheck.RiskLevel)
		result.RejectionReason = "Your request could not be processed due to security concerns."
		result.RejectionLayer = "injection_detector"
		return result, nil
	}
	result.Checks["injection"] = injectionCheck

	// Layer 6: Sanitisation
	userInput = p.sanitizer.Sanitize(userInput)

	if len(recentInputs) > 0 {
		deduped, err := p.sanitizer.Deduplicate(userInput, recentInputs, 0.9)
		if err != nil {
			result.RejectionReason = err.Error()
			result.RejectionLayer = "deduplication"
			return result, nil
		}
		userInput = deduped
	}

	result.CleanedInput = userInput
	result.Passed = true

	layers := make([]string, 0, len(result.Checks))
	for k := range result.Checks {
		layers = append(layers, k)
	}
	p.log("guardrail.passed", "user_id", userID, "layers_checked", layers)

	return result, nil
}

func (p *InputGuardrailPipeline) log(msg string, args ...any) {
	p.logger.Info(msg, args...)
}

// ---------------------------------------------------------------------------
// JSON helper for demo
// ---------------------------------------------------------------------------

func toJSON(v any) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}
