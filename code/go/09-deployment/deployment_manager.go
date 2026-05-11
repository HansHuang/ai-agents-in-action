// Package deployment provides production deployment primitives for AI agents.
//
// Includes gradual rollout management, canary evaluation, per-user cost
// control, multi-region routing, and atomic rollback.
//
// Reference: docs/09-from-dev-to-production/01-deployment-strategies.md
package main

import (
	"context"
	"crypto/md5" //nolint:gosec // MD5 used only for deterministic bucketing, not security
	"encoding/hex"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

// VersionMetrics holds health metrics for one agent version.
type VersionMetrics struct {
	ErrorRate        float64
	P50Latency       float64
	P95Latency       float64
	P99Latency       float64
	AvgCost          float64
	SafetyBlockRate  float64
	TaskSuccessRate  float64
	UserSatisfaction float64
}

// CanaryEvaluation is the result of comparing canary vs stable metrics.
type CanaryEvaluation struct {
	CanaryPct      int
	HasIssues      bool
	Issues         []string
	StableMetrics  VersionMetrics
	CanaryMetrics  VersionMetrics
	Recommendation string
}

// BudgetCheck is the result of a pre-request budget check.
type BudgetCheck struct {
	Allowed        bool
	Reason         string
	CurrentUserCost float64
	UserBudget     float64
}

// RollbackItemResult is the result of rolling back a single artifact.
type RollbackItemResult struct {
	Name    string
	Success bool
	Error   string
}

// RollbackResult is the result of an entire rollback operation.
type RollbackResult struct {
	Reason            string
	Items             []RollbackItemResult
	TotalTimeSeconds  int
	Success           bool
	Timestamp         time.Time
}

// ---------------------------------------------------------------------------
// Feature Flag Service (in-memory stub)
// ---------------------------------------------------------------------------

// FeatureFlagService is a minimal in-memory feature-flag store.
// Replace with LaunchDarkly, Unleash, or similar in production.
type FeatureFlagService struct {
	mu           sync.RWMutex
	flags        map[string]interface{}
	internalUsers map[string]struct{}
}

// NewFeatureFlagService creates a new FeatureFlagService.
func NewFeatureFlagService() *FeatureFlagService {
	return &FeatureFlagService{
		flags: make(map[string]interface{}),
		internalUsers: map[string]struct{}{
			"internal-001": {},
			"internal-002": {},
			"dev-team":     {},
		},
	}
}

// IsInternalUser returns true if userId belongs to the internal team.
func (f *FeatureFlagService) IsInternalUser(userID string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	_, ok := f.internalUsers[userID]
	return ok
}

// GetString returns a string flag value, or defaultValue if not set.
func (f *FeatureFlagService) GetString(key, defaultValue string) string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if v, ok := f.flags[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return defaultValue
}

// GetInt returns an int flag value, or defaultValue if not set.
func (f *FeatureFlagService) GetInt(key string, defaultValue int) int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if v, ok := f.flags[key]; ok {
		if n, ok := v.(int); ok {
			return n
		}
	}
	return defaultValue
}

// SetInt sets an integer flag.
func (f *FeatureFlagService) SetInt(key string, value int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flags[key] = value
}

// SetString sets a string flag.
func (f *FeatureFlagService) SetString(key, value string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flags[key] = value
}

// ---------------------------------------------------------------------------
// Harness Metrics (in-memory stub)
// ---------------------------------------------------------------------------

type versionData struct {
	requests    int
	errors      int
	latencies   []float64
	costs       []float64
	safetyBlocks int
	taskSuccesses int
}

// HarnessMetrics stores per-version request metrics.
type HarnessMetrics struct {
	mu   sync.Mutex
	data map[string]*versionData
}

// NewHarnessMetrics creates a new HarnessMetrics.
func NewHarnessMetrics() *HarnessMetrics {
	return &HarnessMetrics{data: make(map[string]*versionData)}
}

// Record records a single request result.
func (h *HarnessMetrics) Record(version string, latency, cost float64, isError, safetyBlocked, taskSuccess bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.data[version] == nil {
		h.data[version] = &versionData{}
	}
	d := h.data[version]
	d.requests++
	if isError {
		d.errors++
	}
	d.latencies = append(d.latencies, latency)
	d.costs = append(d.costs, cost)
	if safetyBlocked {
		d.safetyBlocks++
	}
	if taskSuccess {
		d.taskSuccesses++
	}
}

// GetMetrics returns aggregated metrics for a version.
func (h *HarnessMetrics) GetMetrics(version string) VersionMetrics {
	h.mu.Lock()
	defer h.mu.Unlock()
	d := h.data[version]
	if d == nil || d.requests == 0 {
		return VersionMetrics{
			P50Latency: 1.0, P95Latency: 2.0, P99Latency: 3.0,
			AvgCost: 0.02, TaskSuccessRate: 1.0, UserSatisfaction: 0.9,
		}
	}
	sorted := make([]float64, len(d.latencies))
	copy(sorted, d.latencies)
	sort.Float64s(sorted)
	n := len(sorted)
	pct := func(p float64) float64 {
		idx := int(math.Min(float64(n)*p, float64(n-1)))
		return sorted[idx]
	}
	var totalCost float64
	for _, c := range d.costs {
		totalCost += c
	}
	return VersionMetrics{
		ErrorRate:       float64(d.errors) / float64(d.requests),
		P50Latency:      pct(0.50),
		P95Latency:      pct(0.95),
		P99Latency:      pct(0.99),
		AvgCost:         totalCost / float64(len(d.costs)),
		SafetyBlockRate: float64(d.safetyBlocks) / float64(d.requests),
		TaskSuccessRate: float64(d.taskSuccesses) / float64(d.requests),
		UserSatisfaction: 0.85,
	}
}

// ---------------------------------------------------------------------------
// 1. Deployment Manager
// ---------------------------------------------------------------------------

// RolloutStages defines the allowed canary rollout percentages.
var RolloutStages = []int{0, 1, 5, 25, 100}

// DeploymentManager manages gradual rollout of new agent versions.
//
// Rollout stages:
//   Stage 0: Internal only (0% external)
//   Stage 1: 1% canary (monitor 24h)
//   Stage 2: 5% extended canary (monitor 48h)
//   Stage 3: 25% beta (monitor 72h)
//   Stage 4: 100% full rollout
type DeploymentManager struct {
	mu          sync.RWMutex
	flags       *FeatureFlagService
	halted      bool
	haltReason  string
}

// NewDeploymentManager creates a DeploymentManager with default configuration.
func NewDeploymentManager(flags *FeatureFlagService) *DeploymentManager {
	flags.SetInt("canary_rollout_pct", 0)
	flags.SetString("stable_version", "v3.1.0")
	flags.SetString("canary_version", "v3.2.1")
	flags.SetString("internal_version", "v3.2.1")
	return &DeploymentManager{flags: flags}
}

// GetAgentVersion determines which agent version a user should receive.
//
// Resolution order:
//  1. Internal users always get the internal (latest) version.
//  2. If rollout is halted, everyone gets stable.
//  3. Otherwise, hash user_id to determine canary bucket.
func (dm *DeploymentManager) GetAgentVersion(userID string) string {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	if dm.flags.IsInternalUser(userID) {
		return dm.flags.GetString("internal_version", "v3.2.1")
	}
	if dm.halted {
		return dm.flags.GetString("stable_version", "v3.1.0")
	}
	pct := dm.flags.GetInt("canary_rollout_pct", 0)
	if dm.userInRolloutGroup(userID, pct) {
		return dm.flags.GetString("canary_version", "v3.2.1")
	}
	return dm.flags.GetString("stable_version", "v3.1.0")
}

// userInRolloutGroup returns true if MD5(userID) mod 100 < percentage.
func (dm *DeploymentManager) userInRolloutGroup(userID string, percentage int) bool {
	h := md5.Sum([]byte(userID)) //nolint:gosec
	hexStr := hex.EncodeToString(h[:4])
	var val uint64
	fmt.Sscanf(hexStr, "%x", &val) //nolint:errcheck
	return int(val%100) < percentage
}

// PromoteRollout increases the canary rollout percentage.
func (dm *DeploymentManager) PromoteRollout(fromPct, toPct int) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	valid := false
	for _, s := range RolloutStages {
		if s == toPct {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("toPct must be one of %v", RolloutStages)
	}
	current := dm.flags.GetInt("canary_rollout_pct", 0)
	if current != fromPct {
		return fmt.Errorf("current rollout is %d%%, not %d%%", current, fromPct)
	}
	canary := dm.flags.GetString("canary_version", "")
	log.Printf("Promoting %s rollout: %d%% → %d%%", canary, fromPct, toPct)
	dm.flags.SetInt("canary_rollout_pct", toPct)
	dm.halted = false
	return nil
}

// HaltRollout immediately routes all external users back to stable.
func (dm *DeploymentManager) HaltRollout(reason string) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	log.Printf("ROLLOUT HALTED: %s", reason)
	dm.halted = true
	dm.haltReason = reason
	dm.flags.SetInt("canary_rollout_pct", 0)
}

// GetRolloutStatus returns the current rollout state.
func (dm *DeploymentManager) GetRolloutStatus() map[string]interface{} {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	return map[string]interface{}{
		"stable_version":     dm.flags.GetString("stable_version", ""),
		"canary_version":     dm.flags.GetString("canary_version", ""),
		"internal_version":   dm.flags.GetString("internal_version", ""),
		"canary_rollout_pct": dm.flags.GetInt("canary_rollout_pct", 0),
		"halted":             dm.halted,
		"halt_reason":        dm.haltReason,
	}
}

// ---------------------------------------------------------------------------
// 2. Canary Deployer
// ---------------------------------------------------------------------------

// CanaryDeployer routes requests to stable or canary and evaluates health.
type CanaryDeployer struct {
	dm      *DeploymentManager
	metrics *HarnessMetrics
}

// NewCanaryDeployer creates a CanaryDeployer.
func NewCanaryDeployer(dm *DeploymentManager, metrics *HarnessMetrics) *CanaryDeployer {
	return &CanaryDeployer{dm: dm, metrics: metrics}
}

// Process simulates routing a request to the appropriate harness.
func (c *CanaryDeployer) Process(_ context.Context, userInput, userID string) string {
	version := c.dm.GetAgentVersion(userID)
	canaryVersion := c.dm.flags.GetString("canary_version", "v3.2.1")
	if version == canaryVersion {
		c.metrics.Record("canary", 1.0, 0.025, false, false, true)
		return fmt.Sprintf("[canary] Response to: %s", userInput)
	}
	c.metrics.Record("stable", 0.9, 0.020, false, false, true)
	return fmt.Sprintf("[stable] Response to: %s", userInput)
}

// EvaluateCanary compares canary vs stable across 5 dimensions.
func (c *CanaryDeployer) EvaluateCanary() CanaryEvaluation {
	stable := c.metrics.GetMetrics("stable")
	canary := c.metrics.GetMetrics("canary")
	pct := c.dm.flags.GetInt("canary_rollout_pct", 0)

	var issues []string
	if !c.checkErrorRate(stable, canary) {
		issues = append(issues, fmt.Sprintf(
			"Error rate: stable=%.2f%%, canary=%.2f%%",
			stable.ErrorRate*100, canary.ErrorRate*100,
		))
	}
	if !c.checkLatency(stable, canary) {
		issues = append(issues, fmt.Sprintf(
			"P95 latency: stable=%.1fs, canary=%.1fs",
			stable.P95Latency, canary.P95Latency,
		))
	}
	if !c.checkCost(stable, canary) {
		issues = append(issues, fmt.Sprintf(
			"Avg cost: stable=$%.3f, canary=$%.3f",
			stable.AvgCost, canary.AvgCost,
		))
	}
	if !c.checkSafety(stable, canary) {
		issues = append(issues, fmt.Sprintf(
			"Safety block rate: stable=%.2f%%, canary=%.2f%%",
			stable.SafetyBlockRate*100, canary.SafetyBlockRate*100,
		))
	}
	if !c.checkTaskSuccess(stable, canary) {
		issues = append(issues, fmt.Sprintf(
			"Task success: stable=%.2f%%, canary=%.2f%%",
			stable.TaskSuccessRate*100, canary.TaskSuccessRate*100,
		))
	}
	return CanaryEvaluation{
		CanaryPct:      pct,
		HasIssues:      len(issues) > 0,
		Issues:         issues,
		StableMetrics:  stable,
		CanaryMetrics:  canary,
		Recommendation: c.generateRecommendation(issues),
	}
}

func (c *CanaryDeployer) checkErrorRate(stable, canary VersionMetrics) bool {
	if stable.ErrorRate == 0 {
		return canary.ErrorRate == 0
	}
	return canary.ErrorRate <= stable.ErrorRate*1.5
}

func (c *CanaryDeployer) checkLatency(stable, canary VersionMetrics) bool {
	return canary.P95Latency <= stable.P95Latency*1.2
}

func (c *CanaryDeployer) checkCost(stable, canary VersionMetrics) bool {
	return canary.AvgCost <= stable.AvgCost*1.2
}

func (c *CanaryDeployer) checkSafety(stable, canary VersionMetrics) bool {
	if stable.SafetyBlockRate == 0 {
		return canary.SafetyBlockRate == 0
	}
	return canary.SafetyBlockRate <= stable.SafetyBlockRate*1.5
}

func (c *CanaryDeployer) checkTaskSuccess(stable, canary VersionMetrics) bool {
	return canary.TaskSuccessRate >= stable.TaskSuccessRate*0.95
}

func (c *CanaryDeployer) generateRecommendation(issues []string) string {
	switch len(issues) {
	case 0:
		return "Canary is healthy. Consider increasing rollout percentage."
	case 1:
		return "Minor issues detected. Monitor for another hour before promoting."
	default:
		return "Significant issues detected. Halt rollout and investigate."
	}
}

// ---------------------------------------------------------------------------
// 3. Production Cost Controller
// ---------------------------------------------------------------------------

// CostConfig holds budget limits.
type CostConfig struct {
	UserDailyBudget       float64
	TotalDailyBudget      float64
	MaxCostPerRequest     float64
	FreeTierDailyBudget   float64
	EnterpriseDailyBudget float64
}

// DefaultCostConfig returns production-safe defaults.
func DefaultCostConfig() CostConfig {
	return CostConfig{
		UserDailyBudget:       10.0,
		TotalDailyBudget:      1000.0,
		MaxCostPerRequest:     1.0,
		FreeTierDailyBudget:   0.50,
		EnterpriseDailyBudget: 50.0,
	}
}

// ProductionCostController enforces per-user and total daily budgets.
type ProductionCostController struct {
	mu              sync.Mutex
	config          CostConfig
	userDailyCosts  map[string]float64
	totalDailyCost  float64
	freeTierUsers   map[string]struct{}
	enterpriseUsers map[string]struct{}
	lastAlertLevel  string
}

// NewProductionCostController creates a ProductionCostController.
func NewProductionCostController(cfg CostConfig) *ProductionCostController {
	return &ProductionCostController{
		config:          cfg,
		userDailyCosts:  make(map[string]float64),
		freeTierUsers:   make(map[string]struct{}),
		enterpriseUsers: make(map[string]struct{}),
	}
}

// RegisterFreeTier marks a user as free tier.
func (p *ProductionCostController) RegisterFreeTier(userID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.freeTierUsers[userID] = struct{}{}
}

// RegisterEnterprise marks a user as enterprise.
func (p *ProductionCostController) RegisterEnterprise(userID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enterpriseUsers[userID] = struct{}{}
}

func (p *ProductionCostController) userBudget(userID string) float64 {
	if _, ok := p.enterpriseUsers[userID]; ok {
		return p.config.EnterpriseDailyBudget
	}
	if _, ok := p.freeTierUsers[userID]; ok {
		return p.config.FreeTierDailyBudget
	}
	return p.config.UserDailyBudget
}

// CheckBudget performs a pre-request budget check.
func (p *ProductionCostController) CheckBudget(userID string, estimatedCost float64) BudgetCheck {
	p.mu.Lock()
	defer p.mu.Unlock()
	current := p.userDailyCosts[userID]
	budget := p.userBudget(userID)

	if estimatedCost > p.config.MaxCostPerRequest {
		return BudgetCheck{
			Reason:         fmt.Sprintf("Request cost $%.3f exceeds per-request limit $%.2f.", estimatedCost, p.config.MaxCostPerRequest),
			CurrentUserCost: current,
			UserBudget:     budget,
		}
	}
	if current+estimatedCost > budget {
		return BudgetCheck{
			Reason:         fmt.Sprintf("Daily budget of $%.2f exceeded. Current: $%.2f.", budget, current),
			CurrentUserCost: current,
			UserBudget:     budget,
		}
	}
	if p.totalDailyCost+estimatedCost > p.config.TotalDailyBudget {
		return BudgetCheck{
			Reason:         "Service temporarily unavailable due to high demand. Please try again later.",
			CurrentUserCost: current,
			UserBudget:     budget,
		}
	}
	return BudgetCheck{Allowed: true, CurrentUserCost: current, UserBudget: budget}
}

// RecordCost records actual cost after a request completes.
func (p *ProductionCostController) RecordCost(userID string, cost float64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.userDailyCosts[userID] += cost
	p.totalDailyCost += cost
	p.checkAndFireAlerts()
}

func (p *ProductionCostController) checkAndFireAlerts() {
	if p.config.TotalDailyBudget <= 0 {
		return
	}
	pct := p.totalDailyCost / p.config.TotalDailyBudget
	switch {
	case pct >= 1.0:
		p.triggerAlert("critical", fmt.Sprintf("Daily budget exhausted: $%.2f", p.totalDailyCost))
	case pct >= 0.9:
		p.triggerAlert("warning", fmt.Sprintf("Daily budget at %.0f%%: $%.2f", pct*100, p.totalDailyCost))
	case pct >= 0.7:
		p.triggerAlert("info", fmt.Sprintf("Daily budget at %.0f%%: $%.2f", pct*100, p.totalDailyCost))
	}
}

func (p *ProductionCostController) triggerAlert(level, message string) {
	if level != p.lastAlertLevel {
		log.Printf("[COST ALERT %s] %s", level, message)
		p.lastAlertLevel = level
	}
}

// GetCostReport returns a daily cost summary.
func (p *ProductionCostController) GetCostReport() map[string]interface{} {
	p.mu.Lock()
	defer p.mu.Unlock()
	budget := p.config.TotalDailyBudget
	if budget == 0 {
		budget = 1
	}
	return map[string]interface{}{
		"total_daily_cost": p.totalDailyCost,
		"daily_budget":     p.config.TotalDailyBudget,
		"budget_remaining": p.config.TotalDailyBudget - p.totalDailyCost,
		"pct_used":         p.totalDailyCost / budget,
		"user_count":       len(p.userDailyCosts),
	}
}

// ResetDailyCosts resets all daily counters (call at midnight).
func (p *ProductionCostController) ResetDailyCosts() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for k := range p.userDailyCosts {
		delete(p.userDailyCosts, k)
	}
	p.totalDailyCost = 0
	p.lastAlertLevel = ""
	log.Println("Daily costs reset.")
}

// ---------------------------------------------------------------------------
// 4. Multi-Region Deployer
// ---------------------------------------------------------------------------

// RegionConfig holds configuration for a deployment region.
type RegionConfig struct {
	LLMProvider            string
	FallbackProvider       string
	VectorDBEndpoint       string
	LatencyToProviderMS    int
	EUResidencyCompliant   bool
}

var regions = map[string]RegionConfig{
	"us-east": {
		LLMProvider:          "openai",
		FallbackProvider:     "anthropic",
		VectorDBEndpoint:     "https://us-east.qdrant.example.com",
		LatencyToProviderMS:  50,
		EUResidencyCompliant: false,
	},
	"eu-west": {
		LLMProvider:          "openai",
		FallbackProvider:     "anthropic",
		VectorDBEndpoint:     "https://eu-west.qdrant.example.com",
		LatencyToProviderMS:  80,
		EUResidencyCompliant: true,
	},
	"ap-southeast": {
		LLMProvider:          "anthropic",
		FallbackProvider:     "openai",
		VectorDBEndpoint:     "https://ap-se.qdrant.example.com",
		LatencyToProviderMS:  120,
		EUResidencyCompliant: false,
	},
}

var geoMap = [][2]string{
	{"52.", "us-east"},
	{"18.", "us-east"},
	{"34.", "us-east"},
	{"35.", "eu-west"},
	{"13.", "ap-southeast"},
}

var euPrefixes = []string{"195.", "212.", "217.", "82.", "185.", "37.", "31."}

// MultiRegionDeployer routes users to the nearest healthy region.
type MultiRegionDeployer struct {
	mu              sync.RWMutex
	circuitBreakers map[string]bool
}

// NewMultiRegionDeployer creates a MultiRegionDeployer.
func NewMultiRegionDeployer() *MultiRegionDeployer {
	cb := make(map[string]bool)
	for r := range regions {
		cb[r] = false
	}
	return &MultiRegionDeployer{circuitBreakers: cb}
}

// GetRegion determines the best region for a user request.
func (m *MultiRegionDeployer) GetRegion(userIP string, userPrefs map[string]string) (string, error) {
	if pref, ok := userPrefs["region"]; ok {
		if _, exists := regions[pref]; exists && m.IsRegionHealthy(pref) {
			return pref, nil
		}
	}
	var region string
	if m.isEUUser(userIP) {
		region = "eu-west"
	} else {
		region = m.geoRoute(userIP)
	}
	if !m.IsRegionHealthy(region) {
		return m.getNearestHealthyRegion(region)
	}
	return region, nil
}

func (m *MultiRegionDeployer) isEUUser(ip string) bool {
	for _, prefix := range euPrefixes {
		if len(ip) >= len(prefix) && ip[:len(prefix)] == prefix {
			return true
		}
	}
	return false
}

func (m *MultiRegionDeployer) geoRoute(ip string) string {
	for _, pair := range geoMap {
		prefix := pair[0]
		if len(ip) >= len(prefix) && ip[:len(prefix)] == prefix {
			return pair[1]
		}
	}
	return "us-east"
}

// IsRegionHealthy returns true if the circuit breaker for a region is closed.
func (m *MultiRegionDeployer) IsRegionHealthy(region string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return !m.circuitBreakers[region]
}

func (m *MultiRegionDeployer) getNearestHealthyRegion(exclude string) (string, error) {
	best := ""
	bestLatency := math.MaxInt32
	for r, cfg := range regions {
		if r != exclude && m.IsRegionHealthy(r) && cfg.LatencyToProviderMS < bestLatency {
			best = r
			bestLatency = cfg.LatencyToProviderMS
		}
	}
	if best == "" {
		return "", fmt.Errorf("no healthy regions available")
	}
	return best, nil
}

// GetRegionConfig returns the configuration for a region.
func (m *MultiRegionDeployer) GetRegionConfig(region string) (RegionConfig, error) {
	cfg, ok := regions[region]
	if !ok {
		return RegionConfig{}, fmt.Errorf("unknown region: %s", region)
	}
	return cfg, nil
}

// OpenCircuitBreaker marks a region as unhealthy.
func (m *MultiRegionDeployer) OpenCircuitBreaker(region string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	log.Printf("Circuit breaker OPEN for region: %s", region)
	m.circuitBreakers[region] = true
}

// CloseCircuitBreaker marks a region as healthy again.
func (m *MultiRegionDeployer) CloseCircuitBreaker(region string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	log.Printf("Circuit breaker CLOSED for region: %s", region)
	m.circuitBreakers[region] = false
}

// ---------------------------------------------------------------------------
// 5. Rollback Manager
// ---------------------------------------------------------------------------

type rollbackItem struct {
	name        string
	description string
	fn          func(ctx context.Context) error
	timeSeconds int
}

// RollbackManager manages rollbacks for AI agent deployments.
type RollbackManager struct {
	dm      *DeploymentManager
	items   []rollbackItem
	history []RollbackResult
	mu      sync.Mutex
}

// NewRollbackManager creates a RollbackManager.
func NewRollbackManager(dm *DeploymentManager) *RollbackManager {
	rm := &RollbackManager{dm: dm}
	rm.items = []rollbackItem{
		{name: "config", description: "Revert harness configuration", fn: rm.rollbackConfig, timeSeconds: 10},
		{name: "prompt", description: "Revert system prompt", fn: rm.rollbackPrompt, timeSeconds: 10},
		{name: "model", description: "Switch to previous model version", fn: rm.rollbackModel, timeSeconds: 30},
		{name: "tools", description: "Revert tool definitions", fn: rm.rollbackTools, timeSeconds: 30},
		{name: "code", description: "Revert application code", fn: rm.rollbackCode, timeSeconds: 60},
		{name: "documents", description: "Revert knowledge base and re-embed", fn: rm.rollbackDocuments, timeSeconds: 300},
	}
	return rm
}

// Rollback executes a rollback. If itemNames is nil, all items are rolled back.
func (rm *RollbackManager) Rollback(ctx context.Context, reason string, itemNames []string) (RollbackResult, error) {
	log.Printf("ROLLBACK INITIATED: %s", reason)
	rm.dm.HaltRollout(reason)

	candidates := rm.items
	if itemNames != nil {
		nameSet := make(map[string]struct{})
		for _, n := range itemNames {
			nameSet[n] = struct{}{}
		}
		candidates = nil
		for _, item := range rm.items {
			if _, ok := nameSet[item.name]; ok {
				candidates = append(candidates, item)
			}
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].timeSeconds < candidates[j].timeSeconds
	})

	var results []RollbackItemResult
	totalTime := 0
	for _, item := range candidates {
		log.Printf("Rolling back: %s …", item.name)
		totalTime += item.timeSeconds
		err := item.fn(ctx)
		if err != nil {
			results = append(results, RollbackItemResult{Name: item.name, Success: false, Error: err.Error()})
			log.Printf("Rollback failed: %s: %v", item.name, err)
		} else {
			results = append(results, RollbackItemResult{Name: item.name, Success: true})
			log.Printf("Rollback complete: %s", item.name)
		}
	}

	allOK := true
	for _, r := range results {
		if !r.Success {
			allOK = false
			break
		}
	}

	result := RollbackResult{
		Reason:           reason,
		Items:            results,
		TotalTimeSeconds: totalTime,
		Success:          allOK,
		Timestamp:        time.Now(),
	}
	rm.mu.Lock()
	rm.history = append(rm.history, result)
	rm.mu.Unlock()
	return result, nil
}

func (rm *RollbackManager) rollbackCode(_ context.Context) error {
	log.Println("[stub] git revert HEAD && kubectl rollout restart …")
	return nil
}

func (rm *RollbackManager) rollbackModel(_ context.Context) error {
	log.Println("[stub] Reverting model config to previous version …")
	return nil
}

func (rm *RollbackManager) rollbackPrompt(_ context.Context) error {
	log.Println("[stub] Loading previous prompt version from library …")
	return nil
}

func (rm *RollbackManager) rollbackConfig(_ context.Context) error {
	log.Println("[stub] Restoring harness config from previous snapshot …")
	return nil
}

func (rm *RollbackManager) rollbackTools(_ context.Context) error {
	log.Println("[stub] Restoring tool definitions from previous release …")
	return nil
}

func (rm *RollbackManager) rollbackDocuments(_ context.Context) error {
	log.Println("[stub] Restoring previous document snapshot and re-embedding …")
	return nil
}

// GetHistory returns past rollback results.
func (rm *RollbackManager) GetHistory() []RollbackResult {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	cp := make([]RollbackResult, len(rm.history))
	copy(cp, rm.history)
	return cp
}

// TestRollback performs a dry-run of all rollback methods.
func (rm *RollbackManager) TestRollback(ctx context.Context) bool {
	log.Println("DRY-RUN: testing all rollback methods …")
	for _, item := range rm.items {
		if err := item.fn(ctx); err != nil {
			log.Printf("DRY-RUN failed: %s: %v", item.name, err)
			return false
		}
	}
	log.Println("DRY-RUN: all rollback methods succeeded.")
	return true
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

func main() {
	ctx := context.Background()
	fmt.Println("\n" + "============================================================")
	fmt.Println("  AI Agent Deployment Manager (Go) — Demo")
	fmt.Println("============================================================")

	flags := NewFeatureFlagService()
	dm := NewDeploymentManager(flags)
	metrics := NewHarnessMetrics()
	deployer := NewCanaryDeployer(dm, metrics)
	costCtrl := NewProductionCostController(DefaultCostConfig())
	multiRegion := NewMultiRegionDeployer()
	rollbackMgr := NewRollbackManager(dm)

	// Gradual rollout
	fmt.Println("\n--- GRADUAL ROLLOUT ---")
	fmt.Printf("Stage 0: %v\n", dm.GetRolloutStatus())
	fmt.Printf("  internal-001 → %s\n", dm.GetAgentVersion("internal-001"))
	fmt.Printf("  external-42  → %s\n", dm.GetAgentVersion("external-42"))

	if err := dm.PromoteRollout(0, 1); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nStage 1 (1%%): %v\n", dm.GetRolloutStatus())

	for i := 0; i < 20; i++ {
		deployer.Process(ctx, "Hello", fmt.Sprintf("user-%04d", i))
	}

	eval := deployer.EvaluateCanary()
	fmt.Printf("  Canary eval: has_issues=%v, recommendation='%s'\n",
		eval.HasIssues, eval.Recommendation)

	if err := dm.PromoteRollout(1, 5); err != nil {
		log.Fatal(err)
	}
	metrics.Record("canary", 3.5, 0.04, true, false, true)
	metrics.Record("canary", 3.8, 0.04, true, false, true)
	eval2 := deployer.EvaluateCanary()
	fmt.Printf("\nStage 2 (5%%) with error spike:\n")
	fmt.Printf("  Issues: %v\n", eval2.Issues)
	fmt.Printf("  Recommendation: %s\n", eval2.Recommendation)

	if eval2.HasIssues {
		dm.HaltRollout("Error rate spike at 5%")
		fmt.Printf("  Halted: %v\n", dm.GetRolloutStatus())
	}

	// Rollback
	fmt.Println("\n--- ROLLBACK ---")
	rbResult, _ := rollbackMgr.Rollback(ctx, "Error rate exceeded threshold", []string{"config", "prompt"})
	names := make([]string, len(rbResult.Items))
	for i, r := range rbResult.Items {
		names[i] = r.Name
	}
	fmt.Printf("  Success: %v, items: %v\n", rbResult.Success, names)

	// Multi-region
	fmt.Println("\n--- MULTI-REGION ROUTING ---")
	testCases := [][2]string{
		{"52.86.1.1", "US"},
		{"195.50.10.1", "EU (GDPR)"},
		{"13.250.1.1", "AP"},
	}
	for _, tc := range testCases {
		region, _ := multiRegion.GetRegion(tc[0], nil)
		cfg, _ := multiRegion.GetRegionConfig(region)
		fmt.Printf("  %s (%s) → %s (provider: %s, %dms)\n",
			tc[1], tc[0], region, cfg.LLMProvider, cfg.LatencyToProviderMS)
	}

	// Cost controller
	fmt.Println("\n--- COST CONTROLLER ---")
	costCtrl.RegisterFreeTier("free-user-1")
	type costTest struct{ user string; cost float64 }
	for _, tc := range []costTest{{"premium-1", 0.03}, {"free-user-1", 0.60}, {"premium-1", 0.03}} {
		check := costCtrl.CheckBudget(tc.user, tc.cost)
		if check.Allowed {
			fmt.Printf("  %s ($%.2f) → ✓ allowed\n", tc.user, tc.cost)
			costCtrl.RecordCost(tc.user, tc.cost)
		} else {
			fmt.Printf("  %s ($%.2f) → ✗ rejected: %s\n", tc.user, tc.cost, check.Reason)
		}
	}

	fmt.Println("\n" + "============================================================")
}
