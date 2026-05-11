// resilience_layer.go — Retry + Fallback + Circuit Breaker for production AI agents.
//
// Go port of the Python resilience_layer.py implementation.
// Uses context.WithTimeout for deadlines, sync.Mutex for concurrent safety,
// and proper Go error wrapping throughout.
//
// Types:
//
//	RetryConfig      — exponential-backoff retry configuration
//	FallbackLevel    — single level in a fallback chain
//	FallbackExecutor — executes operation through ordered fallback chain
//	RLCircuitBreaker   — three-state (Closed/Open/HalfOpen) circuit breaker
//	ResilienceLayer  — combines all three patterns
//	ResilienceMonitor — health check and alerting
//
// See: docs/07-harness-engineering/04-retry-fallback-and-circuit-breakers.md
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Custom error types
// ---------------------------------------------------------------------------

// MaxRetriesExceeded is returned when all retry attempts are exhausted.
type MaxRetriesExceeded struct {
	Attempts int
	Cause    error
}

func (e *MaxRetriesExceeded) Error() string {
	return fmt.Sprintf("max retries exceeded after %d attempt(s): %v", e.Attempts, e.Cause)
}
func (e *MaxRetriesExceeded) Unwrap() error { return e.Cause }

// NonRetryableError wraps an error that must not be retried (e.g. 400 Bad Request).
type NonRetryableError struct {
	Cause error
}

func (e *NonRetryableError) Error() string {
	return fmt.Sprintf("non-retryable error: %v", e.Cause)
}
func (e *NonRetryableError) Unwrap() error { return e.Cause }

// RateLimitError represents an HTTP 429 Too Many Requests response.
// RetryAfterSec holds the server-specified wait duration, or 0 if absent.
type RateLimitError struct {
	Message       string
	RetryAfterSec float64
}

func (e *RateLimitError) Error() string { return e.Message }

// CircuitBreakerOpenError is returned when the circuit breaker is OPEN.
type CircuitBreakerOpenError struct {
	Name              string
	RecoveryRemaining time.Duration
}

func (e *CircuitBreakerOpenError) Error() string {
	return fmt.Sprintf("circuit breaker '%s' is OPEN (recovery in %.0fs)",
		e.Name, e.RecoveryRemaining.Seconds())
}

// FallbackError records a single fallback level failure.
type FallbackError struct {
	Level     int
	LevelName string
	ErrType   string
	Message   string
}

// AllFallbacksExhausted is returned when every fallback level fails.
type AllFallbacksExhausted struct {
	OperationName string
	Errors        []FallbackError
	TotalTime     time.Duration
}

func (e *AllFallbacksExhausted) Error() string {
	return fmt.Sprintf("all %d fallback level(s) failed for '%s' in %.2fs",
		len(e.Errors), e.OperationName, e.TotalTime.Seconds())
}

// SystemUnavailableError is returned when the primary path and all fallbacks fail.
type SystemUnavailableError struct {
	Name           string
	PrimaryError   string
	FallbackErrors []FallbackError
}

func (e *SystemUnavailableError) Error() string {
	return fmt.Sprintf("'%s' is currently unavailable: primary error=%s; %d fallback(s) failed",
		e.Name, e.PrimaryError, len(e.FallbackErrors))
}

// ---------------------------------------------------------------------------
// Retry
// ---------------------------------------------------------------------------

// RetryConfig holds configuration for exponential-backoff retry behaviour.
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts (initial attempt not counted).
	MaxRetries int
	// BaseDelay is the initial delay before the first retry.
	BaseDelay time.Duration
	// MaxDelay is the upper cap on computed delay.
	MaxDelay time.Duration
	// BackoffMultiplier is the exponential growth factor.
	BackoffMultiplier float64
	// Jitter enables ±JitterFactor randomness.
	Jitter bool
	// JitterFactor is the fraction of computed delay to use as jitter range (0.1 = ±10%).
	JitterFactor float64
	// IsRetryable returns true for errors that may be retried.
	IsRetryable func(err error) bool
	// TotalDeadline is the hard wall-clock deadline; give up even if MaxRetries not reached.
	TotalDeadline time.Duration
}

// DefaultRetryConfig returns a RetryConfig with sensible defaults.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:        3,
		BaseDelay:         time.Second,
		MaxDelay:          60 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            true,
		JitterFactor:      0.1,
		IsRetryable: func(err error) bool {
			var rl *RateLimitError
			return errors.As(err, &rl) ||
				errors.Is(err, context.DeadlineExceeded)
		},
		TotalDeadline: 5 * time.Minute,
	}
}

// CalculateDelay returns the sleep duration before retry attempt *attempt*.
//
// Formula: base * multiplier**attempt (capped at MaxDelay), with optional ±jitter.
func CalculateDelay(attempt int, config RetryConfig) time.Duration {
	delay := float64(config.BaseDelay) * math.Pow(config.BackoffMultiplier, float64(attempt))
	if delay > float64(config.MaxDelay) {
		delay = float64(config.MaxDelay)
	}

	if config.Jitter {
		jitterRange := delay * config.JitterFactor
		delay += (rand.Float64()*2 - 1) * jitterRange
	}

	if delay < 0 {
		delay = 0
	}
	return time.Duration(delay)
}

// RetryWithBackoff executes operation with exponential-backoff retry.
//
// The context deadline, if set, is respected in addition to TotalDeadline.
func RetryWithBackoff(ctx context.Context, config RetryConfig, operation func(ctx context.Context) (any, error)) (any, error) {
	startTime := time.Now()
	var lastErr error

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		result, err := operation(ctx)
		if err == nil {
			return result, nil
		}
		lastErr = err

		// Check retryability.
		if !config.IsRetryable(err) {
			return nil, &NonRetryableError{Cause: err}
		}

		// Check total deadline.
		if time.Since(startTime) >= config.TotalDeadline {
			return nil, &MaxRetriesExceeded{Attempts: attempt + 1, Cause: err}
		}

		// No more retries.
		if attempt >= config.MaxRetries {
			return nil, &MaxRetriesExceeded{Attempts: attempt + 1, Cause: err}
		}

		delay := CalculateDelay(attempt, config)
		log.Printf("Retry attempt %d/%d failed: %v. Retrying in %.2fs…",
			attempt+1, config.MaxRetries+1, err, delay.Seconds())

		select {
		case <-time.After(delay):
			// continue
		case <-ctx.Done():
			return nil, &MaxRetriesExceeded{Attempts: attempt + 1, Cause: ctx.Err()}
		}
	}

	return nil, &MaxRetriesExceeded{Attempts: config.MaxRetries + 1, Cause: lastErr}
}

// RetryWithRateLimitAwareness is like RetryWithBackoff but uses the
// RetryAfterSec field of *RateLimitError when available.
func RetryWithRateLimitAwareness(ctx context.Context, config RetryConfig, operation func(ctx context.Context) (any, error)) (any, error) {
	startTime := time.Now()

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		result, err := operation(ctx)
		if err == nil {
			return result, nil
		}

		if time.Since(startTime) >= config.TotalDeadline || attempt >= config.MaxRetries {
			return nil, &MaxRetriesExceeded{Attempts: attempt + 1, Cause: err}
		}

		var rl *RateLimitError
		var delay time.Duration
		if errors.As(err, &rl) && rl.RetryAfterSec > 0 {
			delay = time.Duration(rl.RetryAfterSec * float64(time.Second))
			log.Printf("Rate limited (attempt %d). Waiting %.2fs (server-specified)…",
				attempt+1, delay.Seconds())
		} else if config.IsRetryable(err) {
			delay = CalculateDelay(attempt, config)
			log.Printf("Attempt %d failed: %v. Retrying in %.2fs (exponential backoff)…",
				attempt+1, err, delay.Seconds())
		} else {
			return nil, &NonRetryableError{Cause: err}
		}

		select {
		case <-time.After(delay):
			// continue
		case <-ctx.Done():
			return nil, &MaxRetriesExceeded{Attempts: attempt + 1, Cause: ctx.Err()}
		}
	}

	return nil, &MaxRetriesExceeded{Attempts: config.MaxRetries + 1, Cause: errors.New("all attempts exhausted")}
}

// ---------------------------------------------------------------------------
// Fallback
// ---------------------------------------------------------------------------

// FallbackLevel represents a single level in a fallback chain.
type FallbackLevel struct {
	// Name is a human-readable identifier.
	Name string
	// Provider is an async-style operation that returns (result, error).
	Provider func(ctx context.Context) (any, error)
	// Timeout is the maximum duration to wait for this level.
	Timeout time.Duration
	// Capability is "full" | "reduced" | "static".
	Capability string
	// CostMultiplier is the relative cost (1.0 = same as primary).
	CostMultiplier float64
}

// RLFallbackResult holds the successful result from the fallback chain.
type RLFallbackResult struct {
	Result     any
	LevelUsed  int
	LevelName  string
	Capability string
	Attempts   int
	TotalTime  time.Duration
	Errors     []FallbackError
}

// FallbackStats tracks aggregate fallback performance.
type FallbackStats struct {
	mu              sync.Mutex
	successByLevel  map[string]int
	failureByLevel  map[string]int
	failureByReason map[string]int
	exhaustionCount int
	totalOperations int
}

func newFallbackStats() *FallbackStats {
	return &FallbackStats{
		successByLevel:  make(map[string]int),
		failureByLevel:  make(map[string]int),
		failureByReason: make(map[string]int),
	}
}

func (s *FallbackStats) RecordSuccess(levelName string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.totalOperations++
	s.successByLevel[levelName]++
}

func (s *FallbackStats) RecordFailure(levelName, reason string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failureByLevel[levelName]++
	s.failureByReason[reason]++
}

func (s *FallbackStats) RecordExhaustion() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exhaustionCount++
	s.totalOperations++
}

// FallbackExecutor executes an operation through an ordered fallback chain.
type FallbackExecutor struct {
	Levels []*FallbackLevel
	Stats  *FallbackStats
}

// NewFallbackExecutor creates a FallbackExecutor with the given levels.
func NewFallbackExecutor(levels []*FallbackLevel) *FallbackExecutor {
	return &FallbackExecutor{
		Levels: levels,
		Stats:  newFallbackStats(),
	}
}

// Execute tries each level in order, returning the first success.
func (fe *FallbackExecutor) Execute(ctx context.Context, operationName string, operationFactory func(level *FallbackLevel) (any, error)) (*RLFallbackResult, error) {
	var errs []FallbackError
	start := time.Now()

	for i, level := range fe.Levels {
		log.Printf("Fallback [%s]: Trying level %d (%s, capability=%s)",
			operationName, i, level.Name, level.Capability)

		levelCtx, cancel := context.WithTimeout(ctx, level.Timeout)
		result, err := operationFactory(level)
		cancel()

		if err == nil {
			elapsed := time.Since(start)
			fe.Stats.RecordSuccess(level.Name)
			log.Printf("Fallback [%s]: Level %d (%s) succeeded in %.2fs",
				operationName, i, level.Name, elapsed.Seconds())

			return &RLFallbackResult{
				Result:     result,
				LevelUsed:  i,
				LevelName:  level.Name,
				Capability: level.Capability,
				Attempts:   len(errs) + 1,
				TotalTime:  elapsed,
				Errors:     errs,
			}, nil
		}

		_ = levelCtx
		msg := err.Error()
		if len(msg) > 200 {
			msg = msg[:200]
		}
		entry := FallbackError{
			Level:     i,
			LevelName: level.Name,
			ErrType:   fmt.Sprintf("%T", err),
			Message:   msg,
		}
		errs = append(errs, entry)
		fe.Stats.RecordFailure(level.Name, fmt.Sprintf("%T", err))

		log.Printf("Fallback [%s]: Level %d (%s) failed: %v", operationName, i, level.Name, err)
	}

	fe.Stats.RecordExhaustion()
	return nil, &AllFallbacksExhausted{
		OperationName: operationName,
		Errors:        errs,
		TotalTime:     time.Since(start),
	}
}

// ---------------------------------------------------------------------------
// Circuit Breaker
// ---------------------------------------------------------------------------

// CircuitState represents the circuit breaker state.
type CircuitBreakerState int

const (
	RLCircuitClosed   CircuitBreakerState = iota // Normal operation.
	RLCircuitOpen                                // Failing — requests rejected immediately.
	RLCircuitHalfOpen                            // Testing recovery — probe request allowed.
)

func (s CircuitBreakerState) String() string {
	switch s {
	case RLCircuitClosed:
		return "closed"
	case RLCircuitOpen:
		return "open"
	case RLCircuitHalfOpen:
		return "half_open"
	default:
		return "unknown"
	}
}

// CircuitBreakerStats holds observable circuit breaker metrics.
type CircuitBreakerStats struct {
	Name                   string
	State                  string
	TotalSuccesses         int
	TotalFailures          int
	TotalRejected          int
	TimesOpened            int
	RecentFailuresInWindow int
	FailureRate            float64
	SecondsInCurrentState  float64
}

// RLCircuitBreaker prevents calls to a persistently failing service.
//
// State machine:
//
//	CLOSED  ──(threshold failures in window)──▶ OPEN
//	OPEN    ──(recovery timeout elapsed)     ──▶ HALF_OPEN
//	HALF_OPEN ──(probe succeeds)             ──▶ CLOSED
//	HALF_OPEN ──(probe fails)                ──▶ OPEN
type RLCircuitBreaker struct {
	name                 string
	failureThreshold     int
	recoveryTimeout      time.Duration
	halfOpenMaxRequests  int
	failureWindowSeconds float64

	mu                sync.Mutex
	state             CircuitBreakerState
	failureTimestamps []time.Time
	halfOpenRequests  int
	lastStateChange   time.Time

	totalSuccesses int
	totalFailures  int
	totalRejected  int
	timesOpened    int
}

// RLNewCircuitBreaker creates a RLCircuitBreaker with the given parameters.
func RLNewCircuitBreaker(
	name string,
	failureThreshold int,
	recoveryTimeout time.Duration,
	halfOpenMaxRequests int,
	failureWindowSeconds float64,
) *RLCircuitBreaker {
	return &RLCircuitBreaker{
		name:                 name,
		failureThreshold:     failureThreshold,
		recoveryTimeout:      recoveryTimeout,
		halfOpenMaxRequests:  halfOpenMaxRequests,
		failureWindowSeconds: failureWindowSeconds,
		state:                RLCircuitClosed,
		lastStateChange:      time.Now(),
	}
}

// Call executes operation through the circuit breaker.
// It is safe for concurrent use.
func (cb *RLCircuitBreaker) Call(ctx context.Context, operation func(ctx context.Context) (any, error)) (any, error) {
	cb.mu.Lock()
	cb.maybeTransition()

	switch cb.state {
	case RLCircuitOpen:
		cb.totalRejected++
		remaining := cb.recoveryRemaining()
		cb.mu.Unlock()
		return nil, &CircuitBreakerOpenError{Name: cb.name, RecoveryRemaining: remaining}

	case RLCircuitHalfOpen:
		if cb.halfOpenRequests >= cb.halfOpenMaxRequests {
			cb.totalRejected++
			cb.mu.Unlock()
			return nil, &CircuitBreakerOpenError{
				Name:              cb.name,
				RecoveryRemaining: 0,
			}
		}
		cb.halfOpenRequests++
	}
	cb.mu.Unlock()

	result, err := operation(ctx)
	if err != nil {
		cb.OnFailure()
		return nil, err
	}
	cb.OnSuccess()
	return result, nil
}

// OnSuccess records a successful call and may close the circuit.
func (cb *RLCircuitBreaker) OnSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.totalSuccesses++
	if cb.state == RLCircuitHalfOpen {
		log.Printf("Circuit breaker '%s': probe succeeded — closing circuit.", cb.name)
		cb.transitionTo(RLCircuitClosed)
	}
}

// OnFailure records a failed call and may open or re-open the circuit.
func (cb *RLCircuitBreaker) OnFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.totalFailures++
	cb.failureTimestamps = append(cb.failureTimestamps, time.Now())
	recent := cb.currentWindowFailures()

	switch cb.state {
	case RLCircuitClosed:
		if len(recent) >= cb.failureThreshold {
			log.Printf("Circuit breaker '%s': %d failure(s) in %.0fs — opening.",
				cb.name, len(recent), cb.failureWindowSeconds)
			cb.transitionTo(RLCircuitOpen)
		}
	case RLCircuitHalfOpen:
		log.Printf("Circuit breaker '%s': probe failed — re-opening.", cb.name)
		cb.transitionTo(RLCircuitOpen)
	}
}

// GetStats returns a snapshot of circuit breaker metrics.
func (cb *RLCircuitBreaker) GetStats() CircuitBreakerStats {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return CircuitBreakerStats{
		Name:                   cb.name,
		State:                  cb.state.String(),
		TotalSuccesses:         cb.totalSuccesses,
		TotalFailures:          cb.totalFailures,
		TotalRejected:          cb.totalRejected,
		TimesOpened:            cb.timesOpened,
		RecentFailuresInWindow: len(cb.currentWindowFailures()),
		FailureRate: float64(cb.totalFailures) / math.Max(
			float64(cb.totalSuccesses+cb.totalFailures), 1,
		),
		SecondsInCurrentState: time.Since(cb.lastStateChange).Seconds(),
	}
}

// maybeTransition checks whether OPEN → HALF_OPEN is due.
// Must be called with cb.mu held.
func (cb *RLCircuitBreaker) maybeTransition() {
	if cb.state == RLCircuitOpen && time.Since(cb.lastStateChange) >= cb.recoveryTimeout {
		cb.transitionTo(RLCircuitHalfOpen)
		cb.halfOpenRequests = 0
	}
}

// currentWindowFailures returns timestamps within the rolling failure window.
// Must be called with cb.mu held.
func (cb *RLCircuitBreaker) currentWindowFailures() []time.Time {
	cutoff := time.Now().Add(-time.Duration(cb.failureWindowSeconds * float64(time.Second)))
	filtered := cb.failureTimestamps[:0]
	for _, t := range cb.failureTimestamps {
		if t.After(cutoff) {
			filtered = append(filtered, t)
		}
	}
	cb.failureTimestamps = filtered
	return filtered
}

// transitionTo moves to newState and logs the transition.
// Must be called with cb.mu held.
func (cb *RLCircuitBreaker) transitionTo(newState CircuitBreakerState) {
	oldState := cb.state
	cb.state = newState
	cb.lastStateChange = time.Now()
	if newState == RLCircuitOpen {
		cb.timesOpened++
	}
	log.Printf("Circuit breaker '%s': %s → %s", cb.name, oldState, newState)
}

// recoveryRemaining returns how long until the recovery timeout expires.
// Must be called with cb.mu held.
func (cb *RLCircuitBreaker) recoveryRemaining() time.Duration {
	elapsed := time.Since(cb.lastStateChange)
	if elapsed >= cb.recoveryTimeout {
		return 0
	}
	return cb.recoveryTimeout - elapsed
}

// ---------------------------------------------------------------------------
// ResilienceLayer
// ---------------------------------------------------------------------------

// ResilienceResult holds the outcome of a ResilienceLayer.Execute call.
type ResilienceResult struct {
	Result         any
	Path           string // "primary" | "fallback_level_N"
	Attempts       int
	TotalTime      time.Duration
	FallbackErrors []FallbackError
}

// ResilienceLayer combines RLCircuitBreaker + RetryWithBackoff + FallbackExecutor.
//
// Flow:
//  1. Circuit breaker check.
//  2. If closed/half-open: attempt with retry.
//  3. If retries exhausted: try fallback chain.
//  4. If circuit open: go directly to fallback.
//  5. If all fallbacks fail: return SystemUnavailableError.
type ResilienceLayer struct {
	Name             string
	RLCircuitBreaker   *RLCircuitBreaker
	RetryConfig      RetryConfig
	FallbackExecutor *FallbackExecutor
}

// Execute runs operation with full resilience protection.
func (rl *ResilienceLayer) Execute(ctx context.Context, operation func(ctx context.Context) (any, error)) (*ResilienceResult, error) {
	start := time.Now()
	var primaryErrStr string

	// Wrap the operation so the circuit breaker wraps the retry loop.
	wrapped := func(innerCtx context.Context) (any, error) {
		return RetryWithBackoff(innerCtx, rl.RetryConfig, operation)
	}

	result, err := rl.RLCircuitBreaker.Call(ctx, wrapped)
	if err == nil {
		elapsed := time.Since(start)
		log.Printf("Resilience [%s]: primary path succeeded in %.2fs", rl.Name, elapsed.Seconds())
		return &ResilienceResult{
			Result:    result,
			Path:      "primary",
			Attempts:  1,
			TotalTime: elapsed,
		}, nil
	}

	primaryErrStr = err.Error()
	log.Printf("Resilience [%s]: primary path failed (%v) — activating fallback chain.", rl.Name, err)

	fallbackResult, fallbackErr := rl.FallbackExecutor.Execute(
		ctx,
		rl.Name,
		func(level *FallbackLevel) (any, error) {
			levelCtx, cancel := context.WithTimeout(ctx, level.Timeout)
			defer cancel()
			return level.Provider(levelCtx)
		},
	)

	if fallbackErr == nil {
		elapsed := time.Since(start)
		return &ResilienceResult{
			Result:         fallbackResult.Result,
			Path:           fmt.Sprintf("fallback_level_%d", fallbackResult.LevelUsed),
			Attempts:       fallbackResult.Attempts,
			TotalTime:      elapsed,
			FallbackErrors: fallbackResult.Errors,
		}, nil
	}

	elapsed := time.Since(start)
	log.Printf("Resilience [%s]: all paths exhausted in %.2fs.", rl.Name, elapsed.Seconds())

	var afe *AllFallbacksExhausted
	if errors.As(fallbackErr, &afe) {
		return nil, &SystemUnavailableError{
			Name:           rl.Name,
			PrimaryError:   primaryErrStr,
			FallbackErrors: afe.Errors,
		}
	}
	return nil, fallbackErr
}

// ---------------------------------------------------------------------------
// ResilienceMonitor
// ---------------------------------------------------------------------------

// HealthReport contains a snapshot of resilience layer health.
type HealthReport struct {
	RLCircuitBreaker CircuitBreakerStats
	Fallback       map[string]int
	Alerts         []string
}

// ResilienceMonitor inspects a ResilienceLayer and generates alerts.
type ResilienceMonitor struct {
	Layer *ResilienceLayer

	PrimaryRateWarning   float64
	FallbackRateWarning  float64
	ExhaustionCritical   float64
	CircuitReopenWarning int
}

// NewResilienceMonitor creates a monitor with default alert thresholds.
func NewResilienceMonitor(layer *ResilienceLayer) *ResilienceMonitor {
	return &ResilienceMonitor{
		Layer:                layer,
		PrimaryRateWarning:   0.95,
		FallbackRateWarning:  0.10,
		ExhaustionCritical:   0.01,
		CircuitReopenWarning: 3,
	}
}

// CheckHealth returns a HealthReport for the resilience layer.
func (m *ResilienceMonitor) CheckHealth() HealthReport {
	cbStats := m.Layer.RLCircuitBreaker.GetStats()

	m.Layer.FallbackExecutor.Stats.mu.Lock()
	fbSuccess := make(map[string]int, len(m.Layer.FallbackExecutor.Stats.successByLevel))
	for k, v := range m.Layer.FallbackExecutor.Stats.successByLevel {
		fbSuccess[k] = v
	}
	total := m.Layer.FallbackExecutor.Stats.totalOperations
	exhaustion := m.Layer.FallbackExecutor.Stats.exhaustionCount
	m.Layer.FallbackExecutor.Stats.mu.Unlock()

	alerts := m.generateAlerts(cbStats, total, exhaustion, fbSuccess)
	return HealthReport{
		RLCircuitBreaker: cbStats,
		Fallback:       fbSuccess,
		Alerts:         alerts,
	}
}

func (m *ResilienceMonitor) generateAlerts(
	cb CircuitBreakerStats,
	total, exhaustion int,
	successByLevel map[string]int,
) []string {
	var alerts []string

	if cb.State == RLCircuitOpen.String() {
		alerts = append(alerts, fmt.Sprintf("CRITICAL: Circuit breaker '%s' is OPEN", cb.Name))
	}
	if cb.TimesOpened > m.CircuitReopenWarning {
		alerts = append(alerts, fmt.Sprintf(
			"WARNING: Circuit breaker '%s' has opened %d time(s)", cb.Name, cb.TimesOpened,
		))
	}

	safeTotal := math.Max(float64(total), 1)
	var firstLevel string
	for k := range successByLevel {
		firstLevel = k
		break
	}
	primarySuccesses := float64(successByLevel[firstLevel])
	primaryRate := primarySuccesses / safeTotal
	if primaryRate < m.PrimaryRateWarning {
		alerts = append(alerts, fmt.Sprintf(
			"WARNING: Primary success rate is %.1f%% (threshold: %.0f%%)",
			primaryRate*100, m.PrimaryRateWarning*100,
		))
	}

	exhaustionRate := float64(exhaustion) / safeTotal
	if exhaustionRate > m.ExhaustionCritical {
		alerts = append(alerts, fmt.Sprintf(
			"CRITICAL: Fallback exhaustion rate is %.1f%%", exhaustionRate*100,
		))
	}

	allSuccesses := 0
	for _, v := range successByLevel {
		allSuccesses += v
	}
	fallbackActivations := float64(allSuccesses) - primarySuccesses
	fallbackRate := fallbackActivations / safeTotal
	if fallbackRate > m.FallbackRateWarning {
		alerts = append(alerts, fmt.Sprintf(
			"WARNING: Fallback activated for %.1f%% of requests", fallbackRate*100,
		))
	}

	return alerts
}
