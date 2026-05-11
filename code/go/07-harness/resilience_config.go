// resilience_config.go — Resilience configuration and circuit breaker settings (Go port).
//
// See: docs/07-harness-engineering/01-the-harness-mindset.md
package main

import (
	"fmt"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

// CircuitState represents the current state of a circuit breaker.
type CircuitState string

const (
	CircuitClosed   CircuitState = "closed"   // normal operation
	CircuitOpen     CircuitState = "open"      // failing, reject all calls
	CircuitHalfOpen CircuitState = "half_open" // testing recovery
)

// CircuitBreakerConfig holds tuneable circuit breaker parameters.
type CircuitBreakerConfig struct {
	FailureThreshold  int           // consecutive failures before opening
	RecoveryTimeout   time.Duration // how long to stay open before half-open
	SuccessThreshold  int           // successes in half-open before closing
}

// DefaultCircuitBreakerConfig returns sensible defaults.
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		FailureThreshold: 5,
		RecoveryTimeout:  30 * time.Second,
		SuccessThreshold: 2,
	}
}

// CircuitBreaker implements the circuit breaker pattern for LLM calls.
type CircuitBreaker struct {
	mu               sync.Mutex
	config           CircuitBreakerConfig
	state            CircuitState
	consecutiveFails int
	consecutivePasses int
	openedAt         time.Time
}

// NewCircuitBreaker creates a closed circuit breaker.
func NewCircuitBreaker(cfg CircuitBreakerConfig) *CircuitBreaker {
	return &CircuitBreaker{config: cfg, state: CircuitClosed}
}

// Allow returns true if the call should proceed.
func (cb *CircuitBreaker) Allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case CircuitClosed:
		return true
	case CircuitOpen:
		if time.Since(cb.openedAt) >= cb.config.RecoveryTimeout {
			cb.state = CircuitHalfOpen
			cb.consecutivePasses = 0
			return true
		}
		return false
	case CircuitHalfOpen:
		return true
	}
	return false
}

// RecordSuccess records a successful call.
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.consecutiveFails = 0
	if cb.state == CircuitHalfOpen {
		cb.consecutivePasses++
		if cb.consecutivePasses >= cb.config.SuccessThreshold {
			cb.state = CircuitClosed
		}
	}
}

// RecordFailure records a failed call.
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.consecutiveFails++
	cb.consecutivePasses = 0
	if cb.consecutiveFails >= cb.config.FailureThreshold {
		cb.state = CircuitOpen
		cb.openedAt = time.Now()
	}
}

// State returns the current circuit state.
func (cb *CircuitBreaker) State() CircuitState { return cb.state }

// ResilienceConfig wraps all resilience settings together.
type ResilienceConfig struct {
	CircuitBreaker  CircuitBreakerConfig
	RetryMaxAttempts int
	RetryBackoff     time.Duration
	TimeoutPerCall   time.Duration
}

// DefaultResilienceConfig returns production-ready defaults.
func DefaultResilienceConfig() ResilienceConfig {
	return ResilienceConfig{
		CircuitBreaker:  DefaultCircuitBreakerConfig(),
		RetryMaxAttempts: 3,
		RetryBackoff:     500 * time.Millisecond,
		TimeoutPerCall:   30 * time.Second,
	}
}

// RunResilienceConfigDemo demonstrates the circuit breaker.
func RunResilienceConfigDemo() {
	cb := NewCircuitBreaker(CircuitBreakerConfig{
		FailureThreshold:  3,
		RecoveryTimeout:   5 * time.Second,
		SuccessThreshold:  2,
	})

	for i := 0; i < 5; i++ {
		if cb.Allow() {
			fmt.Printf("Call %d: allowed (state=%s)\n", i+1, cb.State())
			cb.RecordFailure()
		} else {
			fmt.Printf("Call %d: rejected (state=%s)\n", i+1, cb.State())
		}
	}
}
