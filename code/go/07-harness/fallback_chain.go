// fallback_chain.go — Cascading fallback chain for LLM provider failures (Go port).
//
// See: docs/07-harness-engineering/02-input-guardrails-and-validation.md
package main

import (
	"context"
	"fmt"
	"time"
)

// ---------------------------------------------------------------------------
// Fallback types
// ---------------------------------------------------------------------------

// FallbackProvider is a single provider in the fallback chain.
type FallbackProvider struct {
	Name       string
	Model      string
	Priority   int // lower = higher priority
	MaxRetries int
	TimeoutMs  int
}

// FallbackResult holds the outcome of a fallback chain attempt.
type FallbackResult struct {
	ProviderUsed string
	Answer       string
	Attempts     int
	TotalElapsed time.Duration
	Error        error
}

// FallbackChain tries providers in priority order until one succeeds.
type FallbackChain struct {
	providers []FallbackProvider
	call      func(ctx context.Context, provider FallbackProvider, prompt string) (string, error)
}

// NewFallbackChain creates a fallback chain with a caller function.
func NewFallbackChain(
	providers []FallbackProvider,
	call func(ctx context.Context, provider FallbackProvider, prompt string) (string, error),
) *FallbackChain {
	// Sort by priority (simple insertion sort)
	sorted := make([]FallbackProvider, len(providers))
	copy(sorted, providers)
	for i := range sorted {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].Priority < sorted[i].Priority {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	return &FallbackChain{providers: sorted, call: call}
}

// Execute tries each provider in order, retrying up to MaxRetries times.
func (c *FallbackChain) Execute(ctx context.Context, prompt string) FallbackResult {
	start := time.Now()
	attempts := 0

	for _, provider := range c.providers {
		for retry := 0; retry <= provider.MaxRetries; retry++ {
			attempts++
			answer, err := c.call(ctx, provider, prompt)
			if err == nil {
				return FallbackResult{
					ProviderUsed: provider.Name,
					Answer:       answer,
					Attempts:     attempts,
					TotalElapsed: time.Since(start),
				}
			}
			if retry < provider.MaxRetries {
				fmt.Printf("[Fallback] %s attempt %d failed: %v — retrying\n", provider.Name, retry+1, err)
			}
		}
		fmt.Printf("[Fallback] %s exhausted, trying next provider\n", provider.Name)
	}

	return FallbackResult{
		Attempts:     attempts,
		TotalElapsed: time.Since(start),
		Error:        fmt.Errorf("all providers exhausted after %d attempts", attempts),
	}
}

// DefaultFallbackProviders returns a sensible default provider chain.
func DefaultFallbackProviders() []FallbackProvider {
	return []FallbackProvider{
		{Name: "gpt-4o", Model: "gpt-4o", Priority: 1, MaxRetries: 2, TimeoutMs: 30000},
		{Name: "gpt-4o-mini", Model: "gpt-4o-mini", Priority: 2, MaxRetries: 3, TimeoutMs: 15000},
		{Name: "gpt-3.5-turbo", Model: "gpt-3.5-turbo", Priority: 3, MaxRetries: 3, TimeoutMs: 10000},
	}
}

// RunFallbackChainDemo demonstrates the fallback chain.
func RunFallbackChainDemo() {
	callCount := 0
	chain := NewFallbackChain(DefaultFallbackProviders(), func(ctx context.Context, p FallbackProvider, prompt string) (string, error) {
		callCount++
		if p.Priority <= 2 && callCount <= 4 {
			return "", fmt.Errorf("simulated failure for %s", p.Name)
		}
		return fmt.Sprintf("[%s] Answer to: %s", p.Name, prompt), nil
	})

	result := chain.Execute(context.Background(), "What is 2+2?")
	if result.Error != nil {
		fmt.Printf("Fallback chain failed: %v\n", result.Error)
	} else {
		fmt.Printf("Provider: %s, Attempts: %d, Elapsed: %v\n",
			result.ProviderUsed, result.Attempts, result.TotalElapsed)
		fmt.Printf("Answer: %s\n", result.Answer)
	}
}
