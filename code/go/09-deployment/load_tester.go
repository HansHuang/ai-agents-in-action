package main

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// ---------------------------------------------------------------------------
// LoadTester
// ---------------------------------------------------------------------------

// LoadTestResult holds the aggregated results of a load test.
type LoadTestResult struct {
	TotalRequests  int
	Successful     int
	Failed         int
	AvgLatencyMs   float64
	P50LatencyMs   float64
	P95LatencyMs   float64
	P99LatencyMs   float64
	MaxLatencyMs   float64
	RequestsPerSec float64
	ErrorRate      float64
	Duration       time.Duration
}

// LoadTestConfig parameterises the load test.
type LoadTestConfig struct {
	Concurrency  int           // concurrent virtual users
	Duration     time.Duration // how long to run
	RampUpPeriod time.Duration // linear ramp-up time
	TargetRPS    int           // optional target requests/sec (0 = unlimited)
}

// DefaultLoadTestConfig returns a conservative load test config.
func DefaultLoadTestConfig() LoadTestConfig {
	return LoadTestConfig{
		Concurrency:  5,
		Duration:     10 * time.Second,
		RampUpPeriod: 2 * time.Second,
		TargetRPS:    10,
	}
}

// LoadTester runs a configurable load test against a target function.
type LoadTester struct {
	cfg    LoadTestConfig
	target func() error // function under test
}

// NewLoadTester creates a tester with the given config and target.
func NewLoadTester(cfg LoadTestConfig, target func() error) *LoadTester {
	return &LoadTester{cfg: cfg, target: target}
}

// Run executes the load test and returns aggregated results.
func (lt *LoadTester) Run() LoadTestResult {
	latencies := make([]int64, 0, 1000)
	var mu sync.Mutex
	var successCount, failCount int64

	start := time.Now()
	deadline := start.Add(lt.cfg.Duration)

	var wg sync.WaitGroup
	for i := 0; i < lt.cfg.Concurrency; i++ {
		// Ramp up: stagger start times
		rampDelay := time.Duration(float64(lt.cfg.RampUpPeriod) * float64(i) / float64(lt.cfg.Concurrency))
		wg.Add(1)
		go func(workerID int, delay time.Duration) {
			defer wg.Done()
			time.Sleep(delay)
			for time.Now().Before(deadline) {
				t0 := time.Now()
				err := lt.target()
				latencyMs := time.Since(t0).Milliseconds()
				mu.Lock()
				latencies = append(latencies, latencyMs)
				mu.Unlock()
				if err != nil {
					atomic.AddInt64(&failCount, 1)
				} else {
					atomic.AddInt64(&successCount, 1)
				}
				// Throttle if targetRPS is set
				if lt.cfg.TargetRPS > 0 {
					sleepDuration := time.Second/time.Duration(lt.cfg.TargetRPS) - time.Since(t0)
					if sleepDuration > 0 {
						time.Sleep(sleepDuration)
					}
				}
			}
		}(i, rampDelay)
	}
	wg.Wait()

	elapsed := time.Since(start)
	total := int(successCount + failCount)

	result := LoadTestResult{
		TotalRequests: total,
		Successful:    int(successCount),
		Failed:        int(failCount),
		Duration:      elapsed,
	}
	if total > 0 {
		result.ErrorRate = float64(failCount) / float64(total)
		result.RequestsPerSec = float64(total) / elapsed.Seconds()
		result.AvgLatencyMs = calcMean(latencies)
		result.P50LatencyMs = calcPercentile(latencies, 50)
		result.P95LatencyMs = calcPercentile(latencies, 95)
		result.P99LatencyMs = calcPercentile(latencies, 99)
		result.MaxLatencyMs = calcMax(latencies)
	}
	return result
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func calcMean(vals []int64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum int64
	for _, v := range vals {
		sum += v
	}
	return float64(sum) / float64(len(vals))
}

func calcPercentile(vals []int64, pct int) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]int64, len(vals))
	copy(sorted, vals)
	// insertion sort (fine for typical load test sizes)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	idx := int(float64(pct)/100.0*float64(len(sorted)-1) + 0.5)
	return float64(sorted[idx])
}

func calcMax(vals []int64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var mx int64
	for _, v := range vals {
		if v > mx {
			mx = v
		}
	}
	return float64(mx)
}

// PrintLoadTestReport prints a formatted report.
func PrintLoadTestReport(r LoadTestResult) {
	fmt.Println("Load Test Results:")
	fmt.Printf("  Duration       : %v\n", r.Duration.Round(time.Millisecond))
	fmt.Printf("  Total Requests : %d\n", r.TotalRequests)
	fmt.Printf("  Successful     : %d\n", r.Successful)
	fmt.Printf("  Failed         : %d\n", r.Failed)
	fmt.Printf("  Error Rate     : %.1f%%\n", r.ErrorRate*100)
	fmt.Printf("  RPS            : %.1f\n", r.RequestsPerSec)
	fmt.Printf("  Avg Latency    : %.1f ms\n", r.AvgLatencyMs)
	fmt.Printf("  p50 Latency    : %.1f ms\n", r.P50LatencyMs)
	fmt.Printf("  p95 Latency    : %.1f ms\n", r.P95LatencyMs)
	fmt.Printf("  p99 Latency    : %.1f ms\n", r.P99LatencyMs)
	fmt.Printf("  Max Latency    : %.1f ms\n", r.MaxLatencyMs)
}

// RunLoadTesterDemo demonstrates the load tester with a mock target.
func RunLoadTesterDemo() {
	cfg := LoadTestConfig{
		Concurrency:  3,
		Duration:     2 * time.Second,
		RampUpPeriod: 500 * time.Millisecond,
		TargetRPS:    20,
	}

	// Mock target: 10% error rate, 10-100ms latency
	mockTarget := func() error {
		time.Sleep(time.Duration(10+rand.Intn(90)) * time.Millisecond)
		if rand.Float64() < 0.10 {
			return fmt.Errorf("simulated error")
		}
		return nil
	}

	tester := NewLoadTester(cfg, mockTarget)
	fmt.Println("Running load test (2s)...")
	result := tester.Run()
	PrintLoadTestReport(result)
}
