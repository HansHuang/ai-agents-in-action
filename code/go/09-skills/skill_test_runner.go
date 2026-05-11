package skills

import (
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// TestReport
// ---------------------------------------------------------------------------

// TestReport aggregates the result of running one skill's test suite.
type TestReport struct {
	SkillName       string
	TotalTests      int
	Passed          int
	Failed          int
	Failures        []FailureDetail
	ExecutionTimeMs int64
}

// FailureDetail holds information about one failed test case.
type FailureDetail struct {
	TestInput string
	Reason    string
}

// String returns a CI-friendly one-line (or multi-line) summary.
func (r TestReport) String() string {
	icon := "PASS"
	if r.Failed > 0 {
		icon = "FAIL"
	}
	line := fmt.Sprintf("[%s] %s: %d/%d passed (%d ms)",
		icon, r.SkillName, r.Passed, r.TotalTests, r.ExecutionTimeMs)
	for _, f := range r.Failures {
		line += fmt.Sprintf("\n       - input=%q: %s", f.TestInput, f.Reason)
	}
	return line
}

// ---------------------------------------------------------------------------
// SkillTestRunner
// ---------------------------------------------------------------------------

// SkillTestRunner runs all test cases for skills in a registry.
// No LLM. No API keys. No agent loop.
type SkillTestRunner struct {
	registry *SkillRegistry
}

// NewSkillTestRunner creates a runner backed by the given registry.
func NewSkillTestRunner(registry *SkillRegistry) *SkillTestRunner {
	return &SkillTestRunner{registry: registry}
}

// RunSkill runs all test cases for a single registered skill.
func (r *SkillTestRunner) RunSkill(skillName string) (TestReport, error) {
	skill, err := r.registry.Get(skillName)
	if err != nil {
		return TestReport{}, err
	}

	start := time.Now()
	results := skill.RunTests()
	elapsed := time.Since(start).Milliseconds()

	report := TestReport{
		SkillName:       skillName,
		TotalTests:      len(results),
		ExecutionTimeMs: elapsed,
	}
	for _, res := range results {
		if res.Passed {
			report.Passed++
		} else {
			report.Failed++
			report.Failures = append(report.Failures, FailureDetail{
				TestInput: fmt.Sprintf("%v", res.TestInput),
				Reason:    res.Reason,
			})
		}
	}
	return report, nil
}

// RunAll runs tests for every skill in the registry.
func (r *SkillTestRunner) RunAll() []TestReport {
	var reports []TestReport
	for name := range r.registry.skills {
		report, err := r.RunSkill(name)
		if err != nil {
			// Record a single-failure report for unresolvable skills
			reports = append(reports, TestReport{
				SkillName:  name,
				TotalTests: 1,
				Failed:     1,
				Failures:   []FailureDetail{{Reason: err.Error()}},
			})
			continue
		}
		reports = append(reports, report)
	}
	return reports
}

// RunIntegrationTest tests multiple skills working in sequence.
// scenario keys:
//   - "steps": []map[string]interface{} each with "skill" (string) and "input" (Params)
//   - "expect_final_output_contains": []string
func (r *SkillTestRunner) RunIntegrationTest(skillNames []string, scenario map[string]interface{}) bool {
	steps, _ := scenario["steps"].([]map[string]interface{})
	expectContains, _ := scenario["expect_final_output_contains"].([]string)
	var lastResult *SkillResult

	for _, step := range steps {
		name, _ := step["skill"].(string)
		input, _ := step["input"].(Params)
		if input == nil {
			input = Params{}
		}
		result, err := r.registry.Execute(name, input)
		if err != nil || !result.Success {
			errMsg := ""
			if err != nil {
				errMsg = err.Error()
			} else {
				errMsg = result.Error
			}
			fmt.Printf("Integration step failed: skill=%s error=%s\n", name, errMsg)
			return false
		}
		lastResult = result
	}

	if len(expectContains) == 0 {
		return true
	}
	finalStr := ""
	if lastResult != nil {
		finalStr = fmt.Sprintf("%v", lastResult.Data)
	}
	for _, keyword := range expectContains {
		if !strings.Contains(finalStr, keyword) {
			fmt.Printf("Integration test: %q not found in output\n", keyword)
			return false
		}
	}
	return true
}

// PrintReports prints all test reports and returns the total failure count.
func PrintReports(reports []TestReport) int {
	totalFailed := 0
	for _, r := range reports {
		fmt.Println(r.String())
		totalFailed += r.Failed
	}
	return totalFailed
}

// RunSkillTestRunnerDemo demonstrates the runner with a trivial echo skill.
func RunSkillTestRunnerDemo() {
	registry := NewSkillRegistry()

	// Register a simple echo skill with test cases
	echo := NewSkill(
		"echo",
		"Returns its input unchanged",
		func(p Params) (Params, error) {
			return Params{"output": p["message"]}, nil
		},
		map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"message": map[string]interface{}{"type": "string"},
			},
			"required": []string{"message"},
		},
	)
	echo.TestCases = []SkillTest{
		{Input: Params{"message": "hello"}, ExpectSuccess: true, ExpectOutputContains: []string{"hello"}},
		{Input: Params{"message": "world"}, ExpectSuccess: true, ExpectOutputContains: []string{"world"}},
	}
	_ = registry.Register(echo)

	runner := NewSkillTestRunner(registry)
	fmt.Println("=== Running all skill tests ===")
	reports := runner.RunAll()
	failed := PrintReports(reports)

	fmt.Println()
	if failed > 0 {
		fmt.Printf("%d test(s) failed.\n", failed)
	} else {
		fmt.Println("All tests passed.")
	}
}
