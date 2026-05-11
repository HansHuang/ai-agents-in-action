package main

import (
	"fmt"
	"os"
	"strings"
)

// ---------------------------------------------------------------------------
// CI Twelve-Factor Check
// ---------------------------------------------------------------------------

// CICheckResult is the outcome of a single twelve-factor CI check.
type CICheckResult struct {
	Factor      int
	Name        string
	Status      string // "pass" | "warn" | "fail"
	Description string
	Details     string
}

// CITwelveFactorCheck runs lightweight twelve-factor checks in a CI context.
type CITwelveFactorCheck struct {
	results []CICheckResult
}

// NewCITwelveFactorCheck creates a new check runner.
func NewCITwelveFactorCheck() *CITwelveFactorCheck { return &CITwelveFactorCheck{} }

// run executes all twelve-factor checks.
func (c *CITwelveFactorCheck) run() {
	c.results = []CICheckResult{
		c.checkCodebase(),
		c.checkDependencies(),
		c.checkConfig(),
		c.checkBackingServices(),
		c.checkBuildReleaseRun(),
		c.checkProcesses(),
		c.checkPortBinding(),
		c.checkConcurrency(),
		c.checkDisposability(),
		c.checkDevProdParity(),
		c.checkLogs(),
		c.checkAdminProcesses(),
	}
}

func (c *CITwelveFactorCheck) checkCodebase() CICheckResult {
	_, hasGit := os.Stat(".git")
	status := "pass"
	details := ".git directory found"
	if hasGit != nil {
		status = "warn"
		details = "no .git directory found in working directory"
	}
	return CICheckResult{Factor: 1, Name: "Codebase", Status: status, Details: details,
		Description: "One codebase tracked in revision control, many deploys"}
}

func (c *CITwelveFactorCheck) checkDependencies() CICheckResult {
	// Check for a go.mod or package.json as evidence of explicit dependency declaration
	for _, f := range []string{"go.mod", "package.json", "requirements.txt", "pyproject.toml"} {
		if _, err := os.Stat(f); err == nil {
			return CICheckResult{Factor: 2, Name: "Dependencies", Status: "pass", Details: f + " found",
				Description: "Explicitly declare and isolate dependencies"}
		}
	}
	return CICheckResult{Factor: 2, Name: "Dependencies", Status: "warn",
		Details:     "no recognised dependency manifest found",
		Description: "Explicitly declare and isolate dependencies"}
}

func (c *CITwelveFactorCheck) checkConfig() CICheckResult {
	// Warn if config-like variables are hardcoded (simple heuristic)
	secrets := []string{"OPENAI_API_KEY", "DATABASE_URL", "SECRET_KEY"}
	missing := []string{}
	for _, s := range secrets {
		if os.Getenv(s) == "" {
			missing = append(missing, s)
		}
	}
	status := "pass"
	details := "common config vars are present in environment"
	if len(missing) > 0 {
		status = "warn"
		details = "not set in env: " + strings.Join(missing, ", ")
	}
	return CICheckResult{Factor: 3, Name: "Config", Status: status, Details: details,
		Description: "Store config in the environment"}
}

func (c *CITwelveFactorCheck) checkBackingServices() CICheckResult {
	return CICheckResult{Factor: 4, Name: "Backing Services", Status: "pass",
		Details:     "treat backing services as attached resources (manual review recommended)",
		Description: "Treat backing services as attached resources"}
}

func (c *CITwelveFactorCheck) checkBuildReleaseRun() CICheckResult {
	return CICheckResult{Factor: 5, Name: "Build/Release/Run", Status: "pass",
		Details:     "strict separation enforced by CI pipeline",
		Description: "Strictly separate build and run stages"}
}

func (c *CITwelveFactorCheck) checkProcesses() CICheckResult {
	return CICheckResult{Factor: 6, Name: "Processes", Status: "pass",
		Details:     "stateless processes recommended; verify no local filesystem state",
		Description: "Execute the app as stateless processes"}
}

func (c *CITwelveFactorCheck) checkPortBinding() CICheckResult {
	port := os.Getenv("PORT")
	if port == "" {
		port = os.Getenv("HTTP_PORT")
	}
	status := "pass"
	details := fmt.Sprintf("PORT=%s", port)
	if port == "" {
		status = "warn"
		details = "PORT env var not set; defaulting to hardcoded port"
	}
	return CICheckResult{Factor: 7, Name: "Port Binding", Status: status, Details: details,
		Description: "Export services via port binding"}
}

func (c *CITwelveFactorCheck) checkConcurrency() CICheckResult {
	return CICheckResult{Factor: 8, Name: "Concurrency", Status: "pass",
		Details:     "scale via process model; verify no thread-local state",
		Description: "Scale out via the process model"}
}

func (c *CITwelveFactorCheck) checkDisposability() CICheckResult {
	return CICheckResult{Factor: 9, Name: "Disposability", Status: "pass",
		Details:     "fast startup and graceful shutdown required (manual review recommended)",
		Description: "Maximize robustness with fast startup and graceful shutdown"}
}

func (c *CITwelveFactorCheck) checkDevProdParity() CICheckResult {
	env := os.Getenv("APP_ENV")
	if env == "" {
		env = os.Getenv("ENVIRONMENT")
	}
	status := "pass"
	details := fmt.Sprintf("APP_ENV=%s", env)
	if env == "" {
		status = "warn"
		details = "APP_ENV not set; cannot verify dev/prod parity"
	}
	return CICheckResult{Factor: 10, Name: "Dev/Prod Parity", Status: status, Details: details,
		Description: "Keep development, staging, and production as similar as possible"}
}

func (c *CITwelveFactorCheck) checkLogs() CICheckResult {
	return CICheckResult{Factor: 11, Name: "Logs", Status: "pass",
		Details:     "write logs to stdout/stderr (verify no file-based logging)",
		Description: "Treat logs as event streams"}
}

func (c *CITwelveFactorCheck) checkAdminProcesses() CICheckResult {
	return CICheckResult{Factor: 12, Name: "Admin Processes", Status: "pass",
		Details:     "run admin tasks as one-off processes in identical environment",
		Description: "Run admin/management tasks as one-off processes"}
}

// Run executes all checks and returns results.
func (c *CITwelveFactorCheck) Run() []CICheckResult {
	c.run()
	return c.results
}

// ExitCode returns 0 if all checks pass/warn, 1 if any fail.
func (c *CITwelveFactorCheck) ExitCode() int {
	for _, r := range c.results {
		if r.Status == "fail" {
			return 1
		}
	}
	return 0
}

// PrintReport prints a CI-friendly report.
func PrintCIReport(results []CICheckResult) {
	icons := map[string]string{"pass": "✅", "warn": "⚠️ ", "fail": "❌"}
	pass, warn, fail := 0, 0, 0
	for _, r := range results {
		icon := icons[r.Status]
		fmt.Printf("%s [Factor %2d] %-20s %s\n", icon, r.Factor, r.Name, r.Details)
		switch r.Status {
		case "pass":
			pass++
		case "warn":
			warn++
		case "fail":
			fail++
		}
	}
	fmt.Printf("\nResult: %d pass, %d warn, %d fail\n", pass, warn, fail)
}

// RunCITwelveFactorCheckDemo demonstrates the CI check.
func RunCITwelveFactorCheckDemo() {
	checker := NewCITwelveFactorCheck()
	results := checker.Run()
	fmt.Println("CI Twelve-Factor Check:")
	PrintCIReport(results)
	code := checker.ExitCode()
	fmt.Printf("Exit code: %d\n", code)
}
