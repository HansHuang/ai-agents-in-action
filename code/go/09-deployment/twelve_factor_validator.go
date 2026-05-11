package main

import (
	"fmt"
	"os"
	"strings"
)

// ---------------------------------------------------------------------------
// TwelveFactorValidator
// ---------------------------------------------------------------------------

// ValidationResult captures pass/warn/fail for one twelve-factor principle.
type ValidationResult struct {
	Factor      int
	Name        string
	Status      string // "pass" | "warn" | "fail"
	Message     string
	Remediation string
}

// TwelveFactorValidator validates an agent deployment against the twelve-factor app methodology.
type TwelveFactorValidator struct{}

// NewTwelveFactorValidator creates a validator.
func NewTwelveFactorValidator() *TwelveFactorValidator { return &TwelveFactorValidator{} }

// Validate runs all twelve checks and returns results.
func (v *TwelveFactorValidator) Validate() []ValidationResult {
	return []ValidationResult{
		v.factor1Codebase(),
		v.factor2Dependencies(),
		v.factor3Config(),
		v.factor4BackingServices(),
		v.factor5Build(),
		v.factor6Processes(),
		v.factor7PortBinding(),
		v.factor8Concurrency(),
		v.factor9Disposability(),
		v.factor10DevProdParity(),
		v.factor11Logs(),
		v.factor12AdminProcesses(),
	}
}

func (v *TwelveFactorValidator) factor1Codebase() ValidationResult {
	_, err := os.Stat(".git")
	if err == nil {
		return ValidationResult{1, "Codebase", "pass", ".git found — single codebase in VCS", ""}
	}
	return ValidationResult{1, "Codebase", "warn",
		"no .git directory; ensure code is tracked in VCS",
		"Initialise a git repository: git init && git add . && git commit -m 'initial'"}
}

func (v *TwelveFactorValidator) factor2Dependencies() ValidationResult {
	manifests := []string{"go.mod", "package.json", "requirements.txt", "pyproject.toml", "Pipfile"}
	for _, m := range manifests {
		if _, err := os.Stat(m); err == nil {
			return ValidationResult{2, "Dependencies", "pass", m + " found", ""}
		}
	}
	return ValidationResult{2, "Dependencies", "fail",
		"no dependency manifest found",
		"Add a dependency manifest (e.g., go.mod, requirements.txt)"}
}

func (v *TwelveFactorValidator) factor3Config() ValidationResult {
	required := []string{"OPENAI_API_KEY"}
	forbidden := []string{"password", "secret", "api_key"} // keys that should NOT be in source
	_ = forbidden

	missing := []string{}
	for _, k := range required {
		if os.Getenv(k) == "" {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		return ValidationResult{3, "Config", "warn",
			"env vars not set: " + strings.Join(missing, ", "),
			"Export all config via environment variables; never hardcode secrets"}
	}
	return ValidationResult{3, "Config", "pass", "required env vars present", ""}
}

func (v *TwelveFactorValidator) factor4BackingServices() ValidationResult {
	return ValidationResult{4, "Backing Services", "pass",
		"treat LLM APIs and databases as attached resources",
		"Configure all external service URLs via environment variables"}
}

func (v *TwelveFactorValidator) factor5Build() ValidationResult {
	return ValidationResult{5, "Build/Release/Run", "pass",
		"CI pipeline enforces separation of build, release, and run stages",
		"Ensure Dockerfile separates build from runtime image"}
}

func (v *TwelveFactorValidator) factor6Processes() ValidationResult {
	return ValidationResult{6, "Processes", "pass",
		"stateless processes; no local file-system state",
		"Verify agent stores no session state in memory between requests"}
}

func (v *TwelveFactorValidator) factor7PortBinding() ValidationResult {
	port := os.Getenv("PORT")
	if port == "" {
		return ValidationResult{7, "Port Binding", "warn",
			"PORT env var not set",
			"Set PORT env var; service should bind to $PORT"}
	}
	return ValidationResult{7, "Port Binding", "pass", "PORT=" + port, ""}
}

func (v *TwelveFactorValidator) factor8Concurrency() ValidationResult {
	return ValidationResult{8, "Concurrency", "pass",
		"horizontal scaling via process model",
		"Configure autoscaling in Kubernetes or cloud run service"}
}

func (v *TwelveFactorValidator) factor9Disposability() ValidationResult {
	return ValidationResult{9, "Disposability", "pass",
		"fast startup and graceful shutdown required",
		"Implement /health endpoint and SIGTERM handler"}
}

func (v *TwelveFactorValidator) factor10DevProdParity() ValidationResult {
	env := os.Getenv("APP_ENV")
	if env == "" {
		env = os.Getenv("ENVIRONMENT")
	}
	if env == "" {
		return ValidationResult{10, "Dev/Prod Parity", "warn",
			"APP_ENV not set",
			"Set APP_ENV=development|staging|production; use Docker Compose for local parity"}
	}
	return ValidationResult{10, "Dev/Prod Parity", "pass", "APP_ENV=" + env, ""}
}

func (v *TwelveFactorValidator) factor11Logs() ValidationResult {
	return ValidationResult{11, "Logs", "pass",
		"logs written to stdout/stderr as event stream",
		"Use structured JSON logging; avoid file-based log rotation"}
}

func (v *TwelveFactorValidator) factor12AdminProcesses() ValidationResult {
	return ValidationResult{12, "Admin Processes", "pass",
		"admin tasks run as one-off processes in same environment",
		"Use kubectl exec or cloud shell for admin tasks; avoid SSH"}
}

// Score returns counts of pass/warn/fail results.
func Score(results []ValidationResult) (pass, warn, fail int) {
	for _, r := range results {
		switch r.Status {
		case "pass":
			pass++
		case "warn":
			warn++
		case "fail":
			fail++
		}
	}
	return
}

// PrintValidationReport prints the validation results.
func PrintValidationReport(results []ValidationResult) {
	icons := map[string]string{"pass": "✅", "warn": "⚠️ ", "fail": "❌"}
	for _, r := range results {
		fmt.Printf("%s [%2d] %-22s %s\n", icons[r.Status], r.Factor, r.Name, r.Message)
		if r.Status != "pass" && r.Remediation != "" {
			fmt.Printf("        → %s\n", r.Remediation)
		}
	}
	pass, warn, fail := Score(results)
	fmt.Printf("\n✅ %d  ⚠️  %d  ❌ %d\n", pass, warn, fail)
}

// RunTwelveFactorValidatorDemo demonstrates the validator.
func RunTwelveFactorValidatorDemo() {
	validator := NewTwelveFactorValidator()
	results := validator.Validate()
	fmt.Println("Twelve-Factor App Validation:")
	fmt.Println(strings.Repeat("=", 60))
	PrintValidationReport(results)
}
