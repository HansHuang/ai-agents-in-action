// tool_validator.go — Tool definition quality validator (Go port).
//
// Checks tool definitions against documented best practices and produces a
// quality score (0–100) with actionable warnings and fix suggestions.
//
// See docs/02-the-agent-loop/02-tool-design-patterns.md
package main

import (
	"fmt"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

var verbPrefixes = []string{
	"get_", "search_", "create_", "update_", "delete_", "generate_",
	"list_", "fetch_", "send_", "check_", "validate_", "calculate_",
	"convert_", "format_", "submit_", "cancel_", "approve_", "reject_",
}

var vagueNames = map[string]bool{
	"data": true, "input": true, "output": true, "param": true, "value": true,
	"info": true, "result": true, "thing": true, "item": true, "object": true,
	"payload": true,
}

var returnWords = []string{
	"return", "returns", "gives", "provides", "fetches", "retrieves",
	"outputs", "yields", "contains", "responds with",
}

var exampleIndicators = []string{
	"e.g.", "eg ", "ex:", "example:", "for example", "like ", "such as", "for instance",
}

var snakeCaseRe = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

// ToolWarning describes a single validation issue.
type ToolWarning struct {
	Category   string // "naming" | "description" | "parameters"
	Message    string
	Suggestion string
	Penalty    int // points deducted from 100
}

// ToolValidationResult is the outcome for a single tool.
type ToolValidationResult struct {
	ToolName string
	Score    int
	Warnings []ToolWarning
}

// Passed returns true when the score meets the passing threshold (70).
func (r ToolValidationResult) Passed() bool { return r.Score >= 70 }

// Summary returns a human-readable multi-line summary.
func (r ToolValidationResult) Summary() string {
	grade := "✓ PASS"
	if !r.Passed() {
		grade = "✗ FAIL"
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "  Tool: '%s'  Score: %d/100  %s\n", r.ToolName, r.Score, grade)
	for _, w := range r.Warnings {
		fmt.Fprintf(&sb, "  [%s] %s\n", strings.ToUpper(w.Category), w.Message)
		fmt.Fprintf(&sb, "    → Fix: %s\n", w.Suggestion)
	}
	if len(r.Warnings) == 0 {
		sb.WriteString("  No issues found.\n")
	}
	return sb.String()
}

// ToolValidator validates OpenAI-format tool definitions.
type ToolValidator struct{}

// NewToolValidator returns a new ToolValidator.
func NewToolValidator() *ToolValidator { return &ToolValidator{} }

// ValidateTool validates a single tool definition and returns its result.
// The tool should be in OpenAI function-calling format.
func (v *ToolValidator) ValidateTool(tool map[string]interface{}) ToolValidationResult {
	fn, _ := tool["function"].(map[string]interface{})
	name, _ := fn["name"].(string)
	description, _ := fn["description"].(string)
	params, _ := fn["parameters"].(map[string]interface{})

	var warnings []ToolWarning
	penalty := 0

	// ── Naming checks ──────────────────────────────────────────────────────
	if !snakeCaseRe.MatchString(name) {
		w := ToolWarning{"naming", fmt.Sprintf("'%s' is not snake_case", name),
			"Use lowercase letters, digits, and underscores only.", 10}
		warnings = append(warnings, w)
		penalty += w.Penalty
	}

	hasVerb := false
	for _, p := range verbPrefixes {
		if strings.HasPrefix(name, p) {
			hasVerb = true
			break
		}
	}
	if !hasVerb {
		w := ToolWarning{"naming", fmt.Sprintf("'%s' does not start with a recognised verb prefix", name),
			fmt.Sprintf("Use one of: %s", strings.Join(verbPrefixes[:8], ", ")), 10}
		warnings = append(warnings, w)
		penalty += w.Penalty
	}

	if len(name) > 64 {
		w := ToolWarning{"naming", "Tool name exceeds 64 characters", "Shorten the name.", 5}
		warnings = append(warnings, w)
		penalty += w.Penalty
	}

	// ── Description checks ─────────────────────────────────────────────────
	if len(description) < 20 {
		w := ToolWarning{"description", "Description is too short (< 20 chars)",
			"Add a clear description of what the tool does and returns.", 15}
		warnings = append(warnings, w)
		penalty += w.Penalty
	}

	hasReturnWord := false
	descLower := strings.ToLower(description)
	for _, rw := range returnWords {
		if strings.Contains(descLower, rw) {
			hasReturnWord = true
			break
		}
	}
	if !hasReturnWord {
		w := ToolWarning{"description", "Description doesn't mention what the tool returns",
			"Add a sentence like 'Returns ...' describing the output.", 10}
		warnings = append(warnings, w)
		penalty += w.Penalty
	}

	// ── Parameter checks ───────────────────────────────────────────────────
	if properties, ok := params["properties"].(map[string]interface{}); ok {
		for pname, praw := range properties {
			prop, _ := praw.(map[string]interface{})
			pdesc, _ := prop["description"].(string)

			if vagueNames[pname] {
				w := ToolWarning{"parameters", fmt.Sprintf("Parameter name '%s' is too vague", pname),
					"Use a specific descriptive name, e.g. 'city_name' instead of 'data'.", 5}
				warnings = append(warnings, w)
				penalty += w.Penalty
			}

			if len(pdesc) == 0 {
				w := ToolWarning{"parameters", fmt.Sprintf("Parameter '%s' has no description", pname),
					"Add a description explaining the parameter's purpose and format.", 10}
				warnings = append(warnings, w)
				penalty += w.Penalty
			} else {
				hasExample := false
				pdescLower := strings.ToLower(pdesc)
				for _, ei := range exampleIndicators {
					if strings.Contains(pdescLower, ei) {
						hasExample = true
						break
					}
				}
				if !hasExample {
					w := ToolWarning{"parameters", fmt.Sprintf("Parameter '%s' description lacks an example value", pname),
						fmt.Sprintf("Add an example, e.g. \"e.g. 'value'\""), 5}
					warnings = append(warnings, w)
					penalty += w.Penalty
				}
			}
		}
	}

	score := 100 - penalty
	if score < 0 {
		score = 0
	}
	return ToolValidationResult{ToolName: name, Score: score, Warnings: warnings}
}

// ValidateSet validates a set of tool definitions and prints a report.
func (v *ToolValidator) ValidateSet(tools []map[string]interface{}) []ToolValidationResult {
	results := make([]ToolValidationResult, 0, len(tools))
	for _, tool := range tools {
		results = append(results, v.ValidateTool(tool))
	}
	return results
}

// PrintReport prints a validation summary for a set of results.
func PrintValidationReport(results []ToolValidationResult) {
	passed, failed := 0, 0
	for _, r := range results {
		fmt.Println(r.Summary())
		if r.Passed() {
			passed++
		} else {
			failed++
		}
	}
	fmt.Printf("\n=== Summary: %d/%d tools passed ===\n", passed, len(results))
	if failed > 0 {
		fmt.Printf("  %d tool(s) need improvement.\n", failed)
	}
}
