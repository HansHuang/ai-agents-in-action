// Condition Engine — evaluate DSL conditions for dynamic prompt sections.
//
// Go port of code/python/05-context-assembly/condition_engine.py
//
// Supports a simple, safe condition DSL for including or excluding prompt
// sections based on runtime variables. No eval() is used; conditions are
// parsed with regex-based tokenisation.
//
// Simple conditions:
//
//	"plan == 'premium'"
//	"user.plan in ['premium', 'enterprise']"
//	"sentiment_score > 0.7"
//	"user.email contains '@enterprise'"
//	"conversation_history exists"
//
// Compound conditions:
//
//	"plan == 'premium' AND country == 'US'"
//	"country == 'DE' OR country == 'FR'"
//
// See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
package main

import (
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// Operator patterns — order matters: longer / multi-word first
// ---------------------------------------------------------------------------

type opPattern struct {
	re   *regexp.Regexp
	name string
}

var condOpPatterns = []opPattern{
	{regexp.MustCompile(`(?i)^([\w.]+)\s+not\s+in\s+(.+)$`), "not_in"},
	{regexp.MustCompile(`(?i)^([\w.]+)\s+not_in\s+(.+)$`), "not_in"},
	{regexp.MustCompile(`^([\w.]+)\s+>=\s+(.+)$`), "gte"},
	{regexp.MustCompile(`^([\w.]+)\s+<=\s+(.+)$`), "lte"},
	{regexp.MustCompile(`^([\w.]+)\s+==\s+(.+)$`), "eq"},
	{regexp.MustCompile(`^([\w.]+)\s+!=\s+(.+)$`), "neq"},
	{regexp.MustCompile(`^([\w.]+)\s+>\s+(.+)$`), "gt"},
	{regexp.MustCompile(`^([\w.]+)\s+<\s+(.+)$`), "lt"},
	{regexp.MustCompile(`(?i)^([\w.]+)\s+contains\s+(.+)$`), "contains"},
	{regexp.MustCompile(`(?i)^([\w.]+)\s+exists$`), "exists"},
	{regexp.MustCompile(`(?i)^([\w.]+)\s+in\s+(.+)$`), "in"},
}

var (
	andSplitter = regexp.MustCompile(`\bAND\b`)
	orSplitter  = regexp.MustCompile(`\bOR\b`)
	listItemRe  = regexp.MustCompile(`'([^']*)'|"([^"]*)"|(\S+)`)
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// condGetNestedValue resolves a dotted key path from a nested map[string]any.
// "user.plan" → variables["user"]["plan"]
func condGetNestedValue(variables map[string]any, key string) any {
	parts := strings.Split(key, ".")
	var current any = variables
	for _, part := range parts {
		switch v := current.(type) {
		case map[string]any:
			current = v[part]
		default:
			return nil
		}
		if current == nil {
			return nil
		}
	}
	return current
}

// parseRHS parses the right-hand side of a condition into a typed value.
// Handles: 'string', "string", numbers, booleans, and ['a','b','c'] lists.
func parseRHS(raw string) any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}

	// Single-quoted string
	if strings.HasPrefix(raw, "'") && strings.HasSuffix(raw, "'") {
		return raw[1 : len(raw)-1]
	}
	// Double-quoted string
	if strings.HasPrefix(raw, "\"") && strings.HasSuffix(raw, "\"") {
		return raw[1 : len(raw)-1]
	}
	// List literal: ['a', 'b'] or ["a", "b"]
	if strings.HasPrefix(raw, "[") && strings.HasSuffix(raw, "]") {
		inner := raw[1 : len(raw)-1]
		var list []any
		for _, m := range listItemRe.FindAllStringSubmatch(inner, -1) {
			if m[1] != "" {
				list = append(list, m[1])
			} else if m[2] != "" {
				list = append(list, m[2])
			} else if m[3] != "" {
				list = append(list, m[3])
			}
		}
		return list
	}
	// Boolean
	if raw == "true" {
		return true
	}
	if raw == "false" {
		return false
	}
	// Integer
	if i, err := strconv.ParseInt(raw, 10, 64); err == nil {
		return float64(i)
	}
	// Float
	if f, err := strconv.ParseFloat(raw, 64); err == nil {
		return f
	}
	return raw
}

// parseAtom parses one atomic condition string into (key, opName, value).
func parseAtom(atomStr string) (key, opName string, value any, err error) {
	atomStr = strings.TrimSpace(atomStr)
	for _, p := range condOpPatterns {
		m := p.re.FindStringSubmatch(atomStr)
		if m == nil {
			continue
		}
		key = m[1]
		opName = p.name
		if len(m) > 2 {
			value = parseRHS(m[2])
		}
		return key, opName, value, nil
	}
	return "", "", nil, fmt.Errorf("cannot parse condition atom: %q", atomStr)
}

// condEvaluateAtom evaluates one atomic condition against variables.
func condEvaluateAtom(key, op string, value any, variables map[string]any) (bool, error) {
	left := condGetNestedValue(variables, key)

	switch op {
	case "exists":
		return left != nil, nil
	case "eq":
		return condCompareValues(left, value) == 0, nil
	case "neq":
		return condCompareValues(left, value) != 0, nil
	case "gt":
		cmp, err := condNumericCompare(left, value)
		return err == nil && cmp > 0, err
	case "lt":
		cmp, err := condNumericCompare(left, value)
		return err == nil && cmp < 0, err
	case "gte":
		cmp, err := condNumericCompare(left, value)
		return err == nil && cmp >= 0, err
	case "lte":
		cmp, err := condNumericCompare(left, value)
		return err == nil && cmp <= 0, err
	case "in":
		return condValueInList(left, value), nil
	case "not_in":
		return !condValueInList(left, value), nil
	case "contains":
		ls, ok1 := condToString(left)
		rs, ok2 := condToString(value)
		if !ok1 || !ok2 {
			return false, nil
		}
		return strings.Contains(ls, rs), nil
	}
	return false, fmt.Errorf("unknown operator: %q", op)
}

func condToString(v any) (string, bool) {
	switch s := v.(type) {
	case string:
		return s, true
	}
	return "", false
}

func condToFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case string:
		if f, err := strconv.ParseFloat(n, 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

func condNumericCompare(a, b any) (int, error) {
	fa, okA := condToFloat(a)
	fb, okB := condToFloat(b)
	if !okA || !okB {
		return 0, fmt.Errorf("cannot compare %T and %T numerically", a, b)
	}
	switch {
	case fa < fb:
		return -1, nil
	case fa > fb:
		return 1, nil
	default:
		return 0, nil
	}
}

// condCompareValues returns 0 if equal, non-zero otherwise (string or numeric).
func condCompareValues(a, b any) int {
	if cmp, err := condNumericCompare(a, b); err == nil {
		return cmp
	}
	sa, _ := condToString(a)
	sb, _ := condToString(b)
	if sa == sb {
		return 0
	}
	return -1
}

func condValueInList(value, list any) bool {
	items, ok := list.([]any)
	if !ok {
		return false
	}
	for _, item := range items {
		if condCompareValues(value, item) == 0 {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// ConditionEngine
// ---------------------------------------------------------------------------

// ConditionEngine evaluates conditions for dynamic prompt sections.
// Uses a simple, safe DSL — no eval() is invoked.
type ConditionEngine struct{}

// NewConditionEngine creates a new ConditionEngine.
func NewConditionEngine() *ConditionEngine {
	return &ConditionEngine{}
}

// Evaluate evaluates a condition string against variables.
//
// Supported formats:
//   - "query_type == 'billing'"
//   - "user.plan in ['premium', 'enterprise']"
//   - "user.country not_in ['US', 'CA']"
//   - "sentiment_score > 0.7"
//   - "user.email contains '@enterprise'"
//   - "conversation_history exists"
//   - "plan == 'premium' AND country == 'US'"
//   - "country == 'DE' OR country == 'FR'"
func (ce *ConditionEngine) Evaluate(condition string, variables map[string]any) bool {
	condition = strings.TrimSpace(condition)
	// OR has lower precedence than AND.
	for _, orBranch := range orSplitter.Split(condition, -1) {
		andAtoms := andSplitter.Split(orBranch, -1)
		branchTrue := true
		for _, atom := range andAtoms {
			atom = strings.TrimSpace(atom)
			key, op, value, err := parseAtom(atom)
			if err != nil {
				log.Printf("condition_engine: parse error in %q: %v", condition, err)
				branchTrue = false
				break
			}
			result, err := condEvaluateAtom(key, op, value, variables)
			if err != nil {
				log.Printf("condition_engine: eval error in %q: %v", condition, err)
				branchTrue = false
				break
			}
			if !result {
				branchTrue = false
				break
			}
		}
		if branchTrue {
			return true
		}
	}
	return false
}

// EvaluateAll evaluates multiple named conditions and returns those that are true.
func (ce *ConditionEngine) EvaluateAll(conditions map[string]string, variables map[string]any) []string {
	var matched []string
	for name, cond := range conditions {
		if ce.Evaluate(cond, variables) {
			matched = append(matched, name)
		}
	}
	return matched
}

// ValidateCondition validates a condition string for syntax errors.
// Returns (true, "") if valid; (false, errorMessage) if not.
func (ce *ConditionEngine) ValidateCondition(condition string) (bool, string) {
	for _, orBranch := range orSplitter.Split(condition, -1) {
		for _, atom := range andSplitter.Split(orBranch, -1) {
			_, _, _, err := parseAtom(strings.TrimSpace(atom))
			if err != nil {
				return false, err.Error()
			}
		}
	}
	return true, ""
}

// Explain returns a human-readable explanation of why a condition evaluated to true or false.
func (ce *ConditionEngine) Explain(condition string, variables map[string]any) string {
	var lines []string
	lines = append(lines, fmt.Sprintf("Evaluating: %q", condition))

	orBranches := orSplitter.Split(condition, -1)
	var branchResults []bool

	for bIdx, orBranch := range orBranches {
		orBranch = strings.TrimSpace(orBranch)
		andAtoms := andSplitter.Split(orBranch, -1)

		if len(orBranches) > 1 {
			lines = append(lines, fmt.Sprintf("\n  [OR branch %d] %q", bIdx+1, orBranch))
		}

		var atomResults []bool
		for _, atomStr := range andAtoms {
			atomStr = strings.TrimSpace(atomStr)
			key, op, value, err := parseAtom(atomStr)
			if err != nil {
				atomResults = append(atomResults, false)
				lines = append(lines, fmt.Sprintf("    %q  →  ERROR: %v", atomStr, err))
				continue
			}
			left := condGetNestedValue(variables, key)
			result, err := condEvaluateAtom(key, op, value, variables)
			if err != nil {
				atomResults = append(atomResults, false)
				lines = append(lines, fmt.Sprintf("    %q  →  ERROR: %v", atomStr, err))
				continue
			}
			atomResults = append(atomResults, result)
			check := "✓"
			if !result {
				check = "✗"
			}
			lines = append(lines, fmt.Sprintf("    %-40q  →  %s=%v  %s  %v  [%s]",
				atomStr, key, left, op, value, check))
		}

		branchOk := len(atomResults) > 0
		for _, r := range atomResults {
			if !r {
				branchOk = false
				break
			}
		}
		branchResults = append(branchResults, branchOk)
		if len(orBranches) > 1 {
			status := "True ✓"
			if !branchOk {
				status = "False ✗"
			}
			lines = append(lines, fmt.Sprintf("    Branch result: %s", status))
		}
	}

	final := false
	for _, r := range branchResults {
		if r {
			final = true
			break
		}
	}
	overall := "True ✓"
	if !final {
		overall = "False ✗"
	}
	lines = append(lines, fmt.Sprintf("\nOverall: %s", overall))
	return strings.Join(lines, "\n")
}
