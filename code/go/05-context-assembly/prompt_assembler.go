// Prompt Assembler — dynamic prompt construction from templates, conditional
// sections, and multi-source context injection.
//
// Go port of code/python/05-context-assembly/prompt_assembler.py
//
// Same class-equivalent struct names (PromptAssembler), same method names
// (PascalCase for exported), same {variable} syntax, same conditional section
// evaluation via ConditionEngine, same YAML template format.
//
// See: docs/04-context-engineering/02-dynamic-prompt-assembly.md
package main

import (
	"fmt"
	"log"
	"regexp"
	"sort"
	"strconv"
	"strings"

	tiktoken "github.com/pkoukk/tiktoken-go"
)

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

func countTokensStr(text string) int {
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return len(strings.Fields(text)) // rough fallback
	}
	return len(enc.Encode(text, nil, nil))
}

func truncateToTokens(text string, maxTokens int) string {
	if countTokensStr(text) <= maxTokens {
		return text
	}
	words := strings.Fields(text)
	lo, hi := 0, len(words)
	for lo < hi {
		mid := (lo + hi) / 2
		if countTokensStr(strings.Join(words[:mid], " ")) <= maxTokens {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo <= 1 {
		return ""
	}
	return strings.Join(words[:lo-1], " ") + "…"
}

// ---------------------------------------------------------------------------
// Template variable regex
// ---------------------------------------------------------------------------

var varRE = regexp.MustCompile(`\{(\w+)\}`)

// fillTemplate replaces {key} placeholders with values from vars.
// Returns the filled string and the first missing key (if any).
func fillTemplate(template string, vars map[string]string) (string, string) {
	missing := ""
	result := varRE.ReplaceAllStringFunc(template, func(match string) string {
		key := match[1 : len(match)-1]
		if val, ok := vars[key]; ok {
			return val
		}
		if missing == "" {
			missing = key
		}
		return match // leave as-is
	})
	return result, missing
}

// ---------------------------------------------------------------------------
// Condition evaluation (inline — same logic as condition_engine.py)
// ---------------------------------------------------------------------------

func getNestedValue(variables map[string]any, key string) any {
	parts := strings.Split(key, ".")
	var obj any = variables
	for _, part := range parts {
		if obj == nil {
			return nil
		}
		if m, ok := obj.(map[string]any); ok {
			obj = m[part]
		} else {
			return nil
		}
	}
	return obj
}

// parseRhsValue tries to parse a raw string into a float64, bool, []any, or string.
func parseRhsValue(raw string) any {
	raw = strings.TrimSpace(raw)
	// JSON array
	if strings.HasPrefix(raw, "[") {
		var items []any
		raw = strings.Trim(raw, "[]")
		for _, part := range strings.Split(raw, ",") {
			part = strings.TrimSpace(part)
			part = strings.Trim(part, `'"`)
			items = append(items, part)
		}
		return items
	}
	// Quoted string
	if (strings.HasPrefix(raw, "'") && strings.HasSuffix(raw, "'")) ||
		(strings.HasPrefix(raw, `"`) && strings.HasSuffix(raw, `"`)) {
		return raw[1 : len(raw)-1]
	}
	// Number
	if f, err := strconv.ParseFloat(raw, 64); err == nil {
		return f
	}
	return raw
}

type atomResult struct {
	result bool
	err    error
}

func evaluateAtom(atomStr string, variables map[string]any) (bool, error) {
	atomStr = strings.TrimSpace(atomStr)

	type pattern struct {
		re     *regexp.Regexp
		opName string
	}

	patterns := []pattern{
		{regexp.MustCompile(`(?i)^([\w.]+)\s+not\s+in\s+(.+)$`), "not_in"},
		{regexp.MustCompile(`(?i)^([\w.]+)\s+not_in\s+(.+)$`), "not_in"},
		{regexp.MustCompile(`^([\w.]+)\s+>=(.+)$`), "gte"},
		{regexp.MustCompile(`^([\w.]+)\s+<=(.+)$`), "lte"},
		{regexp.MustCompile(`^([\w.]+)\s+==(.+)$`), "eq"},
		{regexp.MustCompile(`^([\w.]+)\s+!=(.+)$`), "neq"},
		{regexp.MustCompile(`^([\w.]+)\s+>(.+)$`), "gt"},
		{regexp.MustCompile(`^([\w.]+)\s+<(.+)$`), "lt"},
		{regexp.MustCompile(`(?i)^([\w.]+)\s+contains\s+(.+)$`), "contains"},
		{regexp.MustCompile(`(?i)^([\w.]+)\s+exists$`), "exists"},
		{regexp.MustCompile(`(?i)^([\w.]+)\s+in\s+(.+)$`), "in"},
	}

	for _, p := range patterns {
		m := p.re.FindStringSubmatch(atomStr)
		if m == nil {
			continue
		}
		key := m[1]
		left := getNestedValue(variables, key)

		if p.opName == "exists" {
			return left != nil, nil
		}
		rhs := parseRhsValue(m[2])

		switch p.opName {
		case "eq":
			return fmt.Sprintf("%v", left) == fmt.Sprintf("%v", rhs), nil
		case "neq":
			return fmt.Sprintf("%v", left) != fmt.Sprintf("%v", rhs), nil
		case "in":
			if items, ok := rhs.([]any); ok {
				for _, item := range items {
					if fmt.Sprintf("%v", left) == fmt.Sprintf("%v", item) {
						return true, nil
					}
				}
				return false, nil
			}
			return strings.Contains(fmt.Sprintf("%v", rhs), fmt.Sprintf("%v", left)), nil
		case "not_in":
			if items, ok := rhs.([]any); ok {
				for _, item := range items {
					if fmt.Sprintf("%v", left) == fmt.Sprintf("%v", item) {
						return false, nil
					}
				}
				return true, nil
			}
			return !strings.Contains(fmt.Sprintf("%v", rhs), fmt.Sprintf("%v", left)), nil
		case "gt":
			lf, _ := toFloat(left)
			rf, _ := toFloat(rhs)
			return lf > rf, nil
		case "lt":
			lf, _ := toFloat(left)
			rf, _ := toFloat(rhs)
			return lf < rf, nil
		case "gte":
			lf, _ := toFloat(left)
			rf, _ := toFloat(rhs)
			return lf >= rf, nil
		case "lte":
			lf, _ := toFloat(left)
			rf, _ := toFloat(rhs)
			return lf <= rf, nil
		case "contains":
			ls := fmt.Sprintf("%v", left)
			rs := fmt.Sprintf("%v", rhs)
			return strings.Contains(ls, rs), nil
		}
	}
	return false, fmt.Errorf("cannot parse condition atom: %q", atomStr)
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case string:
		f, err := strconv.ParseFloat(n, 64)
		return f, err == nil
	}
	return 0, false
}

var andRE = regexp.MustCompile(`(?i)\bAND\b`)
var orRE = regexp.MustCompile(`(?i)\bOR\b`)

// EvaluateCondition evaluates a DSL condition string against variables.
func EvaluateCondition(condition string, variables map[string]any) bool {
	condition = strings.TrimSpace(condition)
	for _, orBranch := range orRE.Split(condition, -1) {
		andAtoms := andRE.Split(orBranch, -1)
		allTrue := true
		for _, atom := range andAtoms {
			ok, err := evaluateAtom(strings.TrimSpace(atom), variables)
			if err != nil || !ok {
				allTrue = false
				break
			}
		}
		if allTrue {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// PromptSection
// ---------------------------------------------------------------------------

// PromptSection is a conditional block of text appended to the prompt.
type PromptSection struct {
	Name      string
	Content   string
	Condition func(variables map[string]any) bool
}

// ShouldInclude evaluates the section's condition.
func (s *PromptSection) ShouldInclude(variables map[string]any) bool {
	if s.Condition == nil {
		return false
	}
	return s.Condition(variables)
}

// ---------------------------------------------------------------------------
// ContextSource
// ---------------------------------------------------------------------------

// ContextSource holds a registered context source with its formatter.
type ContextSource struct {
	Name      string
	Formatter func(data any) string
	Priority  int
	MaxTokens int // 0 = unlimited
}

// ---------------------------------------------------------------------------
// Built-in formatters
// ---------------------------------------------------------------------------

// RagDocument is a retrieved document with text, score, and metadata.
type RagDocument struct {
	Text     string
	Score    float64
	Metadata map[string]string
}

// FormatRagResults formats a slice of retrieved documents.
func FormatRagResults(docs []RagDocument) string {
	if len(docs) == 0 {
		return "(no documents retrieved)"
	}
	var parts []string
	for i, doc := range docs {
		source := doc.Metadata["source"]
		if source == "" {
			source = "unknown"
		}
		parts = append(parts, fmt.Sprintf("[%d] Source: %s (relevance: %.0f%%)\n%s",
			i+1, source, doc.Score*100, doc.Text))
	}
	return strings.Join(parts, "\n\n---\n\n")
}

// UserProfile holds customer profile data for prompt injection.
type UserProfile struct {
	Name         string
	Plan         string
	Location     string
	Preferences  string
	MemberSince  string
	RecentOrders int
	OpenTickets  int
}

// FormatUserProfile formats a UserProfile.
func FormatUserProfile(p UserProfile) string {
	var lines []string
	if p.Name != "" {
		lines = append(lines, "Name: "+p.Name)
	}
	if p.Plan != "" {
		lines = append(lines, "Plan: "+p.Plan)
	}
	if p.Location != "" {
		lines = append(lines, "Location: "+p.Location)
	}
	if p.Preferences != "" {
		lines = append(lines, "Preferences: "+p.Preferences)
	}
	if p.MemberSince != "" {
		lines = append(lines, "Member since: "+p.MemberSince)
	}
	if len(lines) == 0 {
		return "(no profile data)"
	}
	return strings.Join(lines, "\n")
}

// PromptToolResult is the outcome of a tool invocation used in prompt context.
type PromptToolResult struct {
	ToolName string
	Success  bool
	Summary  string
}

// FormatToolResults formats a slice of tool results.
func FormatToolResults(results []PromptToolResult) string {
	if len(results) == 0 {
		return "(no tool results)"
	}
	var lines []string
	for _, r := range results {
		mark := "✓"
		if !r.Success {
			mark = "✗"
		}
		lines = append(lines, fmt.Sprintf("%s %s: %s", mark, r.ToolName, r.Summary))
	}
	return strings.Join(lines, "\n")
}

// FormatConversationSummary formats a conversation history summary.
func FormatConversationSummary(summary string) string {
	if summary == "" {
		return "(no conversation history)"
	}
	return "Previous conversation summary:\n" + summary
}

// FormatBusinessRules formats a slice of business rule strings.
func FormatBusinessRules(rules []string) string {
	if len(rules) == 0 {
		return "(no business rules)"
	}
	var lines []string
	for _, r := range rules {
		lines = append(lines, "- "+r)
	}
	return strings.Join(lines, "\n")
}

// ---------------------------------------------------------------------------
// PromptAssembler
// ---------------------------------------------------------------------------

// PromptAssembler assembles prompts from templates, conditional sections,
// and multi-source context injection.
type PromptAssembler struct {
	baseTemplates    map[string]string
	sections         map[string]*PromptSection
	sourceFormatters map[string]*ContextSource
	// Preserve insertion order for deterministic section ordering
	sectionOrder []string
}

// NewPromptAssembler creates a new PromptAssembler.
func NewPromptAssembler() *PromptAssembler {
	return &PromptAssembler{
		baseTemplates:    make(map[string]string),
		sections:         make(map[string]*PromptSection),
		sourceFormatters: make(map[string]*ContextSource),
	}
}

// RegisterTemplate registers a base template with {placeholder} variables.
func (a *PromptAssembler) RegisterTemplate(name, template string) {
	a.baseTemplates[name] = template
}

// RegisterSection registers a conditional prompt section.
func (a *PromptAssembler) RegisterSection(
	name, content string,
	condition func(variables map[string]any) bool,
) {
	if _, exists := a.sections[name]; !exists {
		a.sectionOrder = append(a.sectionOrder, name)
	}
	a.sections[name] = &PromptSection{
		Name:      name,
		Content:   content,
		Condition: condition,
	}
}

// RegisterSourceFormatter registers a context source with its formatter.
// priority: higher = included first.
// maxTokens: 0 = unlimited.
func (a *PromptAssembler) RegisterSourceFormatter(
	name string,
	formatter func(data any) string,
	priority int,
	maxTokens int,
) {
	a.sourceFormatters[name] = &ContextSource{
		Name:      name,
		Formatter: formatter,
		Priority:  priority,
		MaxTokens: maxTokens,
	}
}

// Assemble builds a complete prompt from the template, active sections,
// and formatted context sources.
func (a *PromptAssembler) Assemble(
	templateName string,
	variables map[string]any,
	contextSources map[string]any,
) (string, error) {
	template, ok := a.baseTemplates[templateName]
	if !ok {
		return "", fmt.Errorf("template %q not registered", templateName)
	}

	// 2. Evaluate conditional sections (in registration order)
	var activeSections []*PromptSection
	for _, name := range a.sectionOrder {
		s := a.sections[name]
		if s.ShouldInclude(variables) {
			activeSections = append(activeSections, s)
		}
	}

	// 3+4. Format and sort context sources
	type fmtSource struct {
		priority int
		name     string
		text     string
	}
	var formatted []fmtSource
	for srcName, data := range contextSources {
		source, found := a.sourceFormatters[srcName]
		if !found {
			continue
		}
		text := source.Formatter(data)
		if source.MaxTokens > 0 && countTokensStr(text) > source.MaxTokens {
			text = truncateToTokens(text, source.MaxTokens)
		}
		formatted = append(formatted, fmtSource{source.Priority, srcName, text})
	}
	sort.Slice(formatted, func(i, j int) bool {
		return formatted[i].priority > formatted[j].priority
	})

	// 5. Build blocks
	var contextParts []string
	var sourceNames []string
	for _, f := range formatted {
		contextParts = append(contextParts, "## "+f.name+"\n"+f.text)
		sourceNames = append(sourceNames, f.name)
	}
	contextBlock := strings.Join(contextParts, "\n\n")

	var sectionContents []string
	var sectionNames []string
	for _, s := range activeSections {
		sectionContents = append(sectionContents, s.Content)
		sectionNames = append(sectionNames, s.Name)
	}
	sectionsBlock := strings.Join(sectionContents, "\n\n")

	// 6. Fill template
	strVars := map[string]string{
		"context":  contextBlock,
		"sections": sectionsBlock,
	}
	for k, v := range variables {
		strVars[k] = fmt.Sprintf("%v", v)
	}
	result, missing := fillTemplate(template, strVars)
	if missing != "" {
		return "", fmt.Errorf("template variable '{%s}' not provided", missing)
	}

	// Append blocks that have no placeholder
	if !strings.Contains(template, "{sections}") && sectionsBlock != "" {
		result = strings.TrimRight(result, "\n") + "\n\n" + sectionsBlock
	}
	if !strings.Contains(template, "{context}") && contextBlock != "" {
		result = strings.TrimRight(result, "\n") + "\n\n" + contextBlock
	}

	// 7. Log
	log.Printf("Assembled prompt | template=%s | sections=[%s] | sources=[%s] | tokens=%d",
		templateName,
		strings.Join(sectionNames, ", "),
		strings.Join(sourceNames, ", "),
		countTokensStr(result),
	)

	return result, nil
}

// AssembleWithBudget assembles a prompt while enforcing a total token budget.
// Low-priority context sources are dropped first.
func (a *PromptAssembler) AssembleWithBudget(
	templateName string,
	variables map[string]any,
	contextSources map[string]any,
	maxTokens int,
) (string, error) {
	result, err := a.Assemble(templateName, variables, contextSources)
	if err != nil {
		return "", err
	}
	if countTokensStr(result) <= maxTokens {
		return result, nil
	}

	// Sort sources ascending by priority (drop lowest first)
	type namedPriority struct {
		name     string
		priority int
	}
	var order []namedPriority
	for name := range contextSources {
		p := 0
		if src, ok := a.sourceFormatters[name]; ok {
			p = src.Priority
		}
		order = append(order, namedPriority{name, p})
	}
	sort.Slice(order, func(i, j int) bool {
		return order[i].priority < order[j].priority
	})

	remaining := make(map[string]any)
	for k, v := range contextSources {
		remaining[k] = v
	}

	for _, item := range order {
		delete(remaining, item.name)
		log.Printf("Budget exceeded (%d tokens) — dropping source %q", maxTokens, item.name)
		result, err = a.Assemble(templateName, variables, remaining)
		if err != nil {
			return "", err
		}
		if countTokensStr(result) <= maxTokens {
			return result, nil
		}
	}
	return result, nil
}

// GetAvailableVariables returns all {variable} names used in the template.
func (a *PromptAssembler) GetAvailableVariables(templateName string) ([]string, error) {
	template, ok := a.baseTemplates[templateName]
	if !ok {
		return nil, fmt.Errorf("template %q not registered", templateName)
	}
	seen := make(map[string]bool)
	var result []string
	for _, m := range varRE.FindAllStringSubmatch(template, -1) {
		name := m[1]
		if !seen[name] {
			seen[name] = true
			result = append(result, name)
		}
	}
	return result, nil
}
