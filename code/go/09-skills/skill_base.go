// skill_base.go — Skill base class and registry for composable agent capabilities.
//
// Go port of code/python/09-skills/skill_base.py
//
// A Skill bundles a tool with:
//   - Input validation   (runs before the tool)
//   - Output normalisation  (runs after the tool)
//   - A fallback  (runs when the tool returns an error)
//   - A prompt fragment  (injected into the agent system prompt)
//   - Test cases  (runnable without an LLM or API key)
//
// See: docs/02-the-agent-loop/05-skills-composing-capabilities.md

package skills

import (
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Function types
// ---------------------------------------------------------------------------

// Params is the type for tool input/output maps.
type Params map[string]interface{}

// ToolFunc is the function a Skill wraps.
type ToolFunc func(params Params) (Params, error)

// ValidatorFunc validates and optionally corrects params before the tool runs.
// Returns the (potentially modified) params or an error.
type ValidatorFunc func(params Params) (Params, error)

// NormalizerFunc normalises the raw tool output.
type NormalizerFunc func(raw Params) (Params, error)

// FallbackFunc is called when the tool returns an error.
// Returns a user-facing message string.
type FallbackFunc func(params Params, err error) string

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// SkillInputError is returned by a ValidatorFunc when parameters are invalid.
type SkillInputError struct {
	Message    string
	Suggestion string
	FixAction  string
}

func (e *SkillInputError) Error() string { return e.Message }

// CircularDependencyError is returned when a dependency cycle is detected.
type CircularDependencyError struct{ Msg string }

func (e *CircularDependencyError) Error() string { return e.Msg }

// MissingDependencyError is returned when a declared dependency is not registered.
type MissingDependencyError struct{ Msg string }

func (e *MissingDependencyError) Error() string { return e.Msg }

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

// SkillResult holds the outcome of executing a skill.
type SkillResult struct {
	Success         bool
	Data            Params
	Error           string
	ErrorType       string // "invalid_input" | "unavailable" | "internal"
	Suggestion      string
	ExecutionTimeMs int64
}

// SkillTest defines a single test case for a skill.
type SkillTest struct {
	Input                Params
	ExpectSuccess        bool
	ExpectOutputContains []string
	ExpectFallback       bool
}

// TestResult holds the outcome of running a single SkillTest.
type TestResult struct {
	TestInput Params
	Passed    bool
	Reason    string
	Result    *SkillResult
}

// ---------------------------------------------------------------------------
// OpenAI schema helper
// ---------------------------------------------------------------------------

// OpenAIFunction is the nested struct in the function-calling schema.
type OpenAIFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// OpenAISchema is the full function-calling schema passed to the LLM.
type OpenAISchema struct {
	Type     string         `json:"type"` // always "function"
	Function OpenAIFunction `json:"function"`
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

// Skill is a composable, testable unit of agent capability.
type Skill struct {
	Name             string
	Description      string
	Tool             ToolFunc
	Parameters       map[string]interface{}
	Version          string
	Tags             []string
	PromptFragment   string
	InputValidator   ValidatorFunc
	OutputNormalizer NormalizerFunc
	Fallback         FallbackFunc
	Dependencies     []string
	TestCases        []SkillTest
}

// NewSkill creates a Skill with sensible defaults.
func NewSkill(name, description string, tool ToolFunc, parameters map[string]interface{}) *Skill {
	return &Skill{
		Name:        name,
		Description: description,
		Tool:        tool,
		Parameters:  parameters,
		Version:     "1.0.0",
	}
}

// Execute runs the full validate → tool → normalise → fallback pipeline.
//
// Returns a SkillResult; never returns an error unless no fallback is defined
// and the tool fails (in that case the error is returned directly).
func (s *Skill) Execute(params Params) (*SkillResult, error) {
	start := time.Now()

	// 1. Validate input
	p := params
	if s.InputValidator != nil {
		var err error
		p, err = s.InputValidator(p)
		if err != nil {
			elapsed := time.Since(start).Milliseconds()
			if e, ok := err.(*SkillInputError); ok {
				return &SkillResult{
					Success:         false,
					Error:           e.Message,
					ErrorType:       "invalid_input",
					Suggestion:      e.Suggestion,
					ExecutionTimeMs: elapsed,
				}, nil
			}
			return &SkillResult{
				Success:         false,
				Error:           err.Error(),
				ErrorType:       "internal",
				ExecutionTimeMs: elapsed,
			}, nil
		}
	}

	// 2. Run tool
	raw, err := s.Tool(p)
	if err != nil {
		elapsed := time.Since(start).Milliseconds()
		if s.Fallback != nil {
			return &SkillResult{
				Success:         false,
				Error:           s.Fallback(params, err),
				ErrorType:       "unavailable",
				ExecutionTimeMs: elapsed,
			}, nil
		}
		return nil, err
	}

	// 3. Normalise output
	if s.OutputNormalizer != nil {
		raw, err = s.OutputNormalizer(raw)
		if err != nil {
			elapsed := time.Since(start).Milliseconds()
			if s.Fallback != nil {
				return &SkillResult{
					Success:         false,
					Error:           s.Fallback(params, err),
					ErrorType:       "unavailable",
					ExecutionTimeMs: elapsed,
				}, nil
			}
			return nil, err
		}
	}

	return &SkillResult{
		Success:         true,
		Data:            raw,
		ExecutionTimeMs: time.Since(start).Milliseconds(),
	}, nil
}

// GetOpenAISchema returns the OpenAI function-calling schema for this skill.
func (s *Skill) GetOpenAISchema() OpenAISchema {
	return OpenAISchema{
		Type: "function",
		Function: OpenAIFunction{
			Name:        s.Name,
			Description: s.Description,
			Parameters:  s.Parameters,
		},
	}
}

// GetPromptFragment returns the prompt fragment to inject into the system message.
func (s *Skill) GetPromptFragment() string {
	if s.PromptFragment != "" {
		return strings.TrimSpace(s.PromptFragment)
	}
	return fmt.Sprintf("Use %s when: %s", s.Name, s.Description)
}

// RunTests executes all test cases in isolation. No agent or LLM required.
func (s *Skill) RunTests() []TestResult {
	results := make([]TestResult, 0, len(s.TestCases))

	for _, test := range s.TestCases {
		result, execErr := s.Execute(test.Input)
		passed := true
		reason := ""

		// If Execute itself errored (no fallback, tool panicked), treat as failure
		if execErr != nil {
			if test.ExpectFallback || !test.ExpectSuccess {
				// Expected some kind of failure
			} else {
				passed = false
				reason = fmt.Sprintf("Unexpected execution error: %v", execErr)
			}
			results = append(results, TestResult{
				TestInput: test.Input,
				Passed:    passed,
				Reason:    reason,
			})
			continue
		}

		if test.ExpectFallback {
			if result.Success || result.ErrorType == "invalid_input" {
				passed = false
				reason = fmt.Sprintf(
					"Expected fallback but got success=%v errorType=%q",
					result.Success, result.ErrorType,
				)
			}
		} else if !test.ExpectSuccess {
			if result.Success {
				passed = false
				reason = "Expected failure but skill reported success"
			} else if len(test.ExpectOutputContains) > 0 {
				combined := result.Error + " " + result.Suggestion
				for _, kw := range test.ExpectOutputContains {
					if !strings.Contains(combined, kw) {
						passed = false
						reason = fmt.Sprintf("Expected %q in error/suggestion", kw)
						break
					}
				}
			}
		} else {
			if !result.Success {
				passed = false
				reason = fmt.Sprintf("Expected success but got error: %s", result.Error)
			} else if len(test.ExpectOutputContains) > 0 {
				dataStr := fmt.Sprintf("%v", result.Data)
				for _, kw := range test.ExpectOutputContains {
					if !strings.Contains(dataStr, kw) {
						passed = false
						reason = fmt.Sprintf("Expected %q in output data", kw)
						break
					}
				}
			}
		}

		results = append(results, TestResult{
			TestInput: test.Input,
			Passed:    passed,
			Reason:    reason,
			Result:    result,
		})
	}

	return results
}

// Validate returns a list of warnings about this skill's definition.
func (s *Skill) Validate() []string {
	var warnings []string

	if s.Description == "" {
		warnings = append(warnings, fmt.Sprintf("[%s] has no description", s.Name))
	}

	props, _ := s.Parameters["properties"].(map[string]interface{})
	if len(props) == 0 {
		warnings = append(warnings, fmt.Sprintf("[%s] parameters has no properties defined", s.Name))
	} else {
		for paramName, schema := range props {
			m, _ := schema.(map[string]interface{})
			if _, ok := m["description"]; !ok {
				warnings = append(warnings,
					fmt.Sprintf("[%s] parameter %q has no description", s.Name, paramName))
			}
		}
	}

	if s.Fallback == nil {
		warnings = append(warnings,
			fmt.Sprintf("[%s] no fallback defined — tool failures will propagate", s.Name))
	}

	if s.PromptFragment == "" {
		warnings = append(warnings,
			fmt.Sprintf("[%s] no PromptFragment — agent will use default wording", s.Name))
	}

	return warnings
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

// SkillRegistry manages skill registration, discovery, and dependency resolution.
type SkillRegistry struct {
	skills map[string]*Skill
}

// NewSkillRegistry creates an empty SkillRegistry.
func NewSkillRegistry() *SkillRegistry {
	return &SkillRegistry{skills: make(map[string]*Skill)}
}

// Register adds a skill to the registry.
// Returns an error if the name is already taken, a dependency is missing,
// or the registration would create a cycle.
func (r *SkillRegistry) Register(skill *Skill) error {
	if _, exists := r.skills[skill.Name]; exists {
		return fmt.Errorf("skill %q is already registered", skill.Name)
	}
	r.skills[skill.Name] = skill
	if err := r.checkDependencies(skill); err != nil {
		delete(r.skills, skill.Name)
		return err
	}
	return nil
}

// RegisterMany registers multiple skills in order.
func (r *SkillRegistry) RegisterMany(skills []*Skill) error {
	for _, s := range skills {
		if err := r.Register(s); err != nil {
			return err
		}
	}
	return nil
}

// Get returns a registered skill by name, or an error if not found.
func (r *SkillRegistry) Get(name string) (*Skill, error) {
	s, ok := r.skills[name]
	if !ok {
		return nil, fmt.Errorf("skill %q is not registered", name)
	}
	return s, nil
}

// FindByTags returns all skills that share at least one of the given tags.
func (r *SkillRegistry) FindByTags(tags []string) []*Skill {
	tagSet := make(map[string]struct{}, len(tags))
	for _, t := range tags {
		tagSet[t] = struct{}{}
	}
	var result []*Skill
	for _, s := range r.skills {
		for _, t := range s.Tags {
			if _, ok := tagSet[t]; ok {
				result = append(result, s)
				break
			}
		}
	}
	return result
}

// GetAllSchemas returns OpenAI function-calling schemas for all registered skills.
func (r *SkillRegistry) GetAllSchemas() []OpenAISchema {
	schemas := make([]OpenAISchema, 0, len(r.skills))
	for _, s := range r.skills {
		schemas = append(schemas, s.GetOpenAISchema())
	}
	return schemas
}

// GetCombinedPrompt returns concatenated prompt fragments for the given skills.
func (r *SkillRegistry) GetCombinedPrompt(skillNames []string) (string, error) {
	var parts []string
	for _, name := range skillNames {
		s, err := r.Get(name)
		if err != nil {
			return "", err
		}
		parts = append(parts, fmt.Sprintf("### %s\n%s", s.Name, s.GetPromptFragment()))
	}
	return strings.Join(parts, "\n\n"), nil
}

// Execute runs a registered skill by name.
func (r *SkillRegistry) Execute(name string, params Params) (*SkillResult, error) {
	s, err := r.Get(name)
	if err != nil {
		return nil, err
	}
	return s.Execute(params)
}

// ResolveDependencies returns the skill and its transitive dependencies,
// topologically sorted (dependencies first).
func (r *SkillRegistry) ResolveDependencies(skill *Skill) ([]*Skill, error) {
	var order []string
	visited := make(map[string]bool)
	visiting := make(map[string]bool)

	var visit func(name string) error
	visit = func(name string) error {
		if visiting[name] {
			return &CircularDependencyError{
				Msg: fmt.Sprintf("circular dependency detected involving %q", name),
			}
		}
		if visited[name] {
			return nil
		}
		visiting[name] = true
		s, ok := r.skills[name]
		if !ok {
			return &MissingDependencyError{
				Msg: fmt.Sprintf("dependency %q is not registered", name),
			}
		}
		for _, dep := range s.Dependencies {
			if err := visit(dep); err != nil {
				return err
			}
		}
		visiting[name] = false
		visited[name] = true
		order = append(order, name)
		return nil
	}

	if err := visit(skill.Name); err != nil {
		return nil, err
	}

	result := make([]*Skill, len(order))
	for i, n := range order {
		result[i] = r.skills[n]
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (r *SkillRegistry) checkDependencies(skill *Skill) error {
	for _, dep := range skill.Dependencies {
		if _, ok := r.skills[dep]; !ok {
			return &MissingDependencyError{
				Msg: fmt.Sprintf(
					"skill %q depends on %q which is not registered",
					skill.Name, dep,
				),
			}
		}
	}

	// Full DFS cycle detection
	visiting := make(map[string]bool)
	visited := make(map[string]bool)

	var visit func(name string) error
	visit = func(name string) error {
		if visiting[name] {
			return &CircularDependencyError{
				Msg: fmt.Sprintf("circular dependency involving %q", name),
			}
		}
		if visited[name] {
			return nil
		}
		visiting[name] = true
		if s, ok := r.skills[name]; ok {
			for _, dep := range s.Dependencies {
				if _, ok := r.skills[dep]; ok {
					if err := visit(dep); err != nil {
						return err
					}
				}
			}
		}
		visiting[name] = false
		visited[name] = true
		return nil
	}

	return visit(skill.Name)
}
