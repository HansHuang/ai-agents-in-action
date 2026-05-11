package skills

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// SkilledAgent
// ---------------------------------------------------------------------------

// ConversationMessage is a single turn in the agent's conversation history.
type ConversationMessage struct {
	Role    string // "user" | "assistant" | "tool"
	Content string
}

// SkilledAgentConfig controls the agent's behaviour.
type SkilledAgentConfig struct {
	// MaxIterations caps the agent loop to prevent run-away tool calls.
	MaxIterations int
	// SystemPromptTemplate is a format string that receives the skill
	// documentation when the agent is initialised.
	SystemPromptTemplate string
}

// DefaultSkilledAgentConfig returns sensible defaults.
func DefaultSkilledAgentConfig() SkilledAgentConfig {
	return SkilledAgentConfig{
		MaxIterations: 10,
		SystemPromptTemplate: `You are a helpful AI assistant with access to the following skills:

%s

When the user asks a question, decide which skill(s) to use and call them in
order. Return a final natural-language answer once all required information
has been gathered.`,
	}
}

// AgentTurn is the result of one user → agent exchange.
type AgentTurn struct {
	UserInput     string
	FinalResponse string
	SkillsInvoked []string
	Iterations    int
	ElapsedMs     int64
}

// SkilledAgent wraps a SkillRegistry and executes a simplified agent loop.
// It does NOT call an LLM; instead it simulates tool dispatch with a
// deterministic keyword router so the demo runs without API keys.
type SkilledAgent struct {
	cfg      SkilledAgentConfig
	registry *SkillRegistry
	history  []ConversationMessage
	prompt   string // combined system prompt
}

// NewSkilledAgent creates an agent backed by the given registry.
func NewSkilledAgent(cfg SkilledAgentConfig, registry *SkillRegistry) *SkilledAgent {
	skillDoc := buildSkillDoc(registry)
	prompt := fmt.Sprintf(cfg.SystemPromptTemplate, skillDoc)
	return &SkilledAgent{cfg: cfg, registry: registry, prompt: prompt}
}

// SystemPrompt returns the agent's current system prompt.
func (a *SkilledAgent) SystemPrompt() string { return a.prompt }

// History returns all turns so far.
func (a *SkilledAgent) History() []ConversationMessage { return a.history }

// Reset clears the conversation history.
func (a *SkilledAgent) Reset() { a.history = nil }

// Run processes one user message and returns the agent turn result.
// The ctx parameter allows callers to cancel long-running tool chains.
func (a *SkilledAgent) Run(ctx context.Context, userInput string) (*AgentTurn, error) {
	start := time.Now()
	a.history = append(a.history, ConversationMessage{Role: "user", Content: userInput})

	turn := &AgentTurn{UserInput: userInput}
	skillsInvoked := []string{}
	var finalResponse string

	for i := 0; i < a.cfg.MaxIterations; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		turn.Iterations = i + 1

		// Skill dispatch: attempt to match user input to a registered skill
		matched, params, ok := a.matchSkill(userInput)
		if !ok {
			// No skill matched — return a direct response
			finalResponse = "I don't have a skill that can help with that request."
			break
		}

		result, err := a.registry.Execute(matched, params)
		skillsInvoked = append(skillsInvoked, matched)

		if err != nil || !result.Success {
			errMsg := err.Error()
			if err == nil {
				errMsg = result.Error
			}
			finalResponse = fmt.Sprintf("I tried to use the %q skill but encountered an error: %s", matched, errMsg)
			break
		}

		// In a real agent we'd feed the result back to the LLM; here we format it directly.
		finalResponse = fmt.Sprintf("Result from %s: %v", matched, result.Data)
		break
	}

	a.history = append(a.history, ConversationMessage{Role: "assistant", Content: finalResponse})
	turn.FinalResponse = finalResponse
	turn.SkillsInvoked = skillsInvoked
	turn.ElapsedMs = time.Since(start).Milliseconds()
	return turn, nil
}

// matchSkill does a simple keyword match to select a skill from the registry.
// Returns skill name, extracted params, and whether a match was found.
func (a *SkilledAgent) matchSkill(input string) (string, Params, bool) {
	lower := strings.ToLower(input)
	for _, schema := range a.registry.GetAllSchemas() {
		name := schema.Function.Name
		// Use skill name keywords as a simple heuristic
		if strings.Contains(lower, strings.ToLower(strings.ReplaceAll(name, "_", " "))) ||
			strings.Contains(lower, strings.ToLower(strings.ReplaceAll(name, "_", ""))) {
			// Build minimal params — real agents would parse LLM JSON tool call
			params := Params{"input": input}
			return name, params, true
		}
	}
	return "", nil, false
}

// buildSkillDoc generates markdown documentation for all registered skills.
func buildSkillDoc(r *SkillRegistry) string {
	schemas := r.GetAllSchemas()
	if len(schemas) == 0 {
		return "(no skills registered)"
	}
	var sb strings.Builder
	for _, s := range schemas {
		fmt.Fprintf(&sb, "- **%s**: %s\n", s.Function.Name, s.Function.Description)
	}
	return sb.String()
}

// PrintTurnReport prints a formatted summary of an AgentTurn.
func PrintTurnReport(turn *AgentTurn) {
	fmt.Println("─────────────────────────────────────────")
	fmt.Printf("User        : %s\n", turn.UserInput)
	fmt.Printf("Response    : %s\n", turn.FinalResponse)
	fmt.Printf("Skills used : %s\n", strings.Join(turn.SkillsInvoked, ", "))
	fmt.Printf("Iterations  : %d\n", turn.Iterations)
	fmt.Printf("Elapsed     : %d ms\n", turn.ElapsedMs)
}
