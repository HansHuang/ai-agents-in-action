// State Manager — explicit conversation state that survives context truncation.
//
// Go port of code/python/05-context-assembly/state_manager.py
//
// The message list is temporary.  State is durable.
//
// Every turn updates ConversationState, which is then injected back into
// the system prompt so the agent never loses track of goals, collected
// user information, or recommendations already made.
//
// See: docs/04-context-engineering/04-multi-turn-context-management.md
package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// ConversationState
// ---------------------------------------------------------------------------

// ConversationState is the agent's durable memory — it survives context
// window limits.  Inject ToPromptContext into every system prompt.
type ConversationState struct {
	// Task tracking
	CurrentGoal       string   `json:"current_goal"`
	SubtasksCompleted []string `json:"subtasks_completed"`
	SubtasksPending   []string `json:"subtasks_pending"`
	GoalSetAtTurn     int      `json:"goal_set_at_turn"`
	// User context
	UserName         string            `json:"user_name"`
	UserPreferences  map[string]string `json:"user_preferences"`
	UserProvidedInfo map[string]string `json:"user_provided_info"`
	// Agent context
	AgentRecommendations []string `json:"agent_recommendations"`
	AgentQuestionsAsked  []string `json:"agent_questions_asked"`
	AgentMode            string   `json:"agent_mode"`
	// Conversation health
	TurnsSinceGoalMentioned int      `json:"turns_since_goal_mentioned"`
	UserFrustrationSignals  int      `json:"user_frustration_signals"`
	TopicChanges            []string `json:"topic_changes"`
	TurnCount               int      `json:"turn_count"`
}

// NewConversationState returns a ConversationState with sensible defaults.
func NewConversationState() *ConversationState {
	return &ConversationState{
		SubtasksCompleted:    []string{},
		SubtasksPending:      []string{},
		UserPreferences:      map[string]string{},
		UserProvidedInfo:     map[string]string{},
		AgentRecommendations: []string{},
		AgentQuestionsAsked:  []string{},
		AgentMode:            "general",
		TopicChanges:         []string{},
	}
}

// ToPromptContext returns a compact, multi-line state summary for prompt injection.
func (s *ConversationState) ToPromptContext() string {
	var parts []string

	if s.UserName != "" {
		parts = append(parts, "User: "+s.UserName)
	}
	if s.CurrentGoal != "" {
		parts = append(parts, "Current goal: "+s.CurrentGoal)
	}
	if len(s.SubtasksCompleted) > 0 {
		parts = append(parts, "Completed: "+strings.Join(s.SubtasksCompleted, "; "))
	}
	if len(s.SubtasksPending) > 0 {
		parts = append(parts, "Pending: "+strings.Join(s.SubtasksPending, "; "))
	}
	if len(s.AgentRecommendations) > 0 {
		recs := s.AgentRecommendations
		if len(recs) > 5 {
			recs = recs[len(recs)-5:]
		}
		parts = append(parts, "Previous recommendations: "+strings.Join(recs, "; "))
	}
	if len(s.AgentQuestionsAsked) > 0 {
		qs := s.AgentQuestionsAsked
		if len(qs) > 5 {
			qs = qs[len(qs)-5:]
		}
		parts = append(parts, "Already asked about: "+strings.Join(qs, "; "))
	}
	if len(s.UserProvidedInfo) > 0 {
		var pairs []string
		for k, v := range s.UserProvidedInfo {
			pairs = append(pairs, k+"="+v)
		}
		parts = append(parts, "User has provided: "+strings.Join(pairs, ", "))
	}
	if len(s.UserPreferences) > 0 {
		var pairs []string
		for k, v := range s.UserPreferences {
			pairs = append(pairs, k+"="+v)
		}
		parts = append(parts, "User preferences: "+strings.Join(pairs, ", "))
	}
	if s.AgentMode != "general" {
		parts = append(parts, "Agent mode: "+s.AgentMode)
	}
	return strings.Join(parts, "\n")
}

// ToDict serialises the state to a JSON-round-trippable map.
func (s *ConversationState) ToDict() map[string]any {
	b, _ := json.Marshal(s)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}

// ConversationStateFromDict deserialises state from a map.
func ConversationStateFromDict(data map[string]any) *ConversationState {
	b, _ := json.Marshal(data)
	s := NewConversationState()
	_ = json.Unmarshal(b, s)
	if s.SubtasksCompleted == nil {
		s.SubtasksCompleted = []string{}
	}
	if s.SubtasksPending == nil {
		s.SubtasksPending = []string{}
	}
	if s.UserPreferences == nil {
		s.UserPreferences = map[string]string{}
	}
	if s.UserProvidedInfo == nil {
		s.UserProvidedInfo = map[string]string{}
	}
	if s.AgentRecommendations == nil {
		s.AgentRecommendations = []string{}
	}
	if s.AgentQuestionsAsked == nil {
		s.AgentQuestionsAsked = []string{}
	}
	if s.TopicChanges == nil {
		s.TopicChanges = []string{}
	}
	if s.AgentMode == "" {
		s.AgentMode = "general"
	}
	return s
}

// ---------------------------------------------------------------------------
// Information extraction helpers
// ---------------------------------------------------------------------------

var (
	reOrderNum   = regexp.MustCompile(`(?i)\border\s*(?:number|#|num)?\s*(?:is\s*)?[:#]?\s*([A-Z0-9\-]{4,20})`)
	reUserName   = regexp.MustCompile(`(?i)\b(?:my\s+name\s+is|i(?:'m| am)\s+called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)`)
	reBudget     = regexp.MustCompile(`(?i)\b(?:budget|spend|cost).*?[\$£€]?\s*([0-9][0-9,]*(?:\.[0-9]{2})?)\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)?`)
	rePreference = regexp.MustCompile(`(?i)\bi\s+prefer\s+([^.!?]{3,60})`)
	reEmail      = regexp.MustCompile(`\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b`)

	reFrustration = regexp.MustCompile(
		`(?i)\b(?:already\s+told|said\s+(?:this|that)|you\s+asked\s+(?:me\s+)?(?:that|this|about))\b|` +
			`\b(?:again\??|for\s+the\s+\w+\s+time)\b|` +
			`\b(?:stop\s+repeating|stop\s+asking|can.t\s+you|why\s+(?:do|are|is|can.t))\b|` +
			`\b(?:frustrated|annoyed|useless|terrible|awful)\b|!{2,}`)

	reGoalKeywords = regexp.MustCompile(
		`(?i)\b(?:i\s+(?:need|want|would\s+like|must|have\s+to)|` +
			`(?:please|can\s+you|could\s+you|help\s+me\s+(?:to|with)?)\s+|` +
			`(?:my\s+goal|the\s+goal|objective|task)\s+is)\b`)
)

type infoExtractor struct {
	key     string
	pattern *regexp.Regexp
}

var infoExtractors = []infoExtractor{
	{"order_number", reOrderNum},
	{"name", reUserName},
	{"budget", reBudget},
	{"preference", rePreference},
	{"email", reEmail},
}

func extractUserInfo(message string) map[string]string {
	found := map[string]string{}
	for _, ex := range infoExtractors {
		if m := ex.pattern.FindStringSubmatch(message); len(m) > 1 {
			val := strings.Trim(m[1], " ,.")
			if _, exists := found[ex.key]; !exists {
				found[ex.key] = val
			}
		}
	}
	return found
}

func hasFrustration(message string) bool {
	return reFrustration.MatchString(message)
}

// goalRelevance returns a 0–1 score of how relevant message is to currentGoal.
func goalRelevance(message, currentGoal string) float64 {
	if currentGoal == "" {
		return 1.0
	}
	reWord := regexp.MustCompile(`\b\w{4,}\b`)
	goalWords := set(reWord.FindAllString(strings.ToLower(currentGoal), -1))
	msgWords := set(reWord.FindAllString(strings.ToLower(message), -1))
	if len(goalWords) == 0 {
		return 1.0
	}
	overlap := 0
	for w := range goalWords {
		if _, ok := msgWords[w]; ok {
			overlap++
		}
	}
	return float64(overlap) / float64(len(goalWords))
}

func set(words []string) map[string]struct{} {
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}

// ---------------------------------------------------------------------------
// StateManager
// ---------------------------------------------------------------------------

const (
	driftTurnsThreshold = 5
	checkpointInterval  = 10
	maxTrackedItems     = 20
)

// UserTurnChanges describes what changed when processing a user turn.
type UserTurnChanges struct {
	GoalDetected        bool
	InfoExtracted       map[string]string
	FrustrationDetected bool
	TopicChanged        bool
	Turn                int
}

// AgentTurnChanges describes what changed when processing an agent turn.
type AgentTurnChanges struct {
	RecommendationsAdded int
	QuestionsAsked       int
	SubtasksCompleted    int
	GoalComplete         bool
}

// ToolCall represents a function tool call from the agent.
type ToolCall struct {
	Name     string
	Function struct {
		Name string
	}
}

// StateManager manages conversation state across turns.
type StateManager struct {
	State *ConversationState
}

// NewStateManager returns a new StateManager with an empty ConversationState.
func NewStateManager() *StateManager {
	return &StateManager{State: NewConversationState()}
}

// SetGoal sets the current conversation goal and optional subtasks.
func (m *StateManager) SetGoal(goal string, subtasks []string) {
	m.State.CurrentGoal = goal
	m.State.SubtasksPending = append([]string{}, subtasks...)
	m.State.SubtasksCompleted = []string{}
	m.State.GoalSetAtTurn = m.State.TurnCount
	m.State.TurnsSinceGoalMentioned = 0
}

// MarkSubtaskComplete moves a subtask from pending to completed.
func (m *StateManager) MarkSubtaskComplete(subtask string) {
	for i, item := range m.State.SubtasksPending {
		if item == subtask ||
			strings.Contains(strings.ToLower(item), strings.ToLower(subtask)) ||
			strings.Contains(strings.ToLower(subtask), strings.ToLower(item)) {
			m.State.SubtasksPending = append(
				m.State.SubtasksPending[:i], m.State.SubtasksPending[i+1:]...)
			m.State.SubtasksCompleted = append(m.State.SubtasksCompleted, item)
			return
		}
	}
}

// CheckGoalDrift returns true when the conversation has drifted from the goal.
func (m *StateManager) CheckGoalDrift() bool {
	if m.State.CurrentGoal == "" {
		return false
	}
	return m.State.TurnsSinceGoalMentioned >= driftTurnsThreshold
}

// CheckGoalComplete returns true when all subtasks are completed.
func (m *StateManager) CheckGoalComplete() bool {
	if m.State.CurrentGoal == "" {
		return false
	}
	return len(m.State.SubtasksPending) == 0 && len(m.State.SubtasksCompleted) > 0
}

// ProcessUserTurn updates state based on a user message.
func (m *StateManager) ProcessUserTurn(userMessage string) UserTurnChanges {
	m.State.TurnCount++
	changes := UserTurnChanges{
		InfoExtracted: map[string]string{},
		Turn:          m.State.TurnCount,
	}

	// Relevance / drift
	relevance := goalRelevance(userMessage, m.State.CurrentGoal)
	if relevance < 0.25 && m.State.CurrentGoal != "" {
		m.State.TurnsSinceGoalMentioned++
		if m.State.TurnsSinceGoalMentioned >= driftTurnsThreshold {
			snippet := userMessage
			if len(snippet) > 80 {
				snippet = snippet[:80]
			}
			snippet = strings.TrimSpace(snippet)
			n := len(m.State.TopicChanges)
			if n == 0 || m.State.TopicChanges[n-1] != snippet {
				m.State.TopicChanges = append(m.State.TopicChanges, snippet)
			}
			changes.TopicChanged = true
		}
	} else {
		m.State.TurnsSinceGoalMentioned = 0
	}

	// Goal detection
	if reGoalKeywords.MatchString(userMessage) {
		changes.GoalDetected = true
	}

	// Info extraction
	extracted := extractUserInfo(userMessage)
	if len(extracted) > 0 {
		for k, v := range extracted {
			m.State.UserProvidedInfo[k] = v
		}
		if name, ok := extracted["name"]; ok && m.State.UserName == "" {
			m.State.UserName = name
		}
		changes.InfoExtracted = extracted
	}

	// Frustration
	if hasFrustration(userMessage) {
		m.State.UserFrustrationSignals++
		changes.FrustrationDetected = true
	}

	return changes
}

// ProcessAgentTurn updates state based on an agent response and optional tool calls.
func (m *StateManager) ProcessAgentTurn(agentResponse string, toolCalls []ToolCall) AgentTurnChanges {
	changes := AgentTurnChanges{}

	// Recommendations
	reRec := regexp.MustCompile(`(?i)\b(?:I\s+(?:recommend|suggest|advise|propose)|` +
		`you\s+(?:should|could|might\s+want\s+to))\s+([^.!?]{5,120})`)
	for _, m2 := range reRec.FindAllStringSubmatch(agentResponse, -1) {
		rec := strings.TrimRight(strings.TrimSpace(m2[1]), ".,")
		found := false
		for _, r := range m.State.AgentRecommendations {
			if r == rec {
				found = true
				break
			}
		}
		if !found {
			m.State.AgentRecommendations = append(m.State.AgentRecommendations, rec)
			changes.RecommendationsAdded++
		}
	}
	if len(m.State.AgentRecommendations) > maxTrackedItems {
		m.State.AgentRecommendations = m.State.AgentRecommendations[len(m.State.AgentRecommendations)-maxTrackedItems:]
	}

	// Questions
	reBoundary := regexp.MustCompile(`(?:[.!?])\s+`)
	sentences := reBoundary.Split(agentResponse, -1)
	for _, sentence := range sentences {
		if strings.Contains(sentence, "?") {
			q := strings.TrimSpace(sentence)
			if len(q) > 120 {
				q = q[:120]
			}
			if q == "" {
				continue
			}
			found := false
			for _, existing := range m.State.AgentQuestionsAsked {
				if existing == q {
					found = true
					break
				}
			}
			if !found {
				m.State.AgentQuestionsAsked = append(m.State.AgentQuestionsAsked, q)
				changes.QuestionsAsked++
			}
		}
	}
	if len(m.State.AgentQuestionsAsked) > maxTrackedItems {
		m.State.AgentQuestionsAsked = m.State.AgentQuestionsAsked[len(m.State.AgentQuestionsAsked)-maxTrackedItems:]
	}

	// Tool calls → subtask completion
	for _, call := range toolCalls {
		fnName := strings.ReplaceAll(call.Function.Name, "_", " ")
		fnName = strings.ToLower(fnName)
		if fnName == "" {
			fnName = strings.ToLower(call.Name)
		}
		for _, pending := range m.State.SubtasksPending {
			if strings.Contains(strings.ToLower(pending), fnName) {
				m.MarkSubtaskComplete(pending)
				changes.SubtasksCompleted++
			}
		}
	}

	if m.CheckGoalComplete() {
		changes.GoalComplete = true
	}
	return changes
}

// GetRecoveryAction returns the most appropriate recovery action or "".
func (m *StateManager) GetRecoveryAction() string {
	if m.State.UserFrustrationSignals >= 3 {
		return "ask_user_to_clarify"
	}
	if m.CheckGoalDrift() {
		return "remind_goal"
	}
	if m.State.TurnCount > 0 &&
		m.State.TurnCount%checkpointInterval == 0 &&
		m.State.CurrentGoal != "" {
		return "inject_checkpoint"
	}
	if m.State.CurrentGoal != "" &&
		len(m.State.SubtasksPending) == 0 &&
		len(m.State.SubtasksCompleted) > 0 {
		return "summarize_progress"
	}
	return ""
}

// BuildSystemPromptWithState augments basePrompt with the current conversation state.
func (m *StateManager) BuildSystemPromptWithState(basePrompt string) string {
	ctx := m.State.ToPromptContext()
	if ctx == "" {
		return basePrompt
	}
	return fmt.Sprintf(
		"%s\n\n## Conversation State (maintained across turns)\n%s\n\n"+
			"Use this state to maintain continuity. "+
			"Do not repeat questions already asked. "+
			"Do not contradict previous recommendations.",
		basePrompt, ctx,
	)
}
