// Recovery Manager — goal detection and conversation health recovery.
//
// Go port of code/python/05-context-assembly/recovery_manager.py
//
// When a long conversation drifts, stalls, or gets stuck in a repetition loop,
// the recovery manager diagnoses what's wrong and injects the right intervention
// into the next prompt.
//
// Key types:
//   - GoalDetector     — extracts and tracks user goals from messages.
//   - RecoveryManager  — diagnoses conversation health and generates interventions.
//
// See: docs/04-context-engineering/04-multi-turn-context-management.md
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

// GoalResult is the output of GoalDetector.DetectGoal.
type GoalResult struct {
	Goal               string   `json:"goal"`
	IsNewGoal          bool     `json:"is_new_goal"`
	SupersedesPrevious bool     `json:"supersedes_previous"`
	Subtasks           []string `json:"subtasks"`
	Priority           string   `json:"priority"`
	EstimatedTurns     int      `json:"estimated_turns"`
}

// CompletionResult is the output of GoalDetector.CheckGoalCompletion.
type CompletionResult struct {
	IsComplete        bool     `json:"is_complete"`
	CompletionPct     float64  `json:"completion_pct"`
	RemainingSubtasks []string `json:"remaining_subtasks"`
	Evidence          string   `json:"evidence"`
}

// ConversationIssue is a single diagnosed conversation problem.
type ConversationIssue struct {
	Type            string // "goal_drift", "repetition", "contradiction", "lost_context", "frustration", "stalemate"
	Severity        string // "low", "medium", "high"
	Description     string
	SuggestedAction string
}

// ---------------------------------------------------------------------------
// LLM helpers
// ---------------------------------------------------------------------------

func recoveryLLMAvailable() bool {
	return os.Getenv("OPENAI_API_KEY") != ""
}

func recoveryLLMCall(prompt, model string) (string, error) {
	type oaiMsg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type oaiReq struct {
		Model       string   `json:"model"`
		Messages    []oaiMsg `json:"messages"`
		Temperature float64  `json:"temperature"`
	}
	type oaiChoice struct {
		Message oaiMsg `json:"message"`
	}
	type oaiResp struct {
		Choices []oaiChoice `json:"choices"`
	}

	reqBody, _ := json.Marshal(oaiReq{
		Model:       model,
		Messages:    []oaiMsg{{Role: "user", Content: prompt}},
		Temperature: 0.2,
	})

	req, err := http.NewRequest("POST",
		"https://api.openai.com/v1/chat/completions",
		bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	var parsed oaiResp
	if err := json.Unmarshal(body, &parsed); err != nil || len(parsed.Choices) == 0 {
		return "", fmt.Errorf("parse error or no choices")
	}
	return strings.TrimSpace(parsed.Choices[0].Message.Content), nil
}

// stripCodeFences removes markdown code fences from raw LLM JSON output.
func stripCodeFences(s string) string {
	return regexp.MustCompile("```(?:json)?\\s*|\\s*```").ReplaceAllString(s, "")
}

// ---------------------------------------------------------------------------
// Prompt templates
// ---------------------------------------------------------------------------

const goalDetectionPrompt = `Analyze the user's message and determine their goal.

Output JSON only — no other text:
{
    "goal": "Clear one-sentence description of what the user wants",
    "is_new_goal": true,
    "supersedes_previous": false,
    "subtasks": ["step1", "step2"],
    "priority": "medium",
    "estimated_turns": 3
}

Rules:
- is_new_goal: true if this is a new topic, false if continuing the previous goal
- supersedes_previous: true if this goal replaces the previous one
- subtasks: break complex goals into 2-5 concrete steps; empty list for simple goals
- priority: high if time-sensitive or critical; low if casual or exploratory
- estimated_turns: estimate how many assistant turns this goal requires

Previous conversation context:
%s

User message:
%s`

const completionCheckPrompt = `Assess whether this conversation goal has been accomplished.

Goal: %s

Conversation summary:
%s

Output JSON only — no other text:
{
    "is_complete": false,
    "completion_pct": 40.0,
    "remaining_subtasks": ["step still needed"],
    "evidence": "phrase that shows how far along we are"
}`

const topicRelevancePrompt = `Score how much this user message relates to the stated goal.

Goal: %s
User message: %s

Output a single JSON number between 0.0 and 1.0.
1.0 = directly on-topic. 0.0 = completely different subject.
Output the number only, no other text.`

// ---------------------------------------------------------------------------
// GoalDetector
// ---------------------------------------------------------------------------

var goalVerbsRe = regexp.MustCompile(
	`(?i)\b(?:need|want|would\s+like|must|have\s+to|trying\s+to|help\s+me` +
		`|please|can\s+you|could\s+you|looking\s+to)\b`)

var completionWordsRe = regexp.MustCompile(
	`(?i)\b(?:confirmed|completed|done|booked|finished|ready|sent|paid|` +
		`success|all\s+set|thank\s+you|you're\s+all\s+set)\b`)

var goalFragmentSplitRe = regexp.MustCompile(`[,;]`)
var goalSentenceSplitRe = regexp.MustCompile(`[.!?]`)
var goalWordRe = regexp.MustCompile(`\b\w{4,}\b`)

// GoalDetector detects and tracks user goals across conversation turns.
// Uses the OpenAI API when available; falls back to deterministic heuristics.
type GoalDetector struct {
	Model  string
	useLLM bool
}

// NewGoalDetector creates a new GoalDetector.
func NewGoalDetector(model string) *GoalDetector {
	if model == "" {
		model = "gpt-4o-mini"
	}
	return &GoalDetector{Model: model, useLLM: recoveryLLMAvailable()}
}

// DetectGoal detects the user's goal from a message.
func (gd *GoalDetector) DetectGoal(userMessage, conversationContext string) *GoalResult {
	if gd.useLLM {
		if ctx := conversationContext; ctx == "" {
			ctx = "(none)"
		}
		prompt := fmt.Sprintf(goalDetectionPrompt, conversationContext, userMessage)
		raw, err := recoveryLLMCall(prompt, gd.Model)
		if err == nil {
			raw = strings.TrimSpace(stripCodeFences(raw))
			var data GoalResult
			if err := json.Unmarshal([]byte(raw), &data); err == nil {
				return &data
			}
		}
	}
	return gd.heuristicGoal(userMessage)
}

// CheckGoalCompletion checks whether goal has been accomplished.
func (gd *GoalDetector) CheckGoalCompletion(goal, conversationSummary string) *CompletionResult {
	if gd.useLLM {
		prompt := fmt.Sprintf(completionCheckPrompt, goal, conversationSummary)
		raw, err := recoveryLLMCall(prompt, gd.Model)
		if err == nil {
			raw = strings.TrimSpace(stripCodeFences(raw))
			var data CompletionResult
			if err := json.Unmarshal([]byte(raw), &data); err == nil {
				return &data
			}
		}
	}
	return gd.heuristicCompletion(goal, conversationSummary)
}

// DetectTopicChange scores how much userMessage relates to currentGoal (0.0–1.0).
func (gd *GoalDetector) DetectTopicChange(currentGoal, userMessage string) float64 {
	if gd.useLLM {
		prompt := fmt.Sprintf(topicRelevancePrompt, currentGoal, userMessage)
		raw, err := recoveryLLMCall(prompt, gd.Model)
		if err == nil {
			raw = strings.TrimSpace(raw)
			var score float64
			if err := json.Unmarshal([]byte(raw), &score); err == nil {
				if score < 0 {
					return 0.0
				}
				if score > 1 {
					return 1.0
				}
				return score
			}
		}
	}
	return gd.heuristicRelevance(currentGoal, userMessage)
}

// heuristicGoal extracts a goal using keyword heuristics.
func (gd *GoalDetector) heuristicGoal(message string) *GoalResult {
	isGoal := goalVerbsRe.MatchString(message)
	firstSentence := goalSentenceSplitRe.Split(message, 2)[0]
	goalText := strings.TrimSpace(firstSentence)
	if len(goalText) > 100 {
		goalText = goalText[:100]
	}
	if goalText == "" {
		if len(message) > 100 {
			goalText = message[:100]
		} else {
			goalText = message
		}
	}

	var subtasks []string
	for _, frag := range goalFragmentSplitRe.Split(message, -1) {
		frag = strings.TrimSpace(frag)
		if len(frag) > 10 && frag != message {
			if len(frag) > 60 {
				frag = frag[:60]
			}
			subtasks = append(subtasks, frag)
		}
	}
	if len(subtasks) > 4 {
		subtasks = subtasks[:4]
	}

	estimatedTurns := len(subtasks) + 1
	if estimatedTurns < 2 {
		estimatedTurns = 2
	}

	return &GoalResult{
		Goal:               goalText,
		IsNewGoal:          isGoal,
		SupersedesPrevious: false,
		Subtasks:           subtasks,
		Priority:           "medium",
		EstimatedTurns:     estimatedTurns,
	}
}

// heuristicCompletion estimates goal completion from keyword presence.
func (gd *GoalDetector) heuristicCompletion(goal, summary string) *CompletionResult {
	matches := len(completionWordsRe.FindAllString(summary, -1))
	pct := float64(matches) * 20.0
	if pct > 100 {
		pct = 100
	}
	return &CompletionResult{
		IsComplete:        pct >= 80.0,
		CompletionPct:     pct,
		RemainingSubtasks: []string{},
		Evidence:          fmt.Sprintf("%d completion signal(s) in summary", matches),
	}
}

// heuristicRelevance uses word-overlap to score relevance.
func (gd *GoalDetector) heuristicRelevance(goal, message string) float64 {
	goalWords := goalWordRe.FindAllString(strings.ToLower(goal), -1)
	msgWords := goalWordRe.FindAllString(strings.ToLower(message), -1)
	if len(goalWords) == 0 {
		return 1.0
	}
	goalSet := map[string]bool{}
	for _, w := range goalWords {
		goalSet[w] = true
	}
	msgSet := map[string]bool{}
	for _, w := range msgWords {
		msgSet[w] = true
	}
	overlap := 0
	for w := range goalSet {
		if msgSet[w] {
			overlap++
		}
	}
	return float64(overlap) / float64(len(goalSet))
}

// ---------------------------------------------------------------------------
// Intervention templates
// ---------------------------------------------------------------------------

var interventionTemplates = map[string]string{
	"goal_drift": "[REMINDER: The current goal is: %s. " +
		"Pending steps: %s. " +
		"Please return focus to this goal, or ask the user if they want to change it.]",
	"repetition": "[NOTE: The following information has already been provided by the user and " +
		"should NOT be asked again: %s. " +
		"Do not repeat these questions.]",
	"contradiction": "[CONSISTENCY CHECK: Previous recommendations were: %s. " +
		"Ensure any new suggestions are consistent with these, or explicitly " +
		"acknowledge the change.]",
	"lost_context": "[CONTEXT SUMMARY: Goal: %s. " +
		"Completed: %s. Pending: %s. " +
		"User info: %s. Resume from this point.]",
	"frustration": "[USER FRUSTRATION DETECTED: The user appears frustrated. " +
		"Acknowledge their concern, apologise for any confusion, " +
		"and focus on resolving their core need: %s.]",
	"stalemate": "[STALEMATE: The conversation has not made progress for several turns. " +
		"Offer a concrete next step or ask the user directly: " +
		"'What would be most helpful right now?']",
}

// ---------------------------------------------------------------------------
// RecoveryManager
// ---------------------------------------------------------------------------

// RecoveryManager detects and recovers from conversation health problems.
// Works alongside a StateManager to monitor health and generate interventions.
type RecoveryManager struct {
	sm                   *StateManager
	turnsWithoutProgress int
	lastSubtaskCount     int
}

// Thresholds for diagnosis.
const (
	recoveryDriftThreshold       = 5
	recoveryFrustrationThreshold = 2
	recoveryRepetitionThreshold  = 3
	recoveryStalemateThreshold   = 8
)

// NewRecoveryManager creates a new RecoveryManager.
func NewRecoveryManager(sm *StateManager) *RecoveryManager {
	return &RecoveryManager{sm: sm}
}

// Diagnose diagnoses current conversation health.
// Returns a list of ConversationIssue sorted with the highest-severity issues first.
func (rm *RecoveryManager) Diagnose() []ConversationIssue {
	var issues []ConversationIssue
	state := rm.sm.State

	// Goal drift
	if state.CurrentGoal != "" && state.TurnsSinceGoalMentioned >= recoveryDriftThreshold {
		severity := "medium"
		if state.TurnsSinceGoalMentioned >= 8 {
			severity = "high"
		}
		issues = append(issues, ConversationIssue{
			Type:     "goal_drift",
			Severity: severity,
			Description: fmt.Sprintf(
				"Conversation has drifted %d turns away from goal: '%s'",
				state.TurnsSinceGoalMentioned, state.CurrentGoal),
			SuggestedAction: "remind_goal",
		})
	}

	// User frustration
	if state.UserFrustrationSignals >= recoveryFrustrationThreshold {
		severity := "medium"
		if state.UserFrustrationSignals >= 3 {
			severity = "high"
		}
		issues = append(issues, ConversationIssue{
			Type:     "frustration",
			Severity: severity,
			Description: fmt.Sprintf(
				"User frustration detected: %d signal(s)",
				state.UserFrustrationSignals),
			SuggestedAction: "ask_user_to_clarify",
		})
	}

	// Repetition
	if len(state.AgentQuestionsAsked) >= recoveryRepetitionThreshold {
		unique := map[string]bool{}
		duplicates := 0
		nonWordRe := regexp.MustCompile(`\W+`)
		for _, q := range state.AgentQuestionsAsked {
			key := strings.TrimSpace(nonWordRe.ReplaceAllString(strings.ToLower(q), " "))
			if len(key) > 30 {
				key = key[:30]
			}
			if unique[key] {
				duplicates++
			} else {
				unique[key] = true
			}
		}
		if duplicates >= 2 {
			issues = append(issues, ConversationIssue{
				Type:            "repetition",
				Severity:        "medium",
				Description:     fmt.Sprintf("Agent has repeated similar questions %d time(s)", duplicates),
				SuggestedAction: "remind_goal",
			})
		}
	}

	// Stalemate
	currentCompleted := len(state.SubtasksCompleted)
	if currentCompleted > rm.lastSubtaskCount {
		rm.lastSubtaskCount = currentCompleted
		rm.turnsWithoutProgress = 0
	} else if state.CurrentGoal != "" {
		rm.turnsWithoutProgress++
	}

	if rm.turnsWithoutProgress >= recoveryStalemateThreshold {
		issues = append(issues, ConversationIssue{
			Type:     "stalemate",
			Severity: "high",
			Description: fmt.Sprintf(
				"No progress in %d turns; pending: %v",
				rm.turnsWithoutProgress, state.SubtasksPending),
			SuggestedAction: "inject_checkpoint",
		})
	}

	// Sort by severity
	sevOrder := map[string]int{"high": 0, "medium": 1, "low": 2}
	for i := 1; i < len(issues); i++ {
		for j := i; j > 0; j-- {
			ai := sevOrder[issues[j].Severity]
			aj := sevOrder[issues[j-1].Severity]
			if ai < aj {
				issues[j], issues[j-1] = issues[j-1], issues[j]
			} else {
				break
			}
		}
	}
	return issues
}

// GetIntervention generates an intervention message to inject into the prompt.
func (rm *RecoveryManager) GetIntervention(issues []ConversationIssue) string {
	if len(issues) == 0 {
		return ""
	}

	state := rm.sm.State
	primary := issues[0]
	tmpl, ok := interventionTemplates[primary.Type]
	if !ok {
		return fmt.Sprintf("[CONVERSATION HEALTH ISSUE: %s]", primary.Description)
	}

	goal := state.CurrentGoal
	if goal == "" {
		goal = "(unknown)"
	}
	pending := strings.Join(state.SubtasksPending, ", ")
	if pending == "" {
		pending = "none"
	}
	completed := strings.Join(state.SubtasksCompleted, ", ")
	if completed == "" {
		completed = "none"
	}

	lastQs := state.AgentQuestionsAsked
	if len(lastQs) > 5 {
		lastQs = lastQs[len(lastQs)-5:]
	}
	alreadyAsked := strings.Join(lastQs, "; ")
	if alreadyAsked == "" {
		alreadyAsked = "none"
	}

	lastRecs := state.AgentRecommendations
	if len(lastRecs) > 5 {
		lastRecs = lastRecs[len(lastRecs)-5:]
	}
	recommendations := strings.Join(lastRecs, "; ")
	if recommendations == "" {
		recommendations = "none"
	}

	var infoPairs []string
	for k, v := range state.UserProvidedInfo {
		infoPairs = append(infoPairs, k+"="+v)
	}
	userInfo := strings.Join(infoPairs, ", ")
	if userInfo == "" {
		userInfo = "none"
	}

	switch primary.Type {
	case "goal_drift":
		return fmt.Sprintf(tmpl, goal, pending)
	case "repetition":
		return fmt.Sprintf(tmpl, alreadyAsked)
	case "contradiction":
		return fmt.Sprintf(tmpl, recommendations)
	case "lost_context":
		return fmt.Sprintf(tmpl, goal, completed, pending, userInfo)
	case "frustration":
		return fmt.Sprintf(tmpl, goal)
	case "stalemate":
		return tmpl
	}
	return fmt.Sprintf("[CONVERSATION HEALTH ISSUE: %s]", primary.Description)
}

// ShouldReset returns true if the conversation is beyond recovery.
func (rm *RecoveryManager) ShouldReset(issues []ConversationIssue) bool {
	highCount := 0
	issueTypes := map[string]bool{}
	for _, i := range issues {
		if i.Severity == "high" {
			highCount++
		}
		issueTypes[i.Type] = true
	}
	stalemate := issueTypes["stalemate"]
	frustration := issueTypes["frustration"]
	return highCount >= 2 || (stalemate && frustration)
}

// GenerateProgressReport generates a user-facing progress summary.
func (rm *RecoveryManager) GenerateProgressReport() string {
	state := rm.sm.State
	var parts []string

	if state.CurrentGoal != "" {
		parts = append(parts, "Goal: "+state.CurrentGoal)
	}
	if len(state.SubtasksCompleted) > 0 {
		parts = append(parts, "Done: "+strings.Join(state.SubtasksCompleted, ", "))
	}
	if len(state.SubtasksPending) > 0 {
		parts = append(parts, "Still to do: "+strings.Join(state.SubtasksPending, ", "))
	} else {
		parts = append(parts, "All steps complete.")
	}
	if len(state.AgentRecommendations) > 0 {
		recs := state.AgentRecommendations
		if len(recs) > 3 {
			recs = recs[len(recs)-3:]
		}
		parts = append(parts, "Recommendations made: "+strings.Join(recs, "; "))
	}

	if len(parts) == 0 {
		return "No active goal."
	}
	return strings.Join(parts, "\n")
}
