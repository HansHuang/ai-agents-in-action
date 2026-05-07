// Session Manager — multi-session conversation management.
//
// Go port of code/python/05-context-assembly/session_manager.py
//
// Handles session creation, expiry, branching, reset, and JSON persistence.
//
// See: docs/04-context-engineering/04-multi-turn-context-management.md
package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

// Message is a single LLM chat message.
type Message struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ToolCalls []any  `json:"tool_calls,omitempty"`
}

// SessionData is the serialisable form of Session.
type SessionData struct {
	SessionID        string         `json:"session_id"`
	UserID           string         `json:"user_id"`
	CreatedAt        float64        `json:"created_at"`
	LastActivity     float64        `json:"last_activity"`
	State            map[string]any `json:"state"`
	Messages         []Message      `json:"messages"`
	InheritedContext string         `json:"inherited_context"`
	IsActive         bool           `json:"is_active"`
	Summarizer       SummarizerData `json:"summarizer"`
}

// Session is a single user conversation.
type Session struct {
	SessionID        string
	UserID           string
	CreatedAt        float64
	LastActivity     float64
	StateManager     *StateManager
	Messages         []Message
	Summarizer       *ProgressiveSummarizer
	InheritedContext string
	IsActive         bool
}

// NewSession creates a new Session for userId.
func NewSession(userID string) *Session {
	now := float64(time.Now().UnixNano()) / 1e9
	return &Session{
		SessionID:    uuid.New().String(),
		UserID:       userID,
		CreatedAt:    now,
		LastActivity: now,
		StateManager: NewStateManager(),
		Messages:     []Message{},
		Summarizer:   NewProgressiveSummarizer(5, 10, "gpt-4o-mini", 1500),
		IsActive:     true,
	}
}

// State returns the session's ConversationState for convenience.
func (s *Session) State() *ConversationState {
	return s.StateManager.State
}

// IsExpired returns true when the session has been idle longer than ttlMinutes.
func (s *Session) IsExpired(ttlMinutes int) bool {
	now := float64(time.Now().UnixNano()) / 1e9
	return (now - s.LastActivity) > float64(ttlMinutes)*60
}

// Touch updates the last-activity timestamp.
func (s *Session) Touch() {
	s.LastActivity = float64(time.Now().UnixNano()) / 1e9
}

// AddUserMessage appends a user message and updates conversation state.
func (s *Session) AddUserMessage(message string) {
	s.Touch()
	s.Messages = append(s.Messages, Message{Role: "user", Content: message})
	s.StateManager.ProcessUserTurn(message)
}

// AddAgentMessage appends an agent message, updates state, and feeds the
// turn to the progressive summarizer.
func (s *Session) AddAgentMessage(message string, toolCalls []ToolCall) {
	s.Touch()
	msg := Message{Role: "assistant", Content: message}
	if len(toolCalls) > 0 {
		calls := make([]any, len(toolCalls))
		for i, tc := range toolCalls {
			calls[i] = tc
		}
		msg.ToolCalls = calls
	}
	s.Messages = append(s.Messages, msg)
	s.StateManager.ProcessAgentTurn(message, toolCalls)

	// Find preceding user message for the summarizer
	userContent := ""
	for i := len(s.Messages) - 2; i >= 0; i-- {
		if s.Messages[i].Role == "user" {
			userContent = s.Messages[i].Content
			break
		}
	}
	s.Summarizer.AddTurn(userContent, message)
}

// BuildMessagesForLLM assembles the message list for the next LLM call.
func (s *Session) BuildMessagesForLLM(systemPrompt string, maxTokens int) []Message {
	augmented := s.StateManager.BuildSystemPromptWithState(systemPrompt)
	summaryCtx := s.Summarizer.GetContext()
	if summaryCtx != "" {
		augmented += "\n\n## Conversation History Summary\n" + summaryCtx
	}
	if s.InheritedContext != "" {
		augmented += "\n\n## Inherited from Previous Session\n" + s.InheritedContext
	}

	systemMsg := Message{Role: "system", Content: augmented}
	sysToks, _ := CountTokens(augmented, "gpt-4o")
	budget := maxTokens - sysToks

	var kept []Message
	running := 0
	for i := len(s.Messages) - 1; i >= 0; i-- {
		msg := s.Messages[i]
		toks, _ := CountTokens(msg.Content, "gpt-4o")
		if running+toks > budget {
			break
		}
		kept = append([]Message{msg}, kept...)
		running += toks
	}

	return append([]Message{systemMsg}, kept...)
}

// ToDict serialises the session for persistence.
func (s *Session) ToDict() SessionData {
	return SessionData{
		SessionID:        s.SessionID,
		UserID:           s.UserID,
		CreatedAt:        s.CreatedAt,
		LastActivity:     s.LastActivity,
		State:            s.StateManager.State.ToDict(),
		Messages:         append([]Message{}, s.Messages...),
		InheritedContext: s.InheritedContext,
		IsActive:         s.IsActive,
		Summarizer:       s.Summarizer.ToDict(),
	}
}

// SessionFromDict restores a Session from serialised data.
func SessionFromDict(data SessionData) *Session {
	s := NewSession(data.UserID)
	s.SessionID = data.SessionID
	s.CreatedAt = data.CreatedAt
	s.LastActivity = data.LastActivity
	s.StateManager.State = ConversationStateFromDict(data.State)
	s.Messages = append([]Message{}, data.Messages...)
	s.InheritedContext = data.InheritedContext
	s.IsActive = data.IsActive
	s.Summarizer = ProgressiveSummarizerFromDict(data.Summarizer)
	return s
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

// SessionManager manages conversation sessions across users and time.
type SessionManager struct {
	TTLMinutes  int
	MaxSessions int
	sessions    map[string]*Session
}

// NewSessionManager returns a SessionManager with the given TTL and capacity.
func NewSessionManager(ttlMinutes, maxSessions int) *SessionManager {
	return &SessionManager{
		TTLMinutes:  ttlMinutes,
		MaxSessions: maxSessions,
		sessions:    map[string]*Session{},
	}
}

// GetSession returns the active session for userID, creating one if expired or absent.
func (m *SessionManager) GetSession(userID string) *Session {
	if s, ok := m.sessions[userID]; ok && !s.IsExpired(m.TTLMinutes) {
		return s
	}
	return m.CreateSession(userID, "")
}

// CreateSession creates a new session, optionally inheriting context from inheritFrom.
func (m *SessionManager) CreateSession(userID, inheritFrom string) *Session {
	m.evictIfNeeded()
	s := NewSession(userID)

	if inheritFrom != "" {
		if parent, ok := m.sessions[inheritFrom]; ok {
			s.State().UserName = parent.State().UserName
			s.State().UserPreferences = copyMap(parent.State().UserPreferences)
			if parent.State().CurrentGoal != "" {
				completed := strings.Join(parent.State().SubtasksCompleted, ", ")
				if completed == "" {
					completed = "none"
				}
				s.InheritedContext = fmt.Sprintf(
					"Previous conversation goal: %s. Completed steps: %s.",
					parent.State().CurrentGoal, completed,
				)
			}
		}
	}

	m.sessions[userID] = s
	return s
}

// BranchSession creates a new session inheriting selected context from the current one.
// contextKeys may contain "user_profile", "goal_summary", "preferences".
func (m *SessionManager) BranchSession(userID string, contextKeys []string) *Session {
	parent := m.sessions[userID]
	m.evictIfNeeded()
	newSession := NewSession(userID)

	if parent != nil {
		wantsProfile := contains(contextKeys, "user_profile")
		wantsPreferences := contains(contextKeys, "preferences")
		wantsGoal := contains(contextKeys, "goal_summary")

		if wantsProfile || wantsPreferences {
			newSession.State().UserName = parent.State().UserName
			newSession.State().UserPreferences = copyMap(parent.State().UserPreferences)
		}
		if wantsProfile {
			newSession.State().UserProvidedInfo = copyMap(parent.State().UserProvidedInfo)
		}
		if wantsGoal && parent.State().CurrentGoal != "" {
			completed := strings.Join(parent.State().SubtasksCompleted, ", ")
			if completed == "" {
				completed = "none"
			}
			newSession.InheritedContext = fmt.Sprintf(
				"Previous conversation goal: %s. Completed: %s.",
				parent.State().CurrentGoal, completed,
			)
		}
	}

	m.sessions[userID] = newSession
	return newSession
}

// ResetSession resets a user's session, optionally preserving user identity.
func (m *SessionManager) ResetSession(userID string, keepIdentity bool) *Session {
	old := m.sessions[userID]
	m.evictIfNeeded()
	s := NewSession(userID)

	if keepIdentity && old != nil {
		s.State().UserName = old.State().UserName
		s.State().UserPreferences = copyMap(old.State().UserPreferences)
	}

	m.sessions[userID] = s
	return s
}

// EndSession marks a session inactive and removes it from memory.
func (m *SessionManager) EndSession(userID string) {
	if s, ok := m.sessions[userID]; ok {
		s.IsActive = false
		delete(m.sessions, userID)
	}
}

// CleanupExpired removes expired sessions and returns the count removed.
func (m *SessionManager) CleanupExpired() int {
	count := 0
	for uid, s := range m.sessions {
		if s.IsExpired(m.TTLMinutes) {
			delete(m.sessions, uid)
			count++
		}
	}
	return count
}

// GetActiveSessions returns the count of non-expired sessions.
func (m *SessionManager) GetActiveSessions() int {
	count := 0
	for _, s := range m.sessions {
		if !s.IsExpired(m.TTLMinutes) {
			count++
		}
	}
	return count
}

func (m *SessionManager) evictIfNeeded() {
	if len(m.sessions) < m.MaxSessions {
		return
	}
	var oldestUID string
	var oldestTime float64 = 1<<62 - 1
	for uid, s := range m.sessions {
		if s.LastActivity < oldestTime {
			oldestTime = s.LastActivity
			oldestUID = uid
		}
	}
	if oldestUID != "" {
		delete(m.sessions, oldestUID)
	}
}

// PersistSession serialises a session to a JSON string.
func (m *SessionManager) PersistSession(userID string) (string, error) {
	s, ok := m.sessions[userID]
	if !ok {
		return "{}", nil
	}
	b, err := json.Marshal(s.ToDict())
	if err != nil {
		return "", fmt.Errorf("marshal session: %w", err)
	}
	return string(b), nil
}

// RestoreSession deserialises a session from a JSON string.
func (m *SessionManager) RestoreSession(userID, data string) (*Session, error) {
	var sd SessionData
	if err := json.Unmarshal([]byte(data), &sd); err != nil {
		return nil, fmt.Errorf("unmarshal session: %w", err)
	}
	s := SessionFromDict(sd)
	s.UserID = userID
	m.sessions[userID] = s
	return s, nil
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

func copyMap(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
