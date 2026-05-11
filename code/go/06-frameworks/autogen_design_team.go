// autogen_design_team.go — AutoGen-equivalent conversational design team in Go.
//
// Python AutoGen has no official Go port. This file implements the same
// CONCEPT — a group-chat orchestrator that routes messages between
// specialised agents until consensus is reached — using Go-native code.
//
// Agents in the team:
//   - ProductManager  : defines user requirements and acceptance criteria
//   - Architect       : designs high-level system structure
//   - SecurityReviewer: identifies risks and enforces least-privilege principle
//
// The orchestrator runs up to maxConversationRuns rounds; agents signal
// "APPROVED" when the spec is complete. If no consensus is reached, the
// best draft after the final round is returned.
//
// Run:
//
//	go run autogen_design_team.go hybrid_rag_agent.go langgraph_alternative.go \
//	           multi_agent_from_scratch.go framework_comparison.go framework_advisor.go \
//	           framework_extraction.go over_engineering_detector.go langsmith_tracer.go
//
// See: docs/06-frameworks-in-practice/03-crewai-autogen.md
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

// DesignResult holds the output of a conversational design session.
type DesignResult struct {
	Requirement         string
	FinalSpec           string
	ConversationHistory []map[string]string // keys: role, name, content
	RoundsTaken         int
	DecisionsMade       []string
	ExecutionTimeSec    float64
	TokenUsage          int
}

// DesignConversationAnalysis analyses the quality of a design conversation.
type DesignConversationAnalysis struct {
	SpeakerTurns      map[string]int
	ProductiveRounds  int
	CircularRounds    int
	Decisions         []string
	DeadlocksDetected int
	TotalRounds       int
}

// ---------------------------------------------------------------------------
// Design team orchestrator
// ---------------------------------------------------------------------------

// dtAgentSpec defines one participant in the design conversation.
type dtAgentSpec struct {
	Name      string
	Role      string
	SystemMsg string
}

// dtDesignAgents returns the three agents of the team.
func dtDesignAgents() []dtAgentSpec {
	return []dtAgentSpec{
		{
			Name: "ProductManager",
			Role: "product manager",
			SystemMsg: "You are a product manager. Your job is to:\n" +
				"1. Define clear user requirements and acceptance criteria.\n" +
				"2. Prioritise features (MUST/SHOULD/NICE-TO-HAVE).\n" +
				"3. Challenge over-engineering.\n" +
				"Respond concisely. When the spec is complete write 'APPROVED' on the last line.",
		},
		{
			Name: "Architect",
			Role: "solutions architect",
			SystemMsg: "You are a solutions architect. Your job is to:\n" +
				"1. Design a scalable, maintainable system structure.\n" +
				"2. Choose appropriate data stores and APIs.\n" +
				"3. Identify integration points and dependencies.\n" +
				"Respond concisely. When the design is finalized write 'APPROVED' on the last line.",
		},
		{
			Name: "SecurityReviewer",
			Role: "application security engineer",
			SystemMsg: "You are an application security engineer. Your job is to:\n" +
				"1. Identify security risks (OWASP Top-10, data exposure, authentication gaps).\n" +
				"2. Enforce least-privilege access and data minimisation.\n" +
				"3. Require mitigations before approving.\n" +
				"Respond concisely. When all risks are mitigated write 'APPROVED' on the last line.",
		},
	}
}

// dtRunAgent calls the LLM for a single agent turn.
func dtRunAgent(ctx context.Context, client *openai.Client, spec dtAgentSpec, history []openai.ChatCompletionMessage) (string, int, error) {
	msgs := append([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: spec.SystemMsg},
	}, history...)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:     "gpt-4o-mini",
		Messages:  msgs,
		MaxTokens: 400,
	})
	if err != nil {
		return "", 0, err
	}
	return resp.Choices[0].Message.Content, resp.Usage.TotalTokens, nil
}

// dtRunDesignSession orchestrates the group-chat conversation.
func dtRunDesignSession(ctx context.Context, client *openai.Client, requirement string) (DesignResult, error) {
	t0 := time.Now()
	maxRounds := 4
	agents := dtDesignAgents()

	result := DesignResult{Requirement: requirement}

	// history for the LLM context (grows with each turn)
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Design requirement:\n" + requirement},
	}
	// readable record for the result
	var convHistory []map[string]string
	convHistory = append(convHistory, map[string]string{
		"role": "user", "name": "Initiator", "content": requirement,
	})

	approvals := make(map[string]bool)

	for round := 1; round <= maxRounds; round++ {
		result.RoundsTaken = round
		allApproved := true

		for _, agent := range agents {
			if approvals[agent.Name] {
				continue
			}

			reply, tokens, err := dtRunAgent(ctx, client, agent, history)
			if err != nil {
				return result, fmt.Errorf("agent %s round %d: %w", agent.Name, round, err)
			}
			result.TokenUsage += tokens

			convHistory = append(convHistory, map[string]string{
				"role": "assistant", "name": agent.Name, "content": reply,
			})
			history = append(history, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: fmt.Sprintf("[%s]: %s", agent.Name, reply),
			})

			if strings.Contains(strings.ToUpper(reply), "APPROVED") {
				approvals[agent.Name] = true
			} else {
				allApproved = false
			}
		}

		if allApproved {
			break
		}
	}

	result.ConversationHistory = convHistory
	result.DecisionsMade = dtExtractDecisions(convHistory)
	result.FinalSpec = dtBuildFinalSpec(convHistory, result.DecisionsMade)
	result.ExecutionTimeSec = time.Since(t0).Seconds()

	return result, nil
}

// ---------------------------------------------------------------------------
// Analysis helpers
// ---------------------------------------------------------------------------

// dtExtractDecisions scans the conversation for lines that look like decisions.
func dtExtractDecisions(history []map[string]string) []string {
	decisionIndicators := []string{
		"we should", "we will", "we need", "must use", "must not",
		"decision:", "agreed:", "require", "will use", "will not",
	}
	seen := make(map[string]bool)
	var decisions []string

	for _, msg := range history {
		if msg["role"] != "assistant" {
			continue
		}
		for _, line := range strings.Split(msg["content"], "\n") {
			lineLower := strings.ToLower(strings.TrimSpace(line))
			if lineLower == "" || lineLower == "approved" {
				continue
			}
			for _, indicator := range decisionIndicators {
				if strings.Contains(lineLower, indicator) {
					key := strings.ToLower(line[:min(len(line), 80)])
					if !seen[key] {
						seen[key] = true
						trimmed := strings.TrimSpace(line)
						if len(trimmed) > 120 {
							trimmed = trimmed[:120] + "…"
						}
						decisions = append(decisions, trimmed)
					}
					break
				}
			}
		}
	}
	return decisions
}

// dtBuildFinalSpec assembles the specification from the last assistant message
// in the conversation that contains more than one sentence.
func dtBuildFinalSpec(history []map[string]string, decisions []string) string {
	var lastSubstantive string
	for _, msg := range history {
		if msg["role"] == "assistant" {
			content := strings.TrimSpace(msg["content"])
			// Remove "APPROVED" trailing line
			content = strings.TrimSuffix(strings.TrimSpace(strings.ToUpper(content)), "APPROVED")
			content = strings.TrimSpace(msg["content"])
			if len(content) > 100 {
				lastSubstantive = content
			}
		}
	}

	var sb strings.Builder
	sb.WriteString("=== DESIGN SPECIFICATION ===\n\n")
	if lastSubstantive != "" {
		sb.WriteString(lastSubstantive)
		sb.WriteString("\n\n")
	}
	if len(decisions) > 0 {
		sb.WriteString("=== KEY DECISIONS ===\n")
		for i, d := range decisions {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, d))
		}
	}
	return sb.String()
}

// analyzeDesignConversation returns statistics about the conversation quality.
func analyzeDesignConversation(history []map[string]string) DesignConversationAnalysis {
	analysis := DesignConversationAnalysis{
		SpeakerTurns: make(map[string]int),
		Decisions:    dtExtractDecisions(history),
	}
	analysis.TotalRounds = len(history)

	prevContent := make(map[string]string)
	for _, msg := range history {
		if msg["role"] != "assistant" {
			continue
		}
		name := msg["name"]
		analysis.SpeakerTurns[name]++

		content := strings.TrimSpace(msg["content"])
		if prev, ok := prevContent[name]; ok {
			// Circular if the message is very similar to the previous one from the same agent
			if len(content) > 0 && len(prev) > 0 {
				// Simple overlap check
				shorter, longer := prev, content
				if len(shorter) > len(longer) {
					shorter, longer = longer, shorter
				}
				if len(shorter) > 20 && strings.Contains(strings.ToLower(longer), strings.ToLower(shorter[:20])) {
					analysis.CircularRounds++
				}
			}
		} else {
			analysis.ProductiveRounds++
		}
		prevContent[name] = content
	}
	return analysis
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// runAutogenDesignTeamDemo is the demo entry point (no main()).
func runAutogenDesignTeamDemo(ctx context.Context, client *openai.Client) {
	requirements := []string{
		"Build a real-time dashboard that shows live KPIs from our PostgreSQL database. " +
			"The dashboard needs to handle 1000 concurrent viewers with sub-second refresh.",
		"Create a customer-facing AI chatbot for product support. " +
			"It must answer questions about our product catalogue and escalate complex issues to human agents.",
	}

	for i, req := range requirements {
		fmt.Printf("\n%s\n  DESIGN SESSION %d\n%s\n\n", strings.Repeat("═", 60), i+1, strings.Repeat("═", 60))
		fmt.Printf("Requirement:\n  %s\n\n", req)

		result, err := dtRunDesignSession(ctx, client, req)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("Rounds taken    : %d\n", result.RoundsTaken)
		fmt.Printf("Tokens used     : %d\n", result.TokenUsage)
		fmt.Printf("Execution time  : %.2fs\n", result.ExecutionTimeSec)
		fmt.Printf("Decisions made  : %d\n", len(result.DecisionsMade))

		if len(result.DecisionsMade) > 0 {
			fmt.Printf("\nKey Decisions:\n")
			for j, d := range result.DecisionsMade {
				fmt.Printf("  %d. %s\n", j+1, d)
			}
		}

		fmt.Printf("\nFinal Spec (preview):\n")
		spec := result.FinalSpec
		if len(spec) > 500 {
			spec = spec[:500] + "\n…(truncated)"
		}
		fmt.Printf("%s\n", spec)

		analysis := analyzeDesignConversation(result.ConversationHistory)
		fmt.Printf("\nConversation Analysis:\n")
		fmt.Printf("  Speaker turns   : %v\n", analysis.SpeakerTurns)
		fmt.Printf("  Productive turns: %d\n", analysis.ProductiveRounds)
		fmt.Printf("  Circular turns  : %d\n", analysis.CircularRounds)
		fmt.Printf("  Decisions found : %d\n", len(analysis.Decisions))
	}
}
