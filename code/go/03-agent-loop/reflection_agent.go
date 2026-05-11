// reflection_agent.go — Reflection agent: Generate → Reflect → Revise loop (Go port).
//
// Implements the self-critique pattern described in:
// docs/02-the-agent-loop/03-planning-strategies.md
//
// The agent generates an initial answer, then evaluates it with a separate
// critic call. If the score is below the threshold, it revises the answer and
// critiques again, up to maxReflections times.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	reflectionModel       = "gpt-4o"
	maxReflections        = 3
	satisfactionThreshold = 8
)

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

// CritiqueResult is the structured output from the critic LLM call.
type CritiqueResult struct {
	OverallScore int      `json:"overall_score"`
	IsSatisfied  bool     `json:"is_satisfied"`
	Feedback     string   `json:"feedback"`
	Strengths    []string `json:"strengths"`
	Weaknesses   []string `json:"weaknesses"`
}

// IterationRecord is a single generate-or-revise iteration with its critique.
type IterationRecord struct {
	Iteration int
	Answer    string
	Critique  *CritiqueResult
}

// ---------------------------------------------------------------------------
// System prompts
// ---------------------------------------------------------------------------

const generatorSystem = `Answer the user's question thoroughly and accurately.
Provide specific details and cite sources or data where possible.
Structure your response clearly with headings if the topic warrants it.`

const criticSystem = `You are a strict quality reviewer. Evaluate the answer against the original question.

Score on 1-10 for:
1. Completeness: Did it answer everything asked?
2. Accuracy: Are there factual errors or unsupported claims?
3. Clarity: Is it easy to understand?
4. Structure: Is it well-organised with appropriate formatting?
5. Actionability: Can the user act on this information?

Output ONLY valid JSON with this exact schema (no markdown fences):
{
  "overall_score": <int 1-10>,
  "is_satisfied": <bool>,
  "feedback": "<specific, actionable critique>",
  "strengths": ["<strength 1>", "..."],
  "weaknesses": ["<weakness 1>", "..."]
}

Set is_satisfied=true when overall_score >= 8 and there are no major weaknesses.
Be honest and specific. Vague feedback like 'good answer' is not acceptable.`

// ---------------------------------------------------------------------------
// ReflectionAgent
// ---------------------------------------------------------------------------

// ReflectionAgent implements the generate → reflect → revise loop.
type ReflectionAgent struct {
	client *openai.Client
	_      struct{}
}

// NewReflectionAgent creates a new ReflectionAgent.
func NewReflectionAgent() *ReflectionAgent {
	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	return &ReflectionAgent{client: &client}
}

// Generate calls the LLM to produce an initial answer or revision.
func (a *ReflectionAgent) Generate(ctx context.Context, question, feedback string) (string, error) {
	userContent := question
	if feedback != "" {
		userContent = fmt.Sprintf("%s\n\nPrevious feedback to address:\n%s", question, feedback)
	}
	resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(generatorSystem),
			openai.UserMessage(userContent),
		},
	})
	if err != nil {
		return "", fmt.Errorf("generate: %w", err)
	}
	return resp.Choices[0].Message.Content, nil
}

// Critique evaluates an answer and returns structured feedback.
func (a *ReflectionAgent) Critique(ctx context.Context, question, answer string) (*CritiqueResult, error) {
	prompt := fmt.Sprintf("Question: %s\n\nAnswer to evaluate:\n%s", question, answer)
	resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: openai.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(criticSystem),
			openai.UserMessage(prompt),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("critique: %w", err)
	}
	raw := resp.Choices[0].Message.Content
	var critique CritiqueResult
	if err := json.Unmarshal([]byte(raw), &critique); err != nil {
		return nil, fmt.Errorf("parse critique JSON: %w\nRaw: %s", err, raw)
	}
	return &critique, nil
}

// Run executes the full generate → reflect → revise loop.
// Returns the final answer and the iteration history.
func (a *ReflectionAgent) Run(ctx context.Context, question string) (string, []IterationRecord) {
	var history []IterationRecord
	feedback := ""

	for i := 1; i <= maxReflections+1; i++ {
		log.Printf("[Reflection] Iteration %d: generating answer", i)
		answer, err := a.Generate(ctx, question, feedback)
		if err != nil {
			log.Printf("[Reflection] Generate failed: %v", err)
			break
		}

		record := IterationRecord{Iteration: i, Answer: answer}

		if i > maxReflections {
			// Final iteration — no further critique
			history = append(history, record)
			log.Printf("[Reflection] Max reflections reached, returning final answer")
			return answer, history
		}

		log.Printf("[Reflection] Iteration %d: critiquing answer", i)
		critique, err := a.Critique(ctx, question, answer)
		if err != nil {
			log.Printf("[Reflection] Critique failed: %v — returning current answer", err)
			history = append(history, record)
			return answer, history
		}

		record.Critique = critique
		history = append(history, record)

		log.Printf("[Reflection] Score: %d/10, satisfied: %v", critique.OverallScore, critique.IsSatisfied)
		if critique.IsSatisfied || critique.OverallScore >= satisfactionThreshold {
			log.Printf("[Reflection] Satisfied after %d iteration(s)", i)
			return answer, history
		}

		feedback = critique.Feedback
	}

	// Return best answer from history
	if len(history) > 0 {
		return history[len(history)-1].Answer, history
	}
	return "", history
}

// RunReflectionAgent demonstrates the reflection agent with a sample question.
func RunReflectionAgent() {
	agent := NewReflectionAgent()
	question := "What are the key differences between RAG and fine-tuning for improving LLM outputs?"

	fmt.Printf("Question: %s\n%s\n", question, repeatStr("=", 60))

	answer, history := agent.Run(context.Background(), question)

	fmt.Printf("\nFinal Answer (%d iteration(s)):\n%s\n", len(history), answer)
	fmt.Println("\n--- Iteration History ---")
	for _, rec := range history {
		fmt.Printf("\nIteration %d:\n", rec.Iteration)
		preview := rec.Answer
		if len(preview) > 200 {
			preview = preview[:200] + "…"
		}
		fmt.Printf("  Answer: %s\n", preview)
		if rec.Critique != nil {
			fmt.Printf("  Score:  %d/10\n", rec.Critique.OverallScore)
			fmt.Printf("  Feedback: %s\n", rec.Critique.Feedback)
		}
	}
}
