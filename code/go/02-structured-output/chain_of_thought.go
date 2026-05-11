// Chain-of-thought prompting demonstration.
//
// Sends the same multi-step math problem with and without chain-of-thought
// instructions. Both calls use temperature=0 for a fair, deterministic comparison.
//
// Port of code/python/02-structured-output/chain_of_thought.py
// Run: go run . -mode=chain-of-thought
//
// See docs/01-foundations/02-prompt-engineering.md — "Chain-of-Thought"
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

const cotModel = "gpt-4o"

// cotProblem is a multi-step arithmetic problem that requires tracking multiple
// quantities — easy to get wrong without explicit step-by-step reasoning.
const cotProblem = "A store sells apples for $1.20 each and bananas for $0.40 each. " +
	"Alice buys 5 apples and 8 bananas. She pays with a $20 bill. " +
	"How much change does she receive?"

var cotWithoutMessages = []openai.ChatCompletionMessage{
	{Role: openai.ChatMessageRoleSystem, Content: "You are a math assistant. Answer concisely."},
	{Role: openai.ChatMessageRoleUser, Content: cotProblem},
}

var cotWithMessages = []openai.ChatCompletionMessage{
	{Role: openai.ChatMessageRoleSystem, Content: "You are a math assistant."},
	{
		Role:    openai.ChatMessageRoleUser,
		Content: cotProblem + "\n\nThink step by step before giving the final answer.",
	},
}

// cotCall sends a chat completion request and returns the trimmed response text.
func cotCall(ctx context.Context, messages []openai.ChatCompletionMessage, client *openai.Client) (string, error) {
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       cotModel,
		Messages:    messages,
		Temperature: 0,
	})
	if err != nil {
		return "", fmt.Errorf("chat completion: %w", err)
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

// RunChainOfThought demonstrates chain-of-thought prompting versus direct answering.
//
// Why chain-of-thought helps:
//   - The model must commit to intermediate values before reaching the answer.
//   - Wrong steps become visible and debuggable — critical in an agent loop.
//   - For complex multi-step problems, CoT can lift accuracy by 20-40% at temp=0.
//   - Trade-off: CoT costs more output tokens. Skip it for simple lookups.
func RunChainOfThought() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	fmt.Println("Problem:", cotProblem)
	fmt.Println(strings.Repeat("=", 60))

	fmt.Println("\n[WITHOUT chain-of-thought]")
	without, err := cotCall(ctx, cotWithoutMessages, client)
	if err != nil {
		log.Fatalf("without COT: %v", err)
	}
	fmt.Println(without)

	fmt.Println("\n[WITH chain-of-thought]")
	withCOT, err := cotCall(ctx, cotWithMessages, client)
	if err != nil {
		log.Fatalf("with COT: %v", err)
	}
	fmt.Println(withCOT)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("Observation: the CoT response shows every arithmetic step.")
	fmt.Println("If the answer is wrong, you can see exactly which step failed.")
	fmt.Println("In an agent loop this means a wrong tool call is debuggable,")
	fmt.Println("not just a black-box failure.")
}
