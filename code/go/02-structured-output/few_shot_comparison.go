// Zero-shot vs few-shot classification comparison.
//
// Demonstrates the reliability gain from few-shot examples for sentiment
// classification. Prints both results side-by-side with token counts so
// you can measure the reliability/cost trade-off directly.
//
// Port of code/python/02-structured-output/few_shot_comparison.py
// Run: go run . -mode=few-shot
//
// See docs/01-foundations/02-prompt-engineering.md — "Few-Shot Prompting"
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	tiktoken "github.com/pkoukk/tiktoken-go"
	openai "github.com/sashabaranov/go-openai"
)

const fewShotModel = "gpt-4o"

const zeroShotSystem = "Classify the sentiment of the following text as exactly one of: " +
	"Positive, Negative, or Neutral."

const fewShotSystem = `Classify the sentiment of the following text as exactly one of: Positive, Negative, or Neutral.
Respond with exactly one word.

Examples:
Text: "I love this product!" → Positive
Text: "This is absolutely terrible." → Negative
Text: "It arrived on time." → Neutral`

// fewShotTestInput is a tricky hedged input — models often disagree zero-shot.
const fewShotTestInput = "The new update is fine, I guess. Not bad, but nothing to get excited about."

// countFewShotTokens estimates the prompt token count using tiktoken.
// Falls back to a rough character-based estimate if the encoding is unavailable.
func countFewShotTokens(messages []openai.ChatCompletionMessage) int {
	enc, err := tiktoken.EncodingForModel(fewShotModel)
	if err != nil {
		total := 0
		for _, m := range messages {
			total += 3 + len(m.Content)/4 + len(m.Role)/4
		}
		return total + 3
	}
	total := 0
	for _, m := range messages {
		total += 3 +
			len(enc.Encode(m.Role, nil, nil)) +
			len(enc.Encode(m.Content, nil, nil))
	}
	return total + 3
}

// fewShotClassify sends a zero-token classification request and returns
// the label plus estimated prompt token count.
func fewShotClassify(ctx context.Context, systemPrompt, text string, client *openai.Client) (string, int, error) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
		{Role: openai.ChatMessageRoleUser, Content: fmt.Sprintf("Text: %q", text)},
	}
	tokens := countFewShotTokens(messages)
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       fewShotModel,
		Messages:    messages,
		Temperature: 0,
		MaxTokens:   10,
	})
	if err != nil {
		return "", tokens, fmt.Errorf("classify: %w", err)
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), tokens, nil
}

// RunFewShotComparison demonstrates zero-shot vs few-shot sentiment classification.
//
// When few-shot is worth the extra tokens:
//   - Classification where the output format MUST be a single word/label.
//   - Tasks where the model often adds unwanted explanation.
//   - Edge cases where zero-shot returns inconsistent casing or phrasing.
//
// For anything more complex, switch to structured output (json_schema).
func RunFewShotComparison() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	fmt.Printf("Input: %q\n\n", fewShotTestInput)
	fmt.Printf("%-12s %-12s %12s\n", "Approach", "Result", "Tokens sent")
	fmt.Println(strings.Repeat("-", 40))

	zeroResult, zeroTokens, err := fewShotClassify(ctx, zeroShotSystem, fewShotTestInput, client)
	if err != nil {
		log.Fatalf("zero-shot: %v", err)
	}
	fmt.Printf("%-12s %-12s %12d\n", "Zero-shot", zeroResult, zeroTokens)

	fewResult, fewTokens, err := fewShotClassify(ctx, fewShotSystem, fewShotTestInput, client)
	if err != nil {
		log.Fatalf("few-shot: %v", err)
	}
	fmt.Printf("%-12s %-12s %12d\n", "Few-shot", fewResult, fewTokens)

	overhead := fewTokens - zeroTokens
	fmt.Printf("\nFew-shot overhead: +%d tokens per request\n", overhead)
}
