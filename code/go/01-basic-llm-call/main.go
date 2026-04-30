// Token counting for OpenAI-compatible models using tiktoken-go.
//
// Shows how to count tokens for a plain string and for a messages slice
// (the format used by chat.completions.create). The messages slice count
// mirrors what the API actually charges you for.
package main

import (
	"fmt"
	"log"

	tiktoken "github.com/pkoukk/tiktoken-go"
)

const model = "gpt-4o"

// countTokens returns the number of tokens in text for the given model.
func countTokens(text, model string) (int, error) {
	enc, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return 0, err
	}
	return len(enc.Encode(text, nil, nil)), nil
}

// message mirrors the OpenAI chat completions messages format.
type message struct {
	Role    string
	Name    string // optional
	Content string
}

// countMessagesTokens returns the token cost of a messages slice for chat
// completions. Accounts for the per-message overhead (3 tokens) and reply
// primer (3 tokens) that the API adds automatically.
// See: https://platform.openai.com/docs/guides/chat/managing-tokens
func countMessagesTokens(messages []message, model string) (int, error) {
	enc, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return 0, err
	}
	const tokensPerMessage = 3
	const tokensPerName = 1
	total := 0
	for _, msg := range messages {
		total += tokensPerMessage
		total += len(enc.Encode(msg.Role, nil, nil))
		total += len(enc.Encode(msg.Content, nil, nil))
		if msg.Name != "" {
			total += len(enc.Encode(msg.Name, nil, nil))
			total += tokensPerName
		}
	}
	total += 3 // reply is primed with <|start|>assistant<|message|>
	return total, nil
}

func main() {
	text := "The quick brown fox jumps over the lazy dog."
	n, err := countTokens(text, model)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Text  : %q\n", text)
	fmt.Printf("Tokens: %d\n", n)

	messages := []message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of France?"},
	}
	total, err := countMessagesTokens(messages, model)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nMessages array token count: %d\n", total)
}
