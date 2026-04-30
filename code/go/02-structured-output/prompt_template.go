package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	tiktoken "github.com/pkoukk/tiktoken-go"
	openai "github.com/sashabaranov/go-openai"
)

const model = "gpt-4o"

const systemPromptTemplate = "You are a technical summarizer.\n" +
	"Summarize the following article in 3 bullet points.\n" +
	"Focus on: %s"

const userPromptTemplate = "Article:\n%s"

// message mirrors the OpenAI chat completions messages format.
type message struct {
	Role    string
	Content string
}

// buildSystemPrompt returns the filled system prompt for the given focus area.
func buildSystemPrompt(focusArea string) (string, error) {
	if strings.TrimSpace(focusArea) == "" {
		return "", errors.New("focusArea must not be empty")
	}
	return fmt.Sprintf(systemPromptTemplate, focusArea), nil
}

// buildUserPrompt returns the filled user prompt for the given article text.
func buildUserPrompt(articleText string) (string, error) {
	if strings.TrimSpace(articleText) == "" {
		return "", errors.New("articleText must not be empty")
	}
	return fmt.Sprintf(userPromptTemplate, articleText), nil
}

// buildMessages returns a messages slice ready for chat.completions.create.
func buildMessages(focusArea, articleText string) ([]message, error) {
	sys, err := buildSystemPrompt(focusArea)
	if err != nil {
		return nil, err
	}
	usr, err := buildUserPrompt(articleText)
	if err != nil {
		return nil, err
	}
	return []message{
		{Role: "system", Content: sys},
		{Role: "user", Content: usr},
	}, nil
}

// countTokens returns the token cost of a messages slice (includes API overhead).
// Accounts for the per-message overhead (3 tokens) and reply primer (3 tokens).
func countTokens(messages []message, modelName string) (int, error) {
	enc, err := tiktoken.EncodingForModel(modelName)
	if err != nil {
		return 0, err
	}
	const tokensPerMessage = 3
	total := 0
	for _, msg := range messages {
		total += tokensPerMessage
		total += len(enc.Encode(msg.Role, nil, nil))
		total += len(enc.Encode(msg.Content, nil, nil))
	}
	total += 3 // reply is primed with <|start|>assistant<|message|>
	return total, nil
}

func runPromptTemplate() {
	focusArea := "practical implementation details"
	articleText := "AI agents are software systems that use large language models as their " +
		"reasoning engine. Unlike chatbots, agents can take actions: call APIs, " +
		"search the web, write code, and orchestrate other agents. The key " +
		"architectural pattern is the agent loop: perceive, think, act, observe. " +
		"Production agents require harness engineering — input validation, retry " +
		"logic, output guardrails, and human-in-the-loop checkpoints."

	messages, err := buildMessages(focusArea, articleText)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	tokens, err := countTokens(messages, model)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("Token count before sending: %d\n", tokens)

	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	var chatMessages []openai.ChatCompletionMessage
	for _, msg := range messages {
		chatMessages = append(chatMessages, openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model:       model,
		Messages:    chatMessages,
		Temperature: 0.3,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Println("\nResponse:")
	fmt.Println(resp.Choices[0].Message.Content)
	fmt.Printf("\nActual tokens used — prompt: %d, completion: %d\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
}
