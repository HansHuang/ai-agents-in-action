// Reusable parse-validate-retry handler for structured LLM extraction.
//
// ExtractWithRetry is a generic function (requires Go 1.18+) that:
//   - Calls the OpenAI chat completions API with a caller-supplied JSON schema
//   - Passes the raw response bytes to a caller-supplied validate function
//   - On validation failure, appends a human-readable error to the message history
//     and retries, giving the model a chance to self-correct
//   - Logs each attempt number, success/failure, and error details
//   - Returns an error when maxRetries is exhausted
//
// Import and reuse across any extraction task in this repo.
//
// See docs/01-foundations/03-structured-output.md — "The Parse-Validate-Retry Pattern"

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

// ExtractWithRetry calls the LLM and validates the response using the provided
// validate function, retrying up to maxRetries times on failure.
//
// Type parameter T is the expected output type; validate must unmarshal the
// raw JSON bytes into *T and return a non-nil error to trigger a retry.
//
// Parameters:
//
//	messages    - Initial message array; extended in-place on each retry.
//	schemaJSON  - JSON Schema as a map; marshalled to bytes and sent to the API.
//	schemaName  - Name field in the json_schema response_format object.
//	validate    - Function that parses raw bytes into *T; returns non-nil error on failure.
//	maxRetries  - Number of retry attempts after the first (0 = try once only).
//
// Returns (*T, nil) on success, or (nil, error) after maxRetries exhausted.
func ExtractWithRetry[T any](
	messages []openai.ChatCompletionMessage,
	schemaJSON map[string]interface{},
	schemaName string,
	validate func([]byte) (*T, error),
	maxRetries int,
) (*T, error) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	schemaBytes, err := json.Marshal(schemaJSON)
	if err != nil {
		return nil, fmt.Errorf("ExtractWithRetry: marshal schema: %w", err)
	}

	workingMessages := make([]openai.ChatCompletionMessage, len(messages))
	copy(workingMessages, messages)

	for attempt := 1; attempt <= maxRetries+1; attempt++ {
		log.Printf("[ExtractWithRetry] Attempt %d/%d", attempt, maxRetries+1)

		resp, apiErr := client.CreateChatCompletion(context.Background(),
			openai.ChatCompletionRequest{
				Model:    openai.GPT4o,
				Messages: workingMessages,
				ResponseFormat: &openai.ChatCompletionResponseFormat{
					Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
					JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
						Name:   schemaName,
						Schema: json.RawMessage(schemaBytes),
						Strict: true,
					},
				},
			})
		if apiErr != nil {
			// API errors are not retried — they indicate a structural problem.
			return nil, fmt.Errorf("[ExtractWithRetry] API error: %w", apiErr)
		}

		raw := resp.Choices[0].Message.Content
		result, valErr := validate([]byte(raw))
		if valErr == nil {
			log.Printf("[ExtractWithRetry] Attempt %d succeeded", attempt)
			return result, nil
		}

		log.Printf("[ExtractWithRetry] Attempt %d failed — %v", attempt, valErr)

		if attempt > maxRetries {
			return nil, fmt.Errorf("[ExtractWithRetry] max retries exceeded: last error: %w", valErr)
		}

		workingMessages = append(workingMessages,
			openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: raw},
			openai.ChatCompletionMessage{
				Role: openai.ChatMessageRoleUser,
				Content: fmt.Sprintf(
					"Your response did not match the required schema. Error: %v. Please fix and retry.",
					valErr,
				),
			},
		)
	}

	return nil, fmt.Errorf("[ExtractWithRetry] unreachable")
}
