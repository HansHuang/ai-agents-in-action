// Structured output extraction with automatic parse-validate-retry.
//
// Go port of code/python/02-structured-output/instructor_extraction.py.
// Python uses the Instructor library (instructor.from_openai()) to add
// automatic parse-validate-retry around the OpenAI client. This file
// achieves the same pattern using ExtractWithRetry from retry_handler.go.
//
// Run: go run . -mode=instructor
//
// See docs/01-foundations/03-structured-output.md — "Language-Specific Patterns"
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// InstructorSentiment is the validated output schema, equivalent to Python's
// SentimentResponse Pydantic model in instructor_extraction.py.
type InstructorSentiment struct {
	Sentiment  string   `json:"sentiment"`
	Confidence float64  `json:"confidence"`
	KeyPhrases []string `json:"key_phrases,omitempty"`
}

// instructorSentimentSchema is the JSON Schema for InstructorSentiment.
// All properties except key_phrases are required; additionalProperties is false
// for OpenAI strict-mode compatibility.
var instructorSentimentSchema = map[string]interface{}{
	"type": "object",
	"properties": map[string]interface{}{
		"sentiment": map[string]interface{}{
			"type": "string",
			"enum": []string{"positive", "negative", "neutral"},
			"description": "The overall sentiment of the text. Must be exactly one of: " +
				"positive (favorable, happy, satisfied), " +
				"negative (unfavorable, unhappy, dissatisfied), or " +
				"neutral (neither clearly positive nor negative).",
		},
		"confidence": map[string]interface{}{
			"type":    "number",
			"minimum": 0.0,
			"maximum": 1.0,
			"description": "How confident you are in the classification, as a float " +
				"between 0.0 (no confidence) and 1.0 (completely certain).",
		},
		"key_phrases": map[string]interface{}{
			"type":  "array",
			"items": map[string]interface{}{"type": "string"},
			"description": "Up to 5 short phrases from the text that most influenced " +
				"the sentiment classification. Omit if no clear phrases stand out.",
		},
	},
	"required":             []string{"sentiment", "confidence", "key_phrases"},
	"additionalProperties": false,
}

// ExtractSentimentInstructor extracts sentiment from text with up to 2 automatic
// retries on validation failure, mirroring Instructor's max_retries=2 behaviour.
//
// Returns a validated InstructorSentiment or an error after retries are exhausted.
func ExtractSentimentInstructor(text string) (*InstructorSentiment, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text must not be empty")
	}

	messages := []openai.ChatCompletionMessage{
		{
			Role: openai.ChatMessageRoleSystem,
			Content: "You are a precise sentiment analysis engine. " +
				"Classify the sentiment of the user's text accurately.",
		},
		{Role: openai.ChatMessageRoleUser, Content: text},
	}

	return ExtractWithRetry(
		messages,
		instructorSentimentSchema,
		"sentiment_response",
		func(raw []byte) (*InstructorSentiment, error) {
			var r InstructorSentiment
			if err := json.Unmarshal(raw, &r); err != nil {
				return nil, fmt.Errorf("unmarshal: %w", err)
			}
			switch r.Sentiment {
			case "positive", "negative", "neutral":
			default:
				return nil, fmt.Errorf("sentiment %q is not one of positive/negative/neutral", r.Sentiment)
			}
			if r.Confidence < 0 || r.Confidence > 1 {
				return nil, fmt.Errorf("confidence %.2f is out of range [0.0, 1.0]", r.Confidence)
			}
			return &r, nil
		},
		2, // max_retries
	)
}

// RunInstructorExtraction demonstrates Instructor-style structured extraction.
func RunInstructorExtraction() {
	tests := []string{
		"I absolutely love this, it changed my life!",
		"It's fine I guess, nothing special.",
		"Terrible product, broke after one day.",
	}

	for _, text := range tests {
		result, err := ExtractSentimentInstructor(text)
		if err != nil {
			log.Printf("extraction failed for %q: %v\n", text, err)
			continue
		}
		fmt.Printf("Text       : %q\n", text)
		fmt.Printf("Sentiment  : %s\n", result.Sentiment)
		fmt.Printf("Confidence : %.2f\n", result.Confidence)
		fmt.Printf("Key Phrases: %v\n", result.KeyPhrases)
		fmt.Println()
	}
}
