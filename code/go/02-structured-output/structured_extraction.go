// Structured output extraction using Go struct tags and the OpenAI API.
//
// Demonstrates:
//   - Defining an output struct with json and jsonschema tags
//   - GenerateJSONSchema(): reflection-based schema generation from struct tags
//   - extractSentiment(): strict json_schema response_format, json.Unmarshal,
//     manual enum/range validation, and a retry loop (max 2 retries)
//
// Run:  go run . -mode=structured
//
// See docs/01-foundations/03-structured-output.md — "Language-Specific Patterns"

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

// SentimentResponse is the structured output we expect from the model.
// json tags control the JSON field names.
// jsonschema tags drive GenerateJSONSchema (see below).
type SentimentResponse struct {
	Sentiment  string   `json:"sentiment"   jsonschema:"enum=positive,enum=negative,enum=neutral,description=Overall sentiment: positive / negative / neutral"`
	Confidence float64  `json:"confidence"  jsonschema:"minimum=0,maximum=1,description=Confidence 0.0–1.0"`
	KeyPhrases []string `json:"key_phrases,omitempty" jsonschema:"description=Up to 5 key phrases that influenced the classification"`
}

// ---------------------------------------------------------------------------
// GenerateJSONSchema — reflection-based JSON Schema builder
// ---------------------------------------------------------------------------

// GenerateJSONSchema generates a strict JSON Schema (draft 7 compatible) from
// a struct using its json and jsonschema field tags.
//
// Supported jsonschema tag keys (comma-separated key=value pairs):
//
//	description=<text>
//	enum=<value>          (may appear multiple times for multiple enum values)
//	minimum=<number>
//	maximum=<number>
func GenerateJSONSchema(v interface{}) map[string]interface{} {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	properties := map[string]interface{}{}
	required := []string{}

	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)

		// Determine the JSON name and whether the field is optional.
		jsonTag := f.Tag.Get("json")
		name := f.Name
		optional := false
		if jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "" && parts[0] != "-" {
				name = parts[0]
			}
			for _, p := range parts[1:] {
				if p == "omitempty" {
					optional = true
				}
			}
		}
		if name == "-" {
			continue
		}

		fieldSchema := goTypeToSchema(f.Type)

		// Apply jsonschema annotations.
		schemaTag := f.Tag.Get("jsonschema")
		if schemaTag != "" {
			for _, part := range strings.Split(schemaTag, ",") {
				kv := strings.SplitN(part, "=", 2)
				if len(kv) != 2 {
					continue
				}
				k, val := kv[0], kv[1]
				switch k {
				case "description":
					fieldSchema["description"] = val
				case "minimum":
					if n, err := strconv.ParseFloat(val, 64); err == nil {
						fieldSchema["minimum"] = n
					}
				case "maximum":
					if n, err := strconv.ParseFloat(val, 64); err == nil {
						fieldSchema["maximum"] = n
					}
				case "enum":
					enums, _ := fieldSchema["enum"].([]interface{})
					fieldSchema["enum"] = append(enums, val)
				}
			}
		}

		properties[name] = fieldSchema
		if !optional {
			required = append(required, name)
		}
	}

	schema := map[string]interface{}{
		"type":                 "object",
		"properties":           properties,
		"additionalProperties": false,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// goTypeToSchema returns a minimal JSON Schema fragment for a Go type.
func goTypeToSchema(t reflect.Type) map[string]interface{} {
	if t.Kind() == reflect.Ptr {
		return goTypeToSchema(t.Elem())
	}
	if t.Kind() == reflect.Slice {
		return map[string]interface{}{
			"type":  "array",
			"items": goTypeToSchema(t.Elem()),
		}
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]interface{}{"type": "string"}
	case reflect.Float32, reflect.Float64:
		return map[string]interface{}{"type": "number"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]interface{}{"type": "integer"}
	case reflect.Bool:
		return map[string]interface{}{"type": "boolean"}
	default:
		return map[string]interface{}{"type": "string"}
	}
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

var validSentiments = map[string]bool{
	"positive": true,
	"negative": true,
	"neutral":  true,
}

func validateSentimentResponse(s *SentimentResponse) error {
	if !validSentiments[s.Sentiment] {
		return fmt.Errorf("invalid sentiment %q: must be positive, negative, or neutral", s.Sentiment)
	}
	if s.Confidence < 0 || s.Confidence > 1 || math.IsNaN(s.Confidence) {
		return fmt.Errorf("confidence %v is out of range [0, 1]", s.Confidence)
	}
	if len(s.KeyPhrases) > 5 {
		return fmt.Errorf("key_phrases has %d entries; maximum is 5", len(s.KeyPhrases))
	}
	return nil
}

// ---------------------------------------------------------------------------
// extractSentiment
// ---------------------------------------------------------------------------

func extractSentiment(text string) (*SentimentResponse, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text must not be empty")
	}

	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	schema := GenerateJSONSchema(SentimentResponse{})
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("marshal schema: %w", err)
	}

	const maxRetries = 2
	messages := []openai.ChatCompletionMessage{
		{
			Role: openai.ChatMessageRoleSystem,
			Content: "You are a precise sentiment analysis engine. " +
				"Classify the sentiment of the user's text accurately.",
		},
		{Role: openai.ChatMessageRoleUser, Content: text},
	}

	for attempt := 1; attempt <= maxRetries+1; attempt++ {
		log.Printf("extractSentiment: attempt %d/%d", attempt, maxRetries+1)

		resp, err := client.CreateChatCompletion(context.Background(),
			openai.ChatCompletionRequest{
				Model:    openai.GPT4o,
				Messages: messages,
				ResponseFormat: &openai.ChatCompletionResponseFormat{
					Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
					JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
						Name:   "sentiment_response",
						Schema: json.RawMessage(schemaBytes),
						Strict: true,
					},
				},
			})
		if err != nil {
			return nil, fmt.Errorf("API error: %w", err)
		}

		raw := resp.Choices[0].Message.Content

		var result SentimentResponse
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			log.Printf("extractSentiment: attempt %d — JSON parse error: %v", attempt, err)
			if attempt > maxRetries {
				return nil, fmt.Errorf("max retries exceeded: last JSON parse error: %w", err)
			}
			messages = append(messages,
				openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: raw},
				openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: fmt.Sprintf("Your response was not valid JSON. Error: %v. Please fix and retry.", err),
				},
			)
			continue
		}

		if valErr := validateSentimentResponse(&result); valErr != nil {
			log.Printf("extractSentiment: attempt %d — validation error: %v", attempt, valErr)
			if attempt > maxRetries {
				return nil, fmt.Errorf("max retries exceeded: last validation error: %w", valErr)
			}
			messages = append(messages,
				openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: raw},
				openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: fmt.Sprintf("Your response did not match the required schema. Error: %v. Please fix and retry.", valErr),
				},
			)
			continue
		}

		log.Printf("extractSentiment: attempt %d succeeded", attempt)
		return &result, nil
	}

	return nil, fmt.Errorf("extractSentiment: unreachable")
}

// ---------------------------------------------------------------------------
// runStructuredExtraction — called by main()
// ---------------------------------------------------------------------------

func runStructuredExtraction() {
	tests := []string{
		"I absolutely love this, it changed my life!",
		"It's fine I guess, nothing special.",
		"Terrible product, broke after one day.",
	}

	for _, text := range tests {
		result, err := extractSentiment(text)
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}
		fmt.Printf("Text       : %q\n", text)
		fmt.Printf("Sentiment  : %s\n", result.Sentiment)
		fmt.Printf("Confidence : %.2f\n", result.Confidence)
		fmt.Printf("Key Phrases: %v\n\n", result.KeyPhrases)
	}
}
