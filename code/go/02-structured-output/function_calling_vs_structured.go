// Function Calling vs. Structured Output: side-by-side comparison.
//
// Runs the same sentiment-extraction task through both API paths on 5 test
// texts and prints a per-text table plus a summary comparing success rate,
// total tokens, and average latency for each method.
//
// Port of code/python/02-structured-output/function_calling_vs_structured.py
// Run: go run . -mode=compare
//
// See docs/01-foundations/03-structured-output.md
// — "Function Calling vs. Structured Output: The Real Difference"
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

const fcsModel = "gpt-4o"

// fcsSentimentTool is the function-calling tool definition (Path A).
var fcsSentimentTool = openai.Tool{
	Type: openai.ToolTypeFunction,
	Function: &openai.FunctionDefinition{
		Name:        "classify_sentiment",
		Description: "Classify the sentiment of the provided text.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"sentiment": map[string]interface{}{
					"type":        "string",
					"enum":        []string{"positive", "negative", "neutral"},
					"description": "Overall sentiment.",
				},
				"confidence": map[string]interface{}{
					"type":        "number",
					"minimum":     0,
					"maximum":     1,
					"description": "Confidence score between 0 and 1.",
				},
				"key_phrases": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"maxItems":    5,
					"description": "Key phrases that influenced the classification.",
				},
			},
			"required": []string{"sentiment", "confidence"},
		},
	},
}

// fcsSentimentSchema is the JSON Schema for structured output / json_schema (Path B).
// Strict mode requires additionalProperties: false and all properties listed in required.
var fcsSentimentSchema = map[string]interface{}{
	"type": "object",
	"properties": map[string]interface{}{
		"sentiment": map[string]interface{}{
			"type": "string",
			"enum": []string{"positive", "negative", "neutral"},
		},
		"confidence": map[string]interface{}{
			"type":    "number",
			"minimum": 0,
			"maximum": 1,
		},
		"key_phrases": map[string]interface{}{
			"type":  "array",
			"items": map[string]interface{}{"type": "string"},
		},
	},
	"required":             []string{"sentiment", "confidence", "key_phrases"},
	"additionalProperties": false,
}

// fcsResult holds the outcome of one extraction attempt.
type fcsResult struct {
	method           string
	text             string
	success          bool
	sentiment        string
	confidence       float64
	promptTokens     int
	completionTokens int
	latencyMS        float64
	errStr           string
}

// fcsTestTexts is the shared set of test inputs for both extraction paths.
var fcsTestTexts = []string{
	"I absolutely love this product! Best purchase I've ever made.",
	"This is absolutely terrible. Complete waste of money.",
	"It arrived. Haven't tried it yet.",
	"Oh sure, because *that's* exactly what I needed — another broken feature.",
	"Ce produit est fantastique, je le recommande vivement.",
}

// fcsFunctionCalling extracts sentiment via function calling (Path A).
func fcsFunctionCalling(ctx context.Context, text string, client *openai.Client) fcsResult {
	start := time.Now()
	res := fcsResult{method: "function_calling", text: text}

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    fcsModel,
		Messages: []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: text}},
		Tools:    []openai.Tool{fcsSentimentTool},
		ToolChoice: openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: "classify_sentiment"},
		},
	})
	res.latencyMS = float64(time.Since(start).Milliseconds())

	if err != nil {
		res.errStr = err.Error()
		return res
	}
	if len(resp.Choices) == 0 || len(resp.Choices[0].Message.ToolCalls) == 0 {
		res.errStr = "no tool call in response"
		return res
	}

	var args struct {
		Sentiment  string  `json:"sentiment"`
		Confidence float64 `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.ToolCalls[0].Function.Arguments), &args); err != nil {
		res.errStr = err.Error()
		return res
	}

	res.success = true
	res.sentiment = args.Sentiment
	res.confidence = args.Confidence
	res.promptTokens = resp.Usage.PromptTokens
	res.completionTokens = resp.Usage.CompletionTokens
	return res
}

// fcsStructuredOutput extracts sentiment via structured output / json_schema (Path B).
func fcsStructuredOutput(ctx context.Context, text string, client *openai.Client) fcsResult {
	start := time.Now()
	res := fcsResult{method: "structured_output", text: text}

	schemaBytes, _ := json.Marshal(fcsSentimentSchema)

	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    fcsModel,
		Messages: []openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: text}},
		ResponseFormat: &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   "sentiment_response",
				Schema: json.RawMessage(schemaBytes),
				Strict: true,
			},
		},
	})
	res.latencyMS = float64(time.Since(start).Milliseconds())

	if err != nil {
		res.errStr = err.Error()
		return res
	}

	var parsed struct {
		Sentiment  string  `json:"sentiment"`
		Confidence float64 `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &parsed); err != nil {
		res.errStr = err.Error()
		return res
	}

	res.success = true
	res.sentiment = parsed.Sentiment
	res.confidence = parsed.Confidence
	res.promptTokens = resp.Usage.PromptTokens
	res.completionTokens = resp.Usage.CompletionTokens
	return res
}

// RunFunctionCallingVsStructured compares function calling and structured output
// for sentiment extraction across five test texts.
func RunFunctionCallingVsStructured() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()

	var fcResults, soResults []fcsResult

	for _, text := range fcsTestTexts {
		fc := fcsFunctionCalling(ctx, text, client)
		so := fcsStructuredOutput(ctx, text, client)
		fcResults = append(fcResults, fc)
		soResults = append(soResults, so)

		agree := "N/A"
		if fc.success && so.success {
			if fc.sentiment == so.sentiment {
				agree = "true"
			} else {
				agree = "false"
			}
		}

		preview := text
		if len(preview) > 65 {
			preview = preview[:65]
		}
		fmt.Printf("\nText: %q\n", preview)
		fmt.Printf("  %-22s %-12s %8s %10s  Status\n", "Method", "Result", "Tokens", "Latency")
		fmt.Println("  " + strings.Repeat("-", 58))

		fcSentiment, soSentiment := fc.sentiment, so.sentiment
		if !fc.success {
			fcSentiment = "FAIL"
		}
		if !so.success {
			soSentiment = "FAIL"
		}
		fcStatus, soStatus := "✓", "✓"
		if !fc.success {
			fcStatus = "✗"
		}
		if !so.success {
			soStatus = "✗"
		}

		fcTok := fc.promptTokens + fc.completionTokens
		soTok := so.promptTokens + so.completionTokens
		fmt.Printf("  %-22s %-12s %8d %9.0fms  %s\n", "function_calling", fcSentiment, fcTok, fc.latencyMS, fcStatus)
		fmt.Printf("  %-22s %-12s %8d %9.0fms  %s\n", "structured_output", soSentiment, soTok, so.latencyMS, soStatus)
		fmt.Printf("  Results agree: %s\n", agree)
	}

	// Summary
	fmt.Printf("\n%s\n", strings.Repeat("=", 58))
	fmt.Println("SUMMARY")
	fmt.Println(strings.Repeat("=", 58))

	type resultPair struct {
		label   string
		results []fcsResult
	}
	for _, p := range []resultPair{
		{"Function Calling", fcResults},
		{"Structured Output", soResults},
	} {
		var successes []fcsResult
		totalTok := 0
		for _, r := range p.results {
			if r.success {
				successes = append(successes, r)
				totalTok += r.promptTokens + r.completionTokens
			}
		}
		avgTok := 0
		if len(successes) > 0 {
			avgTok = totalTok / len(successes)
		}
		var totalLat float64
		for _, r := range p.results {
			totalLat += r.latencyMS
		}
		avgLat := 0.0
		if len(p.results) > 0 {
			avgLat = totalLat / float64(len(p.results))
		}
		fmt.Printf("\n%s:\n", p.label)
		fmt.Printf("  Success rate : %d/%d\n", len(successes), len(p.results))
		fmt.Printf("  Avg tokens   : %d\n", avgTok)
		fmt.Printf("  Avg latency  : %.0fms\n", avgLat)
	}
}
