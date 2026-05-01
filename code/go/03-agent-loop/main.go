// ReAct agent: Reason → Act → Observe loop in Go.
//
// Structurally identical to code/python/03-agent-loop/agent.py.
// Run:  go run .
//
// See docs/02-the-agent-loop/01-anatomy-of-an-agent.md

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const maxIterations = 10

const systemPrompt = `You are an AI assistant with access to tools.

## Your Process
1. When the user asks a question, determine if you need a tool to answer it.
2. If yes, call the appropriate tool with the correct parameters.
3. Wait for the tool result, then determine if you need more tools or can answer.
4. Never guess tool results. Always wait for the actual result.
5. If a tool fails, explain the failure to the user and suggest alternatives.

## Tool Usage Rules
- Call only one tool at a time unless they are independent.
- If you don't have enough information to call a tool, ask the user.
- Never make up parameters. If unsure, ask for clarification.

## Answer Format
- Use the tool results to answer the user's question directly.
- Cite specific data from tool results.
- If multiple tools were used, synthesize the information.`

// WeatherResult holds the data returned by GetWeather.
type WeatherResult struct {
	City            string `json:"city"`
	TemperatureC    int    `json:"temperature_c"`
	Condition       string `json:"condition"`
	HumidityPercent int    `json:"humidity_percent"`
	WindKPH         int    `json:"wind_kph"`
}

var weatherMock = map[string]WeatherResult{
	"Shanghai": {TemperatureC: 22, Condition: "light rain", HumidityPercent: 85, WindKPH: 15},
	"London":   {TemperatureC: 14, Condition: "overcast", HumidityPercent: 78, WindKPH: 20},
	"New York": {TemperatureC: 18, Condition: "partly cloudy", HumidityPercent: 60, WindKPH: 25},
	"Paris":    {TemperatureC: 16, Condition: "sunny", HumidityPercent: 55, WindKPH: 12},
	"Sydney":   {TemperatureC: 28, Condition: "clear", HumidityPercent: 45, WindKPH: 18},
}

// GetWeather returns current weather conditions for the given city.
func GetWeather(city string) WeatherResult {
	cityKey := strings.TrimSpace(strings.SplitN(city, ",", 2)[0])
	data, ok := weatherMock[cityKey]
	if !ok {
		data = WeatherResult{TemperatureC: 20, Condition: "clear", HumidityPercent: 55, WindKPH: 10}
	}
	data.City = city
	return data
}

// StockResult holds the data returned by GetStockPrice.
type StockResult struct {
	Ticker        string  `json:"ticker"`
	PriceUSD      float64 `json:"price_usd"`
	ChangePercent float64 `json:"change_percent"`
	Currency      string  `json:"currency"`
	MarketStatus  string  `json:"market_status"`
}

var stockMock = map[string]StockResult{
	"AAPL":  {PriceUSD: 192.35, ChangePercent: 1.2, Currency: "USD"},
	"GOOGL": {PriceUSD: 171.80, ChangePercent: -0.5, Currency: "USD"},
	"MSFT":  {PriceUSD: 415.10, ChangePercent: 0.8, Currency: "USD"},
	"TSLA":  {PriceUSD: 175.20, ChangePercent: -2.3, Currency: "USD"},
	"AMZN":  {PriceUSD: 188.40, ChangePercent: 0.3, Currency: "USD"},
}

// GetStockPrice returns current stock data for the given ticker symbol.
func GetStockPrice(ticker string) StockResult {
	upper := strings.ToUpper(ticker)
	data, ok := stockMock[upper]
	if !ok {
		data = StockResult{PriceUSD: 100.0, ChangePercent: 0.0, Currency: "USD"}
	}
	data.Ticker = upper
	data.MarketStatus = "open"
	return data
}

var agentTools = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name: "get_weather",
			Description: openai.String(
				"Get current weather conditions for a city. " +
					"Use this when the user asks about weather, temperature, rain, " +
					"humidity, wind, or whether to bring an umbrella or coat. " +
					"Always call this tool rather than guessing — weather is dynamic.",
			),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"city": map[string]string{
						"type":        "string",
						"description": "City name with optional ISO country code, e.g. 'Shanghai, CN'.",
					},
				},
				"required": []string{"city"},
			},
		},
	},
	{
		Function: openai.FunctionDefinitionParam{
			Name: "get_stock_price",
			Description: openai.String(
				"Get the current stock price and daily percentage change for a publicly " +
					"traded company. Use this when the user asks about stock price, share " +
					"value, investment potential, or financial performance. " +
					"Always call this tool rather than using stale training data.",
			),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"ticker": map[string]string{
						"type":        "string",
						"description": "Stock ticker symbol in uppercase, e.g. 'AAPL', 'GOOGL', 'MSFT'.",
					},
				},
				"required": []string{"ticker"},
			},
		},
	},
}

// RunAgent runs the ReAct loop until a final answer or maxIterations.
func RunAgent(
	ctx context.Context,
	userInput string,
	messages []openai.ChatCompletionMessageParamUnion,
	registry ToolRegistry,
) (string, []openai.ChatCompletionMessageParamUnion, error) {
	if strings.TrimSpace(userInput) == "" {
		return "", messages, fmt.Errorf("userInput must not be empty")
	}
	if messages == nil {
		messages = []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
		}
	}
	messages = append(messages, openai.UserMessage(userInput))

	client := openai.NewClient(option.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

	for iteration := 1; iteration <= maxIterations; iteration++ {
		log.Printf("Agent iteration %d/%d", iteration, maxIterations)

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    openai.ChatModelGPT4o,
			Messages: messages,
			Tools:    agentTools,
			ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("auto"),
			},
		})
		if err != nil {
			return "", messages, fmt.Errorf("LLM call failed on iteration %d: %w", iteration, err)
		}

		choice := resp.Choices[0]

		// Append the assistant turn BEFORE processing tool calls.
		messages = append(messages, choice.Message.ToParam())

		if len(choice.Message.ToolCalls) == 0 {
			log.Printf("Agent finished after %d iteration(s)", iteration)
			return choice.Message.Content, messages, nil
		}

		for _, tc := range choice.Message.ToolCalls {
			toolMsg := DispatchTool(tc, registry)
			messages = append(messages, toolMsg)
		}
	}

	const maxMsg = "I was unable to complete your request within the allowed number of " +
		"steps. Please try rephrasing your question or breaking it into smaller parts."
	return maxMsg, messages, nil
}

func main() {
	ctx := context.Background()

	registry := ToolRegistry{
		"get_weather": func(args map[string]interface{}) (interface{}, error) {
			city, _ := args["city"].(string)
			return GetWeather(city), nil
		},
		"get_stock_price": func(args map[string]interface{}) (interface{}, error) {
			ticker, _ := args["ticker"].(string)
			return GetStockPrice(ticker), nil
		},
	}

	queries := []string{
		"What's the weather in Shanghai?",
		"Should I invest in Apple stock right now?",
	}

	for _, query := range queries {
		messages := []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
		}

		fmt.Printf("\n%s\n", strings.Repeat("=", 60))
		fmt.Printf("Query: %s\n", query)
		fmt.Println(strings.Repeat("=", 60))

		answer, finalMessages, err := RunAgent(ctx, query, messages, registry)
		if err != nil {
			log.Printf("Agent error: %v", err)
			continue
		}

		fmt.Printf("\nFinal Answer:\n%s\n", answer)
		fmt.Println("\n--- Full Conversation History ---")
		for _, msg := range finalMessages {
			printMessage(msg)
		}
	}
}

func printMessage(msg openai.ChatCompletionMessageParamUnion) {
	switch {
	case msg.OfSystem != nil:
		content := msg.OfSystem.Content.OfString.Value
		if len(content) > 60 {
			content = content[:60] + "…"
		}
		fmt.Printf("  [SYSTEM] %s\n", strings.ReplaceAll(content, "\n", " "))

	case msg.OfUser != nil:
		var content string
		if msg.OfUser.Content.OfString.Valid() {
			content = msg.OfUser.Content.OfString.Value
		}
		fmt.Printf("  [USER] %s\n", content)

	case msg.OfAssistant != nil:
		if len(msg.OfAssistant.ToolCalls) > 0 {
			for _, tc := range msg.OfAssistant.ToolCalls {
				fmt.Printf("  [ASSISTANT] → tool call: %s(%s)\n",
					tc.Function.Name, tc.Function.Arguments)
			}
		} else {
			content := msg.OfAssistant.Content.OfString.Value
			if len(content) > 80 {
				content = content[:80] + "…"
			}
			fmt.Printf("  [ASSISTANT] %s\n", content)
		}

	case msg.OfTool != nil:
		raw := msg.OfTool.Content.OfString.Value
		var result map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &result); err == nil {
			fmt.Printf("  [TOOL] ← result: %v\n", result)
		} else {
			fmt.Printf("  [TOOL] ← %s\n", raw)
		}
	}
}
