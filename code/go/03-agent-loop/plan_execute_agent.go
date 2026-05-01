// Plan-and-Execute agent in Go.
//
// Three-phase pattern:
//
//	Phase 1 PLAN:      LLM generates a structured JSON plan.
//	Phase 2 EXECUTE:   Steps run in dependency order; independent steps in parallel.
//	Phase 3 SYNTHESIZE: LLM combines results into a final answer.
//
// Structurally equivalent to code/python/03-agent-loop/plan_execute_agent.py.
//
// Run:  go run .
// See:  docs/02-the-agent-loop/03-planning-strategies.md

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

// PlanStep represents one step in the agent's plan.
type PlanStep struct {
	StepNumber     int                    `json:"step_number"`
	Description    string                 `json:"description"`
	ToolName       *string                `json:"tool_name"`
	ToolParams     map[string]interface{} `json:"tool_params"`
	DependsOn      []int                  `json:"depends_on"`
	ExpectedOutput string                 `json:"expected_output"`
}

// AgentPlan is the full structured plan returned by the planner LLM.
type AgentPlan struct {
	UserQuestion       string     `json:"user_question"`
	Steps              []PlanStep `json:"steps"`
	EstimatedToolCalls int        `json:"estimated_tool_calls"`
}

// StepResult holds the outcome of executing one plan step.
type StepResult struct {
	StepNumber int                    `json:"step_number"`
	Success    bool                   `json:"success"`
	Data       map[string]interface{} `json:"data,omitempty"`
	Error      string                 `json:"error,omitempty"`
	DurationMs int64                  `json:"duration_ms"`
}

// RunOutput is the value returned by PlanAndExecuteAgent.Run.
type RunOutput struct {
	Plan    []PlanStep   `json:"plan"`
	Results []StepResult `json:"results"`
	Answer  string       `json:"answer"`
}

// ---------------------------------------------------------------------------
// Mock tools
// ---------------------------------------------------------------------------

var peWeatherMock = map[string]map[string]interface{}{
	"Tokyo":    {"temperature_c": 22, "condition": "light rain", "humidity_percent": 85, "wind_kph": 15},
	"London":   {"temperature_c": 14, "condition": "overcast", "humidity_percent": 78, "wind_kph": 20},
	"New York": {"temperature_c": 18, "condition": "partly cloudy", "humidity_percent": 60, "wind_kph": 25},
	"Paris":    {"temperature_c": 16, "condition": "sunny", "humidity_percent": 55, "wind_kph": 12},
	"Sydney":   {"temperature_c": 28, "condition": "clear", "humidity_percent": 45, "wind_kph": 18},
}

var peStockMock = map[string]map[string]interface{}{
	"AAPL":  {"price_usd": 192.35, "change_percent": 1.2, "currency": "USD", "weekly_change_percent": 3.1},
	"GOOGL": {"price_usd": 171.80, "change_percent": -0.5, "currency": "USD", "weekly_change_percent": 1.5},
	"MSFT":  {"price_usd": 415.10, "change_percent": 0.8, "currency": "USD", "weekly_change_percent": 2.4},
	"TSLA":  {"price_usd": 175.20, "change_percent": -2.3, "currency": "USD", "weekly_change_percent": -4.1},
	"AMZN":  {"price_usd": 188.40, "change_percent": 0.3, "currency": "USD", "weekly_change_percent": 0.8},
}

var peNewsMock = map[string][]map[string]string{
	"AAPL": {
		{"headline": "Apple unveils M4 Ultra chip with record AI performance", "sentiment": "positive"},
		{"headline": "iPhone 17 pre-orders exceed expectations", "sentiment": "positive"},
	},
	"MSFT": {
		{"headline": "Microsoft Azure revenue grows 33% year-over-year", "sentiment": "positive"},
		{"headline": "Copilot+ PC sales drive record Surface quarter", "sentiment": "positive"},
	},
}

func peGetWeather(params map[string]interface{}) (map[string]interface{}, error) {
	cityRaw, ok := params["city"].(string)
	if !ok || cityRaw == "" {
		return nil, fmt.Errorf("city parameter is required")
	}
	key := strings.TrimSpace(strings.SplitN(cityRaw, ",", 2)[0])
	data, found := peWeatherMock[key]
	if !found {
		data = map[string]interface{}{"temperature_c": 20, "condition": "clear", "humidity_percent": 55, "wind_kph": 10}
	}
	result := make(map[string]interface{}, len(data)+1)
	result["city"] = cityRaw
	for k, v := range data {
		result[k] = v
	}
	return result, nil
}

func peGetStockPrice(params map[string]interface{}) (map[string]interface{}, error) {
	tickerRaw, ok := params["ticker"].(string)
	if !ok || tickerRaw == "" {
		return nil, fmt.Errorf("ticker parameter is required")
	}
	upper := strings.ToUpper(tickerRaw)
	data, found := peStockMock[upper]
	if !found {
		data = map[string]interface{}{"price_usd": 100.0, "change_percent": 0.0, "currency": "USD", "weekly_change_percent": 0.0}
	}
	result := make(map[string]interface{}, len(data)+2)
	result["ticker"] = upper
	result["market_status"] = "open"
	for k, v := range data {
		result[k] = v
	}
	return result, nil
}

func peSearchNews(params map[string]interface{}) (map[string]interface{}, error) {
	queryRaw, ok := params["query"].(string)
	if !ok || queryRaw == "" {
		return nil, fmt.Errorf("query parameter is required")
	}
	ticker := strings.ToUpper(strings.SplitN(queryRaw, " ", 2)[0])
	articles, found := peNewsMock[ticker]
	if !found {
		articles = []map[string]string{{"headline": "No major news for '" + queryRaw + "'", "sentiment": "neutral"}}
	}
	// Convert to []interface{} for generic JSON
	articlesIface := make([]interface{}, len(articles))
	for i, a := range articles {
		articlesIface[i] = a
	}
	return map[string]interface{}{
		"query":    queryRaw,
		"articles": articlesIface,
		"count":    len(articles),
	}, nil
}

type toolFuncPE func(map[string]interface{}) (map[string]interface{}, error)

var peToolDispatch = map[string]toolFuncPE{
	"get_weather":     peGetWeather,
	"get_stock_price": peGetStockPrice,
	"search_news":     peSearchNews,
}

// ---------------------------------------------------------------------------
// System prompts
// ---------------------------------------------------------------------------

const pePlannerSystem = `You are a planning assistant. Break the user's request into concrete sequential steps.

Rules:
- step_number must start at 1 and increment by 1.
- tool_name must be one of: get_weather, get_stock_price, search_news — or null for reasoning/synthesis steps.
- tool_params must contain the exact arguments (or null for reasoning steps).
- depends_on lists the step_numbers this step must wait for.
- expected_output describes the result (at least 15 chars).
- Keep the plan to at most 10 steps.

Output ONLY valid JSON:
{"user_question":"<q>","steps":[{"step_number":1,"description":"...","tool_name":"get_stock_price","tool_params":{"ticker":"AAPL"},"depends_on":[],"expected_output":"AAPL price and weekly change"}],"estimated_tool_calls":1}`

const peSynthesizerSystem = `You are a synthesizer. Given the original question and all execution results,
write a complete, well-structured answer. Cite specific numbers and data.
Use markdown formatting where it improves readability.`

// ---------------------------------------------------------------------------
// PlanAndExecuteAgent
// ---------------------------------------------------------------------------

// PlanAndExecuteAgent implements the three-phase planning strategy.
type PlanAndExecuteAgent struct {
	model    string
	maxSteps int
	client   *openai.Client
}

// NewPlanAndExecuteAgent creates a new agent using the API key from the
// OPENAI_API_KEY environment variable.
func NewPlanAndExecuteAgent(model string, maxSteps int) *PlanAndExecuteAgent {
	apiKey := os.Getenv("OPENAI_API_KEY")
	client := openai.NewClient(option.WithAPIKey(apiKey))
	if model == "" {
		model = "gpt-4o"
	}
	if maxSteps <= 0 {
		maxSteps = 20
	}
	return &PlanAndExecuteAgent{model: model, maxSteps: maxSteps, client: &client}
}

// Run executes the Plan → Execute → Synthesize cycle.
func (a *PlanAndExecuteAgent) Run(ctx context.Context, userInput string) (RunOutput, error) {
	plan, err := a.generatePlan(ctx, userInput)
	if err != nil {
		return RunOutput{}, fmt.Errorf("planning failed: %w", err)
	}
	if len(plan) > a.maxSteps {
		plan = plan[:a.maxSteps]
	}

	results, err := a.executePlan(ctx, plan)
	if err != nil {
		return RunOutput{}, fmt.Errorf("execution failed: %w", err)
	}

	answer, err := a.synthesize(ctx, userInput, plan, results)
	if err != nil {
		return RunOutput{}, fmt.Errorf("synthesis failed: %w", err)
	}

	return RunOutput{Plan: plan, Results: results, Answer: answer}, nil
}

// ---------------------------------------------------------------------------
// Phase 1: Plan
// ---------------------------------------------------------------------------

func (a *PlanAndExecuteAgent) generatePlan(ctx context.Context, userInput string) ([]PlanStep, error) {
	resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: a.model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(pePlannerSystem),
			openai.UserMessage(userInput),
		},
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		},
		Temperature: openai.Float(0),
	})
	if err != nil {
		return nil, err
	}

	raw := resp.Choices[0].Message.Content
	var plan AgentPlan
	if err := json.Unmarshal([]byte(raw), &plan); err != nil || len(plan.Steps) == 0 {
		log.Printf("[PlanAndExecute] Plan parse error (%v); using fallback", err)
		return []PlanStep{
			{
				StepNumber:     1,
				Description:    "Answer the user's question using available tools.",
				ExpectedOutput: "A complete answer to the user's question",
			},
		}, nil
	}
	return plan.Steps, nil
}

// ---------------------------------------------------------------------------
// Phase 2: Execute
// ---------------------------------------------------------------------------

func (a *PlanAndExecuteAgent) executePlan(ctx context.Context, steps []PlanStep) ([]StepResult, error) {
	results := make(map[int]StepResult)
	remaining := make([]PlanStep, len(steps))
	copy(remaining, steps)

	maxRounds := len(steps) + 1
	for round := 0; round < maxRounds && len(remaining) > 0; round++ {
		// Collect ready steps (all dependencies satisfied)
		var ready []PlanStep
		for _, s := range remaining {
			allMet := true
			for _, dep := range s.DependsOn {
				if _, ok := results[dep]; !ok {
					allMet = false
					break
				}
			}
			if allMet {
				ready = append(ready, s)
			}
		}

		if len(ready) == 0 {
			for _, s := range remaining {
				results[s.StepNumber] = StepResult{
					StepNumber: s.StepNumber,
					Success:    false,
					Error:      "unresolved dependencies",
				}
			}
			break
		}

		// Execute ready steps in parallel
		type outcome struct {
			stepNum int
			result  StepResult
		}
		ch := make(chan outcome, len(ready))
		var wg sync.WaitGroup
		for _, step := range ready {
			wg.Add(1)
			go func(s PlanStep) {
				defer wg.Done()
				r := a.executeStep(ctx, s, results)
				ch <- outcome{stepNum: s.StepNumber, result: r}
			}(step)
		}
		wg.Wait()
		close(ch)

		for out := range ch {
			results[out.stepNum] = out.result
		}

		// Remove completed steps
		var next []PlanStep
		for _, s := range remaining {
			if _, done := results[s.StepNumber]; !done {
				next = append(next, s)
			}
		}
		remaining = next
	}

	ordered := make([]StepResult, len(steps))
	for i, s := range steps {
		r, ok := results[s.StepNumber]
		if !ok {
			r = StepResult{StepNumber: s.StepNumber, Success: false, Error: "not executed"}
		}
		ordered[i] = r
	}
	return ordered, nil
}

func (a *PlanAndExecuteAgent) executeStep(
	ctx context.Context,
	step PlanStep,
	priorResults map[int]StepResult,
) StepResult {
	start := time.Now()
	if step.ToolName != nil {
		data, err := a.callTool(*step.ToolName, step.ToolParams)
		durationMs := time.Since(start).Milliseconds()
		if err != nil {
			return StepResult{StepNumber: step.StepNumber, Success: false, Error: err.Error(), DurationMs: durationMs}
		}
		return StepResult{StepNumber: step.StepNumber, Success: true, Data: data, DurationMs: durationMs}
	}

	// Reasoning step: call LLM
	data, err := a.reason(ctx, step, priorResults)
	durationMs := time.Since(start).Milliseconds()
	if err != nil {
		return StepResult{StepNumber: step.StepNumber, Success: false, Error: err.Error(), DurationMs: durationMs}
	}
	return StepResult{StepNumber: step.StepNumber, Success: true, Data: data, DurationMs: durationMs}
}

func (a *PlanAndExecuteAgent) callTool(toolName string, params map[string]interface{}) (map[string]interface{}, error) {
	fn, ok := peToolDispatch[toolName]
	if !ok {
		return nil, fmt.Errorf("unknown tool: %q", toolName)
	}
	return fn(params)
}

func (a *PlanAndExecuteAgent) reason(
	ctx context.Context,
	step PlanStep,
	priorResults map[int]StepResult,
) (map[string]interface{}, error) {
	var contextParts []string
	for _, depNum := range step.DependsOn {
		if dep, ok := priorResults[depNum]; ok && dep.Success {
			b, _ := json.MarshalIndent(dep.Data, "", "  ")
			contextParts = append(contextParts, fmt.Sprintf("Step %d result:\n%s", depNum, string(b)))
		}
	}

	userMessage := step.Description
	if len(contextParts) > 0 {
		userMessage += "\n\nContext from previous steps:\n" + strings.Join(contextParts, "\n\n")
	}

	resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: a.model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("You are an executor. Complete the given reasoning step concisely."),
			openai.UserMessage(userMessage),
		},
		Temperature: openai.Float(0.3),
	})
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{"reasoning": resp.Choices[0].Message.Content}, nil
}

// ---------------------------------------------------------------------------
// Phase 3: Synthesize
// ---------------------------------------------------------------------------

func (a *PlanAndExecuteAgent) synthesize(
	ctx context.Context,
	question string,
	steps []PlanStep,
	results []StepResult,
) (string, error) {
	var lines []string
	lines = append(lines, fmt.Sprintf("Original question: %s\n", question))
	for i, step := range steps {
		result := results[i]
		status := "✓"
		if !result.Success {
			status = "✗"
		}
		lines = append(lines, fmt.Sprintf("Step %d [%s]: %s", step.StepNumber, status, step.Description))
		if result.Success && result.Data != nil {
			b, _ := json.Marshal(result.Data)
			lines = append(lines, "Result: "+string(b))
		} else if result.Error != "" {
			lines = append(lines, "Error: "+result.Error)
		}
		lines = append(lines, "")
	}

	resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: a.model,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(peSynthesizerSystem),
			openai.UserMessage(strings.Join(lines, "\n")),
		},
		Temperature: openai.Float(0.5),
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}

// ---------------------------------------------------------------------------
// Demo (called from main.go when --plan flag is passed, or standalone)
// ---------------------------------------------------------------------------

func runPlanAndExecuteDemo() {
	ctx := context.Background()
	agent := NewPlanAndExecuteAgent("gpt-4o", 20)
	query := "Compare Apple and Microsoft stock performance this week"

	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("Query: %s\n", query)
	fmt.Println(strings.Repeat("=", 60))

	output, err := agent.Run(ctx, query)
	if err != nil {
		log.Fatalf("Agent error: %v", err)
	}

	fmt.Println("\n--- Plan ---")
	for _, step := range output.Plan {
		toolLabel := "[reasoning]"
		if step.ToolName != nil {
			toolLabel = "[" + *step.ToolName + "]"
		}
		fmt.Printf("  %d. %s %s\n", step.StepNumber, step.Description, toolLabel)
	}

	fmt.Println("\n--- Results ---")
	for _, result := range output.Results {
		status := "✓"
		if !result.Success {
			status = "✗"
		}
		fmt.Printf("  Step %d [%s] (%dms)\n", result.StepNumber, status, result.DurationMs)
		if result.Success && result.Data != nil {
			b, _ := json.Marshal(result.Data)
			fmt.Printf("    %s\n", string(b))
		} else if result.Error != "" {
			fmt.Printf("    ERROR: %s\n", result.Error)
		}
	}

	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("FINAL ANSWER:")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println(output.Answer)
}
