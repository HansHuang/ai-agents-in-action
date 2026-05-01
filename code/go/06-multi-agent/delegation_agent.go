// Delegation multi-agent system in Go.
//
// The coordinator runs a ReAct loop where its "tools" are specialist agents.
// Each specialist has its own tools and system prompt.
//
// Structurally equivalent to code/python/06-multi-agent/delegation_agent.py.
//
// Run:  go run .
// See:  docs/02-the-agent-loop/04-multi-agent-patterns.md

package multiagent

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

const maxSpecialistIterations = 5
const maxDelegationsPerAgent = 3

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

var stockMock = map[string]map[string]interface{}{
	"AAPL":  {"price_usd": 192.35, "change_percent": 1.2, "weekly_change_percent": 3.1},
	"MSFT":  {"price_usd": 415.10, "change_percent": 0.8, "weekly_change_percent": 2.4},
	"GOOGL": {"price_usd": 171.80, "change_percent": -0.5, "weekly_change_percent": 1.5},
	"TSLA":  {"price_usd": 175.20, "change_percent": -2.3, "weekly_change_percent": -4.1},
	"AMZN":  {"price_usd": 188.40, "change_percent": 0.3, "weekly_change_percent": 0.8},
}

var financialsMock = map[string]map[string]interface{}{
	"Apple":     {"revenue_ttm_b": 391.0, "net_income_b": 97.0, "gross_margin_pct": 45.0, "yoy_revenue_growth_pct": 5.1},
	"Microsoft": {"revenue_ttm_b": 245.0, "net_income_b": 88.0, "gross_margin_pct": 70.0, "yoy_revenue_growth_pct": 15.7},
	"Google":    {"revenue_ttm_b": 350.0, "net_income_b": 76.0, "gross_margin_pct": 58.0, "yoy_revenue_growth_pct": 14.3},
}

var newsMock = map[string][]string{
	"Apple": {
		"Apple reports record services revenue of $25B in Q3.",
		"Apple Vision Pro sales expected to reach 1M units by year-end.",
	},
	"Microsoft": {
		"Microsoft Copilot adds 5M enterprise users in Q3.",
		"Azure cloud revenue surpasses $30B quarterly run rate.",
	},
}

// ---------------------------------------------------------------------------
// Mock tools
// ---------------------------------------------------------------------------

func mockGetStockPrice(args map[string]interface{}) map[string]interface{} {
	ticker := strings.ToUpper(fmt.Sprintf("%v", args["ticker"]))
	if data, ok := stockMock[ticker]; ok {
		result := map[string]interface{}{"ticker": ticker}
		for k, v := range data {
			result[k] = v
		}
		return result
	}
	return map[string]interface{}{"ticker": ticker, "price_usd": 100.0, "change_percent": 0.0}
}

func mockGetCompanyFinancials(args map[string]interface{}) map[string]interface{} {
	company := strings.ToLower(fmt.Sprintf("%v", args["company"]))
	for k, v := range financialsMock {
		if strings.Contains(company, strings.ToLower(k)) {
			result := map[string]interface{}{"company": k}
			for fk, fv := range v {
				result[fk] = fv
			}
			return result
		}
	}
	return map[string]interface{}{"company": args["company"], "error": "No financials found"}
}

func mockWebSearch(args map[string]interface{}) map[string]interface{} {
	query := strings.ToLower(fmt.Sprintf("%v", args["query"]))
	for k, articles := range newsMock {
		if strings.Contains(query, strings.ToLower(k)) {
			return map[string]interface{}{"query": args["query"], "results": articles}
		}
	}
	return map[string]interface{}{"query": args["query"], "results": []string{fmt.Sprintf("No mock results for '%v'", args["query"])}}
}

func mockFetchArticle(args map[string]interface{}) map[string]interface{} {
	return map[string]interface{}{"url": args["url"], "title": "Mock article", "content": "Mock article body."}
}

var toolDispatch = map[string]func(map[string]interface{}) map[string]interface{}{
	"get_stock_price":        mockGetStockPrice,
	"get_company_financials": mockGetCompanyFinancials,
	"web_search":             mockWebSearch,
	"fetch_article":          mockFetchArticle,
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

var financeTools = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "get_stock_price",
			Description: openai.String("Get the current stock price and weekly change for a ticker symbol."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"ticker": map[string]string{"type": "string", "description": "Uppercase ticker, e.g. 'AAPL'."},
				},
				"required": []string{"ticker"},
			},
		},
	},
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "get_company_financials",
			Description: openai.String("Get TTM financials for a company."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"company": map[string]string{"type": "string", "description": "Company name, e.g. 'Apple'."},
				},
				"required": []string{"company"},
			},
		},
	},
}

var researchTools = []openai.ChatCompletionToolParam{
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "web_search",
			Description: openai.String("Search the web for recent news and information about a topic."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]string{"type": "string", "description": "Search query string."},
				},
				"required": []string{"query"},
			},
		},
	},
	{
		Function: openai.FunctionDefinitionParam{
			Name:        "fetch_article",
			Description: openai.String("Fetch the full text of a web article by URL."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"url": map[string]string{"type": "string", "description": "Full article URL."},
				},
				"required": []string{"url"},
			},
		},
	},
}

// ---------------------------------------------------------------------------
// Handoff / HandoffResult
// ---------------------------------------------------------------------------

// Handoff carries the minimum context a specialist needs.
type Handoff struct {
	FromAgent string                 `json:"from_agent"`
	ToAgent   string                 `json:"to_agent"`
	Task      string                 `json:"task"`
	Context   map[string]interface{} `json:"context"`
}

// HandoffResult is the response from a specialist.
type HandoffResult struct {
	FromAgent string `json:"from_agent"`
	Status    string `json:"status"` // "complete" | "failed" | "need_clarification"
	Result    string `json:"result"`
	Error     string `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// SpecialistAgent
// ---------------------------------------------------------------------------

// SpecialistAgent is a focused LLM with its own tools.
type SpecialistAgent struct {
	Name         string
	Role         string
	Tools        []openai.ChatCompletionToolParam
	SystemPrompt string
	TaskGuidance string
	client       *openai.Client
}

// NewSpecialistAgent creates a SpecialistAgent with an OpenAI client.
func NewSpecialistAgent(name, role, systemPrompt, taskGuidance string, tools []openai.ChatCompletionToolParam, apiKey string) *SpecialistAgent {
	c := openai.NewClient(option.WithAPIKey(apiKey))
	return &SpecialistAgent{
		Name:         name,
		Role:         role,
		Tools:        tools,
		SystemPrompt: systemPrompt,
		TaskGuidance: taskGuidance,
		client:       &c,
	}
}

// Run executes a delegated task and returns a HandoffResult.
func (s *SpecialistAgent) Run(h Handoff) HandoffResult {
	userContent := h.Task
	if len(h.Context) > 0 {
		ctxBytes, _ := json.MarshalIndent(h.Context, "", "  ")
		userContent += "\n\nContext provided:\n" + string(ctxBytes)
	}

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(s.SystemPrompt),
		openai.UserMessage(userContent),
	}

	for i := 0; i < maxSpecialistIterations; i++ {
		params := openai.ChatCompletionNewParams{
			Model:    openai.ChatModelGPT4o,
			Messages: messages,
		}
		if len(s.Tools) > 0 {
			params.Tools = s.Tools
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("auto"),
			}
		}

		resp, err := s.client.Chat.Completions.New(context.Background(), params)
		if err != nil {
			return HandoffResult{FromAgent: s.Name, Status: "failed", Error: err.Error()}
		}

		msg := resp.Choices[0].Message
		messages = append(messages, msg.ToParam())

		if len(msg.ToolCalls) == 0 {
			return HandoffResult{FromAgent: s.Name, Status: "complete", Result: msg.Content}
		}

		for _, tc := range msg.ToolCalls {
			var args map[string]interface{}
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			fn, ok := toolDispatch[tc.Function.Name]
			var content string
			if ok {
				b, _ := json.Marshal(fn(args))
				content = string(b)
			} else {
				content = fmt.Sprintf(`{"error":"Unknown tool: %s"}`, tc.Function.Name)
			}
			messages = append(messages, openai.ToolMessage(content, tc.ID))
		}
	}

	return HandoffResult{FromAgent: s.Name, Status: "complete", Result: "Max iterations reached."}
}

// ---------------------------------------------------------------------------
// CoordinatorAgent
// ---------------------------------------------------------------------------

const coordinatorSystem = `You are a coordinator agent. Your ONLY job is to delegate tasks to specialist agents.
Do NOT answer questions directly. Do NOT perform analysis yourself.

Your specialists:
- delegate_to_finance_agent:   financial data, stock prices, earnings, revenue figures
- delegate_to_research_agent:  web search, news, recent events, background research
- delegate_to_writing_agent:   synthesis, summaries, reports, polished prose

Rules:
1. Always delegate immediately — never answer from your own knowledge.
2. If a request spans multiple domains, delegate to multiple specialists.
3. Give each specialist a complete, self-contained task with all the context they need.
4. After all specialists have responded, synthesize their results into a final answer.
5. If a specialist reports failure, acknowledge it in your final answer and continue.`

// DelegationRecord tracks one delegation call.
type DelegationRecord struct {
	Specialist string `json:"specialist"`
	Task       string `json:"task"`
	Status     string `json:"status"`
}

// CoordinatorAgent routes requests to specialists and synthesises results.
type CoordinatorAgent struct {
	specialists      map[string]*SpecialistAgent
	delegationCounts map[string]int
	delegationTools  []openai.ChatCompletionToolParam
	client           *openai.Client
}

// NewCoordinatorAgent creates a CoordinatorAgent.
func NewCoordinatorAgent(specialists map[string]*SpecialistAgent, apiKey string) *CoordinatorAgent {
	c := openai.NewClient(option.WithAPIKey(apiKey))
	counts := make(map[string]int, len(specialists))
	for name := range specialists {
		counts[name] = 0
	}
	coord := &CoordinatorAgent{
		specialists:      specialists,
		delegationCounts: counts,
		client:           &c,
	}
	coord.delegationTools = coord.buildDelegationTools()
	return coord
}

// CoordinatorOutput is the result of a coordinator run.
type CoordinatorOutput struct {
	Answer      string             `json:"answer"`
	Delegations []DelegationRecord `json:"delegations"`
}

// Run processes the user input through the delegation pattern.
func (coord *CoordinatorAgent) Run(userInput string) (CoordinatorOutput, error) {
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(coordinatorSystem),
		openai.UserMessage(userInput),
	}
	var delegations []DelegationRecord

	for i := 0; i < 10; i++ {
		resp, err := coord.client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
			Model:    openai.ChatModelGPT4o,
			Messages: messages,
			Tools:    coord.delegationTools,
			ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("auto"),
			},
		})
		if err != nil {
			return CoordinatorOutput{}, err
		}

		msg := resp.Choices[0].Message
		messages = append(messages, msg.ToParam())

		if len(msg.ToolCalls) == 0 {
			return CoordinatorOutput{Answer: msg.Content, Delegations: delegations}, nil
		}

		for _, tc := range msg.ToolCalls {
			specialistName := strings.TrimSuffix(strings.TrimPrefix(tc.Function.Name, "delegate_to_"), "_agent")
			specialist, ok := coord.specialists[specialistName]
			if !ok {
				content := fmt.Sprintf(`{"error":"Unknown specialist: %s"}`, specialistName)
				messages = append(messages, openai.ToolMessage(content, tc.ID))
				continue
			}

			coord.delegationCounts[specialistName]++
			if coord.delegationCounts[specialistName] > maxDelegationsPerAgent {
				messages = append(messages, openai.ToolMessage(`{"error":"Delegation limit reached"}`, tc.ID))
				continue
			}

			var args struct {
				Task    string                 `json:"task"`
				Context map[string]interface{} `json:"context"`
			}
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)

			handoff := Handoff{
				FromAgent: "coordinator",
				ToAgent:   specialistName,
				Task:      args.Task,
				Context:   args.Context,
			}
			result := specialist.Run(handoff)
			delegations = append(delegations, DelegationRecord{
				Specialist: specialistName,
				Task:       args.Task,
				Status:     result.Status,
			})

			b, _ := json.Marshal(map[string]string{"result": result.Result, "status": result.Status})
			messages = append(messages, openai.ToolMessage(string(b), tc.ID))
		}
	}

	return CoordinatorOutput{Answer: "Maximum iterations reached.", Delegations: delegations}, nil
}

func (coord *CoordinatorAgent) buildDelegationTools() []openai.ChatCompletionToolParam {
	var tools []openai.ChatCompletionToolParam
	for name, agent := range coord.specialists {
		toolName := fmt.Sprintf("delegate_to_%s_agent", name)
		desc := fmt.Sprintf("Delegate a task to the %s specialist. %s", name, agent.TaskGuidance)
		tools = append(tools, openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        toolName,
				Description: openai.String(desc),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"task":    map[string]string{"type": "string", "description": "Complete self-contained task description."},
						"context": map[string]interface{}{"type": "object", "description": "Relevant context.", "additionalProperties": true},
					},
					"required": []string{"task"},
				},
			},
		})
	}
	return tools
}

// ---------------------------------------------------------------------------
// Build default specialists
// ---------------------------------------------------------------------------

// BuildSpecialists returns the default set of specialist agents.
func BuildSpecialists(apiKey string) map[string]*SpecialistAgent {
	return map[string]*SpecialistAgent{
		"finance": NewSpecialistAgent(
			"finance",
			"Financial analysis expert",
			"You are a financial analyst. Use your tools to retrieve accurate financial data. Present numbers clearly with units.",
			"Use for stock prices, earnings reports, revenue figures, and financial comparisons.",
			financeTools,
			apiKey,
		),
		"research": NewSpecialistAgent(
			"research",
			"Research and news expert",
			"You are a research specialist. Search for relevant information and synthesise key findings with source references.",
			"Use for web searches, news, recent events, and background research.",
			researchTools,
			apiKey,
		),
		"writing": NewSpecialistAgent(
			"writing",
			"Professional writer",
			"You are a professional writer. Turn data and findings into polished, well-structured prose.",
			"Use to create summaries, reports, and polished final deliverables.",
			nil,
			apiKey,
		),
	}
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

// RunDelegationDemo runs the delegation pattern demo.
func RunDelegationDemo() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY not set")
	}

	specialists := BuildSpecialists(apiKey)
	coordinator := NewCoordinatorAgent(specialists, apiKey)

	task := "Research Apple's financial performance and write a summary"
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("Task: %s\n", task)
	fmt.Println(strings.Repeat("=", 60))

	output, err := coordinator.Run(task)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Println("\n--- Delegations ---")
	for _, d := range output.Delegations {
		preview := d.Task
		if len(preview) > 60 {
			preview = preview[:60] + "…"
		}
		fmt.Printf("  [%s] %s: %s\n", d.Specialist, d.Status, preview)
	}

	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("FINAL ANSWER:")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println(output.Answer)
}
