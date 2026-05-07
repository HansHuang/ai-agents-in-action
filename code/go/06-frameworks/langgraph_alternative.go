// langgraph_alternative.go — Go-native state-machine agent.
//
// LangGraph and LangChain have limited Go support (May 2026).
// This file demonstrates that the CONCEPTS transfer perfectly even when the
// framework doesn't.  The same graph-based workflow from
// code/python/06-frameworks/langgraph_react_agent.py is implemented here
// using only Go's standard library and the OpenAI SDK.
//
// Architecture:
//
//	Go state machine equivalent of LangGraph's StateGraph:
//	  • AgentState        — explicit typed struct (vs. TypedDict)
//	  • NodeFn            — function type per node
//	  • StateGraph        — tiny graph runtime (add nodes, edges, run)
//	  • ConditionalEdge   — routing function (same concept as LangGraph)
//
// Go framework situation (May 2026):
//
//   - github.com/tmc/langchaingo exists but lags Python LangChain by 6-12
//     months.  Most production Go teams build from scratch.
//   - This is typically an ADVANTAGE: Go agents are cleaner, faster, and
//     easier to debug than Python agents built on heavy frameworks.
//   - For vector search, use Qdrant Go SDK or Weaviate Go client directly.
//
// Run:
//
//	go run langgraph_alternative.go
//
// See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	reactLLMModel  = openai.GPT4o
	reactMaxIter   = 10
	reactNodeAgent = "agent"
	reactNodeTool  = "tools"
	reactNodeEnd   = "__end__"
)

// ---------------------------------------------------------------------------
// State — the equivalent of LangGraph's TypedDict state
// ---------------------------------------------------------------------------

// AgentState holds the full runtime state of the agent.
// In LangGraph this is a TypedDict; in Go it's a plain struct.
type AgentState struct {
	Messages       []openai.ChatCompletionMessage
	IterationCount int
	ToolCallsMade  []string
}

// AgentResult is the structured output returned by Graph.Run.
type AgentResult struct {
	Answer        string
	Iterations    int
	ToolCallsMade []string
	ElapsedMs     float64
}

// ---------------------------------------------------------------------------
// Mock tool data (mirrors tools.py)
// ---------------------------------------------------------------------------

type weatherData struct {
	City            string `json:"city"`
	TemperatureC    int    `json:"temperature_c"`
	Condition       string `json:"condition"`
	HumidityPercent int    `json:"humidity_percent"`
	WindKph         int    `json:"wind_kph"`
}

type stockData struct {
	Ticker        string  `json:"ticker"`
	PriceUSD      float64 `json:"price_usd"`
	ChangePercent float64 `json:"change_percent"`
	Currency      string  `json:"currency"`
	MarketStatus  string  `json:"market_status"`
}

var weatherMock = map[string]weatherData{
	"Tokyo":    {TemperatureC: 18, Condition: "partly cloudy", HumidityPercent: 65, WindKph: 14},
	"Shanghai": {TemperatureC: 22, Condition: "light rain", HumidityPercent: 85, WindKph: 15},
	"London":   {TemperatureC: 14, Condition: "overcast", HumidityPercent: 78, WindKph: 20},
	"New York": {TemperatureC: 18, Condition: "partly cloudy", HumidityPercent: 60, WindKph: 25},
	"Paris":    {TemperatureC: 16, Condition: "sunny", HumidityPercent: 55, WindKph: 12},
}

var stockMock = map[string]stockData{
	"AAPL":  {PriceUSD: 192.35, ChangePercent: 1.2, Currency: "USD"},
	"GOOGL": {PriceUSD: 171.80, ChangePercent: -0.5, Currency: "USD"},
	"MSFT":  {PriceUSD: 415.10, ChangePercent: 0.8, Currency: "USD"},
	"TSLA":  {PriceUSD: 175.20, ChangePercent: -2.3, Currency: "USD"},
	"AMZN":  {PriceUSD: 188.40, ChangePercent: 0.3, Currency: "USD"},
}

// ---------------------------------------------------------------------------
// Tool implementations (the "hands" of the agent)
// ---------------------------------------------------------------------------

func getWeather(city string) string {
	cityKey := strings.TrimSpace(strings.SplitN(city, ",", 2)[0])
	data, ok := weatherMock[cityKey]
	if !ok {
		data = weatherData{TemperatureC: 20, Condition: "clear", HumidityPercent: 55, WindKph: 10}
	}
	data.City = city
	b, _ := json.Marshal(data)
	return string(b)
}

func getStockPrice(ticker string) string {
	t := strings.ToUpper(ticker)
	data, ok := stockMock[t]
	if !ok {
		data = stockData{PriceUSD: 100.0, ChangePercent: 0.0, Currency: "USD"}
	}
	data.Ticker = t
	data.MarketStatus = "open"
	b, _ := json.Marshal(data)
	return string(b)
}

func calculator(expression string) string {
	// Minimal safe evaluator: only handles basic numeric expressions.
	// In production use a proper expression parser.
	return fmt.Sprintf("(expression %q requires a proper math parser)", expression)
}

// dispatchTool routes a tool call to the correct implementation.
func dispatchTool(name string, argsJSON string) string {
	var args map[string]string
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return fmt.Sprintf(`{"error": "bad args: %s"}`, err)
	}
	switch name {
	case "get_weather":
		return getWeather(args["city"])
	case "get_stock_price":
		return getStockPrice(args["ticker"])
	case "calculator":
		return calculator(args["expression"])
	default:
		return fmt.Sprintf(`{"error": "unknown tool: %s"}`, name)
	}
}

// ---------------------------------------------------------------------------
// OpenAI tool definitions
// ---------------------------------------------------------------------------

var tools = []openai.Tool{
	{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name: "get_weather",
			Description: "Get current weather for a city. " +
				"City must include the city name, e.g. 'Tokyo'.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{
						"type":        "string",
						"description": "City name, e.g. 'Tokyo'.",
					},
				},
				"required":             []string{"city"},
				"additionalProperties": false,
			},
		},
	},
	{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name: "get_stock_price",
			Description: "Get current stock price and daily change for a ticker symbol, " +
				"e.g. 'AAPL'.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"ticker": map[string]any{
						"type":        "string",
						"description": "Ticker symbol in uppercase, e.g. 'AAPL'.",
					},
				},
				"required":             []string{"ticker"},
				"additionalProperties": false,
			},
		},
	},
	{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "calculator",
			Description: "Evaluate a simple arithmetic expression, e.g. '2 + 2'.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"expression": map[string]any{
						"type":        "string",
						"description": "Arithmetic expression.",
					},
				},
				"required":             []string{"expression"},
				"additionalProperties": false,
			},
		},
	},
}

// ---------------------------------------------------------------------------
// Go-native StateGraph
//
// This tiny runtime replicates LangGraph's core concept:
//   • Nodes      — named functions that transform state
//   • Edges      — fixed transitions (node A → node B)
//   • ConditionalEdges — routing function (A → one of several B)
//   • Run        — execute the graph from the entry node
//
// Compare with LangGraph Python:
//   workflow = StateGraph(AgentState)
//   workflow.add_node("agent", agent_node)
//   workflow.add_node("tools", tool_node)
//   workflow.add_conditional_edges("agent", should_continue, {...})
//   workflow.add_edge("tools", "agent")
//   graph = workflow.compile()
//   result = graph.invoke(initial_state)
// ---------------------------------------------------------------------------

// NodeFn is a function that processes state and returns updated state.
type NodeFn func(ctx context.Context, state AgentState) (AgentState, error)

// RouteFn chooses the next node given the current state.
type RouteFn func(state AgentState) string

// edge represents either a fixed transition or a conditional one.
type edge struct {
	target string            // non-empty for fixed edges
	route  RouteFn           // non-nil for conditional edges
	routes map[string]string // result → target mapping for conditional edges
}

// StateGraph is a minimal directed graph that runs NodeFns.
type StateGraph struct {
	nodes      map[string]NodeFn
	edges      map[string]edge
	entryPoint string
}

// NewStateGraph creates an empty graph.
func NewStateGraph() *StateGraph {
	return &StateGraph{
		nodes: make(map[string]NodeFn),
		edges: make(map[string]edge),
	}
}

// AddNode registers a named node.
func (g *StateGraph) AddNode(name string, fn NodeFn) {
	g.nodes[name] = fn
}

// SetEntryPoint sets the first node to execute.
func (g *StateGraph) SetEntryPoint(name string) {
	g.entryPoint = name
}

// AddEdge adds a fixed transition: from → to.
func (g *StateGraph) AddEdge(from, to string) {
	g.edges[from] = edge{target: to}
}

// AddConditionalEdges adds a routing-function transition from a node.
// routeMap maps the return value of route() to a target node name.
func (g *StateGraph) AddConditionalEdges(from string, route RouteFn, routeMap map[string]string) {
	g.edges[from] = edge{route: route, routes: routeMap}
}

// Run executes the graph starting from the entry point.
func (g *StateGraph) Run(ctx context.Context, state AgentState) (AgentState, error) {
	current := g.entryPoint
	for current != reactNodeEnd {
		nodeFn, ok := g.nodes[current]
		if !ok {
			return state, fmt.Errorf("unknown node %q", current)
		}

		var err error
		state, err = nodeFn(ctx, state)
		if err != nil {
			return state, fmt.Errorf("node %q: %w", current, err)
		}

		e, ok := g.edges[current]
		if !ok {
			// No outgoing edge — done
			break
		}
		if e.target != "" {
			current = e.target
		} else if e.route != nil {
			result := e.route(state)
			next, ok := e.routes[result]
			if !ok {
				return state, fmt.Errorf("route %q not found in edge from %q", result, current)
			}
			current = next
		}
	}
	return state, nil
}

// ---------------------------------------------------------------------------
// Agent implementation using the Go StateGraph
// ---------------------------------------------------------------------------

const systemPrompt = `You are an AI assistant with access to tools.

## Your Process
1. When the user asks a question, determine if you need a tool.
2. If yes, call the appropriate tool with the correct parameters.
3. Wait for the tool result, then decide: need more tools or ready to answer?
4. Never guess tool results. Always wait for the actual result.

## Answer Format
- Use tool results to answer the user's question directly.
- Cite specific data from tool results.
- If multiple tools were used, synthesise the information.`

// GoNativeReActAgent is the Go equivalent of LangGraphReActAgent.
// It uses the local StateGraph instead of LangGraph.
type GoNativeReActAgent struct {
	client        *openai.Client
	maxIterations int
	graph         *StateGraph
}

// NewGoNativeReActAgent creates and wires the agent graph.
func NewGoNativeReActAgent(maxIter int) *GoNativeReActAgent {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	a := &GoNativeReActAgent{client: client, maxIterations: maxIter}
	a.graph = a.buildGraph()
	return a
}

func (a *GoNativeReActAgent) buildGraph() *StateGraph {
	g := NewStateGraph()

	// ── Agent node ──────────────────────────────────────────────────
	// Equivalent to agent_node in LangGraph Python:
	//   def agent_node(state): response = llm.invoke(state["messages"]); return {...}
	g.AddNode(reactNodeAgent, func(ctx context.Context, state AgentState) (AgentState, error) {
		resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    reactLLMModel,
			Messages: state.Messages,
			Tools:    tools,
		})
		if err != nil {
			return state, err
		}

		msg := resp.Choices[0].Message
		state.Messages = append(state.Messages, msg)
		state.IterationCount++
		return state, nil
	})

	// ── Tool node ────────────────────────────────────────────────────
	// Equivalent to tool_node in LangGraph Python:
	//   def tool_node(state): execute each tool_call; append ToolMessages
	g.AddNode(reactNodeTool, func(ctx context.Context, state AgentState) (AgentState, error) {
		last := state.Messages[len(state.Messages)-1]
		for _, tc := range last.ToolCalls {
			result := dispatchTool(tc.Function.Name, tc.Function.Arguments)
			state.Messages = append(state.Messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    result,
				ToolCallID: tc.ID,
			})
			state.ToolCallsMade = append(state.ToolCallsMade, tc.Function.Name)
		}
		return state, nil
	})

	// ── Routing ──────────────────────────────────────────────────────
	// Equivalent to should_continue in LangGraph Python:
	//   def should_continue(state): return "tools" or "end"
	shouldContinue := func(state AgentState) string {
		if state.IterationCount >= a.maxIterations {
			return "end"
		}
		last := state.Messages[len(state.Messages)-1]
		if len(last.ToolCalls) > 0 {
			return "tools"
		}
		return "end"
	}

	// ── Wire the graph ───────────────────────────────────────────────
	// LangGraph Python equivalent:
	//   workflow.set_entry_point("agent")
	//   workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
	//   workflow.add_edge("tools", "agent")
	g.SetEntryPoint(reactNodeAgent)
	g.AddConditionalEdges(reactNodeAgent, shouldContinue, map[string]string{
		"tools": reactNodeTool,
		"end":   reactNodeEnd,
	})
	g.AddEdge(reactNodeTool, reactNodeAgent)

	return g
}

// Run executes the ReAct loop and returns a structured result.
func (a *GoNativeReActAgent) Run(ctx context.Context, userInput string) (AgentResult, error) {
	start := time.Now()

	initial := AgentState{
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
			{Role: openai.ChatMessageRoleUser, Content: userInput},
		},
		IterationCount: 0,
		ToolCallsMade:  []string{},
	}

	final, err := a.graph.Run(ctx, initial)
	if err != nil {
		return AgentResult{}, err
	}

	elapsed := float64(time.Since(start).Milliseconds())
	lastMsg := final.Messages[len(final.Messages)-1]

	return AgentResult{
		Answer:        lastMsg.Content,
		Iterations:    final.IterationCount,
		ToolCallsMade: final.ToolCallsMade,
		ElapsedMs:     elapsed,
	}, nil
}

// ---------------------------------------------------------------------------
// Visualise the graph as ASCII art
// ---------------------------------------------------------------------------

func visualize() string {
	return `
Go-Native State Machine Agent
(Graph-based workflow without LangGraph)
──────────────────────────────────────────

┌─────────┐
│  START  │
└────┬────┘
     ▼
┌─────────┐     has ToolCalls     ┌─────────┐
│  agent  │──────────────────────▶│  tools  │
└─────────┘                       └────┬────┘
     │                                 │
     │  no ToolCalls / maxIterations   │ (always)
     ▼                                 │
┌─────────┐ ◀──────────────────────────┘
│   END   │
└─────────┘

Go StateGraph concepts vs. LangGraph Python:
  Go NodeFn      ←→  Python def agent_node(state) -> dict
  Go RouteFn     ←→  Python def should_continue(state) -> str
  g.AddNode      ←→  workflow.add_node(...)
  g.AddEdge      ←→  workflow.add_edge(...)
  g.AddConditionalEdges ←→ workflow.add_conditional_edges(...)
  g.Run(ctx, state)  ←→  graph.invoke(initial_state)

Key insight: the concepts are identical.
Go just requires you to write the ~50-line runtime yourself.
In return you get: type safety, zero framework dependency, trivial debugging.
`
}

// ---------------------------------------------------------------------------
// Main / demo
// ---------------------------------------------------------------------------

// RunLangraphAlternativeDemo is the entry point for the standalone demo.
// To run: go run langgraph_alternative.go hybrid_rag_agent.go
func RunLangraphAlternativeDemo() {
	fmt.Println(visualize())

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Set OPENAI_API_KEY to run the demo.")
		fmt.Println("(The state machine architecture is shown above regardless.)")
		return
	}

	agent := NewGoNativeReActAgent(reactMaxIter)
	query := "What's the weather in Tokyo and should I invest in AAPL?"

	fmt.Printf("Query: %q\n\n", query)
	fmt.Println("Running Go-native state machine agent...")

	ctx := context.Background()
	start := time.Now()
	result, err := agent.Run(ctx, query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Agent error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Answer: %s\n\n", result.Answer)
	fmt.Printf("Iterations:     %d\n", result.Iterations)
	fmt.Printf("Tool calls:     %s\n", strings.Join(result.ToolCallsMade, ", "))
	fmt.Printf("Elapsed:        %.0f ms\n", float64(time.Since(start).Milliseconds()))

	fmt.Println("\nConclusion:")
	fmt.Println("  The same graph-based ReAct pattern works in Go without LangGraph.")
	fmt.Println("  The Go StateGraph above is ~50 lines; LangGraph is a multi-MB framework.")
	fmt.Println("  For simple agents, the native approach is preferable in Go.")
}
