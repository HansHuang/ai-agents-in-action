// MCP-Enabled Agent — Go
//
// Discovers and calls tools from MCP servers using the
// {serverName}__{toolName} namespace pattern.
//
// Usage:
//
//	OPENAI_API_KEY=sk-... go run mcp_agent.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	openai "github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// AgentResult holds the outcome of MCPAgent.Run().
type AgentResult struct {
	Answer      string
	Messages    []openai.ChatCompletionMessage
	ToolsCalled []string
}

// ToolResult holds the outcome of one MCP tool execution.
type ToolResult struct {
	ServerName string
	ToolName   string
	Output     string
	IsError    bool
}

// ---------------------------------------------------------------------------
// ServerConnection — one long-lived MCP session
// ---------------------------------------------------------------------------

// ServerConnection wraps an mcp-go StdioMCPClient and the tools it exposes.
type ServerConnection struct {
	Name      string
	Client    *client.StdioMCPClient
	Tools     []mcp.Tool
	Connected bool
}

// Connect starts the server subprocess and initialises the MCP session.
func (sc *ServerConnection) Connect(command string, args []string) error {
	c, err := client.NewStdioMCPClient(command, args)
	if err != nil {
		return fmt.Errorf("failed to create stdio client for %q: %w", sc.Name, err)
	}
	sc.Client = c

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    fmt.Sprintf("agent-client-%s", sc.Name),
		Version: "1.0.0",
	}
	initReq.Params.Capabilities = mcp.ClientCapabilities{}

	if _, err = c.Initialize(ctx, initReq); err != nil {
		return fmt.Errorf("MCP initialize failed for %q: %w", sc.Name, err)
	}

	toolsCtx, cancel2 := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel2()

	toolsResult, err := c.ListTools(toolsCtx, mcp.ListToolsRequest{})
	if err != nil {
		return fmt.Errorf("list_tools failed for %q: %w", sc.Name, err)
	}
	sc.Tools = toolsResult.Tools
	sc.Connected = true

	fmt.Fprintf(os.Stderr,
		"[registry] Connected to %q: %d tools discovered\n",
		sc.Name, len(sc.Tools),
	)
	return nil
}

// CallTool executes toolName on this server and returns a ToolResult.
func (sc *ServerConnection) CallTool(toolName string, args map[string]any) ToolResult {
	if !sc.Connected || sc.Client == nil {
		return ToolResult{
			ServerName: sc.Name,
			ToolName:   toolName,
			Output:     fmt.Sprintf(`{"error":"Server %q is not connected"}`, sc.Name),
			IsError:    true,
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := mcp.CallToolRequest{}
	req.Params.Name = toolName
	req.Params.Arguments = args

	result, err := sc.Client.CallTool(ctx, req)
	if err != nil {
		sc.Connected = false
		return ToolResult{
			ServerName: sc.Name,
			ToolName:   toolName,
			Output:     fmt.Sprintf(`{"error":"tool execution failed: %v"}`, err),
			IsError:    true,
		}
	}

	var sb strings.Builder
	for _, c := range result.Content {
		if tc, ok := c.(mcp.TextContent); ok {
			sb.WriteString(tc.Text)
		}
	}
	output := sb.String()
	if output == "" {
		output = "(empty response)"
	}

	return ToolResult{
		ServerName: sc.Name,
		ToolName:   toolName,
		Output:     output,
		IsError:    result.IsError,
	}
}

// Close terminates the subprocess.
func (sc *ServerConnection) Close() {
	sc.Connected = false
	if sc.Client != nil {
		sc.Client.Close()
	}
}

// ---------------------------------------------------------------------------
// ServerRegistry — manages multiple connections
// ---------------------------------------------------------------------------

type connectParams struct {
	Command string
	Args    []string
}

// ServerRegistry manages connections to multiple MCP servers.
type ServerRegistry struct {
	Servers map[string]*ServerConnection
	params  map[string]connectParams
}

// NewServerRegistry creates an empty registry.
func NewServerRegistry() *ServerRegistry {
	return &ServerRegistry{
		Servers: make(map[string]*ServerConnection),
		params:  make(map[string]connectParams),
	}
}

// Connect starts a server and returns its tools in OpenAI format.
func (r *ServerRegistry) Connect(
	name, command string, args []string,
) ([]openai.Tool, error) {
	conn := &ServerConnection{Name: name}
	if err := conn.Connect(command, args); err != nil {
		return nil, err
	}
	r.Servers[name] = conn
	r.params[name] = connectParams{Command: command, Args: args}
	return r.formatTools(conn, "openai"), nil
}

func (r *ServerRegistry) formatTools(conn *ServerConnection, fmt_ string) []openai.Tool {
	tools := make([]openai.Tool, 0, len(conn.Tools))
	for _, t := range conn.Tools {
		tools = append(tools, r.toOpenAI(t, conn.Name))
	}
	return tools
}

func (r *ServerRegistry) toOpenAI(t mcp.Tool, serverName string) openai.Tool {
	// mcp.Tool.InputSchema is map[string]any — marshal to json.RawMessage
	paramBytes, _ := json.Marshal(t.InputSchema)
	def := &openai.FunctionDefinition{
		Name:        fmt.Sprintf("%s__%s", serverName, t.Name),
		Description: t.Description,
		Parameters:  json.RawMessage(paramBytes),
	}
	return openai.Tool{Type: openai.ToolTypeFunction, Function: def}
}

// GetAllTools returns tools from all connected servers in OpenAI format.
func (r *ServerRegistry) GetAllTools() []openai.Tool {
	var tools []openai.Tool
	for _, conn := range r.Servers {
		if conn.Connected {
			tools = append(tools, r.formatTools(conn, "openai")...)
		}
	}
	return tools
}

// CallTool routes a tool call to the appropriate server.
func (r *ServerRegistry) CallTool(
	serverName, toolName string,
	args map[string]any,
) string {
	conn, ok := r.Servers[serverName]
	if !ok {
		return fmt.Sprintf(`{"error":"unknown server: %q"}`, serverName)
	}
	result := conn.CallTool(toolName, args)
	return result.Output
}

// Reconnect re-establishes a disconnected server connection.
func (r *ServerRegistry) Reconnect(serverName string) bool {
	p, ok := r.params[serverName]
	if !ok {
		return false
	}
	if old, exists := r.Servers[serverName]; exists {
		old.Close()
	}
	conn := &ServerConnection{Name: serverName}
	if err := conn.Connect(p.Command, p.Args); err != nil {
		fmt.Fprintf(os.Stderr, "[registry] Reconnect failed for %q: %v\n", serverName, err)
		return false
	}
	r.Servers[serverName] = conn
	fmt.Fprintf(os.Stderr, "[registry] Reconnected to %q\n", serverName)
	return true
}

// DisconnectAll closes every server connection.
func (r *ServerRegistry) DisconnectAll() {
	for _, conn := range r.Servers {
		conn.Close()
	}
	r.Servers = make(map[string]*ServerConnection)
}

// HealthCheck returns a status snapshot for all registered servers.
func (r *ServerRegistry) HealthCheck() map[string]map[string]any {
	result := make(map[string]map[string]any, len(r.Servers))
	for name, conn := range r.Servers {
		toolNames := make([]string, len(conn.Tools))
		for i, t := range conn.Tools {
			toolNames[i] = t.Name
		}
		result[name] = map[string]any{
			"connected":  conn.Connected,
			"tool_count": len(conn.Tools),
			"tools":      toolNames,
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// MCPAgent
// ---------------------------------------------------------------------------

const maxIterations = 10

// MCPAgent discovers and uses tools from MCP servers.
type MCPAgent struct {
	LLM      *openai.Client
	Model    string
	Registry *ServerRegistry
	Tools    []openai.Tool
}

// NewMCPAgent creates a new agent with an OpenAI client.
func NewMCPAgent(model string) *MCPAgent {
	apiKey := os.Getenv("OPENAI_API_KEY")
	return &MCPAgent{
		LLM:      openai.NewClient(apiKey),
		Model:    model,
		Registry: NewServerRegistry(),
	}
}

// ConnectServer connects to an MCP server and registers its tools.
// Returns the number of tools discovered.
func (a *MCPAgent) ConnectServer(name, command string, args []string) (int, error) {
	newTools, err := a.Registry.Connect(name, command, args)
	if err != nil {
		return 0, err
	}
	a.Tools = append(a.Tools, newTools...)
	fmt.Fprintf(os.Stderr,
		"[agent] +%d tools from %q. Total: %d\n",
		len(newTools), name, len(a.Tools),
	)
	return len(newTools), nil
}

// DisconnectServer disconnects from a server and removes its tools.
func (a *MCPAgent) DisconnectServer(name string) {
	if conn, ok := a.Registry.Servers[name]; ok {
		conn.Close()
		delete(a.Registry.Servers, name)
	}
	a.Tools = a.Registry.GetAllTools()
	fmt.Fprintf(os.Stderr, "[agent] Disconnected from %q\n", name)
}

// Run executes the agent loop with all MCP-discovered tools.
func (a *MCPAgent) Run(userInput string) (AgentResult, error) {
	messages := []openai.ChatCompletionMessage{
		{
			Role: openai.ChatMessageRoleSystem,
			Content: "You are a helpful assistant. Use the available tools " +
				"whenever needed to answer the user's question accurately.",
		},
		{Role: openai.ChatMessageRoleUser, Content: userInput},
	}
	var toolsCalled []string

	for i := 0; i < maxIterations; i++ {
		req := openai.ChatCompletionRequest{
			Model:    a.Model,
			Messages: messages,
		}
		if len(a.Tools) > 0 {
			req.Tools = a.Tools
		}

		resp, err := a.LLM.CreateChatCompletion(context.Background(), req)
		if err != nil {
			return AgentResult{}, fmt.Errorf("LLM error: %w", err)
		}

		msg := resp.Choices[0].Message
		messages = append(messages, msg)

		if len(msg.ToolCalls) == 0 {
			return AgentResult{
				Answer:      msg.Content,
				Messages:    messages,
				ToolsCalled: toolsCalled,
			}, nil
		}

		for _, tc := range msg.ToolCalls {
			fullName := tc.Function.Name
			result, execErr := a.executeMCPTool(fullName, tc.Function.Arguments)
			if execErr != nil {
				result = &ToolResult{Output: fmt.Sprintf(`{"error":"%v"}`, execErr)}
			}
			toolsCalled = append(toolsCalled, fullName)
			messages = append(messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				ToolCallID: tc.ID,
				Content:    result.Output,
			})
		}
	}

	return AgentResult{
		Answer:      "Maximum iterations reached without a final answer.",
		Messages:    messages,
		ToolsCalled: toolsCalled,
	}, nil
}

func (a *MCPAgent) executeMCPTool(fullName, argsJSON string) (*ToolResult, error) {
	idx := strings.Index(fullName, "__")
	if idx < 0 {
		return &ToolResult{
			Output:  fmt.Sprintf(`{"error":"invalid tool name %q"}`, fullName),
			IsError: true,
		}, nil
	}
	serverName := fullName[:idx]
	toolName := fullName[idx+2:]

	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		args = map[string]any{}
	}

	fmt.Fprintf(os.Stderr, "[agent] → %s/%s args=%s\n", serverName, toolName, argsJSON)
	output := a.Registry.CallTool(serverName, toolName, args)
	return &ToolResult{
		ServerName: serverName,
		ToolName:   toolName,
		Output:     output,
	}, nil
}

// Close disconnects from all servers.
func (a *MCPAgent) Close() {
	a.Registry.DisconnectAll()
	fmt.Fprintln(os.Stderr, "[agent] All servers disconnected.")
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

func main() {
	// Locate the weather server relative to this file
	_, thisFile, _, _ := runtime.Caller(0)
	serverPath := filepath.Join(filepath.Dir(thisFile), "weather_mcp_server")

	fmt.Fprintln(os.Stderr, strings.Repeat("=", 60))
	fmt.Fprintln(os.Stderr, "MCP Agent Demo (Go)")
	fmt.Fprintln(os.Stderr, strings.Repeat("=", 60))

	agent := NewMCPAgent("gpt-4o")

	// Connect to the weather server
	if _, err := agent.ConnectServer(
		"weather",
		"go",
		[]string{"run", serverPath},
	); err != nil {
		fmt.Fprintf(os.Stderr, "[demo] Failed to connect to weather server: %v\n", err)
		os.Exit(1)
	}

	// Show discovered tools
	fmt.Fprintln(os.Stderr, "\n--- Discovered Tools ---")
	for _, t := range agent.Tools {
		fmt.Fprintf(os.Stderr, "  %s\n", t.Function.Name)
	}

	// Run queries
	queries := []string{
		"What's the weather in Tokyo?",
		"What's the weather in London and Dubai?",
		"Give me a 3-day forecast for Paris.",
	}

	for _, q := range queries {
		fmt.Fprintln(os.Stderr, "\n"+strings.Repeat("─", 50))
		fmt.Fprintf(os.Stderr, "Query: %s\n", q)
		result, err := agent.Run(q)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		fmt.Fprintf(os.Stderr, "Answer: %s\n", result.Answer)
		fmt.Fprintf(os.Stderr, "Tools called: %s\n", strings.Join(result.ToolsCalled, ", "))
	}

	// Health check
	healthJSON, _ := json.MarshalIndent(agent.Registry.HealthCheck(), "", "  ")
	fmt.Fprintf(os.Stderr, "\n--- Health Check ---\n%s\n", healthJSON)

	agent.Close()
}
