package main

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// ---------------------------------------------------------------------------
// SimpleMCPServer — minimal boilerplate MCP server builder
// ---------------------------------------------------------------------------

// ToolHandler is the function signature for a registered tool.
// It receives a map of string→any arguments and returns a result string or error.
type ToolHandler func(args map[string]any) (string, error)

// registeredTool bundles a tool definition with its handler.
type registeredTool struct {
	tool    mcp.Tool
	handler ToolHandler
}

// registeredResource bundles a resource URI with its content provider.
type registeredResource struct {
	uri     string
	getName func() string
	getBody func() string
}

// SimpleMCPServer wraps mcp-go with a higher-level registration API.
type SimpleMCPServer struct {
	name      string
	version   string
	tools     []registeredTool
	resources []registeredResource
}

// NewSimpleMCPServer creates a server with the given name and version.
func NewSimpleMCPServer(name, version string) *SimpleMCPServer {
	return &SimpleMCPServer{name: name, version: version}
}

// AddTool registers a tool with the server.
//
//	def is an mcp.Tool; handler is called when the client invokes the tool.
func (s *SimpleMCPServer) AddTool(def mcp.Tool, handler ToolHandler) {
	s.tools = append(s.tools, registeredTool{tool: def, handler: handler})
}

// AddResource registers a static-content resource at the given URI.
func (s *SimpleMCPServer) AddResource(uri string, getName func() string, getBody func() string) {
	s.resources = append(s.resources, registeredResource{uri: uri, getName: getName, getBody: getBody})
}

// Build constructs and returns an mcp-go server.Server instance.
func (s *SimpleMCPServer) Build() *server.MCPServer {
	srv := server.NewMCPServer(s.name, s.version)

	for _, rt := range s.tools {
		// Capture loop variable
		handler := rt.handler
		srv.AddTool(rt.tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			// Convert typed arguments to map[string]any
			args := map[string]any{}
			if req.Params.Arguments != nil {
				v := reflect.ValueOf(req.Params.Arguments)
				if v.Kind() == reflect.Map {
					for _, key := range v.MapKeys() {
						args[fmt.Sprint(key.Interface())] = v.MapIndex(key).Interface()
					}
				}
			}
			result, err := handler(args)
			if err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText(result), nil
		})
	}

	return srv
}

// ---------------------------------------------------------------------------
// Schema helpers
// ---------------------------------------------------------------------------

// StringProp returns a JSON Schema property descriptor for a string parameter.
func StringProp(description string) map[string]any {
	return map[string]any{"type": "string", "description": description}
}

// IntProp returns a JSON Schema property descriptor for an integer parameter.
func IntProp(description string) map[string]any {
	return map[string]any{"type": "integer", "description": description}
}

// FloatProp returns a JSON Schema property descriptor for a number parameter.
func FloatProp(description string) map[string]any {
	return map[string]any{"type": "number", "description": description}
}

// BuildToolSchema returns an mcp.Tool with a JSON Schema input spec.
func BuildToolSchema(name, description string, properties map[string]any, required []string) mcp.Tool {
	schema := map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
	schemaBytes, _ := json.Marshal(schema)
	return mcp.NewToolWithRawSchema(name, description, schemaBytes)
}

// ---------------------------------------------------------------------------
// Demo server
// ---------------------------------------------------------------------------

// buildDemoServer constructs a demo SimpleMCPServer with example tools.
func buildDemoServer() *SimpleMCPServer {
	srv := NewSimpleMCPServer("demo-tools", "1.0.0")

	// Tool 1: echo
	echoTool := BuildToolSchema("echo", "Return the input string unchanged",
		map[string]any{"message": StringProp("The text to echo")},
		[]string{"message"})
	srv.AddTool(echoTool, func(args map[string]any) (string, error) {
		msg, _ := args["message"].(string)
		return msg, nil
	})

	// Tool 2: add
	addTool := BuildToolSchema("add", "Add two integers",
		map[string]any{
			"a": IntProp("First operand"),
			"b": IntProp("Second operand"),
		},
		[]string{"a", "b"})
	srv.AddTool(addTool, func(args map[string]any) (string, error) {
		a := toInt(args["a"])
		b := toInt(args["b"])
		return fmt.Sprintf("%d", a+b), nil
	})

	// Tool 3: upper
	upperTool := BuildToolSchema("upper", "Convert text to upper case",
		map[string]any{"text": StringProp("Text to convert")},
		[]string{"text"})
	srv.AddTool(upperTool, func(args map[string]any) (string, error) {
		text, _ := args["text"].(string)
		return strings.ToUpper(text), nil
	})

	return srv
}

// toInt coerces a JSON-decoded value to int.
func toInt(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	}
	return 0
}

// RunSimpleMCPServerDemo prints server info and starts the stdio MCP server.
func RunSimpleMCPServerDemo() {
	srv := buildDemoServer()

	fmt.Printf("SimpleMCPServer %q v%s — %d tool(s) registered\n",
		srv.name, srv.version, len(srv.tools))
	for _, rt := range srv.tools {
		fmt.Printf("  • %s\n", rt.tool.Name)
	}
	fmt.Println()
	fmt.Println("Starting MCP server on stdio (send JSON-RPC 2.0 messages)...")

	mcpSrv := srv.Build()
	if err := server.ServeStdio(mcpSrv); err != nil {
		fmt.Printf("server error: %v\n", err)
	}
}
