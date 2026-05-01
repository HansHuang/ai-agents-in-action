// Tool execution dispatcher for the ReAct agent loop.
//
// DispatchTool() bridges the LLM's tool call and your Go functions.
// Tool execution errors are returned as tool messages (not Go errors) so the
// LLM can explain failures to the user without crashing the loop.
//
// See docs/02-the-agent-loop/01-anatomy-of-an-agent.md — "The Hands (Tools)"

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/openai/openai-go"
)

// ToolFunc is the signature every registered tool must implement.
type ToolFunc func(args map[string]interface{}) (interface{}, error)

// ToolRegistry maps tool names to their implementations.
type ToolRegistry map[string]ToolFunc

// DispatchTool executes the tool requested by the LLM and returns a
// properly formatted tool message for the messages array.
func DispatchTool(
	tc openai.ChatCompletionMessageToolCall,
	registry ToolRegistry,
) openai.ChatCompletionMessageParamUnion {
	name := tc.Function.Name
	toolCallID := tc.ID

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		log.Printf("[DispatchTool] Tool %q: invalid argument JSON: %v", name, err)
		return errorToolMessage(toolCallID, fmt.Sprintf("Invalid arguments JSON: %v", err))
	}

	fn, ok := registry[name]
	if !ok {
		keys := make([]string, 0, len(registry))
		for k := range registry {
			keys = append(keys, k)
		}
		available := strings.Join(keys, ", ")
		log.Printf("[DispatchTool] Tool %q not found. Available: %s", name, available)
		return errorToolMessage(
			toolCallID,
			fmt.Sprintf("Tool '%s' is not available. Available tools: %s", name, available),
		)
	}

	start := time.Now()
	result, err := fn(args)
	elapsed := time.Since(start)

	if err != nil {
		log.Printf("[DispatchTool] Tool %q failed (%v): %v", name, elapsed, err)
		return errorToolMessage(toolCallID, fmt.Sprintf("Tool '%s' failed: %v", name, err))
	}

	resultBytes, marshalErr := json.Marshal(result)
	if marshalErr != nil {
		log.Printf("[DispatchTool] Tool %q: failed to marshal result: %v", name, marshalErr)
		return errorToolMessage(toolCallID, fmt.Sprintf("Tool '%s' returned an unmarshalable result", name))
	}

	resultJSON := string(resultBytes)
	preview := resultJSON
	if len(preview) > 200 {
		preview = preview[:200] + "…"
	}
	log.Printf("[DispatchTool] %s(%s) → %s  [%v]", name, tc.Function.Arguments, preview, elapsed.Round(time.Millisecond))

	// openai.ToolMessage signature: ToolMessage(content, toolCallID)
	return openai.ToolMessage(resultJSON, toolCallID)
}

// errorToolMessage builds a tool message whose content describes the failure.
func errorToolMessage(toolCallID, errText string) openai.ChatCompletionMessageParamUnion {
	content, _ := json.Marshal(map[string]string{"error": errText})
	return openai.ToolMessage(string(content), toolCallID)
}
