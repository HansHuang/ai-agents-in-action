// Programmatic tool definition builder with built-in validation.
//
// Provides ToolDef and Param types for constructing OpenAI-compatible
// function-calling tool definitions and validating arguments at runtime.
//
// Structurally identical to code/python/03-agent-loop/tool_builder.py
// and code/nodejs/03-agent-loop/tool_builder.ts.
//
// Usage:
//
//	tool := ToolDef{
//	    Name: "get_weather",
//	    Description: "Get current weather for a city. Returns temperature and conditions.",
//	    Parameters: []Param{
//	        {
//	            Name:        "city",
//	            Type:        "string",
//	            Required:    true,
//	            Description: "City name with country code. Example: 'Shanghai, SH'",
//	        },
//	    },
//	}
//
//	schema := tool.ToOpenAISchema()
//	err := tool.ValidateArgs(map[string]interface{}{"city": "Shanghai, SH"}) // nil
//	err  = tool.ValidateArgs(map[string]interface{}{"city": 123})            // error
//
// See docs/02-the-agent-loop/02-tool-design-patterns.md

package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// Supported types
// ---------------------------------------------------------------------------

var validParamTypes = map[string]bool{
	"string":  true,
	"integer": true,
	"number":  true,
	"boolean": true,
	"array":   true,
	"object":  true,
}

// ---------------------------------------------------------------------------
// Param
// ---------------------------------------------------------------------------

// Param describes a single parameter in a tool definition.
//
// Minimum and Maximum use pointers so that zero is distinguishable from "not set".
type Param struct {
	Name        string
	Type        string
	Required    bool
	Description string
	Enum        []interface{}
	Minimum     *float64
	Maximum     *float64
	Default     interface{}
}

// Validate returns an error if the Param is misconfigured.
func (p Param) Validate() error {
	if !validParamTypes[p.Type] {
		types := make([]string, 0, len(validParamTypes))
		for t := range validParamTypes {
			types = append(types, t)
		}
		return fmt.Errorf(
			"param %q: type must be one of [%s], got %q",
			p.Name, strings.Join(types, ", "), p.Type,
		)
	}
	return nil
}

// ToSchema returns the JSON Schema fragment for this parameter.
func (p Param) ToSchema() map[string]interface{} {
	schema := map[string]interface{}{
		"type":        p.Type,
		"description": p.Description,
	}
	if p.Enum != nil {
		schema["enum"] = p.Enum
	}
	if p.Minimum != nil {
		schema["minimum"] = *p.Minimum
	}
	if p.Maximum != nil {
		schema["maximum"] = *p.Maximum
	}
	if p.Default != nil {
		schema["default"] = p.Default
	}
	return schema
}

// ---------------------------------------------------------------------------
// ToolDef
// ---------------------------------------------------------------------------

// ToolDef is an OpenAI-compatible function-calling tool definition.
//
// When Strict is true, all properties are placed in required and
// additionalProperties is set to false, which is required by the API.
type ToolDef struct {
	Name        string
	Description string
	Parameters  []Param
	Strict      bool
}

// ToOpenAISchema generates the exact map expected by the OpenAI tools parameter.
//
// The returned map has the form:
//
//	{
//	    "type": "function",
//	    "function": {
//	        "name": "...",
//	        "description": "...",
//	        "strict": false,
//	        "parameters": {
//	            "type": "object",
//	            "properties": {...},
//	            "required": [...],
//	            "additionalProperties": false
//	        }
//	    }
//	}
func (td ToolDef) ToOpenAISchema() map[string]interface{} {
	properties := make(map[string]interface{}, len(td.Parameters))
	required := make([]string, 0)

	for _, p := range td.Parameters {
		properties[p.Name] = p.ToSchema()
		if td.Strict || p.Required {
			required = append(required, p.Name)
		}
	}

	return map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        td.Name,
			"description": td.Description,
			"strict":      td.Strict,
			"parameters": map[string]interface{}{
				"type":                 "object",
				"properties":           properties,
				"required":             required,
				"additionalProperties": false,
			},
		},
	}
}

// ValidateArgs validates args against this tool's parameter definitions.
//
// Checks:
//   - Required parameters are present.
//   - Each value matches its declared type.
//   - Values respect Enum, Minimum, and Maximum constraints.
//
// Returns a descriptive error on the first violation:
//
//	"parameter 'city' must be a string, got float64 (42)"
func (td ToolDef) ValidateArgs(args map[string]interface{}) error {
	// --- Required presence ---
	for _, p := range td.Parameters {
		if p.Required {
			if _, ok := args[p.Name]; !ok {
				return fmt.Errorf("missing required parameter: %q", p.Name)
			}
		}
	}

	// --- Per-parameter checks ---
	for _, p := range td.Parameters {
		value, ok := args[p.Name]
		if !ok {
			continue
		}
		if err := validateParamType(p.Name, p.Type, value); err != nil {
			return err
		}
		if err := validateEnum(p.Name, p.Enum, value); err != nil {
			return err
		}
		if err := validateRange(p.Name, p.Minimum, p.Maximum, value); err != nil {
			return err
		}
	}
	return nil
}

// FromDict creates a ToolDef from a plain map (e.g. parsed from JSON/YAML).
//
// Expected map structure mirrors the Python/TypeScript from_dict format:
//
//	{
//	    "name": "get_weather",
//	    "description": "...",
//	    "strict": false,
//	    "parameters": [
//	        {"name": "city", "type": "string", "required": true, "description": "..."}
//	    ]
//	}
func FromDict(data map[string]interface{}) (*ToolDef, error) {
	name, _ := data["name"].(string)
	if name == "" {
		return nil, fmt.Errorf("tool definition missing required field 'name'")
	}
	description, _ := data["description"].(string)
	strict, _ := data["strict"].(bool)

	rawParams, _ := data["parameters"].([]interface{})
	params := make([]Param, 0, len(rawParams))
	for _, rp := range rawParams {
		pm, ok := rp.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("parameter entry is not a map")
		}
		pName, _ := pm["name"].(string)
		pType, _ := pm["type"].(string)
		pRequired, _ := pm["required"].(bool)
		pDescription, _ := pm["description"].(string)

		p := Param{
			Name:        pName,
			Type:        pType,
			Required:    pRequired,
			Description: pDescription,
		}
		if enum, ok := pm["enum"].([]interface{}); ok {
			p.Enum = enum
		}
		if min, ok := pm["minimum"].(float64); ok {
			p.Minimum = &min
		}
		if max, ok := pm["maximum"].(float64); ok {
			p.Maximum = &max
		}
		p.Default = pm["default"]

		if err := p.Validate(); err != nil {
			return nil, err
		}
		params = append(params, p)
	}

	return &ToolDef{
		Name:        name,
		Description: description,
		Parameters:  params,
		Strict:      strict,
	}, nil
}

// ---------------------------------------------------------------------------
// Internal validation helpers
// ---------------------------------------------------------------------------

// validateParamType checks that value matches the declared JSON Schema type.
//
// Note: JSON numbers parsed by encoding/json arrive as float64. An integer
// check accepts float64 values that have no fractional part.
func validateParamType(name, expectedType string, value interface{}) error {
	switch expectedType {
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf(
				"parameter '%s' must be a string, got %T (%v)", name, value, value,
			)
		}
	case "integer":
		switch v := value.(type) {
		case int, int8, int16, int32, int64,
			uint, uint8, uint16, uint32, uint64:
			// native Go integer — OK
		case float64:
			if v != float64(int64(v)) {
				return fmt.Errorf(
					"parameter '%s' must be an integer, got float64 (%v)", name, value,
				)
			}
		default:
			return fmt.Errorf(
				"parameter '%s' must be an integer, got %T (%v)", name, value, value,
			)
		}
	case "number":
		switch value.(type) {
		case int, int8, int16, int32, int64,
			uint, uint8, uint16, uint32, uint64,
			float32, float64:
			// OK
		default:
			return fmt.Errorf(
				"parameter '%s' must be a number, got %T (%v)", name, value, value,
			)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf(
				"parameter '%s' must be a boolean, got %T (%v)", name, value, value,
			)
		}
	case "array":
		if _, ok := value.([]interface{}); !ok {
			return fmt.Errorf(
				"parameter '%s' must be an array, got %T (%v)", name, value, value,
			)
		}
	case "object":
		if _, ok := value.(map[string]interface{}); !ok {
			return fmt.Errorf(
				"parameter '%s' must be an object, got %T (%v)", name, value, value,
			)
		}
	}
	return nil
}

func validateEnum(name string, enum []interface{}, value interface{}) error {
	if len(enum) == 0 {
		return nil
	}
	for _, allowed := range enum {
		if allowed == value {
			return nil
		}
	}
	return fmt.Errorf("parameter '%s' must be one of %v, got %v", name, enum, value)
}

func validateRange(name string, minimum, maximum *float64, value interface{}) error {
	var num float64
	switch v := value.(type) {
	case float64:
		num = v
	case int:
		num = float64(v)
	case int64:
		num = float64(v)
	default:
		return nil
	}
	if minimum != nil && num < *minimum {
		return fmt.Errorf(
			"parameter '%s' must be >= %v, got %v", name, *minimum, value,
		)
	}
	if maximum != nil && num > *maximum {
		return fmt.Errorf(
			"parameter '%s' must be <= %v, got %v", name, *maximum, value,
		)
	}
	return nil
}

// jsonMarshalSchema is a convenience helper for embedding a schema in JSON output.
func jsonMarshalSchema(td ToolDef) ([]byte, error) {
	return json.Marshal(td.ToOpenAISchema())
}
