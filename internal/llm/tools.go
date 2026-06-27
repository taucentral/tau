// tools.go — ToolSchema helpers shared by all providers.
//
// ToolSchema is the provider-agnostic shape a tool implementation publishes:
// a name, a human-facing description, and a JSON Schema for parameters. The
// struct itself lives in client.go (it's part of the Request payload); this
// file adds constructors, validation, and accessor helpers used by the
// marshal layers in internal/llm/provider/{anthropic,openai}/.
//
// Wire-format conversion is intentionally NOT in this file — each provider
// has its own JSON shape for tools, so the conversion lives in that
// provider's marshal.go. The helpers here only do provider-independent work:
//
//   - NewSchema       build a ToolSchema from a Go value via invopop/jsonschema
//   - Validate       ensure Name is non-empty and Parameters is a valid object schema
//   - ParameterMap   parse Parameters into a map[string]any for provider accessors

package llm

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/invopop/jsonschema"
)

// ErrInvalidToolSchema is returned by Validate when a ToolSchema is malformed
// (empty name, parameters not a JSON object, parameters with type!=object).
var ErrInvalidToolSchema = errors.New("invalid tool schema")

// NewToolSchemaFromStruct builds a ToolSchema by reflecting over an example
// value of T. The struct MUST have json tags or jsonschema tags; fields
// without tags get their Go field name. This is the primary constructor for
// built-in tools whose parameters are typed Go structs.
func NewToolSchemaFromStruct[T any](name, description string, sample T) (ToolSchema, error) {
	r := new(jsonschema.Reflector)
	r.DoNotReference = true
	schema := r.Reflect(sample)
	if schema == nil {
		return ToolSchema{}, fmt.Errorf("%w: jsonschema.Reflect returned nil", ErrInvalidToolSchema)
	}
	// The reflector produces a $def-referenced schema by default; we want
	// the flat inline form for tools. DoNotReference flattens it but the
	// top-level still has $schema, $id, etc. Strip them.
	schema.Type = "object"
	raw, err := json.Marshal(schema)
	if err != nil {
		return ToolSchema{}, fmt.Errorf("%w: marshal reflected schema: %v", ErrInvalidToolSchema, err)
	}
	var flat map[string]any
	if err := json.Unmarshal(raw, &flat); err != nil {
		return ToolSchema{}, fmt.Errorf("%w: unmarshal reflected schema: %v", ErrInvalidToolSchema, err)
	}
	delete(flat, "$schema")
	delete(flat, "$id")
	delete(flat, "$defs")
	delete(flat, "$comment")
	raw, err = json.Marshal(flat)
	if err != nil {
		return ToolSchema{}, err
	}
	ts := ToolSchema{Name: name, Description: description, Parameters: raw}
	if err := ts.Validate(); err != nil {
		return ToolSchema{}, err
	}
	return ts, nil
}

// NewToolSchemaFromJSON builds a ToolSchema from a hand-written JSON Schema
// string. Useful in tests and for plugins that ship schemas verbatim.
func NewToolSchemaFromJSON(name, description, paramsJSON string) (ToolSchema, error) {
	ts := ToolSchema{
		Name:        name,
		Description: description,
		Parameters:  json.RawMessage(paramsJSON),
	}
	if err := ts.Validate(); err != nil {
		return ToolSchema{}, err
	}
	return ts, nil
}

// Validate enforces the minimum invariants all providers expect:
//   - Name is non-empty
//   - Parameters parses as a JSON object
//   - Parameters.type is "object" (or omitted, which we normalize to "object")
func (t ToolSchema) Validate() error {
	if t.Name == "" {
		return fmt.Errorf("%w: empty name", ErrInvalidToolSchema)
	}
	if len(t.Parameters) == 0 {
		return fmt.Errorf("%w: parameters missing", ErrInvalidToolSchema)
	}
	var probe map[string]any
	if err := json.Unmarshal(t.Parameters, &probe); err != nil {
		return fmt.Errorf("%w: parameters not a JSON object: %v", ErrInvalidToolSchema, err)
	}
	if probe["type"] != nil && probe["type"] != "object" {
		return fmt.Errorf("%w: parameters.type = %v, want object", ErrInvalidToolSchema, probe["type"])
	}
	return nil
}

// ParameterMap returns Parameters as a parsed map. Provider marshalers use
// this when they need to mutate the schema before serialization (e.g., add
// "additionalProperties": false for OpenAI strict mode).
func (t ToolSchema) ParameterMap() (map[string]any, error) {
	var m map[string]any
	if err := json.Unmarshal(t.Parameters, &m); err != nil {
		return nil, fmt.Errorf("%w: parameters not a JSON object: %v", ErrInvalidToolSchema, err)
	}
	return m, nil
}
