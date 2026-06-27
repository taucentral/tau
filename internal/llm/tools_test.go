package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolSchema_Validate_OK(t *testing.T) {
	ts := ToolSchema{
		Name:        "read",
		Description: "Read a file",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
	}
	if err := ts.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

func TestToolSchema_Validate_EmptyName(t *testing.T) {
	ts := ToolSchema{
		Name:       "",
		Parameters: json.RawMessage(`{"type":"object"}`),
	}
	if err := ts.Validate(); err == nil {
		t.Errorf("expected error for empty name")
	}
}

func TestToolSchema_Validate_MissingParameters(t *testing.T) {
	ts := ToolSchema{Name: "x"}
	if err := ts.Validate(); err == nil {
		t.Errorf("expected error for missing parameters")
	}
}

func TestToolSchema_Validate_MalformedJSON(t *testing.T) {
	ts := ToolSchema{
		Name:       "x",
		Parameters: json.RawMessage(`not json`),
	}
	if err := ts.Validate(); err == nil {
		t.Errorf("expected error for malformed JSON")
	}
}

func TestToolSchema_Validate_NonObjectSchema(t *testing.T) {
	ts := ToolSchema{
		Name:       "x",
		Parameters: json.RawMessage(`{"type":"string"}`),
	}
	if err := ts.Validate(); err == nil {
		t.Errorf("expected error for type=string")
	}
}

func TestToolSchema_Validate_TypeImplicitObject(t *testing.T) {
	// No "type" field — providers assume object.
	ts := ToolSchema{
		Name:       "x",
		Parameters: json.RawMessage(`{"properties":{}}`),
	}
	if err := ts.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

func TestToolSchema_ParameterMap(t *testing.T) {
	ts := ToolSchema{
		Name:       "x",
		Parameters: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`),
	}
	m, err := ts.ParameterMap()
	if err != nil {
		t.Fatalf("ParameterMap: %v", err)
	}
	if m["type"] != "object" {
		t.Errorf("type = %v", m["type"])
	}
	props, ok := m["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties not a map: %T", m["properties"])
	}
	if _, ok := props["a"]; !ok {
		t.Errorf("properties.a missing")
	}
}

type sampleParams struct {
	Path    string `json:"path" jsonschema:"required,description=Path to read"`
	MaxRows int    `json:"maxRows,omitempty" jsonschema:"minimum=1"`
}

func TestNewToolSchemaFromStruct(t *testing.T) {
	ts, err := NewToolSchemaFromStruct("read", "Read a file", sampleParams{})
	if err != nil {
		t.Fatalf("NewToolSchemaFromStruct: %v", err)
	}
	if ts.Name != "read" {
		t.Errorf("Name = %q", ts.Name)
	}
	if ts.Description != "Read a file" {
		t.Errorf("Description = %q", ts.Description)
	}
	// The reflected schema should contain "path" in properties.
	m, err := ts.ParameterMap()
	if err != nil {
		t.Fatalf("ParameterMap: %v", err)
	}
	props, ok := m["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties not a map: %T", m["properties"])
	}
	if _, ok := props["path"]; !ok {
		t.Errorf("path property missing; schema=%s", ts.Parameters)
	}
}

func TestNewToolSchemaFromStruct_StripsMetaFields(t *testing.T) {
	ts, _ := NewToolSchemaFromStruct("x", "y", sampleParams{})
	if strings.Contains(string(ts.Parameters), "$schema") {
		t.Errorf("$schema not stripped: %s", ts.Parameters)
	}
	if strings.Contains(string(ts.Parameters), "$id") {
		t.Errorf("$id not stripped: %s", ts.Parameters)
	}
	if strings.Contains(string(ts.Parameters), "$defs") {
		t.Errorf("$defs not stripped: %s", ts.Parameters)
	}
}

func TestNewToolSchemaFromJSON(t *testing.T) {
	ts, err := NewToolSchemaFromJSON("bash", "Run a command", `{"type":"object","properties":{"cmd":{"type":"string"}}}`)
	if err != nil {
		t.Fatalf("NewToolSchemaFromJSON: %v", err)
	}
	if ts.Name != "bash" {
		t.Errorf("Name = %q", ts.Name)
	}
}

func TestNewToolSchemaFromJSON_RejectsBadSchema(t *testing.T) {
	_, err := NewToolSchemaFromJSON("x", "y", `{"type":"string"}`)
	if err == nil {
		t.Errorf("expected validation error")
	}
}
