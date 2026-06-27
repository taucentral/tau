package state

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/coevin/tau/internal/llm"
)

func TestNewEntry_DerivesKindFromPayload(t *testing.T) {
	cases := []struct {
		payload Payload
		kind    Kind
	}{
		{SessionHeaderPayload{SessionID: "x", Version: 1}, KindSessionHeader},
		{MessagePayload{Role: llm.RoleUser}, KindMessage},
		{ThinkingLevelChangePayload{Level: "high"}, KindThinkingLevelChange},
		{ModelChangePayload{Model: "gpt-4o"}, KindModelChange},
		{CompactionPayload{Summary: "s", FirstKeptEntryID: "abc"}, KindCompaction},
		{BranchSummaryPayload{Summary: "s"}, KindBranchSummary},
		{LabelPayload{Label: "L"}, KindLabel},
		{SessionInfoPayload{Key: "k", Value: "v"}, KindSessionInfo},
		{CustomPayload{Type: "plugin/x", Data: json.RawMessage(`{}`)}, KindCustom},
		{CustomMessagePayload{Role: "user", Content: json.RawMessage(`"x"`)}, KindCustomMessage},
	}
	for i, tc := range cases {
		e := NewEntry("parent", tc.payload)
		if e.Kind != tc.kind {
			t.Errorf("case %d: kind = %q, want %q", i, e.Kind, tc.kind)
		}
		if e.ParentID != "parent" {
			t.Errorf("case %d: parent = %q", i, e.ParentID)
		}
		if e.Timestamp.IsZero() {
			t.Errorf("case %d: timestamp is zero", i)
		}
	}
}

func TestEntry_MarshalUnmarshal_RoundTrip(t *testing.T) {
	ts := time.Date(2025, 6, 15, 12, 0, 0, 0, time.UTC)
	original := Entry{
		ID:        "abc12345",
		ParentID:  "parent00",
		Kind:      KindMessage,
		Timestamp: ts,
		Payload: MessagePayload{
			Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{
				llm.TextContent{Text: "hello"},
				llm.ToolUse{ID: "call_1", Name: "bash", Input: json.RawMessage(`{"cmd":"ls"}`)},
			},
		},
	}
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	s := string(data)
	for _, want := range []string{
		`"id":"abc12345"`,
		`"parentId":"parent00"`,
		`"kind":"Message"`,
		`"role":"assistant"`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("marshaled JSON missing %q\n%s", want, s)
		}
	}

	var decoded Entry
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if decoded.ID != original.ID || decoded.ParentID != original.ParentID {
		t.Errorf("ID/ParentID = %q/%q", decoded.ID, decoded.ParentID)
	}
	if decoded.Kind != original.Kind {
		t.Errorf("Kind = %q", decoded.Kind)
	}
	if !decoded.Timestamp.Equal(original.Timestamp) {
		t.Errorf("Timestamp drift: %v vs %v", decoded.Timestamp, original.Timestamp)
	}
	mp, ok := decoded.Payload.(MessagePayload)
	if !ok {
		t.Fatalf("Payload type = %T, want MessagePayload", decoded.Payload)
	}
	if mp.Role != llm.RoleAssistant {
		t.Errorf("Role = %q", mp.Role)
	}
	if len(mp.Content) != 2 {
		t.Fatalf("Content len = %d", len(mp.Content))
	}
	if tc, ok := mp.Content[0].(llm.TextContent); !ok || tc.Text != "hello" {
		t.Errorf("Content[0] = %+v", mp.Content[0])
	}
	if tu, ok := mp.Content[1].(llm.ToolUse); !ok || tu.Name != "bash" {
		t.Errorf("Content[1] = %+v", mp.Content[1])
	}
}

func TestEntry_AllKindsRoundTrip(t *testing.T) {
	payloads := []Payload{
		SessionHeaderPayload{SessionID: "s", Version: 1, CreatedAt: time.Now()},
		MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "hi"}}},
		ThinkingLevelChangePayload{Level: "high"},
		ModelChangePayload{Model: "gpt-4o"},
		CompactionPayload{Summary: "summary", FirstKeptEntryID: "abc12345"},
		BranchSummaryPayload{Summary: "parent summary"},
		LabelPayload{Label: "after-merge"},
		SessionInfoPayload{Key: "exit", Value: "done"},
		CustomPayload{Type: "plugin", Data: json.RawMessage(`{"x":1}`)},
		CustomMessagePayload{Role: "user", Content: json.RawMessage(`"injected"`)},
	}
	for i, p := range payloads {
		e := Entry{
			ID:        "entry0001",
			Kind:      kindOf(p),
			Timestamp: time.Now().UTC(),
			Payload:   p,
		}
		data, err := json.Marshal(e)
		if err != nil {
			t.Fatalf("case %d: Marshal: %v", i, err)
		}
		var decoded Entry
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("case %d: Unmarshal: %v", i, err)
		}
		if decoded.Kind != e.Kind {
			t.Errorf("case %d (%s): kind drift to %q", i, e.Kind, decoded.Kind)
		}
		if decoded.Payload == nil {
			t.Errorf("case %d (%s): nil payload after decode", i, e.Kind)
			continue
		}
		if got := kindOf(decoded.Payload); got != e.Kind {
			t.Errorf("case %d (%s): payload kind drift to %q", i, e.Kind, got)
		}
	}
}

func TestEntry_UnknownKindFallbackToCustom(t *testing.T) {
	// Forward compatibility: a future-kind entry should decode without error
	// and land as CustomPayload.
	raw := `{
		"id":"future01",
		"parentId":"parent00",
		"kind":"SomeFutureKind",
		"timestamp":"2025-06-15T12:00:00Z",
		"payload":{"foo":"bar"}
	}`
	var e Entry
	if err := json.Unmarshal([]byte(raw), &e); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if e.Kind != "SomeFutureKind" {
		t.Errorf("Kind = %q", e.Kind)
	}
	if _, ok := e.Payload.(CustomPayload); !ok {
		t.Errorf("Payload type = %T, want CustomPayload", e.Payload)
	}
}

func TestMessagePayload_AsMessage(t *testing.T) {
	p := MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "x"}}}
	m := p.AsMessage()
	if m.Role != llm.RoleUser {
		t.Errorf("Role = %q", m.Role)
	}
	if len(m.Content) != 1 {
		t.Errorf("Content len = %d", len(m.Content))
	}
}
