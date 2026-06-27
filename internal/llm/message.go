// message.go — provider-agnostic message model.
//
// Every provider (Anthropic Messages, OpenAI Chat Completions) speaks its own
// wire format, but internally tau uses the types in this file. Provider
// marshal/unmarshal layers in internal/llm/provider/{anthropic,openai}/
// translate between this canonical shape and the wire payload.
//
// The canonical JSON shape (used for state persistence) is a tagged union on
// ContentBlock:
//
//	{ "type": "text",      "text": "..." }
//	{ "type": "image",     "data": "<base64>", "mimeType": "image/png" }
//	{ "type": "toolUse",   "id": "...", "name": "...", "input": { ... } }
//	{ "type": "toolResult","toolUseId": "...", "content": [ ... ], "isError": false }
//	{ "type": "thinking",  "thinking": "...", "signature": "...", "redacted": false }
//
// A Message carries Role plus Content; role "tool" messages also carry the
// originating ToolCallID so providers that need a separate tool message
// (OpenAI) can populate tool_call_id on the wire.

package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Role identifies the speaker of a Message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// StopReason is why the model stopped generating this turn.
type StopReason string

const (
	StopReasonEndTurn StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

// Usage reports per-turn token accounting. The agent loop reads Usage to drive
// compaction decisions. Cost is filled in by the provider when the model
// definition in models.json carries nonzero Cost rates.
type Usage struct {
	Input       int  `json:"input"`
	Output      int  `json:"output"`
	CacheRead   int  `json:"cacheRead"`
	CacheWrite  int  `json:"cacheWrite"`
	TotalTokens int  `json:"totalTokens"`
	Cost        Cost `json:"cost"`
}

// Cost is the dollar cost of a single turn. All fields are in USD.
type Cost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
}

// ContentBlock is a sealed interface: only the five concrete types in this
// file implement it (the unexported isContentBlock method seals it). Variants:
//
//   - TextContent     plain text
//   - ThinkingContent extended-thinking output (reasoning models)
//   - ImageContent    base64 image
//   - ToolUse         model requests a tool call
//   - ToolResult      result of a previous ToolUse
//
// The interface value is the persistence shape — it round-trips through JSON
// without loss via the type discriminator.
type ContentBlock interface {
	isContentBlock()
}

// TextContent is plain text.
type TextContent struct {
	Text string `json:"text"`
}

// isContentBlock seals TextContent into the ContentBlock union.
func (TextContent) isContentBlock() {}

// ThinkingContent is the output of an extended-thinking ("reasoning") block.
// Signature is an opaque provider-specific token used for multi-turn
// continuity (Anthropic signature, OpenAI reasoning item id). When Redacted
// is true the model declined to share the reasoning text; the opaque payload
// is in Signature so subsequent turns still see it.
type ThinkingContent struct {
	Thinking  string `json:"thinking"`
	Signature string `json:"signature,omitempty"`
	Redacted  bool   `json:"redacted,omitempty"`
}

// isContentBlock seals ThinkingContent into the ContentBlock union.
func (ThinkingContent) isContentBlock() {}

// ImageContent is a base64-encoded image. Data MUST NOT include a "data:"
// URL prefix — the provider marshaler adds the prefix on the wire if needed.
type ImageContent struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// isContentBlock seals ImageContent into the ContentBlock union.
func (ImageContent) isContentBlock() {}

// ToolUse is the model's request to invoke a tool. Input is the raw JSON
// arguments payload from the model; callers validate it against the tool's
// parameter schema before execution.
type ToolUse struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// isContentBlock seals ToolUse into the ContentBlock union.
func (ToolUse) isContentBlock() {}

// ToolResult is the result of executing a ToolUse. Content typically holds a
// single TextContent; the slice shape matches Anthropic's tool_result.content
// which can carry multiple blocks (text + image). When IsError is true the
// provider renders the result as a tool-call failure to the model.
type ToolResult struct {
	ToolUseID string         `json:"toolUseId"`
	Content   []ContentBlock `json:"content"`
	IsError   bool           `json:"isError,omitempty"`
}

// isContentBlock seals ToolResult into the ContentBlock union.
func (ToolResult) isContentBlock() {}

// contentBlockType is the JSON tag discriminator value.
const (
	contentBlockTypeText       = "text"
	contentBlockTypeImage      = "image"
	contentBlockTypeToolUse    = "toolUse"
	contentBlockTypeToolResult = "toolResult"
	contentBlockTypeThinking   = "thinking"
)

// MarshalJSON emits TextContent as {"type":"text","text":"..."}.
func (t TextContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{contentBlockTypeText, t.Text})
}

// UnmarshalJSON accepts the same shape produced by MarshalJSON.
func (t *TextContent) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Type != contentBlockTypeText {
		return fmt.Errorf("llm: TextContent type tag = %q, want %q", raw.Type, contentBlockTypeText)
	}
	t.Text = raw.Text
	return nil
}

// MarshalJSON emits ImageContent as {"type":"image","data":"...","mimeType":"..."}.
func (i ImageContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type     string `json:"type"`
		Data     string `json:"data"`
		MimeType string `json:"mimeType"`
	}{contentBlockTypeImage, i.Data, i.MimeType})
}

// UnmarshalJSON accepts the same shape produced by MarshalJSON.
func (i *ImageContent) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type     string `json:"type"`
		Data     string `json:"data"`
		MimeType string `json:"mimeType"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Type != contentBlockTypeImage {
		return fmt.Errorf("llm: ImageContent type tag = %q, want %q", raw.Type, contentBlockTypeImage)
	}
	i.Data = raw.Data
	i.MimeType = raw.MimeType
	return nil
}

// MarshalJSON emits ToolUse as {"type":"toolUse",...}.
func (u ToolUse) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type  string          `json:"type"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	}{contentBlockTypeToolUse, u.ID, u.Name, u.Input})
}

// UnmarshalJSON accepts the same shape produced by MarshalJSON.
func (u *ToolUse) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type  string          `json:"type"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Type != contentBlockTypeToolUse {
		return fmt.Errorf("llm: ToolUse type tag = %q, want %q", raw.Type, contentBlockTypeToolUse)
	}
	u.ID = raw.ID
	u.Name = raw.Name
	u.Input = raw.Input
	return nil
}

// MarshalJSON emits ThinkingContent as {"type":"thinking",...}.
func (t ThinkingContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type      string `json:"type"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature,omitempty"`
		Redacted  bool   `json:"redacted,omitempty"`
	}{contentBlockTypeThinking, t.Thinking, t.Signature, t.Redacted})
}

// UnmarshalJSON accepts the same shape produced by MarshalJSON.
func (t *ThinkingContent) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type      string `json:"type"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature,omitempty"`
		Redacted  bool   `json:"redacted,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Type != contentBlockTypeThinking {
		return fmt.Errorf("llm: ThinkingContent type tag = %q, want %q", raw.Type, contentBlockTypeThinking)
	}
	t.Thinking = raw.Thinking
	t.Signature = raw.Signature
	t.Redacted = raw.Redacted
	return nil
}

// MarshalJSON emits ToolResult as {"type":"toolResult",...} with nested content
// blocks recursively tagged.
func (r ToolResult) MarshalJSON() ([]byte, error) {
	type plain struct {
		Type      string         `json:"type"`
		ToolUseID string         `json:"toolUseId"`
		Content   []ContentBlock `json:"content"`
		IsError   bool           `json:"isError,omitempty"`
	}
	return json.Marshal(plain{contentBlockTypeToolResult, r.ToolUseID, r.Content, r.IsError})
}

// UnmarshalJSON accepts the same shape produced by MarshalJSON.
func (r *ToolResult) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type      string            `json:"type"`
		ToolUseID string            `json:"toolUseId"`
		Content   []json.RawMessage `json:"content"`
		IsError   bool              `json:"isError,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Type != contentBlockTypeToolResult {
		return fmt.Errorf("llm: ToolResult type tag = %q, want %q", raw.Type, contentBlockTypeToolResult)
	}
	r.ToolUseID = raw.ToolUseID
	r.IsError = raw.IsError
	r.Content = make([]ContentBlock, 0, len(raw.Content))
	for _, raw2 := range raw.Content {
		block, err := UnmarshalContentBlock(raw2)
		if err != nil {
			return err
		}
		r.Content = append(r.Content, block)
	}
	return nil
}

// ErrUnknownContentBlock is returned by UnmarshalContentBlock when the JSON
// "type" tag is missing or unknown.
var ErrUnknownContentBlock = errors.New("unknown content block type")

// UnmarshalContentBlock decodes one JSON object into the appropriate
// ContentBlock variant based on the "type" discriminator. It is the single
// entry point for parsing the tagged union from any context (top-level
// Message.Content, nested ToolResult.Content, fixtures, etc.).
func UnmarshalContentBlock(data []byte) (ContentBlock, error) {
	if len(data) == 0 {
		return nil, ErrUnknownContentBlock
	}
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Type {
	case contentBlockTypeText:
		var t TextContent
		if err := json.Unmarshal(data, &t); err != nil {
			return nil, err
		}
		return t, nil
	case contentBlockTypeImage:
		var i ImageContent
		if err := json.Unmarshal(data, &i); err != nil {
			return nil, err
		}
		return i, nil
	case contentBlockTypeToolUse:
		var u ToolUse
		if err := json.Unmarshal(data, &u); err != nil {
			return nil, err
		}
		return u, nil
	case contentBlockTypeToolResult:
		var r ToolResult
		if err := json.Unmarshal(data, &r); err != nil {
			return nil, err
		}
		return r, nil
	case contentBlockTypeThinking:
		var t ThinkingContent
		if err := json.Unmarshal(data, &t); err != nil {
			return nil, err
		}
		return t, nil
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnknownContentBlock, probe.Type)
	}
}

// Message is a single conversational turn.
//
// Field usage by Role:
//
//   - RoleSystem:    Content = [TextContent{...}] (one block, no more)
//   - RoleUser:      Content = [TextContent, ImageContent, ToolResult, ...]
//   - RoleAssistant: Content = [TextContent, ThinkingContent, ToolUse, ...]
//     Model/ProviderID/ResponseID/Usage/StopReason set
//   - RoleTool:      Content = [ToolResult{...}] (or [TextContent{...}])
//     ToolCallID and ToolName set
//
// The Marshalers in internal/llm/provider/{anthropic,openai}/ read these
// fields and translate to the provider's wire format. OpenAI uses RoleTool +
// ToolCallID directly; Anthropic inlines ToolResult as a user-role content
// block.
type Message struct {
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content"`
	ToolCallID string         `json:"toolCallId,omitempty"`
	ToolName   string         `json:"toolName,omitempty"`
	Model      string         `json:"model,omitempty"`
	ProviderID string         `json:"providerId,omitempty"`
	ResponseID string         `json:"responseId,omitempty"`
	Usage      *Usage         `json:"usage,omitempty"`
	StopReason StopReason     `json:"stopReason,omitempty"`
	Timestamp  time.Time      `json:"timestamp"`
}

// MarshalJSON encodes Message, dispatching each ContentBlock to its concrete
// MarshalJSON so the type discriminator is emitted.
func (m Message) MarshalJSON() ([]byte, error) {
	type plain Message
	return json.Marshal(plain(m))
}

// UnmarshalJSON decodes Message, dispatching each Content block via
// UnmarshalContentBlock.
func (m *Message) UnmarshalJSON(data []byte) error {
	type plain Message
	var raw struct {
		*plain
		Content []json.RawMessage `json:"content"`
	}
	raw.plain = (*plain)(m)
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Content = make([]ContentBlock, 0, len(raw.Content))
	for _, raw2 := range raw.Content {
		block, err := UnmarshalContentBlock(raw2)
		if err != nil {
			return err
		}
		m.Content = append(m.Content, block)
	}
	return nil
}
