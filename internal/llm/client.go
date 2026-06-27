// client.go — provider-agnostic LLM client interface.
//
// The single method `Stream(ctx, req) (<-chan Delta, error)` is the entire
// provider abstraction. Provider implementations in
// internal/llm/provider/{anthropic,openai}/ satisfy this interface; nothing
// in the agent loop imports provider packages directly.
//
// Streaming contract (see llm-client spec):
//
//   - The error return is non-nil only if the request could not be started
//     (e.g., auth failure, retry budget exhausted, malformed Request).
//   - On success the channel emits zero or more TextDelta/ThinkingDelta/
//     ToolCallDelta, then exactly one UsageDelta, then exactly one Final,
//     then the channel is closed.
//   - If ctx is cancelled mid-stream, the provider closes the underlying
//     HTTP connection, drains and closes the channel, and returns
//     ctx.Err() (returned via Final.Err when applicable).
//
// The provider MUST emit deltas in order: text/thinking/tool-call deltas
// first (interleaved by content-block index), then UsageDelta, then Final.

package llm

import (
	"context"
	"encoding/json"
	"errors"
)

// Transport selects the streaming transport for a single request.
type Transport string

const (
	TransportAuto      Transport = "auto"
	TransportSSE       Transport = "sse"
	TransportWebSocket Transport = "websocket"
)

// Request is the provider-agnostic request shape.
//
// Fields are pointers where zero needs to be distinguishable from unset
// (MaxTokens, Temperature, ThinkingBudget). The System slice is rendered as
// the provider expects (Anthropic: top-level system blocks; OpenAI: a
// role="system" message prepended to the conversation).
type Request struct {
	Model          string            // provider-specific model id
	System         []ContentBlock    // system prompt blocks (typically [TextContent])
	Messages       []Message         // conversation turns
	Tools          []ToolSchema      // tool schemas offered this turn
	ThinkingBudget *int              // 0 disables thinking; nil = provider default
	MaxTokens      *int              // hard output cap; nil = provider default
	Temperature    *float64          // 0..2; nil = provider default
	Stop           []string          // stop sequences
	Transport      Transport         // SSE / WebSocket / auto
	Headers        map[string]string // extra headers merged after auth
}

// ToolSchema describes one tool to the model. It mirrors the fields a tool
// implementation publishes: Name, Description, and a JSON Schema for
// parameters. The schema is encoded as raw JSON so any schema source
// (Go struct tags via invopop/jsonschema, hand-written maps, dynamic plugin
// schemas) plugs in without conversion.
type ToolSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// LLMClient is the provider abstraction. Implementations MUST be safe for
// concurrent use; the agent loop may issue parallel Stream calls in tests
// and via plugins.
//
//nolint:revive // name is intentional: "llm.LLMClient" is the canonical handle to a model.
type LLMClient interface {
	// Stream issues a streaming completion request. See the package comment
	// for the channel protocol.
	Stream(ctx context.Context, req Request) (<-chan Delta, error)
}

// Delta is a sealed interface: only the five concrete types in this file
// implement it (the unexported isDelta method seals it). Variants:
//
//   - TextDelta       incremental assistant text
//   - ThinkingDelta   incremental thinking text
//   - ToolCallDelta   incremental tool-call input JSON for a given tool-use id
//   - UsageDelta      per-turn token accounting (emitted once)
//   - Final           terminal marker carrying StopReason and optional error
//
// Concatenation: TextDelta.Text fragments concatenate in order to form the
// assistant text. ToolCallDelta.PartialInput fragments with the same ID
// concatenate to form the full tool-input JSON (which is then parsed).
type Delta interface {
	isDelta()
}

// TextDelta carries an incremental fragment of assistant text output.
type TextDelta struct {
	// ContentIndex is the position of the text block within the assistant
	// message. The agent loop uses this to write into the correct slot when
	// the model interleaves text and tool_use blocks.
	ContentIndex int
	Text         string
}

// isDelta seals TextDelta.
func (TextDelta) isDelta() {}

// ThinkingDelta carries an incremental fragment of thinking text.
type ThinkingDelta struct {
	ContentIndex int
	Text         string
	// Signature is non-empty only when the provider includes one with this
	// fragment (Anthropic signature_delta events).
	Signature string
}

// isDelta seals ThinkingDelta.
func (ThinkingDelta) isDelta() {}

// ToolCallDelta carries an incremental fragment of a tool-call's input JSON.
// The provider streams the JSON arguments in chunks; the consumer
// concatenates PartialInput for matching ID to recover the full payload.
type ToolCallDelta struct {
	// ContentIndex is the position of the tool_use block within the
	// assistant message.
	ContentIndex int
	// ID is the tool-use id assigned by the provider. The first delta for a
	// given content index carries the id; subsequent deltas for the same
	// content index repeat it.
	ID string
	// Name is the tool name. Only set on the first delta for a content
	// index; consumers should latch it on first sight.
	Name string
	// PartialInput is a fragment of the input JSON. Concatenate in order.
	PartialInput string
}

// isDelta seals ToolCallDelta.
func (ToolCallDelta) isDelta() {}

// UsageDelta carries per-turn token accounting. Emitted exactly once per
// stream, after the last content delta and before Final. Some providers
// (OpenAI) only emit usage when stream_options.include_usage is set; tau's
// OpenAI provider always requests it.
type UsageDelta struct {
	// InputTokens counts the prompt tokens.
	InputTokens int
	// OutputTokens counts the generated tokens.
	OutputTokens int
	// CacheReadTokens is the count of input tokens served from a prompt
	// cache (Anthropic-only; zero for providers without prompt caching).
	CacheReadTokens int
	// CacheWriteTokens is the count of input tokens written to the prompt
	// cache (Anthropic-only).
	CacheWriteTokens int
}

// isDelta seals UsageDelta.
func (UsageDelta) isDelta() {}

// Final is the terminal marker. The provider MUST emit exactly one Final
// after all other deltas, then close the channel.
//
//   - StopReason is set to StopReasonError if Err is non-nil; the converse
//     is not guaranteed (StopReasonError may have a nil Err when the
//     provider could not classify the failure).
//   - ResponseID carries the provider's response/message id when available
//     (e.g., Anthropic `msg_…`, OpenAI `chatcmpl_…`).
//   - ResponseModel is the concrete model that served the request when the
//     provider returns a different id (e.g., OpenRouter auto-routing).
type Final struct {
	StopReason    StopReason
	ResponseID    string
	ResponseModel string
	Err           error
}

// isDelta seals Final.
func (Final) isDelta() {}

// ErrFinalMissing is returned by stream consumers when a channel was closed
// without a Final marker. It indicates a provider bug.
var ErrFinalMissing = errors.New("stream closed without Final delta")

// IsAbort reports whether err is context.Canceled or context.DeadlineExceeded.
// Providers wrap aborts in Final.Err; the agent loop uses this helper to
// distinguish aborts from provider errors.
func IsAbort(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
