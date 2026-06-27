// events_types.go — typed events for the agent event bus.
//
// Every observable state transition in a session emits one Event on the
// bus. Subscribers (TUI, telemetry, sinks) receive them in emission
// order. The shapes here are stable for the v1 wire contract.
//
// Naming convention: event topics are snake_case identifiers matching
// the agent-loop spec vocabulary. Struct field names follow Go style.
// Each event carries its own When timestamp so consumers can reorder
// or bucket without consulting wall-clock.

package agent

import (
	"time"

	"github.com/coevin/tau/internal/llm"
	"github.com/coevin/tau/internal/tools"
)

// Topic is the routing key for an Event. Subscribers select zero or
// more topics when subscribing; the bus delivers matching events.
type Topic string

const (
	// TopicSessionStart is emitted once per session, immediately before
	// the first turn. Subscribers use it to bootstrap UI state.
	TopicSessionStart Topic = "session_start"
	// TopicTurnStart opens a turn: the agent has accepted user input
	// and is about to dispatch to the model. Pair with TopicTurnEnd.
	TopicTurnStart Topic = "turn_start"
	// TopicMessageStart opens an assistant message: the model began
	// streaming. Zero or more TopicMessageUpdate follow.
	TopicMessageStart Topic = "message_start"
	// TopicMessageUpdate carries an incremental delta from the model
	// stream. Rendered live by the TUI.
	TopicMessageUpdate Topic = "message_update"
	// TopicToolCall is emitted when the agent invokes a tool. Always
	// paired with TopicToolResult for the same call ID.
	TopicToolCall Topic = "tool_call"
	// TopicToolResult carries the tool's output. Emitted in the same
	// order as the corresponding TopicToolCall when running serially;
	// completion order when concurrent.
	TopicToolResult Topic = "tool_result"
	// TopicMessageEnd closes the assistant message: the model returned
	// Final. Emitted before tool execution.
	TopicMessageEnd Topic = "message_end"
	// TopicTurnEnd closes a turn: the assistant's response (and any
	// tool round-trips) are persisted to the state tree.
	TopicTurnEnd Topic = "turn_end"
	// TopicSessionShutdown is emitted exactly once when the session is
	// shutting down. The bus then closes all subscriber channels,
	// allowing range loops to exit.
	TopicSessionShutdown Topic = "session_shutdown"
)

// allTopics enumerates every topic. Used by tests to verify the set
// stays in sync with the const block above.
var allTopics = []Topic{
	TopicSessionStart,
	TopicTurnStart,
	TopicMessageStart,
	TopicMessageUpdate,
	TopicToolCall,
	TopicToolResult,
	TopicMessageEnd,
	TopicTurnEnd,
	TopicSessionShutdown,
}

// Event is the sealed interface implemented by every typed event.
// Concrete events are value types; subscribers type-switch (or type-
// assert) on the underlying struct.
type Event interface {
	Topic() Topic
}

// SessionStartEvent opens a session. Emitted once before the first
// TopicTurnStart. Carries enough identity for log sinks to bootstrap.
type SessionStartEvent struct {
	When      time.Time
	SessionID string
	Model     string
	Cwd       string
}

// Topic implements Event.
func (e SessionStartEvent) Topic() Topic { return TopicSessionStart }

// TurnStartEvent opens a turn. The agent has accepted user input and is
// about to dispatch to the model.
type TurnStartEvent struct {
	When   time.Time
	Turn   int // 1-based
	UserID string
}

// Topic implements Event.
func (e TurnStartEvent) Topic() Topic { return TopicTurnStart }

// MessageStartEvent opens an assistant message: the model began
// streaming a response. Pair with TopicMessageEnd.
type MessageStartEvent struct {
	When time.Time
}

// Topic implements Event.
func (e MessageStartEvent) Topic() Topic { return TopicMessageStart }

// MessageUpdateEvent carries an incremental delta from the model stream
// to subscribers. Rendered live by the TUI. The Delta field is the
// raw llm.Delta; subscribers type-switch to render.
type MessageUpdateEvent struct {
	When  time.Time
	Delta llm.Delta
}

// Topic implements Event.
func (e MessageUpdateEvent) Topic() Topic { return TopicMessageUpdate }

// ToolCallEvent is emitted when the agent invokes a tool. Always paired
// with a ToolResultEvent for the same ToolUseID.
type ToolCallEvent struct {
	When time.Time
	Call tools.ToolCall
}

// Topic implements Event.
func (e ToolCallEvent) Topic() Topic { return TopicToolCall }

// ToolResultEvent carries the tool's output. Emitted after the
// ToolCallEvent for the same ToolUseID.
type ToolResultEvent struct {
	When   time.Time
	Result tools.ToolResult
	ToolID string // matches the Call.ID from ToolCallEvent
}

// Topic implements Event.
func (e ToolResultEvent) Topic() Topic { return TopicToolResult }

// MessageEndEvent closes the assistant message: the model returned
// Final AND any tool execution triggered by this message has completed.
// Per the agent-loop spec scenario "Full-turn event sequence", the order
// within a turn is: message_start, message_update*, tool_call, tool_result,
// message_end. Tools fire before message_end so subscribers can render a
// complete assistant bubble including tool outcomes before closing it.
type MessageEndEvent struct {
	When       time.Time
	StopReason llm.StopReason
}

// Topic implements Event.
func (e MessageEndEvent) Topic() Topic { return TopicMessageEnd }

// TurnEndEvent closes a turn: the assistant's response (and any tool
// round-trips) are persisted to the state tree.
type TurnEndEvent struct {
	When     time.Time
	Turn     int
	Finished bool // false if the turn was aborted mid-flight
}

// Topic implements Event.
func (e TurnEndEvent) Topic() Topic { return TopicTurnEnd }

// SessionShutdownEvent is emitted exactly once when the session is
// shutting down. After this event, the bus closes all subscriber
// channels.
type SessionShutdownEvent struct {
	When   time.Time
	Reason string // "user", "error:<text>", "context_canceled"
}

// Topic implements Event.
func (e SessionShutdownEvent) Topic() Topic { return TopicSessionShutdown }

// AllTopics returns a copy of the canonical topic list. Tests use it to
// verify coverage. The slice is sorted by declaration order, which
// matches the canonical emit sequence for a happy single-tool turn.
func AllTopics() []Topic {
	out := make([]Topic, len(allTopics))
	copy(out, allTopics)
	return out
}
