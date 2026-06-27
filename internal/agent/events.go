// events.go — bounded, non-blocking event bus.
//
// The bus is the single publish point for every observable state
// transition in an agent session. Each subscriber gets its own buffered
// channel; Publish is non-blocking — a full buffer drops the oldest
// event to make room (with a counter exposed via Drops()) so a slow
// subscriber cannot stall the agent loop.
//
// The bus is goroutine-safe. Subscribe may be called concurrently with
// Publish. Shutdown is idempotent: after the first call, it emits the
// SessionShutdownEvent and closes every subscriber channel; subsequent
// calls are no-ops.

package agent

import (
	"sync"
	"sync/atomic"
	"time"
)

// DefaultEventBuffer is the per-subscriber channel buffer size when no
// size is supplied to NewEventBus. Generous enough to absorb a single
// turn's worth of streaming deltas without blocking the model stream.
const DefaultEventBuffer = 64

// EventBus is the bounded pub/sub for agent events. Construct via
// NewEventBus. A nil *EventBus is usable: Subscribe returns a closed
// channel and Publish is a no-op. This makes optional telemetry easy.
type EventBus struct {
	buffer int

	mu          sync.RWMutex
	closed      bool
	subscribers []*subscriber

	// drops counts events dropped due to a full subscriber buffer.
	// Atomic so the publish hot-path doesn't take the mutex.
	drops atomic.Int64
}

// subscriber is one registration. A subscriber receives only the
// topics it filtered for at registration time.
type subscriber struct {
	ch     chan Event
	topics map[Topic]bool
}

// NewEventBus constructs an EventBus with the given per-subscriber
// buffer size. If buffer <= 0, DefaultEventBuffer is used.
func NewEventBus(buffer int) *EventBus {
	if buffer <= 0 {
		buffer = DefaultEventBuffer
	}
	return &EventBus{buffer: buffer}
}

// Subscribe returns a channel that receives every event whose Topic is
// in topics. If no topics are given, the channel receives ALL events.
// The channel has the bus's configured buffer size.
//
// The returned channel is closed when the bus is shut down. Callers
// should range over it and exit when the range terminates.
//
// Subscribe may be called concurrently with Publish and Shutdown.
// Subscribing after Shutdown returns an already-closed channel.
func (b *EventBus) Subscribe(topics ...Topic) <-chan Event {
	if b == nil {
		// Nil-bus: return a closed channel so callers can range and
		// exit immediately. Avoids nil-channel hangs.
		ch := make(chan Event)
		close(ch)
		return ch
	}
	ch := make(chan Event, b.buffer)
	sub := &subscriber{ch: ch, topics: make(map[Topic]bool, len(topics))}
	for _, t := range topics {
		sub.topics[t] = true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		// Bus already shut down; close immediately so the caller's
		// range exits without hanging.
		close(ch)
		return ch
	}
	b.subscribers = append(b.subscribers, sub)
	return ch
}

// Publish emits an event to every matching subscriber. Non-blocking:
// if a subscriber's buffer is full, the OLDEST event in that buffer is
// dropped to make room (ring-buffer style) and the bus's drop counter
// is incremented.
//
// Publishing on a shut-down bus, or to a nil *EventBus, is a no-op.
func (b *EventBus) Publish(evt Event) {
	if b == nil || evt == nil {
		return
	}
	b.mu.RLock()
	if b.closed {
		b.mu.RUnlock()
		return
	}
	subs := b.subscribers
	b.mu.RUnlock()
	for _, s := range subs {
		if len(s.topics) > 0 && !s.topics[evt.Topic()] {
			continue
		}
		publishToSubscriber(s, evt, &b.drops)
	}
}

// publishToSubscriber delivers evt to s, dropping the oldest queued
// event if the buffer is full. Kept as a free function so the inlined
// hot path doesn't carry the bus mutex.
func publishToSubscriber(s *subscriber, evt Event, drops *atomic.Int64) {
	for {
		select {
		case s.ch <- evt:
			return
		default:
		}
		// Buffer full. Pop the oldest to make room. The pop is racy
		// if multiple publishers race the same subscriber, but the
		// drops counter is monotonic and the subscriber's ordering
		// invariant is "newest buffer-worth wins" which is what we
		// want for slow-consumer scenarios.
		select {
		case <-s.ch:
			drops.Add(1)
		default:
			// Another concurrent publisher popped between our two
			// selects; retry the send.
		}
	}
}

// Shutdown emits a SessionShutdownEvent with the given reason, then
// closes every subscriber's channel. Idempotent: subsequent calls are
// no-ops. The reason string is opaque to the bus; callers use it to
// tell subscribers why the session ended ("user", "error: ...",
// "context_canceled").
func (b *EventBus) Shutdown(reason string) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return
	}
	b.closed = true
	subs := b.subscribers
	b.subscribers = nil
	// Emit the shutdown event on each subscriber's channel directly so
	// it is guaranteed to land even though Publish would early-out on
	// the now-closed flag. We bypass the buffer-drop path: shutdown
	// must be visible to consumers.
	for _, s := range subs {
		if len(s.topics) == 0 || s.topics[TopicSessionShutdown] {
			select {
			case s.ch <- SessionShutdownEvent{When: time.Now(), Reason: reason}:
			default:
				// Even on a full buffer, force delivery by dropping
				// one queued event.
				select {
				case <-s.ch:
				default:
				}
				select {
				case s.ch <- SessionShutdownEvent{When: time.Now(), Reason: reason}:
				default:
				}
			}
		}
		close(s.ch)
	}
}

// Drops returns the total number of events dropped due to full
// subscriber buffers since the bus was created. Useful for telemetry
// and tests; not for production control flow.
func (b *EventBus) Drops() int64 {
	if b == nil {
		return 0
	}
	return b.drops.Load()
}

// IsClosed reports whether the bus has been shut down. A closed bus
// accepts no further publishes; Subscribe returns a closed channel.
func (b *EventBus) IsClosed() bool {
	if b == nil {
		return true
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.closed
}
