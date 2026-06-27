package agent

import (
	"sync"
	"testing"
	"time"
)

// TestEventBus_PublishSubscribe verifies the basic shape: a subscriber
// gets the events it subscribed to, in emission order.
func TestEventBus_PublishSubscribe(t *testing.T) {
	bus := NewEventBus(8)
	ch := bus.Subscribe(TopicTurnStart, TopicTurnEnd)

	bus.Publish(TurnStartEvent{When: time.Now(), Turn: 1})
	bus.Publish(TurnEndEvent{When: time.Now(), Turn: 1, Finished: true})

	got := drainN(ch, 2, time.Second)
	if len(got) != 2 {
		t.Fatalf("expected 2 events, got %d", len(got))
	}
	if _, ok := got[0].(TurnStartEvent); !ok {
		t.Errorf("event 0 = %T, want TurnStartEvent", got[0])
	}
	if _, ok := got[1].(TurnEndEvent); !ok {
		t.Errorf("event 1 = %T, want TurnEndEvent", got[1])
	}
}

// TestEventBus_SubscribeAll covers the "no topics = all events" rule.
func TestEventBus_SubscribeAll(t *testing.T) {
	bus := NewEventBus(8)
	ch := bus.Subscribe()

	bus.Publish(TurnStartEvent{When: time.Now(), Turn: 1})
	bus.Publish(MessageStartEvent{When: time.Now()})
	bus.Publish(MessageEndEvent{When: time.Now(), StopReason: "stop"})

	got := drainN(ch, 3, time.Second)
	if len(got) != 3 {
		t.Fatalf("expected 3 events, got %d", len(got))
	}
}

// TestEventBus_UnselectedTopicNotDelivered verifies topic filtering.
func TestEventBus_UnselectedTopicNotDelivered(t *testing.T) {
	bus := NewEventBus(8)
	ch := bus.Subscribe(TopicTurnEnd)

	bus.Publish(TurnStartEvent{When: time.Now(), Turn: 1})
	bus.Publish(TurnEndEvent{When: time.Now(), Turn: 1, Finished: true})

	got := drainN(ch, 1, time.Second)
	if len(got) != 1 {
		t.Fatalf("expected 1 event (filtered), got %d", len(got))
	}
	if _, ok := got[0].(TurnEndEvent); !ok {
		t.Errorf("event = %T, want TurnEndEvent", got[0])
	}
}

// TestEventBus_ShutdownClosesChannels verifies that Shutdown emits the
// SessionShutdownEvent and then closes every subscriber's channel.
func TestEventBus_ShutdownClosesChannels(t *testing.T) {
	bus := NewEventBus(8)
	ch := bus.Subscribe()

	bus.Shutdown("user")

	// First event must be SessionShutdownEvent; then the channel must close.
	select {
	case evt, ok := <-ch:
		if !ok {
			t.Fatalf("channel closed before SessionShutdownEvent was delivered")
		}
		if _, isShutdown := evt.(SessionShutdownEvent); !isShutdown {
			t.Fatalf("first event = %T, want SessionShutdownEvent", evt)
		}
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for SessionShutdownEvent")
	}
	// Channel should now be closed.
	if _, ok := <-ch; ok {
		t.Errorf("channel not closed after SessionShutdownEvent")
	}
}

// TestEventBus_ShutdownIdempotent verifies that calling Shutdown more
// than once doesn't panic, re-emit, or double-close channels.
func TestEventBus_ShutdownIdempotent(t *testing.T) {
	bus := NewEventBus(8)
	ch := bus.Subscribe()

	bus.Shutdown("user")
	bus.Shutdown("again") // must not panic
	bus.Shutdown("again2")

	count := 0
	for range ch { // drains the shutdown event then exits on close
		count++
	}
	if count != 1 {
		t.Errorf("expected exactly 1 shutdown event, got %d", count)
	}
}

// TestEventBus_SubscribeAfterShutdown verifies that subscribing after
// Shutdown returns an already-closed channel (no hang).
func TestEventBus_SubscribeAfterShutdown(t *testing.T) {
	bus := NewEventBus(8)
	bus.Shutdown("user")

	ch := bus.Subscribe()
	if _, ok := <-ch; ok {
		t.Errorf("channel from post-shutdown Subscribe should be closed immediately")
	}
}

// TestEventBus_PublishAfterShutdownNoOp verifies that publishes after
// shutdown do not deliver and do not panic.
func TestEventBus_PublishAfterShutdownNoOp(t *testing.T) {
	bus := NewEventBus(8)
	ch := bus.Subscribe()
	bus.Shutdown("user")

	// Drain shutdown + close.
	for range ch {
		_ = ch // revive:empty-block — intentional drain loop
	}

	bus.Publish(TurnStartEvent{When: time.Now(), Turn: 1}) // must not panic, not deliver
	if bus.Drops() != 0 {
		t.Errorf("Drops should still be 0 after shutdown; got %d", bus.Drops())
	}
}

// TestEventBus_SlowSubscriberDoesNotBlock verifies the non-blocking
// contract: a subscriber whose buffer fills does not stall the
// publisher. Drops are counted.
func TestEventBus_SlowSubscriberDoesNotBlock(t *testing.T) {
	bus := NewEventBus(2)
	stalled := bus.Subscribe()
	// Never drain "stalled" — fill its buffer (2) plus one extra; the
	// third publish must drop and proceed, not block.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 10; i++ {
			bus.Publish(TurnStartEvent{When: time.Now(), Turn: i})
		}
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("Publish blocked on slow subscriber")
	}
	if bus.Drops() == 0 {
		t.Errorf("expected >0 drops on slow subscriber, got 0")
	}
	// Shutdown so the for-range exits. Without this the leaked channel
	// would never close and the test would hang.
	bus.Shutdown("test")
	for range stalled {
		_ = stalled // revive:empty-block — intentional drain loop
	}
}

// TestEventBus_ConcurrentPublishSubscribe verifies goroutine safety:
// multiple publishers and subscribers running concurrently do not race
// (go test -race catches this).
func TestEventBus_ConcurrentPublishSubscribe(t *testing.T) {
	bus := NewEventBus(16)
	var wg sync.WaitGroup
	const publishers = 4
	const eventsPerPub = 100
	for i := 0; i < publishers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < eventsPerPub; j++ {
				bus.Publish(TurnStartEvent{When: time.Now(), Turn: j})
			}
		}()
	}
	// Two subscribers: one for turn_start, one for everything.
	all := bus.Subscribe()
	start := bus.Subscribe(TopicTurnStart)
	wg.Wait()
	bus.Shutdown("done")
	// Drain both to release goroutines.
	go func() {
		for range all {
			_ = all // revive:empty-block — intentional drain loop
		}
	}()
	go func() {
		for range start {
			_ = start // revive:empty-block — intentional drain loop
		}
	}()
	// The ranges exit when Shutdown closes the channels; nothing to
	// assert beyond "no race" which the test runner verifies.
}

// TestEventBus_NilSafe verifies that a nil *EventBus is usable: Publish
// is a no-op and Subscribe returns a closed channel.
func TestEventBus_NilSafe(t *testing.T) {
	var nilBus *EventBus
	nilBus.Publish(TurnStartEvent{}) // must not panic
	ch := nilBus.Subscribe()
	if _, ok := <-ch; ok {
		t.Errorf("nil bus Subscribe should return closed channel")
	}
	nilBus.Shutdown("test") // must not panic
	if !nilBus.IsClosed() {
		t.Errorf("nil bus IsClosed should be true")
	}
	if nilBus.Drops() != 0 {
		t.Errorf("nil bus Drops should be 0")
	}
}

// TestAllTopics_CoversSpec verifies AllTopics returns the canonical
// list from the agent-loop spec.
func TestAllTopics_CoversSpec(t *testing.T) {
	got := AllTopics()
	want := []Topic{
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
	if len(got) != len(want) {
		t.Fatalf("AllTopics len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("AllTopics[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// drainN reads up to n events from ch with a per-event timeout. Returns
// the events collected. Used by tests that publish synchronously and
// then need to verify delivery.
func drainN(ch <-chan Event, n int, perEventTimeout time.Duration) []Event {
	out := make([]Event, 0, n)
	for len(out) < n {
		select {
		case evt := <-ch:
			out = append(out, evt)
		case <-time.After(perEventTimeout):
			return out
		}
	}
	return out
}
