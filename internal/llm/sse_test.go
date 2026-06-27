package llm

import (
	"errors"
	"strings"
	"testing"
)

func TestReadSSE_SingleEvent(t *testing.T) {
	input := "event: ping\ndata: {\"hi\":1}\n\n"
	var got []SSEEvent
	err := ReadSSE(strings.NewReader(input), func(ev SSEEvent) error {
		got = append(got, ev)
		return nil
	})
	if err != nil {
		t.Fatalf("ReadSSE: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("events = %d, want 1", len(got))
	}
	if got[0].Type != "ping" {
		t.Errorf("Type = %q", got[0].Type)
	}
	if got[0].Data != `{"hi":1}` {
		t.Errorf("Data = %q", got[0].Data)
	}
}

func TestReadSSE_DefaultEventType(t *testing.T) {
	// No "event:" line — defaults to "message".
	input := "data: hello\n\n"
	var got []SSEEvent
	_ = ReadSSE(strings.NewReader(input), func(ev SSEEvent) error {
		got = append(got, ev)
		return nil
	})
	if len(got) != 1 {
		t.Fatalf("events = %d", len(got))
	}
	if got[0].Type != "message" {
		t.Errorf("Type = %q, want message", got[0].Type)
	}
}

func TestReadSSE_MultipleEvents(t *testing.T) {
	input := "event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"
	var got []SSEEvent
	_ = ReadSSE(strings.NewReader(input), func(ev SSEEvent) error {
		got = append(got, ev)
		return nil
	})
	if len(got) != 2 {
		t.Fatalf("events = %d", len(got))
	}
	if got[0].Type != "a" || got[1].Type != "b" {
		t.Errorf("types = %q,%q", got[0].Type, got[1].Type)
	}
}

func TestReadSSE_CommentLines(t *testing.T) {
	input := ": this is a comment\nevent: ping\ndata: 1\n\n"
	var got []SSEEvent
	_ = ReadSSE(strings.NewReader(input), func(ev SSEEvent) error {
		got = append(got, ev)
		return nil
	})
	if len(got) != 1 {
		t.Fatalf("events = %d, want 1", len(got))
	}
}

func TestReadSSE_StopsOnHandlerError(t *testing.T) {
	input := "data: 1\n\ndata: 2\n\n"
	wantErr := errors.New("stop")
	var count int
	err := ReadSSE(strings.NewReader(input), func(ev SSEEvent) error {
		count++
		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Errorf("err = %v, want %v", err, wantErr)
	}
	if count != 1 {
		t.Errorf("handler invoked %d times, want 1", count)
	}
}

func TestReadSSE_HandlesCRLF(t *testing.T) {
	input := "event: x\r\ndata: y\r\n\r\n"
	var got []SSEEvent
	err := ReadSSE(strings.NewReader(input), func(ev SSEEvent) error {
		got = append(got, ev)
		return nil
	})
	if err != nil {
		t.Fatalf("ReadSSE: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("events = %d", len(got))
	}
	if got[0].Data != "y" {
		t.Errorf("Data = %q", got[0].Data)
	}
}

func TestReadSSE_EmptyStream(t *testing.T) {
	err := ReadSSE(strings.NewReader(""), func(ev SSEEvent) error {
		t.Errorf("unexpected event")
		return nil
	})
	if err != nil {
		t.Errorf("err = %v", err)
	}
}

func TestReadSSE_LargePayload(t *testing.T) {
	// A 1MB data payload.
	big := strings.Repeat("x", 1024*1024)
	input := "data: " + big + "\n\n"
	var got []SSEEvent
	err := ReadSSE(strings.NewReader(input), func(ev SSEEvent) error {
		got = append(got, ev)
		return nil
	})
	if err != nil {
		t.Fatalf("ReadSSE: %v", err)
	}
	if len(got) != 1 || len(got[0].Data) != len(big) {
		t.Errorf("payload lost: %d events, %d bytes", len(got), len(got[0].Data))
	}
}
