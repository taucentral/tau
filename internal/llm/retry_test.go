package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestRetryPolicy_Defaults(t *testing.T) {
	p := DefaultRetryPolicy()
	if p.MaxRetries != 4 {
		t.Errorf("MaxRetries = %d", p.MaxRetries)
	}
	if p.MaxDelay != 30*time.Second {
		t.Errorf("MaxDelay = %v", p.MaxDelay)
	}
}

func TestIsRetryableStatus(t *testing.T) {
	cases := map[int]bool{
		200: false,
		400: false,
		401: false,
		403: false,
		404: false,
		422: false,
		429: true,
		500: true,
		502: true,
		503: true,
		504: true,
	}
	for status, want := range cases {
		if got := IsRetryableStatus(status); got != want {
			t.Errorf("IsRetryableStatus(%d) = %v, want %v", status, got, want)
		}
	}
}

func TestIsTransientNetworkError_Timeout(t *testing.T) {
	ne := &net.OpError{Op: "read", Err: fmt.Errorf("i/o timeout")}
	if !IsTransientNetworkError(ne) {
		t.Errorf("expected retryable net timeout")
	}
}

func TestIsTransientNetworkError_ConnectionReset(t *testing.T) {
	err := errors.New("read tcp: connection reset by peer")
	if !IsTransientNetworkError(err) {
		t.Errorf("expected retryable connection reset")
	}
}

func TestIsTransientNetworkError_Nil(t *testing.T) {
	if IsTransientNetworkError(nil) {
		t.Errorf("nil should not be retryable")
	}
}

func TestDo_SuccessFirstTry(t *testing.T) {
	p := RetryPolicy{MaxRetries: 4, BaseDelay: time.Millisecond, MaxDelay: 10 * time.Millisecond}
	calls := 0
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: 200, Body: http.NoBody}, nil
	}
	resp, result, err := Do(context.Background(), p, send)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer closeResp(resp)
	if resp == nil || resp.StatusCode != 200 {
		t.Errorf("resp = %+v", resp)
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}
	if result.Attempts != 1 {
		t.Errorf("Attempts = %d", result.Attempts)
	}
}

func TestDo_RetriesOn503(t *testing.T) {
	p := RetryPolicy{MaxRetries: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond}
	calls := 0
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		if calls < 3 {
			return &http.Response{StatusCode: 503, Body: http.NoBody}, nil
		}
		return &http.Response{StatusCode: 200, Body: http.NoBody}, nil
	}
	resp, result, err := Do(context.Background(), p, send)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer closeResp(resp)
	if resp.StatusCode != 200 {
		t.Errorf("status = %d", resp.StatusCode)
	}
	if calls != 3 {
		t.Errorf("calls = %d, want 3", calls)
	}
	if result.Attempts != 3 {
		t.Errorf("Attempts = %d", result.Attempts)
	}
}

func TestDo_RetriesOn429WithRetryAfter(t *testing.T) {
	p := RetryPolicy{MaxRetries: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Second}
	calls := 0
	var gotWait time.Duration
	start := time.Now()
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		if calls == 1 {
			h := http.Header{}
			h.Set("Retry-After", "1")
			return &http.Response{StatusCode: 429, Header: h, Body: http.NoBody}, nil
		}
		gotWait = time.Since(start)
		return &http.Response{StatusCode: 200, Body: http.NoBody}, nil
	}
	resp, _, err := Do(context.Background(), p, send)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer closeResp(resp)
	if resp.StatusCode != 200 {
		t.Errorf("status = %d", resp.StatusCode)
	}
	if gotWait < 900*time.Millisecond {
		t.Errorf("Retry-After not honored: waited %v", gotWait)
	}
}

func TestDo_DoesNotRetryOn400(t *testing.T) {
	p := RetryPolicy{MaxRetries: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond}
	calls := 0
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: 400, Body: http.NoBody}, nil
	}
	resp, _, err := Do(context.Background(), p, send)
	defer closeResp(resp)
	if err == nil {
		t.Fatal("expected error")
	}
	var se *HTTPStatusError
	if !errors.As(err, &se) || se.Status != 400 {
		t.Errorf("err = %v, want HTTPStatusError{400}", err)
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1 (no retry)", calls)
	}
}

func TestDo_RetriesOnRetryableNetworkError(t *testing.T) {
	p := RetryPolicy{MaxRetries: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond}
	calls := 0
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		if calls < 2 {
			return nil, NewRetryableError(errors.New("connection reset by peer"))
		}
		return &http.Response{StatusCode: 200, Body: http.NoBody}, nil
	}
	resp, _, err := Do(context.Background(), p, send)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer closeResp(resp)
	if resp.StatusCode != 200 {
		t.Errorf("status = %d", resp.StatusCode)
	}
	if calls != 2 {
		t.Errorf("calls = %d, want 2", calls)
	}
}

func TestDo_DoesNotRetryOnNonRetryableError(t *testing.T) {
	p := RetryPolicy{MaxRetries: 3, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond}
	calls := 0
	myErr := errors.New("custom non-retryable")
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		return nil, myErr
	}
	resp, _, err := Do(context.Background(), p, send)
	defer closeResp(resp)
	if !errors.Is(err, myErr) {
		t.Errorf("err = %v, want %v", err, myErr)
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}
}

func TestDo_RespectsCtxCancel(t *testing.T) {
	p := RetryPolicy{MaxRetries: 5, BaseDelay: 50 * time.Millisecond, MaxDelay: 100 * time.Millisecond}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()
	calls := 0
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: 503, Body: http.NoBody}, nil
	}
	resp, _, err := Do(ctx, p, send)
	defer closeResp(resp)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestDo_RetriesExhaustedReturnsStatusError(t *testing.T) {
	p := RetryPolicy{MaxRetries: 2, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond}
	send := func(ctx context.Context) (*http.Response, error) {
		return &http.Response{StatusCode: 503, Body: http.NoBody}, nil
	}
	resp, result, err := Do(context.Background(), p, send)
	defer closeResp(resp)
	if err == nil {
		t.Fatal("expected error")
	}
	var se *HTTPStatusError
	if !errors.As(err, &se) || se.Status != 503 {
		t.Errorf("err = %v, want HTTPStatusError{503}", err)
	}
	if result.Attempts != 3 { // 1 initial + 2 retries
		t.Errorf("Attempts = %d, want 3", result.Attempts)
	}
}

func TestParseRetryAfter_IntegerSeconds(t *testing.T) {
	d, ok := parseRetryAfter("30")
	if !ok {
		t.Fatal("ok = false")
	}
	if d != 30*time.Second {
		t.Errorf("d = %v", d)
	}
}

func TestParseRetryAfter_Zero(t *testing.T) {
	d, ok := parseRetryAfter("0")
	if !ok {
		t.Fatal("ok = false")
	}
	if d != 0 {
		t.Errorf("d = %v", d)
	}
}

func TestParseRetryAfter_HTTPDate(t *testing.T) {
	// A date 60 seconds in the future.
	future := time.Now().Add(60 * time.Second).UTC().Format("Mon, 02 Jan 2006 15:04:05 GMT")
	d, ok := parseRetryAfter(future)
	if !ok {
		t.Fatal("ok = false")
	}
	if d < 30*time.Second || d > 90*time.Second {
		t.Errorf("d = %v, want ~60s", d)
	}
}

func TestParseRetryAfter_HTTPDatePast(t *testing.T) {
	past := time.Now().Add(-60 * time.Second).UTC().Format("Mon, 02 Jan 2006 15:04:05 GMT")
	d, ok := parseRetryAfter(past)
	if !ok {
		t.Fatal("ok = false for past date")
	}
	if d != 0 {
		t.Errorf("d = %v, want 0 for past date", d)
	}
}

func TestParseRetryAfter_Garbage(t *testing.T) {
	_, ok := parseRetryAfter("not-a-date-or-number")
	if ok {
		t.Errorf("expected ok=false")
	}
}

func TestParseRetryAfter_Empty(t *testing.T) {
	_, ok := parseRetryAfter("")
	if ok {
		t.Errorf("expected ok=false")
	}
}

func TestBackoff_RespectsMaxDelay(t *testing.T) {
	p := RetryPolicy{BaseDelay: time.Second, MaxDelay: 5 * time.Second}
	for attempt := 0; attempt < 10; attempt++ {
		d := p.backoff(attempt)
		if d > p.MaxDelay {
			t.Errorf("attempt %d: d=%v > MaxDelay=%v", attempt, d, p.MaxDelay)
		}
	}
}

func TestBackoff_Grows(t *testing.T) {
	p := RetryPolicy{BaseDelay: 10 * time.Millisecond, MaxDelay: 1 * time.Second}
	// Average over multiple samples to defeat jitter.
	const samples = 100
	var prevAvg time.Duration
	for attempt := 0; attempt < 5; attempt++ {
		var total time.Duration
		for i := 0; i < samples; i++ {
			total += p.backoff(attempt)
		}
		avg := total / samples
		// Each higher attempt should average higher than the previous (until capped).
		if avg <= prevAvg && attempt > 0 {
			// Allow this only once we've hit the cap.
			halfCap := p.MaxDelay / 2 // average of full-jitter is cap/2
			if avg < halfCap-time.Millisecond {
				t.Errorf("attempt %d avg=%v not growing past prev=%v", attempt, avg, prevAvg)
			}
		}
		prevAvg = avg
	}
}

func TestRetryableError_Unwrap(t *testing.T) {
	inner := errors.New("inner")
	wrapped := NewRetryableError(inner)
	if !errors.Is(wrapped, inner) {
		t.Errorf("errors.Is failed")
	}
	var re *RetryableError
	if !errors.As(wrapped, &re) {
		t.Errorf("errors.As failed")
	}
}

func TestRetryableError_Nil(t *testing.T) {
	if NewRetryableError(nil) != nil {
		t.Errorf("nil should pass through")
	}
}

func TestHTTPStatusError_Error(t *testing.T) {
	se := &HTTPStatusError{Status: 429}
	if !strings.Contains(se.Error(), "429") {
		t.Errorf("Error() = %q", se.Error())
	}
}

func TestDo_ClosesRetryableResponseBody(t *testing.T) {
	// After a 503 the body must be closed so the connection can be reused.
	p := RetryPolicy{MaxRetries: 1, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond}
	calls := 0
	var bodies []io.Closer
	send := func(ctx context.Context) (*http.Response, error) {
		calls++
		body := &bodyCloser{}
		bodies = append(bodies, body)
		if calls == 1 {
			return &http.Response{StatusCode: 503, Body: body}, nil
		}
		return &http.Response{StatusCode: 200, Body: body}, nil
	}
	resp, _, err := Do(context.Background(), p, send)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if !bodies[0].(*bodyCloser).closed {
		t.Errorf("first body not closed after retry")
	}
}

func TestRetryableError_ErrorAndUnwrap(t *testing.T) {
	wrapped := errors.New("transport: connection reset")
	r := NewRetryableError(wrapped)
	if r.Error() != wrapped.Error() {
		t.Errorf("Error() = %q, want %q", r.Error(), wrapped.Error())
	}
	if !errors.Is(r, wrapped) {
		t.Errorf("errors.Is(r, wrapped) = false, want true")
	}
}

// bodyCloser is a minimal io.ReadCloser that records Close calls.
type bodyCloser struct {
	closed bool
}

func (b *bodyCloser) Read(p []byte) (int, error) { return 0, io.EOF }
func (b *bodyCloser) Close() error               { b.closed = true; return nil }

// closeResp is a test helper that closes resp.Body when non-nil. The bodyclose
// linter cannot prove that llm.Do returns nil resp on its error path (per the
// Do contract at retry.go:96), so test callers must defensively close.
func closeResp(resp *http.Response) {
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
}
