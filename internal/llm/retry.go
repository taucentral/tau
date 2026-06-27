// retry.go — retry policy and backoff for LLM HTTP requests.
//
// Provider implementations call Do(req, send) to issue an HTTP request with
// automatic retries. The policy:
//
//   - Retry on HTTP 429, 500, 502, 503, 504, and on transient network
//     errors (timeouts, EOF, connection reset).
//   - Honor `Retry-After` on the first retry of a 429/503, then fall back
//     to exponential backoff with jitter.
//   - Backoff: base * 2^attempt, capped at MaxDelay, with full jitter.
//   - Non-retryable status codes (400, 401, 403, 422) return immediately.
//   - Caller's context cancellation always wins.
//
// Defaults match the llm-client spec: MaxRetries=4, MaxDelay=30s.

package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// RetryPolicy controls the behavior of Do.
type RetryPolicy struct {
	// MaxRetries is the number of retry attempts after the initial request.
	// Zero disables retries. Default 4.
	MaxRetries int
	// BaseDelay is the initial backoff interval. Default 500ms.
	BaseDelay time.Duration
	// MaxDelay caps the backoff interval. Default 30s.
	MaxDelay time.Duration
}

// DefaultRetryPolicy returns the policy matching the llm-client spec
// (MaxRetries=4, MaxDelay=30s, BaseDelay=500ms).
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxRetries: 4,
		BaseDelay:  500 * time.Millisecond,
		MaxDelay:   30 * time.Second,
	}
}

// SendFunc issues one HTTP request and returns the response or an error.
// The response body MUST be closed by the caller (or by tryDecode) — the
// retry loop reads the body only to inspect error payloads.
type SendFunc func(ctx context.Context) (*http.Response, error)

// RetryableError marks a network-layer error as eligible for retry.
// HTTP-status retries are decided by status code; this is for the
// transport layer (DNS, connection, EOF, etc.).
type RetryableError struct {
	Err error
}

// Error implements error.
func (e *RetryableError) Error() string { return e.Err.Error() }

// Unwrap exposes the wrapped error for errors.Is / errors.As.
func (e *RetryableError) Unwrap() error { return e.Err }

// NewRetryableError wraps err as retryable. Returns nil if err is nil.
func NewRetryableError(err error) error {
	if err == nil {
		return nil
	}
	return &RetryableError{Err: err}
}

// Err is the terminal error returned by Do after retries are exhausted or
// a non-retryable error occurs. Use errors.As to extract.
type RetryResult struct {
	Attempts  int           // total requests issued (1 + retries)
	WaitTotal time.Duration // sum of pre-retry sleeps
	LastErr   error         // nil if last attempt succeeded
	Status    int           // HTTP status of last attempt (0 if transport-level)
}

// Do issues the request via send, retrying per policy on retryable failures.
// The caller's ctx bounds the total wait.
//
// On success returns the final *http.Response and a nil error. The caller
// must close resp.Body.
//
// On failure returns a nil response and a non-nil error (which may be a
// *RetryableError wrapping the last transport error, an *HTTPStatusError
// for a non-retryable status, or ctx.Err() if the context was cancelled).
func Do(ctx context.Context, policy RetryPolicy, send SendFunc) (*http.Response, *RetryResult, error) {
	if policy.MaxRetries < 0 {
		policy.MaxRetries = 0
	}
	if policy.BaseDelay <= 0 {
		policy.BaseDelay = 500 * time.Millisecond
	}
	if policy.MaxDelay <= 0 {
		policy.MaxDelay = 30 * time.Second
	}

	result := &RetryResult{}
	var (
		resp         *http.Response
		retryAfter   time.Duration
		retryAfterOK bool
	)

	for attempt := 0; attempt <= policy.MaxRetries; attempt++ {
		result.Attempts = attempt + 1
		if err := ctx.Err(); err != nil {
			return nil, result, err
		}
		var err error
		resp, err = send(ctx)
		result.LastErr = err
		if resp != nil {
			result.Status = resp.StatusCode
		}
		// Success: 2xx status and no transport error.
		if err == nil && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return resp, result, nil
		}
		// Failure path. Decide whether to retry.
		canRetry := false
		if err != nil {
			// Transport-level error: retry only if explicitly retryable.
			var re *RetryableError
			canRetry = errors.As(err, &re)
		} else {
			// HTTP-level error: retryable status.
			canRetry = isRetryableStatus(resp.StatusCode)
			if canRetry {
				// Capture Retry-After if present.
				if ra := resp.Header.Get("Retry-After"); ra != "" {
					retryAfter, retryAfterOK = parseRetryAfter(ra)
				}
			}
			// Drain and close body so the connection can be reused.
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			resp = nil
		}
		if !canRetry || attempt == policy.MaxRetries {
			if err != nil {
				return nil, result, err
			}
			return nil, result, &HTTPStatusError{Status: result.Status}
		}
		// Compute the wait interval.
		wait := policy.backoff(attempt)
		if retryAfterOK && attempt == 0 {
			// First retry honors Retry-After, capped at MaxDelay.
			if retryAfter > 0 && retryAfter <= policy.MaxDelay {
				wait = retryAfter
			} else {
				wait = policy.MaxDelay
			}
			retryAfterOK = false // only first retry honors Retry-After
		}
		result.WaitTotal += wait
		select {
		case <-ctx.Done():
			return nil, result, ctx.Err()
		case <-time.After(wait):
		}
	}
	// Unreachable.
	return nil, result, errors.New("retry: loop exited unexpectedly")
}

// HTTPStatusError is returned when a request fails with a non-retryable HTTP
// status code. The body has already been consumed and discarded.
type HTTPStatusError struct {
	Status int
	Body   string // body preview captured on demand (empty if not requested)
}

// Error implements error.
func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("http status %d", e.Status)
}

// IsRetryableStatus reports whether status is in the retryable set.
func IsRetryableStatus(status int) bool { return isRetryableStatus(status) }

// isRetryableStatus is the retryable-status table for the spec.
// 429 (rate limit), 500/502/503/504 (transient server / gateway).
func isRetryableStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout:      // 504
		return true
	}
	return false
}

// IsTransientNetworkError reports whether err is a network-layer error that
// providers should mark retryable via NewRetryableError.
func IsTransientNetworkError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	// Connection-refused / reset / broken-pipe are typically retryable.
	s := err.Error()
	if strings.Contains(s, "connection reset") ||
		strings.Contains(s, "broken pipe") ||
		strings.Contains(s, "connection refused") ||
		strings.Contains(s, "no such host") ||
		strings.Contains(s, "i/o timeout") ||
		strings.Contains(s, "tls: handshake") ||
		strings.Contains(s, "EOF") {
		return true
	}
	return false
}

// backoff computes the wait for attempt n (0-indexed) using full jitter.
func (p RetryPolicy) backoff(attempt int) time.Duration {
	if attempt < 0 {
		attempt = 0
	}
	// Exponential growth, capped.
	d := p.BaseDelay << attempt
	if d <= 0 || d > p.MaxDelay {
		d = p.MaxDelay
	}
	// Full jitter: uniform random in [0, d).
	if d <= 0 {
		return 0
	}
	jitter := time.Duration(rand.Int64N(int64(d)))
	if jitter < 0 {
		jitter = -jitter
	}
	return jitter
}

// parseRetryAfter parses the Retry-After header. Two forms are accepted:
//
//   - integer seconds ("30") — common case
//   - HTTP-date ("Wed, 21 Oct 2025 07:28:00 GMT") — RFC 7231
//
// Returns (duration, true) on success, (0, false) otherwise.
func parseRetryAfter(value string) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	// Integer seconds.
	if secs, err := strconv.Atoi(value); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second, true
	}
	// HTTP-date.
	layouts := []string{
		"Mon, 02 Jan 2006 15:04:05 GMT",
		time.RFC1123,
		time.RFC850,
		time.ANSIC,
	}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, value); err == nil {
			d := time.Until(t)
			if d < 0 {
				return 0, true
			}
			return d, true
		}
	}
	return 0, false
}
