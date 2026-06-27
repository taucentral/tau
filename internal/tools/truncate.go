package tools

import (
	"fmt"
	"strings"
)

// DefaultMaxLines is the default per-result line cap applied to tool output
// before it reaches the LLM. 500 lines matches the tools spec default.
const DefaultMaxLines = 500

// DefaultMaxBytes is the default per-result byte cap applied to tool output.
// 512 KiB matches the tools spec default.
const DefaultMaxBytes = 512 * 1024

// headTailSplit is the number of lines preserved at the head and tail of
// truncated output. 500/2 = 250 each side, matching the spec scenario.
const headTailSplit = 2

// TruncateHeadTail caps s to at most maxLines lines AND at most maxBytes
// bytes. When truncation happens, the output is split head/tail with an
// ellipsis marker describing the elision count. The default 500-line / 512
// KiB values preserve the first and last 250 lines (or 256 KiB) so the
// model sees both the start and end of long output.
//
// When s already fits both limits, it is returned unchanged.
//
// maxLines or maxBytes <= 0 disables that dimension (use 0 to apply only
// the other cap). To apply the spec defaults, pass DefaultMaxLines and
// DefaultMaxBytes.
func TruncateHeadTail(s string, maxLines, maxBytes int) string {
	if maxLines < 0 {
		maxLines = 0
	}
	if maxBytes < 0 {
		maxBytes = 0
	}
	// Fast path: no limits requested.
	if maxLines == 0 && maxBytes == 0 {
		return s
	}

	lines := strings.Split(s, "\n")
	totalLines := len(lines)
	totalBytes := len(s)

	// Decide which dimension triggers truncation.
	linesOver := maxLines > 0 && totalLines > maxLines
	bytesOver := maxBytes > 0 && totalBytes > maxBytes
	if !linesOver && !bytesOver {
		return s
	}

	// Compute head/tail sizes per the active constraints. Use the
	// smaller of (maxLines/2) and a reasonable floor so we always
	// preserve content on both sides.
	headLines, tailLines := totalLines, 0
	if maxLines > 0 {
		headLines = maxLines / headTailSplit
		if headLines < 1 {
			headLines = 1
		}
		tailLines = maxLines - headLines
		if tailLines < 1 {
			tailLines = 1
		}
		// Clamp to actual available lines.
		if headLines+tailLines > totalLines {
			headLines = totalLines / headTailSplit
			tailLines = totalLines - headLines
		}
	}

	head := strings.Join(lines[:headLines], "\n")
	tail := strings.Join(lines[totalLines-tailLines:], "\n")

	// Elision count messages.
	linesMsg := ""
	if linesOver {
		elided := totalLines - headLines - tailLines
		linesMsg = fmt.Sprintf("[%d lines elided: %d head, %d tail]", elided, headLines, tailLines)
	}
	bytesMsg := ""
	if bytesOver {
		bytesMsg = fmt.Sprintf("[%d bytes total; output capped at %d bytes]", totalBytes, maxBytes)
	}

	parts := []string{head}
	if linesMsg != "" {
		parts = append(parts, "", linesMsg)
	}
	if bytesMsg != "" {
		parts = append(parts, "", bytesMsg)
	}
	parts = append(parts, "", tail)

	result := strings.Join(parts, "\n")

	// Final byte-cap pass. If the assembled output (rare with default
	// sizes) still exceeds maxBytes, drop the tail until it fits.
	if maxBytes > 0 && len(result) > maxBytes {
		if len(head)+len(linesMsg)+len(bytesMsg)+64 < maxBytes {
			headSuffix := head
			if len(headSuffix) > maxBytes/2 {
				headSuffix = headSuffix[:maxBytes/2]
			}
			result = headSuffix + "\n\n" + linesMsg + "\n" + bytesMsg + "\n\n[tail truncated: byte limit]"
		} else {
			result = result[:maxBytes] + "\n[truncated at byte limit]"
		}
	}
	return result
}

// TruncateBytes caps s to at most maxBytes bytes. If truncation occurs,
// the result is the head of s plus an ellipsis marker describing the
// elision. Use this when only a byte cap (no line structure) is desired.
func TruncateBytes(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	cut := maxBytes
	// Walk back to a valid UTF-8 boundary.
	for cut > 0 && (s[cut]&0xC0) == 0x80 {
		cut--
	}
	return s[:cut] + fmt.Sprintf("\n[truncated: %d bytes total, showing first %d]", len(s), cut)
}
