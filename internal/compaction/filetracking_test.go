package compaction

import (
	"strings"
	"testing"
	"time"

	"github.com/coevin/tau/internal/state"
)

func TestExtractFileTracking_ReadAndModify(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "read", "tu1", map[string]any{"path": "/a"}, nowAt(1*time.Second)),
		mkToolUseMessage("e1", "r1", "edit", "tu2", map[string]any{"path": "/b"}, nowAt(2*time.Second)),
		mkToolUseMessage("w1", "e1", "write", "tu3", map[string]any{"path": "/c"}, nowAt(3*time.Second)),
		mkToolUseMessage("p1", "w1", "patch", "tu4", map[string]any{"path": "/d"}, nowAt(4*time.Second)),
	}
	got := ExtractFileTracking(entries, DefaultFileReadTools, DefaultFileWriteTools)
	if len(got.Reads) != 1 {
		t.Errorf("reads = %d, want 1: %+v", len(got.Reads), got.Reads)
	} else if got.Reads[0].Path != "/a" {
		t.Errorf("reads[0].Path = %q, want /a", got.Reads[0].Path)
	}
	if len(got.Modifications) != 3 {
		t.Fatalf("modifications = %d, want 3: %+v", len(got.Modifications), got.Modifications)
	}
	wantPaths := []string{"/b", "/c", "/d"}
	for i, want := range wantPaths {
		if got.Modifications[i].Path != want {
			t.Errorf("modifications[%d].Path = %q, want %q", i, got.Modifications[i].Path, want)
		}
	}
}

func TestExtractFileTracking_CustomToolNames(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "view", "tu1", map[string]any{"path": "/x"}, nowAt(0)),
		mkToolUseMessage("w1", "r1", "update_file", "tu2", map[string]any{"path": "/y"}, nowAt(0)),
	}
	got := ExtractFileTracking(entries, []string{"view"}, []string{"update_file"})
	if len(got.Reads) != 1 || got.Reads[0].Path != "/x" {
		t.Errorf("reads = %+v, want /x", got.Reads)
	}
	if len(got.Modifications) != 1 || got.Modifications[0].Path != "/y" {
		t.Errorf("mods = %+v, want /y", got.Modifications)
	}
}

func TestExtractFileTracking_MultipleReadsSameFile(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "read", "tu1", map[string]any{"path": "/same"}, nowAt(1*time.Second)),
		mkToolUseMessage("r2", "r1", "read", "tu2", map[string]any{"path": "/same"}, nowAt(2*time.Second)),
	}
	got := ExtractFileTracking(entries, DefaultFileReadTools, DefaultFileWriteTools)
	if len(got.Reads) != 2 {
		t.Errorf("reads = %d, want 2 (no dedup at extraction)", len(got.Reads))
	}
	if got.Reads[0].EntryID != "r1" {
		t.Errorf("reads[0] should be r1 (older first), got %q", got.Reads[0].EntryID)
	}
	if got.Reads[1].EntryID != "r2" {
		t.Errorf("reads[1] should be r2 (newer), got %q", got.Reads[1].EntryID)
	}
}

func TestExtractFileTracking_PathFieldFallbacks(t *testing.T) {
	cases := []struct {
		name string
		key  string
	}{
		{"path", "path"},
		{"file_path", "file_path"},
		{"filePath", "filePath"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			entries := []state.Entry{
				mkSessionHeader("h"),
				mkToolUseMessage("r1", "h", "read", "tu1", map[string]any{c.key: "/x"}, nowAt(0)),
			}
			got := ExtractFileTracking(entries, DefaultFileReadTools, DefaultFileWriteTools)
			if len(got.Reads) != 1 || got.Reads[0].Path != "/x" {
				t.Errorf("extractor missed %q field: %+v", c.key, got.Reads)
			}
		})
	}
}

func TestExtractFileTracking_EmptyPathSkipped(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "read", "tu1", map[string]any{"unrelated": "x"}, nowAt(0)),
		mkToolUseMessage("r2", "r1", "read", "tu2", map[string]any{"path": ""}, nowAt(0)),
		mkToolUseMessage("r3", "r2", "read", "tu3", map[string]any{"path": "/ok"}, nowAt(0)),
	}
	got := ExtractFileTracking(entries, DefaultFileReadTools, DefaultFileWriteTools)
	if len(got.Reads) != 1 {
		t.Errorf("reads = %d, want 1 (only /ok qualifies): %+v", len(got.Reads), got.Reads)
	}
}

func TestExtractFileTracking_NonMessageEntriesIgnored(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkLabel("l1", "h", "tag", nowAt(0)),
		mkSessionInfo("s1", "l1", "k", "v", nowAt(0)),
	}
	got := ExtractFileTracking(entries, DefaultFileReadTools, DefaultFileWriteTools)
	if len(got.Reads) != 0 || len(got.Modifications) != 0 {
		t.Errorf("non-message entries produced operations: %+v", got)
	}
}

func TestExtractFileTracking_ChronologicalSort(t *testing.T) {
	// Pass entries out of chronological order; the extractor should still
	// sort the output oldest-first.
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("late", "h", "read", "tuLate", map[string]any{"path": "/late"}, nowAt(50*time.Second)),
		mkToolUseMessage("early", "h", "read", "tuEarly", map[string]any{"path": "/early"}, nowAt(1*time.Second)),
	}
	got := ExtractFileTracking(entries, DefaultFileReadTools, DefaultFileWriteTools)
	if len(got.Reads) != 2 {
		t.Fatalf("reads = %d, want 2", len(got.Reads))
	}
	if got.Reads[0].EntryID != "early" {
		t.Errorf("reads[0] = %q, want 'early' (oldest first)", got.Reads[0].EntryID)
	}
	if got.Reads[1].EntryID != "late" {
		t.Errorf("reads[1] = %q, want 'late'", got.Reads[1].EntryID)
	}
}

func TestFileTracking_DistinctPaths(t *testing.T) {
	ft := FileTracking{
		Reads: []FileOperation{
			{Path: "/a", Operation: "read", EntryID: "r1", Timestamp: nowAt(1 * time.Second)},
			{Path: "/a", Operation: "read", EntryID: "r2", Timestamp: nowAt(10 * time.Second)},
			{Path: "/b", Operation: "read", EntryID: "r3", Timestamp: nowAt(5 * time.Second)},
		},
		Modifications: []FileOperation{
			{Path: "/a", Operation: "modify", EntryID: "m1", Timestamp: nowAt(20 * time.Second)},
		},
	}
	dp := ft.DistinctPaths()
	if len(dp) != 2 {
		t.Fatalf("DistinctPaths len = %d, want 2: %+v", len(dp), dp)
	}
	// /a: latest operation is m1 (modify at 20s).
	if dp["/a"].EntryID != "m1" {
		t.Errorf("/a op = %+v, want m1", dp["/a"])
	}
	// /b: only r3.
	if dp["/b"].EntryID != "r3" {
		t.Errorf("/b op = %+v, want r3", dp["/b"])
	}
}

func TestFileTracking_CriticalContextSection_Empty(t *testing.T) {
	ft := FileTracking{}
	if got := ft.CriticalContextSection(); got != "" {
		t.Errorf("empty CriticalContextSection = %q, want empty", got)
	}
}

func TestFileTracking_CriticalContextSection_Renders(t *testing.T) {
	ft := FileTracking{
		Reads: []FileOperation{
			{Path: "/a", Operation: "read"},
			{Path: "/b", Operation: "read"},
		},
		Modifications: []FileOperation{
			{Path: "/c", Operation: "modify"},
		},
	}
	got := ft.CriticalContextSection()
	if !strings.Contains(got, "/a") || !strings.Contains(got, "/b") {
		t.Errorf("expected /a and /b in section: %s", got)
	}
	if !strings.Contains(got, "/c") {
		t.Errorf("expected /c in section: %s", got)
	}
	if !strings.Contains(got, "Files read:") {
		t.Errorf("expected 'Files read:' header: %s", got)
	}
	if !strings.Contains(got, "Files modified:") {
		t.Errorf("expected 'Files modified:' header: %s", got)
	}
}

func TestFileTracking_CriticalContextSection_DeduplicatesPaths(t *testing.T) {
	// Multiple operations on the same path appear once in the section.
	ft := FileTracking{
		Reads: []FileOperation{
			{Path: "/a", Operation: "read"},
			{Path: "/a", Operation: "read"},
			{Path: "/a", Operation: "read"},
		},
	}
	got := ft.CriticalContextSection()
	if cnt := strings.Count(got, "/a"); cnt != 1 {
		t.Errorf("/a appears %d times, want 1: %s", cnt, got)
	}
}

func TestExtractFileTracking_CaseInsensitiveNames(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "READ", "tu1", map[string]any{"path": "/a"}, nowAt(0)),
		mkToolUseMessage("w1", "r1", "EDIT", "tu2", map[string]any{"path": "/b"}, nowAt(0)),
	}
	got := ExtractFileTracking(entries, DefaultFileReadTools, DefaultFileWriteTools)
	if len(got.Reads) != 1 {
		t.Errorf("reads = %d, want 1 (case-insensitive match)", len(got.Reads))
	}
	if len(got.Modifications) != 1 {
		t.Errorf("mods = %d, want 1 (case-insensitive match)", len(got.Modifications))
	}
}

func TestExtractFileTracking_DefaultsWhenEmpty(t *testing.T) {
	// Passing nil for readTools/writeTools should fall back to defaults.
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "read", "tu1", map[string]any{"path": "/a"}, nowAt(0)),
	}
	// Note: ExtractFileTracking itself does NOT fall back; the caller
	// (compactor) is responsible for substituting defaults. Verify that
	// passing nil produces zero matches (so the contract is explicit).
	got := ExtractFileTracking(entries, nil, nil)
	if len(got.Reads) != 0 {
		t.Errorf("reads = %d, want 0 (nil tool sets)", len(got.Reads))
	}
}

func TestDefaultFileTools_NonEmpty(t *testing.T) {
	// Sanity: the defaults must be non-empty so out-of-the-box tau tracks
	// the canonical tool names.
	if len(DefaultFileReadTools) == 0 {
		t.Errorf("DefaultFileReadTools is empty")
	}
	if len(DefaultFileWriteTools) == 0 {
		t.Errorf("DefaultFileWriteTools is empty")
	}
}
