package compaction

import (
	"testing"
	"time"

	"github.com/coevin/tau/internal/state"
)

func TestProtectionList_Empty(t *testing.T) {
	p := BuildProtectionList(nil, ProtectionConfig{})
	if p.Len() != 0 {
		t.Errorf("Len = %d, want 0", p.Len())
	}
	if got := p.IDs(); len(got) != 0 {
		t.Errorf("IDs() = %v, want empty", got)
	}
	if p.Contains("anything") {
		t.Errorf("Contains on empty list returned true")
	}
}

func TestProtectionList_SessionInfoAndLabel(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkLabel("l1", "h", "stable", nowAt(0)),
		mkSessionInfo("s1", "l1", "k", "v", nowAt(0)),
		mkMessage("m1", "s1", "user", "hi", nowAt(0)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{})
	// Two protected: l1 (Label) and s1 (SessionInfo).
	if p.Len() != 2 {
		t.Fatalf("Len = %d, want 2", p.Len())
	}
	if !p.Contains("l1") {
		t.Errorf("Label l1 not protected")
	}
	if !p.Contains("s1") {
		t.Errorf("SessionInfo s1 not protected")
	}
	if p.Contains("m1") {
		t.Errorf("user message m1 should not be protected")
	}
}

func TestProtectionList_MostRecentReadPerFile(t *testing.T) {
	// Two reads of file A: the LATER read should be protected.
	// One read of file B: protected.
	// One read of file C: protected.
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("rA1", "h", "read", "tuA1", map[string]any{"path": "/a"}, nowAt(1*time.Second)),
		mkToolUseMessage("rA2", "rA1", "read", "tuA2", map[string]any{"path": "/a"}, nowAt(10*time.Second)),
		mkToolUseMessage("rB1", "rA2", "read", "tuB1", map[string]any{"path": "/b"}, nowAt(20*time.Second)),
		mkToolUseMessage("rC1", "rB1", "read", "tuC1", map[string]any{"path": "/c"}, nowAt(30*time.Second)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{MaxRecentFileReads: 5})
	// Three distinct paths, so 3 read entries protected.
	if p.Len() != 3 {
		t.Fatalf("Len = %d, want 3 (most-recent per path)", p.Len())
	}
	if !p.Contains("rA2") {
		t.Errorf("rA2 (most recent /a) should be protected")
	}
	if p.Contains("rA1") {
		t.Errorf("rA1 (older /a) should NOT be protected")
	}
	if !p.Contains("rB1") {
		t.Errorf("rB1 should be protected")
	}
	if !p.Contains("rC1") {
		t.Errorf("rC1 should be protected")
	}
}

func TestProtectionList_CapMaxRecentFileReads(t *testing.T) {
	// 7 distinct file reads; cap=5 → only 5 newest protected.
	entries := []state.Entry{mkSessionHeader("h")}
	parent := "h"
	for i := 0; i < 7; i++ {
		id := "r" + string(rune('1'+i))
		path := "/file" + string(rune('1'+i))
		e := mkToolUseMessage(id, parent, "read", "tu"+id, map[string]any{"path": path}, nowAt(time.Duration(i+1)*time.Second))
		entries = append(entries, e)
		parent = id
	}
	p := BuildProtectionList(entries, ProtectionConfig{MaxRecentFileReads: 5})
	if p.Len() != 5 {
		t.Fatalf("Len = %d, want 5 (cap)", p.Len())
	}
	// Newest 5 = r3..r7 (r1, r2 are oldest two, dropped).
	if p.Contains("r1") {
		t.Errorf("r1 should be evicted (too old)")
	}
	if p.Contains("r2") {
		t.Errorf("r2 should be evicted (too old)")
	}
	for _, want := range []string{"r3", "r4", "r5", "r6", "r7"} {
		if !p.Contains(want) {
			t.Errorf("expected %q protected", want)
		}
	}
}

func TestProtectionList_DefaultMaxFromZero(t *testing.T) {
	entries := []state.Entry{mkSessionHeader("h")}
	parent := "h"
	for i := 0; i < DefaultMaxRecentFileReads+2; i++ {
		id := "r" + string(rune('a'+i))
		path := "/p" + string(rune('a'+i))
		e := mkToolUseMessage(id, parent, "read", "tu"+id, map[string]any{"path": path}, nowAt(time.Duration(i)*time.Second))
		entries = append(entries, e)
		parent = id
	}
	// MaxRecentFileReads=0 → DefaultMaxRecentFileReads (=5).
	p := BuildProtectionList(entries, ProtectionConfig{MaxRecentFileReads: 0})
	if p.Len() != DefaultMaxRecentFileReads {
		t.Errorf("Len = %d, want default %d", p.Len(), DefaultMaxRecentFileReads)
	}
}

func TestProtectionList_CustomReadTools(t *testing.T) {
	// A custom tool "myread" should be picked up via config.
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "myread", "tu1", map[string]any{"path": "/x"}, nowAt(1*time.Second)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{
		MaxRecentFileReads: 5,
		FileReadTools:      []string{"myread"},
	})
	if !p.Contains("r1") {
		t.Errorf("r1 from custom read tool should be protected")
	}
}

func TestProtectionList_CaseInsensitiveToolName(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "READ", "tu1", map[string]any{"path": "/x"}, nowAt(1*time.Second)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{})
	if !p.Contains("r1") {
		t.Errorf("READ (uppercase) should match default 'read' read tool")
	}
}

func TestProtectionList_WriteToolsNotProtected(t *testing.T) {
	// Writes/modifications are tracked but NOT protected (per spec, only
	// reads are protected). The protection list should not include write
	// tool entries.
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("w1", "h", "edit", "tu1", map[string]any{"path": "/x"}, nowAt(1*time.Second)),
		mkToolUseMessage("w2", "h", "write", "tu2", map[string]any{"path": "/y"}, nowAt(2*time.Second)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{})
	if p.Len() != 0 {
		t.Errorf("Len = %d, want 0 (writes not protected)", p.Len())
	}
}

func TestProtectionList_MissingPathSkipped(t *testing.T) {
	// A read tool call without a recognizable path field is skipped.
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("r1", "h", "read", "tu1", map[string]any{"not_a_path": "x"}, nowAt(1*time.Second)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{})
	if p.Len() != 0 {
		t.Errorf("Len = %d, want 0 (no recognizable path)", p.Len())
	}
}

func TestProtectionList_IDsAreSorted(t *testing.T) {
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkLabel("zzz", "h", "late", nowAt(0)),
		mkLabel("aaa", "h", "early", nowAt(0)),
		mkLabel("mmm", "h", "middle", nowAt(0)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{})
	got := p.IDs()
	want := []string{"aaa", "mmm", "zzz"}
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Errorf("IDs() = %v, want %v (sorted)", got, want)
	}
}

func TestProtectionList_AnyFilePathFieldWorks(t *testing.T) {
	// The extractor tries "path", then "file_path", then "filePath".
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
				mkToolUseMessage("r1", "h", "read", "tu1", map[string]any{c.key: "/x"}, nowAt(1*time.Second)),
			}
			p := BuildProtectionList(entries, ProtectionConfig{MaxRecentFileReads: 5})
			if !p.Contains("r1") {
				t.Errorf("read using %q field should be protected", c.key)
			}
		})
	}
}

func TestProtectionList_LeafIsNewest(t *testing.T) {
	// The most-recent read should win even when entries are passed in
	// chronological order; the protection builder should not assume any
	// particular input order.
	entries := []state.Entry{
		mkSessionHeader("h"),
		mkToolUseMessage("old", "h", "read", "tuOld", map[string]any{"path": "/a"}, nowAt(1*time.Second)),
		mkToolUseMessage("new", "h", "read", "tuNew", map[string]any{"path": "/a"}, nowAt(50*time.Second)),
	}
	p := BuildProtectionList(entries, ProtectionConfig{MaxRecentFileReads: 5})
	if p.Contains("old") {
		t.Errorf("older read of /a should not be protected")
	}
	if !p.Contains("new") {
		t.Errorf("newest read of /a should be protected")
	}
}
