package components

import (
	"strings"
	"testing"
)

func testTheme() Theme { return DarkTheme }

// --- Markdown ---

func TestMarkdownRenderer_Render(t *testing.T) {
	mr := NewMarkdownRenderer(testTheme(), 80)
	out, err := mr.Render("# Hello\n\nSome **bold** text.")
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out == "" {
		t.Error("Render returned empty string")
	}
}

func TestMarkdownRenderer_RenderCodeBlock(t *testing.T) {
	mr := NewMarkdownRenderer(testTheme(), 80)
	md := "```go\npackage main\n```"
	out, err := mr.Render(md)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	// glamour inserts ANSI escape codes around keywords, so we check
	// for both words independently rather than the literal substring.
	if !strings.Contains(out, "package") {
		t.Errorf("code block output missing 'package': %q", out)
	}
	if !strings.Contains(out, "main") {
		t.Errorf("code block output missing 'main': %q", out)
	}
}

func TestMarkdownRenderer_ConcurrentRender(t *testing.T) {
	mr := NewMarkdownRenderer(testTheme(), 80)
	done := make(chan struct{}, 10)
	for i := 0; i < 10; i++ {
		go func() {
			_, err := mr.Render("# Concurrent")
			if err != nil {
				t.Errorf("Render: %v", err)
			}
			done <- struct{}{}
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestMarkdownRenderer_SetWidth(t *testing.T) {
	mr := NewMarkdownRenderer(testTheme(), 40)
	mr.SetWidth(120)
	out, err := mr.Render(strings.Repeat("x", 100))
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out == "" {
		t.Error("Render returned empty after width change")
	}
}

// --- Viewport ---

func TestViewport_AppendText(t *testing.T) {
	vp := NewViewport(testTheme(), 80, 24)
	vp.AppendText("user", "hello")
	vp.AppendText("assistant", "hi there")
	if len(vp.elements) != 2 {
		t.Errorf("elements = %d, want 2", len(vp.elements))
	}
}

func TestViewport_AutoScroll(t *testing.T) {
	vp := NewViewport(testTheme(), 80, 24)
	if !vp.AutoScroll() {
		t.Error("auto-scroll should default to true")
	}
	vp.ScrollUp()
	if vp.AutoScroll() {
		t.Error("auto-scroll should be false after ScrollUp")
	}
}

func TestViewport_SetDimensions(t *testing.T) {
	vp := NewViewport(testTheme(), 80, 24)
	vp.SetDimensions(120, 40)
	if vp.width != 120 || vp.height != 40 {
		t.Errorf("dimensions = %dx%d, want 120x40", vp.width, vp.height)
	}
}

func TestViewport_ScrollbackCap(t *testing.T) {
	vp := NewViewport(testTheme(), 80, 24)
	vp.maxLines = 5
	for i := 0; i < 20; i++ {
		vp.AppendText("user", "line")
	}
	// The viewport's TotalLineCount reflects the content set via
	// SetContent, which is capped by maxLines.
	total := vp.vp.TotalLineCount()
	if total > 10 {
		t.Errorf("scrollback not capped: TotalLineCount = %d, want <= 10", total)
	}
}

// --- Accordion ---

func TestAccordion_Collapsed(t *testing.T) {
	acc := NewAccordion(testTheme(), "bash", `{"command":"ls -la"}`)
	acc.SetWidth(80)
	out := acc.View()
	if !strings.Contains(out, "bash") {
		t.Errorf("collapsed view missing tool name: %q", out)
	}
	if !strings.Contains(out, "▸") {
		t.Errorf("collapsed view missing expand indicator: %q", out)
	}
}

func TestAccordion_Expanded(t *testing.T) {
	acc := NewAccordion(testTheme(), "bash", `{"command":"ls"}`)
	acc.Expanded = true
	acc.SetResult("file1\nfile2")
	acc.SetWidth(80)
	out := acc.View()
	if !strings.Contains(out, "▾") {
		t.Errorf("expanded view missing collapse indicator: %q", out)
	}
	if !strings.Contains(out, "file1") {
		t.Errorf("expanded view missing result: %q", out)
	}
}

func TestAccordion_Running(t *testing.T) {
	acc := NewAccordion(testTheme(), "bash", `{"command":"sleep 10"}`)
	acc.Running = true
	acc.SetWidth(80)
	out := acc.View()
	if !strings.Contains(out, "◐") {
		t.Errorf("running view missing spinner icon: %q", out)
	}
}

func TestAccordion_Error(t *testing.T) {
	acc := NewAccordion(testTheme(), "bash", `{"command":"false"}`)
	acc.IsError = true
	acc.SetResult("exit code 1")
	acc.SetWidth(80)
	out := acc.View()
	if !strings.Contains(out, "✗") {
		t.Errorf("error view missing error icon: %q", out)
	}
}

func TestAccordion_Success(t *testing.T) {
	acc := NewAccordion(testTheme(), "read", `{"path":"/etc/host"}`)
	acc.SetResult("file contents")
	acc.SetWidth(80)
	out := acc.View()
	if !strings.Contains(out, "✓") {
		t.Errorf("success view missing checkmark: %q", out)
	}
}

func TestAccordion_Toggle(t *testing.T) {
	acc := NewAccordion(testTheme(), "bash", "")
	if acc.Expanded {
		t.Error("accordion should start collapsed")
	}
	acc.Toggle()
	if !acc.Expanded {
		t.Error("accordion should be expanded after toggle")
	}
	acc.Toggle()
	if acc.Expanded {
		t.Error("accordion should be collapsed after second toggle")
	}
}

func TestTruncateArgs(t *testing.T) {
	short := `{"a":"b"}`
	got := truncateArgs(short, 80, "bash")
	if got != short {
		t.Errorf("truncateArgs short: got %q, want %q", got, short)
	}
	long := `{"command":"` + strings.Repeat("x", 200) + `"}`
	got = truncateArgs(long, 40, "bash")
	if len(got) > 40 {
		t.Errorf("truncateArgs long: len = %d, want <= 40", len(got))
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("truncateArgs long: should end with …, got %q", got)
	}
}

// --- Footer ---

func TestFooter_View(t *testing.T) {
	t.Setenv("HOME", "/home/user")
	f := NewFooter(testTheme(), 80)
	f.SetCwd("/home/user/dev/project")
	f.SetModel("glm-5.2")
	f.SetProviderAPI("anthropic")
	f.SetThinkingLevel("high")
	f.SetTokens(100000, 200000)
	out := f.View()
	for _, want := range []string{"~", "50%", "200k", "glm-5.2", "high", "[anthropic]"} {
		if !strings.Contains(out, want) {
			t.Errorf("footer missing %q: %q", want, out)
		}
	}
}

func TestFooter_EmptyDefaults(t *testing.T) {
	f := NewFooter(testTheme(), 80)
	out := f.View()
	if out == "" {
		t.Error("empty footer should still produce output")
	}
	// View always emits exactly two lines (cwd + status).
	if n := strings.Count(out, "\n") + 1; n != FooterHeightLines {
		t.Errorf("footer line count = %d, want %d: %q", n, FooterHeightLines, out)
	}
}

func TestFooter_TokenGauge(t *testing.T) {
	f := NewFooter(testTheme(), 80)
	f.SetTokens(0, 0)
	out := f.View()
	if !strings.Contains(out, "?") {
		t.Errorf("zero-max tokens should show '?': %q", out)
	}
}

// TestFooter_TokenColorThresholds verifies the gauge picks the right
// style at each threshold. We can't inspect ANSI color codes without
// reaching into lipgloss internals, so we just verify the percentage
// itself renders at each boundary.
func TestFooter_TokenColorThresholds(t *testing.T) {
	cases := []struct {
		used int
		max  int
		want string
	}{
		{50000, 200000, "25%"},  // muted band
		{150000, 200000, "75%"}, // accent band (>70)
		{195000, 200000, "98%"}, // warning band (>90)
	}
	for _, tc := range cases {
		f := NewFooter(testTheme(), 80)
		f.SetTokens(tc.used, tc.max)
		out := f.View()
		if !strings.Contains(out, tc.want) {
			t.Errorf("tokens %d/%d: want %q in output, got %q", tc.used, tc.max, tc.want, out)
		}
	}
}

func TestFooter_ThinkingOff(t *testing.T) {
	f := NewFooter(testTheme(), 80)
	f.SetModel("glm-5.2")
	f.SetThinkingLevel("off")
	out := f.View()
	if strings.Contains(out, "*") {
		t.Errorf("thinking 'off' should suppress '* <level>': %q", out)
	}

	// Empty thinking is the same as 'off'.
	f2 := NewFooter(testTheme(), 80)
	f2.SetModel("glm-5.2")
	f2.SetThinkingLevel("")
	out2 := f2.View()
	if strings.Contains(out2, "*") {
		t.Errorf("empty thinking should suppress '* <level>': %q", out2)
	}
}

func TestFooter_HomeSubstitution(t *testing.T) {
	t.Setenv("HOME", "/home/user")
	f := NewFooter(testTheme(), 80)
	f.SetCwd("/home/user/dev/project")
	out := f.View()
	if !strings.Contains(out, "~/dev/project") {
		t.Errorf("expected $HOME substituted with '~': %q", out)
	}
	// $HOME exactly (no trailing slash) renders as "~" alone.
	f2 := NewFooter(testTheme(), 80)
	f2.SetCwd("/home/user")
	out2 := f2.View()
	if !strings.Contains(out2, "~") {
		t.Errorf("expected '~' for $HOME exactly: %q", out2)
	}
}

// --- Input ---

func TestInput_Value(t *testing.T) {
	m := NewInput(testTheme(), 80, 3)
	m.SetValue("hello world")
	if m.Value() != "hello world" {
		t.Errorf("Value = %q, want 'hello world'", m.Value())
	}
}

func TestInput_Reset(t *testing.T) {
	m := NewInput(testTheme(), 80, 3)
	m.SetValue("hello")
	m.Reset()
	if m.Value() != "" {
		t.Errorf("Value after reset = %q, want empty", m.Value())
	}
}

func TestInput_History(t *testing.T) {
	m := NewInput(testTheme(), 80, 3)
	m.AppendHistory("first")
	m.AppendHistory("second")
	m.AppendHistory("third")

	// Navigate backward.
	m.HistoryPrev()
	if m.Value() != "third" {
		t.Errorf("HistoryPrev: Value = %q, want 'third'", m.Value())
	}
	m.HistoryPrev()
	if m.Value() != "second" {
		t.Errorf("HistoryPrev: Value = %q, want 'second'", m.Value())
	}

	// Navigate forward.
	m.HistoryNext()
	if m.Value() != "third" {
		t.Errorf("HistoryNext: Value = %q, want 'third'", m.Value())
	}
	m.HistoryNext()
	// After the last entry, should restore the saved draft (empty).
	if m.Value() != "" {
		t.Errorf("HistoryNext past end: Value = %q, want empty", m.Value())
	}
}

func TestInput_HistoryNoConsecutiveDups(t *testing.T) {
	m := NewInput(testTheme(), 80, 3)
	m.AppendHistory("hello")
	m.AppendHistory("hello")
	if len(m.history) != 1 {
		t.Errorf("history = %v, want 1 entry", m.history)
	}
}

func TestInput_Autocomplete(t *testing.T) {
	m := NewInput(testTheme(), 80, 3)
	m.SetCommands([]string{"/help", "/fork", "/quit"})

	m.SetValue("/h")
	m.AutocompleteStart()
	if !m.AutocompleteActive() {
		t.Error("autocomplete should be active for '/h'")
	}
	if len(m.autoMatch) != 1 || m.autoMatch[0] != "/help" {
		t.Errorf("autoMatch = %v, want ['/help']", m.autoMatch)
	}

	m.AutocompleteAccept()
	if m.Value() != "/help " {
		t.Errorf("Value after accept = %q, want '/help '", m.Value())
	}
	if m.AutocompleteActive() {
		t.Error("autocomplete should be inactive after accept")
	}
}

func TestInput_AutocompleteCycle(t *testing.T) {
	m := NewInput(testTheme(), 80, 3)
	m.SetCommands([]string{"/help", "/history"})
	m.SetValue("/")
	m.AutocompleteStart()
	if len(m.autoMatch) < 2 {
		t.Fatalf("expected 2+ matches, got %d", len(m.autoMatch))
	}
	m.AutocompleteNext()
	// autoIndex should have wrapped to the next match.
	if m.autoIndex != 1 {
		t.Errorf("autoIndex = %d, want 1", m.autoIndex)
	}
}

// --- TreeView ---

func TestTreeView_Navigation(t *testing.T) {
	tv := NewTreeView(testTheme(), 80, 24)
	nodes := []*TreeNode{
		{ID: "1", Label: "root", Depth: 0},
		{ID: "2", Label: "child", Depth: 1},
	}
	tv.SetNodes(nodes)
	tv.Show()
	tv.CursorDown()
	sel := tv.Selected()
	if sel == nil || sel.ID != "2" {
		t.Errorf("Selected = %v, want ID '2'", sel)
	}
}

func TestTreeView_ShowHide(t *testing.T) {
	tv := NewTreeView(testTheme(), 80, 24)
	if tv.Visible() {
		t.Error("tree should start hidden")
	}
	tv.Show()
	if !tv.Visible() {
		t.Error("tree should be visible after Show()")
	}
	tv.Hide()
	if tv.Visible() {
		t.Error("tree should be hidden after Hide()")
	}
}

func TestTreeView_CursorNavigation(t *testing.T) {
	tv := NewTreeView(testTheme(), 80, 24)
	nodes := []*TreeNode{
		{ID: "1", Label: "first", Depth: 0},
		{ID: "2", Label: "second", Depth: 0},
		{ID: "3", Label: "third", Depth: 0},
	}
	tv.SetNodes(nodes)
	tv.Show()

	if tv.cursor != 0 {
		t.Errorf("cursor = %d, want 0", tv.cursor)
	}
	tv.CursorDown()
	if tv.cursor != 1 {
		t.Errorf("cursor after down = %d, want 1", tv.cursor)
	}
	tv.CursorUp()
	if tv.cursor != 0 {
		t.Errorf("cursor after up = %d, want 0", tv.cursor)
	}
}

func TestTreeView_Selected(t *testing.T) {
	tv := NewTreeView(testTheme(), 80, 24)
	nodes := []*TreeNode{
		{ID: "1", Label: "first", Depth: 0},
		{ID: "2", Label: "second", Depth: 0},
	}
	tv.SetNodes(nodes)
	tv.Show()
	tv.CursorDown()

	sel := tv.Selected()
	if sel == nil {
		t.Fatal("Selected returned nil")
	}
	if sel.ID != "2" {
		t.Errorf("Selected.ID = %q, want '2'", sel.ID)
	}
}

func TestTreeView_ViewRendersNodes(t *testing.T) {
	tv := NewTreeView(testTheme(), 80, 24)
	nodes := []*TreeNode{
		{ID: "root", Label: "Session Start", Depth: 0, Active: true},
		{ID: "msg1", Label: "User: hello", Depth: 1},
	}
	tv.SetNodes(nodes)
	tv.Show()
	out := tv.View()
	if out == "" {
		t.Error("View returned empty string")
	}
	if !strings.Contains(out, "Session Start") {
		t.Errorf("View missing 'Session Start': %q", out)
	}
	if !strings.Contains(out, "User: hello") {
		t.Errorf("View missing 'User: hello': %q", out)
	}
}

func TestTreeView_HiddenReturnsEmpty(t *testing.T) {
	tv := NewTreeView(testTheme(), 80, 24)
	tv.SetNodes([]*TreeNode{
		{ID: "1", Label: "test", Depth: 0},
	})
	if tv.View() != "" {
		t.Error("hidden tree should return empty string")
	}
}

func TestTreeView_FlattenNestedChildren(t *testing.T) {
	nodes := []*TreeNode{
		{ID: "1", Depth: 0, Children: []*TreeNode{
			{ID: "2", Depth: 1},
		}},
		{ID: "3", Depth: 0},
	}
	flat := flatten(nodes)
	if len(flat) != 3 {
		t.Errorf("flatten: len = %d, want 3", len(flat))
	}
}
