// Package components — treeview.go — branch tree viewer.
//
// The tree viewer renders the session's state-tree DAG as an ASCII
// tree. The user can navigate with arrow keys and press Enter to
// checkout a selected entry (which moves the active leaf pointer).
package components

import (
	"fmt"
	"strings"
)

// TreeNode is one entry in the rendered tree.
type TreeNode struct {
	ID       string
	Label    string
	Kind     string // "message", "branch", "compact", "root"
	Depth    int
	Active   bool
	Children []*TreeNode
}

// TreeView renders a session DAG as a navigable ASCII tree.
type TreeView struct {
	theme   Theme
	width   int
	height  int
	nodes   []*TreeNode
	flat    []*TreeNode
	cursor  int
	visible bool
}

// NewTreeView returns a hidden tree view.
func NewTreeView(theme Theme, width, height int) *TreeView {
	return &TreeView{
		theme:  theme,
		width:  width,
		height: height,
	}
}

// SetDimensions updates the tree view size.
func (tv *TreeView) SetDimensions(width, height int) {
	tv.width = width
	tv.height = height
}

// SetNodes replaces the tree data and flattens it for navigation.
func (tv *TreeView) SetNodes(nodes []*TreeNode) {
	tv.nodes = nodes
	tv.flat = flatten(nodes)
	// Clamp cursor.
	if tv.cursor >= len(tv.flat) {
		tv.cursor = len(tv.flat) - 1
	}
	if tv.cursor < 0 {
		tv.cursor = 0
	}
}

// Visible reports whether the tree view is shown.
func (tv *TreeView) Visible() bool { return tv.visible }

// Show makes the tree view visible and positions the cursor on the
// active node.
func (tv *TreeView) Show() {
	tv.visible = true
	// Find the active node.
	for i, n := range tv.flat {
		if n.Active {
			tv.cursor = i
			return
		}
	}
	tv.cursor = 0
}

// Hide removes the tree view from display.
func (tv *TreeView) Hide() { tv.visible = false }

// CursorUp moves the selection up.
func (tv *TreeView) CursorUp() {
	if tv.cursor > 0 {
		tv.cursor--
	}
}

// CursorDown moves the selection down.
func (tv *TreeView) CursorDown() {
	if tv.cursor < len(tv.flat)-1 {
		tv.cursor++
	}
}

// Selected returns the currently selected node, or nil when empty.
func (tv *TreeView) Selected() *TreeNode {
	if tv.cursor < 0 || tv.cursor >= len(tv.flat) {
		return nil
	}
	return tv.flat[tv.cursor]
}

// View renders the tree. Returns an empty string when not visible.
func (tv *TreeView) View() string {
	if !tv.visible || len(tv.flat) == 0 {
		return ""
	}
	var b strings.Builder
	for i, node := range tv.flat {
		// Indent by depth.
		prefix := strings.Repeat("  ", node.Depth)
		// Tree connector.
		connector := "•"
		// Active marker.
		marker := " "
		if node.Active {
			marker = "▸"
		}
		// Cursor marker.
		cursor := " "
		if i == tv.cursor {
			cursor = "▶"
		}
		// Label.
		label := node.Label
		if label == "" {
			label = node.ID
		}
		// Truncate to width.
		maxLabel := tv.width - len(prefix) - 4
		if maxLabel > 0 && len(label) > maxLabel {
			label = label[:maxLabel-1] + "…"
		}

		// Colourise.
		line := fmt.Sprintf("%s%s%s %s %s", cursor, prefix, marker, connector, label)
		if node.Active {
			line = tv.theme.AccentStyle().Render(line)
		} else if i == tv.cursor {
			line = tv.theme.PrimaryStyle().Render(line)
		} else {
			line = tv.theme.MutedStyle().Render(line)
		}
		b.WriteString(line)
		b.WriteString("\n")
	}
	return b.String()
}

// flatten walks the tree into a depth-ordered slice.
func flatten(nodes []*TreeNode) []*TreeNode {
	var out []*TreeNode
	for _, n := range nodes {
		out = append(out, n)
		if len(n.Children) > 0 {
			out = append(out, flatten(n.Children)...)
		}
	}
	return out
}
