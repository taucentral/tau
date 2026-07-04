package slash

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestClsCommand_Name(t *testing.T) {
	if got := (clsCommand{}).Name(); got != "/cls" {
		t.Errorf("Name = %q, want %q", got, "/cls")
	}
}

func TestClsCommand_ShortHelp_MentionsViewport(t *testing.T) {
	help := (clsCommand{}).ShortHelp()
	if !strings.Contains(help, "viewport") {
		t.Errorf("ShortHelp = %q; want substring \"viewport\"", help)
	}
}

func TestClsCommand_ReturnsErrClearScreen(t *testing.T) {
	out, err := (clsCommand{}).Execute(context.Background(), "", nil)
	if !errors.Is(err, ErrClearScreen) {
		t.Errorf("err = %v, want ErrClearScreen", err)
	}
	if out != "" {
		t.Errorf("out = %q, want empty", out)
	}
}

// TestClsCommand_DoesNotMutateState verifies that /cls is purely a UI
// affordance: invoking it does not change the session's leaf pointer or
// append any entry. The state tree is untouched.
func TestClsCommand_DoesNotMutateState(t *testing.T) {
	sess := newTestSession(t)
	leafBefore := sess.Runtime().State.LeafID()
	treeBefore, err := sess.Runtime().State.Tree()
	if err != nil {
		t.Fatalf("Tree before: %v", err)
	}
	lenBefore := treeBefore.Len()

	if _, err := (clsCommand{}).Execute(context.Background(), "", sess.AsCommandSession()); err != nil && !errors.Is(err, ErrClearScreen) {
		t.Fatalf("/cls: %v", err)
	}

	if got := sess.Runtime().State.LeafID(); got != leafBefore {
		t.Errorf("LeafID changed: %q → %q; /cls must not mutate state", leafBefore, got)
	}
	treeAfter, err := sess.Runtime().State.Tree()
	if err != nil {
		t.Fatalf("Tree after: %v", err)
	}
	if treeAfter.Len() != lenBefore {
		t.Errorf("Tree length changed: %d → %d; /cls must not append entries", lenBefore, treeAfter.Len())
	}
}
