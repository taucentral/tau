package slash

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/llm"
	"github.com/taucentral/tau/internal/state"
)

func TestClearCommand_Name(t *testing.T) {
	if got := (clearCommand{}).Name(); got != "/clear" {
		t.Errorf("Name = %q, want %q", got, "/clear")
	}
}

func TestClearCommand_ShortHelp_MentionsResetAndCheckout(t *testing.T) {
	help := (clearCommand{}).ShortHelp()
	if !strings.Contains(help, "Reset") {
		t.Errorf("ShortHelp = %q; want substring \"Reset\"", help)
	}
	if !strings.Contains(strings.ToLower(help), "/checkout") {
		t.Errorf("ShortHelp = %q; want substring \"/checkout\" (recovery hint)", help)
	}
}

// TestClearCommand_NilSession verifies the guard: /clear without a
// session returns a descriptive error rather than panicking on the
// nil-pointer dereference inside Runtime().
func TestClearCommand_NilSession(t *testing.T) {
	_, err := (clearCommand{}).Execute(context.Background(), "", nil)
	if err == nil {
		t.Fatal("/clear with nil session: err = nil, want error")
	}
	if !strings.Contains(err.Error(), "nil") {
		t.Errorf("/clear nil-session err = %q; want substring \"nil\"", err.Error())
	}
}

// TestClearCommand_AppendsClearMarker verifies the core contract: after
// /clear, the new leaf is a KindClearMarker entry whose parent is the
// pre-clear leaf. The pre-clear leaf remains in the tree (recovery path).
func TestClearCommand_AppendsClearMarker(t *testing.T) {
	sess := newTestSession(t)
	// Seed a message so we have a non-trivial pre-clear leaf.
	if _, err := sess.Runtime().State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "pre-clear"}}},
	}); err != nil {
		t.Fatalf("Append pre-clear: %v", err)
	}
	oldLeaf := sess.Runtime().State.LeafID()

	out, err := (clearCommand{}).Execute(context.Background(), "", sess.AsCommandSession())
	if !errors.Is(err, ErrContextReset) {
		t.Errorf("err = %v, want ErrContextReset", err)
	}
	if !strings.Contains(out, oldLeaf) {
		t.Errorf("output = %q; want old leaf %q in recovery hint", out, oldLeaf)
	}

	newLeaf := sess.Runtime().State.LeafID()
	if newLeaf == oldLeaf {
		t.Fatal("LeafID unchanged after /clear; expected new ClearMarker leaf")
	}

	// The new leaf's payload must be a ClearMarkerPayload whose parent
	// is the pre-clear leaf.
	tree, err := sess.Runtime().State.Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	entry, ok := tree.Get(newLeaf)
	if !ok {
		t.Fatalf("new leaf %q not in tree", newLeaf)
	}
	if entry.Kind != state.KindClearMarker {
		t.Errorf("new leaf Kind = %q, want %q", entry.Kind, state.KindClearMarker)
	}
	if entry.ParentID != oldLeaf {
		t.Errorf("ClearMarker ParentID = %q, want old leaf %q", entry.ParentID, oldLeaf)
	}

	// Old leaf must still be in the tree (recovery path).
	if _, ok := tree.Get(oldLeaf); !ok {
		t.Errorf("old leaf %q missing from tree after /clear; recovery invariant broken", oldLeaf)
	}
}

// TestClearCommand_BuildContextEmptyAfterClear verifies the model-facing
// effect: after /clear, BuildContext returns no messages. The model
// starts with a blank slate.
func TestClearCommand_BuildContextEmptyAfterClear(t *testing.T) {
	sess := newTestSession(t)
	// Seed pre-clear messages.
	if _, err := sess.Runtime().State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "pre-clear user"}}},
	}); err != nil {
		t.Fatalf("Append user: %v", err)
	}
	if _, err := sess.Runtime().State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextContent{Text: "pre-clear assistant"}}},
	}); err != nil {
		t.Fatalf("Append assistant: %v", err)
	}

	if _, err := (clearCommand{}).Execute(context.Background(), "", sess.AsCommandSession()); err != nil && !errors.Is(err, ErrContextReset) {
		t.Fatalf("/clear: %v", err)
	}

	ctx, err := sess.Runtime().State.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("post-clear BuildContext: %v", err)
	}
	if len(ctx.Messages) != 0 {
		t.Errorf("post-clear Messages len = %d, want 0; got %+v", len(ctx.Messages), ctx.Messages)
	}
}

// TestClearCommand_PostClearAppendsVisible verifies that messages
// appended AFTER the /clear are visible to BuildContext (they live in
// the ClearMarker's subtree, not past the barrier).
func TestClearCommand_PostClearAppendsVisible(t *testing.T) {
	sess := newTestSession(t)
	if _, err := sess.Runtime().State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "pre-clear"}}},
	}); err != nil {
		t.Fatalf("Append pre-clear: %v", err)
	}

	if _, err := (clearCommand{}).Execute(context.Background(), "", sess.AsCommandSession()); err != nil && !errors.Is(err, ErrContextReset) {
		t.Fatalf("/clear: %v", err)
	}

	// Append post-clear messages.
	if _, err := sess.Runtime().State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "post-clear user"}}},
	}); err != nil {
		t.Fatalf("Append post-clear: %v", err)
	}

	ctx, err := sess.Runtime().State.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext: %v", err)
	}
	if len(ctx.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1 (only post-clear user)", len(ctx.Messages))
	}
	tc, _ := ctx.Messages[0].Content[0].(llm.TextContent)
	if tc.Text != "post-clear user" {
		t.Errorf("Messages[0] text = %q, want %q", tc.Text, "post-clear user")
	}
}

// TestClearCommand_RecoverableViaCheckout verifies the recovery path:
// after /clear, SetLeaf(oldLeaf) restores the original pre-clear context
// because walking from oldLeaf never visits the ClearMarker (it's a
// descendant, not an ancestor).
func TestClearCommand_RecoverableViaCheckout(t *testing.T) {
	sess := newTestSession(t)
	if _, err := sess.Runtime().State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "pre-clear user"}}},
	}); err != nil {
		t.Fatalf("Append pre-clear: %v", err)
	}
	oldLeaf := sess.Runtime().State.LeafID()

	if _, err := (clearCommand{}).Execute(context.Background(), "", sess.AsCommandSession()); err != nil && !errors.Is(err, ErrContextReset) {
		t.Fatalf("/clear: %v", err)
	}

	// Sanity: post-clear context is empty.
	postCtx, err := sess.Runtime().State.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("post-clear BuildContext: %v", err)
	}
	if len(postCtx.Messages) != 0 {
		t.Fatalf("post-clear Messages len = %d, want 0", len(postCtx.Messages))
	}

	// Recovery: hop back to the pre-clear leaf.
	if err := sess.Runtime().State.SetLeaf(oldLeaf); err != nil {
		t.Fatalf("SetLeaf(oldLeaf): %v", err)
	}
	recovered, err := sess.Runtime().State.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("recovered BuildContext: %v", err)
	}
	if len(recovered.Messages) != 1 {
		t.Errorf("recovered Messages len = %d, want 1 (pre-clear user)", len(recovered.Messages))
	}
	tc, _ := recovered.Messages[0].Content[0].(llm.TextContent)
	if tc.Text != "pre-clear user" {
		t.Errorf("recovered Messages[0] = %q, want %q", tc.Text, "pre-clear user")
	}
}

// TestClearCommand_SuccessMessageFormat verifies the user-facing message
// includes the recovery hint mentioning /checkout and the old leaf ID.
func TestClearCommand_SuccessMessageFormat(t *testing.T) {
	sess := newTestSession(t)
	if _, err := sess.Runtime().State.Append(state.Entry{
		Kind:    state.KindMessage,
		Payload: state.MessagePayload{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextContent{Text: "seed"}}},
	}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	oldLeaf := sess.Runtime().State.LeafID()

	out, _ := (clearCommand{}).Execute(context.Background(), "", sess.AsCommandSession())
	for _, want := range []string{"Cleared", "/checkout", oldLeaf} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q: %q", want, out)
		}
	}
}
