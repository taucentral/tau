package state

import (
	"errors"
	"testing"
	"time"
)

// buildLinearTree constructs a tree with entries r←a←b←c (root first, leaf
// last) for tests that need a known-good linear path.
func buildLinearTree(t *testing.T) *Tree {
	t.Helper()
	ts := time.Now().UTC()
	entries := []Entry{
		{ID: "r", ParentID: "", Kind: KindSessionHeader, Timestamp: ts, Payload: SessionHeaderPayload{SessionID: "s"}},
		{ID: "a", ParentID: "r", Kind: KindMessage, Timestamp: ts.Add(time.Second), Payload: MessagePayload{}},
		{ID: "b", ParentID: "a", Kind: KindMessage, Timestamp: ts.Add(2 * time.Second), Payload: MessagePayload{}},
		{ID: "c", ParentID: "b", Kind: KindMessage, Timestamp: ts.Add(3 * time.Second), Payload: MessagePayload{}},
	}
	tree, err := NewTree(entries)
	if err != nil {
		t.Fatalf("NewTree: %v", err)
	}
	return tree
}

func TestNewTree_LinearChain(t *testing.T) {
	tree := buildLinearTree(t)
	if tree.RootID() != "r" {
		t.Errorf("RootID = %q, want r", tree.RootID())
	}
	if tree.Len() != 4 {
		t.Errorf("Len = %d, want 4", tree.Len())
	}
}

func TestNewTree_RejectsEmpty(t *testing.T) {
	_, err := NewTree(nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrTreeInvalid) {
		t.Errorf("err = %v, want wrap of ErrTreeInvalid", err)
	}
}

func TestNewTree_RejectsNoRoot(t *testing.T) {
	// All entries reference a parent that doesn't exist; no root.
	entries := []Entry{
		{ID: "a", ParentID: "missing"},
	}
	_, err := NewTree(entries)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrTreeInvalid) {
		t.Errorf("err = %v, want ErrTreeInvalid", err)
	}
}

func TestNewTree_RejectsMultipleRoots(t *testing.T) {
	entries := []Entry{
		{ID: "r1", ParentID: ""},
		{ID: "r2", ParentID: ""},
	}
	_, err := NewTree(entries)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrTreeInvalid) {
		t.Errorf("err = %v, want ErrTreeInvalid", err)
	}
}

func TestNewTree_RejectsDuplicateIDs(t *testing.T) {
	entries := []Entry{
		{ID: "r", ParentID: ""},
		{ID: "r", ParentID: "r"},
	}
	_, err := NewTree(entries)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrTreeInvalid) {
		t.Errorf("err = %v, want ErrTreeInvalid", err)
	}
}

func TestNewTree_RejectsMissingParent(t *testing.T) {
	entries := []Entry{
		{ID: "r", ParentID: ""},
		{ID: "orphan", ParentID: "ghost"},
	}
	_, err := NewTree(entries)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrTreeInvalid) {
		t.Errorf("err = %v, want ErrTreeInvalid", err)
	}
}

func TestNewTree_DetectsCycle(t *testing.T) {
	// Two-entry cycle with no real root. NewTree's root-count check fails
	// first, so synthesize a real root + a separate cyclic island.
	entries := []Entry{
		{ID: "root", ParentID: ""},
		{ID: "x", ParentID: "y"},
		{ID: "y", ParentID: "x"},
	}
	_, err := NewTree(entries)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrTreeInvalid) {
		t.Errorf("err = %v, want ErrTreeInvalid", err)
	}
}

func TestTree_RootAndRootID(t *testing.T) {
	tree := buildLinearTree(t)
	root := tree.Root()
	if root.ID != "r" {
		t.Errorf("Root().ID = %q, want r", root.ID)
	}
}

func TestTree_Get(t *testing.T) {
	tree := buildLinearTree(t)
	e, ok := tree.Get("b")
	if !ok {
		t.Fatal("Get(b) returned ok=false")
	}
	if e.ID != "b" {
		t.Errorf("Get(b).ID = %q", e.ID)
	}
	if _, ok := tree.Get("nonexistent"); ok {
		t.Error("Get(nonexistent) returned ok=true")
	}
}

func TestTree_IDs(t *testing.T) {
	tree := buildLinearTree(t)
	got := tree.IDs()
	want := []string{"a", "b", "c", "r"}
	if len(got) != len(want) {
		t.Fatalf("IDs = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("IDs[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

func TestTree_Children_InsertionOrder(t *testing.T) {
	// Verify children come back in insertion order, not sorted, so /tree
	// displays siblings in append order.
	ts := time.Now().UTC()
	entries := []Entry{
		{ID: "r", ParentID: "", Timestamp: ts, Payload: SessionHeaderPayload{}},
		{ID: "z-child", ParentID: "r", Timestamp: ts.Add(3 * time.Second)},
		{ID: "a-child", ParentID: "r", Timestamp: ts.Add(1 * time.Second)},
		{ID: "m-child", ParentID: "r", Timestamp: ts.Add(2 * time.Second)},
	}
	tree, err := NewTree(entries)
	if err != nil {
		t.Fatalf("NewTree: %v", err)
	}
	got := tree.Children("r")
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	// Insertion order from the entries slice, NOT timestamp-sorted.
	want := []string{"z-child", "a-child", "m-child"}
	for i, e := range got {
		if e.ID != want[i] {
			t.Errorf("Children[%d].ID = %q, want %q", i, e.ID, want[i])
		}
	}
}

func TestTree_Children_LeafReturnsNil(t *testing.T) {
	tree := buildLinearTree(t)
	if got := tree.Children("c"); got != nil {
		t.Errorf("Children(leaf) = %v, want nil", got)
	}
}

func TestTree_WalkFromLeaf(t *testing.T) {
	tree := buildLinearTree(t)
	got, err := tree.WalkFromLeaf("c")
	if err != nil {
		t.Fatalf("WalkFromLeaf: %v", err)
	}
	want := []string{"c", "b", "a", "r"}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i, e := range got {
		if e.ID != want[i] {
			t.Errorf("WalkFromLeaf[%d].ID = %q, want %q", i, e.ID, want[i])
		}
	}
}

func TestTree_WalkFromLeaf_FromRoot(t *testing.T) {
	tree := buildLinearTree(t)
	got, err := tree.WalkFromLeaf("r")
	if err != nil {
		t.Fatalf("WalkFromLeaf: %v", err)
	}
	if len(got) != 1 || got[0].ID != "r" {
		t.Errorf("got = %+v, want [r]", got)
	}
}

func TestTree_WalkFromLeaf_UnknownID(t *testing.T) {
	tree := buildLinearTree(t)
	_, err := tree.WalkFromLeaf("nonexistent")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrTreeInvalid) {
		t.Errorf("err = %v, want wrap of ErrTreeInvalid", err)
	}
}

func TestTree_Path(t *testing.T) {
	tree := buildLinearTree(t)
	got, err := tree.Path("c")
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	want := []string{"r", "a", "b", "c"}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i, e := range got {
		if e.ID != want[i] {
			t.Errorf("Path[%d].ID = %q, want %q", i, e.ID, want[i])
		}
	}
}

func TestTree_Path_IsReverseOfWalk(t *testing.T) {
	tree := buildLinearTree(t)
	walk, err := tree.WalkFromLeaf("c")
	if err != nil {
		t.Fatalf("WalkFromLeaf: %v", err)
	}
	path, err := tree.Path("c")
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	if len(walk) != len(path) {
		t.Fatalf("len mismatch: walk=%d path=%d", len(walk), len(path))
	}
	for i := range walk {
		j := len(path) - 1 - i
		if walk[i].ID != path[j].ID {
			t.Errorf("walk[%d]=%q should equal path[%d]=%q", i, walk[i].ID, j, path[j].ID)
		}
	}
}

func TestTree_Ancestors(t *testing.T) {
	tree := buildLinearTree(t)
	got, err := tree.Ancestors("c")
	if err != nil {
		t.Fatalf("Ancestors: %v", err)
	}
	want := []string{"b", "a", "r"}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i, id := range got {
		if id != want[i] {
			t.Errorf("Ancestors[%d] = %q, want %q", i, id, want[i])
		}
	}
}

func TestTree_Ancestors_FromRootIsEmpty(t *testing.T) {
	tree := buildLinearTree(t)
	got, err := tree.Ancestors("r")
	if err != nil {
		t.Fatalf("Ancestors: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want empty", got)
	}
}

func TestTree_BranchedStructure(t *testing.T) {
	// Verify the tree handles a real fork: r → a → b
	//                                  └→ c → d
	// WalkFromLeaf(d) returns [d, c, r]; WalkFromLeaf(b) returns [b, a, r].
	ts := time.Now().UTC()
	entries := []Entry{
		{ID: "r", ParentID: "", Timestamp: ts, Payload: SessionHeaderPayload{}},
		{ID: "a", ParentID: "r", Timestamp: ts.Add(time.Second)},
		{ID: "b", ParentID: "a", Timestamp: ts.Add(2 * time.Second)},
		{ID: "c", ParentID: "r", Timestamp: ts.Add(3 * time.Second)},
		{ID: "d", ParentID: "c", Timestamp: ts.Add(4 * time.Second)},
	}
	tree, err := NewTree(entries)
	if err != nil {
		t.Fatalf("NewTree: %v", err)
	}
	kids := tree.Children("r")
	if len(kids) != 2 {
		t.Errorf("r has %d children, want 2", len(kids))
	}
	pathD, err := tree.Path("d")
	if err != nil {
		t.Fatalf("Path(d): %v", err)
	}
	if len(pathD) != 3 || pathD[0].ID != "r" || pathD[1].ID != "c" || pathD[2].ID != "d" {
		t.Errorf("Path(d) = %+v, want [r c d]", pathD)
	}
	pathB, err := tree.Path("b")
	if err != nil {
		t.Fatalf("Path(b): %v", err)
	}
	if len(pathB) != 3 || pathB[0].ID != "r" || pathB[1].ID != "a" || pathB[2].ID != "b" {
		t.Errorf("Path(b) = %+v, want [r a b]", pathB)
	}
}
