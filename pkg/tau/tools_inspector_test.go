// tools_inspector_test.go — verifies AgentSession.Tools().
//
// (a) Tools() returns sorted names.
// (b) mutating the returned slice does not affect a subsequent Tools() call.

package tau

import (
	"context"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/taucentral/tau/internal/config"
	"github.com/taucentral/tau/internal/fauxprovider"
	"github.com/taucentral/tau/internal/tools"
)

func TestToolsReturnsSortedNames(t *testing.T) {
	opts := Options{
		Cwd:       t.TempDir(),
		Model:     "faux",
		LLMClient: fauxprovider.NewWithResponse("ok"),
		Tools: []HeadlessTool{
			tools.NewReadTool(tools.OSReadOperations{}),
			tools.NewLSTool(tools.OSLSOperations{}),
			tools.NewBashTool(tools.OSBashOperations{}),
			tools.NewEditTool(tools.OSEditOperations{}),
			tools.NewWriteTool(tools.OSWriteOperations{}),
			tools.NewGrepTool(tools.OSGrepOperations{}),
			tools.NewFindTool(tools.OSFindOperations{}),
		},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
	}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	got := sess.Tools()
	// Sort a copy and confirm the returned slice is already sorted.
	sorted := append([]string(nil), got...)
	sort.Strings(sorted)
	if !reflect.DeepEqual(got, sorted) {
		t.Errorf("Tools() not sorted: got %v, want %v", got, sorted)
	}

	// All seven built-in names should appear.
	wantNames := []string{"read", "bash", "edit", "write", "grep", "find", "ls"}
	gotJoined := strings.Join(got, ",")
	for _, w := range wantNames {
		if !strings.Contains(gotJoined, w) {
			t.Errorf("Tools() missing %q in %v", w, got)
		}
	}
}

func TestToolsReturnsFreshCopy(t *testing.T) {
	opts := Options{
		Cwd:           t.TempDir(),
		Model:         "faux",
		LLMClient:     fauxprovider.NewWithResponse("ok"),
		Tools:         []HeadlessTool{tools.NewReadTool(tools.OSReadOperations{})},
		Settings:      config.DefaultSettings(),
		ContextWindow: 200000,
	}
	sess, err := CreateAgentSession(context.Background(), opts)
	if err != nil {
		t.Fatalf("CreateAgentSession: %v", err)
	}
	t.Cleanup(func() { sess.Shutdown(context.Background()) })

	first := sess.Tools()
	if len(first) == 0 {
		t.Fatal("first Tools() returned empty slice")
	}
	// Mutate the returned slice.
	first[0] = "mutated"
	if len(first) > 1 {
		first = append(first, "extra")
	}

	second := sess.Tools()
	// The second call must not reflect the mutation.
	for _, name := range second {
		if name == "mutated" || name == "extra" {
			t.Errorf("Tools() mutation leaked into next call: %v contains %q", second, name)
		}
	}
	// And the first element should now be back to "read" (the only tool).
	if len(second) != 1 || second[0] != "read" {
		t.Errorf("Tools() = %v, want [read]", second)
	}
}
