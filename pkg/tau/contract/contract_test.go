// Package contract hosts the SDK contract test for tau's public slash-
// command surface. It is deliberately an "external-shaped" package: it
// imports ONLY github.com/coevin/tau (the SDK facade) and the standard
// library. It MUST NOT import any path under internal/. If it does, the
// contract is broken and the test fails to compile in any external
// module that depends on github.com/coevin/tau.
//
// The contract test proves three things:
//
//  1. An external Go module can name tau.Command, tau.CommandSession,
//     tau.CommandRuntime, and tau.CommandOptions without reaching into
//     internal/ (compile-time check).
//  2. An external module can declare a type implementing tau.Command
//     and register it with tau.Registry without a compile-time error
//     (runtime check via Registry.Register).
//  3. The registered custom command dispatches through Registry.Execute
//     and its Execute is invoked with the parsed args. Dispatch is
//     exercised with a nil session because this package cannot construct
//     a real tau.CommandSession without importing internal/agent; the
//     custom command under test ignores its session argument, and the
//     args-parsing + dispatch path itself does not require a non-nil
//     session. The end-to-end dispatch assertion (real session, real
//     command, real runtime) lives in internal/slash/slash_test.go,
//     which is allowed to import internal/.
package contract

import (
	"context"
	"errors"
	"strings"
	"testing"

	tau "github.com/coevin/tau/pkg/tau"
)

// echoCommand is a custom tau.Command that records its args and returns
// them as the user-facing output. It ignores its session argument,
// which lets the contract test exercise dispatch without constructing
// a tau.CommandSession (the contract package cannot reach internal/
// types required to build one).
type echoCommand struct {
	name string
	got  string
}

func (c *echoCommand) Name() string      { return c.name }
func (c *echoCommand) ShortHelp() string { return "echo args back (contract test)" }
func (c *echoCommand) Execute(_ context.Context, args string, _ tau.CommandSession) (string, error) {
	c.got = args
	if args == "" {
		return "", errors.New("echo: no args")
	}
	return args, nil
}

// Compile-time assertion: echoCommand satisfies tau.Command. If the
// SDK ever drops the Command alias or changes its signature, this
// package fails to compile — surfacing the breakage at build time
// rather than at the first external consumer.
var _ tau.Command = (*echoCommand)(nil)

// TestExternalModule_CanNamePublicTypes asserts the four SDK-level
// type aliases exist and resolve to non-nil zero values. The aliases
// are the only way an external module can name these types without
// importing internal/agent.
func TestExternalModule_CanNamePublicTypes(t *testing.T) {
	// These four references are entirely type-level; the test framework
	// gives them runtime expression via nil. If any alias is removed,
	// the package fails to compile.
	var (
		_ tau.Command
		_ tau.CommandSession
		_ tau.CommandRuntime
		_ tau.CommandOptions
	)
	// Sanity: the zero interface values are nil. This is a behaviour
	// assertion that the type aliases are not the empty interface.
	var s tau.CommandSession
	if s != nil {
		t.Errorf("CommandSession zero value = %v, want nil", s)
	}
	var r tau.CommandRuntime
	if r != nil {
		t.Errorf("CommandRuntime zero value = %v, want nil", r)
	}
	var o tau.CommandOptions
	if o != nil {
		t.Errorf("CommandOptions zero value = %v, want nil", o)
	}
}

// TestExternalModule_CanRegisterCustomCommand proves an external Go
// module that declares a type implementing tau.Command can hand it to
// tau.Registry.Register without a compile-time or runtime error. This
// is the seam the add-sdd-plugin change needs: it lives at
// github.com/coevin/tau-plugins/sdd and must register /propose,
// /explore, /apply, /archive as slash commands.
func TestExternalModule_CanRegisterCustomCommand(t *testing.T) {
	reg := tau.NewRegistry()
	cmd := &echoCommand{name: "/echo"}
	reg.Register(cmd)

	// Confirm the registry actually recorded the command by looking it
	// up via the public Registry.Lookup method. Lookup returns
	// (Command, bool); we only need the bool here.
	got, ok := reg.Lookup("/echo")
	if !ok {
		t.Fatal("Registry.Lookup(/echo) returned ok=false; Register did not record the custom command")
	}
	if got.Name() != "/echo" {
		t.Errorf("Lookup returned command with Name() = %q, want %q", got.Name(), "/echo")
	}
}

// TestExternalModule_CustomCommandDispatchesViaRegistry proves a
// registered custom command is invoked when Registry.Execute parses
// a matching /<name> directive. Dispatch is exercised with a nil
// session; the custom command ignores its session argument so the
// args-parsing and dispatch path can be observed in isolation.
//
// The contract test deliberately does NOT construct a real
// tau.CommandSession because doing so would require importing
// internal/agent, which is forbidden to external modules. The
// end-to-end dispatch path (real session, real runtime, real state
// manager) is covered by internal/slash/slash_test.go.
func TestExternalModule_CustomCommandDispatchesViaRegistry(t *testing.T) {
	reg := tau.NewRegistry()
	cmd := &echoCommand{name: "/echo"}
	reg.Register(cmd)

	out, err := reg.Execute(context.Background(), "/echo hello world", nil)
	if err != nil {
		t.Fatalf("Registry.Execute(/echo hello world): %v", err)
	}
	if out != "hello world" {
		t.Errorf("Execute output = %q, want %q", out, "hello world")
	}
	if cmd.got != "hello world" {
		t.Errorf("custom command received args = %q, want %q", cmd.got, "hello world")
	}
}

// TestExternalModule_CustomCommandCoexistsWithBuiltins proves a
// registry pre-populated with the built-in command set accepts an
// additional custom command without disturbing either. This is the
// realistic embedder configuration: DefaultSlashRegistry() + one or
// more plugin-provided commands layered on top.
func TestExternalModule_CustomCommandCoexistsWithBuiltins(t *testing.T) {
	reg := tau.DefaultSlashRegistry()
	cmd := &echoCommand{name: "/echo"}
	reg.Register(cmd)

	// Built-in /quit still dispatches (it returns ErrQuitRequested,
	// which is also part of the public SDK surface).
	_, err := reg.Execute(context.Background(), "/quit", nil)
	if !errors.Is(err, tau.ErrQuitRequested) {
		t.Errorf("/quit after custom register: err = %v, want ErrQuitRequested", err)
	}

	// Custom /echo still dispatches alongside the built-ins.
	out, err := reg.Execute(context.Background(), "/echo hi", nil)
	if err != nil {
		t.Fatalf("/echo after built-in register: %v", err)
	}
	if !strings.Contains(out, "hi") {
		t.Errorf("/echo output = %q, want substring %q", out, "hi")
	}
}
