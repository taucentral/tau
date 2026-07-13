// Command tau-plugin-minimal is a minimal plugin binary used by
// internal/plugins integration tests. It advertises three tools
// (minimal.echo, minimal.fail, minimal.log) and implements the proto.Plugin
// service by way of the plugins.PluginAdapter helper.
//
// The minimal.log tool exercises plugin→host RPCs by calling Host.Log on
// the adapter's HostClient. Tests use it to verify the host-side Host
// service registrar is wired into the broker.
//
// This binary is built by the test via `go build` and run as a real
// subprocess; it is never installed or shipped.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/taucentral/tau/internal/plugins"
	tauproto "github.com/taucentral/tau/internal/proto"
)

// echoServer implements proto.PluginServer. Handshake returns the host
// protocol version; ListTools streams a fixed tool list; Execute handles
// each by name; Shutdown is a no-op (the host kills the process via
// go-plugin after grace).
//
// The adapter field is wired in main() so the minimal.log handler can
// reach the host's Host service via HostClient().
type echoServer struct {
	tauproto.UnimplementedPluginServer

	adapter *plugins.PluginAdapter
}

func (s echoServer) Handshake(_ context.Context, _ *tauproto.Empty) (*tauproto.ProtocolVersion, error) {
	return &tauproto.ProtocolVersion{Version: int32(plugins.ProtocolVersion)}, nil
}

// ListTools advertises fully-qualified tool names. The host stores the
// advertised name verbatim (so "minimal.echo" becomes the tool's Name());
// the wire RPC back to the plugin uses just the local portion ("echo"),
// which is what the switch in Execute matches.
func (echoServer) ListTools(_ *tauproto.Empty, stream tauproto.Plugin_ListToolsServer) error {
	tools := []*tauproto.ToolSchema{
		{
			Name:        "minimal.echo",
			Description: "Echo back the request arguments as a text block.",
			JsonSchema:  `{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`,
		},
		{
			Name:        "minimal.fail",
			Description: "Return an error result with the supplied text.",
			JsonSchema:  `{"type":"object","properties":{"text":{"type":"string"}}}`,
		},
		{
			Name:        "minimal.log",
			Description: "Forward the supplied text to the host's Host.Log RPC and echo back acknowledgement.",
			JsonSchema:  `{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`,
		},
	}
	for _, t := range tools {
		if err := stream.Send(t); err != nil {
			return err
		}
	}
	return nil
}

func (s echoServer) Execute(ctx context.Context, call *tauproto.ToolCall) (*tauproto.ToolResult, error) {
	switch call.GetName() {
	case "echo":
		return &tauproto.ToolResult{
			CallId: call.GetId(),
			Content: []*tauproto.ContentBlock{
				{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: "echo: " + string(call.GetArgs())}}},
			},
		}, nil
	case "fail":
		return &tauproto.ToolResult{
			CallId:  call.GetId(),
			IsError: true,
			Content: []*tauproto.ContentBlock{
				{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: "deliberate failure"}}},
			},
		}, nil
	case "log":
		return s.handleLog(ctx, call)
	case "panic":
		panic("integration test: deliberate panic")
	}
	return nil, fmt.Errorf("unknown tool: %s", call.GetName())
}

// handleLog forwards the request args to Host.Log at INFO level. The
// HostClient comes from the adapter, which lazy-dials the host's brokered
// Host service on first use. Returns an error result when the host service
// is unreachable so the test surfaces a clear failure rather than a nil
// dereference.
func (s echoServer) handleLog(ctx context.Context, call *tauproto.ToolCall) (*tauproto.ToolResult, error) {
	if s.adapter == nil {
		return &tauproto.ToolResult{
			CallId:  call.GetId(),
			IsError: true,
			Content: []*tauproto.ContentBlock{
				{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: "minimal.log: adapter not wired"}}},
			},
		}, nil
	}
	hostCli := s.adapter.HostClient()
	if hostCli == nil {
		return &tauproto.ToolResult{
			CallId:  call.GetId(),
			IsError: true,
			Content: []*tauproto.ContentBlock{
				{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: "minimal.log: host Host service unavailable"}}},
			},
		}, nil
	}
	if _, err := hostCli.Log(ctx, &tauproto.LogRequest{
		Level:   tauproto.LogRequest_LEVEL_INFO,
		Message: string(call.GetArgs()),
	}); err != nil {
		return &tauproto.ToolResult{
			CallId:  call.GetId(),
			IsError: true,
			Content: []*tauproto.ContentBlock{
				{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: fmt.Sprintf("minimal.log: Host.Log RPC failed: %v", err)}}},
			},
		}, nil
	}
	return &tauproto.ToolResult{
		CallId: call.GetId(),
		Content: []*tauproto.ContentBlock{
			{Variant: &tauproto.ContentBlock_Text{Text: &tauproto.TextBlock{Text: "logged: " + string(call.GetArgs())}}},
		},
	}, nil
}

func (echoServer) Shutdown(_ context.Context, _ *tauproto.Empty) (*tauproto.Empty, error) {
	return &tauproto.Empty{}, nil
}

func main() {
	if _, ok := os.LookupEnv(plugins.MagicCookieKey); !ok {
		// Refuse to start without the magic cookie. This prevents the
		// binary from being mistaken for a normal CLI tool.
		fmt.Fprintln(os.Stderr, "minimal plugin: missing magic cookie; refusing to start")
		os.Exit(ExitNoCookie)
	}
	// Construct the adapter first, then build the server with the adapter
	// pointer wired in, then assign the server onto the adapter. Order
	// matters: PluginAdapter.Server is a value (tauproto.PluginServer),
	// so the echoServer registered with go-plugin is a copy. Setting
	// adapter onto server BEFORE adapter.Server = server guarantees the
	// copy carried through PluginMap()→GRPCPlugin.ServerImpl→RegisterPluginServer
	// holds the correct *PluginAdapter pointer for handleLog to reach
	// HostClient().
	adapter := &plugins.PluginAdapter{}
	server := echoServer{adapter: adapter}
	adapter.Server = server
	adapter.Serve()
}

// ExitNoCookie is the exit code when the magic cookie is missing. The
// host's go-plugin runner interprets a non-zero exit as a startup
// failure; we keep a distinct code so the integration test can probe it.
const ExitNoCookie = 2
