// Command tau-plugin-minimal is a minimal plugin binary used by
// internal/plugins integration tests. It advertises two tools
// (minimal.echo and minimal.fail) and implements the proto.Plugin
// service by way of the plugins.PluginAdapter helper.
//
// This binary is built by the test via `go build` and run as a real
// subprocess; it is never installed or shipped.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/coevin/tau/internal/plugins"
	tauproto "github.com/coevin/tau/internal/proto"
)

// echoServer implements proto.PluginServer. Handshake returns the host
// protocol version; ListTools streams a fixed tool list; Execute handles
// each by name; Shutdown is a no-op (the host kills the process via
// go-plugin after grace).
type echoServer struct {
	tauproto.UnimplementedPluginServer
}

func (echoServer) Handshake(_ context.Context, _ *tauproto.Empty) (*tauproto.ProtocolVersion, error) {
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
	}
	for _, t := range tools {
		if err := stream.Send(t); err != nil {
			return err
		}
	}
	return nil
}

func (s echoServer) Execute(_ context.Context, call *tauproto.ToolCall) (*tauproto.ToolResult, error) {
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
	case "panic":
		panic("integration test: deliberate panic")
	}
	return nil, fmt.Errorf("unknown tool: %s", call.GetName())
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
	(&plugins.PluginAdapter{Server: echoServer{}}).Serve()
}

// ExitNoCookie is the exit code when the magic cookie is missing. The
// host's go-plugin runner interprets a non-zero exit as a startup
// failure; we keep a distinct code so the integration test can probe it.
const ExitNoCookie = 2
