package plugins

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/hashicorp/go-plugin"
	"github.com/invopop/jsonschema"
	"google.golang.org/grpc"

	"github.com/taucentral/tau/internal/llm"
	tauproto "github.com/taucentral/tau/internal/proto"
	"github.com/taucentral/tau/internal/tools"
)

// GRPCPlugin is the bridge between go-plugin's plugin.Plugin interface and
// the proto Plugin service. The same type is registered in both the host's
// and the plugin's PluginMap:
//
//   - On the plugin side, ServerImpl returns the proto.PluginServer
//     implementation. The plugin author passes it via NewPluginAdapter.
//   - On the host side, GRPCClient returns a Client that wraps the gRPC
//     connection; the host uses it via HostClient (host.go).
//
// The split exists because go-plugin's net/rpc and gRPC subtypes want one
// factory per "plugin name", but the host and plugin processes occupy
// opposite ends of the same wire.
type GRPCPlugin struct {
	plugin.NetRPCUnsupportedPlugin

	// ServerImpl is the proto.PluginServer the plugin process exposes.
	// nil on the host side.
	ServerImpl tauproto.PluginServer
}

// GRPCServer is called by go-plugin on the plugin process to register the
// gRPC service. Host side never calls this.
func (p *GRPCPlugin) GRPCServer(_ *plugin.GRPCBroker, s *grpc.Server) error {
	if p.ServerImpl == nil {
		return errors.New("plugins: GRPCPlugin has no ServerImpl; cannot serve")
	}
	tauproto.RegisterPluginServer(s, p.ServerImpl)
	return nil
}

// GRPCClient is called by go-plugin on the host to wrap the dispensed
// gRPC client. Plugin side never calls this. The returned value is what
// HostClient.pluginClient holds.
func (p *GRPCPlugin) GRPCClient(_ context.Context, _ *plugin.GRPCBroker, c *grpc.ClientConn) (any, error) {
	return tauproto.NewPluginClient(c), nil
}

// PluginMap is the host-side PluginMap. The plugin side builds its own
// equivalent via NewPluginAdapter (server.go).
var PluginMap = map[string]plugin.Plugin{
	PluginName: &GRPCPlugin{},
}

// PluginTool is the host-side tools.Tool implementation that routes
// Execute to a Plugin.Execute RPC. The Registry treats it identically to
// built-in tools; the agent loop is unaware a tool is remote.
//
// PluginTool instances are constructed by HostClient after ListTools
// returns. They hold an indirect clientProvider so a restarted plugin
// can swap its HostClient without rebuilding every PluginTool instance.
//
// Naming convention: the plugin advertises fully-qualified names like
// "git.status" in its ToolSchema; the host stores the tool under that
// name verbatim. Cross-plugin collisions (two plugins expose the same
// qualified name) are resolved first-wins by the Manager, with a
// diagnostic emitted for the loser.
type PluginTool struct {
	// pluginShortName is the plugin's discovery short name (e.g.
	// "git"). Used for routing and diagnostics; does not appear in
	// Name() (the plugin supplies the full name).
	pluginShortName string
	// toolName is the fully-qualified tool name as the plugin
	// advertised it ("git.status"). Name() returns this verbatim.
	toolName       string
	description    string
	jsonSchema     string
	schemaCache    atomic.Pointer[schemaValue]
	clientProvider func() (tauproto.PluginClient, error)
}

// schemaValue is the cached parsed schema; carried through atomic.Pointer
// so Parameters() is goroutine-safe without a mutex.
type schemaValue struct {
	schema jsonschema.Schema
}

// Compile-time interface check.
var _ tools.Tool = (*PluginTool)(nil)

// NewPluginTool constructs a PluginTool. toolName is the fully-qualified
// name the plugin advertised (e.g. "git.status"); pluginShortName is
// the plugin's discovery name for routing and diagnostics. The
// clientProvider returns the live gRPC client; it is called on every
// Execute so restarts swap transparently.
func NewPluginTool(pluginShortName, toolName, description, jsonSchema string, provider func() (tauproto.PluginClient, error)) *PluginTool {
	return &PluginTool{
		pluginShortName: pluginShortName,
		toolName:        toolName,
		description:     description,
		jsonSchema:      jsonSchema,
		clientProvider:  provider,
	}
}

// Name returns the fully-qualified tool name the plugin advertised.
func (t *PluginTool) Name() string { return t.toolName }

// LocalName returns the un-namespaced name sent to the plugin on the
// wire. Everything after the first "." in the advertised name; if there
// is no ".", the full name is returned as-is. The spec mandates the
// plugin receive the un-namespaced form.
func (t *PluginTool) LocalName() string {
	if idx := strings.IndexByte(t.toolName, '.'); idx >= 0 {
		return t.toolName[idx+1:]
	}
	return t.toolName
}

// PluginShortName returns the plugin's discovery short name. Used by
// the Manager for routing and diagnostics.
func (t *PluginTool) PluginShortName() string { return t.pluginShortName }

// Description returns the plugin-advertised description verbatim.
func (t *PluginTool) Description() string { return t.description }

// Parameters parses the plugin-advertised JSON Schema string into the
// jsonschema.Schema shape. The parse is cached after the first call.
func (t *PluginTool) Parameters() jsonschema.Schema {
	if cached := t.schemaCache.Load(); cached != nil {
		return cached.schema
	}
	parsed := tools.ParseJSONSchema(t.jsonSchema)
	t.schemaCache.Store(&schemaValue{schema: parsed})
	return parsed
}

// Execute routes the call to the plugin via the gRPC client. RPC errors
// (network, plugin crash, context cancellation) are converted to a
// ToolResult with IsError=true; the agent loop treats them as tool
// failures. The model sees the error text and can retry or proceed.
//
// Context cancellation propagates to the plugin via gRPC; the plugin is
// responsible for honoring ctx.Done() promptly.
func (t *PluginTool) Execute(ctx context.Context, call tools.ToolCall) (tools.ToolResult, error) {
	client, err := t.clientProvider()
	if err != nil {
		return tools.NewErrorResult(fmt.Sprintf(
			"%s: plugin unavailable: %v", t.Name(), err,
		)), nil
	}
	pbCall := &tauproto.ToolCall{
		Id:   call.ID,
		Name: t.LocalName(),
		Args: call.Args,
		Cwd:  call.Cwd,
	}
	ctx, cancel := context.WithTimeout(ctx, executeTimeout)
	defer cancel()
	resp, err := client.Execute(ctx, pbCall)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			// Host-level cancellation/deadline should propagate as an
			// error rather than be swallowed into a ToolResult, so the
			// agent loop can react (turn timeout, user abort).
			return tools.ToolResult{}, err
		}
		return tools.NewErrorResult(fmt.Sprintf(
			"%s: plugin RPC failed: %v", t.Name(), err,
		)), nil
	}
	return protoToToolResult(resp), nil
}

// RenderCall renders the invocation for the TUI. Plugin tools render
// their namespaced name plus the raw args JSON, matching the style of
// the built-in tools.
func (t *PluginTool) RenderCall(args json.RawMessage, theme *tools.Theme) string {
	out := theme.Wrap(theme.Primary, t.Name())
	if len(args) == 0 || string(args) == "null" {
		return out
	}
	return out + " " + theme.Wrap(theme.Muted, truncateForDisplay(string(args), 80))
}

// RenderResult renders the result for the TUI. Errors get an "error: "
// prefix; otherwise the content blocks are concatenated via the shared
// tools.RenderContentBlocks helper.
func (t *PluginTool) RenderResult(result tools.ToolResult, theme *tools.Theme) string {
	prefix := ""
	if result.IsError {
		prefix = theme.Wrap(theme.Error, "error: ")
	}
	return prefix + tools.RenderContentBlocks(result.Content, theme)
}

// truncateForDisplay caps a string at maxLen chars with an ellipsis. Used
// only for TUI rendering; never on the result that reaches the model.
func truncateForDisplay(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-1] + "…"
}

// protoToToolResult converts the proto ToolResult into the internal
// tools.ToolResult. Content blocks map 1:1; unknown variants become an
// error-annotated text block so the model sees the failure rather than
// a silent drop.
func protoToToolResult(pb *tauproto.ToolResult) tools.ToolResult {
	out := tools.ToolResult{IsError: pb.GetIsError()}
	for _, blk := range pb.GetContent() {
		switch v := blk.GetVariant().(type) {
		case *tauproto.ContentBlock_Text:
			out.Content = append(out.Content, textBlock(v.Text.GetText()))
		case *tauproto.ContentBlock_Image:
			out.Content = append(out.Content, imageBlock(v.Image.GetMimeType(), v.Image.GetData()))
		case *tauproto.ContentBlock_Json:
			out.Content = append(out.Content, textBlock("```json\n"+v.Json.GetJson()+"\n```"))
		default:
			out.Content = append(out.Content, textBlock("[unrecognized plugin content block]"))
		}
	}
	if len(out.Content) == 0 {
		// Avoid an empty-content ToolResult; the agent loop assumes at
		// least one block. Mirrors tools.NewTextResult behavior.
		out.Content = append(out.Content, textBlock(""))
	}
	return out
}

// executeTimeout caps a single Plugin.Execute RPC. Plugins doing
// long-running work should expose their own progress / cancellation;
// this is the host's safety net for a wedged plugin. Generous default
// (60s) lets a real plugin finish; the agent loop's own turn-level
// timeout still applies above this.
const executeTimeout = 60 * time.Second

// textBlock is the local constructor for llm.TextContent. Exists so
// protoToToolResult doesn't need to spell out llm.TextContent{} at
// every block.
func textBlock(s string) llm.TextContent { return llm.TextContent{Text: s} }

// imageBlock is the local constructor for llm.ImageContent from the
// proto ImageBlock payload. The proto carries raw bytes; the llm shape
// also wants raw bytes (the provider marshaler handles base64 encoding).
func imageBlock(mimeType string, data []byte) llm.ImageContent {
	return llm.ImageContent{MimeType: mimeType, Data: string(data)}
}
