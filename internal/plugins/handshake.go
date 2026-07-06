// Package plugins implements tau's process-isolated plugin host.
//
// The plugin protocol is gRPC over an anonymous Unix-domain socket, with
// hashicorp/go-plugin handling process lifecycle (spawn, heartbeat, crash
// detection, graceful shutdown). A panicking plugin cannot panic the host.
//
// Two sides:
//   - Host side (this package): spawns plugin subprocesses, dispatches tool
//     calls, observes crashes, restarts per the policy.
//   - Plugin side (an external Go binary): calls plugin.Serve with a
//     Server implementation of the proto.Plugin service.
//
// Both sides reference HandshakeConfig and PluginMap. Host-side code uses
// HostClient (host.go) to wrap a single spawned plugin. Manager (manager.go)
// discovers, owns, and recovers HostClients.
package plugins

import (
	"context"
	"fmt"
	"time"

	"github.com/hashicorp/go-plugin"

	tauproto "github.com/taucentral/tau/internal/proto"
)

// ProtocolVersion is the wire-protocol version this host build expects.
// Plugins announce their version via Plugin.Handshake; the host rejects
// mismatches. Bumping this constant is a breaking change and requires a
// tau minor-version bump.
//
// History:
//
//	1 = initial release.
//
// The type is `uint` to satisfy go-plugin's HandshakeConfig.ProtocolVersion
// field; handshake comparisons cast to int32 to match the proto response.
const ProtocolVersion uint = 1

// MagicCookieKey is the environment variable name the host sets before
// spawning a plugin. The plugin MUST verify the value matches MagicCookieValue
// before initializing; this prevents random binaries from being treated as
// plugins when misused as go-plugin servers.
const (
	MagicCookieKey   = "TAU_PLUGIN_MAGIC_COOKIE"
	MagicCookieValue = "tau-plugin-protocol-v1"
)

// HandshakeConfig is the go-plugin handshake shared by host and plugin.
// Both sides reference this exact value.
var HandshakeConfig = plugin.HandshakeConfig{
	ProtocolVersion:  ProtocolVersion,
	MagicCookieKey:   MagicCookieKey,
	MagicCookieValue: MagicCookieValue,
}

// PluginName is the key the host and plugin use to register the gRPC
// service in their respective PluginMaps. go-plugin supports multiple
// plugins per process; tau ships exactly one (the Plugin service) so a
// fixed name suffices.
const PluginName = "tau"

// ErrProtocolMismatch is returned by Handshake when a plugin announces a
// protocol version incompatible with this host's ProtocolVersion.
type ErrProtocolMismatch struct {
	PluginName string
	PluginVer  int32
	HostVer    int32
}

func (e *ErrProtocolMismatch) Error() string {
	return fmt.Sprintf(
		"plugins: %s announced protocol version %d, host requires %d",
		e.PluginName, e.PluginVer, e.HostVer,
	)
}

// expectedHostVer returns the host's protocol version as the int32 the
// proto response carries.
func expectedHostVer() int32 { return int32(ProtocolVersion) }

// errNoHandshake is the typed sentinel returned by Handshake when the
// underlying RPC fails. Distinct from ErrProtocolMismatch so callers can
// branch on "could not even ask" vs "answered wrong".
type errNoHandshake struct {
	pluginName string
	cause      error
}

func (e *errNoHandshake) Error() string {
	return fmt.Sprintf("plugins: %s handshake RPC failed: %v", e.pluginName, e.cause)
}

func (e *errNoHandshake) Unwrap() error { return e.cause }

// handshakeDeadline caps the time the host waits for the plugin's
// Handshake RPC response. go-plugin's own protocol negotiation is short,
// but a misbehaving plugin might accept the gRPC connection and then stall
// before answering Handshake. Five seconds is generous for a localhost
// round-trip.
const handshakeDeadline = 5 * time.Second

// Handshake calls Plugin.Handshake on the client and verifies the version.
// Returns ErrProtocolMismatch on version drift, errNoHandshake on RPC
// failure. Caller is responsible for emitting diagnostics and deciding
// whether to skip-and-continue or fail-fast.
func Handshake(ctx context.Context, pluginName string, client tauproto.PluginClient) error {
	ctx, cancel := context.WithTimeout(ctx, handshakeDeadline)
	defer cancel()
	resp, err := client.Handshake(ctx, &tauproto.Empty{})
	if err != nil {
		return &errNoHandshake{pluginName: pluginName, cause: err}
	}
	if resp.GetVersion() != expectedHostVer() {
		return &ErrProtocolMismatch{
			PluginName: pluginName,
			PluginVer:  resp.GetVersion(),
			HostVer:    expectedHostVer(),
		}
	}
	return nil
}
