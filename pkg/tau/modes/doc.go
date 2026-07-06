// Package modes exposes tau's print and rpc run-mode handlers as a public
// subpackage of the SDK. The handlers drive an agentic session and emit
// either stdout-formatted output (RunPrint) or newline-delimited JSON-RPC
// (RunRPC); embedders building their own CLI or editor integration reuse
// these without reimplementing the formatting.
//
// The subpackage imports only github.com/taucentral/tau/pkg/tau (the SDK
// facade). It never imports any internal/* path, so any external module
// can pull it in without paying for internal-package coupling. The
// interactive mode handler (which depends on the TUI, and therefore on
// charmbracelet/*) intentionally lives in the tau-cli module rather than
// here; see openspec/changes/split-tui-into-tau-cli design.md §D1.1-D1.4
// for the rationale.
//
// RunPrint and RunRPC signatures are stable across future minor releases
// of tau. Behaviour changes are additive (new optional fields on
// PrintOptions / RPCOptions) rather than signature changes; the
// stability contract is enforced by pkg/tau/contract/contract_test.go.
package modes
