// rpc_protocol.go — JSON-RPC 2.0 message shapes used by rpc.go.
//
// The shapes here are deliberately minimal: only the fields the v1
// RPC mode reads or writes. JSON-RPC 2.0 spec:
// https://www.jsonrpc.org/specification

package modes

import "encoding/json"

// rpcRequest is the inbound JSON-RPC message. ID is a RawMessage so
// the server can echo it verbatim regardless of whether the client
// used a number, string, or null id.
type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// rpcResponse is the outbound JSON-RPC message. Exactly one of
// Result, Error, or (Method + Params for notifications) is set.
type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// rpcError is the standard JSON-RPC error object. rpcResponse holds
// it by pointer so the JSON encoder emits the field only when set;
// value-typed rpcError would always serialize as {} on no-error.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}
