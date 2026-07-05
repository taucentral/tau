// lazy_bench_test.go — benchmark for Registry.Schemas at scale.
//
// Task 8.3: with N = 50 stub tools (half eager, half lazy), measure
// Schemas(ctx, signals) latency. Target: under 1ms median on commodity
// hardware.

package tools

import (
	"context"
	"fmt"
	"testing"

	"github.com/invopop/jsonschema"
)

// BenchmarkSchemas_MixedEagerLazy_50 measures Schemas latency with
// 25 eager + 25 lazy stubs. The user message matches roughly half the
// lazy tools' intents, so the heuristic evaluates every trigger path.
//
// Run with:
//
//	go test -bench=. -benchmem ./internal/tools/...
func BenchmarkSchemas_MixedEagerLazy_50(b *testing.B) {
	r := NewRegistry()
	for i := 0; i < 25; i++ {
		r.MustRegister(&eagerStub{
			name: fmt.Sprintf("eager-%02d", i),
			desc: "eager stub",
		})
	}
	for i := 0; i < 25; i++ {
		r.MustRegister(&lazyStub{
			name:   fmt.Sprintf("lazy-%02d", i),
			tag:    ToolTag{Intent: fmt.Sprintf("intent-%02d", i%2)},
			schema: jsonschema.Schema{Type: "object"},
		})
	}

	ctx := context.Background()
	sig := TurnSignals{
		UserMessage:     "intent-00 intent-01 search",
		Mode:            HydrationModeHeuristic,
		RecentUseWindow: 5,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := r.Schemas(ctx, sig)
		if err != nil {
			b.Fatalf("Schemas: %v", err)
		}
	}
}

// BenchmarkSchemas_EagerOnly_50 measures the eager fast path with
// 50 eager stubs. This is the pre-lazy-registration performance
// baseline.
func BenchmarkSchemas_EagerOnly_50(b *testing.B) {
	r := NewRegistry()
	for i := 0; i < 50; i++ {
		r.MustRegister(&eagerStub{
			name: fmt.Sprintf("eager-%02d", i),
			desc: "eager stub",
		})
	}

	ctx := context.Background()
	sig := TurnSignals{Mode: HydrationModeHeuristic}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := r.Schemas(ctx, sig)
		if err != nil {
			b.Fatalf("Schemas: %v", err)
		}
	}
}

// BenchmarkSchemas_AllLazyHidden_50 measures the worst case where
// all 50 lazy tools' triggers miss (no match). This stresses the
// trigger-evaluation path without any Hydrate calls.
func BenchmarkSchemas_AllLazyHidden_50(b *testing.B) {
	r := NewRegistry()
	for i := 0; i < 50; i++ {
		r.MustRegister(&lazyStub{
			name:   fmt.Sprintf("lazy-%02d", i),
			tag:    ToolTag{Intent: "never-matches"},
			schema: jsonschema.Schema{Type: "object"},
		})
	}

	ctx := context.Background()
	sig := TurnSignals{
		UserMessage:     "unrelated message",
		Mode:            HydrationModeHeuristic,
		RecentUseWindow: 5,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := r.Schemas(ctx, sig)
		if err != nil {
			b.Fatalf("Schemas: %v", err)
		}
	}
}
