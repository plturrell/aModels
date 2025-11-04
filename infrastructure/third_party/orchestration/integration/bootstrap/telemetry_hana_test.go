//go:build hana

package bootstrap

import (
	"context"
	"testing"
	"time"

	langintegration "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/lang_integration"
)

func TestLangOperationHandlerLogsSuccess(t *testing.T) {
	var logged []*langintegration.LangOperation

	base := time.Date(2025, time.January, 1, 0, 0, 0, 0, time.UTC)

	handler := &LangOperationHandler{
		operations: make(map[string]*operationState),
		now:        func() time.Time { return base },
		logFunc: func(ctx context.Context, op *langintegration.LangOperation) error {
			logged = append(logged, op)
			return nil
		},
	}

	ctx := context.Background()
	handler.HandleChainStart(ctx, map[string]any{"prompt": "ping"})

	handler.now = func() time.Time { return base.Add(2 * time.Second) }
	handler.HandleChainEnd(ctx, map[string]any{"text": "pong"})

	if len(logged) != 1 {
		t.Fatalf("expected 1 operation logged, got %d", len(logged))
	}

	op := logged[0]

	if op.Status != "success" {
		t.Fatalf("expected status success, got %s", op.Status)
	}
	if op.LatencyMs != 2000 {
		t.Fatalf("expected latency 2000ms, got %d", op.LatencyMs)
	}
	if op.Operation != "chain" {
		t.Fatalf("expected operation chain, got %s", op.Operation)
	}
	if op.PrivacyLevel != "default" {
		t.Fatalf("expected default privacy level, got %s", op.PrivacyLevel)
	}
	if _, ok := op.Input["prompt"]; !ok {
		t.Fatalf("expected prompt in input map")
	}
	if _, ok := op.Output["text"]; !ok {
		t.Fatalf("expected text in output map")
	}
	if op.LibraryType != langintegration.LangChain {
		t.Fatalf("expected library type LangChain, got %s", op.LibraryType)
	}
	if op.CompletedAt == nil {
		t.Fatalf("expected completed timestamp")
	}
}
