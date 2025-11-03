package flight

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/apache/arrow/go/v16/arrow/array"
	"github.com/apache/arrow/go/v16/arrow/memory"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/models"
)

func TestBuildOperationsRecord(t *testing.T) {
	allocator := memory.NewGoAllocator()
	now := time.Now().UTC()
	ops := []*models.LangOperation{
		{
			ID:          "abc",
			LibraryType: "test",
			Operation:   "op",
			Status:      models.OperationStatusSuccess,
			LatencyMs:   42,
			CreatedAt:   now,
			CompletedAt: &now,
			Input:       map[string]any{"foo": "bar"},
			Output:      map[string]any{"baz": 1},
		},
	}

	record, err := buildOperationsRecord(allocator, ops)
	if err != nil {
		t.Fatalf("buildOperationsRecord returned error: %v", err)
	}
	defer record.Release()

	if record.NumRows() != 1 {
		t.Fatalf("expected 1 row, got %d", record.NumRows())
	}

	// Validate JSON payloads are emitted
	inputCol := record.Column(11)
	if inputCol == nil {
		t.Fatal("input column not populated")
	}
	val := inputCol.(*array.String).Value(0)
	var payload map[string]any
	if err := json.Unmarshal([]byte(val), &payload); err != nil {
		t.Fatalf("input column does not contain valid json: %v", err)
	}
}
