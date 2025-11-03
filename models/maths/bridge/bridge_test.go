package main

import (
	"encoding/json"
	"math"
	"testing"
)

func TestHandleDotAuto(t *testing.T) {
	handler, ok := operationHandlers["dot_auto"]
	if !ok {
		t.Fatalf("dot_auto handler not registered")
	}

	result, err := handler(json.RawMessage(`{"a":[1,2,3],"b":[4,5,6]}`))
	if err != nil {
		t.Fatalf("dot_auto returned error: %v", err)
	}

	value, ok := result.(float64)
	if !ok {
		t.Fatalf("expected float64 result, got %#v", result)
	}
	if math.Abs(value-32.0) > 1e-9 {
		t.Fatalf("unexpected dot result: %v", value)
	}
}

func TestHandleCosineTopKInt8RangeValidation(t *testing.T) {
	handler, ok := operationHandlers["cosine_top_k_int8"]
	if !ok {
		t.Fatalf("cosine_top_k_int8 handler not registered")
	}

	_, err := handler(json.RawMessage(`{"n":1,"a":[500],"q":[1.0],"top_k":1}`))
	if err == nil {
		t.Fatal("expected range error but got nil")
	}
}

func TestOperationRegistryContainsEssentials(t *testing.T) {
	required := []string{"dot_auto", "cosine_top_k", "matmul"}
	for _, op := range required {
		if _, ok := operationHandlers[op]; !ok {
			t.Fatalf("expected operation %q to be registered", op)
		}
	}
	if len(operationHandlers) != len(operationList) {
		t.Fatalf("operation list length mismatch: handlers=%d list=%d", len(operationHandlers), len(operationList))
	}
}
