package maths

import (
	"context"
	"testing"
)

func TestNewDotGraph(t *testing.T) {
	g, err := NewDotGraph(nil)
	if err != nil {
		t.Fatalf("NewDotGraph error: %v", err)
	}
	out, err := g.Invoke(context.Background(), DotInput{A: []float64{1, 2}, B: []float64{3, 4}})
	if err != nil {
		t.Fatalf("invoke dot graph: %v", err)
	}
	if got := out.(float64); got != 11 {
		t.Fatalf("unexpected dot output: %v", got)
	}
}

func TestBuildGraphRequiresNodes(t *testing.T) {
	if _, err := BuildGraph("entry", "", nil, nil); err == nil {
		t.Fatalf("expected error for empty node set")
	}
}
