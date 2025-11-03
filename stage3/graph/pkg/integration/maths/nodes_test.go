package maths

import (
	"context"
	"testing"
)

func TestDotNode(t *testing.T) {
	handler := DotNode(nil)
	out, err := handler(context.Background(), DotInput{A: []float64{1, 2}, B: []float64{3, 4}})
	if err != nil {
		t.Fatalf("dot node error: %v", err)
	}
	if out.(float64) != 11 {
		t.Fatalf("unexpected dot output: %v", out)
	}
}

func TestCosineTopKNode(t *testing.T) {
	handler := CosineTopKNode(nil)
	input := CosineTopKInput{
		Dimension: 2,
		Matrix:    []float64{1, 0, 0, 1},
		Query:     []float64{1, 0},
		TopK:      1,
	}
	out, err := handler(context.Background(), input)
	if err != nil {
		t.Fatalf("cosine node error: %v", err)
	}
	result := out.(CosineTopKResult)
	if len(result.Indices) != 1 || result.Indices[0] != 0 {
		t.Fatalf("unexpected indices: %v", result.Indices)
	}
	if len(result.Scores) != 1 || result.Scores[0] < 0.99 {
		t.Fatalf("unexpected scores: %v", result.Scores)
	}
}

func TestMatMulNode(t *testing.T) {
	handler := MatMulNode(nil)
	input := MatMulInput{
		M: 2,
		N: 2,
		K: 2,
		A: []float64{1, 2, 3, 4},
		B: []float64{5, 6, 7, 8},
	}
	out, err := handler(context.Background(), input)
	if err != nil {
		t.Fatalf("matmul node error: %v", err)
	}
	vals := out.([]float64)
	expect := []float64{19, 22, 43, 50}
	for i, v := range vals {
		if v != expect[i] {
			t.Fatalf("unexpected result: %v", vals)
		}
	}
}
