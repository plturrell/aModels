package maths

import (
	"context"
	"testing"
	"time"
)

func TestEngineDot(t *testing.T) {
	engine := NewEngine(nil)
	val, err := engine.Dot(context.Background(), []float64{1, 2, 3}, []float64{4, 5, 6})
	if err != nil {
		t.Fatalf("Dot error: %v", err)
	}
	if val != 32 {
		t.Fatalf("unexpected dot value: %v", val)
	}
}

func TestEngineCosineTopK(t *testing.T) {
	engine := NewEngine(nil)
	A := []float64{
		1, 0,
		0, 1,
	}
	q := []float64{1, 0}
	idx, scores, err := engine.CosineTopK(context.Background(), 2, A, q, 1)
	if err != nil {
		t.Fatalf("CosineTopK error: %v", err)
	}
	if len(idx) != 1 || idx[0] != 0 {
		t.Fatalf("unexpected indices: %v", idx)
	}
	if len(scores) != 1 || scores[0] < 0.999 {
		t.Fatalf("unexpected scores: %v", scores)
	}
}

func TestEngineRespectsContext(t *testing.T) {
	engine := NewEngine(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := engine.Dot(ctx, []float64{1}, []float64{1}); err == nil {
		t.Fatalf("expected context error")
	}
}

func TestEngineMatMul(t *testing.T) {
	engine := NewEngine(nil)
	A := []float64{
		1, 2,
		3, 4,
	}
	B := []float64{
		5, 6,
		7, 8,
	}
	out, err := engine.MatMul(context.Background(), 2, 2, 2, A, B)
	if err != nil {
		t.Fatalf("MatMul error: %v", err)
	}
	want := []float64{19, 22, 43, 50}
	for i, v := range out {
		if v != want[i] {
			t.Fatalf("unexpected MatMul result %v != %v", out, want)
		}
	}
}

func TestEngineProject(t *testing.T) {
	engine := NewEngine(nil)
	A := []float64{
		1, 0,
		0, 1,
	}
	P := []float64{
		1,
		1,
	}
	out, err := engine.Project(context.Background(), 2, 2, 1, A, P)
	if err != nil {
		t.Fatalf("Project error: %v", err)
	}
	if len(out) != 2 || out[0] != 1 || out[1] != 1 {
		t.Fatalf("unexpected projection result: %v", out)
	}
}

func TestEngineCosineTopKInt8(t *testing.T) {
	engine := NewEngine(nil)
	A := []int8{1, 0, 0, 1}
	q := []float64{1, 0}
	idx, scores, err := engine.CosineTopKInt8(context.Background(), 2, A, q, 1)
	if err != nil {
		t.Fatalf("CosineTopKInt8 error: %v", err)
	}
	if idx[0] != 0 || scores[0] < 0.999 {
		t.Fatalf("unexpected outputs idx=%v scores=%v", idx, scores)
	}
}

func TestEngineCosineMultiTopK(t *testing.T) {
	engine := NewEngine(nil)
	A := []float64{
		1, 0,
		0, 1,
	}
	Q := []float64{
		1, 0,
		0, 1,
	}
	idx, scores, err := engine.CosineMultiTopK(context.Background(), 2, A, Q, 1)
	if err != nil {
		t.Fatalf("CosineMultiTopK error: %v", err)
	}
	if len(idx) != 2 || idx[0][0] != 0 || idx[1][0] != 1 {
		t.Fatalf("unexpected idx: %v", idx)
	}
	if len(scores) != 2 || scores[0][0] < 0.999 || scores[1][0] < 0.999 {
		t.Fatalf("unexpected scores: %v", scores)
	}
}

func TestEngineTimeout(t *testing.T) {
	engine := NewEngine(nil)
	ctx, cancel := context.WithTimeout(context.Background(), 0)
	defer cancel()
	time.Sleep(time.Millisecond)
	if _, err := engine.Dot(ctx, []float64{1}, []float64{1}); err == nil {
		t.Fatalf("expected deadline error")
	}
}
