package fusions

import (
    "math"
    "testing"
)

func TestSoftmaxCrossEntropy(t *testing.T) {
    logits := [][]float64{{1, 2, 3}, {3, 2, 1}}
    labels := []int{2, 0}
    loss, probs := SoftmaxCrossEntropy(logits, labels)
    if len(probs) != 2 || len(probs[0]) != 3 { t.Fatalf("unexpected probs shape") }
    naive := func(row []float64) []float64 {
        m := row[0]; for _, v := range row { if v > m { m = v } }
        sum := 0.0; out := make([]float64, len(row))
        for i, v := range row { out[i] = math.Exp(v-m); sum += out[i] }
        for i := range out { out[i] /= sum }
        return out
    }
    p0 := naive(logits[0]); p1 := naive(logits[1])
    want := -0.5*(math.Log(p0[2]) + math.Log(p1[0]))
    if math.Abs(loss-want) > 1e-6 { t.Fatalf("loss mismatch got=%v want=%v", loss, want) }
}

func TestMatMulBiasGELU(t *testing.T) {
    A := [][]float64{{1, 2}, {3, 4}}
    B := [][]float64{{5, 6}, {7, 8}}
    bias := []float64{0.1, -0.2}
    C, err := MatMulBiasGELU(A, B, bias)
    if err != nil { t.Fatalf("err: %v", err) }
    if len(C) != 2 || len(C[0]) != 2 { t.Fatalf("unexpected shape") }
    for i := range C { for j := range C[i] {
        if math.IsNaN(C[i][j]) { t.Fatalf("nan at %d,%d", i,j) }
    } }
}

