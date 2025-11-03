package fusions

import (
    "math/rand"
    "testing"
)

func makeLogits32(m, n int) ([][]float32, []int) {
    X := make([][]float32, m)
    labels := make([]int, m)
    for i := 0; i < m; i++ {
        row := make([]float32, n)
        for j := 0; j < n; j++ { row[j] = rand.Float32()*4 - 2 }
        X[i] = row
        labels[i] = rand.Intn(n)
    }
    return X, labels
}

func BenchmarkSoftmaxRow32(b *testing.B) {
    x := make([]float32, 1024)
    for i := range x { x[i] = float32(i%17) - 8 }
    b.ReportAllocs()
    b.ResetTimer()
    var out []float32
    for i := 0; i < b.N; i++ {
        out = SoftmaxRow32(x)
    }
    if len(out) == 0 { b.Fatal("no output") }
}

func BenchmarkSoftmaxCrossEntropy32(b *testing.B) {
    logits, labels := makeLogits32(256, 128)
    b.ReportAllocs()
    b.ResetTimer()
    var loss float32
    var probs [][]float32
    for i := 0; i < b.N; i++ {
        loss, probs = SoftmaxCrossEntropy32(logits, labels)
    }
    _ = loss
    if len(probs) == 0 { b.Fatal("no output") }
}

