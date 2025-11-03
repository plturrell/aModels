package fusions

import (
    "math/rand"
    "testing"
)

func benchArrayF64(n int) []float64 { a:=make([]float64,n); for i:=range a{ a[i]=rand.NormFloat64() }; return a }

func BenchmarkMatMulBiasGELU(b *testing.B) {
    m, k, n := 256, 256, 256
    A := make([][]float64, m)
    for i := 0; i < m; i++ { A[i] = benchArrayF64(k) }
    B := make([][]float64, k)
    for i := 0; i < k; i++ { B[i] = benchArrayF64(n) }
    bias := benchArrayF64(n)
    b.ResetTimer()
    for i := 0; i < b.N; i++ { _, _ = MatMulBiasGELU(A, B, bias) }
}

func BenchmarkSoftmaxCrossEntropy(b *testing.B) {
    m, n := 1024, 512
    logits := make([][]float64, m)
    labels := make([]int, m)
    for i := 0; i < m; i++ { logits[i] = benchArrayF64(n); labels[i] = rand.Intn(n) }
    b.ResetTimer()
    for i := 0; i < b.N; i++ { _, _ = SoftmaxCrossEntropy(logits, labels) }
}

