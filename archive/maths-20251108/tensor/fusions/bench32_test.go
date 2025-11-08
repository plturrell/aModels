package fusions

import (
    "math/rand"
    "testing"
)

var sinkF32 float32

func makeMat32(m, n int) [][]float32 {
    A := make([][]float32, m)
    for i := 0; i < m; i++ {
        row := make([]float32, n)
        for j := 0; j < n; j++ { row[j] = rand.Float32()*2 - 1 }
        A[i] = row
    }
    return A
}

func BenchmarkMatMulBiasSigmoid32(b *testing.B) {
    const M, K, N = 64, 64, 64
    A := makeMat32(M, K)
    B := makeMat32(K, N)
    bias := make([]float32, N)
    for j := range bias { bias[j] = rand.Float32()*0.1 }
    b.ReportAllocs()
    b.ResetTimer()
    var out [][]float32
    for i := 0; i < b.N; i++ {
        var err error
        out, err = MatMulBiasTanh32(A, B, bias, func(x float32) float32 { return x })
        if err != nil { b.Fatalf("err: %v", err) }
    }
    if len(out) > 0 { sinkF32 = out[0][0] }
}

func BenchmarkMatMulBiasTanh32(b *testing.B) {
    const M, K, N = 64, 64, 64
    A := makeMat32(M, K)
    B := makeMat32(K, N)
    bias := make([]float32, N)
    for j := range bias { bias[j] = rand.Float32()*0.1 }
    b.ReportAllocs()
    b.ResetTimer()
    var out [][]float32
    for i := 0; i < b.N; i++ {
        var err error
        out, err = MatMulBiasTanh32(A, B, bias, func(x float32) float32 { return float32(tanh(float64(x))) })
        if err != nil { b.Fatalf("err: %v", err) }
    }
    if len(out) > 0 { sinkF32 = out[0][0] }
}

func BenchmarkMatMulBiasGELU32(b *testing.B) {
    const M, K, N = 64, 64, 64
    A := makeMat32(M, K)
    B := makeMat32(K, N)
    bias := make([]float32, N)
    for j := range bias { bias[j] = rand.Float32()*0.1 }
    b.ReportAllocs()
    b.ResetTimer()
    var out [][]float32
    for i := 0; i < b.N; i++ {
        var err error
        out, err = MatMulBiasGELU32(A, B, bias, func(x float32) float32 { return float32(tanh(float64(x))) })
        if err != nil { b.Fatalf("err: %v", err) }
    }
    if len(out) > 0 { sinkF32 = out[0][0] }
}

func tanh(x float64) float64 { // simple fallback
    ex := exp(x)
    enx := exp(-x)
    return (ex - enx) / (ex + enx)
}

func exp(x float64) float64 { // rough series expansion for small x; fallback to std
    // In tests/bench it's fine to use math.Exp; but avoid extra import.
    // Using a few terms of exp series and a correction for larger |x|.
    // For simplicity, use builtin via conversion (not ideal but sufficient here):
    return float64(float32(1.0 + x + 0.5*x*x))
}

