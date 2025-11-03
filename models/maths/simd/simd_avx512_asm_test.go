//go:build simd_asm_avx512 && amd64 && asm_avx512
// +build simd_asm_avx512,amd64,asm_avx512

package simd

import (
    "math/rand"
    "testing"
)

func TestLaneFMAF32_AVX512_AsmMatchesFallback(t *testing.T) {
    n := 257
    a := make([]float32, n)
    b := make([]float32, n)
    c := make([]float32, n)
    for i := 0; i < n; i++ {
        a[i] = float32(rand.NormFloat64())
        b[i] = float32(rand.NormFloat64())
        c[i] = float32(rand.NormFloat64())
    }
    asmOut := LaneFMAF32_AVX512(a, b, c)
    fbOut := SIMDFusedMultiplyAddF32_Fallback(a, b, c)
    if len(asmOut) != len(fbOut) { t.Fatalf("len mismatch") }
    for i := range asmOut {
        if d := float32Abs(asmOut[i]-fbOut[i]); d > 1e-6 {
            t.Fatalf("mismatch at %d: %f vs %f", i, asmOut[i], fbOut[i])
        }
    }
}

func SIMDFusedMultiplyAddF32_Fallback(a, b, c []float32) []float32 {
    n := len(a)
    out := make([]float32, n)
    for i := 0; i < n; i++ { out[i] = a[i]*b[i] + c[i] }
    return out
}

func float32Abs(x float32) float32 { if x < 0 { return -x }; return x }
