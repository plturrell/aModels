//go:build simd_asm_neon && arm64 && asm_neon
// +build simd_asm_neon,arm64,asm_neon

package simd

import (
    "math/rand"
    "testing"
)

func TestNEONFMAF32_AsmMatchesFallback(t *testing.T) {
    n := 513
    a := make([]float32, n)
    b := make([]float32, n)
    c := make([]float32, n)
    for i := 0; i < n; i++ {
        a[i] = float32(rand.NormFloat64())
        b[i] = float32(rand.NormFloat64())
        c[i] = float32(rand.NormFloat64())
    }
    asmOut := NEONFMAF32(a, b, c)
    fbOut := neonFMAFallback(a, b, c)
    if len(asmOut) != len(fbOut) { t.Fatalf("len mismatch") }
    for i := range asmOut {
        if d := float32Abs(asmOut[i]-fbOut[i]); d > 1e-6 {
            t.Fatalf("mismatch at %d: %f vs %f", i, asmOut[i], fbOut[i])
        }
    }
}

func neonFMAFallback(a, b, c []float32) []float32 {
    n := len(a)
    out := make([]float32, n)
    for i := 0; i < n; i++ { out[i] = a[i]*b[i] + c[i] }
    return out
}

func float32Abs(x float32) float32 { if x < 0 { return -x }; return x }
