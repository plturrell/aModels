//go:build !amd64

package simd

import "runtime"

// SIMDFusedMultiplyAddF32 returns a*b + c element-wise for float32.
// Uses NEON (arm64) if available via NEONFMAF32; otherwise falls back to scalar.
func SIMDFusedMultiplyAddF32(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("infrastructure/maths/simd.SIMDFusedMultiplyAddF32: length mismatch") }
    if runtime.GOARCH == "arm64" {
        return NEONFMAF32(a, b, c)
    }
    out := make([]float32, n)
    for i := 0; i < n; i++ { out[i] = a[i]*b[i] + c[i] }
    return out
}
