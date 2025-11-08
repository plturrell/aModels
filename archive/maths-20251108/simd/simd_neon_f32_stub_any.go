//go:build !arm64

package simd

// Fallback stub for NEON F32 FMA on non-arm64 architectures.
func NEONFMAF32(a, b, c []float32) []float32 {
    n := len(a)
    out := make([]float32, n)
    for i := 0; i < n; i++ { out[i] = a[i]*b[i] + c[i] }
    return out
}
