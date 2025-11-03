//go:build simd_asm_neon && arm64 && !asm_neon
// +build simd_asm_neon,arm64,!asm_neon

package simd

// Placeholder NEON implementations; replace with hand-tuned assembly.

func NEONAddF32(a, b []float32) []float32 {
    n := len(a)
    if len(b) != n { panic("infrastructure/maths/simd.NEONAddF32: length mismatch") }
    out := make([]float32, n)
    for i := range a { out[i] = a[i] + b[i] }
    return out
}

func NEONFMAF32(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("infrastructure/maths/simd.NEONFMAF32: length mismatch") }
    out := make([]float32, n)
    for i := range a { out[i] = a[i]*b[i] + c[i] }
    return out
}

func LaneDotF32_NEON(a, b []float32) float32 {
    n := len(a)
    if len(b) != n { panic("infrastructure/maths/simd.LaneDotF32_NEON: length mismatch") }
    var sum float32
    for i := range a { sum += a[i] * b[i] }
    return sum
}
