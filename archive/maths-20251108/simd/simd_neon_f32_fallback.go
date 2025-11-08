//go:build arm64 && !(simd_asm_neon && asm_neon)

package simd

import "unsafe"

// NEONFMAF32 fallback when NEON asm tags are not enabled.
func NEONFMAF32(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("infrastructure/maths/simd.NEONFMAF32: length mismatch") }
    out := make([]float32, n)
    for i := 0; i < n; i++ { out[i] = a[i]*b[i] + c[i] }
    return out
}

// Provide the fallback body used by the asm stub when asm_neon tags are enabled.
// This symbol is resolved from simd_neon_asm_fallback_body.go under asm_neon tags.
func laneFMAF32_NEON_body_fallbackGo(dst, aa, bb, cc *float32, n int) {
    a := unsafe.Slice(aa, n)
    b := unsafe.Slice(bb, n)
    c := unsafe.Slice(cc, n)
    d := unsafe.Slice(dst, n)
    for i := 0; i < n; i++ { d[i] = a[i]*b[i] + c[i] }
}
