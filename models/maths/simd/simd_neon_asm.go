//go:build simd_asm_neon && arm64 && asm_neon
// +build simd_asm_neon,arm64,asm_neon

package simd

// laneFMAF32_NEON_body is provided under the same build tags in
// simd_neon_asm_body_go.go so asm-tag builds compute correctly.

func NEONFMAF32(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("infrastructure/maths/simd.NEONFMAF32: length mismatch") }
    out := make([]float32, n)
    if n == 0 { return out }
    laneFMAF32_NEON_body(&out[0], &a[0], &b[0], &c[0], n)
    return out
}
