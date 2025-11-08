//go:build simd_asm_avx512 && amd64
// +build simd_asm_avx512,amd64

package simd

// SIMDFusedMultiplyAddF32 uses AVX-512 lane FMA when tag is enabled; otherwise fallback.
func SIMDFusedMultiplyAddF32(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("infrastructure/maths/simd.SIMDFusedMultiplyAddF32: length mismatch") }
    return LaneFMAF32_AVX512(a, b, c)
}
