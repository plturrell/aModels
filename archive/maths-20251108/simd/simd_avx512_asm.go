//go:build simd_asm_avx512 && amd64 && asm_avx512
// +build simd_asm_avx512,amd64,asm_avx512

package simd

// laneFMAF32_AVX512_body is provided under the same build tags in
// simd_avx512_asm_body_go.go so asm-tag builds compute correctly.

func LaneFMAF32_AVX512(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n {
        panic("infrastructure/maths/simd.LaneFMAF32_AVX512: length mismatch")
    }
    out := make([]float32, n)
    if n == 0 {
        return out
    }
    laneFMAF32_AVX512_body(&out[0], &a[0], &b[0], &c[0], n)
    return out
}
