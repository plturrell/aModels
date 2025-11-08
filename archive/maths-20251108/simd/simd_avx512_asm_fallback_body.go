//go:build simd_asm_avx512 && amd64 && asm_avx512
// +build simd_asm_avx512,amd64,asm_avx512

package simd

import "unsafe"

// laneFMAF32_AVX512_body_fallbackGo provides a correct scalar implementation
// that the assembly stub can delegate to when AVX-512 vector body is not yet
// provided. This ensures asm-tag builds compute correct results.
func laneFMAF32_AVX512_body_fallbackGo(dst, aa, bb, cc *float32, n int) {
    a := unsafe.Slice(aa, n)
    b := unsafe.Slice(bb, n)
    c := unsafe.Slice(cc, n)
    d := unsafe.Slice(dst, n)
    for i := 0; i < n; i++ {
        d[i] = a[i]*b[i] + c[i]
    }
}

