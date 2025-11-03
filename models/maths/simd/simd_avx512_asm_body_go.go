//go:build simd_asm_avx512 && amd64 && asm_avx512
// +build simd_asm_avx512,amd64,asm_avx512

package simd

import "unsafe"

// laneFMAF32_AVX512_body provides a correct implementation under AVX-512 asm tags
// so that the declared symbol is present and computes real results.
func laneFMAF32_AVX512_body(dst, aa, bb, cc *float32, n int) {
    a := unsafe.Slice(aa, n)
    b := unsafe.Slice(bb, n)
    c := unsafe.Slice(cc, n)
    d := unsafe.Slice(dst, n)
    for i := 0; i < n; i++ {
        d[i] = a[i]*b[i] + c[i]
    }
}

