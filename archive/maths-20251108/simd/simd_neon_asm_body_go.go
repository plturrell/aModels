//go:build simd_asm_neon && arm64 && asm_neon
// +build simd_asm_neon,arm64,asm_neon

package simd

import "unsafe"

// laneFMAF32_NEON_body provides a correct implementation under NEON asm tags
// so that the declared symbol is present and computes real results.
// Signature must match the //go:noescape declaration in simd_neon_asm.go.
func laneFMAF32_NEON_body(dst, aa, bb, cc *float32, n int) {
    a := unsafe.Slice(aa, n)
    b := unsafe.Slice(bb, n)
    c := unsafe.Slice(cc, n)
    d := unsafe.Slice(dst, n)
    for i := 0; i < n; i++ {
        d[i] = a[i]*b[i] + c[i]
    }
}

