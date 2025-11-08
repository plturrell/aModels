//go:build simd_asm_neon && arm64 && asm_neon
// +build simd_asm_neon,arm64,asm_neon

package simd

import "unsafe"

// Provide the fallback body used by the NEON asm stub when asm_neon tags are enabled.
func laneFMAF32_NEON_body_fallbackGo(dst, aa, bb, cc *float32, n int) {
    a := unsafe.Slice(aa, n)
    b := unsafe.Slice(bb, n)
    c := unsafe.Slice(cc, n)
    d := unsafe.Slice(dst, n)
    for i := 0; i < n; i++ {
        d[i] = a[i]*b[i] + c[i]
    }
}

