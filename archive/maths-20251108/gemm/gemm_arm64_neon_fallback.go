//go:build arm64 && neonasm

package gemm

import "unsafe"

// Go fallback for NEON entrypoint: computes an 8x4 tile C += A(8xK)*B(Kx4) for row-packed B.
func gemm8x4_kernel_neon_fallbackGo(a *float64, b *float64, kn int, jn int, c *float64, cstride int) {
    as := unsafe.Slice(a, kn*8)
    bs := unsafe.Slice(b, kn*jn)
    cs := unsafe.Slice(c, cstride*8)
    for r := 0; r < 8; r++ {
        aBase := r * kn
        cBase := r * cstride
        for j := 0; j < 4; j++ {
            sum := 0.0
            bcol := j
            for p := 0; p < kn; p++ {
                sum += as[aBase+p] * bs[p*jn+bcol]
            }
            cs[cBase+j] += sum
        }
    }
}
