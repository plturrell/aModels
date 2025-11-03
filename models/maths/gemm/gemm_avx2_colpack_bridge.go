//go:build amd64 && avx2asm
// +build amd64,avx2asm

package gemm

// Bridge implementation for column-packed B using the existing 8x4 AVX2 kernel.
// It repacks 4 columns of length kn into a temporary row-major panel (kn x 4)
// and then calls the row-packed 8x4 kernel. This ensures correctness with
// avx2asm tags without relying on a separate assembly body.

// gemm8x4_kernel_avx2 is implemented in gemm_avx2_amd64.s
func gemm8x4_kernel_avx2(a *float64, b *float64, kn int, jn int, c *float64, cstride int)

func gemm8x4_kernel_avx2_colpack(a *float64, b *float64, kn int, colStride int, c *float64, cstride int) {
    if kn <= 0 { return }
    // Allocate a small temporary panel kn x 4 in row-major
    buf := make([]float64, kn*4)
    // Pack: for each p, take 4 columns at offsets 0, colStride, 2*colStride, 3*colStride
    for p := 0; p < kn; p++ {
        // Compute base indices
        b0 := *(*float64)(ptrOffset(b, (p+0*colStride)*8))
        b1 := *(*float64)(ptrOffset(b, (p+1*colStride)*8))
        b2 := *(*float64)(ptrOffset(b, (p+2*colStride)*8))
        b3 := *(*float64)(ptrOffset(b, (p+3*colStride)*8))
        off := p*4
        buf[off+0] = b0
        buf[off+1] = b1
        buf[off+2] = b2
        buf[off+3] = b3
    }
    // Call row-packed kernel with jn=4
    gemm8x4_kernel_avx2(a, &buf[0], kn, 4, c, cstride)
}

import "unsafe"

func ptrOffset(p *float64, bytes int) *float64 {
    return (*float64)(unsafe.Add(unsafe.Pointer(p), bytes))
}
