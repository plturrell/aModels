//go:build arm64 && neonasm
// +build arm64,neonasm

package gemm

import "unsafe"

// Pure-Go bodies for NEON kernel symbols to ensure correctness when the
// assembly versions are not present. These match the signatures referenced by
// gemm_asm_arm64_neon.go.

func gemm8x4_kernel_neon(a *float64, b *float64, kn int, jn int, c *float64, cstride int) {
    // Compute an 8x4 tile starting at C, using A(8xkn) and B(kn x jn) row-major
    aTile := unsafe.Slice(a, 8*kn)
    bTile := unsafe.Slice(b, kn*jn)
    // c row slices will be constructed per row using stride
    for r := 0; r < 8; r++ {
        var s0, s1, s2, s3 float64
        for p := 0; p < kn; p++ {
            av := aTile[r*kn+p]
            bp := bTile[p*jn : p*jn+4]
            s0 += av * bp[0]
            s1 += av * bp[1]
            s2 += av * bp[2]
            s3 += av * bp[3]
        }
        crow := unsafe.Slice((*float64)(unsafe.Add(unsafe.Pointer(c), uintptr(r*cstride*8))), 4)
        crow[0] += s0
        crow[1] += s1
        crow[2] += s2
        crow[3] += s3
    }
}

func gemm8x8_kernel_neon(a *float64, b *float64, kn int, jn int, c *float64, cstride int) {
    // Compose two 8x4 tiles to cover 8x8
    // Left 4 cols
    gemm8x4_kernel_neon(a, b, kn, jn, c, cstride)
    // Right 4 cols (offset B by 4, C by 4)
    bRight := (*float64)(unsafe.Add(unsafe.Pointer(b), uintptr(4*8)))
    cRight := (*float64)(unsafe.Add(unsafe.Pointer(c), uintptr(4*8)))
    gemm8x4_kernel_neon(a, bRight, kn, jn, cRight, cstride)
}
