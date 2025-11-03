//go:build amd64 && avx2asm

package gemm

import "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"

// gemm8x4_kernel_avx2_colpack is implemented in gemm_avx2_colpack_amd64.s
func gemm8x4_kernel_avx2_colpack(a *float64, b *float64, kn int, colStride int, c *float64, cstride int)

// Try AVX2 asm kernels directly on column-packed B to avoid per-tile repack.
func tryAsmColPack(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) bool {
    if jn < 4 || im < 8 { return false }
    handled := false
    // 1) 8x16
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+15 < jn; j += 16 {
            base := (ic+i)*C.Stride + jc + j
            gemm8x4_kernel_avx2_colpack(&aPack[i*kn], &bPack[(j+0)*kn],  kn, kn, &C.Data[base+0],  C.Stride)
            gemm8x4_kernel_avx2_colpack(&aPack[i*kn], &bPack[(j+4)*kn],  kn, kn, &C.Data[base+4],  C.Stride)
            gemm8x4_kernel_avx2_colpack(&aPack[i*kn], &bPack[(j+8)*kn],  kn, kn, &C.Data[base+8],  C.Stride)
            gemm8x4_kernel_avx2_colpack(&aPack[i*kn], &bPack[(j+12)*kn], kn, kn, &C.Data[base+12], C.Stride)
            handled = true
        }
    }
    // 2) 8x8
    for i := 0; i+7 < im; i += 8 {
        jTail16 := (jn / 16) * 16
        for j := jTail16; j+7 < jn; j += 8 {
            base := (ic+i)*C.Stride + jc + j
            gemm8x4_kernel_avx2_colpack(&aPack[i*kn], &bPack[(j+0)*kn], kn, kn, &C.Data[base+0], C.Stride)
            gemm8x4_kernel_avx2_colpack(&aPack[i*kn], &bPack[(j+4)*kn], kn, kn, &C.Data[base+4], C.Stride)
            handled = true
        }
    }
    // 3) 8x4
    for i := 0; i+7 < im; i += 8 {
        jTail16 := (jn / 16) * 16
        jTail8 := jTail16 + ((jn - jTail16) / 8) * 8
        for j := jTail8; j+3 < jn; j += 4 {
            gemm8x4_kernel_avx2_colpack(&aPack[i*kn], &bPack[j*kn], kn, kn, &C.Data[(ic+i)*C.Stride + jc + j], C.Stride)
            handled = true
        }
    }
    return handled
}
