//go:build amd64 && !avx2asm
// +build amd64,!avx2asm

package gemm

import "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"

// gemm8x4_kernel is implemented in gemm_amd64.s
func gemm8x4_kernel(a *float64, b *float64, kn int, jn int, c *float64, cstride int)

func asmGemm8x4(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) == 0 { return false }
    // Tile rows in steps of 8 and columns in steps of 4
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+3 < jn; j += 4 {
            cptr := &C.Data[(ic+i)*C.Stride + jc + j]
            gemm8x4_kernel(&aPack[i*kn], &bPack[j], kn, jn, cptr, C.Stride)
        }
    }
    // Handle tails generically
    iTail := (im / 8) * 8
    jTail := (jn / 4) * 4
    if iTail < im || jTail < jn {
        // Generic fallback for remaining region
        // Build a small loop to complete the block
        for i := iTail; i < im; i++ {
            for j := 0; j < jn; j++ {
                sum := 0.0
                for p := 0; p < kn; p++ {
                    sum += aPack[i*kn+p] * bPack[p*jn+j]
                }
                C.Data[(ic+i)*C.Stride + jc + j] += sum
            }
        }
        // Rows covered but leftover cols (if any)
        for i := 0; i < iTail; i++ {
            for j := jTail; j < jn; j++ {
                sum := 0.0
                for p := 0; p < kn; p++ {
                    sum += aPack[i*kn+p] * bPack[p*jn+j]
                }
                C.Data[(ic+i)*C.Stride + jc + j] += sum
            }
        }
    }
    return true
}

func asmGemm8x8(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) < 4 { return false }
    // Tile 8x8 via two 8x4 kernel calls per tile
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+7 < jn; j += 8 {
            cptr0 := &C.Data[(ic+i)*C.Stride + jc + j]
            cptr1 := &C.Data[(ic+i)*C.Stride + jc + j + 4]
            gemm8x4_kernel(&aPack[i*kn], &bPack[j], kn, jn, cptr0, C.Stride)
            gemm8x4_kernel(&aPack[i*kn], &bPack[j+4], kn, jn, cptr1, C.Stride)
        }
    }
    // Handle tails with asmGemm8x4 and generic path
    iTail := (im / 8) * 8
    jTail8 := (jn / 8) * 8
    if jTail8 < jn {
        // Use 8x4 tiles for the next 4 cols if available
        if jn-jTail8 >= 4 {
            for i := 0; i+7 < im; i += 8 {
                cptr := &C.Data[(ic+i)*C.Stride + jc + jTail8]
                gemm8x4_kernel(&aPack[i*kn], &bPack[jTail8], kn, jn, cptr, C.Stride)
            }
            jTail8 += 4
        }
    }
    // Any remaining rows or cols are done generically
    if iTail < im || jTail8 < jn {
        for i := iTail; i < im; i++ {
            for j := 0; j < jn; j++ {
                sum := 0.0
                for p := 0; p < kn; p++ {
                    sum += aPack[i*kn+p] * bPack[p*jn+j]
                }
                C.Data[(ic+i)*C.Stride + jc + j] += sum
            }
        }
        for i := 0; i < iTail; i++ {
            for j := jTail8; j < jn; j++ {
                sum := 0.0
                for p := 0; p < kn; p++ {
                    sum += aPack[i*kn+p] * bPack[p*jn+j]
                }
                C.Data[(ic+i)*C.Stride + jc + j] += sum
            }
        }
    }
    return true
}

// asmGemm8x16 composes 8x16 tiles out of four 8x4 kernel calls per tile.
func asmGemm8x16(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) < 16 { return false }
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+15 < jn; j += 16 {
            baseC := (ic+i)*C.Stride + jc + j
            // 4 contiguous 8x4 blocks across columns
            gemm8x4_kernel(&aPack[i*kn], &bPack[j+0],  kn, jn, &C.Data[baseC+0],  C.Stride)
            gemm8x4_kernel(&aPack[i*kn], &bPack[j+4],  kn, jn, &C.Data[baseC+4],  C.Stride)
            gemm8x4_kernel(&aPack[i*kn], &bPack[j+8],  kn, jn, &C.Data[baseC+8],  C.Stride)
            gemm8x4_kernel(&aPack[i*kn], &bPack[j+12], kn, jn, &C.Data[baseC+12], C.Stride)
        }
    }
    // Tails generically
    iTail := (im / 8) * 8
    jTail16 := (jn / 16) * 16
    if iTail < im || jTail16 < jn {
        for i := iTail; i < im; i++ {
            for j := 0; j < jn; j++ {
                sum := 0.0
                for p := 0; p < kn; p++ { sum += aPack[i*kn+p] * bPack[p*jn+j] }
                C.Data[(ic+i)*C.Stride + jc + j] += sum
            }
        }
        for i := 0; i < iTail; i++ {
            for j := jTail16; j < jn; j++ {
                sum := 0.0
                for p := 0; p < kn; p++ { sum += aPack[i*kn+p] * bPack[p*jn+j] }
                C.Data[(ic+i)*C.Stride + jc + j] += sum
            }
        }
    }
    return true
}
