//go:build arm64 && neonasm

package gemm

import "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"

// NEON 8x4 kernel (assembly) for row-packed B; composes into 8x8 and 8x16 tiles.

// gemm8x4_kernel_neon and gemm8x8_kernel_neon are provided under the same
// build tags. Implementations may be in Go for correctness.

func asmGemm8x4(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) == 0 { return false }
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+3 < jn; j += 4 {
            cptr := &C.Data[(ic+i)*C.Stride + jc + j]
            gemm8x4_kernel_neon(&aPack[i*kn], &bPack[j], kn, jn, cptr, C.Stride)
        }
    }
    // tails generic
    iTail := (im / 8) * 8
    jTail := (jn / 4) * 4
    for i := iTail; i < im; i++ {
        for j := 0; j < jn; j++ {
            s := 0.0
            for p := 0; p < kn; p++ { s += aPack[i*kn+p] * bPack[p*jn+j] }
            C.Data[(ic+i)*C.Stride + jc + j] += s
        }
    }
    for i := 0; i < iTail; i++ {
        for j := jTail; j < jn; j++ {
            s := 0.0
            for p := 0; p < kn; p++ { s += aPack[i*kn+p] * bPack[p*jn+j] }
            C.Data[(ic+i)*C.Stride + jc + j] += s
        }
    }
    return true
}

func asmGemm8x8(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) < 8 { return false }
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+7 < jn; j += 8 {
            base := (ic+i)*C.Stride + jc + j
            gemm8x8_kernel_neon(&aPack[i*kn], &bPack[j], kn, jn, &C.Data[base], C.Stride)
        }
    }
    // tails
    iTail := (im / 8) * 8
    jTail := (jn / 8) * 8
    if jn-jTail >= 4 {
        for i := 0; i+7 < im; i += 8 {
            gemm8x4_kernel_neon(&aPack[i*kn], &bPack[jTail], kn, jn, &C.Data[(ic+i)*C.Stride + jc + jTail], C.Stride)
        }
        jTail += 4
    }
    for i := iTail; i < im; i++ {
        for j := 0; j < jn; j++ {
            s := 0.0
            for p := 0; p < kn; p++ { s += aPack[i*kn+p] * bPack[p*jn+j] }
            C.Data[(ic+i)*C.Stride + jc + j] += s
        }
    }
    for i := 0; i < iTail; i++ {
        for j := jTail; j < jn; j++ {
            s := 0.0
            for p := 0; p < kn; p++ { s += aPack[i*kn+p] * bPack[p*jn+j] }
            C.Data[(ic+i)*C.Stride + jc + j] += s
        }
    }
    return true
}

func asmGemm8x16(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) < 16 { return false }
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+15 < jn; j += 16 {
            base := (ic+i)*C.Stride + jc + j
            gemm8x4_kernel_neon(&aPack[i*kn], &bPack[j+0],  kn, jn, &C.Data[base+0],  C.Stride)
            gemm8x4_kernel_neon(&aPack[i*kn], &bPack[j+4],  kn, jn, &C.Data[base+4],  C.Stride)
            gemm8x4_kernel_neon(&aPack[i*kn], &bPack[j+8],  kn, jn, &C.Data[base+8],  C.Stride)
            gemm8x4_kernel_neon(&aPack[i*kn], &bPack[j+12], kn, jn, &C.Data[base+12], C.Stride)
        }
    }
    // tails generic
    iTail := (im / 8) * 8
    jTail := (jn / 16) * 16
    for i := iTail; i < im; i++ {
        for j := 0; j < jn; j++ {
            s := 0.0
            for p := 0; p < kn; p++ { s += aPack[i*kn+p] * bPack[p*jn+j] }
            C.Data[(ic+i)*C.Stride + jc + j] += s
        }
    }
    for i := 0; i < iTail; i++ {
        for j := jTail; j < jn; j++ {
            s := 0.0
            for p := 0; p < kn; p++ { s += aPack[i*kn+p] * bPack[p*jn+j] }
            C.Data[(ic+i)*C.Stride + jc + j] += s
        }
    }
    return true
}
