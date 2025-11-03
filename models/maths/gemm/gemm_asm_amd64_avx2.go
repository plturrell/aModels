//go:build amd64 && avx2asm
// +build amd64,avx2asm

package gemm

import (
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	cpu "golang.org/x/sys/cpu"
)

// gemm8x4_kernel_avx2 is implemented in gemm_avx2_amd64.s
func gemm8x4_kernel_avx2(a *float64, b *float64, kn int, jn int, c *float64, cstride int)

// gemm8x4_kernel_avx2_fma is implemented in gemm_avx2_fma_amd64.s (uses FMA)
func gemm8x4_kernel_avx2_fma(a *float64, b *float64, kn int, jn int, c *float64, cstride int)

func callKernel8x4(a *float64, b *float64, kn int, jn int, c *float64, cstride int) {
    if cpu.X86.HasFMA {
        gemm8x4_kernel_avx2_fma(a, b, kn, jn, c, cstride)
    } else {
        gemm8x4_kernel_avx2(a, b, kn, jn, c, cstride)
    }
}

func asmGemm8x4(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) == 0 { return false }
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+3 < jn; j += 4 {
            cptr := &C.Data[(ic+i)*C.Stride + jc + j]
            callKernel8x4(&aPack[i*kn], &bPack[j], kn, jn, cptr, C.Stride)
        }
    }
    // tails
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
    if len(aPack) == 0 || len(bPack) < 4 { return false }
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+7 < jn; j += 8 {
            base := (ic+i)*C.Stride + jc + j
            callKernel8x4(&aPack[i*kn], &bPack[j],   kn, jn, &C.Data[base+0], C.Stride)
            callKernel8x4(&aPack[i*kn], &bPack[j+4], kn, jn, &C.Data[base+4], C.Stride)
        }
    }
    // tails
    iTail := (im / 8) * 8
    jTail := (jn / 8) * 8
        if jn-jTail >= 4 {
            for i := 0; i+7 < im; i += 8 {
                callKernel8x4(&aPack[i*kn], &bPack[jTail], kn, jn, &C.Data[(ic+i)*C.Stride + jc + jTail], C.Stride)
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
            callKernel8x4(&aPack[i*kn], &bPack[j+0],  kn, jn, &C.Data[base+0],  C.Stride)
            callKernel8x4(&aPack[i*kn], &bPack[j+4],  kn, jn, &C.Data[base+4],  C.Stride)
            callKernel8x4(&aPack[i*kn], &bPack[j+8],  kn, jn, &C.Data[base+8],  C.Stride)
            callKernel8x4(&aPack[i*kn], &bPack[j+12], kn, jn, &C.Data[base+12], C.Stride)
        }
    }
    // tails
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
