//go:build arm64 && !neonasm
// +build arm64,!neonasm

package gemm

import "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"

// For now, provide Go-implemented microkernel-like tiling under arm64 build
// to exercise the asm path; can be replaced with real NEON .s later.

func asmGemm8x4(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) == 0 { return false }
    // Tile rows in steps of 8 and columns in steps of 4
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+3 < jn; j += 4 {
            // Accumulate 8x4
            for r := 0; r < 8; r++ {
                for c := 0; c < 4; c++ {
                    sum := 0.0
                    for p := 0; p < kn; p++ {
                        sum += aPack[(i+r)*kn+p] * bPack[p*jn + (j+c)]
                    }
                    C.Data[(ic+i+r)*C.Stride + jc + j + c] += sum
                }
            }
        }
    }
    // Generic tails
    iTail := (im / 8) * 8
    jTail := (jn / 4) * 4
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
        for j := jTail; j < jn; j++ {
            sum := 0.0
            for p := 0; p < kn; p++ {
                sum += aPack[i*kn+p] * bPack[p*jn+j]
            }
            C.Data[(ic+i)*C.Stride + jc + j] += sum
        }
    }
    return true
}

func asmGemm8x8(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    // Use 8x4 twice where possible
    handled := false
    if jn >= 8 {
        handled = true
        for i := 0; i+7 < im; i += 8 {
            for j := 0; j+7 < jn; j += 8 {
                // Left 4
                for r := 0; r < 8; r++ {
                    for c := 0; c < 4; c++ {
                        sum := 0.0
                        for p := 0; p < kn; p++ {
                            sum += aPack[(i+r)*kn+p] * bPack[p*jn + (j+c)]
                        }
                        C.Data[(ic+i+r)*C.Stride + jc + j + c] += sum
                    }
                }
                // Right 4
                for r := 0; r < 8; r++ {
                    for c := 4; c < 8; c++ {
                        sum := 0.0
                        for p := 0; p < kn; p++ {
                            sum += aPack[(i+r)*kn+p] * bPack[p*jn + (j+c)]
                        }
                        C.Data[(ic+i+r)*C.Stride + jc + j + c] += sum
                    }
                }
            }
        }
    }
    if !handled {
        return false
    }
    // Tails
    iTail := (im / 8) * 8
    jTail8 := (jn / 8) * 8
    // rows tail
    for i := iTail; i < im; i++ {
        for j := 0; j < jn; j++ {
            sum := 0.0
            for p := 0; p < kn; p++ {
                sum += aPack[i*kn+p] * bPack[p*jn+j]
            }
            C.Data[(ic+i)*C.Stride + jc + j] += sum
        }
    }
    // cols tail
    for i := 0; i < iTail; i++ {
        for j := jTail8; j < jn; j++ {
            sum := 0.0
            for p := 0; p < kn; p++ {
                sum += aPack[i*kn+p] * bPack[p*jn+j]
            }
            C.Data[(ic+i)*C.Stride + jc + j] += sum
        }
    }
    return true
}

func asmGemm8x16(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    if len(aPack) == 0 || len(bPack) < 16 { return false }
    for i := 0; i+7 < im; i += 8 {
        for j := 0; j+15 < jn; j += 16 {
            // Left 8
            for r := 0; r < 8; r++ {
                for c := 0; c < 8; c++ {
                    sum := 0.0
                    for p := 0; p < kn; p++ {
                        sum += aPack[(i+r)*kn+p] * bPack[p*jn + (j+c)]
                    }
                    C.Data[(ic+i+r)*C.Stride + jc + j + c] += sum
                }
            }
            // Right 8
            for r := 0; r < 8; r++ {
                for c := 8; c < 16; c++ {
                    sum := 0.0
                    for p := 0; p < kn; p++ {
                        sum += aPack[(i+r)*kn+p] * bPack[p*jn + (j+c)]
                    }
                    C.Data[(ic+i+r)*C.Stride + jc + j + c] += sum
                }
            }
        }
    }
    // tails
    iTail := (im / 8) * 8
    jTail := (jn / 16) * 16
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
        for j := jTail; j < jn; j++ {
            sum := 0.0
            for p := 0; p < kn; p++ {
                sum += aPack[i*kn+p] * bPack[p*jn+j]
            }
            C.Data[(ic+i)*C.Stride + jc + j] += sum
        }
    }
    return true
}
