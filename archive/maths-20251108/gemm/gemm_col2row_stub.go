//go:build ! (amd64 && avx2asm)

package gemm

import "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"

// Default stub when AVX2 asm is not available: do not handle col-pack via asm.
func tryAsmColPack(im, jn, kn int, aPack, bPack []float64, C *util.Matrix64, ic, jc int) bool {
    return false
}
