//go:build !amd64 && !arm64
// +build !amd64,!arm64

package gemm

import "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"

// asmGemm8x8 tries to compute the full (im x jn) block using 8x8 tiles with an
// architecture-specific microkernel. Returns true if handled, false to fall back to Go.
func asmGemm8x8(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    return false
}

// asmGemm8x4 tries to compute the full (im x jn) block using 8x4 tiles with an
// architecture-specific microkernel. Returns true if handled, false to fall back to Go.
func asmGemm8x4(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    return false
}

// asmGemm8x16 tries to compute the full (im x jn) block using 8x16 tiles with an
// architecture-specific microkernel. Returns true if handled, false to fall back to Go.
func asmGemm8x16(aPack, bPack []float64, C *util.Matrix64, ic, jc, im, jn, kn int) bool {
    return false
}
