package conv

import (
    "fmt"
)

// standardConv2D performs a straightforward 2D convolution with stride.
func standardConv2D(input, kernel [][]float64, stride int) ([][]float64, error) {
    if stride <= 0 { stride = 1 }
    inH, inW := len(input), len(input[0])
    kH, kW := len(kernel), len(kernel[0])
    if inH < kH || inW < kW { return nil, fmt.Errorf("kernel larger than input") }
    outH := (inH-kH)/stride + 1
    outW := (inW-kW)/stride + 1
    if outH <= 0 || outW <= 0 { return nil, fmt.Errorf("invalid output shape") }
    out := make([][]float64, outH)
    for i := 0; i < outH; i++ {
        out[i] = make([]float64, outW)
        for j := 0; j < outW; j++ {
            s := 0.0
            for ki := 0; ki < kH; ki++ {
                for kj := 0; kj < kW; kj++ {
                    s += input[i*stride+ki][j*stride+kj] * kernel[ki][kj]
                }
            }
            out[i][j] = s
        }
    }
    return out, nil
}

// WinogradConv2D provides a Winograd-API-compatible entry; currently delegates
// to standard convolution for correctness (keep API stable after refactor).
func WinogradConv2D(input [][]float64, kernel [][]float64, stride int) ([][]float64, error) {
    return standardConv2D(input, kernel, stride)
}

