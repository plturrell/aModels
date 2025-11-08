package fusions

import (
    "errors"
    "math"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
    simd "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/simd"
    lanes "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/lanes"
    cpu "golang.org/x/sys/cpu"
)

// MatMulBiasGELU computes C = GELU(A*B + bias).
func MatMulBiasGELU(A, B [][]float64, bias []float64) ([][]float64, error) {
    m, k1 := len(A), len(A[0])
    k2, n := len(B), len(B[0])
    if k1 != k2 || len(bias) != n { return nil, errors.New("dimension mismatch") }
    C := make([][]float64, m)
    for i := 0; i < m; i++ { C[i] = make([]float64, n) }
    const blockI, blockJ, blockK = 64,64,64
    for ii := 0; ii < m; ii += blockI {
        iEnd := ints.Min(ii+blockI, m)
        for jj := 0; jj < n; jj += blockJ {
            jEnd := ints.Min(jj+blockJ, n)
            for kk := 0; kk < k1; kk += blockK {
                kEnd := ints.Min(kk+blockK, k1)
                for i := ii; i < iEnd; i++ {
                    for j := jj; j < jEnd; j++ {
                        sum := C[i][j]
                        for p := kk; p < kEnd; p++ { sum += A[i][p]*B[p][j] }
                        if kk+blockK >= k1 || kEnd == k1 {
                            z := sum + bias[j]
                            C[i][j] = geluFast(z)
                        } else {
                            C[i][j] = sum
                        }
                    }
                }
            }
        }
    }
    return C, nil
}

// MatMulBiasSigmoid computes C = sigmoid(A*B + bias)
func MatMulBiasSigmoid(A, B [][]float64, bias []float64) ([][]float64, error) {
    m, k1 := len(A), len(A[0])
    k2, n := len(B), len(B[0])
    if k1 != k2 || len(bias) != n { return nil, errors.New("dimension mismatch") }
    C := make([][]float64, m)
    for i := 0; i < m; i++ { C[i] = make([]float64, n) }
    const blockI, blockJ, blockK = 64,64,64
    for ii := 0; ii < m; ii += blockI {
        iEnd := ints.Min(ii+blockI, m)
        for jj := 0; jj < n; jj += blockJ {
            jEnd := ints.Min(jj+blockJ, n)
            for kk := 0; kk < k1; kk += blockK {
                kEnd := ints.Min(kk+blockK, k1)
                for i := ii; i < iEnd; i++ {
                    for j := jj; j < jEnd; j++ {
                        sum := C[i][j]
                        for p := kk; p < kEnd; p++ { sum += A[i][p]*B[p][j] }
                        if kk+blockK >= k1 || kEnd == k1 {
                            z := sum + bias[j]
                            C[i][j] = 1.0 / (1.0 + math.Exp(-z))
                        } else {
                            C[i][j] = sum
                        }
                    }
                }
            }
        }
    }
    return C, nil
}

// MatMulBiasTanh computes C = tanh(A*B + bias)
func MatMulBiasTanh(A, B [][]float64, bias []float64, tanh func(float64) float64) ([][]float64, error) {
    m, k1 := len(A), len(A[0])
    k2, n := len(B), len(B[0])
    if k1 != k2 || len(bias) != n { return nil, errors.New("dimension mismatch") }
    C := make([][]float64, m)
    for i := 0; i < m; i++ { C[i] = make([]float64, n) }
    const blockI, blockJ, blockK = 64,64,64
    for ii := 0; ii < m; ii += blockI {
        iEnd := ints.Min(ii+blockI, m)
        for jj := 0; jj < n; jj += blockJ {
            jEnd := ints.Min(jj+blockJ, n)
            for kk := 0; kk < k1; kk += blockK {
                kEnd := ints.Min(kk+blockK, k1)
                for i := ii; i < iEnd; i++ {
                    for j := jj; j < jEnd; j++ {
                        sum := C[i][j]
                        for p := kk; p < kEnd; p++ { sum += A[i][p]*B[p][j] }
                        if kk+blockK >= k1 || kEnd == k1 {
                            z := sum + bias[j]
                            C[i][j] = tanh(z)
                        } else {
                            C[i][j] = sum
                        }
                    }
                }
            }
        }
    }
    return C, nil
}

// float32 variants
func MatMulBiasGELU32(A, B [][]float32, bias []float32, tanhF32 func(float32) float32) ([][]float32, error) {
    m, k1 := len(A), len(A[0])
    k2, n := len(B), len(B[0])
    if k1 != k2 || len(bias) != n { return nil, errors.New("dimension mismatch") }
    C := make([][]float32, m)
    for i := 0; i < m; i++ { C[i] = make([]float32, n) }
    const blockI, blockJ, blockK = 64,64,64
    for ii := 0; ii < m; ii += blockI {
        iEnd := min(ii+blockI, m)
        for jj := 0; jj < n; jj += blockJ {
            jEnd := min(jj+blockJ, n)
            for kk := 0; kk < k1; kk += blockK {
                kEnd := min(kk+blockK, k1)
                for i := ii; i < iEnd; i++ {
                    row := A[i][kk:kEnd]
                    for j := jj; j < jEnd; j++ {
                        colLen := kEnd - kk
                        col := make([]float32, colLen)
                        for p := 0; p < colLen; p++ { col[p] = B[kk+p][j] }
                        s := dotF32SIMD(row, col)
                        if kk == 0 { C[i][j] = s } else { C[i][j] += s }
                        if kk+blockK >= k1 || kEnd == k1 {
                            z := float32(float64(C[i][j] + bias[j]))
                            const c0 = float32(0.7978845608028654)
                            const c1 = float32(0.044715)
                            yz := c0 * (z + c1*z*z*z)
                            C[i][j] = 0.5 * z * (1 + tanhF32(yz))
                        }
                    }
                }
            }
        }
    }
    return C, nil
}

func MatMulBiasTanh32(A, B [][]float32, bias []float32, tanhF32 func(float32) float32) ([][]float32, error) {
    m, k1 := len(A), len(A[0])
    k2, n := len(B), len(B[0])
    if k1 != k2 || len(bias) != n { return nil, errors.New("dimension mismatch") }
    C := make([][]float32, m)
    for i := 0; i < m; i++ { C[i] = make([]float32, n) }
    const blockI, blockJ, blockK = 64,64,64
    for ii := 0; ii < m; ii += blockI {
        iEnd := min(ii+blockI, m)
        for jj := 0; jj < n; jj += blockJ {
            jEnd := min(jj+blockJ, n)
            for kk := 0; kk < k1; kk += blockK {
                kEnd := min(kk+blockK, k1)
                for i := ii; i < iEnd; i++ {
                    row := A[i][kk:kEnd]
                    for j := jj; j < jEnd; j++ {
                        colLen := kEnd - kk
                        col := make([]float32, colLen)
                        for p := 0; p < colLen; p++ { col[p] = B[kk+p][j] }
                        s := dotF32SIMD(row, col)
                        if kk == 0 { C[i][j] = s } else { C[i][j] += s }
                        if kk+blockK >= k1 || kEnd == k1 {
                            z := float32(float64(C[i][j] + bias[j]))
                            C[i][j] = float32(math.Tanh(float64(z)))
                        }
                    }
                }
            }
        }
    }
    return C, nil
}

func dotF32SIMD(a, b []float32) float32 {
    n := len(a)
    if len(b) != n { panic("fusions.dotF32SIMD: length mismatch") }
    // Prefer AVX2 8-wide dot on amd64
    if cpu.X86.HasAVX2 {
        return lanes.DotF32_AVX2(a, b)
    }
    // Fallback: use FMA + scalar reduce
    const chunk = 256
    var sum float32
    zeros := make([]float32, chunk)
    i := 0
    for ; i+chunk <= n; i += chunk {
        prod := simd.SIMDFusedMultiplyAddF32(a[i:i+chunk], b[i:i+chunk], zeros)
        for _, v := range prod { sum += v }
    }
    if i < n {
        rem := n - i
        prod := simd.SIMDFusedMultiplyAddF32(a[i:], b[i:], zeros[:rem])
        for _, v := range prod { sum += v }
    }
    return sum
}

func geluFast(x float64) float64 {
    const c0 = 0.7978845608028654
    const c1 = 0.044715
    y := c0 * (x + c1*x*x*x)
    return 0.5 * x * (1 + math.Tanh(y))
}

// min moved to internal/ints
