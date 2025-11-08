package quant

// QuantizedMatrix represents an INT8 quantized matrix with scale and zero-point.
type QuantizedMatrix struct {
    Data  [][]int8
    Scale float64
    Zero  int8
}

// QuantizeMatrix quantizes float64 matrix to INT8 (symmetric-ish per-matrix scale).
func QuantizeMatrix(A [][]float64) *QuantizedMatrix {
    m, n := len(A), len(A[0])
    // Find min/max for quantization
    minVal, maxVal := A[0][0], A[0][0]
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if A[i][j] < minVal { minVal = A[i][j] }
            if A[i][j] > maxVal { maxVal = A[i][j] }
        }
    }
    scale := (maxVal - minVal) / 255.0
    if scale == 0 { scale = 1 }
    zero := int8(-minVal / scale)
    // Quantize
    data := make([][]int8, m)
    for i := 0; i < m; i++ {
        data[i] = make([]int8, n)
        for j := 0; j < n; j++ {
            data[i][j] = int8(A[i][j]/scale) + zero
        }
    }
    return &QuantizedMatrix{ Data: data, Scale: scale, Zero: zero }
}

// QuantizedMatMul performs INT8 matmul and returns INT8 with composed scale.
func QuantizedMatMul(A, B *QuantizedMatrix) (*QuantizedMatrix, error) {
    m, k := len(A.Data), len(A.Data[0])
    n := len(B.Data[0])
    // INT32 accumulation
    Cacc := make([][]int32, m)
    for i := 0; i < m; i++ { Cacc[i] = make([]int32, n) }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            sum := int32(0)
            for p := 0; p < k; p++ { sum += int32(A.Data[i][p]) * int32(B.Data[p][j]) }
            Cacc[i][j] = sum
        }
    }
    // Re-quantize
    scale := A.Scale * B.Scale
    if scale == 0 { scale = 1 }
    out := make([][]int8, m)
    for i := 0; i < m; i++ {
        out[i] = make([]int8, n)
        for j := 0; j < n; j++ {
            out[i][j] = int8(float64(Cacc[i][j]) / scale)
        }
    }
    return &QuantizedMatrix{ Data: out, Scale: scale, Zero: 0 }, nil
}

