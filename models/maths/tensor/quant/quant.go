package quant

import (
    "fmt"
    "math"
)

// Float16 represents a 16-bit floating point number (simplified)
type Float16 uint16

// Float32ToFloat16 converts float32 to float16 (simplified)
func Float32ToFloat16(f float32) Float16 {
    bits := math.Float32bits(f)
    sign := bits >> 31
    exp := ((bits >> 23) & 0xFF) - 127 + 15
    mantissa := (bits >> 13) & 0x3FF
    if exp <= 0 { return Float16(sign << 15) }
    if exp >= 31 { return Float16((sign << 15) | 0x7C00) }
    return Float16((sign << 15) | (uint32(exp) << 10) | mantissa)
}

// Float16ToFloat32 converts float16 to float32
func Float16ToFloat32(h Float16) float32 {
    bits := uint32(h)
    sign := bits >> 15
    exp := (bits >> 10) & 0x1F
    mantissa := bits & 0x3FF
    switch exp {
    case 0:
        if mantissa == 0 { return math.Float32frombits(sign << 31) }
        exp = 1
    case 31:
        return math.Float32frombits((sign << 31) | 0x7F800000 | (mantissa << 13))
    }
    exp = exp - 15 + 127
    return math.Float32frombits((sign << 31) | (exp << 23) | (mantissa << 13))
}

// MixedPrecisionGEMM performs matrix multiplication with FP16 accumulation
func MixedPrecisionGEMM(A, B [][]float64) ([][]float64, error) {
    m, k1 := len(A), len(A[0])
    k2, n := len(B), len(B[0])
    if k1 != k2 { return nil, fmt.Errorf("dimension mismatch") }
    k := k1
    C := make([][]float64, m)
    for i := 0; i < m; i++ {
        C[i] = make([]float64, n)
        for j := 0; j < n; j++ {
            var sum float32
            for p := 0; p < k; p++ { sum += float32(A[i][p]) * float32(B[p][j]) }
            C[i][j] = float64(sum)
        }
    }
    return C, nil
}

// QuantizeInt8 quantizes float64 matrix to int8 with scale factor
func QuantizeInt8(A [][]float64) ([][]int8, float64, error) {
    m, n := len(A), len(A[0])
    maxAbs := 0.0
    for i := 0; i < m; i++ { for j := 0; j < n; j++ { if abs := math.Abs(A[i][j]); abs > maxAbs { maxAbs = abs } } }
    if maxAbs == 0 { return nil, 0, fmt.Errorf("cannot quantize zero matrix") }
    scale := 127.0 / maxAbs
    Q := make([][]int8, m)
    for i := 0; i < m; i++ { Q[i] = make([]int8, n); for j:=0;j<n;j++{ Q[i][j] = int8(math.Round(A[i][j]*scale)) } }
    return Q, scale, nil
}

// DequantizeInt8 converts int8 matrix back to float64
func DequantizeInt8(Q [][]int8, scale float64) [][]float64 {
    m, n := len(Q), len(Q[0])
    R := make([][]float64, m)
    for i:=0;i<m;i++{ R[i]=make([]float64,n); for j:=0;j<n;j++{ R[i][j] = float64(Q[i][j])/scale } }
    return R
}

// Int8MatMul performs quantized matmul and dequantizes output
func Int8MatMul(A, B [][]int8, scaleA, scaleB float64) ([][]float64, error) {
    m, k1 := len(A), len(A[0]); k2, n := len(B), len(B[0])
    if k1 != k2 { return nil, fmt.Errorf("dimension mismatch") }
    k := k1
    C := make([][]float64, m)
    for i := 0; i < m; i++ { C[i] = make([]float64, n); for j:=0;j<n;j++{ var sum int32; for p:=0;p<k;p++{ sum += int32(A[i][p]) * int32(B[p][j]) }; C[i][j] = float64(sum)/(scaleA*scaleB) } }
    return C, nil
}

