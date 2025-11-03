package ops

import "math"

// SoftmaxRow computes a numerically-stable softmax of a row vector.
func SoftmaxRow(x []float64) []float64 {
    maxVal := x[0]
    for _, v := range x { if v > maxVal { maxVal = v } }
    out := make([]float64, len(x))
    sum := 0.0
    for i, v := range x { ev := math.Exp(v - maxVal); out[i] = ev; sum += ev }
    inv := 1.0 / (sum + 1e-12)
    for i := range out { out[i] *= inv }
    return out
}

// SoftmaxMatrix applies SoftmaxRow to each row.
func SoftmaxMatrix(X [][]float64) [][]float64 {
    m := len(X)
    R := make([][]float64, m)
    for i := 0; i < m; i++ { R[i] = SoftmaxRow(X[i]) }
    return R
}

// KahanSum does Kahan compensated summation.
func KahanSum(values []float64) float64 {
    sum := 0.0
    c := 0.0
    for _, v := range values {
        y := v - c
        t := sum + y
        c = (t - sum) - y
        sum = t
    }
    return sum
}

// Dot computes the dot product of two equal-length vectors.
func Dot(a, b []float64) float64 {
    s := 0.0
    for i := range a { s += a[i] * b[i] }
    return s
}

