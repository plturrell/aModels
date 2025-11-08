package fusions

import (
    "math"
    "runtime"
    lanes "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/lanes"
    cpu "golang.org/x/sys/cpu"
)

// SoftmaxCrossEntropy computes softmax and cross-entropy in one pass.
func SoftmaxCrossEntropy(logits [][]float64, labels []int) (loss float64, probs [][]float64) {
    m, n := len(logits), len(logits[0])
    probs = make([][]float64, m)
    for i := 0; i < m; i++ {
        probs[i] = make([]float64, n)
        // Max using arch-specific reductions when available
        maxv := logits[i][0]
        if runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 {
            maxv = lanes.ReduceMaxF64_AVX2(logits[i])
        } else if runtime.GOARCH == "arm64" {
            maxv = lanes.ReduceMaxF64_NEON(logits[i])
        } else {
            for j := 1; j < n; j++ { if logits[i][j] > maxv { maxv = logits[i][j] } }
        }
        sum := 0.0
        for j := 0; j < n; j++ { v := math.Exp(logits[i][j]-maxv); probs[i][j] = v; sum += v }
        inv := 1.0 / sum
        for j := 0; j < n; j++ { probs[i][j] *= inv }
        y := labels[i]
        if y >= 0 && y < n { loss += -math.Log(probs[i][y] + 1e-12) }
    }
    loss /= float64(m)
    return
}

// AddMulExp computes out = exp(a + b*c).
func AddMulExp(a, b, c []float64) []float64 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("fusions.AddMulExp: length mismatch") }
    out := make([]float64, n)
    const block = 256
    for i := 0; i < n; i += block {
        end := i + block
        if end > n { end = n }
        if end < n { _ = a[end-1] } // light prefetch
        for j := i; j < end; j++ { out[j] = math.Exp(a[j] + b[j]*c[j]) }
    }
    return out
}

// SoftmaxRow64 computes softmax over a single float64 row.
func SoftmaxRow64(x []float64) []float64 {
    n := len(x)
    out := make([]float64, n)
    if n == 0 { return out }
    maxv := x[0]
    if runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 { maxv = lanes.ReduceMaxF64_AVX2(x) } else if runtime.GOARCH == "arm64" { maxv = lanes.ReduceMaxF64_NEON(x) } else {
        for i := 1; i < n; i++ { if x[i] > maxv { maxv = x[i] } }
    }
    sum := 0.0
    for i := 0; i < n; i++ { v := math.Exp(x[i]-maxv); out[i] = v; sum += v }
    if sum == 0 { return out }
    inv := 1.0 / sum
    for i := 0; i < n; i++ { out[i] *= inv }
    return out
}

// Softmax2D64 applies softmax per row on [][]float64.
func Softmax2D64(X [][]float64) [][]float64 {
    m := len(X)
    Y := make([][]float64, m)
    for i := 0; i < m; i++ { Y[i] = SoftmaxRow64(X[i]) }
    return Y
}

// SoftmaxRow32 computes softmax over a single float32 row.
func SoftmaxRow32(x []float32) []float32 {
    n := len(x)
    out := make([]float32, n)
    if n == 0 { return out }
    // max for numerical stability
    var maxv float32
    if runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 {
        maxv = lanes.ReduceMaxF32_AVX2(x)
    } else if runtime.GOARCH == "arm64" {
        maxv = lanes.ReduceMaxF32_NEON(x)
    } else {
        maxv = x[0]
        for i := 1; i < n; i++ { if x[i] > maxv { maxv = x[i] } }
    }
    // exponentiate and sum
    // exponentiate and sum
    var sum float32
    for i := 0; i < n; i++ {
        v := float32(math.Exp(float64(x[i] - maxv)))
        out[i] = v
        sum += v
    }
    if sum == 0 { return out }
    inv := 1.0 / sum
    for i := 0; i < n; i++ { out[i] *= float32(inv) }
    return out
}

// Softmax2D32 applies softmax per row on [][]float32.
func Softmax2D32(X [][]float32) [][]float32 {
    m := len(X)
    Y := make([][]float32, m)
    for i := 0; i < m; i++ { Y[i] = SoftmaxRow32(X[i]) }
    return Y
}

// SoftmaxCrossEntropy32 computes softmax and cross-entropy (float32 variant) in one pass.
func SoftmaxCrossEntropy32(logits [][]float32, labels []int) (loss float32, probs [][]float32) {
    m := len(logits)
    if m == 0 { return 0, make([][]float32, 0) }
    n := len(logits[0])
    probs = make([][]float32, m)
    for i := 0; i < m; i++ {
        row := logits[i]
        // max
        var maxv float32
        if runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 { maxv = lanes.ReduceMaxF32_AVX2(row) } else if runtime.GOARCH == "arm64" { maxv = lanes.ReduceMaxF32_NEON(row) } else { maxv = row[0]; for j := 1; j < n; j++ { if row[j] > maxv { maxv = row[j] } } }
        // exp and sum
        probs[i] = make([]float32, n)
        var sum float32
        for j := 0; j < n; j++ {
            v := float32(math.Exp(float64(row[j] - maxv)))
            probs[i][j] = v
            sum += v
        }
        if sum == 0 { sum = 1 }
        inv := 1.0 / sum
        for j := 0; j < n; j++ { probs[i][j] *= float32(inv) }
        y := labels[i]
        if y >= 0 && y < n {
            // loss accum as float64 for numerical stability, then cast
            loss += float32(-math.Log(float64(probs[i][y]) + 1e-12))
        }
    }
    loss /= float32(m)
    return
}
