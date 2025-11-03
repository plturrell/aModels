package attention

import (
    "math"
    "os"
    "runtime"
    "strconv"
    lanes "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/lanes"
    cpu "golang.org/x/sys/cpu"
)

// FlashAttention2D32 computes O = softmax(Q K^T * scale) V for float32 matrices.
// Q: (m x d), K: (n x d), V: (n x dv) → O: (m x dv)
// Implementation mirrors the float64 version but uses AVX2-optimized dot/reductions when available.
func FlashAttention2D32(Q, K, V [][]float32, scale float32) [][]float32 {
    m := len(Q)
    if m == 0 { return make([][]float32, 0) }
    d := len(Q[0])
    n := len(K)
    if n == 0 || len(K[0]) != d || len(V) != n { return make([][]float32, 0) }
    dv := len(V[0])
    if dv == 0 { return make([][]float32, m) }

    O := make([][]float32, m)
    for i := 0; i < m; i++ { O[i] = make([]float32, dv) }

    // Threading
    threads := runtime.GOMAXPROCS(0)
    if s := os.Getenv("MATHS_ATTENTION_THREADS"); s != "" {
        if v, err := strconv.Atoi(s); err == nil && v > 0 { threads = v }
    }
    parts := partitionRange(m, threads)

    type nothing struct{}
    done := make(chan nothing, len(parts))
    for _, p := range parts {
        if p.start >= p.end { done <- nothing{}; continue }
        st, en := p.start, p.end
        go func() {
            // Local accumulator in float64 for stability
            acc := make([]float64, dv)
            for i := st; i < en; i++ {
                qi := Q[i]
                // 1) max logit
                maxLogit := float32(math.Inf(-1))
                for j := 0; j < n; j++ {
                    s := dotF32(qi, K[j]) * scale
                    if s > maxLogit { maxLogit = s }
                }
                // 2) denom and acc
                for c := range acc { acc[c] = 0 }
                denom := 0.0
                for j := 0; j < n; j++ {
                    s := dotF32(qi, K[j]) * scale
                    w := math.Exp(float64(s - maxLogit))
                    denom += w
                    vj := V[j]
                    for c := 0; c < dv; c++ { acc[c] += w * float64(vj[c]) }
                }
                if denom == 0 { denom = 1 }
                inv := 1.0 / denom
                out := O[i]
                for c := 0; c < dv; c++ { out[c] = float32(acc[c] * inv) }
            }
            done <- nothing{}
        }()
    }
    for range parts { <-done }
    return O
}

func dotF32(a, b []float32) float32 {
    if len(a) != len(b) { return 0 }
    if runtime.GOARCH == "amd64" && cpu.X86.HasAVX2 { return lanes.DotF32_AVX2(a, b) }
    if runtime.GOARCH == "arm64" { return lanes.LaneDotF32_NEON(a, b) }
    s := float32(0)
    i := 0
    n := len(a)
    for ; i+7 < n; i += 8 {
        s += a[i+0]*b[i+0] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
             a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
    }
    for ; i < n; i++ { s += a[i]*b[i] }
    return s
}
