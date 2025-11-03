package attention

import (
    "math"
    "os"
    "runtime"
    "strconv"
    "sync"
    utilpkg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// FlashAttention computes O = softmax(Q K^T * scale) V without materializing the full
// attention matrix. It computes each output row independently with two passes over K,V.
// Q: (m x d), K: (n x d), V: (n x dv), returns O: (m x dv).
func FlashAttention(Q, K, V *utilpkg.Matrix64, scale float64) *utilpkg.Matrix64 {
    if Q == nil || K == nil || V == nil { return utilpkg.NewMatrix64(0,0) }
    if Q.Cols != K.Cols { return utilpkg.NewMatrix64(0,0) }
    if K.Rows != V.Rows { return utilpkg.NewMatrix64(0,0) }

    m, d := Q.Rows, Q.Cols
    n, _ := K.Rows, K.Cols
    dv := V.Cols
    O := utilpkg.NewMatrix64(m, dv)
    if m == 0 || n == 0 || d == 0 || dv == 0 { return O }

    // Threading
    threads := runtime.GOMAXPROCS(0)
    if s := os.Getenv("MATHS_ATTENTION_THREADS"); s != "" {
        if v, err := strconv.Atoi(s); err == nil && v > 0 { threads = v }
    }
    // Simple work partition by rows
    type rng struct{ start, end int }
    parts := partitionRange(m, threads)
    var wg sync.WaitGroup
    for _, p := range parts {
        if p.start >= p.end { continue }
        wg.Add(1)
        go func(st, en int) {
            defer wg.Done()
            // Local scratch for one row accumulation
            acc := make([]float64, dv)
            for i := st; i < en; i++ {
                // 1) Find max logit for numerical stability
                maxLogit := math.Inf(-1)
                qi := Q.Data[i*Q.Stride : i*Q.Stride+d]
                for j := 0; j < n; j++ {
                    kj := K.Data[j*K.Stride : j*K.Stride+d]
                    s := dotContiguous(qi, kj) * scale
                    if s > maxLogit { maxLogit = s }
                }
                // 2) Compute denominator and accumulate weighted V rows
                denom := 0.0
                // zero acc
                for t := range acc { acc[t] = 0 }
                for j := 0; j < n; j++ {
                    kj := K.Data[j*K.Stride : j*K.Stride+d]
                    vj := V.Data[j*V.Stride : j*V.Stride+dv]
                    s := dotContiguous(qi, kj) * scale
                    w := math.Exp(s - maxLogit)
                    denom += w
                    // axpy into acc
                    for c := 0; c < dv; c++ { acc[c] += w * vj[c] }
                }
                if denom == 0 { denom = 1 }
                // Normalize into output
                out := O.Data[i*O.Stride : i*O.Stride+dv]
                inv := 1.0 / denom
                for c := 0; c < dv; c++ { out[c] = acc[c] * inv }
            }
        }(p.start, p.end)
    }
    wg.Wait()
    return O
}

// FlashAttention2D is a [][] wrapper around FlashAttention.
func FlashAttention2D(Q, K, V [][]float64, scale float64) [][]float64 {
    q := utilpkg.From2D(Q)
    k := utilpkg.From2D(K)
    v := utilpkg.From2D(V)
    o := FlashAttention(q, k, v, scale)
    return utilpkg.To2D(o)
}

// dotContiguous computes dot(q, k) for equal-length slices.
func dotContiguous(a, b []float64) float64 {
    s := 0.0
    // Unroll by 4 for ILP
    i := 0
    n := len(a)
    for ; i+3 < n; i += 4 {
        s += a[i+0]*b[i+0] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
    }
    for ; i < n; i++ { s += a[i]*b[i] }
    return s
}

// partitionRange splits [0,n) into up to k parts.
func partitionRange(n, k int) []struct{ start, end int } {
    if k < 1 { k = 1 }
    if k > n { k = n }
    out := make([]struct{ start, end int }, k)
    base := n / k
    rem := n % k
    off := 0
    for i := 0; i < k; i++ {
        sz := base
        if i < rem { sz++ }
        out[i] = struct{ start, end int }{off, off + sz}
        off += sz
    }
    return out
}
