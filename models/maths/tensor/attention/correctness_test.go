package attention

import (
    "math"
    "testing"
)

// naiveAttention computes softmax(QK^T*scale) V directly (reference, slow)
func naiveAttention(Q, K, V [][]float64, scale float64) [][]float64 {
    m, d := len(Q), 0
    if m > 0 { d = len(Q[0]) }
    n := len(K)
    dv := len(V[0])
    // logits m x n
    logits := make([][]float64, m)
    for i := 0; i < m; i++ {
        logits[i] = make([]float64, n)
        for j := 0; j < n; j++ {
            s := 0.0
            for p := 0; p < d; p++ { s += Q[i][p]*K[j][p] }
            logits[i][j] = s * scale
        }
    }
    // softmax rows and multiply V
    O := make([][]float64, m)
    for i := 0; i < m; i++ {
        // softmax
        maxv := math.Inf(-1)
        for j := 0; j < n; j++ { if logits[i][j] > maxv { maxv = logits[i][j] } }
        denom := 0.0
        for j := 0; j < n; j++ { denom += math.Exp(logits[i][j]-maxv) }
        if denom == 0 { denom = 1 }
        O[i] = make([]float64, dv)
        inv := 1.0/denom
        for j := 0; j < n; j++ {
            w := math.Exp(logits[i][j]-maxv) * inv
            for c := 0; c < dv; c++ { O[i][c] += w * V[j][c] }
        }
    }
    return O
}

// randMat64 provided in bench_test.go for this package

func TestFlashAttention_Correctness(t *testing.T) {
    cases := [][4]int{{2,3,4,5}, {4,8,16,8}, {5,6,7,3}}
    for _, cs := range cases {
        m,d,n,dv := cs[0],cs[1],cs[2],cs[3]
        Q := randMat64(m,d)
        K := randMat64(n,d)
        V := randMat64(n,dv)
        scale := 1.0/math.Sqrt(float64(d))
        want := naiveAttention(Q,K,V,scale)
        got := FlashAttention2D(Q,K,V,scale)
        for i := 0; i < m; i++ {
            for c := 0; c < dv; c++ {
                if diff := math.Abs(got[i][c]-want[i][c]); diff > 1e-6 {
                    t.Fatalf("m=%d d=%d n=%d dv=%d mismatch at (%d,%d): got=%g want=%g diff=%g", m,d,n,dv,i,c, got[i][c], want[i][c], diff)
                }
            }
        }
    }
}
