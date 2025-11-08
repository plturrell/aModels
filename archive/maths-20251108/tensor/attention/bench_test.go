package attention

import (
    "testing"
)

func randMat64(r, c int) [][]float64 {
    m := make([][]float64, r)
    for i := 0; i < r; i++ {
        row := make([]float64, c)
        for j := 0; j < c; j++ { row[j] = float64(i*j%17) - 8.0 }
        m[i] = row
    }
    return m
}

func BenchmarkFlashAttention_Small(b *testing.B) {
    m,d,n,dv := 128, 64, 128, 64
    Q := randMat64(m,d)
    K := randMat64(n,d)
    V := randMat64(n,dv)
    scale := 1.0/float64(d)
    b.ResetTimer()
    for i := 0; i < b.N; i++ { _ = FlashAttention2D(Q,K,V,scale) }
}

