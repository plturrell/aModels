package attention

import (
    "testing"
)

func benchMat32(r, c int) [][]float32 {
    m := make([][]float32, r)
    v := float32(1)
    for i := 0; i < r; i++ {
        row := make([]float32, c)
        for j := 0; j < c; j++ { v = v*1.0001 + 0.0003; row[j] = v }
        m[i] = row
    }
    return m
}

func BenchmarkFlashAttention2D32_Small(b *testing.B) {
    m, d, n, dv := 128, 64, 128, 64
    Q := benchMat32(m, d)
    K := benchMat32(n, d)
    V := benchMat32(n, dv)
    scale := float32(1.0/float32(d))
    b.ReportAllocs()
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = FlashAttention2D32(Q, K, V, scale)
    }
}

