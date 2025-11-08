package linalg

import (
    "math/rand"
    "testing"
    "time"
)

// genSym creates a symmetric n x n matrix with deterministic random values.
func genSym(n int, seed int64) [][]float64 {
    rnd := rand.New(rand.NewSource(seed))
    A := make([][]float64, n)
    for i := 0; i < n; i++ { A[i] = make([]float64, n) }
    for i := 0; i < n; i++ {
        for j := 0; j <= i; j++ {
            v := rnd.Float64()*2 - 1
            A[i][j] = v
            A[j][i] = v
        }
    }
    return A
}

func BenchmarkEigenDivideConquer(b *testing.B) {
    sizes := []int{64, 96, 128}
    for _, n := range sizes {
        b.Run(time.Now().Format("150405")+"_n"+string(rune(n)), func(b *testing.B) {
            A := genSym(n, 42)
            b.ResetTimer()
            for i := 0; i < b.N; i++ {
                _, _, _ = EigenDecompositionDivideConquer(A)
            }
        })
    }
}

