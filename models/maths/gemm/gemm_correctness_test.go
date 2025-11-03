package gemm

import (
    "testing"
)

func naiveGemm(A, B [][]float64) [][]float64 {
    m, k, n := len(A), 0, 0
    if m > 0 { k = len(A[0]) }
    if k > 0 { n = len(B[0]) }
    C := make([][]float64, m)
    for i := 0; i < m; i++ { C[i] = make([]float64, n) }
    for i := 0; i < m; i++ {
        for p := 0; p < k; p++ {
            a := A[i][p]
            bp := B[p]
            for j := 0; j < n; j++ {
                C[i][j] += a * bp[j]
            }
        }
    }
    return C
}

func TestMatMul2DCorrectness(t *testing.T) {
    sizes := []int{1, 2, 3, 5, 8, 17, 31}
    for _, m := range sizes {
        for _, k := range sizes {
            for _, n := range sizes {
                A := randMat(m, k)
                B := randMat(k, n)
                got := MatMul2D(A, B)
                want := naiveGemm(A, B)
                if m == 0 || n == 0 || k == 0 { continue }
                for i := 0; i < m; i++ {
                    for j := 0; j < n; j++ {
                        if diff := abs(got[i][j]-want[i][j]); diff > 1e-9 {
                            t.Fatalf("m=%d k=%d n=%d mismatch at (%d,%d): got=%g want=%g diff=%g", m,k,n,i,j, got[i][j], want[i][j], diff)
                        }
                    }
                }
            }
        }
    }
}

func abs(x float64) float64 { if x < 0 { return -x }; return x }
