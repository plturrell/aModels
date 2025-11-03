package gemm

import (
    "fmt"
    "math/rand"
    "testing"
)

func randMat(rows, cols int) [][]float64 {
    m := make([][]float64, rows)
    for i := 0; i < rows; i++ {
        r := make([]float64, cols)
        for j := 0; j < cols; j++ { r[j] = rand.NormFloat64() }
        m[i] = r
    }
    return m
}

func BenchmarkGemmPacked_256(b *testing.B) {
    A := randMat(256, 256)
    B := randMat(256, 256)
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = MatMul2D(A, B)
    }
}

func BenchmarkGemmPacked_512(b *testing.B) {
    A := randMat(512, 512)
    B := randMat(512, 512)
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = MatMul2D(A, B)
    }
}

func BenchmarkGemmPacked_1024(b *testing.B) {
    A := randMat(1024, 1024)
    B := randMat(1024, 1024)
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = MatMul2D(A, B)
    }
}

func BenchmarkGemmPacked_Grid(b *testing.B) {
    grids := [][3]int{
        {128, 128, 128},
        {256, 256, 256},
        {256, 512, 256},
        {512, 256, 512},
        {512, 512, 512},
        {1024, 512, 512},
        {1024, 1024, 1024},
    }
    for _, g := range grids {
        m, k, n := g[0], g[1], g[2]
        b.Run(
            fmt.Sprintf("m%dk%dn%d", m, k, n),
            func(b *testing.B) {
                A := randMat(m, k)
                B := randMat(k, n)
                b.ResetTimer()
                for i := 0; i < b.N; i++ { _ = MatMul2D(A, B) }
            },
        )
    }
}
