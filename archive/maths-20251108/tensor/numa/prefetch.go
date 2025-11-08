package numa

import (
    par "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/parallel"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
)

// PrefetchMatrix prefetches matrix data into cache
func PrefetchMatrix(A [][]float64, startRow, endRow int) {
    for i := startRow; i < endRow; i++ {
        _ = A[i][0]
        for j := 0; j < len(A[i]); j += 8 { _ = A[i][j] }
    }
}

// CacheBlockedMatMulPrefetch performs cache-blocked matmul with simple prefetching
func CacheBlockedMatMulPrefetch(A, B [][]float64, cfg *par.ParallelConfig) ([][]float64, error) {
    m, k := len(A), len(A[0])
    n := len(B[0])
    C := make([][]float64, m)
    for i := 0; i < m; i++ { C[i] = make([]float64, n) }
    const blockSize = 64
    const prefetchDistance = 8
    par.ParallelFor(0, m, cfg, func(ii int) {
        if ii+prefetchDistance < m { PrefetchMatrix(A, ii+prefetchDistance, ii+prefetchDistance+1) }
        for jj := 0; jj < n; jj += blockSize {
            for kk := 0; kk < k; kk += blockSize {
                iEnd := ints.Min(ii+1, m)
                jEnd := ints.Min(jj+blockSize, n)
                kEnd := ints.Min(kk+blockSize, k)
                for i := ii; i < iEnd; i++ {
                    for j := jj; j < jEnd; j++ {
                        sum := C[i][j]
                        for p := kk; p < kEnd; p++ { sum += A[i][p] * B[p][j] }
                        C[i][j] = sum
                    }
                }
            }
        }
    })
    return C, nil
}

