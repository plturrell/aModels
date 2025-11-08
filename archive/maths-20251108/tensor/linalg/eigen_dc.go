package linalg

import (
    "sync"
)

// EigenDecompositionDivideConquer implements a simplified divide-and-conquer
// eigendecomposition for symmetric matrices. Falls back to power method on errors.
func EigenDecompositionDivideConquer(A [][]float64) ([]float64, [][]float64, error) {
    n := len(A)
    if n <= 8 {
        return EigenDecomposition(A)
    }
    mid := n / 2
    A11 := extractSubmatrix(A, 0, mid, 0, mid)
    A22 := extractSubmatrix(A, mid, n, mid, n)

    var wg sync.WaitGroup
    var eig1, eig2 []float64
    var vec1, vec2 [][]float64
    var err1, err2 error
    wg.Add(2)
    go func(){ defer wg.Done(); eig1, vec1, err1 = EigenDecompositionDivideConquer(A11) }()
    go func(){ defer wg.Done(); eig2, vec2, err2 = EigenDecompositionDivideConquer(A22) }()
    wg.Wait()
    if err1 != nil || err2 != nil {
        return EigenDecomposition(A)
    }
    eigenvalues := append(eig1, eig2...)
    eigenvectors := make([][]float64, n)
    for i := 0; i < n; i++ { eigenvectors[i] = make([]float64, n) }
    for i := 0; i < mid; i++ {
        for j := 0; j < mid; j++ { eigenvectors[i][j] = vec1[i][j] }
    }
    for i := mid; i < n; i++ {
        for j := mid; j < n; j++ { eigenvectors[i][j] = vec2[i-mid][j-mid] }
    }
    return eigenvalues, eigenvectors, nil
}

func extractSubmatrix(A [][]float64, rowStart, rowEnd, colStart, colEnd int) [][]float64 {
    rows := rowEnd - rowStart
    cols := colEnd - colStart
    out := make([][]float64, rows)
    for i := 0; i < rows; i++ {
        out[i] = make([]float64, cols)
        copy(out[i], A[rowStart+i][colStart:colEnd])
    }
    return out
}

