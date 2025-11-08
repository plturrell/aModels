package linalg

import (
    "fmt"
    "math"
)

// EigenDecomposition computes eigenvalues and eigenvectors of a symmetric matrix
// using power iteration with deflation. Returns values and column-major eigenvectors.
func EigenDecomposition(A [][]float64) ([]float64, [][]float64, error) {
    n := len(A)
    if n == 0 {
        return nil, nil, fmt.Errorf("empty matrix")
    }
    eigenvalues := make([]float64, n)
    eigenvectors := make([][]float64, n)
    for i := 0; i < n; i++ { eigenvectors[i] = make([]float64, n) }

    // Copy matrix for deflation
    Acopy := make([][]float64, n)
    for i := 0; i < n; i++ { Acopy[i] = append([]float64(nil), A[i]...) }

    for k := 0; k < n; k++ {
        v := make([]float64, n)
        for i := 0; i < n; i++ { v[i] = 1.0 / math.Sqrt(float64(n)) }
        const maxIter = 100
        const tol = 1e-10
        for iter := 0; iter < maxIter; iter++ {
            vNew := make([]float64, n)
            for i := 0; i < n; i++ {
                for j := 0; j < n; j++ { vNew[i] += Acopy[i][j] * v[j] }
            }
            norm := 0.0
            for i := 0; i < n; i++ { norm += vNew[i]*vNew[i] }
            norm = math.Sqrt(norm)
            if norm < 1e-15 { break }
            for i := 0; i < n; i++ { vNew[i] /= norm }
            diff := 0.0
            for i := 0; i < n; i++ { d := vNew[i]-v[i]; diff += d*d }
            if math.Sqrt(diff) < tol { v = vNew; break }
            v = vNew
        }
        // λ = v^T A v
        Av := make([]float64, n)
        for i := 0; i < n; i++ { for j := 0; j < n; j++ { Av[i] += Acopy[i][j] * v[j] } }
        lambda := 0.0
        for i := 0; i < n; i++ { lambda += v[i] * Av[i] }
        eigenvalues[k] = lambda
        copy(eigenvectors[k], v)
        // Deflate
        for i := 0; i < n; i++ {
            for j := 0; j < n; j++ { Acopy[i][j] -= lambda * v[i] * v[j] }
        }
    }
    // Convert to column-major eigenvectors
    cols := make([][]float64, n)
    for i := 0; i < n; i++ {
        cols[i] = make([]float64, n)
        for j := 0; j < n; j++ { cols[i][j] = eigenvectors[j][i] }
    }
    return eigenvalues, cols, nil
}

