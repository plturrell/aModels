package linalg

// Symmetrize returns 0.5*(A + A^T).
func Symmetrize(A [][]float64) [][]float64 {
    n := len(A)
    R := make([][]float64, n)
    for i := 0; i < n; i++ {
        R[i] = make([]float64, n)
        for j := 0; j < n; j++ { R[i][j] = 0.5 * (A[i][j] + A[j][i]) }
    }
    return R
}

// ReconstructMatrix builds Q*diag(lambda)*Q^T where Q has columns of eigenvectors.
func ReconstructMatrix(Q [][]float64, lambda []float64) [][]float64 {
    n := len(Q)
    R := make([][]float64, n)
    for i := 0; i < n; i++ {
        R[i] = make([]float64, n)
        for j := 0; j < n; j++ {
            s := 0.0
            for k := 0; k < n; k++ { s += Q[i][k] * lambda[k] * Q[j][k] }
            R[i][j] = s
        }
    }
    return R
}

