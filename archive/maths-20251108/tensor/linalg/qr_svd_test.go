package linalg

import (
    "math"
    "testing"
)

func nearlyEqual(a, b float64, tol float64) bool {
    if a == b { return true }
    diff := math.Abs(a-b)
    denom := math.Max(1.0, math.Max(math.Abs(a), math.Abs(b)))
    return diff/denom < tol
}

func matMul(A, B [][]float64) [][]float64 {
    m, k := len(A), len(A[0])
    n := len(B[0])
    out := make([][]float64, m)
    for i := 0; i < m; i++ {
        out[i] = make([]float64, n)
        for j := 0; j < n; j++ {
            s := 0.0
            for p := 0; p < k; p++ { s += A[i][p] * B[p][j] }
            out[i][j] = s
        }
    }
    return out
}

func matDiffNorm(A, B [][]float64) float64 {
    m, n := len(A), len(A[0])
    s := 0.0
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            d := A[i][j] - B[i][j]
            s += d * d
        }
    }
    return math.Sqrt(s)
}

func identity(n int) [][]float64 {
    I := make([][]float64, n)
    for i := 0; i < n; i++ { I[i] = make([]float64, n); I[i][i] = 1 }
    return I
}

func transpose2(A [][]float64) [][]float64 {
    m, n := len(A), len(A[0])
    R := make([][]float64, n)
    for i := 0; i < n; i++ { R[i] = make([]float64, m); for j := 0; j < m; j++ { R[i][j] = A[j][i] } }
    return R
}

func TestQRDecomposition_ReconstructsA(t *testing.T) {
    A := [][]float64{
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41},
    }
    Q, R, err := QRDecomposition(A)
    if err != nil { t.Fatalf("qr error: %v", err) }
    QR := matMul(Q, R)
    if n := matDiffNorm(QR, A); n > 1e-6 {
        t.Fatalf("||QR-A|| too large: %g", n)
    }
    // Check Q^T Q ≈ I
    QtQ := matMul(transpose2(Q), Q)
    I := identity(len(Q))
    if n := matDiffNorm(QtQ, I); n > 1e-6 {
        t.Fatalf("||Q^TQ - I|| too large: %g", n)
    }
}

func TestSVD_ReconstructsA(t *testing.T) {
    A := [][]float64{
        {1, 2, 3},
        {4, 5, 6},
    }
    U, S, Vt, err := SVD(A, false)
    if err != nil { t.Fatalf("svd error: %v", err) }
    US := matMul(U, S)
    USVt := matMul(US, Vt)
    if n := matDiffNorm(USVt, A); n > 1e-6 {
        t.Fatalf("||USVt - A|| too large: %g", n)
    }
}

func TestSVD_SingularAndSorted(t *testing.T) {
    // Rank-1 matrix (rows identical)
    A := [][]float64{
        {2, 4, 6},
        {2, 4, 6},
    }
    U, S, Vt, err := SVD(A, false)
    if err != nil { t.Fatalf("svd error: %v", err) }
    // Check shapes: U (2 x 2), S (2 x 2), Vt (2 x 3)
    if len(U) != 2 || len(U[0]) != 2 { t.Fatalf("U shape %dx%d unexpected", len(U), len(U[0])) }
    if len(S) != 2 || len(S[0]) != 2 { t.Fatalf("S shape %dx%d unexpected", len(S), len(S[0])) }
    if len(Vt) != 2 || len(Vt[0]) != 3 { t.Fatalf("Vt shape %dx%d unexpected", len(Vt), len(Vt[0])) }
    // Singular values non-increasing and one near zero
    s0, s1 := S[0][0], S[1][1]
    if s0 < s1-1e-9 { t.Fatalf("singular values not sorted: s0=%g s1=%g", s0, s1) }
    if s1 > 1e-6 { t.Fatalf("second singular not near zero: %g", s1) }
}

