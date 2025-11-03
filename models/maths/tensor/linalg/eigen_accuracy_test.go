package linalg

import (
    "math"
    "sort"
    "testing"
)

// pair holds an eigenvalue and its corresponding eigenvector column.
type pair struct { val float64; vec []float64 }

func sortPairs(vals []float64, vecs [][]float64) []pair {
    n := len(vals)
    ps := make([]pair, n)
    for i := 0; i < n; i++ {
        col := make([]float64, n)
        for r := 0; r < n; r++ { col[r] = vecs[r][i] }
        ps[i] = pair{val: vals[i], vec: col}
    }
    sort.Slice(ps, func(i, j int) bool { return ps[i].val > ps[j].val })
    return ps
}

func matVec(A [][]float64, x []float64) []float64 {
    y := make([]float64, len(A))
    for i := range A { s := 0.0; for j := range A[i] { s += A[i][j]*x[j] }; y[i] = s }
    return y
}

func normDiff(a, b []float64) float64 {
    s := 0.0
    for i := range a { d := a[i]-b[i]; s += d*d }
    return math.Sqrt(s)
}

func normalize(x []float64) []float64 {
    n := 0.0
    for _, v := range x { n += v*v }
    if n == 0 { return x }
    inv := 1.0/math.Sqrt(n)
    y := make([]float64, len(x))
    for i := range x { y[i] = x[i]*inv }
    return y
}

func frobenius(A [][]float64) float64 {
    s := 0.0
    for i := range A { for j := range A[i] { s += A[i][j]*A[i][j] } }
    return math.Sqrt(s)
}

func reconstruct(vecs [][]float64, vals []float64) [][]float64 {
    n := len(vals)
    R := make([][]float64, n)
    for i := range R { R[i] = make([]float64, n) }
    M := make([][]float64, n)
    for i := range M { M[i] = make([]float64, n) }
    for j := 0; j < n; j++ { for i := 0; i < n; i++ { M[i][j] = vecs[i][j] * vals[j] } }
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            s := 0.0
            for k := 0; k < n; k++ { s += M[i][k]*vecs[j][k] }
            R[i][j] = s
        }
    }
    return R
}

func diffMat(A, B [][]float64) [][]float64 {
    n := len(A)
    D := make([][]float64, n)
    for i := range D { D[i] = make([]float64, n); for j := 0; j < n; j++ { D[i][j] = A[i][j]-B[i][j] } }
    return D
}

func TestEigen_Accuracy_Gonum(t *testing.T) {
    sizes := []int{16, 32}
    for _, n := range sizes {
        A := genSym(n, 7)
        valsG, vecsG, err := EigenDecompositionBLAS(A)
        if err != nil { t.Fatalf("gonum eig failed: %v", err) }
        RG := reconstruct(vecsG, valsG)
        errG := frobenius(A)
        diffG := frobenius(diffMat(A, RG))
        if diffG/errG > 1e-9 {
            t.Fatalf("n=%d: gonum reconstruction error too high: rel=%e", n, diffG/errG)
        }
    }
}

