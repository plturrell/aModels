package linalg

import (
    "math"
    "sort"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
    gemm "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/gemm"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// QRDecomposition performs QR decomposition using Householder reflections.
func QRDecomposition(A [][]float64) (Q, R [][]float64, err error) {
    m, n := len(A), len(A[0])
    R = make([][]float64, m)
    for i := 0; i < m; i++ { R[i] = make([]float64, n); copy(R[i], A[i]) }
    Q = identityMatrix(m)

    for k := 0; k < ints.Min(m-1, n); k++ {
        x := make([]float64, m-k)
        for i := k; i < m; i++ { x[i-k] = R[i][k] }

        normX := 0.0
        for _, v := range x { normX += v*v }
        normX = math.Sqrt(normX)
        if normX < 1e-15 { continue }

        alpha := -sign(x[0]) * normX
        v := make([]float64, len(x))
        v[0] = x[0] - alpha
        for i := 1; i < len(x); i++ { v[i] = x[i] }
        normV := 0.0
        for _, vj := range v { normV += vj*vj }
        normV = math.Sqrt(normV)
        if normV < 1e-15 { continue }
        for i := range v { v[i] /= normV }

        for j := k; j < n; j++ {
            dot := 0.0
            for i := 0; i < len(v); i++ { dot += v[i] * R[k+i][j] }
            for i := 0; i < len(v); i++ { R[k+i][j] -= 2 * dot * v[i] }
        }
        for j := 0; j < m; j++ {
            dot := 0.0
            for i := 0; i < len(v); i++ { dot += v[i] * Q[k+i][j] }
            for i := 0; i < len(v); i++ { Q[k+i][j] -= 2 * dot * v[i] }
        }
    }
    Q = transpose(Q)
    return Q, R, nil
}

// SVD performs a simplified Singular Value Decomposition.
func SVD(A [][]float64, fullMatrices bool) (U, S [][]float64, Vt [][]float64, err error) {
    m, n := len(A), len(A[0])
    if m <= 32 && n <= 32 { return svdSmall(A, fullMatrices) }

    At := transpose(A)
    AtA := matmul(At, A, n, n, m)
    eigenvalues, V, err := EigenDecomposition(AtA)
    if err != nil { return nil, nil, nil, err }

    k := ints.Min(m, n)
    singular := make([]float64, k)
    for i := 0; i < k; i++ { if eigenvalues[i] > 0 { singular[i] = math.Sqrt(eigenvalues[i]) } }

    // Sort singular values descending, reorder V columns accordingly (first k)
    order := make([]int, k)
    for i := 0; i < k; i++ { order[i] = i }
    sort.Slice(order, func(i, j int) bool { return singular[order[i]] > singular[order[j]] })
    singularSorted := make([]float64, k)
    Vsorted := make([][]float64, n)
    for i := 0; i < n; i++ { Vsorted[i] = make([]float64, k) }
    for idx := 0; idx < k; idx++ {
        j := order[idx]
        singularSorted[idx] = singular[j]
        for r := 0; r < n; r++ { Vsorted[r][idx] = V[r][j] }
    }
    singular = singularSorted
    V = Vsorted

    U = make([][]float64, m)
    for i := 0; i < m; i++ {
        U[i] = make([]float64, k)
        for j := 0; j < k; j++ {
            if singular[j] > 1e-15 {
                for p := 0; p < n; p++ { U[i][j] += A[i][p] * V[p][j] / singular[j] }
            }
        }
    }
    S = make([][]float64, k)
    for i := 0; i < k; i++ { S[i] = make([]float64, k); S[i][i] = singular[i] }
    Vt = transpose(V)
    return U, S, Vt, nil
}

// --- local helpers ---

func transpose(A [][]float64) [][]float64 { m,n := len(A), len(A[0]); R:=make([][]float64,n); for i:=0;i<n;i++{ R[i]=make([]float64,m); for j:=0;j<m;j++{ R[i][j]=A[j][i] } }; return R }
func matmul(A,B [][]float64, m,n,k int) [][]float64 { a:=util.From2D(A); b:=util.From2D(B); c:=gemm.MatMulContiguous(a,b); return util.To2D(c) }
func identityMatrix(n int) [][]float64 { I:=make([][]float64,n); for i:=0;i<n;i++{ I[i]=make([]float64,n); I[i][i]=1 }; return I }
func sign(x float64) float64 { if x>=0 { return 1 } ; return -1 }
// minInt moved to internal/ints

func svdSmall(A [][]float64, _ bool) (U, S, Vt [][]float64, err error) {
    m, n := len(A), len(A[0])
    At := transpose(A)
    AtA := matmul(At, A, n, n, m)
    eigenvalues, V, err := EigenDecomposition(AtA)
    if err != nil { return nil, nil, nil, err }
    k := ints.Min(m, n)
    singular := make([]float64, k)
    for i := 0; i < k; i++ { if eigenvalues[i] > 0 { singular[i] = math.Sqrt(eigenvalues[i]) } }

    // Sort descending as above
    order := make([]int, k)
    for i := 0; i < k; i++ { order[i] = i }
    sort.Slice(order, func(i, j int) bool { return singular[order[i]] > singular[order[j]] })
    singularSorted := make([]float64, k)
    Vsorted := make([][]float64, n)
    for i := 0; i < n; i++ { Vsorted[i] = make([]float64, k) }
    for idx := 0; idx < k; idx++ {
        j := order[idx]
        singularSorted[idx] = singular[j]
        for r := 0; r < n; r++ { Vsorted[r][idx] = V[r][j] }
    }
    singular = singularSorted
    V = Vsorted
    U = make([][]float64, m)
    for i := 0; i < m; i++ {
        U[i] = make([]float64, k)
        for j := 0; j < k; j++ {
            if singular[j] > 1e-15 {
                for p := 0; p < n; p++ { U[i][j] += A[i][p] * V[p][j] / singular[j] }
            }
        }
    }
    S = make([][]float64, k); for i:=0;i<k;i++{ S[i]=make([]float64,k); S[i][i]=singular[i] }
    Vt = transpose(V)
    return U, S, Vt, nil
}
