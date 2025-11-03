package linalg

import (
    "fmt"
    "math"
    gemm "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/gemm"
    ops "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/ops"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// SPDExp computes exponential map on SPD manifold using eigendecomposition
// Exp_A(V) = A^(1/2) * exp(A^(-1/2) * V * A^(-1/2)) * A^(1/2)
func SPDExp(A, V [][]float64) ([][]float64, error) {
    n := len(A)
    if n == 0 || len(A[0]) != n { return nil, fmt.Errorf("matrix must be square") }
    symA := Symmetrize(A)
    symV := Symmetrize(V)
    eigenvalues, eigenvectors, err := EigenDecomposition(symA)
    if err != nil { return nil, fmt.Errorf("eigendecomposition failed: %v", err) }
    sqrtLambda := make([]float64, n)
    invSqrtLambda := make([]float64, n)
    for i := 0; i < n; i++ {
        if eigenvalues[i] <= 0 { return nil, fmt.Errorf("matrix not positive definite") }
        sqrtLambda[i] = math.Sqrt(eigenvalues[i])
        invSqrtLambda[i] = 1.0 / sqrtLambda[i]
    }
    invSqrtA := ReconstructMatrix(eigenvectors, invSqrtLambda)
    tmp := matmulLocal(invSqrtA, symV, n, n, n)
    W := matmulLocal(tmp, invSqrtA, n, n, n)
    wEigenvalues, wEigenvectors, err := EigenDecomposition(W)
    if err != nil { return nil, fmt.Errorf("eigendecomposition of W failed: %v", err) }
    expLambdaW := make([]float64, n)
    for i := 0; i < n; i++ { expLambdaW[i] = math.Exp(wEigenvalues[i]) }
    expW := ReconstructMatrix(wEigenvectors, expLambdaW)
    sqrtA := ReconstructMatrix(eigenvectors, sqrtLambda)
    tmp2 := matmulLocal(sqrtA, expW, n, n, n)
    result := matmulLocal(tmp2, sqrtA, n, n, n)
    return result, nil
}

// SPDDistance computes Riemannian geodesic distance on SPD manifold
// d(A,B) = ||log(A^(-1/2) * B * A^(-1/2))||_F
func SPDDistance(A, B [][]float64) (float64, error) {
    n := len(A)
    if n == 0 || len(A[0]) != n || len(B) != n || len(B[0]) != n { return 0, fmt.Errorf("matrices must be square and same size") }
    symA := Symmetrize(A)
    symB := Symmetrize(B)
    eigenvalues, eigenvectors, err := EigenDecomposition(symA)
    if err != nil { return 0, fmt.Errorf("eigendecomposition failed: %v", err) }
    invSqrtLambda := make([]float64, n)
    for i := 0; i < n; i++ {
        if eigenvalues[i] <= 0 { return 0, fmt.Errorf("matrix A not positive definite") }
        invSqrtLambda[i] = 1.0 / math.Sqrt(eigenvalues[i])
    }
    invSqrtA := ReconstructMatrix(eigenvectors, invSqrtLambda)
    tmp := matmulLocal(invSqrtA, symB, n, n, n)
    M := matmulLocal(tmp, invSqrtA, n, n, n)
    mEigenvalues, _, err := EigenDecomposition(M)
    if err != nil { return 0, fmt.Errorf("eigendecomposition of M failed: %v", err) }
    dist := 0.0
    for i := 0; i < n; i++ {
        if mEigenvalues[i] <= 0 { return 0, fmt.Errorf("invalid eigenvalue in distance computation") }
        logLambda := math.Log(mEigenvalues[i])
        dist += logLambda * logLambda
    }
    return math.Sqrt(dist), nil
}

// MobiusAdd performs Möbius addition in Poincaré ball
func MobiusAdd(x, y []float64) ([]float64, error) {
    if len(x) != len(y) { return nil, fmt.Errorf("dimension mismatch") }
    n := len(x)
    normXSq := ops.Dot(x, x)
    normYSq := ops.Dot(y, y)
    xyDot := ops.Dot(x, y)
    denom := 1.0 + 2.0*xyDot + normXSq*normYSq
    result := make([]float64, n)
    for i := 0; i < n; i++ { result[i] = ((1.0+2.0*xyDot+normYSq)*x[i] + (1.0-normXSq)*y[i]) / denom }
    return result, nil
}

// PoincareExp computes exponential map in Poincaré ball
func PoincareExp(x, v []float64) ([]float64, error) {
    if len(x) != len(v) { return nil, fmt.Errorf("dimension mismatch") }
    n := len(x)
    normXSq := ops.Dot(x, x)
    normV := math.Sqrt(ops.Dot(v, v))
    if normV < 1e-12 { return x, nil }
    lambdaX := 2.0 / (1.0 - normXSq)
    factor := math.Tanh(lambdaX*normV/2.0) / normV
    scaled := make([]float64, n)
    for i := 0; i < n; i++ { scaled[i] = v[i] * factor }
    return MobiusAdd(x, scaled)
}

// local helper using packed GEMM path
func matmulLocal(A, B [][]float64, m, n, k int) [][]float64 {
    if m == 0 || n == 0 || k == 0 { return make([][]float64, m) }
    a := util.From2D(A)
    b := util.From2D(B)
    c := gemm.MatMulContiguous(a, b)
    return util.To2D(c)
}

