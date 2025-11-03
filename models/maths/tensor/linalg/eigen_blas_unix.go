//go:build !darwin

package linalg

import (
	"fmt"
	matpkg "gonum.org/v1/gonum/mat"
)

// EigenDecompositionBLAS uses Gonum for symmetric eigendecomposition (where available).
func EigenDecompositionBLAS(A [][]float64) ([]float64, [][]float64, error) {
	n := len(A)
	if n == 0 || len(A[0]) != n {
		return nil, nil, fmt.Errorf("matrix must be square")
	}
	sym := matpkg.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		if len(A[i]) != n {
			return nil, nil, fmt.Errorf("row %d length %d != %d", i, len(A[i]), n)
		}
		for j := 0; j <= i; j++ {
			v := 0.5 * (A[i][j] + A[j][i])
			sym.SetSym(i, j, v)
		}
	}
	var es matpkg.EigenSym
	if !es.Factorize(sym, true) {
		return nil, nil, fmt.Errorf("eigensolver failed")
	}
	vals := es.Values(make([]float64, n))
	var V matpkg.Dense
	es.VectorsTo(&V)
	vecs := make([][]float64, n)
	for i := 0; i < n; i++ {
		vecs[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			vecs[i][j] = V.At(i, j)
		}
	}
	return vals, vecs, nil
}
