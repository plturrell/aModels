//go:build cgo && lapack && (linux || freebsd)

package linalg

/*
#cgo LDFLAGS: -llapacke -llapack -lblas
#include <lapacke.h>
*/
import "C"
import "fmt"

// EigenDecompositionBLAS uses LAPACKE_dsyevd on unix.
func EigenDecompositionBLAS(A [][]float64) ([]float64, [][]float64, error) {
    n := len(A)
    if n == 0 || len(A[0]) != n { return nil, nil, fmt.Errorf("matrix must be square") }
    buf := make([]float64, n*n)
    for i := 0; i < n; i++ {
        if len(A[i]) != n { return nil, nil, fmt.Errorf("row %d length %d != %d", i, len(A[i]), n) }
        for j := 0; j < n; j++ { v := 0.5 * (A[i][j] + A[j][i]); buf[i*n+j] = v }
    }
    w := make([]float64, n)
    info := C.LAPACKE_dsyevd(C.int(101), 'V', 'U', C.int(n), (*C.double)(&buf[0]), C.int(n), (*C.double)(&w[0]))
    if info != 0 { return nil, nil, fmt.Errorf("LAPACKE_dsyevd failed: info=%d", int(info)) }
    vecs := make([][]float64, n)
    for i := 0; i < n; i++ { vecs[i] = make([]float64, n) }
    for j := 0; j < n; j++ { for i := 0; i < n; i++ { vecs[i][j] = buf[i*n+j] } }
    return w, vecs, nil
}

