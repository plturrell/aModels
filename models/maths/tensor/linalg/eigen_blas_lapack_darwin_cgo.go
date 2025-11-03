//go:build cgo && lapack && darwin

package linalg

/*
#cgo LDFLAGS: -framework Accelerate

extern void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda,
                    double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
*/
import "C"
import (
    "fmt"
    "unsafe"
)

// EigenDecompositionBLAS uses LAPACK dsyevd on darwin.
func EigenDecompositionBLAS(A [][]float64) ([]float64, [][]float64, error) {
    n := len(A)
    if n == 0 || len(A[0]) != n { return nil, nil, fmt.Errorf("matrix must be square") }
    buf := make([]float64, n*n)
    for i := 0; i < n; i++ {
        if len(A[i]) != n { return nil, nil, fmt.Errorf("row %d length %d != %d", i, len(A[i]), n) }
        for j := 0; j < n; j++ { v := 0.5 * (A[i][j] + A[j][i]); buf[j*n+i] = v }
    }
    w := make([]float64, n)
    jobz := C.char('V'); uplo := C.char('U'); cn := C.int(n); clda := C.int(n); info := C.int(0)
    lwork := C.int(-1); liwork := C.int(-1); var wkopt C.double; var iwkopt C.int
    C.dsyevd_(&(*(*C.char)(unsafe.Pointer(&jobz))), &(*(*C.char)(unsafe.Pointer(&uplo))), &cn,
        (*C.double)(unsafe.Pointer(&buf[0])), &clda,
        (*C.double)(unsafe.Pointer(&w[0])),
        (*C.double)(unsafe.Pointer(&wkopt)), &lwork,
        (*C.int)(unsafe.Pointer(&iwkopt)), &liwork, &info)
    if info != 0 { return nil, nil, fmt.Errorf("dsyevd query failed: info=%d", int(info)) }
    lwork = C.int(int(wkopt)); liwork = iwkopt
    work := make([]C.double, int(lwork)); iwork := make([]C.int, int(liwork))
    C.dsyevd_(&(*(*C.char)(unsafe.Pointer(&jobz))), &(*(*C.char)(unsafe.Pointer(&uplo))), &cn,
        (*C.double)(unsafe.Pointer(&buf[0])), &clda,
        (*C.double)(unsafe.Pointer(&w[0])),
        &work[0], &lwork,
        &iwork[0], &liwork, &info)
    if info != 0 { return nil, nil, fmt.Errorf("dsyevd failed: info=%d", int(info)) }
    vecs := make([][]float64, n)
    for i := 0; i < n; i++ { vecs[i] = make([]float64, n); for j := 0; j < n; j++ { vecs[i][j] = buf[j*n+i] } }
    return w, vecs, nil
}

