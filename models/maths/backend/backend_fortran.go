//go:build cgo
// +build cgo

package backend

/*
#include <dlfcn.h>
#include <stdlib.h>

typedef void (*lnn_dot_fn)(int, const double*, const double*, double*);
typedef void (*lnn_cosine_fn)(int, const double*, const double*, double*);
typedef void (*lnn_matmul_fn)(int, int, int, const double*, const double*, double*);
typedef void (*lnn_dot_batch_fn)(int, int, const double*, const double*, double*);
typedef void (*lnn_cosine_batch_fn)(int, int, const double*, const double*, double*);
typedef void (*lnn_cosine_topk_fn)(int, int, const double*, const double*, int, int*, double*);
typedef void (*lnn_cosine_topk_i8_fn)(int, int, const signed char*, const double*, int, int*, double*);
typedef void (*lnn_cosine_multi_topk_fn)(int, int, int, const double*, const double*, int, int*, double*);
typedef void (*lnn_cosine_multi_topk_i8_fn)(int, int, int, const signed char*, const double*, int, int*, double*);
typedef void (*lnn_project_fn)(int, int, int, const double*, const double*, double*);

static void* lnn_handle = NULL;
static lnn_dot_fn    p_dot    = NULL;
static lnn_cosine_fn p_cosine = NULL;
static lnn_matmul_fn p_matmul = NULL;
static lnn_dot_batch_fn    p_dot_batch    = NULL;
static lnn_cosine_batch_fn p_cosine_batch = NULL;
static lnn_cosine_topk_fn  p_cosine_topk  = NULL;
static lnn_cosine_topk_i8_fn  p_cosine_topk_i8  = NULL;
static lnn_project_fn      p_project      = NULL;
static lnn_cosine_multi_topk_fn p_cosine_multi_topk = NULL;
static lnn_cosine_multi_topk_i8_fn p_cosine_multi_topk_i8 = NULL;

static int lnn_load_backend(const char* path) {
    if (lnn_handle) return 0;
    lnn_handle = dlopen(path, RTLD_NOW);
    if (!lnn_handle) return 1;
    p_dot    = (lnn_dot_fn)          dlsym(lnn_handle, "lnn_dot");
    p_cosine = (lnn_cosine_fn)       dlsym(lnn_handle, "lnn_cosine");
    p_matmul = (lnn_matmul_fn)       dlsym(lnn_handle, "lnn_matmul");
    p_dot_batch    = (lnn_dot_batch_fn)    dlsym(lnn_handle, "lnn_dot_batch");
    p_cosine_batch = (lnn_cosine_batch_fn) dlsym(lnn_handle, "lnn_cosine_batch");
    p_cosine_topk  = (lnn_cosine_topk_fn)  dlsym(lnn_handle, "lnn_cosine_topk");
    p_cosine_topk_i8 = (lnn_cosine_topk_i8_fn) dlsym(lnn_handle, "lnn_cosine_topk_i8");
    p_project           = (lnn_project_fn)           dlsym(lnn_handle, "lnn_project");
    p_cosine_multi_topk = (lnn_cosine_multi_topk_fn) dlsym(lnn_handle, "lnn_cosine_multi_topk");
    p_cosine_multi_topk_i8 = (lnn_cosine_multi_topk_i8_fn) dlsym(lnn_handle, "lnn_cosine_multi_topk_i8");
    if (!p_dot || !p_cosine || !p_matmul || !p_dot_batch || !p_cosine_batch || !p_cosine_topk || !p_project || !p_cosine_multi_topk || !p_cosine_topk_i8 || !p_cosine_multi_topk_i8) return 2;
    return 0;
}

static int lnn_is_loaded() { return (p_dot && p_cosine && p_matmul) ? 1 : 0; }

static void call_dot(int n, const double* a, const double* b, double* out) { (*p_dot)(n,a,b,out); }
static void call_cos(int n, const double* a, const double* b, double* out) { (*p_cosine)(n,a,b,out); }
static void call_mm(int m, int n, int k, const double* A, const double* B, double* C) { (*p_matmul)(m,n,k,A,B,C); }
static void call_dot_batch(int n, int m, const double* A, const double* B, double* out) { (*p_dot_batch)(n,m,A,B,out); }
static void call_cos_batch(int n, int m, const double* A, const double* B, double* out) { (*p_cosine_batch)(n,m,A,B,out); }
static void call_topk(int n, int m, const double* A, const double* q, int k, int* idx, double* scores) { (*p_cosine_topk)(n,m,A,q,k,idx,scores); }
static void call_topk_i8(int n, int m, const signed char* A8, const double* q, int k, int* idx, double* scores) { (*p_cosine_topk_i8)(n,m,A8,q,k,idx,scores); }
static void call_project(int m, int n, int r, const double* A, const double* P, double* Y) { (*p_project)(m,n,r,A,P,Y); }
static void call_multi_topk(int n, int m, int q, const double* A, const double* Q, int k, int* idx, double* scores) { (*p_cosine_multi_topk)(n,m,q,A,Q,k,idx,scores); }
static void call_multi_topk_i8(int n, int m, int q, const signed char* A8, const double* Q, int k, int* idx, double* scores) { (*p_cosine_multi_topk_i8)(n,m,q,A8,Q,k,idx,scores); }
*/
import "C"

import (
	"os"
	"path/filepath"
	"runtime"
	"unsafe"
)

// fortranProvider uses the LNN Fortran shared library.
type fortranProvider struct{}

func tryLoadFortran() bool {
	// 1) explicit env path
	if p := os.Getenv("LNN_FORTRAN_LIB"); p != "" {
		c := C.CString(p)
		defer C.free(unsafe.Pointer(c))
		if C.lnn_load_backend(c) == 0 && C.lnn_is_loaded() != 0 {
			return true
		}
	}
	// 2) default to infra fortran lib path
	// resolve current source dir
	_, file, _, ok := runtime.Caller(0)
	if ok {
		base := filepath.Dir(file) // .../infrastructure/maths
		libdir := filepath.Join(base, "fortran", "lib")
		// platform name
		cand := filepath.Join(libdir, "liblnn.so")
		if runtime.GOOS == "darwin" {
			cand = filepath.Join(libdir, "liblnn.dylib")
		}
		if _, err := os.Stat(cand); err == nil {
			c := C.CString(cand)
			defer C.free(unsafe.Pointer(c))
			if C.lnn_load_backend(c) == 0 && C.lnn_is_loaded() != 0 {
				return true
			}
		}
	}
	return false
}

func init() {
	if tryLoadFortran() {
		Register(fortranProvider{})
	}
}

// Provider methods — prefer Fortran; fallback to Go provider for safety

func (fortranProvider) Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("infrastructure/maths/backend.Dot: length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	var out C.double
	C.call_dot(C.int(len(a)), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])), &out)
	return float64(out)
}

func (fp fortranProvider) Cos(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("infrastructure/maths/backend.Cos: length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	var out C.double
	C.call_cos(C.int(len(a)), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])), &out)
	return float64(out)
}

func (fp fortranProvider) MatMul(m, n, k int, A, B []float64) []float64 {
	if len(A) != m*k || len(B) != k*n {
		panic("infrastructure/maths/backend.MatMul: buffer size mismatch")
	}
	Cbuf := make([]float64, m*n)
	if m > 0 && n > 0 && k > 0 {
		C.call_mm(C.int(m), C.int(n), C.int(k), (*C.double)(unsafe.Pointer(&A[0])), (*C.double)(unsafe.Pointer(&B[0])), (*C.double)(unsafe.Pointer(&Cbuf[0])))
	}
	return Cbuf
}

func (fp fortranProvider) Project(m, n, r int, A, P []float64) []float64 {
	if m <= 0 || n <= 0 || r <= 0 || len(A) != m*n || len(P) != n*r {
		return nil
	}
	Y := make([]float64, m*r)
	C.call_project(C.int(m), C.int(n), C.int(r), (*C.double)(unsafe.Pointer(&A[0])), (*C.double)(unsafe.Pointer(&P[0])), (*C.double)(unsafe.Pointer(&Y[0])))
	return Y
}

func (fp fortranProvider) CosineTopK(n int, A []float64, q []float64, topK int) ([]int, []float64) {
	if n <= 0 || len(A)%n != 0 || len(q) != n || topK <= 0 {
		return nil, nil
	}
	m := len(A) / n
	idx := make([]C.int, topK)
	scores := make([]C.double, topK)
	if m > 0 {
		C.call_topk(C.int(n), C.int(m), (*C.double)(unsafe.Pointer(&A[0])), (*C.double)(unsafe.Pointer(&q[0])), C.int(topK), (*C.int)(unsafe.Pointer(&idx[0])), (*C.double)(unsafe.Pointer(&scores[0])))
	}
	outIdx := make([]int, topK)
	outScores := make([]float64, topK)
	for i := 0; i < topK; i++ {
		outIdx[i] = int(idx[i]) - 1
		outScores[i] = float64(scores[i])
	}
	return outIdx, outScores
}

func (fp fortranProvider) CosineMultiTopK(n int, A []float64, Q []float64, topK int) ([][]int, [][]float64) {
	if n <= 0 || len(A)%n != 0 || len(Q)%n != 0 || topK <= 0 {
		return nil, nil
	}
	m := len(A) / n
	mq := len(Q) / n
	idx := make([]C.int, mq*topK)
	scores := make([]C.double, mq*topK)
	C.call_multi_topk(C.int(n), C.int(m), C.int(mq), (*C.double)(unsafe.Pointer(&A[0])), (*C.double)(unsafe.Pointer(&Q[0])), C.int(topK), (*C.int)(unsafe.Pointer(&idx[0])), (*C.double)(unsafe.Pointer(&scores[0])))
	outIdx := make([][]int, mq)
	outScores := make([][]float64, mq)
	for q := 0; q < mq; q++ {
		off := q * topK
		rowIdx := make([]int, topK)
		rowScores := make([]float64, topK)
		for k := 0; k < topK; k++ {
			rowIdx[k] = int(idx[off+k]) - 1
			rowScores[k] = float64(scores[off+k])
		}
		outIdx[q], outScores[q] = rowIdx, rowScores
	}
	return outIdx, outScores
}

func (fp fortranProvider) CosineTopKInt8(n int, A8 []int8, q []float64, topK int) ([]int, []float64) {
	if n <= 0 || len(A8)%n != 0 || len(q) != n || topK <= 0 {
		return nil, nil
	}
	m := len(A8) / n
	idx := make([]C.int, topK)
	scores := make([]C.double, topK)
	if m > 0 {
		C.call_topk_i8(C.int(n), C.int(m), (*C.schar)(unsafe.Pointer(&A8[0])), (*C.double)(unsafe.Pointer(&q[0])), C.int(topK), (*C.int)(unsafe.Pointer(&idx[0])), (*C.double)(unsafe.Pointer(&scores[0])))
	}
	outIdx := make([]int, topK)
	outScores := make([]float64, topK)
	for i := 0; i < topK; i++ {
		outIdx[i] = int(idx[i]) - 1
		outScores[i] = float64(scores[i])
	}
	return outIdx, outScores
}
