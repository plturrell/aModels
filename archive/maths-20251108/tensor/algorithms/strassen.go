package algorithms

import (
    "fmt"
    gemm "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/gemm"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

const strassenThreshold = 128

func matmul(A, B [][]float64, m, n, k int) [][]float64 { a:=util.From2D(A); b:=util.From2D(B); c:=gemm.MatMulContiguous(a,b); return util.To2D(c) }

func nextPowerOf2(n int) int { p:=1; for p<n { p<<=1 }; return p }

func padMatrix(A [][]float64, size int) [][]float64 { m,n:=len(A),len(A[0]); P:=make([][]float64,size); for i:=0;i<size;i++{ P[i]=make([]float64,size); if i<m { copy(P[i][:n], A[i]) } }; return P }
func unpadMatrix(A [][]float64, m,n int) [][]float64 { R:=make([][]float64,m); for i:=0;i<m;i++{ R[i]=make([]float64,n); copy(R[i], A[i][:n]) }; return R }
func splitMatrix(A [][]float64, half int)([][]float64,[][]float64,[][]float64,[][]float64){ A11:=make([][]float64,half);A12:=make([][]float64,half);A21:=make([][]float64,half);A22:=make([][]float64,half); for i:=0;i<half;i++{ A11[i]=make([]float64,half);A12[i]=make([]float64,half);A21[i]=make([]float64,half);A22[i]=make([]float64,half); copy(A11[i],A[i][:half]); copy(A12[i],A[i][half:2*half]); copy(A21[i],A[half+i][:half]); copy(A22[i],A[half+i][half:2*half]) }; return A11,A12,A21,A22 }
func mergeMatrix(C11,C12,C21,C22 [][]float64, n int) [][]float64 { half:=n/2; C:=make([][]float64,n); for i:=0;i<half;i++{ C[i]=make([]float64,n); copy(C[i][:half],C11[i]); copy(C[i][half:],C12[i]) }; for i:=0;i<half;i++{ C[half+i]=make([]float64,n); copy(C[half+i][:half],C21[i]); copy(C[half+i][half:],C22[i]) }; return C }
func addMatrices(A,B [][]float64, n int) [][]float64 { C:=make([][]float64,n); for i:=0;i<n;i++{ C[i]=make([]float64,n); for j:=0;j<n;j++{ C[i][j]=A[i][j]+B[i][j] } }; return C }
func subMatrices(A,B [][]float64, n int) [][]float64 { C:=make([][]float64,n); for i:=0;i<n;i++{ C[i]=make([]float64,n); for j:=0;j<n;j++{ C[i][j]=A[i][j]-B[i][j] } }; return C }

func strassenRecursive(A, B [][]float64, n int) [][]float64 {
    if n <= strassenThreshold { return matmul(A,B,n,n,n) }
    half := n/2
    A11,A12,A21,A22 := splitMatrix(A,half)
    B11,B12,B21,B22 := splitMatrix(B,half)
    M1 := strassenRecursive(addMatrices(A11,A22,half), addMatrices(B11,B22,half), half)
    M2 := strassenRecursive(addMatrices(A21,A22,half), B11, half)
    M3 := strassenRecursive(A11, subMatrices(B12,B22,half), half)
    M4 := strassenRecursive(A22, subMatrices(B21,B11,half), half)
    M5 := strassenRecursive(addMatrices(A11,A12,half), B22, half)
    M6 := strassenRecursive(subMatrices(A21,A11,half), addMatrices(B11,B12,half), half)
    M7 := strassenRecursive(subMatrices(A12,A22,half), addMatrices(B21,B22,half), half)
    C11 := subMatrices(addMatrices(addMatrices(M1,M4,half), M7, half), M5, half)
    C12 := addMatrices(M3,M5,half)
    C21 := addMatrices(M2,M4,half)
    C22 := subMatrices(addMatrices(addMatrices(M1,M3,half), M6, half), M2, half)
    return mergeMatrix(C11,C12,C21,C22,n)
}

// StrassenMatMul performs Strassen's algorithm using packed GEMM as leaf.
func StrassenMatMul(A, B [][]float64) ([][]float64, error) {
    m, k1 := len(A), len(A[0]); k2, n := len(B), len(B[0])
    if k1 != k2 { return nil, ErrDim() }
    if m != n || n != k1 || n < strassenThreshold { return matmul(A,B,m,n,k1), nil }
    size := nextPowerOf2(n)
    if size != n { A = padMatrix(A,size); B=padMatrix(B,size) }
    C := strassenRecursive(A,B,size)
    if size != n { C = unpadMatrix(C,m,n) }
    return C,nil
}

func ErrDim() error { return fmt.Errorf("dimension mismatch") }
