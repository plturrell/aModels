package sparse

import (
    "fmt"
    "math"
    "sort"
)

// SparseCOO represents a sparse matrix in COO (Coordinate) format
type SparseCOO struct {
    Rows, Cols []int
    Values     []float64
    Shape      [2]int // [m, n]
}

// SparseCSR represents a sparse matrix in CSR (Compressed Sparse Row) format
type SparseCSR struct {
    RowPtr []int
    ColIdx []int
    Values []float64
    Shape  [2]int // [m, n]
}

func NewSparseCOO(dense [][]float64, threshold float64) *SparseCOO {
    m, n := len(dense), len(dense[0])
    sp := &SparseCOO{Rows: []int{}, Cols: []int{}, Values: []float64{}, Shape: [2]int{m,n}}
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if math.Abs(dense[i][j]) > threshold { sp.Rows = append(sp.Rows,i); sp.Cols=append(sp.Cols,j); sp.Values=append(sp.Values,dense[i][j]) }
        }
    }
    return sp
}

func COOToCSR(coo *SparseCOO) *SparseCSR {
    m := coo.Shape[0]; nnz := len(coo.Values)
    csr := &SparseCSR{RowPtr: make([]int, m+1), ColIdx: make([]int, nnz), Values: make([]float64, nnz), Shape: coo.Shape}
    type entry struct{ row,col int; val float64 }
    entries := make([]entry, nnz)
    for i := 0; i < nnz; i++ { entries[i] = entry{coo.Rows[i], coo.Cols[i], coo.Values[i]} }
    sort.Slice(entries, func(i,j int) bool{ if entries[i].row!=entries[j].row { return entries[i].row<entries[j].row }; return entries[i].col<entries[j].col })
    currentRow := 0
    for i := 0; i < nnz; i++ {
        for currentRow < entries[i].row { currentRow++; csr.RowPtr[currentRow] = i }
        csr.ColIdx[i] = entries[i].col; csr.Values[i] = entries[i].val
    }
    for currentRow < m { currentRow++; csr.RowPtr[currentRow] = nnz }
    return csr
}

func SparseMatVec(csr *SparseCSR, x []float64) ([]float64, error) {
    m, n := csr.Shape[0], csr.Shape[1]
    if len(x) != n { return nil, fmt.Errorf("dimension mismatch: matrix has %d columns, vector has %d elements", n, len(x)) }
    y := make([]float64, m)
    for i := 0; i < m; i++ { sum:=0.0; for j := csr.RowPtr[i]; j<csr.RowPtr[i+1]; j++ { sum += csr.Values[j] * x[csr.ColIdx[j]] }; y[i]=sum }
    return y, nil
}

func SparseMatMul(A, B *SparseCSR) (*SparseCSR, error) {
    if A.Shape[1] != B.Shape[0] { return nil, fmt.Errorf("dimension mismatch: %d != %d", A.Shape[1], B.Shape[0]) }
    m, n := A.Shape[0], B.Shape[1]
    type key struct{ row,col int }
    resultMap := make(map[key]float64)
    for i := 0; i < m; i++ {
        for jIdx := A.RowPtr[i]; jIdx < A.RowPtr[i+1]; jIdx++ {
            j := A.ColIdx[jIdx]; aVal := A.Values[jIdx]
            for kIdx := B.RowPtr[j]; kIdx < B.RowPtr[j+1]; kIdx++ {
                k := B.ColIdx[kIdx]; bVal := B.Values[kIdx]; resultMap[key{i,k}] += aVal*bVal
            }
        }
    }
    entries := make([]struct{row,col int; val float64}, 0, len(resultMap))
    for k,v := range resultMap { if math.Abs(v) > 1e-15 { entries = append(entries, struct{row,col int; val float64}{k.row,k.col,v}) } }
    sort.Slice(entries, func(i,j int) bool{ if entries[i].row!=entries[j].row { return entries[i].row<entries[j].row }; return entries[i].col<entries[j].col })
    result := &SparseCSR{ RowPtr: make([]int,m+1), ColIdx: make([]int,len(entries)), Values: make([]float64,len(entries)), Shape: [2]int{m,n} }
    currentRow := 0
    for i,e := range entries { for currentRow < e.row { currentRow++; result.RowPtr[currentRow] = i }; result.ColIdx[i] = e.col; result.Values[i] = e.val }
    for currentRow < m { currentRow++; result.RowPtr[currentRow] = len(entries) }
    return result, nil
}

func SparseToDense(csr *SparseCSR) [][]float64 {
    m, n := csr.Shape[0], csr.Shape[1]
    dense := make([][]float64, m)
    for i := 0; i < m; i++ { dense[i] = make([]float64, n); for j := csr.RowPtr[i]; j<csr.RowPtr[i+1]; j++ { dense[i][csr.ColIdx[j]] = csr.Values[j] } }
    return dense
}

