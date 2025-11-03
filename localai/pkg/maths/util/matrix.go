package util

// Matrix64 is a simple dense row-major matrix of float64 values.
type Matrix64 struct {
    Data   []float64
    Rows   int
    Cols   int
    Stride int
}

// NewMatrix64 allocates a Matrix64 with given dimensions.
func NewMatrix64(rows, cols int) *Matrix64 {
    if rows < 0 {
        rows = 0
    }
    if cols < 0 {
        cols = 0
    }
    return &Matrix64{
        Data:   make([]float64, rows*cols),
        Rows:   rows,
        Cols:   cols,
        Stride: cols,
    }
}

// Set sets the element at (r,c) to v if within bounds.
func (m *Matrix64) Set(r, c int, v float64) {
    if m == nil {
        return
    }
    if r < 0 || r >= m.Rows || c < 0 || c >= m.Cols {
        return
    }
    m.Data[r*m.Stride+c] = v
}

// MatMul computes a naive matrix multiply of A(rows x k) and B(k x cols).
// Returns a newly allocated slice of length rows*cols.
func MatMul(rows, cols, k int, a, b []float64) []float64 {
    out := make([]float64, rows*cols)
    for i := 0; i < rows; i++ {
        ai := i * k
        oi := i * cols
        for j := 0; j < cols; j++ {
            sum := 0.0
            bj := j
            for t := 0; t < k; t++ {
                sum += a[ai+t] * b[bj]
                bj += cols
            }
            out[oi+j] = sum
        }
    }
    return out
}

