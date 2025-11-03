package util

// Matrix64 is a contiguous row-major matrix of float64 values.
// Stride is elements per row (>= Cols).
type Matrix64 struct {
    Data   []float64
    Rows   int
    Cols   int
    Stride int
}

// NewMatrix64 allocates a zero-initialized contiguous matrix (stride == cols).
func NewMatrix64(rows, cols int) *Matrix64 {
    if rows < 0 || cols < 0 {
        panic("NewMatrix64: negative dimension")
    }
    m := &Matrix64{Rows: rows, Cols: cols, Stride: cols}
    if rows*cols > 0 {
        m.Data = make([]float64, rows*cols)
    }
    return m
}

// From2D copies a [][]float64 into a contiguous Matrix64.
func From2D(a [][]float64) *Matrix64 {
    if len(a) == 0 {
        return NewMatrix64(0, 0)
    }
    rows := len(a)
    cols := len(a[0])
    out := NewMatrix64(rows, cols)
    for i := 0; i < rows; i++ {
        if len(a[i]) != cols {
            panic("From2D: ragged rows")
        }
        copy(out.Data[i*out.Stride:i*out.Stride+cols], a[i])
    }
    return out
}

// To2D copies a Matrix64 into a freshly-allocated [][]float64.
func To2D(m *Matrix64) [][]float64 {
    if m == nil || m.Rows == 0 || m.Cols == 0 {
        return make([][]float64, 0)
    }
    out := make([][]float64, m.Rows)
    for i := 0; i < m.Rows; i++ {
        row := make([]float64, m.Cols)
        copy(row, m.Data[i*m.Stride:i*m.Stride+m.Cols])
        out[i] = row
    }
    return out
}

// At returns the element at (r,c).
func (m *Matrix64) At(r, c int) float64 {
    return m.Data[r*m.Stride+c]
}

// Set sets the element at (r,c).
func (m *Matrix64) Set(r, c int, v float64) {
    m.Data[r*m.Stride+c] = v
}
