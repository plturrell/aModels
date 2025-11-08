package array

import (
	"fmt"
	"math"
	"sort"
)

type Array struct {
	data    []float64
	shape   []int
	strides []int
}

func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func NewArray(data []float64, shape ...int) *Array {
	total := 1
	for _, d := range shape {
		total *= d
	}
	if len(data) != total {
		panic(fmt.Sprintf("data size %d doesn't match shape %v", len(data), shape))
	}
	return &Array{data: data, shape: shape, strides: computeStrides(shape)}
}

func Zeros(shape ...int) *Array {
	total := 1
	for _, d := range shape {
		total *= d
	}
	return NewArray(make([]float64, total), shape...)
}
func Ones(shape ...int) *Array {
	total := 1
	for _, d := range shape {
		total *= d
	}
	data := make([]float64, total)
	for i := range data {
		data[i] = 1
	}
	return NewArray(data, shape...)
}

func Arange(start, stop, step float64) *Array {
	n := int(math.Ceil((stop - start) / step))
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = start + float64(i)*step
	}
	return NewArray(data, n)
}
func Linspace(start, stop float64, num int) *Array {
	data := make([]float64, num)
	if num > 1 {
		step := (stop - start) / float64(num-1)
		for i := 0; i < num; i++ {
			data[i] = start + float64(i)*step
		}
	} else if num == 1 {
		data[0] = start
	}
	return NewArray(data, num)
}
func Eye(n int) *Array {
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		data[i*n+i] = 1
	}
	return NewArray(data, n, n)
}

func Random(shape ...int) *Array {
	total := 1
	for _, d := range shape {
		total *= d
	}
	data := make([]float64, total)
	for i := range data {
		data[i] = math.Sin(float64(i))
	}
	return NewArray(data, shape...)
}

// Copy creates a deep copy of the array.
func (a *Array) Copy() *Array {
	data := make([]float64, len(a.data))
	copy(data, a.data)
	return NewArray(data, a.shape...)
}

// DataCopy returns a copy of the underlying data slice.
func (a *Array) DataCopy() []float64 {
	out := make([]float64, len(a.data))
	copy(out, a.data)
	return out
}

func (a *Array) Shape() []int { return a.shape }
func (a *Array) Reshape(shape ...int) *Array {
	total := 1
	for _, d := range shape {
		total *= d
	}
	if total != len(a.data) {
		panic(fmt.Sprintf("cannot reshape array of size %d into shape %v", len(a.data), shape))
	}
	return NewArray(a.data, shape...)
}
func (a *Array) T() *Array {
	if len(a.shape) != 2 {
		panic("transpose only supported for 2D arrays")
	}
	m, n := a.shape[0], a.shape[1]
	data := make([]float64, len(a.data))
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			data[j*m+i] = a.data[i*n+j]
		}
	}
	return NewArray(data, n, m)
}
func (a *Array) Dot(b *Array) *Array {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("dot only supported for 2D arrays")
	}
	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k1 != k2 {
		panic(fmt.Sprintf("dimension mismatch: %d != %d", k1, k2))
	}
	data := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			s := 0.0
			for k := 0; k < k1; k++ {
				s += a.data[i*k1+k] * b.data[k*n+j]
			}
			data[i*n+j] = s
		}
	}
	return NewArray(data, m, n)
}
func (a *Array) Sum() float64 {
	s := 0.0
	for _, v := range a.data {
		s += v
	}
	return s
}
func (a *Array) Mean() float64 { return a.Sum() / float64(len(a.data)) }
func (a *Array) Std() float64 {
	mean := a.Mean()
	var v float64
	for _, x := range a.data {
		d := x - mean
		v += d * d
	}
	return math.Sqrt(v / float64(len(a.data)))
}
func (a *Array) Max() float64 {
	if len(a.data) == 0 {
		return math.NaN()
	}
	mx := a.data[0]
	for _, v := range a.data[1:] {
		if v > mx {
			mx = v
		}
	}
	return mx
}
func (a *Array) Min() float64 {
	if len(a.data) == 0 {
		return math.NaN()
	}
	mn := a.data[0]
	for _, v := range a.data[1:] {
		if v < mn {
			mn = v
		}
	}
	return mn
}

// Method chaining operations

// Add performs element-wise addition and returns a new array
func (a *Array) Add(other *Array) (*Array, error) {
	if !shapesEqual(a.shape, other.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, other.shape)
	}

	result := make([]float64, len(a.data))
	for i := range a.data {
		result[i] = a.data[i] + other.data[i]
	}

	return NewArray(result, a.shape...), nil
}

// Sub performs element-wise subtraction and returns a new array
func (a *Array) Sub(other *Array) (*Array, error) {
	if !shapesEqual(a.shape, other.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, other.shape)
	}

	result := make([]float64, len(a.data))
	for i := range a.data {
		result[i] = a.data[i] - other.data[i]
	}

	return NewArray(result, a.shape...), nil
}

// Mul performs element-wise multiplication and returns a new array
func (a *Array) Mul(other *Array) (*Array, error) {
	if !shapesEqual(a.shape, other.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, other.shape)
	}

	result := make([]float64, len(a.data))
	for i := range a.data {
		result[i] = a.data[i] * other.data[i]
	}

	return NewArray(result, a.shape...), nil
}

// Div performs element-wise division and returns a new array
func (a *Array) Div(other *Array) (*Array, error) {
	if !shapesEqual(a.shape, other.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, other.shape)
	}

	result := make([]float64, len(a.data))
	for i := range a.data {
		if other.data[i] == 0 {
			return nil, fmt.Errorf("division by zero at index %d", i)
		}
		result[i] = a.data[i] / other.data[i]
	}

	return NewArray(result, a.shape...), nil
}

// AddScalar adds a scalar to all elements
func (a *Array) AddScalar(scalar float64) *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = v + scalar
	}
	return NewArray(result, a.shape...)
}

// MulScalar multiplies all elements by a scalar
func (a *Array) MulScalar(scalar float64) *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = v * scalar
	}
	return NewArray(result, a.shape...)
}

// Pow raises each element to the given power
func (a *Array) Pow(power float64) *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Pow(v, power)
	}
	return NewArray(result, a.shape...)
}

// Sqrt takes the square root of each element
func (a *Array) Sqrt() *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Sqrt(v)
	}
	return NewArray(result, a.shape...)
}

// Abs takes the absolute value of each element
func (a *Array) Abs() *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Abs(v)
	}
	return NewArray(result, a.shape...)
}

// Exp takes the exponential of each element
func (a *Array) Exp() *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Exp(v)
	}
	return NewArray(result, a.shape...)
}

// Log takes the natural logarithm of each element
func (a *Array) Log() *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Log(v)
	}
	return NewArray(result, a.shape...)
}

// Sin takes the sine of each element
func (a *Array) Sin() *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Sin(v)
	}
	return NewArray(result, a.shape...)
}

// Cos takes the cosine of each element
func (a *Array) Cos() *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Cos(v)
	}
	return NewArray(result, a.shape...)
}

// Tan takes the tangent of each element
func (a *Array) Tan() *Array {
	result := make([]float64, len(a.data))
	for i, v := range a.data {
		result[i] = math.Tan(v)
	}
	return NewArray(result, a.shape...)
}

// shapesEqual checks if two shapes are equal
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

// Extra NumPy-like helpers
func (a *Array) ArgMax() int {
	if len(a.data) == 0 {
		return -1
	}
	idx := 0
	mx := a.data[0]
	for i, v := range a.data[1:] {
		if v > mx {
			mx = v
			idx = i + 1
		}
	}
	return idx
}
func (a *Array) ArgMin() int {
	if len(a.data) == 0 {
		return -1
	}
	idx := 0
	mn := a.data[0]
	for i, v := range a.data[1:] {
		if v < mn {
			mn = v
			idx = i + 1
		}
	}
	return idx
}
func (a *Array) Clip(min, max float64) *Array {
	data := make([]float64, len(a.data))
	for i, v := range a.data {
		if v < min {
			data[i] = min
		} else if v > max {
			data[i] = max
		} else {
			data[i] = v
		}
	}
	return NewArray(data, a.shape...)
}
func (a *Array) Where(cond func(float64) bool) []int {
	idxs := []int{}
	for i, v := range a.data {
		if cond(v) {
			idxs = append(idxs, i)
		}
	}
	return idxs
}
func (a *Array) Sort() *Array {
	data := append([]float64(nil), a.data...)
	sort.Float64s(data)
	return NewArray(data, a.shape...)
}
func (a *Array) Unique() *Array {
	seen := map[float64]bool{}
	out := []float64{}
	for _, v := range a.data {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Float64s(out)
	return NewArray(out, len(out))
}

func Concatenate(arrays []*Array, axis int) *Array {
	if len(arrays) == 0 {
		return nil
	}
	if axis != 0 {
		panic("only axis=0 supported")
	}
	total := 0
	for _, a := range arrays {
		total += len(a.data)
	}
	data := make([]float64, 0, total)
	for _, a := range arrays {
		data = append(data, a.data...)
	}
	newShape := append([]int(nil), arrays[0].shape...)
	newShape[0] = 0
	for _, a := range arrays {
		newShape[0] += a.shape[0]
	}
	return NewArray(data, newShape...)
}

func Stack(arrays []*Array, axis int) *Array {
	if len(arrays) == 0 {
		return nil
	}
	total := len(arrays) * len(arrays[0].data)
	data := make([]float64, 0, total)
	for _, a := range arrays {
		data = append(data, a.data...)
	}
	newShape := append([]int{len(arrays)}, arrays[0].shape...)
	return NewArray(data, newShape...)
}

func (a *Array) Norm(ord float64) float64 {
	if ord == 2 {
		s := 0.0
		for _, v := range a.data {
			s += v * v
		}
		return math.Sqrt(s)
	} else if ord == 1 {
		s := 0.0
		for _, v := range a.data {
			s += math.Abs(v)
		}
		return s
	} else if math.IsInf(ord, 1) {
		mx := 0.0
		for _, v := range a.data {
			av := math.Abs(v)
			if av > mx {
				mx = av
			}
		}
		return mx
	}
	s := 0.0
	for _, v := range a.data {
		s += math.Pow(math.Abs(v), ord)
	}
	return math.Pow(s, 1.0/ord)
}

func (a *Array) Inv() (*Array, error) {
	if len(a.shape) != 2 || a.shape[0] != a.shape[1] {
		return nil, fmt.Errorf("matrix must be square")
	}
	n := a.shape[0]
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = a.data[i*n+j]
		}
		aug[i][n+i] = 1
	}
	for i := 0; i < n; i++ {
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(aug[k][i]) > math.Abs(aug[maxRow][i]) {
				maxRow = k
			}
		}
		aug[i], aug[maxRow] = aug[maxRow], aug[i]
		if math.Abs(aug[i][i]) < 1e-15 {
			return nil, fmt.Errorf("matrix is singular")
		}
		pivot := aug[i][i]
		for j := 0; j < 2*n; j++ {
			aug[i][j] /= pivot
		}
		for k := 0; k < n; k++ {
			if k != i {
				factor := aug[k][i]
				for j := 0; j < 2*n; j++ {
					aug[k][j] -= factor * aug[i][j]
				}
			}
		}
	}
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			data[i*n+j] = aug[i][n+j]
		}
	}
	return NewArray(data, n, n), nil
}

func (a *Array) Det() (float64, error) {
	if len(a.shape) != 2 || a.shape[0] != a.shape[1] {
		return 0, fmt.Errorf("matrix must be square")
	}
	n := a.shape[0]
	mat := make([][]float64, n)
	for i := 0; i < n; i++ {
		mat[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			mat[i][j] = a.data[i*n+j]
		}
	}
	det := 1.0
	for i := 0; i < n; i++ {
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(mat[k][i]) > math.Abs(mat[maxRow][i]) {
				maxRow = k
			}
		}
		if maxRow != i {
			mat[i], mat[maxRow] = mat[maxRow], mat[i]
			det *= -1
		}
		if math.Abs(mat[i][i]) < 1e-15 {
			return 0, nil
		}
		det *= mat[i][i]
		for k := i + 1; k < n; k++ {
			factor := mat[k][i] / mat[i][i]
			for j := i; j < n; j++ {
				mat[k][j] -= factor * mat[i][j]
			}
		}
	}
	return det, nil
}

func Solve(A, b *Array) (*Array, error) {
	if len(A.shape) != 2 || len(b.shape) != 1 {
		return nil, fmt.Errorf("A must be 2D and b must be 1D")
	}
	n := A.shape[0]
	if A.shape[1] != n || b.shape[0] != n {
		return nil, fmt.Errorf("dimension mismatch")
	}
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, n+1)
		for j := 0; j < n; j++ {
			aug[i][j] = A.data[i*n+j]
		}
		aug[i][n] = b.data[i]
	}
	for i := 0; i < n; i++ {
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(aug[k][i]) > math.Abs(aug[maxRow][i]) {
				maxRow = k
			}
		}
		aug[i], aug[maxRow] = aug[maxRow], aug[i]
		if math.Abs(aug[i][i]) < 1e-15 {
			return nil, fmt.Errorf("matrix is singular")
		}
		for k := i + 1; k < n; k++ {
			factor := aug[k][i] / aug[i][i]
			for j := i; j <= n; j++ {
				aug[k][j] -= factor * aug[i][j]
			}
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = aug[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= aug[i][j] * x[j]
		}
		x[i] /= aug[i][i]
	}
	return NewArray(x, n), nil
}
