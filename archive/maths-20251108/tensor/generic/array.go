package generic

import (
	"fmt"
	"math"
	"sort"
)

// Number represents numeric types that can be used in arrays
type Number interface {
	~float32 | ~float64 | ~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// Array represents a generic n-dimensional array
type Array[T Number] struct {
	data    []T
	shape   []int
	strides []int
}

// NewArray creates a new generic array
func NewArray[T Number](data []T, shape ...int) (*Array[T], error) {
	total := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("negative dimension: %d", d)
		}
		total *= d
	}
	if len(data) != total {
		return nil, fmt.Errorf("data size %d doesn't match shape %v (expected %d)", len(data), shape, total)
	}
	return &Array[T]{
		data:    data,
		shape:   shape,
		strides: computeStrides(shape),
	}, nil
}

// computeStrides calculates the strides for a given shape
func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// Factory functions for common array types

// Zeros creates an array filled with zeros
func Zeros[T Number](shape ...int) (*Array[T], error) {
	total := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("negative dimension: %d", d)
		}
		total *= d
	}
	data := make([]T, total)
	return &Array[T]{
		data:    data,
		shape:   shape,
		strides: computeStrides(shape),
	}, nil
}

// Ones creates an array filled with ones
func Ones[T Number](shape ...int) (*Array[T], error) {
	total := 1
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("negative dimension: %d", d)
		}
		total *= d
	}
	data := make([]T, total)
	var one T = 1
	for i := range data {
		data[i] = one
	}
	return &Array[T]{
		data:    data,
		shape:   shape,
		strides: computeStrides(shape),
	}, nil
}

// Arange creates an array with values from start to stop with step
func Arange[T Number](start, stop, step T) (*Array[T], error) {
	if step == 0 {
		return nil, fmt.Errorf("step cannot be zero")
	}

	var n int
	if step > 0 {
		n = int(math.Ceil(float64(stop-start) / float64(step)))
	} else {
		n = int(math.Ceil(float64(stop-start) / float64(step)))
	}

	if n <= 0 {
		return &Array[T]{data: []T{}, shape: []int{0}, strides: []int{1}}, nil
	}

	data := make([]T, n)
	for i := 0; i < n; i++ {
		data[i] = start + T(i)*step
	}

	return &Array[T]{
		data:    data,
		shape:   []int{n},
		strides: []int{1},
	}, nil
}

// Linspace creates an array with n evenly spaced values from start to stop
func Linspace[T Number](start, stop T, num int) (*Array[T], error) {
	if num < 0 {
		return nil, fmt.Errorf("num must be non-negative, got %d", num)
	}

	data := make([]T, num)
	if num == 0 {
		return &Array[T]{data: []T{}, shape: []int{0}, strides: []int{1}}, nil
	}
	if num == 1 {
		data[0] = start
	} else {
		step := (stop - start) / T(num-1)
		for i := 0; i < num; i++ {
			data[i] = start + T(i)*step
		}
	}

	return &Array[T]{
		data:    data,
		shape:   []int{num},
		strides: []int{1},
	}, nil
}

// Eye creates an identity matrix
func Eye[T Number](n int) (*Array[T], error) {
	if n < 0 {
		return nil, fmt.Errorf("negative dimension: %d", n)
	}

	data := make([]T, n*n)
	var zero, one T = 0, 1
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				data[i*n+j] = one
			} else {
				data[i*n+j] = zero
			}
		}
	}

	return &Array[T]{
		data:    data,
		shape:   []int{n, n},
		strides: computeStrides([]int{n, n}),
	}, nil
}

// Array methods

// Shape returns the shape of the array
func (a *Array[T]) Shape() []int {
	return a.shape
}

// Size returns the total number of elements
func (a *Array[T]) Size() int {
	return len(a.data)
}

// Data returns the underlying data slice
func (a *Array[T]) Data() []T {
	return a.data
}

// Copy creates a copy of the array
func (a *Array[T]) Copy() *Array[T] {
	data := make([]T, len(a.data))
	copy(data, a.data)
	shape := make([]int, len(a.shape))
	copy(shape, a.shape)
	strides := make([]int, len(a.strides))
	copy(strides, a.strides)

	return &Array[T]{
		data:    data,
		shape:   shape,
		strides: strides,
	}
}

// Reshape changes the shape of the array
func (a *Array[T]) Reshape(shape ...int) error {
	total := 1
	for _, d := range shape {
		if d < 0 {
			return fmt.Errorf("negative dimension: %d", d)
		}
		total *= d
	}
	if total != len(a.data) {
		return fmt.Errorf("cannot reshape array of size %d into shape %v", len(a.data), shape)
	}

	a.shape = shape
	a.strides = computeStrides(shape)
	return nil
}

// Transpose transposes a 2D array
func (a *Array[T]) Transpose() (*Array[T], error) {
	if len(a.shape) != 2 {
		return nil, fmt.Errorf("transpose only supported for 2D arrays, got %dD", len(a.shape))
	}

	m, n := a.shape[0], a.shape[1]
	data := make([]T, len(a.data))

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			data[j*m+i] = a.data[i*n+j]
		}
	}

	return &Array[T]{
		data:    data,
		shape:   []int{n, m},
		strides: computeStrides([]int{n, m}),
	}, nil
}

// Dot performs matrix multiplication for 2D arrays
func (a *Array[T]) Dot(b *Array[T]) (*Array[T], error) {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("dot only supported for 2D arrays")
	}

	m, k1 := a.shape[0], a.shape[1]
	k2, n := b.shape[0], b.shape[1]
	if k1 != k2 {
		return nil, fmt.Errorf("dimension mismatch: %d != %d", k1, k2)
	}

	data := make([]T, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum T = 0
			for k := 0; k < k1; k++ {
				sum += a.data[i*k1+k] * b.data[k*n+j]
			}
			data[i*n+j] = sum
		}
	}

	return &Array[T]{
		data:    data,
		shape:   []int{m, n},
		strides: computeStrides([]int{m, n}),
	}, nil
}

// Statistical operations

// Sum calculates the sum of all elements
func (a *Array[T]) Sum() T {
	var sum T = 0
	for _, v := range a.data {
		sum += v
	}
	return sum
}

// Mean calculates the mean of all elements
func (a *Array[T]) Mean() T {
	if len(a.data) == 0 {
		return 0
	}
	return a.Sum() / T(len(a.data))
}

// Min finds the minimum value
func (a *Array[T]) Min() T {
	if len(a.data) == 0 {
		return 0
	}
	min := a.data[0]
	for _, v := range a.data[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// Max finds the maximum value
func (a *Array[T]) Max() T {
	if len(a.data) == 0 {
		return 0
	}
	max := a.data[0]
	for _, v := range a.data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// ArgMin finds the index of the minimum value
func (a *Array[T]) ArgMin() int {
	if len(a.data) == 0 {
		return -1
	}
	idx := 0
	min := a.data[0]
	for i, v := range a.data[1:] {
		if v < min {
			min = v
			idx = i + 1
		}
	}
	return idx
}

// ArgMax finds the index of the maximum value
func (a *Array[T]) ArgMax() int {
	if len(a.data) == 0 {
		return -1
	}
	idx := 0
	max := a.data[0]
	for i, v := range a.data[1:] {
		if v > max {
			max = v
			idx = i + 1
		}
	}
	return idx
}

// Element-wise operations

// Add performs element-wise addition
func (a *Array[T]) Add(b *Array[T]) (*Array[T], error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, b.shape)
	}

	data := make([]T, len(a.data))
	for i := range a.data {
		data[i] = a.data[i] + b.data[i]
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}, nil
}

// Sub performs element-wise subtraction
func (a *Array[T]) Sub(b *Array[T]) (*Array[T], error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, b.shape)
	}

	data := make([]T, len(a.data))
	for i := range a.data {
		data[i] = a.data[i] - b.data[i]
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}, nil
}

// Mul performs element-wise multiplication
func (a *Array[T]) Mul(b *Array[T]) (*Array[T], error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, b.shape)
	}

	data := make([]T, len(a.data))
	for i := range a.data {
		data[i] = a.data[i] * b.data[i]
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}, nil
}

// Div performs element-wise division
func (a *Array[T]) Div(b *Array[T]) (*Array[T], error) {
	if !shapesEqual(a.shape, b.shape) {
		return nil, fmt.Errorf("shape mismatch: %v vs %v", a.shape, b.shape)
	}

	data := make([]T, len(a.data))
	for i := range a.data {
		if b.data[i] == 0 {
			return nil, fmt.Errorf("division by zero at index %d", i)
		}
		data[i] = a.data[i] / b.data[i]
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}, nil
}

// Scalar operations

// AddScalar adds a scalar to all elements
func (a *Array[T]) AddScalar(scalar T) *Array[T] {
	data := make([]T, len(a.data))
	for i, v := range a.data {
		data[i] = v + scalar
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}
}

// MulScalar multiplies all elements by a scalar
func (a *Array[T]) MulScalar(scalar T) *Array[T] {
	data := make([]T, len(a.data))
	for i, v := range a.data {
		data[i] = v * scalar
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}
}

// Utility functions

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

// Sort sorts the array in-place
func (a *Array[T]) Sort() {
	sort.Slice(a.data, func(i, j int) bool {
		return a.data[i] < a.data[j]
	})
}

// Clip clips values to the range [min, max]
func (a *Array[T]) Clip(min, max T) *Array[T] {
	data := make([]T, len(a.data))
	for i, v := range a.data {
		if v < min {
			data[i] = min
		} else if v > max {
			data[i] = max
		} else {
			data[i] = v
		}
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}
}

// Abs returns the absolute value of all elements
func (a *Array[T]) Abs() *Array[T] {
	data := make([]T, len(a.data))
	for i, v := range a.data {
		if v < 0 {
			data[i] = -v
		} else {
			data[i] = v
		}
	}

	return &Array[T]{
		data:    data,
		shape:   a.shape,
		strides: a.strides,
	}
}

// String returns a string representation of the array
func (a *Array[T]) String() string {
	return fmt.Sprintf("Array%T{shape: %v, data: %v}", *new(T), a.shape, a.data)
}
