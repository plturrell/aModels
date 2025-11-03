package array

import (
	"fmt"
	"reflect"
)

// IndexType represents different types of indexing operations
type IndexType int

const (
	IndexBasic IndexType = iota
	IndexBoolean
	IndexFancy
	IndexSlice
)

// Indexer defines the interface for all indexing operations
type Indexer interface {
	Type() IndexType
	Apply(arr *Array) (*Array, error)
	Validate(arr *Array) error
}

// BooleanIndex represents boolean array indexing
type BooleanIndex struct {
	Mask []bool
}

func (b BooleanIndex) Type() IndexType { return IndexBoolean }

func (b BooleanIndex) Validate(arr *Array) error {
	if len(b.Mask) != len(arr.data) {
		return fmt.Errorf("boolean mask length %d doesn't match array size %d", len(b.Mask), len(arr.data))
	}
	return nil
}

func (b BooleanIndex) Apply(arr *Array) (*Array, error) {
	if err := b.Validate(arr); err != nil {
		return nil, err
	}

	// Count true values
	count := 0
	for _, v := range b.Mask {
		if v {
			count++
		}
	}

	if count == 0 {
		return NewArray([]float64{}, 0), nil
	}

	result := make([]float64, count)
	idx := 0
	for i, v := range b.Mask {
		if v {
			result[idx] = arr.data[i]
			idx++
		}
	}

	return NewArray(result, count), nil
}

// FancyIndex represents integer array indexing
type FancyIndex struct {
	Indices []int
}

func (f FancyIndex) Type() IndexType { return IndexFancy }

func (f FancyIndex) Validate(arr *Array) error {
	for _, idx := range f.Indices {
		if idx < 0 || idx >= len(arr.data) {
			return fmt.Errorf("index %d out of bounds [0, %d)", idx, len(arr.data))
		}
	}
	return nil
}

func (f FancyIndex) Apply(arr *Array) (*Array, error) {
	if err := f.Validate(arr); err != nil {
		return nil, err
	}

	result := make([]float64, len(f.Indices))
	for i, idx := range f.Indices {
		result[i] = arr.data[idx]
	}

	return NewArray(result, len(f.Indices)), nil
}

// SliceIndex represents slice indexing with start:stop:step
type SliceIndex struct {
	Start, Stop, Step int
}

func (s SliceIndex) Type() IndexType { return IndexSlice }

func (s SliceIndex) Validate(arr *Array) error {
	length := len(arr.data)

	// Handle negative indices
	start := s.Start
	if start < 0 {
		start = length + start
	}
	stop := s.Stop
	if stop < 0 {
		stop = length + stop
	}

	// Bounds checking
	if start < 0 || start >= length {
		return fmt.Errorf("start index %d out of bounds [0, %d)", start, length)
	}
	if stop < 0 || stop > length {
		return fmt.Errorf("stop index %d out of bounds [0, %d]", stop, length)
	}
	if s.Step == 0 {
		return fmt.Errorf("step cannot be zero")
	}

	return nil
}

func (s SliceIndex) Apply(arr *Array) (*Array, error) {
	if err := s.Validate(arr); err != nil {
		return nil, err
	}

	length := len(arr.data)
	start := s.Start
	stop := s.Stop
	step := s.Step

	// Handle negative indices
	if start < 0 {
		start = length + start
	}
	if stop < 0 {
		stop = length + stop
	}

	// Handle default values
	if step == 0 {
		step = 1
	}

	// Calculate result size
	var result []float64
	if step > 0 {
		for i := start; i < stop; i += step {
			result = append(result, arr.data[i])
		}
	} else {
		for i := start; i > stop; i += step {
			result = append(result, arr.data[i])
		}
	}

	return NewArray(result, len(result)), nil
}

// Boolean indexing: arr[mask]
func (a *Array) BooleanIndex(mask []bool) (*Array, error) {
	indexer := BooleanIndex{Mask: mask}
	return indexer.Apply(a)
}

// Fancy indexing: arr[indices]
func (a *Array) FancyIndex(indices []int) (*Array, error) {
	indexer := FancyIndex{Indices: indices}
	return indexer.Apply(a)
}

// Slice indexing: arr[start:stop:step]
func (a *Array) SliceIndex(start, stop, step int) (*Array, error) {
	indexer := SliceIndex{Start: start, Stop: stop, Step: step}
	return indexer.Apply(a)
}

// Advanced slicing with NumPy-style syntax
func (a *Array) Slice(slice interface{}) (*Array, error) {
	switch s := slice.(type) {
	case []int:
		// Fancy indexing
		return a.FancyIndex(s)
	case []bool:
		// Boolean indexing
		return a.BooleanIndex(s)
	case string:
		// Parse slice notation like "1:10:2"
		return a.parseSliceString(s)
	default:
		return nil, fmt.Errorf("unsupported slice type: %T", slice)
	}
}

// Parse slice string like "1:10:2" or "::2"
func (a *Array) parseSliceString(s string) (*Array, error) {
	return nil, fmt.Errorf("slice string parsing not yet implemented: %s", s)
}

// Multi-dimensional indexing

// At returns the value at the given multi-dimensional index
func (a *Array) At(indices ...int) (float64, error) {
	if len(indices) != len(a.shape) {
		return 0, fmt.Errorf("index dimensions %d don't match array dimensions %d", len(indices), len(a.shape))
	}

	// Convert multi-dimensional index to flat index
	flatIndex := 0
	for i, idx := range indices {
		if idx < 0 || idx >= a.shape[i] {
			return 0, fmt.Errorf("index %d out of bounds for dimension %d (size %d)", idx, i, a.shape[i])
		}
		flatIndex += idx * a.strides[i]
	}

	return a.data[flatIndex], nil
}

// SetAt sets the value at the given multi-dimensional index
func (a *Array) SetAt(value float64, indices ...int) error {
	if len(indices) != len(a.shape) {
		return fmt.Errorf("index dimensions %d don't match array dimensions %d", len(indices), len(a.shape))
	}

	// Convert multi-dimensional index to flat index
	flatIndex := 0
	for i, idx := range indices {
		if idx < 0 || idx >= a.shape[i] {
			return fmt.Errorf("index %d out of bounds for dimension %d (size %d)", idx, i, a.shape[i])
		}
		flatIndex += idx * a.strides[i]
	}

	a.data[flatIndex] = value
	return nil
}

// Advanced selection operations

// Take selects elements at given indices along an axis
func (a *Array) Take(indices []int, axis int) (*Array, error) {
	if axis < 0 || axis >= len(a.shape) {
		return nil, fmt.Errorf("axis %d out of bounds [0, %d)", axis, len(a.shape))
	}

	// For 1D arrays, this is just fancy indexing
	if len(a.shape) == 1 {
		return a.FancyIndex(indices)
	}

	// For multi-dimensional arrays, this is more complex
	// Implementation would depend on the specific axis
	return nil, fmt.Errorf("multi-dimensional Take not yet implemented")
}

// Put sets values at given indices
func (a *Array) Put(indices []int, values []float64) error {
	if len(indices) != len(values) {
		return fmt.Errorf("indices length %d doesn't match values length %d", len(indices), len(values))
	}

	for i, idx := range indices {
		if idx < 0 || idx >= len(a.data) {
			return fmt.Errorf("index %d out of bounds [0, %d)", idx, len(a.data))
		}
		a.data[idx] = values[i]
	}

	return nil
}

// Choose selects from multiple arrays based on indices
func Choose(indices []int, arrays ...*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("no arrays provided")
	}

	// All arrays must have the same shape
	shape := arrays[0].shape
	for i, arr := range arrays[1:] {
		if !reflect.DeepEqual(arr.shape, shape) {
			return nil, fmt.Errorf("array %d has different shape %v than array 0 shape %v", i+1, arr.shape, shape)
		}
	}

	result := make([]float64, len(indices))
	for i, idx := range indices {
		if idx < 0 || idx >= len(arrays) {
			return nil, fmt.Errorf("choice index %d out of bounds [0, %d)", idx, len(arrays))
		}
		// For now, assume 1D arrays - this would need to be more complex for multi-dimensional
		if len(arrays[idx].data) != len(indices) {
			return nil, fmt.Errorf("array %d size %d doesn't match indices length %d", idx, len(arrays[idx].data), len(indices))
		}
		result[i] = arrays[idx].data[i]
	}

	return NewArray(result, len(indices)), nil
}

// Compress selects elements based on a condition
func (a *Array) Compress(condition []bool, axis int) (*Array, error) {
	if len(condition) != a.shape[axis] {
		return nil, fmt.Errorf("condition length %d doesn't match axis %d size %d", len(condition), axis, a.shape[axis])
	}

	// Count true values
	count := 0
	for _, v := range condition {
		if v {
			count++
		}
	}

	if count == 0 {
		return NewArray([]float64{}, 0), nil
	}

	// For 1D arrays, this is just boolean indexing
	if len(a.shape) == 1 {
		return a.BooleanIndex(condition)
	}

	// For multi-dimensional arrays, this is more complex
	return nil, fmt.Errorf("multi-dimensional Compress not yet implemented")
}

// Nonzero returns indices of non-zero elements
func (a *Array) Nonzero() []int {
	var indices []int
	for i, v := range a.data {
		if v != 0 {
			indices = append(indices, i)
		}
	}
	return indices
}

// Advanced indexing with ellipsis support (placeholder for future implementation)
func (a *Array) EllipsisIndex(indices ...interface{}) (*Array, error) {
	// This would handle NumPy's ellipsis (...) syntax
	// For now, return an error indicating it's not implemented
	return nil, fmt.Errorf("ellipsis indexing not yet implemented")
}

// Indexing utilities

// ValidateIndex checks if an index is valid for the given shape
func ValidateIndex(idx int, shape []int, axis int) error {
	if axis < 0 || axis >= len(shape) {
		return fmt.Errorf("axis %d out of bounds [0, %d)", axis, len(shape))
	}
	if idx < 0 || idx >= shape[axis] {
		return fmt.Errorf("index %d out of bounds for axis %d (size %d)", idx, axis, shape[axis])
	}
	return nil
}

// NormalizeIndex converts negative indices to positive ones
func NormalizeIndex(idx int, size int) int {
	if idx < 0 {
		return size + idx
	}
	return idx
}

// CreateIndexer creates an indexer from various input types
func CreateIndexer(index interface{}) (Indexer, error) {
	switch idx := index.(type) {
	case []bool:
		return BooleanIndex{Mask: idx}, nil
	case []int:
		return FancyIndex{Indices: idx}, nil
	case int:
		// Single integer index
		return FancyIndex{Indices: []int{idx}}, nil
	default:
		return nil, fmt.Errorf("unsupported index type: %T", index)
	}
}
