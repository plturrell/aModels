package array

import (
	"fmt"
	"math"
)

// BroadcastResult represents the result of broadcasting two arrays
type BroadcastResult struct {
	Shape   []int
	Strides []int
	Size    int
}

// BroadcastInfo contains information about how to broadcast arrays
type BroadcastInfo struct {
	Shape    []int
	StridesA []int
	StridesB []int
	Size     int
}

// CanBroadcast checks if two arrays can be broadcast together
func CanBroadcast(a, b *Array) bool {
	// Start from the rightmost dimension
	i := len(a.shape) - 1
	j := len(b.shape) - 1

	for i >= 0 && j >= 0 {
		// Dimensions are compatible if they're equal or one is 1
		if a.shape[i] != b.shape[j] && a.shape[i] != 1 && b.shape[j] != 1 {
			return false
		}
		i--
		j--
	}

	return true
}

// BroadcastShapes returns the broadcasted shape of two arrays
func BroadcastShapes(a, b *Array) ([]int, error) {
	if !CanBroadcast(a, b) {
		return nil, fmt.Errorf("arrays cannot be broadcast together: %v and %v", a.shape, b.shape)
	}

	// Determine the maximum number of dimensions
	maxDims := len(a.shape)
	if len(b.shape) > maxDims {
		maxDims = len(b.shape)
	}

	result := make([]int, maxDims)

	// Fill from the right
	for i := 0; i < maxDims; i++ {
		dimA := 1
		dimB := 1

		if i < len(a.shape) {
			dimA = a.shape[len(a.shape)-1-i]
		}
		if i < len(b.shape) {
			dimB = b.shape[len(b.shape)-1-i]
		}

		// The broadcasted dimension is the maximum of the two
		result[maxDims-1-i] = int(math.Max(float64(dimA), float64(dimB)))
	}

	return result, nil
}

// ComputeBroadcastInfo computes how to broadcast two arrays
func ComputeBroadcastInfo(a, b *Array) (*BroadcastInfo, error) {
	shape, err := BroadcastShapes(a, b)
	if err != nil {
		return nil, err
	}

	// Compute strides for the broadcasted arrays
	stridesA := computeBroadcastStrides(a.shape, shape)
	stridesB := computeBroadcastStrides(b.shape, shape)

	// Compute total size
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	return &BroadcastInfo{
		Shape:    shape,
		StridesA: stridesA,
		StridesB: stridesB,
		Size:     size,
	}, nil
}

// computeBroadcastStrides computes strides for broadcasting
func computeBroadcastStrides(originalShape, broadcastShape []int) []int {
	strides := make([]int, len(broadcastShape))

	// Start from the rightmost dimension
	origIdx := len(originalShape) - 1
	broadIdx := len(broadcastShape) - 1

	stride := 1
	for broadIdx >= 0 {
		if origIdx >= 0 && originalShape[origIdx] == broadcastShape[broadIdx] {
			// Dimensions match, use the original stride
			strides[broadIdx] = stride
			stride *= originalShape[origIdx]
			origIdx--
		} else {
			// Dimension was broadcasted (size 1), stride is 0
			strides[broadIdx] = 0
		}
		broadIdx--
	}

	return strides
}

// BroadcastTo broadcasts an array to a target shape
func (a *Array) BroadcastTo(targetShape []int) (*Array, error) {
	if !canBroadcastTo(a.shape, targetShape) {
		return nil, fmt.Errorf("cannot broadcast shape %v to %v", a.shape, targetShape)
	}

	// If shapes are already the same, return a copy
	if shapesEqual(a.shape, targetShape) {
		return a.Copy(), nil
	}

	// Compute strides for the target shape
	strides := computeBroadcastStrides(a.shape, targetShape)

	// Create new data array
	size := 1
	for _, dim := range targetShape {
		size *= dim
	}

	newData := make([]float64, size)

	// Fill the broadcasted array
	for i := 0; i < size; i++ {
		// Convert flat index to multi-dimensional index
		indices := flatToMultiIndex(i, targetShape)

		// Convert to original array index
		origIdx := 0
		for j, idx := range indices {
			if j < len(a.shape) {
				origIdx += idx * strides[j]
			}
		}

		newData[i] = a.data[origIdx]
	}

	return &Array{
		data:    newData,
		shape:   targetShape,
		strides: computeStrides(targetShape),
	}, nil
}

// canBroadcastTo checks if an array can be broadcast to a target shape
func canBroadcastTo(shape, target []int) bool {
	// Start from the rightmost dimension
	i := len(shape) - 1
	j := len(target) - 1

	for i >= 0 && j >= 0 {
		if shape[i] != target[j] && shape[i] != 1 {
			return false
		}
		i--
		j--
	}

	// Remaining dimensions in target must be 1 or match
	for j >= 0 {
		if target[j] != 1 {
			return false
		}
		j--
	}

	return true
}

// shapesEqual checks if two shapes are equal
// flatToMultiIndex converts a flat index to multi-dimensional indices
func flatToMultiIndex(flatIdx int, shape []int) []int {
	indices := make([]int, len(shape))

	for i := len(shape) - 1; i >= 0; i-- {
		indices[i] = flatIdx % shape[i]
		flatIdx /= shape[i]
	}

	return indices
}

// Broadcasted operations

// AddBroadcast adds two arrays with broadcasting
func (a *Array) AddBroadcast(b *Array) (*Array, error) {
	info, err := ComputeBroadcastInfo(a, b)
	if err != nil {
		return nil, err
	}

	result := make([]float64, info.Size)

	for i := 0; i < info.Size; i++ {
		// Get indices for both arrays
		idxA := getBroadcastIndex(i, info.StridesA)
		idxB := getBroadcastIndex(i, info.StridesB)

		result[i] = a.data[idxA] + b.data[idxB]
	}

	return &Array{
		data:    result,
		shape:   info.Shape,
		strides: computeStrides(info.Shape),
	}, nil
}

// SubBroadcast subtracts two arrays with broadcasting
func (a *Array) SubBroadcast(b *Array) (*Array, error) {
	info, err := ComputeBroadcastInfo(a, b)
	if err != nil {
		return nil, err
	}

	result := make([]float64, info.Size)

	for i := 0; i < info.Size; i++ {
		idxA := getBroadcastIndex(i, info.StridesA)
		idxB := getBroadcastIndex(i, info.StridesB)

		result[i] = a.data[idxA] - b.data[idxB]
	}

	return &Array{
		data:    result,
		shape:   info.Shape,
		strides: computeStrides(info.Shape),
	}, nil
}

// MulBroadcast multiplies two arrays with broadcasting
func (a *Array) MulBroadcast(b *Array) (*Array, error) {
	info, err := ComputeBroadcastInfo(a, b)
	if err != nil {
		return nil, err
	}

	result := make([]float64, info.Size)

	for i := 0; i < info.Size; i++ {
		idxA := getBroadcastIndex(i, info.StridesA)
		idxB := getBroadcastIndex(i, info.StridesB)

		result[i] = a.data[idxA] * b.data[idxB]
	}

	return &Array{
		data:    result,
		shape:   info.Shape,
		strides: computeStrides(info.Shape),
	}, nil
}

// DivBroadcast divides two arrays with broadcasting
func (a *Array) DivBroadcast(b *Array) (*Array, error) {
	info, err := ComputeBroadcastInfo(a, b)
	if err != nil {
		return nil, err
	}

	result := make([]float64, info.Size)

	for i := 0; i < info.Size; i++ {
		idxA := getBroadcastIndex(i, info.StridesA)
		idxB := getBroadcastIndex(i, info.StridesB)

		if b.data[idxB] == 0 {
			return nil, fmt.Errorf("division by zero at index %d", i)
		}

		result[i] = a.data[idxA] / b.data[idxB]
	}

	return &Array{
		data:    result,
		shape:   info.Shape,
		strides: computeStrides(info.Shape),
	}, nil
}

// getBroadcastIndex gets the index in the original array for a broadcasted position
func getBroadcastIndex(flatIdx int, strides []int) int {
	index := 0
	for i, stride := range strides {
		if stride > 0 {
			// Convert flat index to multi-dimensional index
			dimIdx := (flatIdx / product(strides[i+1:])) % len(strides)
			index += dimIdx * stride
		}
	}
	return index
}

// product calculates the product of a slice of integers
func product(nums []int) int {
	result := 1
	for _, num := range nums {
		result *= num
	}
	return result
}

// Broadcasting utilities

// ExpandDims adds new dimensions of size 1
func (a *Array) ExpandDims(axis int) (*Array, error) {
	if axis < 0 || axis > len(a.shape) {
		return nil, fmt.Errorf("axis %d out of bounds [0, %d]", axis, len(a.shape))
	}

	newShape := make([]int, len(a.shape)+1)
	copy(newShape[:axis], a.shape[:axis])
	newShape[axis] = 1
	copy(newShape[axis+1:], a.shape[axis:])

	return &Array{
		data:    a.data,
		shape:   newShape,
		strides: computeStrides(newShape),
	}, nil
}

// Squeeze removes dimensions of size 1
func (a *Array) Squeeze(axis ...int) (*Array, error) {
	newShape := make([]int, 0, len(a.shape))

	for i, dim := range a.shape {
		shouldKeep := true

		if len(axis) > 0 {
			// Only remove specified axes
			shouldKeep = true
			for _, ax := range axis {
				if i == ax {
					shouldKeep = false
					break
				}
			}
		} else {
			// Remove all dimensions of size 1
			shouldKeep = dim != 1
		}

		if shouldKeep {
			newShape = append(newShape, dim)
		}
	}

	if len(newShape) == 0 {
		// All dimensions were squeezed, return scalar
		return NewArray([]float64{a.data[0]}, 1), nil
	}

	return &Array{
		data:    a.data,
		shape:   newShape,
		strides: computeStrides(newShape),
	}, nil
}

// Broadcast comparison operations

// EqualBroadcast checks element-wise equality with broadcasting
func (a *Array) EqualBroadcast(b *Array) (*Array, error) {
	info, err := ComputeBroadcastInfo(a, b)
	if err != nil {
		return nil, err
	}

	result := make([]float64, info.Size)

	for i := 0; i < info.Size; i++ {
		idxA := getBroadcastIndex(i, info.StridesA)
		idxB := getBroadcastIndex(i, info.StridesB)

		if a.data[idxA] == b.data[idxB] {
			result[i] = 1
		} else {
			result[i] = 0
		}
	}

	return &Array{
		data:    result,
		shape:   info.Shape,
		strides: computeStrides(info.Shape),
	}, nil
}

// GreaterBroadcast checks element-wise greater than with broadcasting
func (a *Array) GreaterBroadcast(b *Array) (*Array, error) {
	info, err := ComputeBroadcastInfo(a, b)
	if err != nil {
		return nil, err
	}

	result := make([]float64, info.Size)

	for i := 0; i < info.Size; i++ {
		idxA := getBroadcastIndex(i, info.StridesA)
		idxB := getBroadcastIndex(i, info.StridesB)

		if a.data[idxA] > b.data[idxB] {
			result[i] = 1
		} else {
			result[i] = 0
		}
	}

	return &Array{
		data:    result,
		shape:   info.Shape,
		strides: computeStrides(info.Shape),
	}, nil
}

// LessBroadcast checks element-wise less than with broadcasting
func (a *Array) LessBroadcast(b *Array) (*Array, error) {
	info, err := ComputeBroadcastInfo(a, b)
	if err != nil {
		return nil, err
	}

	result := make([]float64, info.Size)

	for i := 0; i < info.Size; i++ {
		idxA := getBroadcastIndex(i, info.StridesA)
		idxB := getBroadcastIndex(i, info.StridesB)

		if a.data[idxA] < b.data[idxB] {
			result[i] = 1
		} else {
			result[i] = 0
		}
	}

	return &Array{
		data:    result,
		shape:   info.Shape,
		strides: computeStrides(info.Shape),
	}, nil
}
