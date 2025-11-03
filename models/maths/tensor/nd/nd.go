package nd

import (
    "errors"
    "fmt"
    "math"
)

// ============================================================================
// N-DIMENSIONAL OPERATIONS
// - NaN-aware reductions for 3D/ND arrays
// - Masked broadcasting for ND arrays
// - Advanced axis operations
// ============================================================================

// ============================================================================
// 3D NAN-AWARE REDUCTIONS
// ============================================================================

// NanArgMin3D returns argmin for 3D array, ignoring NaN
func NanArgMin3D(A [][][]float64) (int, int, int) {
	minI, minJ, minK := -1, -1, -1
	minVal := math.Inf(1)

	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A[i]); j++ {
			for k := 0; k < len(A[i][j]); k++ {
				if !math.IsNaN(A[i][j][k]) && A[i][j][k] < minVal {
					minVal = A[i][j][k]
					minI, minJ, minK = i, j, k
				}
			}
		}
	}

	return minI, minJ, minK
}

// NanArgMax3D returns argmax for 3D array, ignoring NaN
func NanArgMax3D(A [][][]float64) (int, int, int) {
	maxI, maxJ, maxK := -1, -1, -1
	maxVal := math.Inf(-1)

	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A[i]); j++ {
			for k := 0; k < len(A[i][j]); k++ {
				if !math.IsNaN(A[i][j][k]) && A[i][j][k] > maxVal {
					maxVal = A[i][j][k]
					maxI, maxJ, maxK = i, j, k
				}
			}
		}
	}

	return maxI, maxJ, maxK
}

// NanArgMin3DAxis returns argmin along axis for 3D array
func NanArgMin3DAxis(A [][][]float64, axis int) interface{} {
	d1, d2, d3 := len(A), len(A[0]), len(A[0][0])

	switch axis {
	case 0:
		// Min along first dimension
		result := make([][]int, d2)
		for j := 0; j < d2; j++ {
			result[j] = make([]int, d3)
			for k := 0; k < d3; k++ {
				minIdx := -1
				minVal := math.Inf(1)

				for i := 0; i < d1; i++ {
					if !math.IsNaN(A[i][j][k]) && A[i][j][k] < minVal {
						minVal = A[i][j][k]
						minIdx = i
					}
				}
				result[j][k] = minIdx
			}
		}
		return result

	case 1:
		// Min along second dimension
		result := make([][]int, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]int, d3)
			for k := 0; k < d3; k++ {
				minIdx := -1
				minVal := math.Inf(1)

				for j := 0; j < d2; j++ {
					if !math.IsNaN(A[i][j][k]) && A[i][j][k] < minVal {
						minVal = A[i][j][k]
						minIdx = j
					}
				}
				result[i][k] = minIdx
			}
		}
		return result

	case 2:
		// Min along third dimension
		result := make([][]int, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]int, d2)
			for j := 0; j < d2; j++ {
				minIdx := -1
				minVal := math.Inf(1)

				for k := 0; k < d3; k++ {
					if !math.IsNaN(A[i][j][k]) && A[i][j][k] < minVal {
						minVal = A[i][j][k]
						minIdx = k
					}
				}
				result[i][j] = minIdx
			}
		}
		return result

	default:
		panic(fmt.Sprintf("infrastructure/maths/tensor.NanArgMin3DAxis: invalid axis %d for 3D array", axis))
	}
}

// NanArgMax3DAxis returns argmax along axis for 3D array
func NanArgMax3DAxis(A [][][]float64, axis int) interface{} {
	d1, d2, d3 := len(A), len(A[0]), len(A[0][0])

	switch axis {
	case 0:
		result := make([][]int, d2)
		for j := 0; j < d2; j++ {
			result[j] = make([]int, d3)
			for k := 0; k < d3; k++ {
				maxIdx := -1
				maxVal := math.Inf(-1)

				for i := 0; i < d1; i++ {
					if !math.IsNaN(A[i][j][k]) && A[i][j][k] > maxVal {
						maxVal = A[i][j][k]
						maxIdx = i
					}
				}
				result[j][k] = maxIdx
			}
		}
		return result

	case 1:
		result := make([][]int, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]int, d3)
			for k := 0; k < d3; k++ {
				maxIdx := -1
				maxVal := math.Inf(-1)

				for j := 0; j < d2; j++ {
					if !math.IsNaN(A[i][j][k]) && A[i][j][k] > maxVal {
						maxVal = A[i][j][k]
						maxIdx = j
					}
				}
				result[i][k] = maxIdx
			}
		}
		return result

	case 2:
		result := make([][]int, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]int, d2)
			for j := 0; j < d2; j++ {
				maxIdx := -1
				maxVal := math.Inf(-1)

				for k := 0; k < d3; k++ {
					if !math.IsNaN(A[i][j][k]) && A[i][j][k] > maxVal {
						maxVal = A[i][j][k]
						maxIdx = k
					}
				}
				result[i][j] = maxIdx
			}
		}
		return result

	default:
		panic(fmt.Sprintf("infrastructure/maths/tensor.NanArgMax3DAxis: invalid axis %d for 3D array", axis))
	}
}

// ============================================================================
// MASKED 3D OPERATIONS
// ============================================================================

// MaskedNanArgMin3D returns argmin for 3D array with mask
func MaskedNanArgMin3D(A [][][]float64, mask [][][]bool) (int, int, int) {
	minI, minJ, minK := -1, -1, -1
	minVal := math.Inf(1)

	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A[i]); j++ {
			for k := 0; k < len(A[i][j]); k++ {
				if !mask[i][j][k] && !math.IsNaN(A[i][j][k]) && A[i][j][k] < minVal {
					minVal = A[i][j][k]
					minI, minJ, minK = i, j, k
				}
			}
		}
	}

	return minI, minJ, minK
}

// MaskedNanArgMax3D returns argmax for 3D array with mask
func MaskedNanArgMax3D(A [][][]float64, mask [][][]bool) (int, int, int) {
	maxI, maxJ, maxK := -1, -1, -1
	maxVal := math.Inf(-1)

	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A[i]); j++ {
			for k := 0; k < len(A[i][j]); k++ {
				if !mask[i][j][k] && !math.IsNaN(A[i][j][k]) && A[i][j][k] > maxVal {
					maxVal = A[i][j][k]
					maxI, maxJ, maxK = i, j, k
				}
			}
		}
	}

	return maxI, maxJ, maxK
}

// ============================================================================
// MASKED BROADCASTING FOR 3D
// ============================================================================

// MaskedBroadcastAdd3D broadcasts with mask
func MaskedBroadcastAdd3D(A [][][]float64, B interface{}, mask [][][]bool) ([][][]float64, error) {
	d1, d2, d3 := len(A), len(A[0]), len(A[0][0])
	result := make([][][]float64, d1)

	for i := 0; i < d1; i++ {
		result[i] = make([][]float64, d2)
		for j := 0; j < d2; j++ {
			result[i][j] = make([]float64, d3)
			copy(result[i][j], A[i][j])
		}
	}

	// Handle different broadcast types
	switch b := B.(type) {
	case float64:
		// Scalar broadcast
		for i := 0; i < d1; i++ {
			for j := 0; j < d2; j++ {
				for k := 0; k < d3; k++ {
					if !mask[i][j][k] {
						result[i][j][k] += b
					}
				}
			}
		}

	case []float64:
		// 1D broadcast (along last dimension)
		if len(b) != d3 {
            return nil, errors.New("dimension mismatch")
		}
		for i := 0; i < d1; i++ {
			for j := 0; j < d2; j++ {
				for k := 0; k < d3; k++ {
					if !mask[i][j][k] {
						result[i][j][k] += b[k]
					}
				}
			}
		}

	case [][]float64:
		// 2D broadcast
		if len(b) != d2 || len(b[0]) != d3 {
            return nil, errors.New("dimension mismatch")
		}
		for i := 0; i < d1; i++ {
			for j := 0; j < d2; j++ {
				for k := 0; k < d3; k++ {
					if !mask[i][j][k] {
						result[i][j][k] += b[j][k]
					}
				}
			}
		}

	default:
		return nil, fmt.Errorf("unsupported broadcast type")
	}

	return result, nil
}

// MaskedBroadcastMul3D broadcasts multiplication with mask
func MaskedBroadcastMul3D(A [][][]float64, B interface{}, mask [][][]bool) ([][][]float64, error) {
	d1, d2, d3 := len(A), len(A[0]), len(A[0][0])
	result := make([][][]float64, d1)

	for i := 0; i < d1; i++ {
		result[i] = make([][]float64, d2)
		for j := 0; j < d2; j++ {
			result[i][j] = make([]float64, d3)
			copy(result[i][j], A[i][j])
		}
	}

	switch b := B.(type) {
	case float64:
		for i := 0; i < d1; i++ {
			for j := 0; j < d2; j++ {
				for k := 0; k < d3; k++ {
					if !mask[i][j][k] {
						result[i][j][k] *= b
					}
				}
			}
		}

	case []float64:
		if len(b) != d3 {
            return nil, errors.New("dimension mismatch")
		}
		for i := 0; i < d1; i++ {
			for j := 0; j < d2; j++ {
				for k := 0; k < d3; k++ {
					if !mask[i][j][k] {
						result[i][j][k] *= b[k]
					}
				}
			}
		}

	case [][]float64:
		if len(b) != d2 || len(b[0]) != d3 {
            return nil, errors.New("dimension mismatch")
		}
		for i := 0; i < d1; i++ {
			for j := 0; j < d2; j++ {
				for k := 0; k < d3; k++ {
					if !mask[i][j][k] {
						result[i][j][k] *= b[j][k]
					}
				}
			}
		}

	default:
		return nil, fmt.Errorf("unsupported broadcast type")
	}

	return result, nil
}

// ============================================================================
// NAN-AWARE REDUCTIONS ALONG AXIS (3D)
// ============================================================================

// NanSum3DAxis computes sum along axis, ignoring NaN
func NanSum3DAxis(A [][][]float64, axis int) interface{} {
	d1, d2, d3 := len(A), len(A[0]), len(A[0][0])

	switch axis {
	case 0:
		result := make([][]float64, d2)
		for j := 0; j < d2; j++ {
			result[j] = make([]float64, d3)
			for k := 0; k < d3; k++ {
				sum := 0.0
				for i := 0; i < d1; i++ {
					if !math.IsNaN(A[i][j][k]) {
						sum += A[i][j][k]
					}
				}
				result[j][k] = sum
			}
		}
		return result

	case 1:
		result := make([][]float64, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]float64, d3)
			for k := 0; k < d3; k++ {
				sum := 0.0
				for j := 0; j < d2; j++ {
					if !math.IsNaN(A[i][j][k]) {
						sum += A[i][j][k]
					}
				}
				result[i][k] = sum
			}
		}
		return result

	case 2:
		result := make([][]float64, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]float64, d2)
			for j := 0; j < d2; j++ {
				sum := 0.0
				for k := 0; k < d3; k++ {
					if !math.IsNaN(A[i][j][k]) {
						sum += A[i][j][k]
					}
				}
				result[i][j] = sum
			}
		}
		return result

	default:
		panic(fmt.Sprintf("infrastructure/maths/tensor.NanSum3DAxis: invalid axis %d for 3D array", axis))
	}
}

// NanMean3DAxis computes mean along axis, ignoring NaN
func NanMean3DAxis(A [][][]float64, axis int) interface{} {
	d1, d2, d3 := len(A), len(A[0]), len(A[0][0])

	switch axis {
	case 0:
		result := make([][]float64, d2)
		for j := 0; j < d2; j++ {
			result[j] = make([]float64, d3)
			for k := 0; k < d3; k++ {
				sum := 0.0
				count := 0
				for i := 0; i < d1; i++ {
					if !math.IsNaN(A[i][j][k]) {
						sum += A[i][j][k]
						count++
					}
				}
				if count > 0 {
					result[j][k] = sum / float64(count)
				} else {
					result[j][k] = math.NaN()
				}
			}
		}
		return result

	case 1:
		result := make([][]float64, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]float64, d3)
			for k := 0; k < d3; k++ {
				sum := 0.0
				count := 0
				for j := 0; j < d2; j++ {
					if !math.IsNaN(A[i][j][k]) {
						sum += A[i][j][k]
						count++
					}
				}
				if count > 0 {
					result[i][k] = sum / float64(count)
				} else {
					result[i][k] = math.NaN()
				}
			}
		}
		return result

	case 2:
		result := make([][]float64, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]float64, d2)
			for j := 0; j < d2; j++ {
				sum := 0.0
				count := 0
				for k := 0; k < d3; k++ {
					if !math.IsNaN(A[i][j][k]) {
						sum += A[i][j][k]
						count++
					}
				}
				if count > 0 {
					result[i][j] = sum / float64(count)
				} else {
					result[i][j] = math.NaN()
				}
			}
		}
		return result

	default:
		panic(fmt.Sprintf("infrastructure/maths/tensor.NanMean3DAxis: invalid axis %d for 3D array", axis))
	}
}

// NanStd3DAxis computes standard deviation along axis, ignoring NaN
func NanStd3DAxis(A [][][]float64, axis int) interface{} {
	// First compute mean
	meanResult := NanMean3DAxis(A, axis)

	d1, d2, d3 := len(A), len(A[0]), len(A[0][0])

	switch axis {
	case 0:
		mean := meanResult.([][]float64)
		result := make([][]float64, d2)
		for j := 0; j < d2; j++ {
			result[j] = make([]float64, d3)
			for k := 0; k < d3; k++ {
				variance := 0.0
				count := 0
				for i := 0; i < d1; i++ {
					if !math.IsNaN(A[i][j][k]) {
						diff := A[i][j][k] - mean[j][k]
						variance += diff * diff
						count++
					}
				}
				if count > 0 {
					result[j][k] = math.Sqrt(variance / float64(count))
				} else {
					result[j][k] = math.NaN()
				}
			}
		}
		return result

	case 1:
		mean := meanResult.([][]float64)
		result := make([][]float64, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]float64, d3)
			for k := 0; k < d3; k++ {
				variance := 0.0
				count := 0
				for j := 0; j < d2; j++ {
					if !math.IsNaN(A[i][j][k]) {
						diff := A[i][j][k] - mean[i][k]
						variance += diff * diff
						count++
					}
				}
				if count > 0 {
					result[i][k] = math.Sqrt(variance / float64(count))
				} else {
					result[i][k] = math.NaN()
				}
			}
		}
		return result

	case 2:
		mean := meanResult.([][]float64)
		result := make([][]float64, d1)
		for i := 0; i < d1; i++ {
			result[i] = make([]float64, d2)
			for j := 0; j < d2; j++ {
				variance := 0.0
				count := 0
				for k := 0; k < d3; k++ {
					if !math.IsNaN(A[i][j][k]) {
						diff := A[i][j][k] - mean[i][j]
						variance += diff * diff
						count++
					}
				}
				if count > 0 {
					result[i][j] = math.Sqrt(variance / float64(count))
				} else {
					result[i][j] = math.NaN()
				}
			}
		}
		return result

	default:
		panic(fmt.Sprintf("invalid axis %d for 3D array", axis))
	}
}
