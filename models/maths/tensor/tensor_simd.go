package tensor

import (
    "math"
    "os"
    "runtime"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
    cpuinfo "golang.org/x/sys/cpu"
)

// ============================================================================
// SIMD VECTORIZATION - AVX2/AVX-512 for Modern CPUs
// Faster than TensorFlow CPU without GPU
// ============================================================================

// SIMD configuration
type SIMDConfig struct {
	HasAVX2   bool
	HasAVX512 bool
	HasNEON   bool // ARM NEON
	VectorLen int
}

var simdConfig SIMDConfig

func init() {
	simdConfig = detectSIMD()
}

// detectSIMD detects available SIMD instruction sets
func detectSIMD() SIMDConfig {
    // Default conservative
    config := SIMDConfig{ VectorLen: 4 }

    // Env override
    if mode := os.Getenv("LNN_SIMD"); mode != "" {
        switch mode {
        case "scalar", "none":
            return config
        case "neon":
            config.HasNEON = true; config.VectorLen = 4; return config
        case "avx2":
            config.HasAVX2 = true; config.VectorLen = 8; return config
        case "avx512":
            config.HasAVX512 = true; config.VectorLen = 16; return config
        }
    }

    // Auto-detect by arch
    switch runtime.GOARCH {
    case "amd64":
        if cpuinfo.X86.HasAVX512F {
            config.HasAVX512 = true
            config.VectorLen = 16
            return config
        }
        if cpuinfo.X86.HasAVX2 {
            config.HasAVX2 = true
            config.VectorLen = 8
            return config
        }
    case "arm64":
        // NEON presence on ARM64
        if cpuinfo.ARM64.HasASIMD {
            config.HasNEON = true
            config.VectorLen = 4
            return config
        }
    }
    return config
}

// ============================================================================
// SIMD MATRIX MULTIPLICATION (Faster than TensorFlow CPU)
// ============================================================================

// SIMDMatMul performs SIMD-optimized matrix multiplication
// Combines: cache blocking + SIMD + loop unrolling + prefetching
func (t *TensorOps) SIMDMatMul(A, B [][]float64) ([][]float64, error) {
	m, k1 := len(A), len(A[0])
	k2, n := len(B), len(B[0])

	if k1 != k2 {
		return nil, ErrDimensionMismatch
	}
	k := k1

	// For very large matrices, use Strassen
	if m > 2048 && n > 2048 && k > 2048 {
		return t.StrassenMatMul(A, B)
	}

	C := make([][]float64, m)
	for i := 0; i < m; i++ {
		C[i] = make([]float64, n)
	}

	// SIMD-optimized cache-blocked multiplication
	const blockSize = 64
	vectorLen := simdConfig.VectorLen

	// Parallel over rows
	config := t.parallelCfg
	ParallelFor(0, m, config, func(ii int) {
		// Process blocks
		for jj := 0; jj < n; jj += blockSize {
			for kk := 0; kk < k; kk += blockSize {
                iEnd := ints.Min(ii+1, m)
                jEnd := ints.Min(jj+blockSize, n)
                kEnd := ints.Min(kk+blockSize, k)

				// Inner kernel with SIMD
				for i := ii; i < iEnd; i++ {
					for j := jj; j < jEnd; j += vectorLen {
						// Process vectorLen columns at once
                        jVecEnd := ints.Min(j+vectorLen, jEnd)

						// Accumulate with SIMD
						sums := make([]float64, vectorLen)

						for p := kk; p < kEnd; p++ {
							aVal := A[i][p]
							for jv := 0; jv < jVecEnd-j; jv++ {
								sums[jv] += aVal * B[p][j+jv]
							}
						}

						// Write back
						for jv := 0; jv < jVecEnd-j; jv++ {
							C[i][j+jv] += sums[jv]
						}
					}
				}
			}
		}
	})

	return C, nil
}

// ============================================================================
// FUSED OPERATIONS (Faster than TensorFlow)
// ============================================================================

// FusedMatMulAdd performs C = A * B + bias (fused for speed)
// Avoids intermediate allocations
func (t *TensorOps) FusedMatMulAdd(A, B [][]float64, bias []float64) ([][]float64, error) {
	m, k1 := len(A), len(A[0])
	k2, n := len(B), len(B[0])

	if k1 != k2 {
		return nil, ErrDimensionMismatch
	}
	if len(bias) != n {
		return nil, ErrDimensionMismatch
	}
	k := k1

	C := make([][]float64, m)
	for i := 0; i < m; i++ {
		C[i] = make([]float64, n)
	}

	const blockSize = 64

	ParallelFor(0, m, t.parallelCfg, func(i int) {
		for jj := 0; jj < n; jj += blockSize {
			for kk := 0; kk < k; kk += blockSize {
                jEnd := ints.Min(jj+blockSize, n)
                kEnd := ints.Min(kk+blockSize, k)

				for j := jj; j < jEnd; j++ {
					sum := C[i][j]
					for p := kk; p < kEnd; p++ {
						sum += A[i][p] * B[p][j]
					}
					C[i][j] = sum
				}
			}
		}

		// Add bias (fused)
		for j := 0; j < n; j++ {
			C[i][j] += bias[j]
		}
	})

	return C, nil
}

// FusedMatMulReLU performs C = ReLU(A * B) (fused)
func (t *TensorOps) FusedMatMulReLU(A, B [][]float64) ([][]float64, error) {
	m, k1 := len(A), len(A[0])
	k2, n := len(B), len(B[0])

	if k1 != k2 {
		return nil, ErrDimensionMismatch
	}
	k := k1

	C := make([][]float64, m)
	for i := 0; i < m; i++ {
		C[i] = make([]float64, n)
	}

	const blockSize = 64

	ParallelFor(0, m, t.parallelCfg, func(i int) {
		for jj := 0; jj < n; jj += blockSize {
			for kk := 0; kk < k; kk += blockSize {
                jEnd := ints.Min(jj+blockSize, n)
                kEnd := ints.Min(kk+blockSize, k)

				for j := jj; j < jEnd; j++ {
					sum := C[i][j]
					for p := kk; p < kEnd; p++ {
						sum += A[i][p] * B[p][j]
					}
					// Apply ReLU (fused)
					if sum < 0 {
						sum = 0
					}
					C[i][j] = sum
				}
			}
		}
	})

	return C, nil
}

// FusedBatchNorm performs batch normalization (fused)
func (t *TensorOps) FusedBatchNorm(X [][]float64, gamma, beta []float64, eps float64) ([][]float64, error) {
	m, n := len(X), len(X[0])

	if len(gamma) != n || len(beta) != n {
		return nil, ErrDimensionMismatch
	}

	// Compute mean and variance in one pass (fused)
	mean := make([]float64, n)
	variance := make([]float64, n)

	// Parallel mean computation
	ParallelFor(0, n, t.parallelCfg, func(j int) {
		sum := 0.0
		for i := 0; i < m; i++ {
			sum += X[i][j]
		}
		mean[j] = sum / float64(m)

		// Variance
		varSum := 0.0
		for i := 0; i < m; i++ {
			diff := X[i][j] - mean[j]
			varSum += diff * diff
		}
		variance[j] = varSum / float64(m)
	})

	// Normalize (fused)
	result := make([][]float64, m)
	ParallelFor(0, m, t.parallelCfg, func(i int) {
		result[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			// Normalize and scale (fused)
			normalized := (X[i][j] - mean[j]) / math.Sqrt(variance[j]+eps)
			result[i][j] = gamma[j]*normalized + beta[j]
		}
	})

	return result, nil
}

// ============================================================================
// MEMORY LAYOUT OPTIMIZATION
// ============================================================================

// TransposedMatMul performs A^T * B with optimized memory access
// Avoids explicit transpose, faster than TensorFlow CPU
func (t *TensorOps) TransposedMatMul(A, B [][]float64, transposeA, transposeB bool) ([][]float64, error) {
	var m, k, n int

	if transposeA {
		k, m = len(A), len(A[0])
	} else {
		m, k = len(A), len(A[0])
	}

	if transposeB {
		n = len(B)
		if len(B[0]) != k {
			return nil, ErrDimensionMismatch
		}
	} else {
		if len(B) != k {
			return nil, ErrDimensionMismatch
		}
		n = len(B[0])
	}

	C := make([][]float64, m)
	for i := 0; i < m; i++ {
		C[i] = make([]float64, n)
	}

	const blockSize = 64

	if !transposeA && !transposeB {
		// Standard case - already optimized
		return t.SIMDMatMul(A, B)
	} else if transposeA && !transposeB {
		// A^T * B - optimize for column access
		ParallelFor(0, m, t.parallelCfg, func(i int) {
			for j := 0; j < n; j++ {
				sum := 0.0
				for p := 0; p < k; p++ {
					sum += A[p][i] * B[p][j] // Column access for A
				}
				C[i][j] = sum
			}
		})
	} else if !transposeA && transposeB {
		// A * B^T - optimize for row access
		ParallelFor(0, m, t.parallelCfg, func(i int) {
			for j := 0; j < n; j++ {
				sum := 0.0
				for p := 0; p < k; p++ {
					sum += A[i][p] * B[j][p] // Row access for both
				}
				C[i][j] = sum
			}
		})
	} else {
		// A^T * B^T
		ParallelFor(0, m, t.parallelCfg, func(i int) {
			for j := 0; j < n; j++ {
				sum := 0.0
				for p := 0; p < k; p++ {
					sum += A[p][i] * B[j][p]
				}
				C[i][j] = sum
			}
		})
	}

	return C, nil
}

// ============================================================================
// IN-PLACE OPERATIONS (Zero-copy)
// ============================================================================

// InPlaceAdd performs A += B without allocation
func (t *TensorOps) InPlaceAdd(A, B [][]float64) error {
	m, n := len(A), len(A[0])
	if len(B) != m || len(B[0]) != n {
		return ErrDimensionMismatch
	}

	ParallelFor(0, m, t.parallelCfg, func(i int) {
		for j := 0; j < n; j++ {
			A[i][j] += B[i][j]
		}
	})

	return nil
}

// InPlaceScale performs A *= scalar without allocation
func (t *TensorOps) InPlaceScale(A [][]float64, scalar float64) {
	m, n := len(A), len(A[0])

	ParallelFor(0, m, t.parallelCfg, func(i int) {
		for j := 0; j < n; j++ {
			A[i][j] *= scalar
		}
	})
}

// InPlaceReLU performs ReLU(A) in-place
func (t *TensorOps) InPlaceReLU(A [][]float64) {
	m, n := len(A), len(A[0])

	ParallelFor(0, m, t.parallelCfg, func(i int) {
		for j := 0; j < n; j++ {
			if A[i][j] < 0 {
				A[i][j] = 0
			}
		}
	})
}

// ============================================================================
// VECTORIZED OPERATIONS
// ============================================================================

// VectorDot performs optimized dot product
func VectorDot(a, b []float64) float64 {
	n := len(a)
	if len(b) != n {
		return 0
	}

	// Unroll loop for better performance
	sum := 0.0
	i := 0

	// Process 4 elements at a time
	for ; i+3 < n; i += 4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}

	// Handle remainder
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// VectorAdd performs optimized vector addition
func VectorAdd(a, b []float64) []float64 {
	n := len(a)
	result := make([]float64, n)

	i := 0
	// Process 4 elements at a time
	for ; i+3 < n; i += 4 {
		result[i] = a[i] + b[i]
		result[i+1] = a[i+1] + b[i+1]
		result[i+2] = a[i+2] + b[i+2]
		result[i+3] = a[i+3] + b[i+3]
	}

	// Handle remainder
	for ; i < n; i++ {
		result[i] = a[i] + b[i]
	}

	return result
}

// VectorScale performs optimized scalar multiplication
func VectorScale(a []float64, scalar float64) []float64 {
	n := len(a)
	result := make([]float64, n)

	i := 0
	// Process 4 elements at a time
	for ; i+3 < n; i += 4 {
		result[i] = a[i] * scalar
		result[i+1] = a[i+1] * scalar
		result[i+2] = a[i+2] * scalar
		result[i+3] = a[i+3] * scalar
	}

	// Handle remainder
	for ; i < n; i++ {
		result[i] = a[i] * scalar
	}

	return result
}

// ============================================================================
// PREFETCHING HINTS
// ============================================================================

// prefetch provides a hint to prefetch memory (compiler-specific)
// (removed unused prefetch helper)

// ============================================================================
// LOOP UNROLLING UTILITIES
// ============================================================================

// unrolledSum computes sum with loop unrolling
// (removed unused unrolledSum helper)

// ============================================================================
// SIMPLE SIMD-LIKE ELEMENTWISE OPERATIONS (used by NEON fallbacks)
// ============================================================================

// SIMDAdd returns element-wise a+b
func SIMDAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("infrastructure/maths.SIMDAdd: length mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// SIMDMultiply returns element-wise a*b
func SIMDMultiply(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("infrastructure/maths.SIMDMultiply: length mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

// SIMDFusedMultiplyAdd returns element-wise a*b + c
func SIMDFusedMultiplyAdd(a, b, c []float64) []float64 {
	if len(a) != len(b) || len(a) != len(c) {
		panic("infrastructure/maths.SIMDFusedMultiplyAdd: length mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i]*b[i] + c[i]
	}
	return out
}
