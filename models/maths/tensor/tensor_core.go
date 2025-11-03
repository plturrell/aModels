package tensor

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	gemm "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/gemm"
	ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
	linalg "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/linalg"
	ops "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/ops"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
)

// TensorOps provides high-performance tensor operations
// Can use Fortran backend (via CGO) or pure Go implementation
type TensorOps struct {
	useFortran  bool
	numThreads  int
	blockSize   int // Cache blocking size for matmul
	useBLAS     bool
	blasConfig  *BLASConfig
	parallelCfg *ParallelConfig
	bufferPool  *MatrixBufferPool
	vectorPool  *BufferPool
}

// NewTensorOps creates a new tensor operations engine
func NewTensorOps(useFortran bool) *TensorOps {
	numThreads := runtime.NumCPU()
	blasConfig := InitBLAS(numThreads)

	return &TensorOps{
		useFortran:  useFortran,
		numThreads:  numThreads,
		blockSize:   64, // Optimized for L1 cache (64x64x8 = 32KB)
		useBLAS:     blasConfig.Available,
		blasConfig:  blasConfig,
		parallelCfg: NewParallelConfig(),
		bufferPool:  NewMatrixBufferPool(),
		vectorPool:  NewBufferPool(),
	}
}

// NewTensorOpsWithThreads creates tensor ops with explicit thread count
func NewTensorOpsWithThreads(useFortran bool, numThreads int) *TensorOps {
	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
	}

	blasConfig := InitBLAS(numThreads)
	parallelCfg := NewParallelConfig()
	parallelCfg.NumWorkers = numThreads

	return &TensorOps{
		useFortran:  useFortran,
		numThreads:  numThreads,
		blockSize:   64,
		useBLAS:     blasConfig.Available,
		blasConfig:  blasConfig,
		parallelCfg: parallelCfg,
		bufferPool:  NewMatrixBufferPool(),
		vectorPool:  NewBufferPool(),
	}
}

// SetThreads configures parallelism level
func (t *TensorOps) SetThreads(n int) {
	t.numThreads = n
}

// ============================================================================
// BATCHED OPERATIONS
// ============================================================================

// BatchedGEMM performs batched matrix multiplication with optional softmax
func (t *TensorOps) BatchedGEMM(A, B [][][]float64, applySoftmax bool) ([][][]float64, error) {
	if len(A) != len(B) {
		return nil, fmt.Errorf("batch size mismatch")
	}
	if len(A) == 0 {
		return nil, fmt.Errorf("empty batch")
	}

	batchSize := len(A)
	m := len(A[0])
	k := len(A[0][0])
	n := len(B[0][0])

	result := make([][][]float64, batchSize)

	// Parallel processing across batches
	var wg sync.WaitGroup
	for b := 0; b < batchSize; b++ {
		wg.Add(1)
		go func(batch int) {
			defer wg.Done()
			result[batch] = matmul(A[batch], B[batch], m, n, k)
			if applySoftmax {
				result[batch] = ops.SoftmaxMatrix(result[batch])
			}
		}(b)
	}
	wg.Wait()

	return result, nil
}

// BatchedSoftmax applies stable softmax to each row
func (t *TensorOps) BatchedSoftmax(X [][]float64) ([][]float64, error) {
	if len(X) == 0 {
		return nil, fmt.Errorf("empty input")
	}

	m := len(X)
	result := make([][]float64, m)

	var wg sync.WaitGroup
	for i := 0; i < m; i++ {
		wg.Add(1)
		go func(row int) {
			defer wg.Done()
			result[row] = ops.SoftmaxRow(X[row])
		}(i)
	}
	wg.Wait()

	return result, nil
}

// ============================================================================
// NORMALIZATION
// ============================================================================

// LayerNorm applies layer normalization with learnable parameters
func (t *TensorOps) LayerNorm(x, gamma, beta []float64, eps float64) ([]float64, error) {
	if len(x) != len(gamma) || len(x) != len(beta) {
		return nil, fmt.Errorf("dimension mismatch")
	}

	n := len(x)

	// Compute mean and variance using Welford's algorithm to avoid extra allocations
	mean := 0.0
	m2 := 0.0
	for i, v := range x {
		delta := v - mean
		mean += delta / float64(i+1)
		m2 += delta * (v - mean)
	}
	variance := m2 / float64(n)
	stdDev := math.Sqrt(variance + eps)
	invStd := 1.0 / stdDev

	// Normalize and scale
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = gamma[i]*(x[i]-mean)*invStd + beta[i]
	}

	return result, nil
}

// RMSNorm applies root mean square normalization
func (t *TensorOps) RMSNorm(x, gamma []float64, eps float64) ([]float64, error) {
	if len(x) != len(gamma) {
		return nil, fmt.Errorf("dimension mismatch")
	}

	n := len(x)

	// Compute sum of squares without allocating intermediate slices
	sumSq := 0.0
	for _, v := range x {
		sumSq += v * v
	}
	rms := math.Sqrt(sumSq/float64(n) + eps)
	invRms := 1.0 / rms

	// Normalize
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = gamma[i] * x[i] * invRms
	}

	return result, nil
}

// ============================================================================
// TENSOR CONTRACTIONS
// ============================================================================

// TensorContract3 performs 3-way tensor contraction
func (t *TensorOps) TensorContract3(A [][][]float64, b, c, d []float64) (float64, error) {
	n1 := len(A)
	n2 := len(A[0])
	n3 := len(A[0][0])

	if len(b) != n1 || len(c) != n2 || len(d) != n3 {
		return 0, fmt.Errorf("dimension mismatch")
	}

	// Parallel contraction with Kahan summation
	type partial struct {
		sum float64
		c   float64
	}

	partials := make([]partial, t.numThreads)
	var wg sync.WaitGroup

	chunkSize := (n1 + t.numThreads - 1) / t.numThreads

	for thread := 0; thread < t.numThreads; thread++ {
		wg.Add(1)
		go func(tid int) {
			defer wg.Done()

			start := tid * chunkSize
			end := ints.Min(start+chunkSize, n1)

			sum := 0.0
			comp := 0.0

			for i := start; i < end; i++ {
				for j := 0; j < n2; j++ {
					for k := 0; k < n3; k++ {
						y := A[i][j][k]*b[i]*c[j]*d[k] - comp
						t := sum + y
						comp = (t - sum) - y
						sum = t
					}
				}
			}

			partials[tid].sum = sum
			partials[tid].c = comp
		}(thread)
	}
	wg.Wait()

	// Combine partial sums
	totalSum := 0.0
	totalC := 0.0
	for _, p := range partials {
		y := p.sum - totalC
		t := totalSum + y
		totalC = (t - totalSum) - y
		totalSum = t
	}

	return totalSum, nil
}

// ============================================================================
// FUSED OPERATIONS
// ============================================================================

// FusedMatMulBiasGELU performs fused matrix multiplication with bias addition and GELU activation
// Computes GELU(X @ W + b) where X is [m, k], W is [k, n], b is [n]
// Returns [m, n] matrix
func (t *TensorOps) FusedMatMulBiasGELU(X [][]float64, W [][]float64, b []float64) ([][]float64, error) {
	if len(X) == 0 || len(W) == 0 {
		return nil, fmt.Errorf("empty input matrices")
	}

	m := len(X)    // number of rows in X
	k := len(X[0]) // number of columns in X
	n := len(W[0]) // number of columns in W

	if len(W) != k {
		return nil, fmt.Errorf("matrix dimension mismatch: X is [%d, %d], W is [%d, %d]", m, k, len(W), len(W[0]))
	}
	if len(b) != n {
		return nil, fmt.Errorf("bias dimension mismatch: expected %d, got %d", n, len(b))
	}

	// Perform matrix multiplication
	result := matmul(X, W, m, n, k)

	// Add bias and apply GELU activation
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// Add bias
			result[i][j] += b[j]
			// Apply GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
			x := result[i][j]
			result[i][j] = gelu(x)
		}
	}

	return result, nil
}

// gelu computes the GELU (Gaussian Error Linear Unit) activation function
func gelu(x float64) float64 {
	// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
	const sqrtTwoOverPi = 0.7978845608028654 // sqrt(2/π)
	return 0.5 * x * (1.0 + math.Tanh(sqrtTwoOverPi*(x+0.044715*x*x*x)))
}

// ============================================================================
// SPD and Hyperbolic manifold operations delegated to linalg package
func (t *TensorOps) SPDExp(A, V [][]float64) ([][]float64, error)  { return linalg.SPDExp(A, V) }
func (t *TensorOps) SPDDistance(A, B [][]float64) (float64, error) { return linalg.SPDDistance(A, B) }
func (t *TensorOps) MobiusAdd(x, y []float64) ([]float64, error)   { return linalg.MobiusAdd(x, y) }
func (t *TensorOps) PoincareExp(x, v []float64) ([]float64, error) { return linalg.PoincareExp(x, v) }

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// matmul performs cache-blocked matrix multiplication
// Uses tiling to optimize L1/L2 cache usage
func matmul(A, B [][]float64, m, n, k int) [][]float64 {
	if m == 0 || n == 0 || k == 0 {
		return make([][]float64, m)
	}
	// Route through contiguous packed GEMM for better cache locality.
	// Convert 2D slices to Matrix64, run packed GEMM, convert back.
	a := util.From2D(A)
	b := util.From2D(B)
	c := gemm.MatMulContiguous(a, b)
	return util.To2D(c)
}

// min moved to internal/ints

// ============================================================================
// EIGENDECOMPOSITION (Power Iteration + Deflation)
// ============================================================================

// eigenDecomposition computes eigenvalues and eigenvectors of symmetric matrix
// Uses power iteration with deflation for symmetric matrices
func eigenDecomposition(A [][]float64) ([]float64, [][]float64, error) {
	return linalg.EigenDecomposition(A)
}

// reconstructMatrix reconstructs matrix from eigendecomposition
// M = Q * diag(λ) * Q^T
// helpers moved to ops/linalg
