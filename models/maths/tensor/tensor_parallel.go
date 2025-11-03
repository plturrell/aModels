package tensor

import (
    "fmt"
    "sync"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
    par "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/parallel"
    ops "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/ops"
)

// ============================================================================
// PARALLEL OPTIMIZATION (Rayon-style parallelism for Go)
// ============================================================================

// Re-export parallel primitives from subpackage to preserve API
type ParallelConfig = par.ParallelConfig
func NewParallelConfig() *ParallelConfig { return par.NewParallelConfig() }

// ParallelFor executes function over range in parallel (like rayon)
func ParallelFor(start, end int, config *ParallelConfig, fn func(int)) { par.ParallelFor(start, end, config, fn) }

// ParallelMap applies function to slice in parallel
func ParallelMap[T any, R any](input []T, config *ParallelConfig, fn func(T) R) []R { return par.ParallelMap(input, config, fn) }

// ParallelReduce reduces slice in parallel
func ParallelReduce[T any](input []T, config *ParallelConfig, identity T, fn func(T, T) T) T { return par.ParallelReduce(input, config, identity, fn) }

// ============================================================================
// PARALLEL MATRIX OPERATIONS
// ============================================================================

// ParallelMatMul performs parallel matrix multiplication
func (t *TensorOps) ParallelMatMul(A, B [][]float64) ([][]float64, error) {
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

	config := NewParallelConfig()

	// Parallel over rows with cache blocking
	const blockSize = 64
	ParallelFor(0, m, config, func(i int) {
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
	})

	return C, nil
}

// ParallelBatchGEMM performs batched matrix multiplication in parallel
func (t *TensorOps) ParallelBatchGEMM(A, B [][][]float64, applySoftmax bool) ([][][]float64, error) {
	batchSize := len(A)
	if len(B) != batchSize {
		return nil, ErrDimensionMismatch
	}

	result := make([][][]float64, batchSize)
	config := NewParallelConfig()

	// Parallel over batches
	ParallelFor(0, batchSize, config, func(batch int) {
		m, k := len(A[batch]), len(A[batch][0])
		n := len(B[batch][0])

		result[batch] = matmul(A[batch], B[batch], m, n, k)

        if applySoftmax { result[batch] = ops.SoftmaxMatrix(result[batch]) }
	})

	return result, nil
}

// ============================================================================
// PARALLEL DATA PREPARATION
// ============================================================================

// ParallelWhiten performs parallel data whitening
func (t *TensorOps) ParallelWhiten(data [][]float64, bufferPool *MatrixBufferPool) ([][]float64, error) {
	m, n := len(data), len(data[0])

	// Compute mean in parallel
	config := NewParallelConfig()
	mean := make([]float64, n)

	// Parallel mean computation
	partialSums := make([][]float64, config.NumWorkers)
	for i := range partialSums {
		partialSums[i] = make([]float64, n)
	}

	chunkSize := (m + config.NumWorkers - 1) / config.NumWorkers
	var wg sync.WaitGroup

	for w := 0; w < config.NumWorkers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()

			start := worker * chunkSize
            end := ints.Min(start+chunkSize, m)

			for i := start; i < end; i++ {
				for j := 0; j < n; j++ {
					partialSums[worker][j] += data[i][j]
				}
			}
		}(w)
	}
	wg.Wait()

	// Combine partial sums
	for j := 0; j < n; j++ {
		for w := 0; w < config.NumWorkers; w++ {
			mean[j] += partialSums[w][j]
		}
		mean[j] /= float64(m)
	}

	// Center data in parallel (reuse buffer if provided)
	var centered [][]float64
	if bufferPool != nil {
		centered = bufferPool.Get(m, n)
	} else {
		centered = make([][]float64, m)
		for i := 0; i < m; i++ {
			centered[i] = make([]float64, n)
		}
	}

	ParallelFor(0, m, config, func(i int) {
		for j := 0; j < n; j++ {
			centered[i][j] = data[i][j] - mean[j]
		}
	})

	return centered, nil
}

// ParallelMSE computes mean squared error in parallel
func (t *TensorOps) ParallelMSE(predicted, actual [][]float64) (float64, error) {
	m, n := len(predicted), len(predicted[0])
	if len(actual) != m || len(actual[0]) != n {
		return 0, ErrDimensionMismatch
	}

	config := NewParallelConfig()

	// Parallel MSE computation with Kahan summation
	type partialResult struct {
		sum float64
		c   float64
	}

	partials := make([]partialResult, config.NumWorkers)
	chunkSize := (m + config.NumWorkers - 1) / config.NumWorkers

	var wg sync.WaitGroup
	for w := 0; w < config.NumWorkers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()

			start := worker * chunkSize
            end := ints.Min(start+chunkSize, m)

			sum := 0.0
			comp := 0.0

			for i := start; i < end; i++ {
				for j := 0; j < n; j++ {
					diff := predicted[i][j] - actual[i][j]
					y := diff*diff - comp
					t := sum + y
					comp = (t - sum) - y
					sum = t
				}
			}

			partials[worker].sum = sum
			partials[worker].c = comp
		}(w)
	}
	wg.Wait()

	// Combine with Kahan summation
	totalSum := 0.0
	totalC := 0.0
	for _, p := range partials {
		y := p.sum - totalC
		t := totalSum + y
		totalC = (t - totalSum) - y
		totalSum = t
	}

	return totalSum / float64(m*n), nil
}

// ============================================================================
// PARALLEL TRAINING UTILITIES
// ============================================================================

// ParallelEpochProcessor processes multiple training epochs in parallel
type ParallelEpochProcessor struct {
	config     *ParallelConfig
	bufferPool *MatrixBufferPool
	vectorPool *BufferPool
}

// NewParallelEpochProcessor creates a parallel epoch processor
func NewParallelEpochProcessor() *ParallelEpochProcessor {
	return &ParallelEpochProcessor{
		config:     NewParallelConfig(),
		bufferPool: NewMatrixBufferPool(),
		vectorPool: NewBufferPool(),
	}
}

// ProcessBatch processes a batch of samples in parallel
func (pep *ParallelEpochProcessor) ProcessBatch(
	samples [][]float64,
	fn func([]float64) []float64,
) [][]float64 {
	n := len(samples)
	results := make([][]float64, n)

	ParallelFor(0, n, pep.config, func(i int) {
		results[i] = fn(samples[i])
	})

	return results
}

// ParallelGradientCompute computes gradients in parallel
func (pep *ParallelEpochProcessor) ParallelGradientCompute(
	predictions, targets [][]float64,
	learningRate float64,
) ([][]float64, error) {
	m, n := len(predictions), len(predictions[0])
	if len(targets) != m || len(targets[0]) != n {
		return nil, ErrDimensionMismatch
	}

	gradients := pep.bufferPool.Get(m, n)

	ParallelFor(0, m, pep.config, func(i int) {
		for j := 0; j < n; j++ {
			gradients[i][j] = (predictions[i][j] - targets[i][j]) * learningRate
		}
	})

	return gradients, nil
}

// Cleanup returns buffers to pool
func (pep *ParallelEpochProcessor) Cleanup(matrices ...[][]float64) {
	for _, mat := range matrices {
		pep.bufferPool.Put(mat)
	}
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

var ErrDimensionMismatch = fmt.Errorf("dimension mismatch")

// minInt moved to internal/ints
