package parallel

import (
	"context"
	"runtime"
	"sync"
)

// AutoConfig contains configuration for auto-parallelization
type AutoConfig struct {
	MinElements int             // Minimum elements to trigger parallelization
	MaxWorkers  int             // Maximum number of workers
	ChunkSize   int             // Elements per chunk
	EnableNUMA  bool            // Enable NUMA-aware scheduling
	Context     context.Context // Context for cancellation
}

// DefaultAutoConfig returns a sensible default configuration
func DefaultAutoConfig() *AutoConfig {
	return &AutoConfig{
		MinElements: 10000, // Parallelize operations on 10K+ elements
		MaxWorkers:  runtime.NumCPU(),
		ChunkSize:   1000, // 1K elements per chunk
		EnableNUMA:  false,
		Context:     context.Background(),
	}
}

// AutoParallelizer manages automatic parallelization of operations
type AutoParallelizer struct {
	config     *AutoConfig
	workerPool *WorkerPool
}

// NewAutoParallelizer creates a new auto-parallelizer
func NewAutoParallelizer(config *AutoConfig) *AutoParallelizer {
	if config == nil {
		config = DefaultAutoConfig()
	}

	return &AutoParallelizer{
		config:     config,
		workerPool: NewWorkerPool(config.MaxWorkers),
	}
}

// ShouldParallelize determines if an operation should be parallelized
func (ap *AutoParallelizer) ShouldParallelize(elementCount int) bool {
	return elementCount >= ap.config.MinElements
}

// AutoParallelMap applies a function to each element in parallel.
func AutoParallelMap[T any, R any](ap *AutoParallelizer, data []T, fn func(T) R) []R {
	if !ap.ShouldParallelize(len(data)) {
		result := make([]R, len(data))
		for i, v := range data {
			result[i] = fn(v)
		}
		return result
	}

	result := make([]R, len(data))
	chunks := ap.createChunks(len(data))

	var wg sync.WaitGroup
	for _, chunk := range chunks {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				result[i] = fn(data[i])
			}
		}(chunk.Start, chunk.End)
	}

	wg.Wait()
	return result
}

// AutoParallelReduce reduces an array using a binary operation.
func AutoParallelReduce[T any](ap *AutoParallelizer, data []T, init T, op func(T, T) T) T {
	if !ap.ShouldParallelize(len(data)) {
		result := init
		for _, v := range data {
			result = op(result, v)
		}
		return result
	}

	chunks := ap.createChunks(len(data))
	chunkResults := make([]T, len(chunks))

	var wg sync.WaitGroup
	for i, chunk := range chunks {
		wg.Add(1)
		go func(chunkIdx int, start, end int) {
			defer wg.Done()
			result := init
			for j := start; j < end; j++ {
				result = op(result, data[j])
			}
			chunkResults[chunkIdx] = result
		}(i, chunk.Start, chunk.End)
	}

	wg.Wait()

	result := init
	for _, chunkResult := range chunkResults {
		result = op(result, chunkResult)
	}

	return result
}

// AutoParallelScan performs a parallel prefix scan (inclusive).
func AutoParallelScan[T any](ap *AutoParallelizer, data []T, op func(T, T) T) []T {
	if !ap.ShouldParallelize(len(data)) {
		result := make([]T, len(data))
		if len(data) == 0 {
			return result
		}
		result[0] = data[0]
		for i := 1; i < len(data); i++ {
			result[i] = op(result[i-1], data[i])
		}
		return result
	}

	return parallelScanImpl(ap, data, op)
}

func parallelScanImpl[T any](ap *AutoParallelizer, data []T, op func(T, T) T) []T {
	n := len(data)
	if n == 0 {
		return []T{}
	}

	upSweep := make([]T, n)
	copy(upSweep, data)

	for d := 1; d < n; d *= 2 {
		var wg sync.WaitGroup
		for i := d; i < n; i += 2 * d {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				upSweep[idx] = op(upSweep[idx-d], upSweep[idx])
			}(i)
		}
		wg.Wait()
	}

	downSweep := make([]T, n)
	copy(downSweep, upSweep)
	downSweep[n-1] = data[n-1]

	for d := n / 2; d >= 1; d /= 2 {
		var wg sync.WaitGroup
		for i := d; i < n; i += 2 * d {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				temp := downSweep[idx-d]
				downSweep[idx-d] = downSweep[idx]
				downSweep[idx] = op(downSweep[idx], temp)
			}(i)
		}
		wg.Wait()
	}

	return downSweep
}

// AutoParallelSort sorts data in parallel using a merge sort.
func AutoParallelSort[T any](ap *AutoParallelizer, data []T, less func(T, T) bool) {
	if !ap.ShouldParallelize(len(data)) || len(data) <= 1 {
		return
	}
	parallelMergeSort(data, less)
}

func parallelMergeSort[T any](data []T, less func(T, T) bool) {
	if len(data) <= 1 {
		return
	}

	mid := len(data) / 2
	left := append([]T(nil), data[:mid]...)
	right := append([]T(nil), data[mid:]...)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		parallelMergeSort(left, less)
	}()

	go func() {
		defer wg.Done()
		parallelMergeSort(right, less)
	}()

	wg.Wait()
	merge(data, left, right, less)
}

func merge[T any](data, left, right []T, less func(T, T) bool) {
	i, j, k := 0, 0, 0

	for i < len(left) && j < len(right) {
		if less(left[i], right[j]) {
			data[k] = left[i]
			i++
		} else {
			data[k] = right[j]
			j++
		}
		k++
	}

	for i < len(left) {
		data[k] = left[i]
		i++
		k++
	}

	for j < len(right) {
		data[k] = right[j]
		j++
		k++
	}
}

// Chunk represents a work chunk
type Chunk struct {
	Start int
	End   int
}

// createChunks creates work chunks for parallel processing
func (ap *AutoParallelizer) createChunks(totalElements int) []Chunk {
	if totalElements <= ap.config.ChunkSize {
		return []Chunk{{Start: 0, End: totalElements}}
	}

	numChunks := (totalElements + ap.config.ChunkSize - 1) / ap.config.ChunkSize
	if numChunks > ap.config.MaxWorkers {
		numChunks = ap.config.MaxWorkers
	}

	chunks := make([]Chunk, numChunks)
	chunkSize := totalElements / numChunks
	remainder := totalElements % numChunks

	start := 0
	for i := 0; i < numChunks; i++ {
		end := start + chunkSize
		if i < remainder {
			end++
		}
		chunks[i] = Chunk{Start: start, End: end}
		start = end
	}

	return chunks
}

// WorkerPool manages a pool of workers
type WorkerPool struct {
	workers int
	jobs    chan func()
	wg      sync.WaitGroup
	ctx     context.Context
	cancel  context.CancelFunc
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workers int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	pool := &WorkerPool{
		workers: workers,
		jobs:    make(chan func(), workers*2),
		ctx:     ctx,
		cancel:  cancel,
	}

	// Start workers
	for i := 0; i < workers; i++ {
		pool.wg.Add(1)
		go pool.worker()
	}

	return pool
}

// worker runs a worker goroutine
func (wp *WorkerPool) worker() {
	defer wp.wg.Done()
	for {
		select {
		case job := <-wp.jobs:
			job()
		case <-wp.ctx.Done():
			return
		}
	}
}

// Submit submits a job to the worker pool
func (wp *WorkerPool) Submit(job func()) {
	select {
	case wp.jobs <- job:
	case <-wp.ctx.Done():
		// Pool is shutting down
	}
}

// Close shuts down the worker pool
func (wp *WorkerPool) Close() {
	wp.cancel()
	close(wp.jobs)
	wp.wg.Wait()
}

// NUMA-aware scheduling (placeholder for future implementation)
type NUMAScheduler struct {
	// This would contain NUMA topology information
	// and scheduling logic for NUMA-aware operations
}

// NewNUMAScheduler creates a NUMA-aware scheduler
func NewNUMAScheduler() *NUMAScheduler {
	// This would detect NUMA topology and create appropriate scheduling
	return &NUMAScheduler{}
}

// ScheduleNUMA schedules work across NUMA nodes
func (ns *NUMAScheduler) ScheduleNUMA(work func(int, int), totalWork int) {
	// This would distribute work across NUMA nodes
	// For now, just run sequentially
	work(0, totalWork)
}
