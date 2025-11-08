package parallel

import (
    "runtime"
    "sync"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
)

// ParallelConfig holds parallelization settings
type ParallelConfig struct {
    NumWorkers    int
    ChunkSize     int
    UseGoroutines bool
}

// NewParallelConfig creates optimal parallel configuration
func NewParallelConfig() *ParallelConfig {
    return &ParallelConfig{
        NumWorkers:    runtime.NumCPU(),
        ChunkSize:     64,
        UseGoroutines: true,
    }
}

// ParallelFor executes function over range in parallel
func ParallelFor(start, end int, config *ParallelConfig, fn func(int)) {
    if config == nil {
        cfg := NewParallelConfig()
        config = cfg
    }
    if !config.UseGoroutines || end-start < config.ChunkSize {
        for i := start; i < end; i++ { fn(i) }
        return
    }

    numWorkers := config.NumWorkers
    if numWorkers < 1 { numWorkers = 1 }
    chunkSize := (end - start + numWorkers - 1) / numWorkers

    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func(worker int) {
            defer wg.Done()
            workerStart := start + worker*chunkSize
            workerEnd := ints.Min(workerStart+chunkSize, end)
            for i := workerStart; i < workerEnd; i++ { fn(i) }
        }(w)
    }
    wg.Wait()
}

// ParallelMap applies function to slice in parallel
func ParallelMap[T any, R any](input []T, config *ParallelConfig, fn func(T) R) []R {
    n := len(input)
    result := make([]R, n)
    ParallelFor(0, n, config, func(i int) { result[i] = fn(input[i]) })
    return result
}

// ParallelReduce reduces slice in parallel
func ParallelReduce[T any](input []T, config *ParallelConfig, identity T, fn func(T, T) T) T {
    n := len(input)
    if n == 0 { return identity }
    if n == 1 { return input[0] }

    numWorkers := config.NumWorkers
    if numWorkers < 1 { numWorkers = 1 }
    chunkSize := (n + numWorkers - 1) / numWorkers

    partialResults := make([]T, numWorkers)
    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func(worker int) {
            defer wg.Done()
            workerStart := worker * chunkSize
            workerEnd := ints.Min(workerStart+chunkSize, n)
            if workerStart >= n { partialResults[worker] = identity; return }
            acc := input[workerStart]
            for i := workerStart + 1; i < workerEnd; i++ { acc = fn(acc, input[i]) }
            partialResults[worker] = acc
        }(w)
    }
    wg.Wait()

    result := partialResults[0]
    for i := 1; i < numWorkers; i++ { result = fn(result, partialResults[i]) }
    return result
}

// minInt moved to internal/ints
