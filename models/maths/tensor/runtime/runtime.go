package runtime

import (
    "runtime"
    "sync"
    ints "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/tensor/internal/ints"
)

// LockFreeSum performs a numerically-stable parallel sum using per-worker
// Kahan compensation and a final sequential Kahan combine. This avoids
// undefined behavior of atomic float bit-adding and returns a correct sum.
func LockFreeSum(data []float64) float64 {
    n := len(data)
    if n == 0 { return 0 }

    workers := runtime.NumCPU()
    if workers < 1 { workers = 1 }
    chunk := (n + workers - 1) / workers

    type partial struct{ sum, c float64 }
    parts := make([]partial, workers)

    var wg sync.WaitGroup
    for w := 0; w < workers; w++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            start := id * chunk
            end := ints.Min(start+chunk, n)
            s := 0.0
            c := 0.0
            for i := start; i < end; i++ {
                y := data[i] - c
                t := s + y
                c = (t - s) - y
                s = t
            }
            parts[id] = partial{sum: s, c: c}
        }(w)
    }
    wg.Wait()

    // Final Kahan combine of partials
    total := 0.0
    comp := 0.0
    for i := 0; i < workers; i++ {
        y := parts[i].sum - comp
        t := total + y
        comp = (t - total) - y
        total = t
    }
    return total
}

// minInt moved to internal/ints
