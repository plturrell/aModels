package parallel

import (
    "sync/atomic"
    "testing"
)

func TestParallelFor_WritesAll(t *testing.T) {
    n := 1000
    got := make([]int32, n)
    cfg := NewParallelConfig()
    ParallelFor(0, n, cfg, func(i int) { atomic.StoreInt32(&got[i], 1) })
    for i := 0; i < n; i++ {
        if atomic.LoadInt32(&got[i]) != 1 {
            t.Fatalf("index %d not processed", i)
        }
    }
}

func TestParallelMap_Simple(t *testing.T) {
    in := []int{1,2,3,4,5}
    out := ParallelMap(in, NewParallelConfig(), func(v int) int { return v*v })
    want := []int{1,4,9,16,25}
    if len(out) != len(want) { t.Fatalf("len mismatch: %d vs %d", len(out), len(want)) }
    for i := range want {
        if out[i] != want[i] { t.Fatalf("idx %d: got %d want %d", i, out[i], want[i]) }
    }
}

func TestParallelReduce_Sum(t *testing.T) {
    in := []int{1,2,3,4,5}
    sum := ParallelReduce(in, NewParallelConfig(), 0, func(a,b int) int { return a+b })
    if sum != 15 { t.Fatalf("sum got %d want 15", sum) }
}

func TestParallelReduce_Empty(t *testing.T) {
    var in []int
    sum := ParallelReduce(in, NewParallelConfig(), 7, func(a,b int) int { return a+b })
    if sum != 7 { t.Fatalf("empty reduce got %d want 7", sum) }
}

