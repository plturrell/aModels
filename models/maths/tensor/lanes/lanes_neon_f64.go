package lanes

// NEON-style 2-wide unrolled helpers for float64 reductions.

// ReduceMaxF64_NEON returns the maximum value of a float64 slice. Returns 0 for empty.
func ReduceMaxF64_NEON(x []float64) float64 {
    n := len(x)
    if n == 0 { return 0 }
    i := 0
    m0, m1 := x[0], x[0]
    for ; i < n && i < 2; i++ {
        if i == 0 { m0 = x[i] } else { m1 = x[i] }
    }
    for ; i+1 < n; i += 2 {
        if x[i+0] > m0 { m0 = x[i+0] }
        if x[i+1] > m1 { m1 = x[i+1] }
    }
    maxv := m0
    if m1 > maxv { maxv = m1 }
    for ; i < n; i++ { if x[i] > maxv { maxv = x[i] } }
    return maxv
}

// ReduceSumF64_NEON sums a float64 slice using 2-wide unrolling.
func ReduceSumF64_NEON(x []float64) float64 {
    n := len(x)
    var s0, s1 float64
    i := 0
    for ; i+1 < n; i += 2 {
        s0 += x[i+0]
        s1 += x[i+1]
    }
    sum := s0 + s1
    for ; i < n; i++ { sum += x[i] }
    return sum
}

