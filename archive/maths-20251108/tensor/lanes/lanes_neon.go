package lanes

// NEON-style 4-wide unrolled helpers for float32 reductions. Portable Go loops
// sized to NEON lanes; compilers may auto-vectorize.

// ReduceMaxF32_NEON returns the maximum value of a float32 slice. Returns 0 for empty.
func ReduceMaxF32_NEON(x []float32) float32 {
    n := len(x)
    if n == 0 { return 0 }
    // Initialize with up to 4 elements
    i := 0
    m0, m1, m2, m3 := x[0], x[0], x[0], x[0]
    for ; i < n && i < 4; i++ {
        switch i {
        case 0: m0 = x[i]
        case 1: m1 = x[i]
        case 2: m2 = x[i]
        case 3: m3 = x[i]
        }
    }
    for ; i+3 < n; i += 4 {
        if x[i+0] > m0 { m0 = x[i+0] }
        if x[i+1] > m1 { m1 = x[i+1] }
        if x[i+2] > m2 { m2 = x[i+2] }
        if x[i+3] > m3 { m3 = x[i+3] }
    }
    maxv := m0
    if m1 > maxv { maxv = m1 }
    if m2 > maxv { maxv = m2 }
    if m3 > maxv { maxv = m3 }
    for ; i < n; i++ {
        if x[i] > maxv { maxv = x[i] }
    }
    return maxv
}

// ReduceSumF32_NEON sums a float32 slice using 4-wide unrolling.
func ReduceSumF32_NEON(x []float32) float32 {
    n := len(x)
    var s0, s1, s2, s3 float32
    i := 0
    for ; i+3 < n; i += 4 {
        s0 += x[i+0]
        s1 += x[i+1]
        s2 += x[i+2]
        s3 += x[i+3]
    }
    sum := (s0 + s1) + (s2 + s3)
    for ; i < n; i++ { sum += x[i] }
    return sum
}

