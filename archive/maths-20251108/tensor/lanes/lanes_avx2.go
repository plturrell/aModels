package lanes

// AVX2-style 8-wide unrolled helpers for float32 operations. These are
// portable Go loops sized to common AVX2 lane widths; compilers can
// auto-vectorize them. They preserve ABI by being additive.

func LaneAddF32_AVX2(a, b []float32) []float32 {
    n := len(a)
    if len(b) != n { panic("LaneAddF32_AVX2: length mismatch") }
    out := make([]float32, n)
    i := 0
    for ; i+7 < n; i += 8 {
        out[i+0] = a[i+0] + b[i+0]
        out[i+1] = a[i+1] + b[i+1]
        out[i+2] = a[i+2] + b[i+2]
        out[i+3] = a[i+3] + b[i+3]
        out[i+4] = a[i+4] + b[i+4]
        out[i+5] = a[i+5] + b[i+5]
        out[i+6] = a[i+6] + b[i+6]
        out[i+7] = a[i+7] + b[i+7]
    }
    for ; i < n; i++ { out[i] = a[i] + b[i] }
    return out
}

func LaneMulF32_AVX2(a, b []float32) []float32 {
    n := len(a)
    if len(b) != n { panic("LaneMulF32_AVX2: length mismatch") }
    out := make([]float32, n)
    i := 0
    for ; i+7 < n; i += 8 {
        out[i+0] = a[i+0] * b[i+0]
        out[i+1] = a[i+1] * b[i+1]
        out[i+2] = a[i+2] * b[i+2]
        out[i+3] = a[i+3] * b[i+3]
        out[i+4] = a[i+4] * b[i+4]
        out[i+5] = a[i+5] * b[i+5]
        out[i+6] = a[i+6] * b[i+6]
        out[i+7] = a[i+7] * b[i+7]
    }
    for ; i < n; i++ { out[i] = a[i] * b[i] }
    return out
}

func LaneFMAF32_AVX2(a, b, c []float32) []float32 {
    n := len(a)
    if len(b) != n || len(c) != n { panic("LaneFMAF32_AVX2: length mismatch") }
    out := make([]float32, n)
    i := 0
    for ; i+7 < n; i += 8 {
        out[i+0] = a[i+0]*b[i+0] + c[i+0]
        out[i+1] = a[i+1]*b[i+1] + c[i+1]
        out[i+2] = a[i+2]*b[i+2] + c[i+2]
        out[i+3] = a[i+3]*b[i+3] + c[i+3]
        out[i+4] = a[i+4]*b[i+4] + c[i+4]
        out[i+5] = a[i+5]*b[i+5] + c[i+5]
        out[i+6] = a[i+6]*b[i+6] + c[i+6]
        out[i+7] = a[i+7]*b[i+7] + c[i+7]
    }
    for ; i < n; i++ { out[i] = a[i]*b[i] + c[i] }
    return out
}

// DotF32_AVX2 computes the dot product of two float32 slices using 8-wide
// unrolling to match AVX2 lanes. Panics on length mismatch.
func DotF32_AVX2(a, b []float32) float32 {
    n := len(a)
    if len(b) != n { panic("DotF32_AVX2: length mismatch") }
    var s0, s1, s2, s3, s4, s5, s6, s7 float32
    i := 0
    for ; i+7 < n; i += 8 {
        s0 += a[i+0] * b[i+0]
        s1 += a[i+1] * b[i+1]
        s2 += a[i+2] * b[i+2]
        s3 += a[i+3] * b[i+3]
        s4 += a[i+4] * b[i+4]
        s5 += a[i+5] * b[i+5]
        s6 += a[i+6] * b[i+6]
        s7 += a[i+7] * b[i+7]
    }
    sum := (((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)))
    for ; i < n; i++ { sum += a[i] * b[i] }
    return sum
}

// ReduceSumF32_AVX2 sums a float32 slice using 8-wide unrolling.
func ReduceSumF32_AVX2(x []float32) float32 {
    n := len(x)
    var s0, s1, s2, s3, s4, s5, s6, s7 float32
    i := 0
    for ; i+7 < n; i += 8 {
        s0 += x[i+0]
        s1 += x[i+1]
        s2 += x[i+2]
        s3 += x[i+3]
        s4 += x[i+4]
        s5 += x[i+5]
        s6 += x[i+6]
        s7 += x[i+7]
    }
    sum := (((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7)))
    for ; i < n; i++ { sum += x[i] }
    return sum
}

// ReduceMaxF32_AVX2 returns the maximum value of a float32 slice using
// 8-wide unrolling. Returns 0 for empty input.
func ReduceMaxF32_AVX2(x []float32) float32 {
    n := len(x)
    if n == 0 { return 0 }
    // Initialize with first up to 8 elements
    i := 0
    m0, m1, m2, m3, m4, m5, m6, m7 := x[0], x[0], x[0], x[0], x[0], x[0], x[0], x[0]
    for ; i < n && i < 8; i++ {
        switch i {
        case 0: m0 = x[i]
        case 1: m1 = x[i]
        case 2: m2 = x[i]
        case 3: m3 = x[i]
        case 4: m4 = x[i]
        case 5: m5 = x[i]
        case 6: m6 = x[i]
        case 7: m7 = x[i]
        }
    }
    for ; i+7 < n; i += 8 {
        if x[i+0] > m0 { m0 = x[i+0] }
        if x[i+1] > m1 { m1 = x[i+1] }
        if x[i+2] > m2 { m2 = x[i+2] }
        if x[i+3] > m3 { m3 = x[i+3] }
        if x[i+4] > m4 { m4 = x[i+4] }
        if x[i+5] > m5 { m5 = x[i+5] }
        if x[i+6] > m6 { m6 = x[i+6] }
        if x[i+7] > m7 { m7 = x[i+7] }
    }
    maxv := m0
    if m1 > maxv { maxv = m1 }
    if m2 > maxv { maxv = m2 }
    if m3 > maxv { maxv = m3 }
    if m4 > maxv { maxv = m4 }
    if m5 > maxv { maxv = m5 }
    if m6 > maxv { maxv = m6 }
    if m7 > maxv { maxv = m7 }
    for ; i < n; i++ {
        if x[i] > maxv { maxv = x[i] }
    }
    return maxv
}
