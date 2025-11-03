package lanes

// Vector-lane helpers for AVX-512 (f32) and NEON (f32/f64) style processing.
// These are written in portable Go with block-unrolling sized to common lane
// widths. Compilers may auto-vectorize parts; the structure matches lane sizes
// to simplify future hand-optimized assembly drops.

import (
    "runtime"
    cpu "golang.org/x/sys/cpu"
)

// LaneAddF32_AVX512 adds float32 slices using 16-wide lanes (emulating AVX-512).
func LaneAddF32_AVX512(a, b []float32) []float32 {
	n := len(a)
	if len(b) != n {
		panic("LaneAddF32_AVX512: length mismatch")
	}
	out := make([]float32, n)
	i := 0
	for ; i+15 < n; i += 16 {
		out[i+0] = a[i+0] + b[i+0]
		out[i+1] = a[i+1] + b[i+1]
		out[i+2] = a[i+2] + b[i+2]
		out[i+3] = a[i+3] + b[i+3]
		out[i+4] = a[i+4] + b[i+4]
		out[i+5] = a[i+5] + b[i+5]
		out[i+6] = a[i+6] + b[i+6]
		out[i+7] = a[i+7] + b[i+7]
		out[i+8] = a[i+8] + b[i+8]
		out[i+9] = a[i+9] + b[i+9]
		out[i+10] = a[i+10] + b[i+10]
		out[i+11] = a[i+11] + b[i+11]
		out[i+12] = a[i+12] + b[i+12]
		out[i+13] = a[i+13] + b[i+13]
		out[i+14] = a[i+14] + b[i+14]
		out[i+15] = a[i+15] + b[i+15]
	}
	for ; i < n; i++ {
		out[i] = a[i] + b[i]
	}
	return out
}

// LaneMulF32_AVX512 multiplies float32 slices using 16-wide lanes.
func LaneMulF32_AVX512(a, b []float32) []float32 {
	n := len(a)
	if len(b) != n {
		panic("LaneMulF32_AVX512: length mismatch")
	}
	out := make([]float32, n)
	i := 0
	for ; i+15 < n; i += 16 {
		out[i+0] = a[i+0] * b[i+0]
		out[i+1] = a[i+1] * b[i+1]
		out[i+2] = a[i+2] * b[i+2]
		out[i+3] = a[i+3] * b[i+3]
		out[i+4] = a[i+4] * b[i+4]
		out[i+5] = a[i+5] * b[i+5]
		out[i+6] = a[i+6] * b[i+6]
		out[i+7] = a[i+7] * b[i+7]
		out[i+8] = a[i+8] * b[i+8]
		out[i+9] = a[i+9] * b[i+9]
		out[i+10] = a[i+10] * b[i+10]
		out[i+11] = a[i+11] * b[i+11]
		out[i+12] = a[i+12] * b[i+12]
		out[i+13] = a[i+13] * b[i+13]
		out[i+14] = a[i+14] * b[i+14]
		out[i+15] = a[i+15] * b[i+15]
	}
	for ; i < n; i++ {
		out[i] = a[i] * b[i]
	}
	return out
}

// LaneFMAF32_AVX512 computes a*b + c for float32 with 16-wide unrolling.
func LaneFMAF32_AVX512(a, b, c []float32) []float32 {
	n := len(a)
	if len(b) != n || len(c) != n {
		panic("LaneFMAF32_AVX512: length mismatch")
	}
	out := make([]float32, n)
	i := 0
	for ; i+15 < n; i += 16 {
		out[i+0] = a[i+0]*b[i+0] + c[i+0]
		out[i+1] = a[i+1]*b[i+1] + c[i+1]
		out[i+2] = a[i+2]*b[i+2] + c[i+2]
		out[i+3] = a[i+3]*b[i+3] + c[i+3]
		out[i+4] = a[i+4]*b[i+4] + c[i+4]
		out[i+5] = a[i+5]*b[i+5] + c[i+5]
		out[i+6] = a[i+6]*b[i+6] + c[i+6]
		out[i+7] = a[i+7]*b[i+7] + c[i+7]
		out[i+8] = a[i+8]*b[i+8] + c[i+8]
		out[i+9] = a[i+9]*b[i+9] + c[i+9]
		out[i+10] = a[i+10]*b[i+10] + c[i+10]
		out[i+11] = a[i+11]*b[i+11] + c[i+11]
		out[i+12] = a[i+12]*b[i+12] + c[i+12]
		out[i+13] = a[i+13]*b[i+13] + c[i+13]
		out[i+14] = a[i+14]*b[i+14] + c[i+14]
		out[i+15] = a[i+15]*b[i+15] + c[i+15]
	}
	for ; i < n; i++ {
		out[i] = a[i]*b[i] + c[i]
	}
	return out
}

// LaneDotF32_NEON computes dot product for float32 with NEON-style 4-lane unrolling.
func LaneDotF32_NEON(a, b []float32) float32 {
	n := len(a)
	if len(b) != n {
		panic("LaneDotF32_NEON: length mismatch")
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		s0 += a[i+0] * b[i+0]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	sum := s0 + s1 + s2 + s3
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// SelectOptimalF32Add chooses lane implementation by arch.
func SelectOptimalF32Add(a, b []float32) []float32 {
    if runtime.GOARCH == "arm64" {
        // NEON 4-lane style
        n := len(a)
        if len(b) != n {
            panic("SelectOptimalF32Add: length mismatch")
        }
        out := make([]float32, n)
        i := 0
        for ; i+3 < n; i += 4 {
            out[i+0] = a[i+0] + b[i+0]
            out[i+1] = a[i+1] + b[i+1]
            out[i+2] = a[i+2] + b[i+2]
            out[i+3] = a[i+3] + b[i+3]
        }
        for ; i < n; i++ {
            out[i] = a[i] + b[i]
        }
        return out
    }
    if runtime.GOARCH == "amd64" {
        if cpu.X86.HasAVX512F {
            return LaneAddF32_AVX512(a, b)
        }
        if cpu.X86.HasAVX2 {
            return LaneAddF32_AVX2(a, b)
        }
    }
    // Fallback generic
    n := len(a)
    if len(b) != n { panic("SelectOptimalF32Add: length mismatch") }
    out := make([]float32, n)
    for i := 0; i < n; i++ { out[i] = a[i] + b[i] }
    return out
}
