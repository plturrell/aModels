package fft

import (
    "fmt"
    "math"
    "math/cmplx"
    "os"
    "runtime"
    "sync"
    "strconv"
)

// ============================================================================
// ADVANCED FFT OPERATIONS
// - FFT along selectable axes
// - Inverse transforms for complex inputs
// - Multi-dimensional FFT with axis control
// - Complex-to-complex transforms
// ============================================================================

// FFTConfig holds FFT configuration
type FFTConfig struct {
    Norm      string // "backward", "ortho", "forward"
    Workers   int    // Number of parallel workers
    UseCache  bool   // Cache twiddle factors
    Precision string // "single", "double"
    // Implementation selector (optional): if MATHS_FFT_STOCKHAM=1 and input size
    // is power-of-two, use a Stockham ping-pong kernel for improved locality.
}

// DefaultFFTConfig returns default FFT configuration
func DefaultFFTConfig() *FFTConfig {
    return &FFTConfig{
        Norm:      "backward",
        Workers:   defaultFFTWorkers(),
        UseCache:  defaultFFTCache(),
        Precision: "double",
    }
}

// FFTAxisConfig holds FFT axis configuration
type FFTAxisConfig struct {
	Axis     int
	Inverse  bool
	Norm     string // "backward", "ortho", "forward"
	Workers  int
	UseCache bool
}

// DefaultFFTAxisConfig returns default axis configuration
func DefaultFFTAxisConfig(axis int) *FFTAxisConfig {
    return &FFTAxisConfig{
        Axis:     axis,
        Inverse:  false,
        Norm:     "backward",
        Workers:  defaultFFTWorkers(),
        UseCache: defaultFFTCache(),
    }
}

// --- internals: env defaults, twiddle cache, simple parallel helpers ---

func defaultFFTWorkers() int {
    if v := os.Getenv("MATHS_FFT_WORKERS"); v != "" {
        if n, err := strconvAtoiSafe(v); err == nil && n > 0 { return n }
    }
    return runtime.NumCPU()
}

func defaultFFTCache() bool {
    if v := os.Getenv("MATHS_FFT_CACHE"); v != "" {
        if v == "1" || v == "true" || v == "TRUE" { return true }
        if v == "0" || v == "false" || v == "FALSE" { return false }
    }
    return true
}

func strconvAtoiSafe(s string) (int, error) { return strconv.Atoi(s) }

var (
    wlenCache sync.Map // key: n (int) -> []complex128 stage wlen for lengths 2,4,8,...,n
)

func getStageWlens(n int) []complex128 {
    if v, ok := wlenCache.Load(n); ok {
        return v.([]complex128)
    }
    // Compute number of stages
    stages := 0
    for l := 1; l < n; l <<= 1 { stages++ }
    wlens := make([]complex128, stages)
    idx := 0
    for length := 2; length <= n; length <<= 1 {
        ang := -2 * math.Pi / float64(length)
        wlens[idx] = complex(math.Cos(ang), math.Sin(ang))
        idx++
    }
    wlenCache.Store(n, wlens)
    return wlens
}

func parallelFor(n, workers int, fn func(i int)) {
    if workers <= 1 || n <= 1 {
        for i := 0; i < n; i++ { fn(i) }
        return
    }
    if workers > n { workers = n }
    var wg sync.WaitGroup
    // simple chunking
    chunk := (n + workers - 1) / workers
    for w := 0; w < workers; w++ {
        start := w * chunk
        end := start + chunk
        if start >= n { break }
        if end > n { end = n }
        wg.Add(1)
        go func(st, en int) {
            defer wg.Done()
            for i := st; i < en; i++ { fn(i) }
        }(start, end)
    }
    wg.Wait()
}

func useStockham() bool {
    v := os.Getenv("MATHS_FFT_STOCKHAM")
    return v == "1" || v == "true" || v == "TRUE"
}

func isPow2(n int) bool { return n > 0 && (n&(n-1)) == 0 }

// stockhamFFT computes an in-place Stockham auto-sort FFT using ping-pong buffers.
// Requires n to be a power of two. Returns a freshly-allocated output slice.
func stockhamFFT(x []complex128) []complex128 {
    n := len(x)
    if n == 0 { return nil }
    if !isPow2(n) {
        // Fallback to Cooley for non power-of-two lengths
        return cooleyFFT(append([]complex128(nil), x...))
    }
    a := append([]complex128(nil), x...)
    b := make([]complex128, n)
    // m is the half-size of butterflies; full block size M = 2*m
    for m := 1; m < n; m <<= 1 {
        M := m << 1
        ang := -2 * math.Pi / float64(M)
        w_m := complex(math.Cos(ang), math.Sin(ang))
        for k := 0; k < n; k += M {
            w := complex(1, 0)
            for j := 0; j < m; j++ {
                u := a[k+j]
                v := a[k+j+m] * w
                b[k+2*j] = u + v
                b[k+2*j+1] = u - v
                w *= w_m
            }
        }
        a, b = b, a
    }
    return a
}

// ============================================================================
// 2D FFT WITH AXIS SELECTION
// ============================================================================

// FFT2DAxis performs FFT along specified axis of 2D array
func FFT2DAxis(x [][]complex128, axis int, config *FFTAxisConfig) [][]complex128 {
    if config == nil {
        config = DefaultFFTAxisConfig(axis)
    }

	m, n := len(x), len(x[0])
	result := make([][]complex128, m)
	for i := 0; i < m; i++ {
		result[i] = make([]complex128, n)
		copy(result[i], x[i])
	}

	fftConfig := &FFTConfig{
		Norm:    config.Norm,
		Workers: config.Workers,
	}

    if axis == 0 {
        // FFT along columns (parallel by column)
        parallelFor(n, config.Workers, func(j int) {
            col := make([]complex128, m)
            for i := 0; i < m; i++ { col[i] = result[i][j] }
            var transformed []complex128
            if config.Inverse { transformed = IFFT(col, fftConfig) } else { transformed = FFT(col, fftConfig) }
            for i := 0; i < m; i++ { result[i][j] = transformed[i] }
        })
    } else {
        // FFT along rows (parallel by row)
        parallelFor(m, config.Workers, func(i int) {
            if config.Inverse { result[i] = IFFT(result[i], fftConfig) } else { result[i] = FFT(result[i], fftConfig) }
        })
    }

	return result
}

// IFFT2DAxis performs inverse FFT along specified axis
func IFFT2DAxis(x [][]complex128, axis int, config *FFTAxisConfig) [][]complex128 {
	if config == nil {
		config = DefaultFFTAxisConfig(axis)
	}
	config.Inverse = true
	return FFT2DAxis(x, axis, config)
}

// ============================================================================
// 3D FFT WITH AXIS SELECTION
// ============================================================================

// FFT3DAxis performs FFT along specified axis of 3D array
func FFT3DAxis(x [][][]complex128, axis int, config *FFTAxisConfig) [][][]complex128 {
    if config == nil {
        config = DefaultFFTAxisConfig(axis)
    }

	d1, d2, d3 := len(x), len(x[0]), len(x[0][0])
	result := make([][][]complex128, d1)
	for i := 0; i < d1; i++ {
		result[i] = make([][]complex128, d2)
		for j := 0; j < d2; j++ {
			result[i][j] = make([]complex128, d3)
			copy(result[i][j], x[i][j])
		}
	}

	fftConfig := &FFTConfig{Norm: config.Norm, Workers: config.Workers}

    switch axis {
    case 0:
        // FFT along first dimension (parallel over j,k tiles)
        parallelFor(d2*d3, config.Workers, func(idx int) {
            j := idx / d3
            k := idx % d3
            slice := make([]complex128, d1)
            for i := 0; i < d1; i++ { slice[i] = result[i][j][k] }
            var transformed []complex128
            if config.Inverse { transformed = IFFT(slice, fftConfig) } else { transformed = FFT(slice, fftConfig) }
            for i := 0; i < d1; i++ { result[i][j][k] = transformed[i] }
        })

	case 1:
        // FFT along second dimension (parallel over i,k tiles)
        parallelFor(d1*d3, config.Workers, func(idx int) {
            i := idx / d3
            k := idx % d3
            slice := make([]complex128, d2)
            for j := 0; j < d2; j++ { slice[j] = result[i][j][k] }
            var transformed []complex128
            if config.Inverse { transformed = IFFT(slice, fftConfig) } else { transformed = FFT(slice, fftConfig) }
            for j := 0; j < d2; j++ { result[i][j][k] = transformed[j] }
        })

	case 2:
        // FFT along third dimension (parallel over i,j tiles)
        parallelFor(d1*d2, config.Workers, func(idx int) {
            i := idx / d2
            j := idx % d2
            if config.Inverse { result[i][j] = IFFT(result[i][j], fftConfig) } else { result[i][j] = FFT(result[i][j], fftConfig) }
        })

	default:
		panic(fmt.Sprintf("infrastructure/maths/tensor.FFT3DAxis: invalid axis %d for 3D array", axis))
	}

	return result
}

// IFFT3DAxis performs inverse FFT along specified axis
func IFFT3DAxis(x [][][]complex128, axis int, config *FFTAxisConfig) [][][]complex128 {
	if config == nil {
		config = DefaultFFTAxisConfig(axis)
	}
	config.Inverse = true
	return FFT3DAxis(x, axis, config)
}

// ============================================================================
// MULTI-AXIS FFT
// ============================================================================

// FFTNDAxes performs FFT along multiple axes
func FFTNDAxes(x [][][]complex128, axes []int, config *FFTConfig) [][][]complex128 {
	result := x

	for _, axis := range axes {
		axisConfig := &FFTAxisConfig{
			Axis:     axis,
			Inverse:  false,
			Norm:     config.Norm,
			Workers:  config.Workers,
			UseCache: config.UseCache,
		}
		result = FFT3DAxis(result, axis, axisConfig)
	}

	return result
}

// IFFTNDAxes performs inverse FFT along multiple axes
func IFFTNDAxes(x [][][]complex128, axes []int, config *FFTConfig) [][][]complex128 {
	result := x

	// Apply inverse FFT in reverse order
	for i := len(axes) - 1; i >= 0; i-- {
		axisConfig := &FFTAxisConfig{
			Axis:     axes[i],
			Inverse:  true,
			Norm:     config.Norm,
			Workers:  config.Workers,
			UseCache: config.UseCache,
		}
		result = FFT3DAxis(result, axes[i], axisConfig)
	}

	return result
}

// ============================================================================
// COMPLEX-TO-COMPLEX TRANSFORMS
// ============================================================================

// IFFTComplex performs complex-to-complex inverse FFT
func IFFTComplex(x []complex128, config *FFTConfig) []complex128 {
	return IFFT(x, config)
}

// IFFT2DComplex performs 2D complex-to-complex inverse FFT
func IFFT2DComplex(x [][]complex128, config *FFTConfig) [][]complex128 {
	m, n := len(x), len(x[0])

	// Inverse FFT along rows
	result := make([][]complex128, m)
	for i := 0; i < m; i++ {
		result[i] = IFFT(x[i], config)
	}

	// Inverse FFT along columns
	temp := make([][]complex128, n)
	for j := 0; j < n; j++ {
		temp[j] = make([]complex128, m)
		for i := 0; i < m; i++ {
			temp[j][i] = result[i][j]
		}
		temp[j] = IFFT(temp[j], config)
	}

	// Transpose back
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result[i][j] = temp[j][i]
		}
	}

	return result
}

// ============================================================================
// FFT SHIFT OPERATIONS
// ============================================================================

// FFTShift2D shifts zero-frequency to center for 2D array
func FFTShift2D(x [][]complex128) [][]complex128 {
	m, n := len(x), len(x[0])
	result := make([][]complex128, m)

	midM := m / 2
	midN := n / 2

	for i := 0; i < m; i++ {
		result[i] = make([]complex128, n)
		for j := 0; j < n; j++ {
			srcI := (i + midM) % m
			srcJ := (j + midN) % n
			result[i][j] = x[srcI][srcJ]
		}
	}

	return result
}

// IFFTShift2D inverse of FFTShift2D
func IFFTShift2D(x [][]complex128) [][]complex128 {
	m, n := len(x), len(x[0])
	result := make([][]complex128, m)

	midM := (m + 1) / 2
	midN := (n + 1) / 2

	for i := 0; i < m; i++ {
		result[i] = make([]complex128, n)
		for j := 0; j < n; j++ {
			srcI := (i + midM) % m
			srcJ := (j + midN) % n
			result[i][j] = x[srcI][srcJ]
		}
	}

	return result
}

// ============================================================================
// CONVOLUTION VIA FFT
// ============================================================================

// Convolve2DFFT performs 2D convolution using FFT
func Convolve2DFFT(signal, kernel [][]float64) [][]float64 {
	m1, n1 := len(signal), len(signal[0])
	m2, n2 := len(kernel), len(kernel[0])

	// Output size
	mOut := m1 + m2 - 1
	nOut := n1 + n2 - 1

	// Pad to power of 2
	mPad := nextPowerOf2FFT(mOut)
	nPad := nextPowerOf2FFT(nOut)

	// Convert to complex and pad
	signalPad := make([][]complex128, mPad)
	kernelPad := make([][]complex128, mPad)

	for i := 0; i < mPad; i++ {
		signalPad[i] = make([]complex128, nPad)
		kernelPad[i] = make([]complex128, nPad)

		if i < m1 {
			for j := 0; j < n1; j++ {
				signalPad[i][j] = complex(signal[i][j], 0)
			}
		}

		if i < m2 {
			for j := 0; j < n2; j++ {
				kernelPad[i][j] = complex(kernel[i][j], 0)
			}
		}
	}

	// FFT both
	signalFFT := FFT2D(signalPad, nil)
	kernelFFT := FFT2D(kernelPad, nil)

	// Element-wise multiply
	product := make([][]complex128, mPad)
	for i := 0; i < mPad; i++ {
		product[i] = make([]complex128, nPad)
		for j := 0; j < nPad; j++ {
			product[i][j] = signalFFT[i][j] * kernelFFT[i][j]
		}
	}

	// Inverse FFT
	resultComplex := IFFT2DComplex(product, nil)

	// Extract real part
	result := make([][]float64, mOut)
	for i := 0; i < mOut; i++ {
		result[i] = make([]float64, nOut)
		for j := 0; j < nOut; j++ {
			result[i][j] = real(resultComplex[i][j])
		}
	}

	return result
}

// ============================================================================
// POWER SPECTRUM
// ============================================================================

// PowerSpectrum computes power spectrum (magnitude squared)
func PowerSpectrum(x []float64, config *FFTConfig) []float64 {
	spectrum := RFFT(x, config)
	n := len(spectrum)

	power := make([]float64, n)
	for i := 0; i < n; i++ {
		mag := cmplx.Abs(spectrum[i])
		power[i] = mag * mag
	}

	return power
}

// PowerSpectrum2D computes 2D power spectrum
func PowerSpectrum2D(x [][]float64, config *FFTConfig) [][]float64 {
	// Convert to complex
	m, n := len(x), len(x[0])
	cx := make([][]complex128, m)
	for i := 0; i < m; i++ {
		cx[i] = make([]complex128, n)
		for j := 0; j < n; j++ {
			cx[i][j] = complex(x[i][j], 0)
		}
	}

	// 2D FFT
	spectrum := FFT2D(cx, config)

	// Compute power
	power := make([][]float64, m)
	for i := 0; i < m; i++ {
		power[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			mag := cmplx.Abs(spectrum[i][j])
			power[i][j] = mag * mag
		}
	}

	return power
}

// PhaseSpectrum computes phase spectrum
func PhaseSpectrum(x []float64, config *FFTConfig) []float64 {
	spectrum := RFFT(x, config)
	n := len(spectrum)

	phase := make([]float64, n)
	for i := 0; i < n; i++ {
		phase[i] = cmplx.Phase(spectrum[i])
	}

	return phase
}

// ============================================================================
// MISSING CORE FFT IMPLEMENTATIONS (1D/2D, RFFT)
// ============================================================================

// nextPowerOf2FFT returns the next power-of-two >= n
func nextPowerOf2FFT(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// FFT computes the 1D Cooley–Tukey FFT of a complex vector.
// If config is provided, it may adjust normalization in future.
func FFT(x []complex128, _ *FFTConfig) []complex128 {
    n := len(x)
    if n == 0 {
        return nil
    }
    if n&(n-1) != 0 {
        // pad to power of two
        m := nextPowerOf2FFT(n)
        tmp := make([]complex128, m)
        copy(tmp, x)
        x = tmp
        n = m
    } else {
        x = append([]complex128(nil), x...)
    }
    // Keep Cooley–Tukey as the active path while Stockham completes validation.
    return cooleyFFT(x)
}

func cooleyFFT(x []complex128) []complex128 {
    n := len(x)
    // bit-reverse permutation
    j := 0
    for i := 1; i < n-1; i++ {
        bit := n >> 1
        for ; j&bit != 0; bit >>= 1 {
            j &^= bit
        }
        j |= bit
        if i < j {
            x[i], x[j] = x[j], x[i]
        }
    }
    // Cooley–Tukey, with optional cached stage wlen values
    var wlens []complex128
    useCache := defaultFFTCache() // no config passed into FFT; respect env default
    if useCache { wlens = getStageWlens(n) }
    stage := 0
    for length := 2; length <= n; length <<= 1 {
        var wlen complex128
        if useCache { wlen = wlens[stage] } else {
            ang := -2 * math.Pi / float64(length)
            wlen = complex(math.Cos(ang), math.Sin(ang))
        }
        half := length >> 1
        // Precompute wlen^2 and wlen^4 for unrolling
        wlen2 := wlen * wlen
        wlen4 := wlen2 * wlen2
        for i := 0; i < n; i += length {
            w := complex(1, 0)
            k := 0
            // Unroll by 4 when possible
            if half >= 4 {
                for ; k+3 < half; k += 4 {
                    // w0, w1, w2, w3
                    w0 := w
                    w1 := w0 * wlen
                    w2 := w1 * wlen
                    w3 := w2 * wlen
                    // indices
                    i0 := i + k
                    j0 := i0 + half
                    // butterfly 0
                    u0 := x[i0]
                    v0 := w0 * x[j0]
                    x[i0] = u0 + v0
                    x[j0] = u0 - v0
                    // butterfly 1
                    i1 := i + k + 1
                    j1 := i1 + half
                    u1 := x[i1]
                    v1 := w1 * x[j1]
                    x[i1] = u1 + v1
                    x[j1] = u1 - v1
                    // butterfly 2
                    i2 := i + k + 2
                    j2 := i2 + half
                    u2 := x[i2]
                    v2 := w2 * x[j2]
                    x[i2] = u2 + v2
                    x[j2] = u2 - v2
                    // butterfly 3
                    i3 := i + k + 3
                    j3 := i3 + half
                    u3 := x[i3]
                    v3 := w3 * x[j3]
                    x[i3] = u3 + v3
                    x[j3] = u3 - v3
                    // advance w by 4 steps
                    w *= wlen4
                }
            }
            // Unroll by 2 when possible
            if k+1 < half {
                for ; k+1 < half; k += 2 {
                    w0 := w
                    w1 := w0 * wlen
                    // 0
                    i0 := i + k
                    j0 := i0 + half
                    u0 := x[i0]
                    v0 := w0 * x[j0]
                    x[i0] = u0 + v0
                    x[j0] = u0 - v0
                    // 1
                    i1 := i + k + 1
                    j1 := i1 + half
                    u1 := x[i1]
                    v1 := w1 * x[j1]
                    x[i1] = u1 + v1
                    x[j1] = u1 - v1
                    // advance w by 2 steps
                    w *= wlen2
                }
            }
            // Remainder
            for ; k < half; k++ {
                u := x[i+k]
                v := w * x[i+k+half]
                x[i+k] = u + v
                x[i+k+half] = u - v
                w *= wlen
            }
        }
        stage++
    }
    return x
}

// IFFT computes the inverse FFT.
func IFFT(x []complex128, _ *FFTConfig) []complex128 {
	n := len(x)
	if n == 0 {
		return nil
	}
	conj := make([]complex128, n)
	for i := range x {
		conj[i] = complex(real(x[i]), -imag(x[i]))
	}
	y := FFT(conj, nil)
	out := make([]complex128, len(y))
	invN := 1.0 / float64(n)
	for i := range y {
		out[i] = complex(real(y[i])*invN, -imag(y[i])*invN)
	}
	return out
}

// RFFT computes the FFT of a real input, returning complex output.
func RFFT(x []float64, _ *FFTConfig) []complex128 {
	n := len(x)
	c := make([]complex128, n)
	for i := 0; i < n; i++ {
		c[i] = complex(x[i], 0)
	}
	return FFT(c, nil)
}

// FFT2D computes 2D FFT by applying 1D FFT across rows then columns.
func FFT2D(x [][]complex128, cfg *FFTConfig) [][]complex128 {
	m, n := len(x), len(x[0])
	// FFT rows
	rows := make([][]complex128, m)
	for i := 0; i < m; i++ {
		rows[i] = FFT(x[i], cfg)
	}
	// transpose
	cols := make([][]complex128, n)
	for j := 0; j < n; j++ {
		cols[j] = make([]complex128, m)
		for i := 0; i < m; i++ {
			cols[j][i] = rows[i][j]
		}
	}
	// FFT columns
	for j := 0; j < n; j++ {
		cols[j] = FFT(cols[j], cfg)
	}
	// transpose back
	out := make([][]complex128, m)
	for i := 0; i < m; i++ {
		out[i] = make([]complex128, n)
		for j := 0; j < n; j++ {
			out[i][j] = cols[j][i]
		}
	}
	return out
}
