package fft

import (
    "math/cmplx"
    "math/rand"
    "os"
    "testing"
)

func nearlyEq(a, b complex128, tol float64) bool {
    da := cmplx.Abs(a - b)
    denom := 1.0
    if cmplx.Abs(b) > 1 { denom = cmplx.Abs(b) }
    return da/denom < tol
}

func TestFFT_IFFT_Roundtrip_Pow2(t *testing.T) {
    sizes := []int{8, 16, 64}
    for _, n := range sizes {
        in := make([]complex128, n)
        for i := 0; i < n; i++ { in[i] = complex(float64(i%7)-3, float64((i*3)%5)-2) }
        y := FFT(in, nil)
        z := IFFT(y, nil)
        if len(z) != n { t.Fatalf("len mismatch: %d vs %d", len(z), n) }
        for i := 0; i < n; i++ {
            if !nearlyEq(z[i], in[i], 1e-9) {
                t.Fatalf("roundtrip mismatch at %d: got=%v want=%v", i, z[i], in[i])
            }
        }
    }
}

func TestFFT_Stockham_Matches_Default(t *testing.T) {
    defer os.Unsetenv("MATHS_FFT_STOCKHAM")
    sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024}
    rng := rand.New(rand.NewSource(42))
    for _, n := range sizes {
        // random complex input
        in := make([]complex128, n)
        for i := 0; i < n; i++ {
            in[i] = complex(rng.NormFloat64(), rng.NormFloat64())
        }
        // default (Cooley)
        os.Setenv("MATHS_FFT_STOCKHAM", "0")
        y0 := FFT(in, nil)
        // stockham
        os.Setenv("MATHS_FFT_STOCKHAM", "1")
        y1 := FFT(in, nil)
        if len(y0) != len(y1) { t.Fatalf("n=%d len mismatch: %d vs %d", n, len(y0), len(y1)) }
        for i := range y0 {
            if !nearlyEq(y0[i], y1[i], 1e-9) {
                t.Fatalf("n=%d idx=%d stockham mismatch: %v vs %v", n, i, y0[i], y1[i])
            }
        }
    }
}
