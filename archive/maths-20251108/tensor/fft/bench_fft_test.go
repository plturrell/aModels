package fft

import (
    "math/cmplx"
    "testing"
)

func makeSig(n int) []complex128 {
    s := make([]complex128, n)
    for i := 0; i < n; i++ {
        s[i] = complex(float64((i*13)%17)-8, float64((i*7)%11)-5)
    }
    return s
}

func BenchmarkFFT_1D_1024(b *testing.B) {
    x := makeSig(1024)
    b.ReportAllocs()
    b.ResetTimer()
    var y []complex128
    for i := 0; i < b.N; i++ { y = FFT(x, nil) }
    _ = y
}

func BenchmarkFFT_1D_8192(b *testing.B) {
    x := makeSig(8192)
    b.ReportAllocs()
    b.ResetTimer()
    var y []complex128
    for i := 0; i < b.N; i++ { y = FFT(x, nil) }
    _ = y
}

func BenchmarkIFFT_1D_8192(b *testing.B) {
    x := makeSig(8192)
    y := FFT(x, nil)
    b.ReportAllocs()
    b.ResetTimer()
    var z []complex128
    for i := 0; i < b.N; i++ { z = IFFT(y, nil) }
    _ = z
}

func BenchmarkFFT2D_256x256(b *testing.B) {
    m, n := 256, 256
    X := make([][]complex128, m)
    for i := 0; i < m; i++ { X[i] = make([]complex128, n); for j := 0; j < n; j++ { X[i][j] = complex(float64((i*j)%23)-11, float64((i+2*j)%17)-8) } }
    cfg := DefaultFFTConfig()
    b.ReportAllocs()
    b.ResetTimer()
    var Y [][]complex128
    for i := 0; i < b.N; i++ { Y = FFT2D(X, cfg) }
    _ = Y
}

func BenchmarkFFT_Roundtrip_4096(b *testing.B) {
    x := makeSig(4096)
    b.ReportAllocs()
    b.ResetTimer()
    var rt []complex128
    for i := 0; i < b.N; i++ { rt = IFFT(FFT(x, nil), nil) }
    _ = cmplx.Abs(rt[0])
}

