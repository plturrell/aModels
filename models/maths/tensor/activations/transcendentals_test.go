package activations

import (
    "math"
    "testing"
)

func TestTranscendentalsAccuracy_F64(t *testing.T) {
    SetSIMDAccuracyMode(SIMDFastApproximate)
    xs := []float64{-10, -5, -1, -0.5, 0, 0.3, 0.7, 1, 2, 5, 10}
    for _, x := range xs {
        got := TanhEstrinF64(x); want := math.Tanh(x)
        if math.Abs(got-want) > 2e-2 { t.Fatalf("tanh fast: x=%v got=%v want=%v", x, got, want) }
        if x > -0.99 {
            got = Log1pEstrinF64(x); want = math.Log1p(x)
            if math.Abs(got-want) > 5e-2 { t.Fatalf("log1p fast: x=%v got=%v want=%v", x, got, want) }
        }
        got = Expm1EstrinF64(x); want = math.Expm1(x)
        if math.Abs(got-want) > 5e-2 { t.Fatalf("expm1 fast: x=%v got=%v want=%v", x, got, want) }
    }
    SetSIMDAccuracyMode(SIMDStrictAccuracy)
    for _, x := range xs {
        got := TanhEstrinF64(x); want := math.Tanh(x)
        if math.Abs(got-want) > 1e-5 { t.Fatalf("tanh strict: x=%v got=%v want=%v", x, got, want) }
    }
}

func TestTranscendentalsAccuracy_F32(t *testing.T) {
    xs := []float32{-10, -1, -0.5, 0, 0.2, 1, 3, 10}
    for _, x := range xs {
        got := TanhEstrinF32(x)
        want := float32(math.Tanh(float64(x)))
        if math.Abs(float64(got-want)) > 3e-2 { t.Fatalf("tanh f32: x=%v got=%v want=%v", x, got, want) }
    }
}

