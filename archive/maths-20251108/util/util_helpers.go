package util

import (
	"math"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/backend"
)

// ----- Scalar helpers -----

func Add(a, b float64) float64      { return a + b }
func Subtract(a, b float64) float64 { return a - b }
func Multiply(a, b float64) float64 { return a * b }

// Divide returns a / b. If b == 0, returns +Inf or -Inf depending on sign of a.
func Divide(a, b float64) float64 {
	if b == 0 {
		if a >= 0 {
			return math.Inf(1)
		}
		return math.Inf(-1)
	}
	return a / b
}

func Modulo(a, b float64) float64 { return math.Mod(a, b) }
func Abs(a float64) float64       { return math.Abs(a) }
func Equal(a, b float64) bool     { return a == b }
func Greater(a, b float64) bool   { return a > b }
func Less(a, b float64) bool      { return a < b }
func Round(a float64) float64     { return math.Round(a) }
func Floor(a float64) float64     { return math.Floor(a) }
func Ceil(a float64) float64      { return math.Ceil(a) }

// Sum returns the sum of all values.
func Sum(values []float64) float64 {
	acc := 0.0
	for _, v := range values {
		acc += v
	}
	return acc
}

// Min returns the minimum value (0 for empty slice).
func Min(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] < m {
			m = values[i]
		}
	}
	return m
}

// Max returns the maximum value (0 for empty slice).
func Max(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > m {
			m = values[i]
		}
	}
	return m
}

// Mean returns the arithmetic mean (0 for empty slice).
func Mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	return Sum(values) / float64(len(values))
}

// NearlyEqual checks approximate equality within epsilon (absolute).
func NearlyEqual(a, b, eps float64) bool {
	if eps <= 0 {
		eps = 1e-9
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}

// Sqrt exposes math.Sqrt for convenience.
func Sqrt(v float64) float64 { return math.Sqrt(v) }

// ----- Vector helpers (pure-Go) -----

// L2Norm returns sqrt(sum(x_i^2)).
func L2Norm(a []float64) float64 {
	acc := 0.0
	for _, v := range a {
		acc += v * v
	}
	return math.Sqrt(acc)
}

// VecAdd returns element-wise a + b.
func VecAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("infrastructure/maths/util.VecAdd: length mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// ScalarMul returns scalar * a (element-wise).
func ScalarMul(a []float64, scalar float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * scalar
	}
	return out
}

// Clamp returns x limited to [minV, maxV].
func Clamp(x, minV, maxV float64) float64 {
	if x < minV {
		return minV
	}
	if x > maxV {
		return maxV
	}
	return x
}

// ----- Provider convenience wrappers -----

// Dot computes the dot product using the selected provider.
func Dot(a, b []float64) float64 { return backend.New().Dot(a, b) }

// Cos computes the cosine similarity using the selected provider.
func Cos(a, b []float64) float64 { return backend.New().Cos(a, b) }

// MatMul performs matrix multiply using the selected provider.
func MatMul(m, n, k int, A, B []float64) []float64 { return backend.New().MatMul(m, n, k, A, B) }
