package backend

// Auto helpers mirror legacy per-agent helpers while delegating to the provider.

// DotAuto selects the active backend (delegates to Provider.Dot).
func DotAuto(a, b []float64) float64 { return New().Dot(a, b) }

// CosineAuto selects the active backend (delegates to Provider.Cos).
func CosineAuto(a, b []float64) float64 { return New().Cos(a, b) }

// DotBatchAuto computes per-row dot products for flattened buffers.
// A and B contain m rows of length n each, laid out row-major.
func DotBatchAuto(n int, A, B []float64) []float64 {
	if n <= 0 || len(A) != len(B) || len(A)%n != 0 {
		panic("infrastructure/maths/backend.DotBatchAuto: invalid sizes")
	}
	m := len(A) / n
	out := make([]float64, m)
	p := New()
	base := 0
	for i := 0; i < m; i++ {
		out[i] = p.Dot(A[base:base+n], B[base:base+n])
		base += n
	}
	return out
}

// CosineBatchAuto computes per-row cosine similarity for flattened buffers.
func CosineBatchAuto(n int, A, B []float64) []float64 {
	if n <= 0 || len(A) != len(B) || len(A)%n != 0 {
		panic("infrastructure/maths/backend.CosineBatchAuto: invalid sizes")
	}
	m := len(A) / n
	out := make([]float64, m)
	p := New()
	base := 0
	for i := 0; i < m; i++ {
		out[i] = p.Cos(A[base:base+n], B[base:base+n])
		base += n
	}
	return out
}
