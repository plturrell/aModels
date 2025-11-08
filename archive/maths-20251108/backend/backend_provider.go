package backend

import (
	"math"
	"math/rand"
	"os"
)

// Provider defines the math operations surface.
type Provider interface {
	Dot(a, b []float64) float64
	Cos(a, b []float64) float64
	MatMul(m, n, k int, A, B []float64) []float64
	Project(m, n, r int, A, P []float64) []float64

	CosineTopK(n int, A []float64, q []float64, topK int) ([]int, []float64)
	CosineMultiTopK(n int, A []float64, Q []float64, topK int) ([][]int, [][]float64)
	CosineTopKInt8(n int, A8 []int8, q []float64, topK int) ([]int, []float64)
}

var registered Provider

// Register installs an external provider (e.g., Fortran-backed) for selection via New().
func Register(p Provider) { registered = p }

// New selects a Provider based on env (INFRA_MATHS_BACKEND):
//   - "fortran" returns the registered provider if present, else pure Go
//   - default or "go" returns pure Go
func New() Provider {
	switch os.Getenv("INFRA_MATHS_BACKEND") {
	case "fortran", "FORTRAN", "Fortran":
		if registered != nil {
			return registered
		}
		return goProvider{}
	case "go", "GO", "Go":
		return goProvider{}
	default:
		if registered != nil {
			return registered
		}
		return goProvider{}
	}
}

// --- Convenience functions (package-level) ---

// BuildRandomProjection builds a dense Gaussian random projection matrix P (n x r) with seed.
func BuildRandomProjection(n, r int, seed int64) []float64 {
	if n <= 0 || r <= 0 {
		return nil
	}
	rnd := rand.New(rand.NewSource(seed))
	P := make([]float64, n*r)
	for i := 0; i < n*r; i++ {
		P[i] = rnd.NormFloat64() / float64(n)
	}
	return P
}

// Project multiplies A (m x n) by P (n x r) to produce Y (m x r) using the selected provider.
func Project(m, n, r int, A, P []float64) []float64 { return New().Project(m, n, r, A, P) }

// CosineTopK computes per-row cosine similarity to query and returns topK indices and scores.
func CosineTopK(n int, A []float64, q []float64, topK int) ([]int, []float64) {
	return New().CosineTopK(n, A, q, topK)
}

// CosineMultiTopK computes per-query Top-K cosine.
func CosineMultiTopK(n int, A []float64, Q []float64, topK int) ([][]int, [][]float64) {
	return New().CosineMultiTopK(n, A, Q, topK)
}

// CosineTopKInt8 computes Top-K for int8 docs vs float64 query.
func CosineTopKInt8(n int, A8 []int8, q []float64, topK int) ([]int, []float64) {
	return New().CosineTopKInt8(n, A8, q, topK)
}

// --- Pure-Go provider implementation ---

type goProvider struct{}

func (goProvider) Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("infrastructure/maths/backend.Dot: length mismatch")
	}
	acc := 0.0
	for i := range a {
		acc += a[i] * b[i]
	}
	return acc
}

func l2(a []float64) float64 {
	acc := 0.0
	for _, v := range a {
		acc += v * v
	}
	return math.Sqrt(acc)
}

func (p goProvider) Cos(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("infrastructure/maths/backend.Cos: length mismatch")
	}
	da := l2(a)
	db := l2(b)
	if da == 0 || db == 0 {
		return 0
	}
	return p.Dot(a, b) / (da * db)
}

func (goProvider) MatMul(m, n, k int, A, B []float64) []float64 {
	if m < 0 || n < 0 || k < 0 {
		panic("infrastructure/maths/backend.MatMul: negative dim")
	}
	if len(A) != m*k || len(B) != k*n {
		panic("infrastructure/maths/backend.MatMul: buffer mismatch")
	}
	C := make([]float64, m*n)
	if m == 0 || n == 0 || k == 0 {
		return C
	}
	for i := 0; i < m; i++ {
		arow := i * k
		for p := 0; p < n; p++ {
			sum := 0.0
			for j := 0; j < k; j++ {
				sum += A[arow+j] * B[j*n+p]
			}
			C[i*n+p] = sum
		}
	}
	return C
}

func (goProvider) Project(m, n, r int, A, P []float64) []float64 {
	if m <= 0 || n <= 0 || r <= 0 {
		return nil
	}
	if len(A) != m*n || len(P) != n*r {
		return nil
	}
	Y := make([]float64, m*r)
	for i := 0; i < m; i++ {
		arow := i * n
		for k := 0; k < r; k++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				sum += A[arow+j] * P[j*r+k]
			}
			Y[i*r+k] = sum
		}
	}
	return Y
}

func (p goProvider) CosineTopK(n int, A []float64, q []float64, topK int) ([]int, []float64) {
	if n <= 0 || len(A)%n != 0 || len(q) != n || topK <= 0 {
		return nil, nil
	}
	mrows := len(A) / n
	qn := l2(q)
	if qn == 0 {
		return make([]int, topK), make([]float64, topK)
	}
	idx := make([]int, topK)
	sc := make([]float64, topK)
	for i := 0; i < topK; i++ {
		idx[i] = -1
		sc[i] = -1e300
	}
	heapBuild(sc, idx, topK)
	for i := 0; i < mrows; i++ {
		row := A[i*n : i*n+n]
		rn := l2(row)
		if rn == 0 {
			continue
		}
		s := p.Dot(row, q) / (rn * qn)
		if s > sc[0] {
			sc[0] = s
			idx[0] = i
			pos := 0
			heapSiftDown(sc, idx, topK, &pos)
		}
	}
	heapSortDesc(sc, idx, topK)
	return idx, sc
}

func (p goProvider) CosineMultiTopK(n int, A []float64, Q []float64, topK int) ([][]int, [][]float64) {
	if n <= 0 || len(A)%n != 0 || len(Q)%n != 0 || topK <= 0 {
		return nil, nil
	}
	mrows := len(A) / n
	qrows := len(Q) / n
	outI := make([][]int, qrows)
	outS := make([][]float64, qrows)
	an := make([]float64, mrows)
	for i := 0; i < mrows; i++ {
		an[i] = l2(A[i*n : i*n+n])
	}
	for j := 0; j < qrows; j++ {
		qv := Q[j*n : j*n+n]
		qn := l2(qv)
		idx := make([]int, topK)
		sc := make([]float64, topK)
		for k := 0; k < topK; k++ {
			idx[k] = -1
			sc[k] = -1e300
		}
		heapBuild(sc, idx, topK)
		if qn == 0 {
			outI[j], outS[j] = idx, sc
			continue
		}
		for i := 0; i < mrows; i++ {
			if an[i] == 0 {
				continue
			}
			s := p.Dot(A[i*n:i*n+n], qv) / (an[i] * qn)
			if s > sc[0] {
				sc[0] = s
				idx[0] = i
				pos := 0
				heapSiftDown(sc, idx, topK, &pos)
			}
		}
		heapSortDesc(sc, idx, topK)
		outI[j], outS[j] = idx, sc
	}
	return outI, outS
}

func (goProvider) CosineTopKInt8(n int, A8 []int8, q []float64, topK int) ([]int, []float64) {
	if n <= 0 || len(A8)%n != 0 || len(q) != n || topK <= 0 {
		return nil, nil
	}
	mrows := len(A8) / n
	qn := l2(q)
	if qn == 0 {
		return make([]int, topK), make([]float64, topK)
	}
	idx := make([]int, topK)
	sc := make([]float64, topK)
	for i := 0; i < topK; i++ {
		idx[i] = -1
		sc[i] = -1e300
	}
	heapBuild(sc, idx, topK)
	for i := 0; i < mrows; i++ {
		base := i * n
		var dot, an2 float64
		for j := 0; j < n; j++ {
			v := float64(A8[base+j])
			dot += v * q[j]
			an2 += v * v
		}
		if an2 == 0 {
			continue
		}
		s := dot / (math.Sqrt(an2) * qn)
		if s > sc[0] {
			sc[0] = s
			idx[0] = i
			pos := 0
			heapSiftDown(sc, idx, topK, &pos)
		}
	}
	heapSortDesc(sc, idx, topK)
	return idx, sc
}

// --- minimal heap helpers (0-based indexing) ---

func heapSiftDown(scores []float64, idx []int, heapSize int, pos *int) {
	i := *pos
	for {
		left := 2*i + 1
		right := left + 1
		smallest := i
		if left < heapSize && scores[left] < scores[smallest] {
			smallest = left
		}
		if right < heapSize && scores[right] < scores[smallest] {
			smallest = right
		}
		if smallest == i {
			break
		}
		scores[i], scores[smallest] = scores[smallest], scores[i]
		idx[i], idx[smallest] = idx[smallest], idx[i]
		i = smallest
	}
	*pos = i
}

func heapBuild(scores []float64, idx []int, heapSize int) {
	if heapSize <= 1 {
		return
	}
	for i := heapSize/2 - 1; i >= 0; i-- {
		p := i
		heapSiftDown(scores, idx, heapSize, &p)
	}
}

func heapSortDesc(scores []float64, idx []int, heapSize int) {
	for i := heapSize - 1; i > 0; i-- {
		scores[0], scores[i] = scores[i], scores[0]
		idx[0], idx[i] = idx[i], idx[0]
		p := 0
		heapSiftDown(scores, idx, i, &p)
	}
	for l, r := 0, heapSize-1; l < r; l, r = l+1, r-1 {
		scores[l], scores[r] = scores[r], scores[l]
		idx[l], idx[r] = idx[r], idx[l]
	}
}
