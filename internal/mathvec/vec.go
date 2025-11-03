package mathvec

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// Vectorizer maps text to a fixed-length dense vector with simple hashing.
type Vectorizer struct {
	Dim        int
	UseBigrams bool
	Sublinear  bool // use log1p(tf)
}

func NewVectorizer(dim int) *Vectorizer {
	return &Vectorizer{Dim: dim, UseBigrams: true, Sublinear: true}
}

func (v *Vectorizer) clean(s string) string {
	s = strings.ToLower(s)
	s = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, s)
	return s
}

func hashToken(t string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(t))
	return h.Sum64()
}

// Vec returns a dense TF vector of length Dim.
func (v *Vectorizer) Vec(text string) []float64 {
	if v.Dim <= 0 {
		v.Dim = 2048
	}
	x := make([]float64, v.Dim)
	text = v.clean(text)
	toks := strings.Fields(text)
	for i, tok := range toks {
		idx := int(hashToken(tok) % uint64(v.Dim))
		x[idx] += 1
		if v.UseBigrams && i+1 < len(toks) {
			bi := tok + "_" + toks[i+1]
			bidx := int(hashToken(bi) % uint64(v.Dim))
			x[bidx] += 1
		}
	}
	if v.Sublinear {
		for i := range x {
			if x[i] > 0 {
				x[i] = 1 + math.Log(x[i])
			}
		}
	}
	// L2 normalize
	var norm float64
	for _, v := range x {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		inv := 1.0 / norm
		for i := range x {
			x[i] *= inv
		}
	}
	return x
}

// Cosine returns cosine similarity using pure math operations.
func Cosine(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		ai, bi := a[i], b[i]
		dot += ai * bi
		na += ai * ai
		nb += bi * bi
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
