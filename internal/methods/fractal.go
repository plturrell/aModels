package methods

import (
	"hash/fnv"
	"math"
	"strings"
)

// ScoreOptionsFractal compares approximate fractal dimension (box-counting) between prompt and options.
// Closer dimension to the prompt yields higher score.
func ScoreOptionsFractal(prompt string, options []string) []float64 {
	dp := fractalDim(prompt)
	scores := make([]float64, len(options))
	for i, o := range options {
		do := fractalDim(o)
		// Higher score when |dp - do| is small
		scores[i] = -math.Abs(dp - do)
	}
	return scores
}

func fractalDim(s string) float64 {
	// Map tokens into a 2D binary grid via hashing; compute occupied boxes at scales 1,2,4,8
	toks := strings.Fields(strings.ToLower(s))
	if len(toks) == 0 {
		return 0
	}
	sizes := []int{16, 32, 64}
	counts := make([]float64, len(sizes))
	for k, n := range sizes {
		grid := make([][]bool, n)
		for i := range grid {
			grid[i] = make([]bool, n)
		}
		for _, t := range toks {
			h := h2(t)
			x := int(h % uint64(n))
			y := int((h / uint64(n)) % uint64(n))
			grid[x][y] = true
		}
		c := 0
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if grid[i][j] {
					c++
				}
			}
		}
		counts[k] = float64(c)
	}
	// Fit slope of log N(s) vs log(1/s). Here s ~ 1/n, so log N vs log n
	xs := []float64{}
	ys := []float64{}
	for _, n := range sizes {
		xs = append(xs, math.Log(float64(n)))
	}
	for _, c := range counts {
		ys = append(ys, math.Log(c+1))
	}
	// simple least squares slope
	var sx, sy, sxx, sxy float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
		sxx += xs[i] * xs[i]
		sxy += xs[i] * ys[i]
	}
	n := float64(len(xs))
	denom := n*sxx - sx*sx
	if denom == 0 {
		return 0
	}
	slope := (n*sxy - sx*sy) / denom
	return slope
}

func h2(t string) uint64 { h := fnv.New64a(); _, _ = h.Write([]byte(t)); return h.Sum64() }
