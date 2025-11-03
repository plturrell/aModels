package methods

import (
	"math"
	"math/rand"
	"time"
)

// ARC transform codes and functions
type ARCTransform func([][]int) [][]int

type ArcLib struct {
	Codes []string
	Funcs map[string]ARCTransform
}

func DefaultArcLib() ArcLib {
	// include rotations, flips, small shifts, tiling halves, and stripe fills
	codes := []string{"I", "R90", "R180", "R270", "FH", "FV",
		"SX1", "SX-1", "SY1", "SY-1", "SX2", "SX-2", "SY2", "SY-2",
		"TH", "TV", "StripeH", "StripeV",
	}
	funcs := map[string]ARCTransform{
		"I":       func(x [][]int) [][]int { return x },
		"R90":     rot90,
		"R180":    func(x [][]int) [][]int { return rot90(rot90(x)) },
		"R270":    func(x [][]int) [][]int { return rot90(rot90(rot90(x))) },
		"FH":      flipH,
		"FV":      flipV,
		"SX1":     func(x [][]int) [][]int { return shift(x, 1, 0) },
		"SX-1":    func(x [][]int) [][]int { return shift(x, -1, 0) },
		"SY1":     func(x [][]int) [][]int { return shift(x, 0, 1) },
		"SY-1":    func(x [][]int) [][]int { return shift(x, 0, -1) },
		"SX2":     func(x [][]int) [][]int { return shift(x, 2, 0) },
		"SX-2":    func(x [][]int) [][]int { return shift(x, -2, 0) },
		"SY2":     func(x [][]int) [][]int { return shift(x, 0, 2) },
		"SY-2":    func(x [][]int) [][]int { return shift(x, 0, -2) },
		"TH":      tileHalfH,
		"TV":      tileHalfV,
		"StripeH": stripeH,
		"StripeV": stripeV,
	}
	return ArcLib{Codes: codes, Funcs: funcs}
}

// BuildArcLibFromMask constructs a subset library using a bitmask of groups.
// Mask bits:
// 1=ID, 2=ROT, 4=FLIP, 8=SHIFT, 16=TILE, 32=STRIPE
func BuildArcLibFromMask(mask int) ArcLib {
	if mask == 0 {
		return DefaultArcLib()
	}
	all := DefaultArcLib()
	pick := map[string]bool{}
	// Always include ID if bit set
	if mask&1 != 0 {
		pick["I"] = true
	}
	if mask&2 != 0 {
		pick["R90"], pick["R180"], pick["R270"] = true, true, true
	}
	if mask&4 != 0 {
		pick["FH"], pick["FV"] = true, true
	}
	if mask&8 != 0 {
		for _, c := range []string{"SX1", "SX-1", "SY1", "SY-1", "SX2", "SX-2", "SY2", "SY-2"} {
			pick[c] = true
		}
	}
	if mask&16 != 0 {
		pick["TH"], pick["TV"] = true, true
	}
	if mask&32 != 0 {
		pick["StripeH"], pick["StripeV"] = true, true
	}
	codes := []string{}
	funcs := map[string]ARCTransform{}
	for _, c := range all.Codes {
		if pick[c] {
			codes = append(codes, c)
			funcs[c] = all.Funcs[c]
		}
	}
	if len(codes) == 0 {
		return all
	}
	return ArcLib{Codes: codes, Funcs: funcs}
}

// rot/flip helpers (duplicate minimal versions for method isolation)
func rot90(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, w)
	for i := 0; i < w; i++ {
		out[i] = make([]int, h)
	}
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			out[j][h-1-i] = g[i][j]
		}
	}
	return out
}
func flipH(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
		for j := 0; j < w; j++ {
			out[i][w-1-j] = g[i][j]
		}
	}
	return out
}
func flipV(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
	}
	for i := 0; i < h; i++ {
		copy(out[h-1-i], g[i])
	}
	return out
}

func shift(g [][]int, dx, dy int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
	}
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			ni := i - dy
			nj := j - dx
			if ni >= 0 && ni < h && nj >= 0 && nj < w {
				out[i][j] = g[ni][nj]
			} else {
				out[i][j] = 0
			}
		}
	}
	return out
}

func tileHalfH(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
		half := (w + 1) / 2
		for j := 0; j < w; j++ {
			out[i][j] = g[i][j%half]
		}
	}
	return out
}

func tileHalfV(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	half := (h + 1) / 2
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
		src := g[i%half]
		copy(out[i], src)
	}
	return out
}

func stripeH(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
		// majority non-zero color in row
		freq := map[int]int{}
		for j := 0; j < w; j++ {
			c := g[i][j]
			if c != 0 {
				freq[c]++
			}
		}
		bestC, bestN := 0, -1
		for c, n := range freq {
			if n > bestN {
				bestN, bestC = n, c
			}
		}
		for j := 0; j < w; j++ {
			if bestN > 0 {
				out[i][j] = bestC
			} else {
				out[i][j] = g[i][j]
			}
		}
	}
	return out
}

func stripeV(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
	}
	for j := 0; j < w; j++ {
		freq := map[int]int{}
		for i := 0; i < h; i++ {
			c := g[i][j]
			if c != 0 {
				freq[c]++
			}
		}
		bestC, bestN := 0, -1
		for c, n := range freq {
			if n > bestN {
				bestN, bestC = n, c
			}
		}
		for i := 0; i < h; i++ {
			if bestN > 0 {
				out[i][j] = bestC
			} else {
				out[i][j] = g[i][j]
			}
		}
	}
	return out
}

// RewardFn evaluates a sequence over training pairs and returns a scalar reward.
type RewardFn func(seq []string) float64

// ArcSearchMCTS does a simple UCT search over transform sequences up to maxDepth.
func ArcSearchMCTS(maxDepth, rollouts int, seed int64, lib ArcLib, reward RewardFn, policy []float64) []string {
	if rollouts <= 0 {
		rollouts = 64
	}
	if maxDepth <= 0 {
		maxDepth = 3
	}
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	R := rand.New(rand.NewSource(seed))

	type node struct {
		seq      []string
		n        int
		q        float64
		kids     []*node
		expanded bool
	}
	root := &node{seq: nil}

	ucb := func(n *node, t int) float64 {
		if n.n == 0 {
			return 1e9
		}
		return (n.q / float64(n.n)) + math.Sqrt(2*math.Log(float64(t+1))/float64(n.n))
	}

	total := 0
	bestSeq := []string{}
	bestVal := -1e18
	for total < rollouts {
		// Selection
		path := []*node{root}
		cur := root
		depth := 0
		for cur.expanded && depth < maxDepth && len(cur.kids) > 0 {
			// pick child with max UCB
			var next *node
			best := -1e18
			for _, c := range cur.kids {
				u := ucb(c, total)
				if u > best {
					best, next = u, c
				}
			}
			cur = next
			path = append(path, cur)
			depth++
		}
		// Expansion
		if !cur.expanded && depth < maxDepth {
			for i, code := range lib.Codes {
				child := &node{seq: append(append([]string{}, cur.seq...), code)}
				if policy != nil && i < len(policy) {
					child.n = int(policy[i] * 10) // Scale policy to initialize visit counts
				}
				cur.kids = append(cur.kids, child)
			}
			cur.expanded = true
			// pick one child to simulate
			if len(cur.kids) > 0 {
				cur = cur.kids[R.Intn(len(cur.kids))]
				path = append(path, cur)
				depth++
			}
		}
		// Simulation: evaluate reward of cur.seq
		val := reward(cur.seq)
		if val > bestVal {
			bestVal, bestSeq = val, append([]string{}, cur.seq...)
		}
		// Backprop
		for _, n := range path {
			n.n++
			n.q += val
		}
		total++
	}
	return bestSeq
}
