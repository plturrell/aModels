package methods

import (
	"math"
	"math/rand"
	"time"
)

// Score function signature used in simulations
type scoreFn func(optIdx int) float64

// ScoreOptionsMCTS returns estimated values for each option by running simple UCB1 MCTS
// over arms (each option is an arm). It does not expand further states; it is a bandit MCTS.
func ScoreOptionsMCTS(nOptions int, rollouts int, seed int64, scorer scoreFn) []float64 {
	if rollouts <= 0 {
		rollouts = 64
	}
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	R := rand.New(rand.NewSource(seed))
	// Stats
	N := make([]int, nOptions)
	Q := make([]float64, nOptions)
	// Initialize by pulling each arm once
	for i := 0; i < nOptions; i++ {
		r := scorer(i)
		N[i] = 1
		Q[i] = r
	}
	// Rollouts with UCB1 selection
	for t := nOptions; t < rollouts; t++ {
		// select
		best := 0
		bestU := -1e18
		for i := 0; i < nOptions; i++ {
			u := Q[i]/float64(N[i]) + math.Sqrt(2*math.Log(float64(t+1))/float64(N[i]))
			if u > bestU {
				bestU, best = u, i
			}
		}
		// simulate
		r := scorer(best)
		// update
		N[best]++
		Q[best] += r
		_ = R // deterministic unless scorer is random; R kept for future extensions
	}
	// Return average rewards
	out := make([]float64, nOptions)
	for i := 0; i < nOptions; i++ {
		out[i] = Q[i] / float64(N[i])
	}
	return out
}
