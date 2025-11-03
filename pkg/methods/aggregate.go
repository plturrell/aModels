package methods

import (
	"ai_benchmarks/internal/mathvec"
)

// CombinedScores computes weighted sum of available methods.
// weights keys: w_vec, w_lnn, w_sym, w_mcts, w_frac
func CombinedScores(prompt string, options []string, weights map[string]float64, seed int64, vecDim int, mctsRollouts int) []float64 {
	n := len(options)
	out := make([]float64, n)
	// Vector cosine
	if w := weights["w_vec"]; w != 0 {
		v := mathvec.NewVectorizer(vecDim)
		qp := v.Vec(prompt)
		for i, opt := range options {
			out[i] += w * mathvec.Cosine(qp, v.Vec(opt))
		}
	}
	// LNN
	if w := weights["w_lnn"]; w != 0 {
		s := ScoreOptionsLNN(prompt, options)
		for i := range out {
			out[i] += w * s[i]
		}
	}
	// Symbolic
	if w := weights["w_sym"]; w != 0 {
		s := ScoreOptionsSymbolic(prompt, options)
		for i := range out {
			out[i] += w * s[i]
		}
	}
	// Fractal
	if w := weights["w_frac"]; w != 0 {
		s := ScoreOptionsFractal(prompt, options)
		for i := range out {
			out[i] += w * s[i]
		}
	}
	// MCTS over cosine reward
	if w := weights["w_mcts"]; w != 0 {
		if mctsRollouts <= 0 {
			mctsRollouts = 64
		}
		v := mathvec.NewVectorizer(vecDim)
		qp := v.Vec(prompt)
		vals := ScoreOptionsMCTS(n, mctsRollouts, seed, func(i int) float64 { return mathvec.Cosine(qp, v.Vec(options[i])) })
		for i := range out {
			out[i] += w * vals[i]
		}
	}
	return out
}

// CombinedYesNo scores ["yes","no"] options.
func CombinedYesNo(prompt string, weights map[string]float64, seed int64, vecDim int, mctsRollouts int) (float64, float64) {
	opts := []string{"yes", "no"}
	s := CombinedScores(prompt, opts, weights, seed, vecDim, mctsRollouts)
	return s[0], s[1]
}
