package ai

import (
	"math"
	"math/rand"
	"sort"
)

type SamplingConfig struct {
 Temperature float64
 TopK        int
 TopP        float64
}

const (
 DefaultTopP = 0.9
 DefaultTopK = 50
)

func sampleLogits(logits []float64, cfg SamplingConfig, rng *rand.Rand) int {
    temperature := cfg.Temperature
    if temperature <= 0 {
        temperature = 1
    }

	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = v / temperature
	}

	probs := softmax(scaled)

	if cfg.TopK > 0 && cfg.TopK < len(probs) {
		topIndices := make([]int, len(probs))
		for i := range topIndices {
			topIndices[i] = i
		}
		sort.Slice(topIndices, func(i, j int) bool {
			return probs[topIndices[i]] > probs[topIndices[j]]
		})

		total := 0.0
		clipped := make([]float64, len(probs))
		for i := 0; i < cfg.TopK; i++ {
			idx := topIndices[i]
			clipped[idx] = probs[idx]
			total += probs[idx]
		}
		if total > 0 {
			for i := range clipped {
				clipped[i] /= total
			}
			probs = clipped
		}
	}

	if cfg.TopP > 0 && cfg.TopP < 1 {
		type pair struct {
			idx  int
			prob float64
		}
		items := make([]pair, len(probs))
		for i, p := range probs {
			items[i] = pair{idx: i, prob: p}
		}
		sort.Slice(items, func(i, j int) bool {
			return items[i].prob > items[j].prob
		})
		cumulative := 0.0
		cutoff := len(items)
		for i, pair := range items {
			cumulative += pair.prob
			if cumulative >= cfg.TopP {
				cutoff = i + 1
				break
			}
		}
		clipped := make([]float64, len(probs))
		total := 0.0
		for i := 0; i < cutoff; i++ {
			idx := items[i].idx
			clipped[idx] = items[i].prob
			total += items[i].prob
		}
		if total > 0 {
			for i := range clipped {
				clipped[i] /= total
			}
			probs = clipped
		}
	}

	cumulative := 0.0
	cutoff := rng.Float64()
	for i, p := range probs {
		cumulative += p
		if cutoff <= cumulative {
			return i
		}
	}

	maxIdx := 0
	maxVal := probs[0]
	for i := 1; i < len(probs); i++ {
		if probs[i] > maxVal {
			maxVal = probs[i]
			maxIdx = i
		}
	}
	return maxIdx
}

func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return []float64{}
	}
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	exps := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		exp := math.Exp(v - maxLogit)
		exps[i] = exp
		sum += exp
	}
	if sum == 0 {
		return exps
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}
