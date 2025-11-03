package learn

import (
	"bufio"
	"encoding/json"
	"errors"
	"math"
	"os"
)

// LogOddsModel stores token->weight where weight ~= log P(t|pos) - log P(t|neg)
type LogOddsModel struct {
	Weights map[string]float64 `json:"weights"`
}

type counts struct {
	pos  map[string]int
	neg  map[string]int
	npos int
	nneg int
}

func newCounts() *counts { return &counts{pos: map[string]int{}, neg: map[string]int{}} }

// Add updates counts for tokens, where y=true indicates positive class.
func (c *counts) Add(tokens []string, y bool) {
	if y {
		for _, t := range tokens {
			c.pos[t]++
			c.npos++
		}
	} else {
		for _, t := range tokens {
			c.neg[t]++
			c.nneg++
		}
	}
}

// Fit computes smoothed log-odds weights.
func (c *counts) Fit(alpha float64) *LogOddsModel {
	if alpha <= 0 {
		alpha = 0.5
	}
	Vset := map[string]struct{}{}
	for t := range c.pos {
		Vset[t] = struct{}{}
	}
	for t := range c.neg {
		Vset[t] = struct{}{}
	}
	V := float64(len(Vset))
	w := make(map[string]float64, len(Vset))
	pden := float64(c.npos) + alpha*V
	nden := float64(c.nneg) + alpha*V
	for t := range Vset {
		py := (float64(c.pos[t]) + alpha) / pden
		pn := (float64(c.neg[t]) + alpha) / nden
		w[t] = safeLog(py) - safeLog(pn)
	}
	return &LogOddsModel{Weights: w}
}

func safeLog(x float64) float64 {
	if x <= 1e-18 {
		x = 1e-18
	}
	return math.Log(x)
}

// Save writes model to JSON file.
func (m *LogOddsModel) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(m)
}

// LoadLogOdds reads a model from JSON file.
func LoadLogOdds(path string) (*LogOddsModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))
	var m LogOddsModel
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}
	if m.Weights == nil {
		return nil, errors.New("invalid model: missing weights")
	}
	return &m, nil
}
