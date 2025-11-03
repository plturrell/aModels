package boolq

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"ai_benchmarks/internal/learn"
	"ai_benchmarks/internal/methods"
	"ai_benchmarks/internal/models"
	"ai_benchmarks/internal/registry"
	"ai_benchmarks/internal/textnorm"
)

type Example struct {
	Question string `json:"question"`
	Passage  string `json:"passage,omitempty"`
	Answer   *bool  `json:"answer,omitempty"`
	Label    string `json:"label,omitempty"` // alternative: "yes"/"no" or "true"/"false"
}

func parseBool(e Example) (bool, error) {
	if e.Answer != nil {
		return *e.Answer, nil
	}
	if e.Label != "" {
		v := strings.ToLower(strings.TrimSpace(e.Label))
		switch v {
		case "1", "true", "yes", "y":
			return true, nil
		case "0", "false", "no", "n":
			return false, nil
		}
	}
	return false, errors.New("missing answer/label")
}

type baselineModel string

// predict applies a trivial heuristic for yes/no.
func (m baselineModel) predict(q, passage string) bool {
	s := strings.ToLower(q + " " + passage)
	yesScore := 0
	noScore := 0
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(q)), "is ") ||
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(q)), "are ") ||
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(q)), "does ") {
		yesScore++
	}
	if strings.Contains(s, " not ") || strings.Contains(s, "n't ") {
		noScore += 2
	}
	if strings.Contains(s, "always") || strings.Contains(s, "never") {
		noScore++
	}
	return yesScore >= noScore
}

type runner struct{}

func (runner) ID() string            { return "boolq" }
func (runner) Description() string   { return "BoolQ: yes/no questions; metric=accuracy" }
func (runner) DefaultMetric() string { return "accuracy" }

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	// Optional: load/fit a pure-maths log-odds model
	var lom *learn.LogOddsModel
	if opts.ModelIn != "" {
		m, err := learn.LoadLogOdds(opts.ModelIn)
		if err != nil {
			return nil, fmt.Errorf("load model: %w", err)
		}
		lom = m
	}
	if opts.FitPath != "" {
		alpha := 0.5
		if v, ok := opts.Params["alpha"]; ok {
			alpha = v
		}
		m, err := fitBoolQWithAlpha(opts.FitPath, alpha)
		if err != nil {
			return nil, fmt.Errorf("fit: %w", err)
		}
		lom = m
		if opts.ModelOut != "" {
			_ = lom.Save(opts.ModelOut)
		}
	}
	if fi, err := os.Stat(opts.DataPath); err != nil || fi.IsDir() {
		return nil, registry.Errf("--data=<path to JSONL> (BoolQ)", "data must be a JSONL file: %v", err)
	}
	f, err := os.Open(opts.DataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))
	total := 0
	correct := 0

	started := time.Now().Unix()
	model := baselineModel(opts.Model)
	for {
		if opts.Limit > 0 && total >= opts.Limit {
			break
		}
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if errors.Is(err, os.ErrClosed) {
				break
			}
			if err.Error() == "EOF" {
				break
			}
			if errors.Is(err, context.Canceled) {
				break
			}
			// Allow trailing whitespace/malformed at end
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, fmt.Errorf("decode: %w", err)
		}
		gold, err := parseBool(ex)
		if err != nil {
			continue
		}
		var pred bool
		if lom != nil || opts.Model == "hybrid" {
			prompt := models.JoinPrompt(ex.Question, ex.Passage)
			vecDim := 2048
			if v, ok := opts.Params["vec_dim"]; ok {
				vecDim = int(v)
			}
			mctsRollouts := 64
			if v, ok := opts.Params["mcts_rollouts"]; ok {
				mctsRollouts = int(v)
			}
			weights := map[string]float64{
				"w_vec":  opts.Params["w_vec"],
				"w_lnn":  opts.Params["w_lnn"],
				"w_sym":  opts.Params["w_sym"],
				"w_mcts": opts.Params["w_mcts"],
				"w_frac": opts.Params["w_frac"],
			}
			sy, sn := methods.CombinedYesNo(prompt, weights, opts.Seed, vecDim, mctsRollouts)
			sLog := 0.0
			if lom != nil {
				sLog = sumWeights(lom, textnorm.Tokens(prompt))
			}
			wLo := opts.Params["w_lo"]
			sYes := wLo*sLog + sy
			sNo := wLo*(-sLog) + sn
			pred = sYes >= sNo
		} else if opts.Model == "maths" {
			prompt := models.JoinPrompt(ex.Question, ex.Passage)
			pred = models.MathsYesNo(prompt, 2048)
		} else {
			pred = model.predict(ex.Question, ex.Passage)
		}
		if pred == gold {
			correct++
		}
		total++
	}
	finished := time.Now().Unix()
	if total == 0 {
		return nil, errors.New("no examples loaded (check data format)")
	}
	acc := float64(correct) / float64(total)
	sum := &registry.Summary{
		Task:       "boolq",
		Model:      opts.Model,
		Count:      total,
		Metrics:    map[string]float64{"accuracy": acc},
		StartedAt:  started,
		FinishedAt: finished,
		Details:    map[string]any{"correct": correct},
	}
	return sum, nil
}

func init() { registry.Register(runner{}) }

// ---- training helpers (pure maths log-odds) ----

func fitBoolQWithAlpha(path string, alpha float64) (*learn.LogOddsModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))
	c := newCounts()
	for {
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, err
		}
		y, err := parseBool(ex)
		if err != nil {
			continue
		}
		prompt := models.JoinPrompt(ex.Question, ex.Passage)
		toks := textnorm.Tokens(prompt)
		c.Add(toks, y)
	}
	m := c.Fit(alpha)
	return m, nil
}

type counts struct {
	pos, neg   map[string]int
	npos, nneg int
}

func newCounts() *counts { return &counts{pos: map[string]int{}, neg: map[string]int{}} }
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
func (c *counts) Fit(alpha float64) *learn.LogOddsModel {
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
	pden := float64(c.npos) + alpha*V
	nden := float64(c.nneg) + alpha*V
	w := map[string]float64{}
	for t := range Vset {
		py := (float64(c.pos[t]) + alpha) / pden
		pn := (float64(c.neg[t]) + alpha) / nden
		w[t] = mathLog(py) - mathLog(pn)
	}
	return &learn.LogOddsModel{Weights: w}
}

func mathLog(x float64) float64 { return math.Log(x) }

func sumWeights(m *learn.LogOddsModel, toks []string) float64 {
	s := 0.0
	for _, t := range toks {
		if w, ok := m.Weights[t]; ok {
			s += w
		}
	}
	return s
}
