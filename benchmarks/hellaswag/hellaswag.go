package hellaswag

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
	"ai_benchmarks/internal/rng"
	"ai_benchmarks/internal/textnorm"
)

type Example struct {
	Context     string   `json:"context"`
	Endings     []string `json:"endings"`
	Label       any      `json:"label"` // accept number or string
	IndActivity string   `json:"ind_activity,omitempty"`
	IndWikiHow  string   `json:"ind_wikihow,omitempty"`
	SplitType   string   `json:"split_type,omitempty"`
	SourceID    string   `json:"source_id,omitempty"`
}

func parseLabel(v any) (int, error) {
	switch t := v.(type) {
	case float64:
		return int(t), nil
	case int:
		return t, nil
	case string:
		s := strings.TrimSpace(t)
		if s == "" {
			return -1, errors.New("empty label")
		}
		// try parse digits
		n := 0
		for _, r := range s {
			if r < '0' || r > '9' {
				return -1, fmt.Errorf("bad label: %q", s)
			}
			n = n*10 + int(r-'0')
		}
		return n, nil
	default:
		return -1, fmt.Errorf("unsupported label type %T", v)
	}
}

// baseline selects ending by simple overlap heuristic; ties broken randomly.
func baselinePick(r *strings.Replacer, R *rngSource, ctx string, endings []string) int {
	ctxTokens := bagOfWords(r, ctx)
	best := -1
	bestScore := -1
	for i, e := range endings {
		score := overlap(ctxTokens, bagOfWords(r, e))
		if score > bestScore {
			best, bestScore = i, score
		} else if score == bestScore && R.rand.Intn(2) == 0 {
			best = i
		}
	}
	if best < 0 {
		best = 0
	}
	return best
}

type rngSource struct{ rand *rng.RandFacade }

// Provide minimal facade to allow deterministic tests without importing math/rand directly here.
// Implemented in internal/rng (thin wrapper).

type runner struct{}

func (runner) ID() string            { return "hellaswag" }
func (runner) Description() string   { return "HellaSwag: commonsense completion; metric=accuracy" }
func (runner) DefaultMetric() string { return "accuracy" }

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	// Optional: load/fit a pure-maths log-odds model over option tokens
	var lom *learn.LogOddsModel
	if opts.ModelIn != "" {
		m, err := learn.LoadLogOdds(opts.ModelIn)
		if err != nil {
			return nil, fmt.Errorf("load model: %w", err)
		}
		lom = m
	}
	if opts.FitPath != "" {
		m, err := fitHellaSwag(opts.FitPath)
		if err != nil {
			return nil, fmt.Errorf("fit: %w", err)
		}
		lom = m
		if opts.ModelOut != "" {
			_ = lom.Save(opts.ModelOut)
		}
	}
	if fi, err := os.Stat(opts.DataPath); err != nil || fi.IsDir() {
		return nil, registry.Errf("--data=<path to JSONL> (HellaSwag)", "data must be a JSONL file: %v", err)
	}
	f, err := os.Open(opts.DataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))

	// prepare normalization replacer and rng
	repl := strings.NewReplacer(
		",", " ", ".", " ", ";", " ", ":", " ", "?", " ", "!", " ", "\n", " ", "\t", " ",
		"(", " ", ")", " ", "[", " ", "]", " ", "{", " ", "}", " ",
	)
	rr := &rngSource{rand: rng.NewFacade(opts.Seed)}

	total := 0
	correct := 0
	started := time.Now().Unix()
	for {
		if opts.Limit > 0 && total >= opts.Limit {
			break
		}
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, fmt.Errorf("decode: %w", err)
		}
		if len(ex.Endings) == 0 {
			continue
		}
		gold, err := parseLabel(ex.Label)
		if err != nil {
			continue
		}
		var pred int
		if lom != nil || opts.Model == "hybrid" {
			prompt := repl.Replace(ex.Context)
			vecDim := 4096
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
			scores := methods.CombinedScores(prompt, ex.Endings, weights, opts.Seed, vecDim, mctsRollouts)
			// add log-odds if available
			if lom != nil {
				for i, opt := range ex.Endings {
					s := 0.0
					for _, t := range textnorm.Tokens(opt) {
						if w, ok := lom.Weights[t]; ok {
							s += w
						}
					}
					scores[i] += opts.Params["w_lo"] * s
				}
			}
			// choose best index
			best, bestS := 0, -1e18
			for i, s := range scores {
				if s > bestS {
					best, bestS = i, s
				}
			}
			pred = best
		} else if opts.Model == "maths" {
			prompt := repl.Replace(ex.Context)
			pred = models.MathsMCQ(prompt, ex.Endings, 4096)
		} else {
			pred = baselinePick(repl, rr, ex.Context, ex.Endings)
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
		Task:       "hellaswag",
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

// Helper functions
func bagOfWords(r *strings.Replacer, s string) map[string]int {
	s = strings.ToLower(r.Replace(s))
	m := map[string]int{}
	for _, tok := range strings.Fields(s) {
		m[tok]++
	}
	return m
}

func overlap(a, b map[string]int) int {
	c := 0
	for k, va := range a {
		if vb, ok := b[k]; ok {
			if va < vb {
				c += va
			} else {
				c += vb
			}
		}
	}
	return c
}

// ---- training helpers ----

func fitHellaSwag(path string) (*learn.LogOddsModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))
	cpos := map[string]int{}
	cneg := map[string]int{}
	npos := 0
	nneg := 0
	for {
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, err
		}
		if len(ex.Endings) == 0 {
			continue
		}
		gold, err := parseLabel(ex.Label)
		if err != nil {
			continue
		}
		for i, e := range ex.Endings {
			toks := textnorm.Tokens(e)
			if i == gold {
				for _, t := range toks {
					cpos[t]++
					npos++
				}
			} else {
				for _, t := range toks {
					cneg[t]++
					nneg++
				}
			}
		}
	}
	// Fit
	Vset := map[string]struct{}{}
	for t := range cpos {
		Vset[t] = struct{}{}
	}
	for t := range cneg {
		Vset[t] = struct{}{}
	}
	alpha := 0.5
	V := float64(len(Vset))
	pden := float64(npos) + alpha*V
	nden := float64(nneg) + alpha*V
	w := map[string]float64{}
	for t := range Vset {
		py := (float64(cpos[t]) + alpha) / pden
		pn := (float64(cneg[t]) + alpha) / nden
		w[t] = mathLog(py) - mathLog(pn)
	}
	return &learn.LogOddsModel{Weights: w}, nil
}

func mathLog(x float64) float64 { return math.Log(x) }

func scoreOptionsLogOdds(m *learn.LogOddsModel, options []string) int {
	best := 0
	bestScore := -1e18
	for i, opt := range options {
		s := 0.0
		for _, t := range textnorm.Tokens(opt) {
			if w, ok := m.Weights[t]; ok {
				s += w
			}
		}
		if s > bestScore {
			best, bestScore = i, s
		}
	}
	return best
}

// hybrid scoring factored into methods.CombinedScores above
