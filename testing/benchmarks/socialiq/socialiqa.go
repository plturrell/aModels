package socialiq

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
	Question string `json:"question"`
	A        string `json:"answerA"`
	B        string `json:"answerB"`
	C        string `json:"answerC"`
	Label    string `json:"label"` // "A"/"B"/"C" or "0"/"1"/"2"
}

func parseLabel(s string) (int, error) {
	s = strings.TrimSpace(strings.ToUpper(s))
	switch s {
	case "A", "0":
		return 0, nil
	case "B", "1":
		return 1, nil
	case "C", "2":
		return 2, nil
	default:
		return -1, fmt.Errorf("bad label: %q", s)
	}
}

func bag(s string) map[string]int {
	s = strings.ToLower(s)
	repl := strings.NewReplacer(
		",", " ", ".", " ", ";", " ", ":", " ", "?", " ", "!", " ", "\n", " ", "\t", " ",
		"(", " ", ")", " ", "[", " ", "]", " ", "{", " ", "}", " ",
	)
	s = repl.Replace(s)
	m := map[string]int{}
	for _, w := range strings.Fields(s) {
		m[w]++
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

type runner struct{}

func (runner) ID() string            { return "socialiqa" }
func (runner) Description() string   { return "SocialIQA: social commonsense MCQ; metric=accuracy" }
func (runner) DefaultMetric() string { return "accuracy" }

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	if fi, err := os.Stat(opts.DataPath); err != nil || fi.IsDir() {
		return nil, registry.Errf("--data=<path to JSONL> (SocialIQA)", "data must be a JSONL file: %v", err)
	}
	f, err := os.Open(opts.DataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))

	// Optional: model load/fit
	var lom *learn.LogOddsModel
	if opts.ModelIn != "" {
		m, err := learn.LoadLogOdds(opts.ModelIn)
		if err != nil {
			return nil, fmt.Errorf("load model: %w", err)
		}
		lom = m
	}
	if opts.FitPath != "" {
		m, err := fitSocialIQA(opts.FitPath)
		if err != nil {
			return nil, fmt.Errorf("fit: %w", err)
		}
		lom = m
		if opts.ModelOut != "" {
			_ = lom.Save(opts.ModelOut)
		}
	}

	R := rng.NewFacade(opts.Seed)
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
		gold, err := parseLabel(ex.Label)
		if err != nil {
			continue
		}
		answers := []string{ex.A, ex.B, ex.C}
		var best int
		if lom != nil || opts.Model == "hybrid" {
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
			scores := methods.CombinedScores(ex.Question, answers, weights, opts.Seed, vecDim, mctsRollouts)
			if lom != nil {
				for i, opt := range answers {
					s := 0.0
					for _, t := range textnorm.Tokens(opt) {
						if w, ok := lom.Weights[t]; ok {
							s += w
						}
					}
					scores[i] += opts.Params["w_lo"] * s
				}
			}
			bestIdx, bestS := 0, -1e18
			for i, s := range scores {
				if s > bestS {
					bestIdx, bestS = i, s
				}
			}
			best = bestIdx
		} else if opts.Model == "maths" {
			best = models.MathsMCQ(ex.Question, answers, 4096)
		} else {
			qb := bag(ex.Question)
			bestScore := -1
			for i, ans := range answers {
				sc := overlap(qb, bag(ans))
				if sc > bestScore || (sc == bestScore && R.Intn(2) == 0) {
					best, bestScore = i, sc
				}
			}
		}
		if best == gold {
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
		Task:       "socialiqa",
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

// ---- training helpers ----

func fitSocialIQA(path string) (*learn.LogOddsModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))
	cpos := map[string]int{}
	cneg := map[string]int{}
	npos, nneg := 0, 0
	for {
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, err
		}
		gold, err := parseLabel(ex.Label)
		if err != nil {
			continue
		}
		opts := []string{ex.A, ex.B, ex.C}
		for i, o := range opts {
			toks := textnorm.Tokens(o)
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
		w[t] = math.Log(py) - math.Log(pn)
	}
	return &learn.LogOddsModel{Weights: w}, nil
}

// hybrid scoring factored into methods.CombinedScores above
