package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"ai_benchmarks/internal/lnn"
	"ai_benchmarks/internal/registry"
	"gonum.org/v1/gonum/optimize"
)

type tuneConfig struct {
	Task       string
	Train      string
	Eval       string
	Iter       int
	Seed       int64
	Fixed      map[string]float64
	WeightsOut string
}

func tuneCmd(args []string) {
	fs := flag.NewFlagSet("tune", flag.ExitOnError)
	var task, train, eval, weightsOut string
	var iter int
	var seed int64
	var params []string
	var paramsIn string
	var refine bool
	var refineSteps int
	var refineStep float64
	var optimizer string

	// LNN flags
	var lnnMode bool
	var lnnStateFile string

	fs.StringVar(&task, "task", "", "task id (e.g., piqa, hellaswag, socialiqa, boolq, triviaqa, arc)")
	fs.StringVar(&train, "train", "", "path to training data (used for FitPath)")
	fs.StringVar(&eval, "eval", "", "path to evaluation data (used for DataPath)")
	fs.IntVar(&iter, "iters", 60, "number of random search iterations")
	fs.Int64Var(&seed, "seed", time.Now().UnixNano(), "random seed")
	fs.StringVar(&weightsOut, "weights-out", "", "optional JSON file to save best params")
	fs.Var((*paramList)(&params), "param", "fix param as key=value (repeatable)")
	fs.StringVar(&paramsIn, "params-in", "", "optional JSON file with starting params (also treated as fixed unless overridden)")
	fs.BoolVar(&refine, "refine", true, "run a coordinate local search refine stage after random search")
	fs.IntVar(&refineSteps, "refine-steps", 2, "number of coordinate passes")
	fs.Float64Var(&refineStep, "refine-step", 0.2, "initial step size for weight/alpha coordinates (decays if no improvement)")
	fs.StringVar(&optimizer, "optimizer", "coordinate", "refinement optimizer: coordinate or nelder-mead")
	fs.BoolVar(&lnnMode, "lnn", false, "enable LNN-based recursive tuning mode")
	fs.StringVar(&lnnStateFile, "lnn-state-file", "", "path to save/load LNN state")
	_ = fs.Parse(args)

	if task == "" {
		fmt.Fprintln(os.Stderr, "missing --task")
		os.Exit(2)
	}
	if !lnnMode && eval == "" {
		fmt.Fprintln(os.Stderr, "missing --eval for standard tuning")
		os.Exit(2)
	}
	if lnnMode && lnnStateFile == "" {
		lnnStateFile = task + ".lnn.state"
		fmt.Fprintf(os.Stderr, "info: using default LNN state file: %s\n", lnnStateFile)
	}
	fixed := parseParams(params)
	if paramsIn != "" {
		if mp, err := readParamsFile(paramsIn); err == nil {
			for k, v := range mp {
				if _, ok := fixed[k]; !ok {
					fixed[k] = v
				}
			}
		} else {
			fmt.Fprintln(os.Stderr, "warn: failed to read params-in:", err)
		}
	}
	if iter <= 0 {
		iter = 60
	}
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	r := registry.Lookup(task)
	if r == nil {
		fmt.Fprintln(os.Stderr, "unknown task:", task)
		os.Exit(2)
	}

	if lnnMode {
		// LNN-based tuning
		cfg := lnn.LNNTuneConfig{
			LearningRate:  0.001,
			MaxIterations: iter,
			StateFile:     lnnStateFile,
			UseRecursive:  true,
		}
		tuner, err := lnn.NewLNNTuner(r, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to create LNN tuner: %v\n", err)
			os.Exit(1)
		}
		bestParams, err := tuner.Tune()
		if err != nil {
			fmt.Fprintf(os.Stderr, "LNN tuning failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("\n--- LNN Tuning Complete ---")
		out := map[string]any{"task": task, "best_params": bestParams}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(out)
		if weightsOut != "" {
			_ = writeJSONFile(weightsOut, bestParams)
		}
		return
	}

	R := rand.New(rand.NewSource(seed))

	bestObj := -1.0
	bestParams := map[string]float64{}

	for it := 0; it < iter; it++ {
		params := sampleParams(R, fixed)
		sum, err := r.Run(context.Background(), registry.RunOptions{
			DataPath: eval,
			Model:    "hybrid",
			FitPath:  train,
			Params:   params,
			Seed:     seed + int64(it),
		})
		if err != nil {
			continue
		}
		m := pickMetric(task, sum)
		if m > bestObj {
			bestObj = m
			bestParams = params
			fmt.Fprintf(os.Stderr, "iter %d: %.6f %v\n", it+1, m, params)
		}
	}
	// Coordinate refine
	if refine {
		rp, rm := refineParams(optimizer, task, r, train, eval, bestParams, refineSteps, refineStep, seed)
		if rm > bestObj {
			bestObj, bestParams = rm, rp
		}
	}
	if bestObj < 0 {
		fmt.Fprintln(os.Stderr, "no feasible params found")
		os.Exit(1)
	}
	out := map[string]any{"task": task, "best_metric": bestObj, "params": bestParams}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
	if weightsOut != "" {
		_ = writeJSONFile(weightsOut, bestParams)
	}
}

func sampleParams(R *rand.Rand, fixed map[string]float64) map[string]float64 {
	p := map[string]float64{}
	// candidate keys and ranges
	// alpha, weights, vec_dim, mcts_rollouts, arc_depth
	set := func(k string, v float64) {
		if _, ok := fixed[k]; ok {
			p[k] = fixed[k]
		} else {
			p[k] = v
		}
	}
	// sampling helpers
	rnd := func(lo, hi float64) float64 { return lo + (hi-lo)*R.Float64() }
	set("alpha", rnd(0.1, 2.0))
	set("w_lo", rnd(0.0, 2.0))
	set("w_vec", rnd(0.0, 2.0))
	set("w_lnn", rnd(0.0, 2.0))
	set("w_sym", rnd(0.0, 2.0))
	set("w_mcts", rnd(0.0, 2.0))
	set("w_frac", rnd(0.0, 2.0))
	// discrete choices
	vd := []int{1024, 2048, 4096}
	md := []int{32, 64, 128}
	ad := []int{2, 3, 4}
	set("vec_dim", float64(vd[R.Intn(len(vd))]))
	set("mcts_rollouts", float64(md[R.Intn(len(md))]))
	set("arc_depth", float64(ad[R.Intn(len(ad))]))
	// bool-like toggles
	if _, ok := fixed["palette_soft"]; ok {
		p["palette_soft"] = fixed["palette_soft"]
	} else {
		p["palette_soft"] = float64(R.Intn(2))
	}
	if _, ok := fixed["arc_mask"]; ok {
		p["arc_mask"] = fixed["arc_mask"]
	} else {
		p["arc_mask"] = 63
	}
	return p
}

// refineParams performs a simple coordinate ascent on key weights and alpha.
func refineParams(optimizer, task string, r registry.Runner, train, eval string, start map[string]float64, passes int, step float64, seed int64) (map[string]float64, float64) {
	keys := []string{"w_lo", "w_vec", "w_lnn", "w_sym", "w_mcts", "w_frac", "alpha"}
	evalOnce := func(p map[string]float64) float64 {
		sum, err := r.Run(context.Background(), registry.RunOptions{DataPath: eval, Model: "hybrid", FitPath: train, Params: p, Seed: seed})
		if err != nil {
			return -1
		}
		return pickMetric(task, sum)
	}

	switch optimizer {
	case "nelder-mead":
		obj := func(x []float64) float64 {
			p := map[string]float64{}
			for k, v := range start {
				p[k] = v
			}
			for i, k := range keys {
				p[k] = x[i]
			}
			return -evalOnce(p) // Negate for minimization
		}
		x0 := make([]float64, len(keys))
		for i, k := range keys {
			x0[i] = start[k]
		}
		res, err := optimize.Minimize(optimize.Problem{Func: obj}, x0, nil, &optimize.NelderMead{})
		if err != nil {
			fmt.Fprintf(os.Stderr, "nelder-mead failed: %v\n", err)
			return start, evalOnce(start)
		}
		bestParams := map[string]float64{}
		for k, v := range start {
			bestParams[k] = v
		}
		for i, k := range keys {
			bestParams[k] = res.X[i]
		}
		return bestParams, -res.F

	case "coordinate":
		best := map[string]float64{}
		for k, v := range start {
			best[k] = v
		}
		bestM := evalOnce(best)
		clamp := func(k string, v float64) float64 {
			switch k {
			case "alpha":
				if v < 0.05 {
					v = 0.05
				}
				if v > 3.0 {
					v = 3.0
				}
			default:
				if v < 0 {
					v = 0
				}
				if v > 3.0 {
					v = 3.0
				}
			}
			return v
		}
		for pass := 0; pass < passes; pass++ {
			improved := false
			for _, k := range keys {
				base := best[k]
				tried := []float64{base + step, base - step}
				for _, v := range tried {
					cand := map[string]float64{}
					for kk, vv := range best {
						cand[kk] = vv
					}
					cand[k] = clamp(k, v)
					m := evalOnce(cand)
					if m > bestM {
						bestM = m
						best = cand
						improved = true
						fmt.Fprintf(os.Stderr, "refine %s: %.4f -> %.4f (k=%s)\n", task, base, cand[k], k)
					}
				}
			}
			if !improved {
				step *= 0.5
				if step < 1e-3 {
					break
				}
			}
		}
		return best, bestM
	default:
		fmt.Fprintf(os.Stderr, "unknown optimizer: %s\n", optimizer)
		return start, evalOnce(start)
	}
}

func pickMetric(task string, sum *registry.Summary) float64 {
	switch task {
	case "boolq", "hellaswag", "socialiqa", "piqa", "arc":
		return sum.Metrics["accuracy"]
	case "triviaqa":
		return 0.5*sum.Metrics["exact_match"] + 0.5*sum.Metrics["f1"]
	default:
		// fallback avg of metrics
		s := 0.0
		n := 0
		for _, v := range sum.Metrics {
			s += v
			n++
		}
		if n == 0 {
			return -1
		}
		return s / float64(n)
	}
}

// parseParams & paramList are provided in main.go (same package)
