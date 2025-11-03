package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"ai_benchmarks/internal/lnn"
	"ai_benchmarks/internal/registry"
)

type nashTask struct {
	ID    string `json:"id"`
	Train string `json:"train"`
	Eval  string `json:"eval"`
}

type nashConfig struct {
	Tasks      []nashTask `json:"tasks"`
	Iterations int        `json:"iterations"`
	Seed       int64      `json:"seed"`
}

func nashCmd(args []string) {
	fs := flag.NewFlagSet("nash", flag.ExitOnError)
	var confPath string
	var weightsOut string
	var paramsFixed []string
	var taskWeights []string
	var lnnMode bool
	fs.StringVar(&confPath, "config", "", "path to JSON config with tasks/train/eval")
	fs.StringVar(&weightsOut, "weights-out", "", "optional JSON file to save best params + per-task metrics")
	fs.Var((*paramList)(&paramsFixed), "param", "fix param as key=val (repeatable); others are searched")
	fs.Var((*paramList)(&taskWeights), "task-weight", "per-task weight id=val (repeatable) for Nash objective; defaults to 1.0")
	fs.BoolVar(&lnnMode, "lnn", false, "enable LNN-based calibration using per-task LNNs")
	_ = fs.Parse(args)
	if confPath == "" {
		fmt.Fprintln(os.Stderr, "missing --config")
		os.Exit(2)
	}
	f, err := os.Open(confPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open config:", err)
		os.Exit(2)
	}
	defer f.Close()
	var conf nashConfig
	if err := json.NewDecoder(f).Decode(&conf); err != nil {
		fmt.Fprintln(os.Stderr, "decode config:", err)
		os.Exit(2)
	}
	if conf.Iterations <= 0 {
		conf.Iterations = 50
	}
	if conf.Seed == 0 {
		conf.Seed = time.Now().UnixNano()
	}

	if lnnMode {
		runners := make([]registry.Runner, len(conf.Tasks))
		for i, t := range conf.Tasks {
			runners[i] = registry.Lookup(t.ID)
		}
		// In the new architecture, nash uses pre-trained LNNs, so it doesn't need a single state file.
		cfg := lnn.LNNTuneConfig{
			LearningRate:  0.001, // Not used for inference, but required by struct
			MaxIterations: conf.Iterations,
			UseRecursive:  false, // We are not training in nash mode anymore
		}
		calibrator, err := lnn.NewLNNNashCalibrator(runners, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to create LNN calibrator: %v\n", err)
			os.Exit(1)
		}
		bestParams, err := calibrator.Calibrate()
		if err != nil {
			fmt.Fprintf(os.Stderr, "LNN calibration failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("\n--- LNN Calibration Complete ---")
		out := map[string]any{"best_params": bestParams}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(out)
		if weightsOut != "" {
			_ = writeJSONFile(weightsOut, out)
		}
		return
	}

	R := rand.New(rand.NewSource(conf.Seed))
	fixed := parseParams(paramsFixed)
	tw := parseParams(taskWeights) // id->weight (defaults to 1.0 if absent)
	// Baseline utilities
	base := make(map[string]float64)
	for _, t := range conf.Tasks {
		base[t.ID] = 0
	}

	// Search over params:
	// alpha in [0.1,2], weights in [0,2], vec_dim in {1024,2048,4096}, mcts_rollouts in {32,64,128}
	bestObj := -1.0
	var bestParams map[string]float64
	for it := 0; it < conf.Iterations; it++ {
		params := sampleGlobalParams(R, fixed)
		obj, perTask := evalParams(conf.Tasks, params, base, tw)
		if obj > bestObj {
			bestObj = obj
			bestParams = params
			// stream interim result to stderr
			fmt.Fprintf(os.Stderr, "iter %d: obj=%.6f params=%v perTask=%v\n", it+1, obj, params, perTask)
		}
	}
	if bestParams == nil {
		fmt.Fprintln(os.Stderr, "no feasible params found")
		os.Exit(1)
	}
	// Output best params as JSON
	out := map[string]any{"best_objective": bestObj, "best_params": bestParams}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
	if weightsOut != "" {
		_ = writeJSONFile(weightsOut, out)
	}
}

func evalParams(tasks []nashTask, params map[string]float64, base map[string]float64, taskWeights map[string]float64) (float64, map[string]float64) {
	// Nash product over (metric - base), floored at epsilon
	eps := 1e-9
	sumLog := 0.0
	perTask := map[string]float64{}
	for _, t := range tasks {
		r := registry.Lookup(t.ID)
		if r == nil {
			continue
		}
		// Train and evaluate in one call: FitPath=train, DataPath=eval, Model=hybrid
		sum, err := r.Run(context.Background(), registry.RunOptions{
			DataPath: t.Eval,
			Model:    "hybrid",
			FitPath:  t.Train,
			Params:   params,
			Limit:    0,
		})
		if err != nil {
			continue
		}
		// Pick task metric
		var m float64
		switch t.ID {
		case "boolq", "hellaswag", "socialiqa", "piqa", "arc":
			m = sum.Metrics["accuracy"]
		case "triviaqa":
			// combine EM and F1 as average
			m = 0.5*(sum.Metrics["exact_match"]) + 0.5*(sum.Metrics["f1"])
		default:
			// fallback: mean of metrics
			s := 0.0
			n := 0
			for _, v := range sum.Metrics {
				s += v
				n++
			}
			if n > 0 {
				m = s / float64(n)
			}
		}
		perTask[t.ID] = m
		u := m - base[t.ID]
		if u < eps {
			u = eps
		}
		// weighted log-product objective; default task weight 1.0 if unspecified
		w := taskWeights[t.ID]
		if w == 0 {
			w = 1.0
		}
		sumLog += w * math.Log(u)
	}
	if len(perTask) == 0 {
		return -1, perTask
	}
	return sumLog, perTask
}

func sampleGlobalParams(R *rand.Rand, fixed map[string]float64) map[string]float64 {
	p := map[string]float64{}
	set := func(k string, v float64) {
		if f, ok := fixed[k]; ok {
			p[k] = f
		} else {
			p[k] = v
		}
	}
	rnd := func(lo, hi float64) float64 { return lo + (hi-lo)*R.Float64() }
	set("alpha", rnd(0.1, 2.0))
	set("w_lo", rnd(0.0, 2.0))
	set("w_vec", rnd(0.0, 2.0))
	set("w_lnn", rnd(0.0, 2.0))
	set("w_sym", rnd(0.0, 2.0))
	set("w_mcts", rnd(0.0, 2.0))
	set("w_frac", rnd(0.0, 2.0))
	vd := []int{1024, 2048, 4096}
	md := []int{32, 64, 128}
	ad := []int{2, 3, 4}
	set("vec_dim", float64(vd[R.Intn(len(vd))]))
	set("mcts_rollouts", float64(md[R.Intn(len(md))]))
	set("arc_depth", float64(ad[R.Intn(len(ad))]))
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
