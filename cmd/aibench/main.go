package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"ai_benchmarks/internal/preprocess"
	"ai_benchmarks/internal/registry"

	// Register all benchmarks via side-effect imports
	_ "ai_benchmarks/benchmarks/arc"
	_ "ai_benchmarks/benchmarks/boolq"
	_ "ai_benchmarks/benchmarks/deepseekocr"
	_ "ai_benchmarks/benchmarks/hellaswag"
	_ "ai_benchmarks/benchmarks/piqa"
	_ "ai_benchmarks/benchmarks/socialiq"
	_ "ai_benchmarks/benchmarks/triviaqa"
)

type runFlags struct {
	task      string
	data      string
	model     string
	out       string
	limit     int
	seed      int64
	fit       string
	modelIn   string
	modelOut  string
	params    []string
	paramsIn  string
	paramsOut string
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	cmd := os.Args[1]
	switch cmd {
	case "list":
		listCmd()
	case "run":
		runCmd(os.Args[2:])
	case "tune":
		tuneCmd(os.Args[2:])
	case "nash":
		nashCmd(os.Args[2:])
	case "preprocess":
		preprocessCmd(os.Args[2:])
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		usage()
		os.Exit(2)
	}
}

func preprocessCmd(args []string) {
	var dataIn, dataOut, lnnModel string
	fs := flag.NewFlagSet("preprocess", flag.ExitOnError)
	fs.StringVar(&dataIn, "data-in", "", "path to the input dataset file or directory")
	fs.StringVar(&dataOut, "data-out", "", "path to the output dataset file or directory")
	fs.StringVar(&lnnModel, "lnn-model", "", "path to the LNN model file")
	_ = fs.Parse(args)

	if err := preprocess.Preprocess(dataIn, dataOut, lnnModel); err != nil {
		fmt.Fprintf(os.Stderr, "preprocess failed: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Println("aibench - pure Go AI benchmarks (6 tests)")
	fmt.Println("\nUsage:")
	fmt.Println("  aibench list")
	fmt.Println("  aibench run --task=<id> --data=<path> [--model=<name>] [--out=<file>] [--limit=N] [--seed=S]")
	fmt.Println("  aibench preprocess --data-in=<path> --data-out=<path> --lnn-model=<path>")
	fmt.Println("\nTasks:")
	for _, r := range registry.All() {
		fmt.Printf("  - %s: %s\n", r.ID(), r.Description())
	}
}

func listCmd() {
	fmt.Println("Available tasks:")
	for _, r := range registry.All() {
		fmt.Printf("- %s: %s (metric: %s)\n", r.ID(), r.Description(), r.DefaultMetric())
	}
}

func runCmd(args []string) {
	var rf runFlags
	var preprocessData bool
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	fs.StringVar(&rf.task, "task", "", "task id to run (e.g., triviaqa, boolq, hellaswag, socialiqa, piqa, arc)")
	fs.StringVar(&rf.data, "data", "", "path to dataset file or directory")
	fs.StringVar(&rf.model, "model", "baseline", "model to use (task-dependent; e.g., baseline/random/heuristic)")
	fs.StringVar(&rf.out, "out", "", "optional output JSON file for summary + run details")
	fs.IntVar(&rf.limit, "limit", 0, "optional limit on number of examples (0=all)")
	fs.Int64Var(&rf.seed, "seed", time.Now().UnixNano(), "random seed")
	fs.StringVar(&rf.fit, "fit", "", "optional training dataset path to fit a pure-maths model")
	fs.StringVar(&rf.modelIn, "model-in", "", "optional path to load a previously saved model JSON")
	fs.StringVar(&rf.modelOut, "model-out", "", "optional path to save a trained model JSON")
	fs.Var((*paramList)(&rf.params), "param", "override param as key=value (repeatable); e.g., -param w_vec=1.0 -param vec_dim=4096")
	fs.StringVar(&rf.paramsIn, "params-in", "", "optional JSON file with param map to load (string->float)")
	fs.StringVar(&rf.paramsOut, "params-out", "", "optional path to save the final params used in this run")
	fs.BoolVar(&preprocessData, "preprocess-data", false, "if true, run the autopilot pre-processing layer on the training data")
	var differentialPrivacy bool
	fs.BoolVar(&differentialPrivacy, "differential-privacy", false, "if true, use differentially private training")
	var epsilon, delta float64
	fs.Float64Var(&epsilon, "epsilon", 1.0, "epsilon for differential privacy")
	fs.Float64Var(&delta, "delta", 1e-5, "delta for differential privacy")

	_ = fs.Parse(args)

	if differentialPrivacy {
		cmd := exec.Command("python3", "scripts/train_dp.py", "--model-path", rf.modelIn, "--model-out", rf.modelOut, "--data-api-url", rf.fit, "--epsilon", fmt.Sprintf("%f", epsilon), "--delta", fmt.Sprintf("%f", delta))
		cmd.Dir = "/Users/user/Library/CloudStorage/Dropbox/agenticAiETH/agenticAiETH_layer4_Training"
		output, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "differential privacy training failed: %s\n%s", err, output)
			os.Exit(1)
		}
		fmt.Println("Differential privacy training complete.")
		return
	}

	if preprocessData {
		tempFile, err := os.CreateTemp("", "enriched-*.jsonl")
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to create temp file for enriched data: %v\n", err)
			os.Exit(1)
		}
		defer os.Remove(tempFile.Name())

		if err := preprocess.Preprocess(rf.fit, tempFile.Name(), rf.modelIn); err != nil {
			fmt.Fprintf(os.Stderr, "preprocess failed: %v\n", err)
			os.Exit(1)
		}
		rf.fit = tempFile.Name()
	}

	if strings.TrimSpace(rf.task) == "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "missing --task")
		os.Exit(2)
	}
	if strings.TrimSpace(rf.data) == "" && strings.TrimSpace(rf.fit) == "" {
		fs.Usage()
		fmt.Fprintln(os.Stderr, "missing --data or --fit")
		os.Exit(2)
	}

	r := registry.Lookup(rf.task)
	if r == nil {
		fmt.Fprintf(os.Stderr, "unknown task: %s\n", rf.task)
		os.Exit(2)
	}

	ctx := context.Background()
	// Build params map: load from file then apply CLI overrides
	params := map[string]float64{}
	if rf.paramsIn != "" {
		if pm, err := readParamsFile(rf.paramsIn); err == nil {
			params = pm
		} else {
			fmt.Fprintf(os.Stderr, "warn: failed to load params-in: %v\n", err)
		}
	}
	for k, v := range parseParams(rf.params) {
		params[k] = v
	}

	sum, err := r.Run(ctx, registry.RunOptions{
		DataPath: rf.data,
		Model:    rf.model,
		Limit:    rf.limit,
		Seed:     rf.seed,
		FitPath:  rf.fit,
		ModelIn:  rf.modelIn,
		ModelOut: rf.modelOut,
		Params:   params,
	})
	if err != nil {
		var ue *registry.UsageError
		if errors.As(err, &ue) {
			fmt.Fprintf(os.Stderr, "error: %v\n", ue)
			fmt.Fprintf(os.Stderr, "usage: %s\n", ue.Hint)
			os.Exit(2)
		}
		fmt.Fprintf(os.Stderr, "run failed: %v\n", err)
		os.Exit(1)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(sum); err != nil {
		fmt.Fprintf(os.Stderr, "failed to write summary: %v\n", err)
		os.Exit(1)
	}

	if rf.out != "" {
		if err := writeJSONFile(rf.out, sum); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write --out: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", rf.out)
	}

	if rf.paramsOut != "" {
		if err := writeJSONFile(rf.paramsOut, params); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write --params-out: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", rf.paramsOut)
	}
}

func writeJSONFile(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// --- param parsing helpers ---

type paramList []string

func (p *paramList) String() string { return strings.Join(*p, ",") }
func (p *paramList) Set(s string) error {
	*p = append(*p, s)
	return nil
}

func parseParams(items []string) map[string]float64 {
	out := map[string]float64{}
	for _, it := range items {
		// support comma-separated clusters too
		parts := strings.Split(it, ",")
		for _, seg := range parts {
			seg = strings.TrimSpace(seg)
			if seg == "" {
				continue
			}
			kv := strings.SplitN(seg, "=", 2)
			if len(kv) != 2 {
				continue
			}
			k := strings.TrimSpace(kv[0])
			v := strings.TrimSpace(kv[1])
			if k == "" || v == "" {
				continue
			}
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				out[k] = f
			}
		}
	}
	return out
}

func readParamsFile(path string) (map[string]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	m := map[string]float64{}
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}
	return m, nil
}
