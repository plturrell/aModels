package preprocess

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"ai_benchmarks/benchmarks/arc"
	"ai_benchmarks/internal/lnn"
	"ai_benchmarks/internal/methods"
)

// EnrichedTask holds the original task and the discovered transformation sequence.
type EnrichedTask struct {
	Task     arc.Task `json:"task"`
	Solution []string `json:"solution"`
}

// Preprocess runs the autopilot pre-processing layer.
func Preprocess(dataIn, dataOut, lnnModelPath string) error {
	fmt.Println("Running autopilot pre-processing layer...")
	fmt.Printf("Data in: %s\n", dataIn)
	fmt.Printf("Data out: %s\n", dataOut)
	fmt.Printf("LNN model: %s\n", lnnModelPath)

	// Load the LNN model
	calibrator, err := lnn.LookupCalibrator("arc")
	if err != nil {
		return fmt.Errorf("failed to create calibrator: %w", err)
	}
	if err := calibrator.Load(lnnModelPath); err != nil {
		return fmt.Errorf("failed to load lnn model: %w", err)
	}

	tasks, err := loadTasks(dataIn, 0)
	if err != nil {
		return fmt.Errorf("failed to load tasks: %w", err)
	}

	outFile, err := os.Create(dataOut)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	encoder := json.NewEncoder(outFile)

	for _, task := range tasks {
		enrichedTask, err := processTask(task, calibrator)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to process task: %v\n", err)
			continue
		}

		if err := encoder.Encode(enrichedTask); err != nil {
			return fmt.Errorf("failed to encode enriched task: %w", err)
		}
	}

	return nil
}

func processTask(task arc.Task, calibrator lnn.Calibrator) (EnrichedTask, error) {
	fmt.Printf("Processing task...\n")

	output, err := calibrator.Generate("arc", nil)
	if err != nil {
		return EnrichedTask{}, fmt.Errorf("failed to generate params: %w", err)
	}

	maxDepth := int(output.Params["arc_depth"])
	rollouts := int(output.Params["mcts_rollouts"])

	lib := methods.DefaultArcLib()
	reward := func(seq []string) float64 {
		return calculateReward(seq, task, lib)
	}

	seq := methods.ArcSearchMCTS(maxDepth, rollouts, 0, lib, reward, output.Policy)

	fmt.Printf("Found sequence: %v\n", seq)

	enrichedTask := EnrichedTask{
		Task:     task,
		Solution: seq,
	}

	return enrichedTask, nil
}

func calculateReward(seq []string, task arc.Task, lib methods.ArcLib) float64 {
	// Apply sequence to each train input and compute average match after color map
	s := 0.0
	n := 0
	for _, p := range task.Train {
		g := p.In
		for _, c := range seq {
			g = lib.Funcs[c](g)
		}

		mp, ok := learnColorMap(task.Train)
		if !ok {
			mp = learnColorMapSoft(task.Train)
		}
		pred := applyColorMap(g, mp)

		s += arc.MatchRatio(pred, p.Out)
		n++
	}
	if n == 0 {
		return 0
	}
	return s / float64(n)
}

// Try a simple per-cell color remapping learned from train pairs. If not consistent, fall back to identity.
func learnColorMap(train []arc.Pair) (map[int]int, bool) {
	mp := map[int]int{}
	for _, p := range train {
		in, out := p.In, p.Out
		if len(in) != len(out) {
			return nil, false
		}
		for i := range in {
			if len(in[i]) != len(out[i]) {
				return nil, false
			}
			for j := range in[i] {
				a, b := in[i][j], out[i][j]
				if v, ok := mp[a]; ok {
					if v != b {
						return nil, false
					}
				} else {
					mp[a] = b
				}
			}
		}
	}
	return mp, true
}

// Soft color map: for each input color, pick the most frequent output color at aligned positions.
func learnColorMapSoft(train []arc.Pair) map[int]int {
	counts := map[int]map[int]int{} // a -> (b -> count)
	for _, p := range train {
		in, out := p.In, p.Out
		if len(in) != len(out) {
			continue
		}
		for i := range in {
			if len(in[i]) != len(out[i]) {
				continue
			}
			for j := range in[i] {
				a, b := in[i][j], out[i][j]
				m, ok := counts[a]
				if !ok {
					m = map[int]int{}
					counts[a] = m
				}
				m[b]++
			}
		}
	}
	mp := map[int]int{}
	for a, m := range counts {
		bestB, bestN := 0, -1
		for b, n := range m {
			if n > bestN {
				bestN, bestB = n, b
			}
		}
		mp[a] = bestB
	}
	return mp
}

func applyColorMap(in [][]int, mp map[int]int) [][]int {
	h := len(in)
	out := make([][]int, h)
	for i := range in {
		w := len(in[i])
		row := make([]int, w)
		for j := 0; j < w; j++ {
			c := in[i][j]
			if v, ok := mp[c]; ok {
				row[j] = v
			} else {
				row[j] = c
			}
		}
		out[i] = row
	}
	return out
}

// loadTasks supports a single JSON file (one task) or a directory of JSON files.
func loadTasks(path string, limit int) ([]arc.Task, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	tasks := []arc.Task{}
	if fi.IsDir() {
		err = filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			if filepath.Ext(p) != ".json" {
				return nil
			}
			if limit > 0 && len(tasks) >= limit {
				return fs.SkipDir
			}
			t, e := loadOne(p)
			if e != nil {
				return e
			}
			tasks = append(tasks, t)
			return nil
		})
		return tasks, err
	}
	t, err := loadOne(path)
	if err != nil {
		return nil, err
	}
	if limit > 0 {
		tasks = append(tasks, t)
	} else {
		tasks = append(tasks, t)
	}
	return tasks, nil
}

func loadOne(p string) (arc.Task, error) {
	f, err := os.Open(p)
	if err != nil {
		return arc.Task{}, err
	}
	defer f.Close()
	var t arc.Task
	dec := json.NewDecoder(f)
	if err := dec.Decode(&t); err != nil {
		return arc.Task{}, err
	}
	return t, nil
}
