package arc

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"ai_benchmarks/internal/methods"
	"ai_benchmarks/internal/registry"
)

type Pair struct {
	In  [][]int `json:"input"`
	Out [][]int `json:"output"`
}

type Task struct {
	Train []Pair `json:"train"`
	Test  []Pair `json:"test"`
}

// loadTasks supports a single JSON file (one task) or a directory of JSON files.
func loadTasks(path string, limit int) ([]Task, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	tasks := []Task{}
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

func loadOne(p string) (Task, error) {
	f, err := os.Open(p)
	if err != nil {
		return Task{}, err
	}
	defer f.Close()
	var t Task
	dec := json.NewDecoder(f)
	if err := dec.Decode(&t); err != nil {
		return Task{}, err
	}
	return t, nil
}

// Try a simple per-cell color remapping learned from train pairs. If not consistent, fall back to identity.
func learnColorMap(train []Pair) (map[int]int, bool) {
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
func learnColorMapSoft(train []Pair) map[int]int {
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

// baseline predictor: color map if consistent, else echo input.
func baselinePredict(in [][]int, train []Pair) [][]int {
	if mp, ok := learnColorMap(train); ok {
		return applyColorMap(in, mp)
	}
	return in
}

func equalGrid(a, b [][]int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

type runner struct{}

func (runner) ID() string { return "arc" }
func (runner) Description() string {
	return "ARC (Chollet 2019): reasoning with grid I/O; metric=accuracy"
}
func (runner) DefaultMetric() string { return "accuracy" }

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	tasks, err := loadTasks(opts.DataPath, opts.Limit)
	if err != nil {
		return nil, registry.Errf("--data=<path to ARC task .json or folder>", "failed to load: %v", err)
	}

	// Auto-detect model and get initial hints (not hardcoded optimal values)
	detectedModel := AutoDetectModel(opts.Model)
	modelCfg := GetModelConfig(detectedModel)

	// Use model hints as INITIAL VALUES only if user hasn't provided params
	// The LNN will learn better values through training
	if len(opts.Params) == 0 {
		opts.Params = modelCfg.ToParams()
	} else {
		// User params take full precedence
		opts.Params = MergeParams(modelCfg.ToParams(), opts.Params)
	}

	total := 0
	correct := 0
	started := time.Now().Unix()
	for _, t := range tasks {
		for _, te := range t.Test {
			var pred [][]int
			if opts.Model == "hybrid" || detectedModel != "default" {
				pred = hybridPredictARC(te.In, t.Train, opts)
			} else {
				pred = baselinePredict(te.In, t.Train)
			}
			if equalGrid(pred, te.Out) {
				correct++
			}
			total++
		}
	}
	finished := time.Now().Unix()
	if total == 0 {
		return nil, errors.New("no ARC test cases found")
	}
	acc := float64(correct) / float64(total)
	sum := &registry.Summary{
		Task:       "arc",
		Model:      opts.Model,
		Count:      total,
		Metrics:    map[string]float64{"accuracy": acc},
		StartedAt:  started,
		FinishedAt: finished,
		Details: map[string]any{
			"correct":        correct,
			"tasks":          len(tasks),
			"detected_model": detectedModel,
			"model_config":   modelCfg.Name,
		},
	}
	return sum, nil
}

func init() { registry.Register(runner{}) }

// --- Hybrid ARC methods ---

type transform func([][]int) [][]int

func rot90(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, w)
	for i := 0; i < w; i++ {
		out[i] = make([]int, h)
	}
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			out[j][h-1-i] = g[i][j]
		}
	}
	return out
}
func rot180(g [][]int) [][]int { return rot90(rot90(g)) }
func rot270(g [][]int) [][]int { return rot90(rot180(g)) }
func flipH(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
		for j := 0; j < w; j++ {
			out[i][w-1-j] = g[i][j]
		}
	}
	return out
}
func flipV(g [][]int) [][]int {
	h := len(g)
	if h == 0 {
		return g
	}
	w := len(g[0])
	out := make([][]int, h)
	for i := 0; i < h; i++ {
		out[i] = make([]int, w)
	}
	for i := 0; i < h; i++ {
		copy(out[h-1-i], g[i])
	}
	return out
}

func MatchRatio(a, b [][]int) float64 {
	if len(a) != len(b) {
		return 0
	}
	if len(a) == 0 || len(a[0]) == 0 {
		return 0
	}
	h, w := len(a), len(a[0])
	eq := 0
	tot := h * w
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			if a[i][j] == b[i][j] {
				eq++
			}
		}
	}
	return float64(eq) / float64(tot)
}

func hybridPredictARC(in [][]int, train []Pair, opts registry.RunOptions) [][]int {
	// Use advanced program synthesis if enabled
	useSynthesis := false
	if v, ok := opts.Params["arc_synthesis"]; ok && v > 0 {
		useSynthesis = true
	}

	if useSynthesis {
		// Use LNN-generated parameters for synthesis configuration
		maxDepth := 3
		if v, ok := opts.Params["max_depth"]; ok {
			maxDepth = int(v)
		} else if v, ok := opts.Params["arc_depth"]; ok {
			maxDepth = int(v)
		}

		maxCandidates := 100
		if v, ok := opts.Params["max_candidates"]; ok {
			maxCandidates = int(v)
		}

		backgroundColor := 0
		if v, ok := opts.Params["background_color"]; ok {
			backgroundColor = int(v)
		}

		ps := &ProgramSynthesis{
			MaxDepth:        maxDepth,
			MaxCandidates:   maxCandidates,
			BackgroundColor: backgroundColor,
			Rand:            rand.New(rand.NewSource(opts.Seed)),
		}

		program := ps.Synthesize(train)
		if len(program) > 0 {
			return ApplyProgram(in, program)
		}
	}

	// derive color map (prefer strict, else soft; or force soft via param)
	forceSoft := false
	if v, ok := opts.Params["palette_soft"]; ok && v > 0 {
		forceSoft = true
	}
	var mp map[int]int
	if !forceSoft {
		if tmp, ok := learnColorMap(train); ok {
			mp = tmp
		}
	}
	if mp == nil {
		mp = learnColorMapSoft(train)
	}

	// If MCTS rollouts present, use search over transform sequences
	// Use LNN-generated values, no hardcoded defaults
	rollouts := 0
	if v, ok := opts.Params["mcts_rollouts"]; ok {
		rollouts = int(v)
	}

	depth := 0
	if v, ok := opts.Params["arc_depth"]; ok {
		d := int(v)
		if d > 0 {
			depth = d
		}
	}
	// Only use MCTS if both rollouts and depth are specified
	if depth == 0 && rollouts > 0 {
		depth = 3 // Minimal fallback only if rollouts requested but no depth
	}
	lib := methods.DefaultArcLib()
	if mv, ok := opts.Params["arc_mask"]; ok {
		lib = methods.BuildArcLibFromMask(int(mv))
	}
	if rollouts > 0 {
		reward := func(seq []string) float64 {
			// Apply sequence to each train input and compute average match after color map
			s := 0.0
			n := 0
			for _, p := range train {
				g := p.In
				for _, c := range seq {
					g = lib.Funcs[c](g)
				}
				pred := applyColorMap(g, mp)
				s += MatchRatio(pred, p.Out)
				n++
			}
			if n == 0 {
				return 0
			}
			return s / float64(n)
		}
		seq := methods.ArcSearchMCTS(depth, rollouts, opts.Seed, lib, reward, nil)
		// Apply best sequence to test input
		g := in
		for _, c := range seq {
			g = lib.Funcs[c](g)
		}
		return applyColorMap(g, mp)
	}
	// fallback: pick best single transform
	Ts := []transform{func(x [][]int) [][]int { return x }, rot90, rot180, rot270, flipH, flipV}
	bestT := Ts[0]
	bestScore := -1.0
	for _, T := range Ts {
		s := 0.0
		n := 0
		for _, p := range train {
			pin := T(p.In)
			pred := applyColorMap(pin, mp)
			s += MatchRatio(pred, p.Out)
			n++
		}
		if n > 0 {
			s /= float64(n)
		}
		if s > bestScore {
			bestScore, bestT = s, T
		}
	}
	_ = math.Log // keep import active
	return applyColorMap(bestT(in), mp)
}

// ... rest of the code remains the same ...
