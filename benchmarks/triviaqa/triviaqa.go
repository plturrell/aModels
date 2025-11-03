package triviaqa

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
	"unicode"

	"ai_benchmarks/internal/learn"
	"ai_benchmarks/internal/mathvec"
	"ai_benchmarks/internal/methods"
	"ai_benchmarks/internal/registry"
	"ai_benchmarks/internal/textnorm"
	"math"
)

type Example struct {
	Question string   `json:"question"`
	Answer   string   `json:"answer,omitempty"`
	Answers  []string `json:"answers,omitempty"`
}

func goldAnswers(ex Example) []string {
	if len(ex.Answers) > 0 {
		return ex.Answers
	}
	if ex.Answer != "" {
		return []string{ex.Answer}
	}
	return nil
}

// baseline: naive keyword guess — choose last capitalized token from the question.
func baselinePredict(q string) string {
	toks := strings.Fields(q)
	var cand string
	for _, t := range toks {
		if len(t) == 0 {
			continue
		}
		r := []rune(t)
		if unicode.IsUpper(r[0]) {
			cand = t
		}
	}
	if cand == "" && len(toks) > 0 {
		cand = toks[len(toks)-1]
	}
	return strings.Trim(cand, ",.?!;:\"'()[]{}")
}

type runner struct{}

func (runner) ID() string            { return "triviaqa" }
func (runner) Description() string   { return "TriviaQA: open-domain QA; metrics=EM,F1" }
func (runner) DefaultMetric() string { return "f1" }

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	if fi, err := os.Stat(opts.DataPath); err != nil || fi.IsDir() {
		return nil, registry.Errf("--data=<path to JSONL> (TriviaQA)", "data must be a JSONL file: %v", err)
	}
	f, err := os.Open(opts.DataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))

	// Optional: token log-odds model trained from answer tokens
	var lom *learn.LogOddsModel
	if opts.ModelIn != "" {
		if m, err := learn.LoadLogOdds(opts.ModelIn); err == nil {
			lom = m
		}
	}
	if opts.FitPath != "" {
		if m, err := fitTrivia(opts.FitPath); err == nil {
			lom = m
			if opts.ModelOut != "" {
				_ = lom.Save(opts.ModelOut)
			}
		} else {
			return nil, fmt.Errorf("fit: %w", err)
		}
	}

	total := 0
	emC := 0
	f1Sum := 0.0
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
		golds := goldAnswers(ex)
		if len(golds) == 0 {
			continue
		}
		var pred string
		if opts.Model == "hybrid" || lom != nil {
			pred = hybridPredict(lom, ex.Question, opts)
		} else {
			pred = baselinePredict(ex.Question)
		}
		// evaluate against best matching gold
		bestF1 := 0.0
		matched := false
		for _, g := range golds {
			if textnorm.ExactMatch(pred, g) {
				matched = true
			}
			if f1 := textnorm.F1(pred, g); f1 > bestF1 {
				bestF1 = f1
			}
		}
		if matched {
			emC++
		}
		f1Sum += bestF1
		total++
	}
	finished := time.Now().Unix()
	if total == 0 {
		return nil, errors.New("no examples loaded (check data format)")
	}
	em := float64(emC) / float64(total)
	f1 := f1Sum / float64(total)
	sum := &registry.Summary{
		Task:       "triviaqa",
		Model:      opts.Model,
		Count:      total,
		Metrics:    map[string]float64{"exact_match": em, "f1": f1},
		StartedAt:  started,
		FinishedAt: finished,
		Details:    map[string]any{"em_count": emC},
	}
	return sum, nil
}

func init() { registry.Register(runner{}) }

// --- Hybrid predictor for open QA (pure maths) ---

// fitTrivia learns token log-odds where positives are answer tokens and negatives are question tokens.
func fitTrivia(path string) (*learn.LogOddsModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReader(f))
	pos := map[string]int{}
	neg := map[string]int{}
	npos, nneg := 0, 0
	for {
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, err
		}
		for _, g := range goldAnswers(ex) {
			for _, t := range textnorm.Tokens(g) {
				pos[t]++
				npos++
			}
		}
		for _, t := range textnorm.Tokens(ex.Question) {
			neg[t]++
			nneg++
		}
	}
	// fit
	Vset := map[string]struct{}{}
	for t := range pos {
		Vset[t] = struct{}{}
	}
	for t := range neg {
		Vset[t] = struct{}{}
	}
	alpha := 0.5
	V := float64(len(Vset))
	pden := float64(npos) + alpha*V
	nden := float64(nneg) + alpha*V
	w := map[string]float64{}
	for t := range Vset {
		py := (float64(pos[t]) + alpha) / pden
		pn := (float64(neg[t]) + alpha) / nden
		w[t] = mathLog(py) - mathLog(pn)
	}
	return &learn.LogOddsModel{Weights: w}, nil
}

func mathLog(x float64) float64 { return math.Log(x) }

// hybridPredict builds candidate answers from the question and scores them via components.
func hybridPredict(lom *learn.LogOddsModel, question string, opts registry.RunOptions) string {
	cands := candidatesFromQuestion(question)
	if len(cands) == 0 {
		return baselinePredict(question)
	}
	// params
	wLo := 1.0
	wVe := 1.0
	wLNN := 0.5
	wSym := 0.5
	wFrac := 0.2
	vecDim := 2048
	if v, ok := opts.Params["w_lo"]; ok {
		wLo = v
	}
	if v, ok := opts.Params["w_vec"]; ok {
		wVe = v
	}
	if v, ok := opts.Params["w_lnn"]; ok {
		wLNN = v
	}
	if v, ok := opts.Params["w_sym"]; ok {
		wSym = v
	}
	if v, ok := opts.Params["w_frac"]; ok {
		wFrac = v
	}
	if v, ok := opts.Params["vec_dim"]; ok {
		vecDim = int(v)
	}

	v := mathvec.NewVectorizer(vecDim)
	qp := v.Vec(question)
	sLNN := methods.ScoreOptionsLNN(question, cands)
	sSym := methods.ScoreOptionsSymbolic(question, cands)
	sFrac := methods.ScoreOptionsFractal(question, cands)
	best := cands[0]
	bestScore := -1e18
	for i, cand := range cands {
		s := 0.0
		if lom != nil {
			for _, t := range textnorm.Tokens(cand) {
				if w, ok := lom.Weights[t]; ok {
					s += w
				}
			}
		}
		s = wLo*s + wVe*mathvec.Cosine(qp, v.Vec(cand)) + wLNN*sLNN[i] + wSym*sSym[i] + wFrac*sFrac[i]
		if s > bestScore {
			best, bestScore = cand, s
		}
	}
	return best
}

func candidatesFromQuestion(q string) []string {
	// Capitalized tokens and final token as candidates; also include adjacent bigrams of capitalized tokens
	toks := strings.Fields(q)
	m := map[string]struct{}{}
	add := func(s string) {
		s = strings.Trim(s, ",.?!;:\"'()[]{}")
		if s != "" {
			m[s] = struct{}{}
		}
	}
	for i := 0; i < len(toks); i++ {
		t := toks[i]
		r := []rune(t)
		if len(r) > 0 && unicode.IsUpper(r[0]) {
			add(t)
		}
		if i+1 < len(toks) {
			r2 := []rune(toks[i+1])
			if len(r) > 0 && len(r2) > 0 && unicode.IsUpper(r[0]) && unicode.IsUpper(r2[0]) {
				add(t + " " + toks[i+1])
			}
		}
	}
	if len(toks) > 0 {
		add(toks[len(toks)-1])
	}
	out := make([]string, 0, len(m))
	for s := range m {
		out = append(out, s)
	}
	return out
}
