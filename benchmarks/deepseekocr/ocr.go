package deepseekocr

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"ai_benchmarks/internal/ds"
	"ai_benchmarks/internal/registry"
	"ai_benchmarks/internal/vision"
)

type example struct {
	Image  string `json:"image"`
	Text   string `json:"text"`
	Prompt string `json:"prompt,omitempty"`
}

type runner struct{}

func (runner) ID() string            { return "deepseek_ocr" }
func (runner) Description() string   { return "DeepSeek OCR evaluation; metrics=CER/WER/exact_match" }
func (runner) DefaultMetric() string { return "cer" }

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	if strings.TrimSpace(opts.DataPath) == "" {
		return nil, registry.Errf("--data=<path>", "missing dataset path")
	}

	dsPath := opts.DataPath
	info, err := os.Stat(dsPath)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		dsPath = filepath.Join(dsPath, "annotations.jsonl")
	}
	if _, err := os.Stat(dsPath); err != nil {
		return nil, fmt.Errorf("annotations file not found: %w", err)
	}

	endpoint := firstNonEmpty(os.Getenv("DEEPSEEK_OCR_ENDPOINT"), "http://localhost:9393/v1/ocr")
	apiKey := os.Getenv("DEEPSEEK_OCR_API_KEY")
	modelVariant := os.Getenv("DEEPSEEK_OCR_MODEL")
	defaultPrompt := os.Getenv("DEEPSEEK_OCR_PROMPT")
	timeout := parseDuration(os.Getenv("DEEPSEEK_OCR_TIMEOUT"), 60*time.Second)

	client := vision.NewDeepSeekClient(vision.DeepSeekConfig{
		Endpoint: endpoint,
		APIKey:   apiKey,
		Timeout:  timeout,
	})

	type detail struct {
		Image      string  `json:"image"`
		Expected   string  `json:"expected"`
		Prediction string  `json:"prediction"`
		CER        float64 `json:"cer"`
		WER        float64 `json:"wer"`
		ExactMatch bool    `json:"exact_match"`
		LatencyMs  float64 `json:"latency_ms"`
		Error      string  `json:"error,omitempty"`
	}

	var details []detail
	var errorDetails []detail
	charErrors := 0
	charTotal := 0
	wordErrors := 0
	wordTotal := 0
	exactMatches := 0
	count := 0

	datasetDir := filepath.Dir(dsPath)
	started := time.Now().Unix()

	_, err = ds.ReadJSONL[example](dsPath, opts.Limit, func(ex *example) error {
		count++
		imgPath := ex.Image
		if !filepath.IsAbs(imgPath) {
			imgPath = filepath.Join(datasetDir, imgPath)
		}
		record := detail{Image: ex.Image, Expected: ex.Text}

		imgBytes, err := os.ReadFile(imgPath)
		if err != nil {
			record.Error = fmt.Sprintf("read image: %v", err)
			if count <= 50 {
				details = append(details, record)
			} else {
				errorDetails = append(errorDetails, record)
			}
			return nil
		}

		prompt := firstNonEmpty(ex.Prompt, defaultPrompt)
		callCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		start := time.Now()
		pred, err := client.ExtractText(callCtx, imgBytes, prompt, modelVariant)
		latency := time.Since(start).Seconds() * 1000
		record.LatencyMs = latency

		if err != nil {
			record.Error = err.Error()
			if count <= 50 {
				details = append(details, record)
			} else {
				errorDetails = append(errorDetails, record)
			}
			return nil
		}

		record.Prediction = pred
		normPred := normalize(pred)
		normTruth := normalize(ex.Text)

		cerRatio, totalChars, charDist := charErrorRate(normPred, normTruth)
		werRatio, totalWords, wordDist := wordErrorRate(normPred, normTruth)
		record.CER = cerRatio
		record.WER = werRatio
		record.ExactMatch = normPred == normTruth

		charErrors += charDist
		charTotal += totalChars
		wordErrors += wordDist
		wordTotal += totalWords
		if record.ExactMatch {
			exactMatches++
		}

		if count <= 50 {
			details = append(details, record)
		} else if record.Error != "" || !record.ExactMatch {
			errorDetails = append(errorDetails, record)
		}
		return nil
	})

	if err != nil {
		var usage *registry.UsageError
		if errors.As(err, &usage) {
			return nil, usage
		}
		return nil, err
	}

	if count == 0 {
		return nil, errors.New("dataset yielded zero examples")
	}

	finished := time.Now().Unix()
	cer := ratio(charErrors, charTotal)
	wer := ratio(wordErrors, wordTotal)
	em := float64(exactMatches) / float64(count)

	sum := &registry.Summary{
		Task:  "deepseek_ocr",
		Model: firstNonEmpty(opts.Model, "deepseek-ocr"),
		Count: count,
		Metrics: map[string]float64{
			"cer":         cer,
			"wer":         wer,
			"exact_match": em,
			"char_total":  float64(charTotal),
			"word_total":  float64(wordTotal),
		},
		StartedAt:  started,
		FinishedAt: finished,
	}

	switch {
	case len(details) > 0:
		sum.Details = details
	case len(errorDetails) > 0:
		sum.Details = errorDetails
	}
	return sum, nil
}

func init() { registry.Register(runner{}) }

// Helpers --------------------------------------------------------------------

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func parseDuration(s string, fallback time.Duration) time.Duration {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	if d, err := time.ParseDuration(s); err == nil {
		return d
	}
	if ms, err := strconv.Atoi(s); err == nil {
		return time.Duration(ms) * time.Millisecond
	}
	return fallback
}

func normalize(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	fields := strings.Fields(s)
	return strings.Join(fields, " ")
}

func charErrorRate(pred, truth string) (float64, int, int) {
	predRunes := []rune(pred)
	truthRunes := []rune(truth)
	dist := levenshteinRunes(predRunes, truthRunes)
	total := len(truthRunes)
	if total == 0 {
		if dist == 0 {
			return 0, 0, 0
		}
		return 1, 0, dist
	}
	return float64(dist) / float64(total), total, dist
}

func wordErrorRate(pred, truth string) (float64, int, int) {
	predWords := strings.Fields(pred)
	truthWords := strings.Fields(truth)
	dist := levenshteinStrings(predWords, truthWords)
	total := len(truthWords)
	if total == 0 {
		if dist == 0 {
			return 0, 0, 0
		}
		return 1, 0, dist
	}
	return float64(dist) / float64(total), total, dist
}

func ratio(num, den int) float64 {
	if den <= 0 {
		if num == 0 {
			return 0
		}
		return 1
	}
	return float64(num) / float64(den)
}

func levenshteinRunes(a, b []rune) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := 0; j <= len(b); j++ {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		curr[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			del := prev[j] + 1
			ins := curr[j-1] + 1
			sub := prev[j-1] + cost
			curr[j] = min(del, ins, sub)
		}
		prev, curr = curr, prev
	}
	return prev[len(b)]
}

func levenshteinStrings(a, b []string) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := 0; j <= len(b); j++ {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		curr[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			del := prev[j] + 1
			ins := curr[j-1] + 1
			sub := prev[j-1] + cost
			curr[j] = min(del, ins, sub)
		}
		prev, curr = curr, prev
	}
	return prev[len(b)]
}

func min(values ...int) int {
	m := values[0]
	for _, v := range values[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
