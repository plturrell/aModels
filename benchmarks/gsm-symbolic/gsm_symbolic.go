package gsmsymbolic

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"ai_benchmarks/internal/registry"
)

// GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs
// Paper: https://machinelearning.apple.com/research/gsm-symbolic
// GitHub: https://github.com/apple/ml-gsm-symbolic

type runner struct{}

func (runner) ID() string {
	return "gsm-symbolic"
}

func (runner) Description() string {
	return "GSM-Symbolic: Mathematical reasoning with symbolic perturbations; metric=accuracy"
}

func (runner) DefaultMetric() string {
	return "accuracy"
}

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	// Validate data path
	if fi, err := os.Stat(opts.DataPath); err != nil || fi.IsDir() {
		return nil, registry.Errf("--data=<path to JSONL>", "data must be a JSONL file: %v", err)
	}

	// Determine variant from filename
	variant := determineVariant(opts.DataPath)

	// Load dataset
	examples, err := loadDataset(opts.DataPath)
	if err != nil {
		return nil, fmt.Errorf("load dataset: %w", err)
	}

	if len(examples) == 0 {
		return nil, errors.New("no examples loaded")
	}

	started := time.Now().Unix()
	total := 0
	correct := 0

	// Group by template for variance analysis
	templateGroups := make(map[int][]Example)
	for _, ex := range examples {
		templateGroups[ex.ID] = append(templateGroups[ex.ID], ex)
	}

	// Process each example
	for _, ex := range examples {
		if opts.Limit > 0 && total >= opts.Limit {
			break
		}

		// Extract gold answer
		goldAnswer := extractFinalAnswer(ex.Answer)
		if goldAnswer == "" {
			continue
		}

		// For now, use a simple baseline (can be replaced with actual model inference)
		// In production, this would call the model with the question
		prediction := solveWithBaseline(ex.Question)

		// Compare answers
		if normalizeAnswer(prediction) == normalizeAnswer(goldAnswer) {
			correct++
		}

		total++
	}

	finished := time.Now().Unix()

	if total == 0 {
		return nil, errors.New("no questions processed")
	}

	accuracy := float64(correct) / float64(total)

	// Calculate variance across instances
	variance := calculateVariance(templateGroups)

	sum := &registry.Summary{
		Task:       "gsm-symbolic",
		Model:      opts.Model,
		Count:      total,
		Metrics:    map[string]float64{"accuracy": accuracy, "variance": variance},
		StartedAt:  started,
		FinishedAt: finished,
		Details: map[string]any{
			"correct":       correct,
			"variant":       variant,
			"templates":     len(templateGroups),
			"avg_instances": float64(total) / float64(len(templateGroups)),
		},
	}

	return sum, nil
}

func init() {
	registry.Register(runner{})
}

// loadDataset loads GSM-Symbolic JSONL file
func loadDataset(path string) ([]Example, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var examples []Example
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}

		var ex Example
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			continue // Skip malformed lines
		}

		examples = append(examples, ex)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return examples, nil
}

// extractFinalAnswer extracts the numeric answer after ####
func extractFinalAnswer(answer string) string {
	// Match #### followed by optional whitespace and a number
	re := regexp.MustCompile(`####\s*(-?\d+\.?\d*)`)
	matches := re.FindStringSubmatch(answer)
	if len(matches) > 1 {
		return matches[1]
	}
	return ""
}

// normalizeAnswer normalizes numeric answers for comparison
func normalizeAnswer(s string) string {
	s = strings.TrimSpace(s)
	// Remove commas from numbers
	s = strings.ReplaceAll(s, ",", "")
	// Parse as float and format consistently
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		// Remove trailing zeros and decimal point if integer
		if f == float64(int64(f)) {
			return fmt.Sprintf("%d", int64(f))
		}
		return fmt.Sprintf("%.2f", f)
	}
	return s
}

// solveWithBaseline provides a simple baseline solver
// In production, this would be replaced with actual model inference
func solveWithBaseline(question string) string {
	// This is a placeholder - in real implementation, call the model
	// For now, return empty to demonstrate the infrastructure
	return ""
}

// determineVariant determines which GSM-Symbolic variant from filename
func determineVariant(path string) string {
	filename := strings.ToLower(path)
	if strings.Contains(filename, "p2") {
		return "GSM_symbolic_p2"
	} else if strings.Contains(filename, "p1") {
		return "GSM_symbolic_p1"
	}
	return "GSM_symbolic"
}

// calculateVariance calculates performance variance across template instances
func calculateVariance(templateGroups map[int][]Example) float64 {
	if len(templateGroups) == 0 {
		return 0
	}

	var variances []float64

	for _, instances := range templateGroups {
		if len(instances) < 2 {
			continue
		}

		// Calculate mean performance for this template
		// (This is simplified - in production, track actual correct/incorrect per instance)
		mean := 0.5 // Placeholder

		// Calculate variance
		var sumSquaredDiff float64
		for range instances {
			diff := 0.5 - mean // Placeholder
			sumSquaredDiff += diff * diff
		}

		variance := sumSquaredDiff / float64(len(instances))
		variances = append(variances, variance)
	}

	if len(variances) == 0 {
		return 0
	}

	// Return average variance across all templates
	var sum float64
	for _, v := range variances {
		sum += v
	}

	return sum / float64(len(variances))
}

// extractLastNumber extracts the last numeric value from model response
// This matches the paper's answer extraction heuristic
func extractLastNumber(response string) string {
	// Remove commas
	response = strings.ReplaceAll(response, ",", "")

	// Find the last number
	re := regexp.MustCompile(`-?\d+\.?\d*`)
	matches := re.FindAllString(response, -1)

	if len(matches) > 0 {
		return matches[len(matches)-1]
	}

	return ""
}

// CalculateAccuracyByTemplate calculates per-template accuracy
func CalculateAccuracyByTemplate(examples []Example, predictions map[string]string) map[int]float64 {
	templateAccuracy := make(map[int]float64)
	templateCounts := make(map[int]int)
	templateCorrect := make(map[int]int)

	for _, ex := range examples {
		key := fmt.Sprintf("%d_%d", ex.ID, ex.Instance)
		pred, ok := predictions[key]
		if !ok {
			continue
		}

		gold := extractFinalAnswer(ex.Answer)
		templateCounts[ex.ID]++

		if normalizeAnswer(pred) == normalizeAnswer(gold) {
			templateCorrect[ex.ID]++
		}
	}

	for templateID, count := range templateCounts {
		if count > 0 {
			templateAccuracy[templateID] = float64(templateCorrect[templateID]) / float64(count)
		}
	}

	return templateAccuracy
}

// CalculatePerformanceVariance calculates variance in model performance
// This is a key metric from the GSM-Symbolic paper
func CalculatePerformanceVariance(templateAccuracy map[int]float64) float64 {
	if len(templateAccuracy) == 0 {
		return 0
	}

	// Calculate mean accuracy
	var sum float64
	for _, acc := range templateAccuracy {
		sum += acc
	}
	mean := sum / float64(len(templateAccuracy))

	// Calculate variance
	var sumSquaredDiff float64
	for _, acc := range templateAccuracy {
		diff := acc - mean
		sumSquaredDiff += diff * diff
	}

	return math.Sqrt(sumSquaredDiff / float64(len(templateAccuracy)))
}
