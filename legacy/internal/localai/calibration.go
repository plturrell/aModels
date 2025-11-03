package localai

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// CalibrationResult stores the results of model calibration
type CalibrationResult struct {
	ModelName       string             `json:"model_name"`
	Task            string             `json:"task"`
	Accuracy        float64            `json:"accuracy"`
	F1Score         float64            `json:"f1_score,omitempty"`
	Samples         int                `json:"samples"`
	OptimalTemp     float64            `json:"optimal_temperature"`
	OptimalTopP     float64            `json:"optimal_top_p"`
	ParameterScores map[string]float64 `json:"parameter_scores"`
	Recommendations []string           `json:"recommendations"`
}

// CalibrationConfig defines calibration parameters
type CalibrationConfig struct {
	ModelName        string
	Task             string
	TemperatureRange []float64 // e.g., [0.1, 0.5, 0.7, 1.0]
	TopPRange        []float64 // e.g., [0.7, 0.85, 0.95]
	MaxSamples       int       // Number of samples to test
}

// Calibrator handles model calibration using benchmark data
type Calibrator struct {
	Client *Client
	Config CalibrationConfig
}

// NewCalibrator creates a new calibrator
func NewCalibrator(client *Client, config CalibrationConfig) *Calibrator {
	return &Calibrator{
		Client: client,
		Config: config,
	}
}

// CalibrateParams tests different parameter combinations and finds optimal settings
func (cal *Calibrator) CalibrateParams(testData []TestCase) (*CalibrationResult, error) {
	bestAccuracy := 0.0
	bestTemp := 0.7
	bestTopP := 0.9
	paramScores := make(map[string]float64)

	// Test temperature variations
	for _, temp := range cal.Config.TemperatureRange {
		accuracy, err := cal.testWithParams(testData, temp, 0.9)
		if err != nil {
			return nil, fmt.Errorf("failed to test temperature %.2f: %w", temp, err)
		}

		paramKey := fmt.Sprintf("temp_%.2f", temp)
		paramScores[paramKey] = accuracy

		if accuracy > bestAccuracy {
			bestAccuracy = accuracy
			bestTemp = temp
		}
	}

	// Test top_p variations with best temperature
	for _, topP := range cal.Config.TopPRange {
		accuracy, err := cal.testWithParams(testData, bestTemp, topP)
		if err != nil {
			return nil, fmt.Errorf("failed to test top_p %.2f: %w", topP, err)
		}

		paramKey := fmt.Sprintf("top_p_%.2f", topP)
		paramScores[paramKey] = accuracy

		if accuracy > bestAccuracy {
			bestAccuracy = accuracy
			bestTopP = topP
		}
	}

	// Generate recommendations
	recommendations := cal.generateRecommendations(bestTemp, bestTopP, bestAccuracy)

	result := &CalibrationResult{
		ModelName:       cal.Config.ModelName,
		Task:            cal.Config.Task,
		Accuracy:        bestAccuracy,
		Samples:         len(testData),
		OptimalTemp:     bestTemp,
		OptimalTopP:     bestTopP,
		ParameterScores: paramScores,
		Recommendations: recommendations,
	}

	return result, nil
}

// TestCase represents a single test case for calibration
type TestCase struct {
	Prompt         string
	ExpectedAnswer string
	Choices        []string // For MCQ tasks
}

// testWithParams runs inference with specific parameters
func (cal *Calibrator) testWithParams(testData []TestCase, temp float64, topP float64) (float64, error) {
	correct := 0
	total := len(testData)

	if cal.Config.MaxSamples > 0 && total > cal.Config.MaxSamples {
		total = cal.Config.MaxSamples
	}

	for i := 0; i < total; i++ {
		test := testData[i]

		req := CompletionRequest{
			Model:       cal.Config.ModelName,
			Prompt:      test.Prompt,
			Temperature: temp,
			TopP:        topP,
			MaxTokens:   100,
		}

		resp, err := cal.Client.Complete(req)
		if err != nil {
			return 0, fmt.Errorf("inference failed: %w", err)
		}

		if len(resp.Choices) > 0 {
			answer := resp.Choices[0].Text
			if cal.checkAnswer(answer, test.ExpectedAnswer, test.Choices) {
				correct++
			}
		}
	}

	return float64(correct) / float64(total), nil
}

// checkAnswer validates if the model's answer is correct
func (cal *Calibrator) checkAnswer(modelAnswer, expected string, choices []string) bool {
	// Simple exact match for now
	// TODO: Add fuzzy matching, semantic similarity
	return modelAnswer == expected
}

// generateRecommendations creates actionable recommendations based on calibration
func (cal *Calibrator) generateRecommendations(temp, topP, accuracy float64) []string {
	var recommendations []string

	if accuracy < 0.5 {
		recommendations = append(recommendations, "⚠️ Low accuracy detected. Consider fine-tuning the model or using a larger model.")
	}

	if temp < 0.3 {
		recommendations = append(recommendations, "🔹 Low temperature optimal - model performs best with deterministic outputs")
	} else if temp > 0.8 {
		recommendations = append(recommendations, "🔹 High temperature optimal - task benefits from creative/diverse outputs")
	}

	if topP < 0.8 {
		recommendations = append(recommendations, "🔹 Low top_p optimal - focused sampling improves accuracy")
	}

	if accuracy > 0.8 {
		recommendations = append(recommendations, "✅ Excellent performance - current parameters are well-calibrated")
	} else if accuracy > 0.6 {
		recommendations = append(recommendations, "✓ Good performance - minor tuning may improve results")
	}

	return recommendations
}

// SaveCalibration saves calibration results to a JSON file
func (cal *Calibrator) SaveCalibration(result *CalibrationResult, filepath string) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal calibration: %w", err)
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write calibration file: %w", err)
	}

	return nil
}

// LoadCalibration loads calibration results from a JSON file
func LoadCalibration(filepath string) (*CalibrationResult, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read calibration file: %w", err)
	}

	var result CalibrationResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal calibration: %w", err)
	}

	return &result, nil
}

// CalculateF1Score computes F1 score for binary classification
func CalculateF1Score(truePositives, falsePositives, falseNegatives int) float64 {
	if truePositives == 0 {
		return 0.0
	}

	precision := float64(truePositives) / float64(truePositives+falsePositives)
	recall := float64(truePositives) / float64(truePositives+falseNegatives)

	if precision+recall == 0 {
		return 0.0
	}

	return 2 * (precision * recall) / (precision + recall)
}

// CalculatePerplexity computes perplexity for language modeling tasks
func CalculatePerplexity(logProbs []float64) float64 {
	if len(logProbs) == 0 {
		return math.Inf(1)
	}

	sumLogProbs := 0.0
	for _, lp := range logProbs {
		sumLogProbs += lp
	}

	avgLogProb := sumLogProbs / float64(len(logProbs))
	return math.Exp(-avgLogProb)
}
