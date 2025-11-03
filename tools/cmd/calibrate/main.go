package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"ai_benchmarks/internal/localai"
)

func main() {
	// Command line flags
	localaiURL := flag.String("url", "http://localhost:8080", "LocalAI server URL")
	apiKey := flag.String("key", "", "API key for LocalAI")
	modelName := flag.String("model", "phi-2", "Model name to calibrate")
	task := flag.String("task", "boolq", "Benchmark task (boolq, hellaswag, piqa, socialiqa, triviaqa, arc)")
	dataPath := flag.String("data", "", "Path to test data (JSONL format)")
	outputPath := flag.String("output", "calibration_result.json", "Path to save calibration results")
	maxSamples := flag.Int("samples", 100, "Maximum number of samples to test")
	flag.Parse()

	if *dataPath == "" {
		log.Fatal("Error: --data flag is required")
	}

	// Create LocalAI client
	client := localai.NewClient(*localaiURL, *apiKey)

	// Verify connection
	models, err := client.ListModels()
	if err != nil {
		log.Fatalf("Failed to connect to LocalAI: %v", err)
	}

	fmt.Printf("✓ Connected to LocalAI\n")
	fmt.Printf("✓ Available models: %d\n", len(models.Data))

	// Load test data
	testData, err := loadTestData(*dataPath, *task)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}

	fmt.Printf("✓ Loaded %d test cases\n", len(testData))

	// Configure calibration
	config := localai.CalibrationConfig{
		ModelName:        *modelName,
		Task:             *task,
		TemperatureRange: []float64{0.1, 0.3, 0.5, 0.7, 0.9},
		TopPRange:        []float64{0.7, 0.85, 0.95},
		MaxSamples:       *maxSamples,
	}

	calibrator := localai.NewCalibrator(client, config)

	// Run calibration
	fmt.Printf("\n🔧 Starting calibration for %s on %s task...\n", *modelName, *task)
	result, err := calibrator.CalibrateParams(testData)
	if err != nil {
		log.Fatalf("Calibration failed: %v", err)
	}

	// Display results
	fmt.Printf("\n📊 Calibration Results:\n")
	fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
	fmt.Printf("Model:              %s\n", result.ModelName)
	fmt.Printf("Task:               %s\n", result.Task)
	fmt.Printf("Accuracy:           %.2f%%\n", result.Accuracy*100)
	fmt.Printf("Samples Tested:     %d\n", result.Samples)
	fmt.Printf("Optimal Temperature: %.2f\n", result.OptimalTemp)
	fmt.Printf("Optimal Top-P:      %.2f\n", result.OptimalTopP)

	if result.F1Score > 0 {
		fmt.Printf("F1 Score:           %.4f\n", result.F1Score)
	}

	fmt.Printf("\n📈 Parameter Scores:\n")
	for param, score := range result.ParameterScores {
		fmt.Printf("  %s: %.2f%%\n", param, score*100)
	}

	fmt.Printf("\n💡 Recommendations:\n")
	for _, rec := range result.Recommendations {
		fmt.Printf("  %s\n", rec)
	}
	fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	// Save results
	if err := calibrator.SaveCalibration(result, *outputPath); err != nil {
		log.Fatalf("Failed to save calibration: %v", err)
	}

	fmt.Printf("\n✓ Calibration results saved to: %s\n", *outputPath)
}

// loadTestData loads test data based on task type
func loadTestData(path, task string) ([]localai.TestCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var testCases []localai.TestCase

	switch task {
	case "boolq":
		testCases, err = parseBoolQData(data)
	case "hellaswag":
		testCases, err = parseHellaSwagData(data)
	case "piqa":
		testCases, err = parsePIQAData(data)
	case "socialiqa":
		testCases, err = parseSocialIQAData(data)
	case "triviaqa":
		testCases, err = parseTriviaQAData(data)
	default:
		return nil, fmt.Errorf("unsupported task: %s", task)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to parse data: %w", err)
	}

	return testCases, nil
}

// parseBoolQData parses BoolQ format data
func parseBoolQData(data []byte) ([]localai.TestCase, error) {
	var items []struct {
		Question string `json:"question"`
		Answer   bool   `json:"answer"`
		Passage  string `json:"passage"`
	}

	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}

	testCases := make([]localai.TestCase, len(items))
	for i, item := range items {
		expected := "no"
		if item.Answer {
			expected = "yes"
		}
		testCases[i] = localai.TestCase{
			Prompt:         fmt.Sprintf("Passage: %s\n\nQuestion: %s", item.Passage, item.Question),
			ExpectedAnswer: expected,
		}
	}

	return testCases, nil
}

// parseHellaSwagData parses HellaSwag format data
func parseHellaSwagData(data []byte) ([]localai.TestCase, error) {
	var items []struct {
		Context string   `json:"context"`
		Endings []string `json:"endings"`
		Label   int      `json:"label"`
	}

	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}

	testCases := make([]localai.TestCase, len(items))
	for i, item := range items {
		testCases[i] = localai.TestCase{
			Prompt:         item.Context,
			ExpectedAnswer: fmt.Sprintf("%d", item.Label),
			Choices:        item.Endings,
		}
	}

	return testCases, nil
}

// parsePIQAData parses PIQA format data
func parsePIQAData(data []byte) ([]localai.TestCase, error) {
	var items []struct {
		Goal  string `json:"goal"`
		Sol1  string `json:"sol1"`
		Sol2  string `json:"sol2"`
		Label int    `json:"label"`
	}

	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}

	testCases := make([]localai.TestCase, len(items))
	for i, item := range items {
		testCases[i] = localai.TestCase{
			Prompt:         item.Goal,
			ExpectedAnswer: fmt.Sprintf("%d", item.Label),
			Choices:        []string{item.Sol1, item.Sol2},
		}
	}

	return testCases, nil
}

// parseSocialIQAData parses SocialIQA format data
func parseSocialIQAData(data []byte) ([]localai.TestCase, error) {
	var items []struct {
		Question string `json:"question"`
		AnswerA  string `json:"answerA"`
		AnswerB  string `json:"answerB"`
		AnswerC  string `json:"answerC"`
		Label    string `json:"label"`
	}

	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}

	testCases := make([]localai.TestCase, len(items))
	for i, item := range items {
		testCases[i] = localai.TestCase{
			Prompt:         item.Question,
			ExpectedAnswer: item.Label,
			Choices:        []string{item.AnswerA, item.AnswerB, item.AnswerC},
		}
	}

	return testCases, nil
}

// parseTriviaQAData parses TriviaQA format data
func parseTriviaQAData(data []byte) ([]localai.TestCase, error) {
	var items []struct {
		Question string `json:"question"`
		Answer   string `json:"answer"`
	}

	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}

	testCases := make([]localai.TestCase, len(items))
	for i, item := range items {
		testCases[i] = localai.TestCase{
			Prompt:         item.Question,
			ExpectedAnswer: item.Answer,
		}
	}

	return testCases, nil
}
