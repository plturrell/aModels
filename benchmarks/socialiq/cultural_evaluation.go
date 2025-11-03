package socialiq

import (
	"context"
	"fmt"
	"math"
	"sync"
)

// ============================================================================
// Cultural Robustness Evaluation
// Implements comprehensive cross-cultural testing
// ============================================================================

// CulturalRobustnessReport contains cultural evaluation results
type CulturalRobustnessReport struct {
	BaselineAccuracy      float64
	CrossCulturalAccuracy map[string]float64
	CulturalVariance      float64
	CulturalBias          map[string]float64
	RobustnessScore       float64
	SensitivityAnalysis   map[string][]float64
	RecommendedParameters CulturalParams
	DetailedResults       []CulturalTestResult
}

// CulturalTestResult contains results for a specific cultural setting
type CulturalTestResult struct {
	CulturalParams  CulturalParams
	Accuracy        float64
	F1Score         float64
	PerformanceDrop float64
	BiasMetrics     map[string]float64
}

// CulturalEvaluator performs comprehensive cultural evaluation
type CulturalEvaluator struct {
	coordinator *MetacognitiveCoordinator
	baseParams  CulturalParams
	testParams  []CulturalParams
	perfMonitor *PerformanceMonitor
	mu          sync.RWMutex
}

// NewCulturalEvaluator creates a new cultural evaluator
func NewCulturalEvaluator(baseParams CulturalParams) *CulturalEvaluator {
	return &CulturalEvaluator{
		baseParams:  baseParams,
		testParams:  generateCulturalTestSuite(),
		perfMonitor: NewPerformanceMonitor(),
	}
}

// generateCulturalTestSuite generates diverse cultural parameter sets
func generateCulturalTestSuite() []CulturalParams {
	suite := make([]CulturalParams, 0)

	// Western individualistic culture
	suite = append(suite, CulturalParams{
		Individualism:    0.9,
		PowerDistance:    0.2,
		UncertaintyAvoid: 0.3,
		Masculinity:      0.6,
		LongTermOrient:   0.5,
		Indulgence:       0.7,
		Context:          make(map[string]float64),
	})

	// Eastern collectivistic culture
	suite = append(suite, CulturalParams{
		Individualism:    0.2,
		PowerDistance:    0.7,
		UncertaintyAvoid: 0.6,
		Masculinity:      0.5,
		LongTermOrient:   0.8,
		Indulgence:       0.3,
		Context:          make(map[string]float64),
	})

	// Latin American culture
	suite = append(suite, CulturalParams{
		Individualism:    0.3,
		PowerDistance:    0.8,
		UncertaintyAvoid: 0.7,
		Masculinity:      0.4,
		LongTermOrient:   0.3,
		Indulgence:       0.6,
		Context:          make(map[string]float64),
	})

	// Scandinavian culture
	suite = append(suite, CulturalParams{
		Individualism:    0.7,
		PowerDistance:    0.1,
		UncertaintyAvoid: 0.2,
		Masculinity:      0.2,
		LongTermOrient:   0.6,
		Indulgence:       0.7,
		Context:          make(map[string]float64),
	})

	// Middle Eastern culture
	suite = append(suite, CulturalParams{
		Individualism:    0.3,
		PowerDistance:    0.7,
		UncertaintyAvoid: 0.6,
		Masculinity:      0.6,
		LongTermOrient:   0.4,
		Indulgence:       0.3,
		Context:          make(map[string]float64),
	})

	// African culture
	suite = append(suite, CulturalParams{
		Individualism:    0.2,
		PowerDistance:    0.6,
		UncertaintyAvoid: 0.5,
		Masculinity:      0.5,
		LongTermOrient:   0.3,
		Indulgence:       0.5,
		Context:          make(map[string]float64),
	})

	return suite
}

// EvaluateCulturalRobustness performs comprehensive cultural evaluation
func (ce *CulturalEvaluator) EvaluateCulturalRobustness(ctx context.Context, dataset *QADataset) (*CulturalRobustnessReport, error) {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	report := &CulturalRobustnessReport{
		CrossCulturalAccuracy: make(map[string]float64),
		CulturalBias:          make(map[string]float64),
		SensitivityAnalysis:   make(map[string][]float64),
		DetailedResults:       make([]CulturalTestResult, 0),
	}

	// Evaluate baseline (original cultural parameters)
	baseCoordinator := NewMetacognitiveCoordinator(ce.baseParams)
	baseAccuracy, err := ce.evaluateWithParams(ctx, baseCoordinator, dataset)
	if err != nil {
		return nil, fmt.Errorf("baseline evaluation: %w", err)
	}
	report.BaselineAccuracy = baseAccuracy

	// Evaluate across different cultural settings
	accuracies := make([]float64, 0)

	for i, params := range ce.testParams {
		cultureName := fmt.Sprintf("Culture_%d", i)

		// Create coordinator with test parameters
		testCoordinator := NewMetacognitiveCoordinator(params)

		// Evaluate
		accuracy, err := ce.evaluateWithParams(ctx, testCoordinator, dataset)
		if err != nil {
			continue
		}

		report.CrossCulturalAccuracy[cultureName] = accuracy
		accuracies = append(accuracies, accuracy)

		// Compute performance drop
		perfDrop := (baseAccuracy - accuracy) / baseAccuracy

		// Compute bias metrics
		biasMetrics := ce.computeBiasMetrics(baseAccuracy, accuracy, params)

		// Store detailed result
		report.DetailedResults = append(report.DetailedResults, CulturalTestResult{
			CulturalParams:  params,
			Accuracy:        accuracy,
			F1Score:         accuracy, // Simplified
			PerformanceDrop: perfDrop,
			BiasMetrics:     biasMetrics,
		})
	}

	// Compute cultural variance
	report.CulturalVariance = ce.computeVariance(accuracies)

	// Compute robustness score (inverse of variance, normalized)
	report.RobustnessScore = 1.0 / (1.0 + report.CulturalVariance)

	// Perform sensitivity analysis
	report.SensitivityAnalysis = ce.performSensitivityAnalysis(ctx, dataset)

	// Find optimal parameters
	report.RecommendedParameters = ce.findOptimalParameters(report.DetailedResults)

	// Compute cultural bias
	report.CulturalBias = ce.computeCulturalBias(report.DetailedResults)

	return report, nil
}

// evaluateWithParams evaluates model with specific cultural parameters
func (ce *CulturalEvaluator) evaluateWithParams(ctx context.Context, coordinator *MetacognitiveCoordinator, dataset *QADataset) (float64, error) {
	correct := 0
	total := 0

	for _, question := range dataset.Questions {
		input := MultimodalInput{
			VideoID:  question.VideoID,
			Question: question.Question,
			Answers:  question.Answers,
		}

		result, err := coordinator.ReasonAndAct(ctx, input)
		if err != nil {
			continue
		}

		if result.PredictedIndex == question.CorrectIndex {
			correct++
		}
		total++
	}

	if total == 0 {
		return 0, fmt.Errorf("no questions evaluated")
	}

	return float64(correct) / float64(total), nil
}

// computeVariance computes variance of accuracies
func (ce *CulturalEvaluator) computeVariance(accuracies []float64) float64 {
	if len(accuracies) == 0 {
		return 0
	}

	// Compute mean
	mean := 0.0
	for _, acc := range accuracies {
		mean += acc
	}
	mean /= float64(len(accuracies))

	// Compute variance
	variance := 0.0
	for _, acc := range accuracies {
		diff := acc - mean
		variance += diff * diff
	}
	variance /= float64(len(accuracies))

	return variance
}

// computeBiasMetrics computes bias metrics for a cultural setting
func (ce *CulturalEvaluator) computeBiasMetrics(baseAccuracy, testAccuracy float64, params CulturalParams) map[string]float64 {
	metrics := make(map[string]float64)

	// Performance gap
	metrics["performance_gap"] = baseAccuracy - testAccuracy

	// Relative performance
	if baseAccuracy > 0 {
		metrics["relative_performance"] = testAccuracy / baseAccuracy
	}

	// Cultural distance (simplified)
	baseParams := ce.baseParams
	distance := 0.0
	distance += math.Abs(params.Individualism - baseParams.Individualism)
	distance += math.Abs(params.PowerDistance - baseParams.PowerDistance)
	distance += math.Abs(params.UncertaintyAvoid - baseParams.UncertaintyAvoid)
	distance += math.Abs(params.Masculinity - baseParams.Masculinity)
	distance += math.Abs(params.LongTermOrient - baseParams.LongTermOrient)
	distance += math.Abs(params.Indulgence - baseParams.Indulgence)
	metrics["cultural_distance"] = distance / 6.0

	return metrics
}

// performSensitivityAnalysis analyzes sensitivity to each cultural parameter
func (ce *CulturalEvaluator) performSensitivityAnalysis(ctx context.Context, dataset *QADataset) map[string][]float64 {
	sensitivity := make(map[string][]float64)

	parameters := []string{
		"Individualism",
		"PowerDistance",
		"UncertaintyAvoid",
		"Masculinity",
		"LongTermOrient",
		"Indulgence",
	}

	// Test each parameter independently
	for _, param := range parameters {
		values := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
		accuracies := make([]float64, len(values))

		for i, value := range values {
			// Create params with only this parameter varied
			testParams := ce.baseParams
			switch param {
			case "Individualism":
				testParams.Individualism = value
			case "PowerDistance":
				testParams.PowerDistance = value
			case "UncertaintyAvoid":
				testParams.UncertaintyAvoid = value
			case "Masculinity":
				testParams.Masculinity = value
			case "LongTermOrient":
				testParams.LongTermOrient = value
			case "Indulgence":
				testParams.Indulgence = value
			}

			coordinator := NewMetacognitiveCoordinator(testParams)
			accuracy, err := ce.evaluateWithParams(ctx, coordinator, dataset)
			if err != nil {
				accuracies[i] = 0
			} else {
				accuracies[i] = accuracy
			}
		}

		sensitivity[param] = accuracies
	}

	return sensitivity
}

// findOptimalParameters finds the best cultural parameters
func (ce *CulturalEvaluator) findOptimalParameters(results []CulturalTestResult) CulturalParams {
	if len(results) == 0 {
		return ce.baseParams
	}

	// Find parameters with highest accuracy
	bestAccuracy := 0.0
	var bestParams CulturalParams

	for _, result := range results {
		if result.Accuracy > bestAccuracy {
			bestAccuracy = result.Accuracy
			bestParams = result.CulturalParams
		}
	}

	return bestParams
}

// computeCulturalBias computes overall cultural bias
func (ce *CulturalEvaluator) computeCulturalBias(results []CulturalTestResult) map[string]float64 {
	bias := make(map[string]float64)

	if len(results) == 0 {
		return bias
	}

	// Compute average performance drop
	avgDrop := 0.0
	for _, result := range results {
		avgDrop += result.PerformanceDrop
	}
	avgDrop /= float64(len(results))
	bias["average_performance_drop"] = avgDrop

	// Compute max performance drop
	maxDrop := 0.0
	for _, result := range results {
		if result.PerformanceDrop > maxDrop {
			maxDrop = result.PerformanceDrop
		}
	}
	bias["max_performance_drop"] = maxDrop

	// Compute fairness score (1 - variance in accuracies)
	accuracies := make([]float64, len(results))
	for i, result := range results {
		accuracies[i] = result.Accuracy
	}
	variance := ce.computeVariance(accuracies)
	bias["fairness_score"] = 1.0 / (1.0 + variance)

	return bias
}

// PrintReport prints the cultural robustness report
func (ce *CulturalEvaluator) PrintReport(report *CulturalRobustnessReport) {
	fmt.Println("\n=== Cultural Robustness Report ===")
	fmt.Printf("\nBaseline Accuracy: %.2f%%\n", report.BaselineAccuracy*100)
	fmt.Printf("Robustness Score: %.4f\n", report.RobustnessScore)
	fmt.Printf("Cultural Variance: %.6f\n", report.CulturalVariance)

	fmt.Println("\n--- Cross-Cultural Performance ---")
	for culture, accuracy := range report.CrossCulturalAccuracy {
		drop := (report.BaselineAccuracy - accuracy) / report.BaselineAccuracy * 100
		fmt.Printf("%s: %.2f%% (drop: %.1f%%)\n", culture, accuracy*100, drop)
	}

	fmt.Println("\n--- Cultural Bias Metrics ---")
	for metric, value := range report.CulturalBias {
		fmt.Printf("%s: %.4f\n", metric, value)
	}

	fmt.Println("\n--- Sensitivity Analysis ---")
	for param, values := range report.SensitivityAnalysis {
		fmt.Printf("%s: ", param)
		for _, val := range values {
			fmt.Printf("%.2f%% ", val*100)
		}
		fmt.Println()
	}

	fmt.Println("\n--- Recommended Parameters ---")
	fmt.Printf("Individualism: %.2f\n", report.RecommendedParameters.Individualism)
	fmt.Printf("PowerDistance: %.2f\n", report.RecommendedParameters.PowerDistance)
	fmt.Printf("UncertaintyAvoid: %.2f\n", report.RecommendedParameters.UncertaintyAvoid)
	fmt.Printf("Masculinity: %.2f\n", report.RecommendedParameters.Masculinity)
	fmt.Printf("LongTermOrient: %.2f\n", report.RecommendedParameters.LongTermOrient)
	fmt.Printf("Indulgence: %.2f\n", report.RecommendedParameters.Indulgence)
}

// CulturalBayesUpdate implements Algorithm 6 from the paper
func CulturalBayesUpdate(prior CulturalParams, evidence SocialEvidence, learningRate float64) CulturalParams {
	updated := prior

	// Update each parameter based on evidence
	if evidence.ObservedIndividualism >= 0 {
		updated.Individualism = (1-learningRate)*prior.Individualism +
			learningRate*evidence.ObservedIndividualism
	}

	if evidence.ObservedPowerDistance >= 0 {
		updated.PowerDistance = (1-learningRate)*prior.PowerDistance +
			learningRate*evidence.ObservedPowerDistance
	}

	if evidence.ObservedUncertaintyAvoid >= 0 {
		updated.UncertaintyAvoid = (1-learningRate)*prior.UncertaintyAvoid +
			learningRate*evidence.ObservedUncertaintyAvoid
	}

	// Normalize to [0, 1]
	updated.Individualism = math.Max(0, math.Min(1, updated.Individualism))
	updated.PowerDistance = math.Max(0, math.Min(1, updated.PowerDistance))
	updated.UncertaintyAvoid = math.Max(0, math.Min(1, updated.UncertaintyAvoid))

	return updated
}

// SocialEvidence represents observed cultural evidence
type SocialEvidence struct {
	ObservedIndividualism    float64
	ObservedPowerDistance    float64
	ObservedUncertaintyAvoid float64
	ObservedMasculinity      float64
	ObservedLongTermOrient   float64
	ObservedIndulgence       float64
	Confidence               float64
}
