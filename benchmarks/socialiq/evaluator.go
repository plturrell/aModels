package socialiq

import (
	"context"
	"fmt"
	"math"
)

// Evaluator evaluates model performance on Social-IQ 2.0
type Evaluator struct {
	loader *DataLoader
}

// NewEvaluator creates a new evaluator
func NewEvaluator(loader *DataLoader) *Evaluator {
	return &Evaluator{
		loader: loader,
	}
}

// Evaluate evaluates a model on a dataset
func (e *Evaluator) Evaluate(ctx context.Context, model ModelInterface, dataset *QADataset) (*EvaluationMetrics, error) {
	if len(dataset.Questions) == 0 {
		return nil, fmt.Errorf("empty dataset")
	}

	metrics := &EvaluationMetrics{
		CategoryAccuracy: make(map[string]float64),
		TypeAccuracy:     make(map[string]float64),
		TotalQuestions:   len(dataset.Questions),
	}

	// Track correct predictions by category and type
	categoryCorrect := make(map[string]int)
	categoryTotal := make(map[string]int)
	typeCorrect := make(map[string]int)
	typeTotal := make(map[string]int)

	// Evaluate each question
	for _, question := range dataset.Questions {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Get video metadata
		meta, err := e.loader.LoadVideoMetadata(question.VideoID)
		if err != nil {
			continue // Skip unavailable videos
		}

		// Create multimodal input
		input := MultimodalInput{
			VideoID:    question.VideoID,
			Question:   question.Question,
			Answers:    question.Answers,
			Transcript: "", // Would load actual transcript
		}

		// Get prediction
		prediction, err := model.Predict(input)
		if err != nil {
			continue
		}

		// Check if correct
		isCorrect := prediction.PredictedIndex == question.CorrectIndex

		if isCorrect {
			metrics.CorrectPredictions++
			categoryCorrect[meta.Category]++
			typeCorrect[question.QuestionType]++
		}

		categoryTotal[meta.Category]++
		typeTotal[question.QuestionType]++
	}

	// Calculate overall metrics
	metrics.Accuracy = float64(metrics.CorrectPredictions) / float64(metrics.TotalQuestions)

	// Calculate precision, recall, F1 (treating as binary: correct vs incorrect)
	truePositives := float64(metrics.CorrectPredictions)
	falsePositives := float64(metrics.TotalQuestions - metrics.CorrectPredictions)
	falseNegatives := float64(metrics.TotalQuestions - metrics.CorrectPredictions)

	if truePositives+falsePositives > 0 {
		metrics.Precision = truePositives / (truePositives + falsePositives)
	}
	if truePositives+falseNegatives > 0 {
		metrics.Recall = truePositives / (truePositives + falseNegatives)
	}
	if metrics.Precision+metrics.Recall > 0 {
		metrics.F1Score = 2 * (metrics.Precision * metrics.Recall) / (metrics.Precision + metrics.Recall)
	}

	// Calculate per-category accuracy
	for category, total := range categoryTotal {
		if total > 0 {
			metrics.CategoryAccuracy[category] = float64(categoryCorrect[category]) / float64(total)
		}
	}

	// Calculate per-type accuracy
	for qType, total := range typeTotal {
		if total > 0 {
			metrics.TypeAccuracy[qType] = float64(typeCorrect[qType]) / float64(total)
		}
	}

	return metrics, nil
}

// EvaluateCrossValidation performs k-fold cross-validation
func (e *Evaluator) EvaluateCrossValidation(ctx context.Context, model ModelInterface, dataset *QADataset, k int) (*EvaluationMetrics, error) {
	if k <= 1 {
		return nil, fmt.Errorf("k must be greater than 1")
	}

	foldSize := len(dataset.Questions) / k
	var allMetrics []*EvaluationMetrics

	for i := 0; i < k; i++ {
		// Create train/test split
		testStart := i * foldSize
		testEnd := testStart + foldSize
		if i == k-1 {
			testEnd = len(dataset.Questions)
		}

		trainQuestions := append([]Question{}, dataset.Questions[:testStart]...)
		trainQuestions = append(trainQuestions, dataset.Questions[testEnd:]...)
		testQuestions := dataset.Questions[testStart:testEnd]

		trainDataset := &QADataset{
			Questions: trainQuestions,
			Split:     "train",
		}
		testDataset := &QADataset{
			Questions: testQuestions,
			Split:     "test",
		}

		// Train model
		if err := model.Train(*trainDataset); err != nil {
			return nil, fmt.Errorf("train fold %d: %w", i, err)
		}

		// Evaluate
		metrics, err := e.Evaluate(ctx, model, testDataset)
		if err != nil {
			return nil, fmt.Errorf("evaluate fold %d: %w", i, err)
		}

		allMetrics = append(allMetrics, metrics)
	}

	// Average metrics across folds
	return e.averageMetrics(allMetrics), nil
}

// averageMetrics computes average metrics across multiple evaluations
func (e *Evaluator) averageMetrics(metrics []*EvaluationMetrics) *EvaluationMetrics {
	if len(metrics) == 0 {
		return &EvaluationMetrics{}
	}

	avg := &EvaluationMetrics{
		CategoryAccuracy: make(map[string]float64),
		TypeAccuracy:     make(map[string]float64),
	}

	for _, m := range metrics {
		avg.Accuracy += m.Accuracy
		avg.F1Score += m.F1Score
		avg.Precision += m.Precision
		avg.Recall += m.Recall
		avg.TotalQuestions += m.TotalQuestions
		avg.CorrectPredictions += m.CorrectPredictions

		for cat, acc := range m.CategoryAccuracy {
			avg.CategoryAccuracy[cat] += acc
		}
		for typ, acc := range m.TypeAccuracy {
			avg.TypeAccuracy[typ] += acc
		}
	}

	n := float64(len(metrics))
	avg.Accuracy /= n
	avg.F1Score /= n
	avg.Precision /= n
	avg.Recall /= n

	for cat := range avg.CategoryAccuracy {
		avg.CategoryAccuracy[cat] /= n
	}
	for typ := range avg.TypeAccuracy {
		avg.TypeAccuracy[typ] /= n
	}

	return avg
}

// CompareModels compares multiple models on the same dataset
func (e *Evaluator) CompareModels(ctx context.Context, models map[string]ModelInterface, dataset *QADataset) (map[string]*EvaluationMetrics, error) {
	results := make(map[string]*EvaluationMetrics)

	for name, model := range models {
		metrics, err := e.Evaluate(ctx, model, dataset)
		if err != nil {
			return nil, fmt.Errorf("evaluate %s: %w", name, err)
		}
		results[name] = metrics
	}

	return results, nil
}

// CalculateConfidenceInterval calculates 95% confidence interval for accuracy
func (e *Evaluator) CalculateConfidenceInterval(metrics *EvaluationMetrics) (float64, float64) {
	n := float64(metrics.TotalQuestions)
	p := metrics.Accuracy

	// Standard error
	se := math.Sqrt(p * (1 - p) / n)

	// 95% confidence interval (z = 1.96)
	margin := 1.96 * se

	lower := math.Max(0, p-margin)
	upper := math.Min(1, p+margin)

	return lower, upper
}

// PrintMetrics prints evaluation metrics in a readable format
func (e *Evaluator) PrintMetrics(metrics *EvaluationMetrics) {
	fmt.Println("=== Evaluation Metrics ===")
	fmt.Printf("Accuracy:  %.2f%% (%d/%d)\n",
		metrics.Accuracy*100,
		metrics.CorrectPredictions,
		metrics.TotalQuestions)
	fmt.Printf("Precision: %.4f\n", metrics.Precision)
	fmt.Printf("Recall:    %.4f\n", metrics.Recall)
	fmt.Printf("F1 Score:  %.4f\n", metrics.F1Score)

	lower, upper := e.CalculateConfidenceInterval(metrics)
	fmt.Printf("95%% CI:    [%.2f%%, %.2f%%]\n", lower*100, upper*100)

	if len(metrics.CategoryAccuracy) > 0 {
		fmt.Println("\n=== Per-Category Accuracy ===")
		for cat, acc := range metrics.CategoryAccuracy {
			fmt.Printf("%s: %.2f%%\n", cat, acc*100)
		}
	}

	if len(metrics.TypeAccuracy) > 0 {
		fmt.Println("\n=== Per-Type Accuracy ===")
		for typ, acc := range metrics.TypeAccuracy {
			fmt.Printf("%s: %.2f%%\n", typ, acc*100)
		}
	}
}

// AnalyzeErrors performs error analysis on predictions
func (e *Evaluator) AnalyzeErrors(ctx context.Context, model ModelInterface, dataset *QADataset) (map[string][]Question, error) {
	errors := make(map[string][]Question)

	for _, question := range dataset.Questions {
		input := MultimodalInput{
			VideoID:  question.VideoID,
			Question: question.Question,
			Answers:  question.Answers,
		}

		prediction, err := model.Predict(input)
		if err != nil {
			continue
		}

		if prediction.PredictedIndex != question.CorrectIndex {
			errorType := fmt.Sprintf("predicted_%d_actual_%d",
				prediction.PredictedIndex,
				question.CorrectIndex)
			errors[errorType] = append(errors[errorType], question)
		}
	}

	return errors, nil
}
