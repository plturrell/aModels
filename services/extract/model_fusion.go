package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
)

// ModelFusionFramework combines predictions from multiple models for better accuracy.
type ModelFusionFramework struct {
	logger            *log.Logger
	useRelationalTransformer bool
	useSAPRPT         bool
	useGlove          bool
	weights           ModelWeights
}

// ModelWeights holds weights for ensemble predictions.
type ModelWeights struct {
	RelationalTransformer float64 `json:"relational_transformer"`
	SAPRPT                float64 `json:"sap_rpt"`
	Glove                 float64 `json:"glove"`
}

// DefaultModelWeights returns default weights for models.
func DefaultModelWeights() ModelWeights {
	return ModelWeights{
		RelationalTransformer: 0.4,
		SAPRPT:               0.4,
		Glove:                0.2,
	}
}

// NewModelFusionFramework creates a new model fusion framework.
func NewModelFusionFramework(logger *log.Logger) *ModelFusionFramework {
	return &ModelFusionFramework{
		logger:                  logger,
		useRelationalTransformer: true,
		useSAPRPT:               os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true",
		useGlove:                os.Getenv("USE_GLOVE_EMBEDDINGS") == "true",
		weights:                 DefaultModelWeights(),
	}
}

// ModelPrediction represents a prediction from a single model.
type ModelPrediction struct {
	ModelName  string            `json:"model_name"`
	Prediction any               `json:"prediction"`
	Confidence float64           `json:"confidence"`
	Embedding  []float32         `json:"embedding,omitempty"`
	Metadata   map[string]any    `json:"metadata,omitempty"`
}

// FusedPrediction represents a combined prediction from multiple models.
type FusedPrediction struct {
	FinalPrediction any               `json:"final_prediction"`
	Confidence      float64           `json:"confidence"`
	ModelPredictions []ModelPrediction `json:"model_predictions"`
	FusionMethod    string            `json:"fusion_method"` // "weighted_average", "consensus", "majority_vote"
	Weights         ModelWeights      `json:"weights"`
}

// FusePredictions combines predictions from multiple models.
func (mff *ModelFusionFramework) FusePredictions(
	ctx context.Context,
	predictions []ModelPrediction,
	fusionMethod string,
) (*FusedPrediction, error) {
	if len(predictions) == 0 {
		return nil, fmt.Errorf("no predictions to fuse")
	}

	// Select fusion method
	if fusionMethod == "" {
		fusionMethod = "weighted_average"
	}

	var fusedPrediction *FusedPrediction
	var err error

	switch fusionMethod {
	case "weighted_average":
		fusedPrediction, err = mff.weightedAverageFusion(predictions)
	case "consensus":
		fusedPrediction, err = mff.consensusFusion(predictions)
	case "majority_vote":
		fusedPrediction, err = mff.majorityVoteFusion(predictions)
	default:
		return nil, fmt.Errorf("unknown fusion method: %s", fusionMethod)
	}

	if err != nil {
		return nil, fmt.Errorf("fusion failed: %w", err)
	}

	fusedPrediction.FusionMethod = fusionMethod
	fusedPrediction.Weights = mff.weights

	return fusedPrediction, nil
}

// weightedAverageFusion performs weighted average fusion of predictions.
func (mff *ModelFusionFramework) weightedAverageFusion(
	predictions []ModelPrediction,
) (*FusedPrediction, error) {
	// For embedding predictions, compute weighted average
	if len(predictions) > 0 && predictions[0].Embedding != nil {
		return mff.fuseEmbeddings(predictions)
	}

	// For classification/regression predictions, use weighted average of confidences
	totalWeight := 0.0
	weightedSum := 0.0
	mostConfident := predictions[0]

	for _, pred := range predictions {
		weight := mff.getModelWeight(pred.ModelName) * pred.Confidence
		totalWeight += weight
		weightedSum += weight * pred.Confidence

		if pred.Confidence > mostConfident.Confidence {
			mostConfident = pred
		}
	}

	avgConfidence := weightedSum / totalWeight
	if totalWeight == 0 {
		avgConfidence = mostConfident.Confidence
	}

	return &FusedPrediction{
		FinalPrediction:  mostConfident.Prediction,
		Confidence:       avgConfidence,
		ModelPredictions: predictions,
	}, nil
}

// fuseEmbeddings fuses embeddings from multiple models.
func (mff *ModelFusionFramework) fuseEmbeddings(
	predictions []ModelPrediction,
) (*FusedPrediction, error) {
	if len(predictions) == 0 {
		return nil, fmt.Errorf("no predictions to fuse")
	}

	// Get embedding dimensions from first prediction
	embeddingDim := len(predictions[0].Embedding)
	if embeddingDim == 0 {
		return nil, fmt.Errorf("empty embedding")
	}

	// Weighted average of embeddings
	fusedEmbedding := make([]float32, embeddingDim)
	totalWeight := 0.0

	for _, pred := range predictions {
		if len(pred.Embedding) != embeddingDim {
			continue // Skip mismatched dimensions
		}

		weight := mff.getModelWeight(pred.ModelName) * pred.Confidence
		totalWeight += weight

		for i := range pred.Embedding {
			fusedEmbedding[i] += float32(weight) * pred.Embedding[i]
		}
	}

	// Normalize
	if totalWeight > 0 {
		for i := range fusedEmbedding {
			fusedEmbedding[i] /= float32(totalWeight)
		}
	}

	return &FusedPrediction{
		FinalPrediction:  fusedEmbedding,
		Confidence:        float64(totalWeight) / float64(len(predictions)),
		ModelPredictions: predictions,
	}, nil
}

// consensusFusion performs consensus-based fusion.
func (mff *ModelFusionFramework) consensusFusion(
	predictions []ModelPrediction,
) (*FusedPrediction, error) {
	// Find predictions that agree (within threshold)
	consensusThreshold := 0.8

	// Group predictions by similarity
	predictionGroups := make(map[string][]ModelPrediction)
	for _, pred := range predictions {
		predStr := fmt.Sprintf("%v", pred.Prediction)
		predictionGroups[predStr] = append(predictionGroups[predStr], pred)
	}

	// Find largest consensus group
	maxGroupSize := 0
	consensusPrediction := predictions[0]

	for _, group := range predictionGroups {
		if len(group) > maxGroupSize {
			maxGroupSize = len(group)
			consensusPrediction = group[0]
		}
	}

	consensusRatio := float64(maxGroupSize) / float64(len(predictions))
	confidence := consensusRatio

	if consensusRatio >= consensusThreshold {
		confidence = min(1.0, consensusRatio+0.1)
	}

	return &FusedPrediction{
		FinalPrediction:  consensusPrediction.Prediction,
		Confidence:       confidence,
		ModelPredictions: predictions,
	}, nil
}

// majorityVoteFusion performs majority vote fusion.
func (mff *ModelFusionFramework) majorityVoteFusion(
	predictions []ModelPrediction,
) (*FusedPrediction, error) {
	// Count votes for each prediction
	voteCount := make(map[string]int)
	predictionMap := make(map[string]ModelPrediction)

	for _, pred := range predictions {
		predStr := fmt.Sprintf("%v", pred.Prediction)
		voteCount[predStr]++
		if _, exists := predictionMap[predStr]; !exists {
			predictionMap[predStr] = pred
		}
	}

	// Find majority vote
	maxVotes := 0
	majorityPredStr := ""
	for predStr, votes := range voteCount {
		if votes > maxVotes {
			maxVotes = votes
			majorityPredStr = predStr
		}
	}

	majorityPred := predictionMap[majorityPredStr]
	confidence := float64(maxVotes) / float64(len(predictions))

	return &FusedPrediction{
		FinalPrediction:  majorityPred.Prediction,
		Confidence:       confidence,
		ModelPredictions: predictions,
	}, nil
}

// PredictWithMultipleModels generates predictions from multiple models and fuses them.
func (mff *ModelFusionFramework) PredictWithMultipleModels(
	ctx context.Context,
	artifactType string,
	artifactData map[string]any,
) (*FusedPrediction, error) {
	predictions := []ModelPrediction{}

	// Generate prediction from RelationalTransformer
	if mff.useRelationalTransformer {
		pred, err := mff.predictWithRelationalTransformer(ctx, artifactType, artifactData)
		if err == nil {
			predictions = append(predictions, *pred)
		} else {
			mff.logger.Printf("RelationalTransformer prediction failed: %v", err)
		}
	}

	// Generate prediction from SAP-RPT
	if mff.useSAPRPT {
		pred, err := mff.predictWithSAPRPT(ctx, artifactType, artifactData)
		if err == nil {
			predictions = append(predictions, *pred)
		} else {
			mff.logger.Printf("SAP-RPT prediction failed: %v", err)
		}
	}

	// Generate prediction from Glove
	if mff.useGlove {
		pred, err := mff.predictWithGlove(ctx, artifactType, artifactData)
		if err == nil {
			predictions = append(predictions, *pred)
		} else {
			mff.logger.Printf("Glove prediction failed: %v", err)
		}
	}

	if len(predictions) == 0 {
		return nil, fmt.Errorf("no successful predictions")
	}

	// Fuse predictions
	return mff.FusePredictions(ctx, predictions, "weighted_average")
}

// predictWithRelationalTransformer generates prediction using RelationalTransformer.
func (mff *ModelFusionFramework) predictWithRelationalTransformer(
	ctx context.Context,
	artifactType string,
	artifactData map[string]any,
) (*ModelPrediction, error) {
	// Call embedding script
	cmd := exec.CommandContext(ctx, "python3", "./scripts/embed.py",
		"--artifact-type", artifactType,
	)

	// Add artifact data as JSON
	artifactJSON, err := json.Marshal(artifactData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal artifact data: %w", err)
	}

	cmd.Args = append(cmd.Args, "--artifact-data", string(artifactJSON))

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
	}

	return &ModelPrediction{
		ModelName:  "relational_transformer",
		Prediction: embedding,
		Confidence: 0.8,
		Embedding:  embedding,
	}, nil
}

// predictWithSAPRPT generates prediction using SAP-RPT.
func (mff *ModelFusionFramework) predictWithSAPRPT(
	ctx context.Context,
	artifactType string,
	artifactData map[string]any,
) (*ModelPrediction, error) {
	// Extract text for embedding
	text := ""
	if tableName, ok := artifactData["table_name"].(string); ok {
		text = tableName
	}

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embed_sap_rpt.py",
		"--artifact-type", "text",
		"--text", text,
	)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to generate SAP-RPT embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
	}

	return &ModelPrediction{
		ModelName:  "sap_rpt",
		Prediction: embedding,
		Confidence: 0.85,
		Embedding:  embedding,
	}, nil
}

// predictWithGlove generates prediction using Glove.
func (mff *ModelFusionFramework) predictWithGlove(
	ctx context.Context,
	artifactType string,
	artifactData map[string]any,
) (*ModelPrediction, error) {
	// Placeholder for Glove embedding
	// Would call Go Glove package
	return &ModelPrediction{
		ModelName:  "glove",
		Prediction: []float32{},
		Confidence: 0.7,
		Embedding:  []float32{},
	}, nil
}

// getModelWeight returns the weight for a model.
func (mff *ModelFusionFramework) getModelWeight(modelName string) float64 {
	switch modelName {
	case "relational_transformer":
		return mff.weights.RelationalTransformer
	case "sap_rpt":
		return mff.weights.SAPRPT
	case "glove":
		return mff.weights.Glove
	default:
		return 0.33 // Default equal weight
	}
}

// SelectBestModel selects the best model for a given task type.
func (mff *ModelFusionFramework) SelectBestModel(taskType string) string {
	// Model selection based on task type
	switch taskType {
	case "table_classification":
		return "sap_rpt" // SAP-RPT is best for classification
	case "column_type_inference":
		return "relational_transformer" // RelationalTransformer is best for structure
	case "semantic_similarity":
		return "sap_rpt" // SAP-RPT is best for semantic
	case "sequence_analysis":
		return "relational_transformer" // RelationalTransformer is best for sequences
	default:
		return "weighted_ensemble" // Use ensemble for unknown tasks
	}
}

// ValidateCrossModel validates predictions across models.
func (mff *ModelFusionFramework) ValidateCrossModel(
	predictions []ModelPrediction,
	threshold float64,
) bool {
	if len(predictions) < 2 {
		return true // Single prediction is always valid
	}

	// Check if predictions agree (similarity > threshold)
	for i := 0; i < len(predictions)-1; i++ {
		for j := i + 1; j < len(predictions); j++ {
			similarity := mff.calculatePredictionSimilarity(predictions[i], predictions[j])
			if similarity < threshold {
				return false // Predictions disagree
			}
		}
	}

	return true // All predictions agree
}

// calculatePredictionSimilarity calculates similarity between two predictions.
func (mff *ModelFusionFramework) calculatePredictionSimilarity(
	pred1, pred2 ModelPrediction,
) float64 {
	// If both have embeddings, use cosine similarity
	if len(pred1.Embedding) > 0 && len(pred2.Embedding) > 0 {
		return cosineSimilarity(pred1.Embedding, pred2.Embedding)
	}

	// Otherwise, compare predictions directly
	if pred1.Prediction == pred2.Prediction {
		return 1.0
	}

	return 0.0
}

// Helper functions
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct float64
	var normA, normB float64

	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

func sqrt(x float64) float64 {
	if x == 0 {
		return 0
	}
	if x < 0 {
		return 0
	}
	result := x
	for i := 0; i < 10; i++ {
		result = 0.5 * (result + x/result)
	}
	return result
}

