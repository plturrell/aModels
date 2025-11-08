package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/plturrell/aModels/pkg/localai"
)

// ModelFusionFramework combines predictions from multiple models for better accuracy.
// Phase 8.2: Enhanced with domain-optimized weights for better domain-specific accuracy.
type ModelFusionFramework struct {
	logger                   *log.Logger
	useRelationalTransformer bool
	useSAPRPT                bool
	useGlove                 bool
	useLocalAI                bool
	localaiClient            *localai.Client
	localaiURL               string
	weights                  ModelWeights
	domainDetector           *DomainDetector         // Phase 8.2: Domain detector for domain-aware weights
	domainWeights            map[string]ModelWeights // Phase 8.2: domain_id -> optimized weights
}

// ModelWeights holds weights for ensemble predictions.
type ModelWeights struct {
	RelationalTransformer float64 `json:"relational_transformer"`
	SAPRPT                float64 `json:"sap_rpt"`
	Glove                 float64 `json:"glove"`
	LocalAI               float64 `json:"localai,omitempty"`
}

// DefaultModelWeights returns default weights for models.
func DefaultModelWeights() ModelWeights {
	return ModelWeights{
		RelationalTransformer: 0.3,
		SAPRPT:                0.3,
		Glove:                 0.15,
		LocalAI:               0.25,
	}
}

// NewModelFusionFramework creates a new model fusion framework.
func NewModelFusionFramework(logger *log.Logger) *ModelFusionFramework {
	localaiURL := os.Getenv("LOCALAI_URL")
	var domainDetector *DomainDetector
	var localaiClient *localai.Client
	useLocalAI := false
	
	if localaiURL != "" {
		domainDetector = NewDomainDetector(localaiURL, logger)
		// Initialize LocalAI client if URL is provided
		localaiClient = localai.NewClient(localaiURL)
		useLocalAI = true
		logger.Printf("LocalAI integration enabled: %s", localaiURL)
	}

	return &ModelFusionFramework{
		logger:                   logger,
		useRelationalTransformer: true,
		useSAPRPT:                os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true",
		useGlove:                 os.Getenv("USE_GLOVE_EMBEDDINGS") == "true",
		useLocalAI:                useLocalAI,
		localaiClient:             localaiClient,
		localaiURL:                localaiURL,
		weights:                  DefaultModelWeights(),
		domainDetector:           domainDetector,                // Phase 8.2: Domain detector
		domainWeights:            make(map[string]ModelWeights), // Phase 8.2: Domain-specific weights
	}
}

// FusionModelPrediction represents a prediction from a single model.
type FusionModelPrediction struct {
	ModelName  string         `json:"model_name"`
	Prediction any            `json:"prediction"`
	Confidence float64        `json:"confidence"`
	Embedding  []float32      `json:"embedding,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// FusedPrediction represents a combined prediction from multiple models.
type FusedPrediction struct {
	FinalPrediction  any                     `json:"final_prediction"`
	Confidence       float64                 `json:"confidence"`
	ModelPredictions []FusionModelPrediction `json:"model_predictions"`
	FusionMethod     string                  `json:"fusion_method"` // "weighted_average", "consensus", "majority_vote"
	Weights          ModelWeights            `json:"weights"`
}

// FusePredictions combines predictions from multiple models.
// Phase 8.2: Enhanced with domain-aware weight optimization.
func (mff *ModelFusionFramework) FusePredictions(
	ctx context.Context,
	predictions []FusionModelPrediction,
	fusionMethod string,
	domainID string, // Phase 8.2: Optional domain ID for domain-optimized weights
) (*FusedPrediction, error) {
	if len(predictions) == 0 {
		return nil, fmt.Errorf("no predictions to fuse")
	}

	// Phase 8.2: Use domain-specific weights if domainID provided
	if domainID != "" && mff.domainDetector != nil {
		if domainWeights, exists := mff.domainWeights[domainID]; exists {
			// Use cached domain-specific weights
			mff.weights = domainWeights
		} else {
			// Optimize weights for domain
			domainWeights = mff.optimizeWeightsForDomain(domainID, predictions)
			mff.domainWeights[domainID] = domainWeights
			mff.weights = domainWeights
		}
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

// optimizeWeightsForDomain optimizes model weights based on domain characteristics.
// Phase 8.2: Domain-specific weight optimization.
func (mff *ModelFusionFramework) optimizeWeightsForDomain(
	domainID string,
	predictions []FusionModelPrediction,
) ModelWeights {
	if mff.domainDetector == nil {
		return DefaultModelWeights()
	}

	// Get domain config
	mff.domainDetector.mu.RLock()
	domainConfig, exists := mff.domainDetector.domainConfigs[domainID]
	mff.domainDetector.mu.RUnlock()

	if !exists {
		return DefaultModelWeights()
	}

	// Optimize weights based on domain characteristics
	weights := DefaultModelWeights()

	// Check if domain is semantic-rich (has many keywords/tags)
	keywordCount := len(domainConfig.Keywords)
	tagCount := len(domainConfig.Tags)

	// Semantic-rich domains benefit more from SAP RPT
	if keywordCount > 5 || tagCount > 3 {
		weights.SAPRPT = 0.5
		weights.RelationalTransformer = 0.3
		weights.Glove = 0.2
	} else {
		// Less semantic domains benefit from RelationalTransformer
		weights.RelationalTransformer = 0.5
		weights.SAPRPT = 0.3
		weights.Glove = 0.2
	}

	// Adjust based on layer
	switch domainConfig.Layer {
	case "data":
		// Data layer benefits from RelationalTransformer
		weights.RelationalTransformer = 0.5
		weights.SAPRPT = 0.3
		weights.Glove = 0.2
	case "application", "business":
		// Application/business layers benefit from SAP RPT
		weights.SAPRPT = 0.5
		weights.RelationalTransformer = 0.3
		weights.Glove = 0.2
	default:
		// Default weights
	}

	mff.logger.Printf("Optimized weights for domain %s: RT=%.2f, SAP=%.2f, Glove=%.2f",
		domainID, weights.RelationalTransformer, weights.SAPRPT, weights.Glove)

	return weights
}

// weightedAverageFusion performs weighted average fusion of predictions.
func (mff *ModelFusionFramework) weightedAverageFusion(
	predictions []FusionModelPrediction,
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
	predictions []FusionModelPrediction,
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
		Confidence:       float64(totalWeight) / float64(len(predictions)),
		ModelPredictions: predictions,
	}, nil
}

// consensusFusion performs consensus-based fusion.
func (mff *ModelFusionFramework) consensusFusion(
	predictions []FusionModelPrediction,
) (*FusedPrediction, error) {
	// Find predictions that agree (within threshold)
	consensusThreshold := 0.8

	// Group predictions by similarity
	predictionGroups := make(map[string][]FusionModelPrediction)
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
	predictions []FusionModelPrediction,
) (*FusedPrediction, error) {
	// Count votes for each prediction
	voteCount := make(map[string]int)
	predictionMap := make(map[string]FusionModelPrediction)

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
	predictions := []FusionModelPrediction{}

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

	// Generate predictions from LocalAI models
	if mff.useLocalAI && mff.localaiClient != nil {
		// Try Phi-3.5-mini for general tasks
		if pred, err := mff.predictWithLocalAI(ctx, artifactType, artifactData, "phi-3.5-mini"); err == nil {
			predictions = append(predictions, *pred)
		} else {
			mff.logger.Printf("LocalAI (phi-3.5-mini) prediction failed: %v", err)
		}

		// Try Granite-4.0 for code/technical artifacts
		if artifactType == "code" || artifactType == "sql" || artifactType == "ddl" {
			if pred, err := mff.predictWithLocalAI(ctx, artifactType, artifactData, "granite-4.0"); err == nil {
				predictions = append(predictions, *pred)
			} else {
				mff.logger.Printf("LocalAI (granite-4.0) prediction failed: %v", err)
			}
		}

		// Try VaultGemma as fallback
		if len(predictions) == 0 {
			if pred, err := mff.predictWithLocalAI(ctx, artifactType, artifactData, "vaultgemma"); err == nil {
				predictions = append(predictions, *pred)
			}
		}
	}

	if len(predictions) == 0 {
		return nil, fmt.Errorf("no successful predictions")
	}

	// Fuse predictions
	return mff.FusePredictions(ctx, predictions, "weighted_average", "")
}

// predictWithRelationalTransformer generates prediction using RelationalTransformer.
func (mff *ModelFusionFramework) predictWithRelationalTransformer(
	ctx context.Context,
	artifactType string,
	artifactData map[string]any,
) (*FusionModelPrediction, error) {
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

	return &FusionModelPrediction{
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
) (*FusionModelPrediction, error) {
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

	return &FusionModelPrediction{
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
) (*FusionModelPrediction, error) {
	// Placeholder for Glove embedding
	// Would call Go Glove package
	return &FusionModelPrediction{
		ModelName:  "glove",
		Prediction: []float32{},
		Confidence: 0.7,
		Embedding:  []float32{},
	}, nil
}

// predictWithLocalAI generates prediction using LocalAI models.
func (mff *ModelFusionFramework) predictWithLocalAI(
	ctx context.Context,
	artifactType string,
	artifactData map[string]any,
	modelName string,
) (*FusionModelPrediction, error) {
	if mff.localaiClient == nil {
		return nil, fmt.Errorf("LocalAI client not initialized")
	}

	// Build prompt from artifact data
	prompt := fmt.Sprintf("Analyze this %s artifact and provide metadata extraction:\n\n", artifactType)
	
	// Add artifact data to prompt
	artifactJSON, err := json.MarshalIndent(artifactData, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal artifact data: %w", err)
	}
	prompt += string(artifactJSON)
	prompt += "\n\nProvide structured metadata extraction in JSON format."

	// Call LocalAI
	req := &localai.ChatRequest{
		Model: modelName,
		Messages: []localai.Message{
			{
				Role:    "system",
				Content: "You are a metadata extraction assistant. Analyze artifacts and extract structured metadata.",
			},
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   512,
		Temperature: 0.3,
	}

	resp, err := mff.localaiClient.ChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("LocalAI chat completion: %w", err)
	}

	content := resp.GetContent()
	if content == "" {
		return nil, fmt.Errorf("empty response from LocalAI")
	}

	// Parse response as JSON if possible
	var prediction any
	if err := json.Unmarshal([]byte(content), &prediction); err != nil {
		// If not JSON, use as string
		prediction = content
	}

	return &FusionModelPrediction{
		ModelName:  fmt.Sprintf("localai-%s", modelName),
		Prediction: prediction,
		Confidence: 0.85,
		Metadata: map[string]any{
			"model":      modelName,
			"raw_output": content,
		},
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
	case "localai-phi-3.5-mini", "localai-granite-4.0", "localai-vaultgemma":
		return mff.weights.LocalAI
	default:
		if mff.useLocalAI {
			return mff.weights.LocalAI
		}
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
	predictions []FusionModelPrediction,
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
	pred1, pred2 FusionModelPrediction,
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
