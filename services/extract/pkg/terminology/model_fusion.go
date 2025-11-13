package terminology

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/plturrell/aModels/pkg/localai"
	"github.com/plturrell/aModels/services/extract/pkg/utils"
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
	domainDetector           interface{}         // Phase 8.2: Domain detector for domain-aware weights
	domainWeights            map[string]ModelWeights // Phase 8.2: domain_id -> optimized weights
	batchSize                int                    // Batch size for LocalAI predictions
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

// createPooledHTTPClient creates an HTTP client with connection pooling for LocalAI
func createPooledHTTPClient(logger *log.Logger) *http.Client {
	// Get pool configuration from environment variables with defaults
	maxIdleConns := 10
	if val := os.Getenv("LOCALAI_HTTP_POOL_SIZE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxIdleConns = parsed
		}
	}

	maxIdleConnsPerHost := 5
	if val := os.Getenv("LOCALAI_HTTP_MAX_IDLE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxIdleConnsPerHost = parsed
		}
	}

	transport := &http.Transport{
		MaxIdleConns:        maxIdleConns,
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		IdleConnTimeout:     90 * time.Second,
		DisableKeepAlives:    false,
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   60 * time.Second,
	}

	if logger != nil {
		logger.Printf("LocalAI HTTP client pool configured: maxIdleConns=%d, maxIdleConnsPerHost=%d", maxIdleConns, maxIdleConnsPerHost)
	}

	return client
}

// RetryWithExponentialBackoff retries a function with exponential backoff
func RetryWithExponentialBackoff(ctx context.Context, logger *log.Logger, maxAttempts int, initialBackoff time.Duration, maxBackoff time.Duration, fn func() error) error {
	var lastErr error
	backoff := initialBackoff

	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			// Check context cancellation before waiting
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}

			if logger != nil {
				logger.Printf("Retrying LocalAI call (attempt %d/%d) after %v", attempt+1, maxAttempts, backoff)
			}
		}

		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Exponential backoff: double the backoff time, but cap at maxBackoff
		backoff = time.Duration(float64(backoff) * 2)
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}

	return fmt.Errorf("failed after %d attempts: %w", maxAttempts, lastErr)
}

// IsRetryableError determines if an error is retryable
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())
	// Retry on network errors, timeouts, and 5xx status codes
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "connection") ||
		strings.Contains(errStr, "network") ||
		strings.Contains(errStr, "status=5") ||
		strings.Contains(errStr, "status=502") ||
		strings.Contains(errStr, "status=503") ||
		strings.Contains(errStr, "status=504")
}

// NewModelFusionFramework creates a new model fusion framework.
func NewModelFusionFramework(logger *log.Logger) *ModelFusionFramework {
	localaiURL := os.Getenv("LOCALAI_URL")
	var domainDetector interface{}
	var localaiClient *localai.Client
	useLocalAI := false
	
	if localaiURL != "" {
		// TODO: Initialize domain detector when implementation is available
		// domainDetector = NewDomainDetector(localaiURL, logger)
		domainDetector = nil
		// Initialize LocalAI client with connection pooling if URL is provided
		pooledHTTPClient := createPooledHTTPClient(logger)
		localaiClient = localai.NewClientWithHTTPClient(localaiURL, pooledHTTPClient)
		useLocalAI = true
		logger.Printf("LocalAI integration enabled: %s (with connection pooling)", localaiURL)
	}

	// Get batch size from environment (default: 5)
	batchSize := 5
	if val := os.Getenv("LOCALAI_BATCH_SIZE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			batchSize = parsed
		}
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
		batchSize:                batchSize,
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

	// Get domain config - use type assertion for interface{}
	var domainConfig struct {
		Keywords []string
		Tags     []string
		Layer    string
	}
	exists := false
	
	// Type assert domainDetector to check if it has Config method
	if detector, ok := mff.domainDetector.(interface{ Config(string) (struct {
		Keywords []string
		Tags     []string
		Layer    string
	}, bool) }); ok {
		domainConfig, exists = detector.Config(domainID)
	}

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
		// Use math.Min for float64 comparison
		confidence = math.Min(1.0, consensusRatio+0.1)
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

// BatchPredictWithLocalAI generates predictions for multiple artifacts in a batch
// This reduces the number of HTTP calls and improves throughput
func (mff *ModelFusionFramework) BatchPredictWithLocalAI(
	ctx context.Context,
	requests []struct {
		ArtifactType string
		ArtifactData  map[string]any
		ModelName     string
	},
) ([]*FusionModelPrediction, error) {
	if mff.localaiClient == nil {
		return nil, fmt.Errorf("LocalAI client not initialized")
	}

	if len(requests) == 0 {
		return nil, fmt.Errorf("no requests provided")
	}

	// Process in batches
	results := make([]*FusionModelPrediction, 0, len(requests))
	
	for i := 0; i < len(requests); i += mff.batchSize {
		end := i + mff.batchSize
		if end > len(requests) {
			end = len(requests)
		}

		batch := requests[i:end]
		batchResults, err := mff.processBatch(ctx, batch)
		if err != nil {
			// Log error but continue with other batches
			mff.logger.Printf("Batch prediction error: %v", err)
			// Add nil results for failed batch
			for range batch {
				results = append(results, nil)
			}
			continue
		}

		results = append(results, batchResults...)
	}

	return results, nil
}

// processBatch processes a batch of prediction requests concurrently
func (mff *ModelFusionFramework) processBatch(
	ctx context.Context,
	requests []struct {
		ArtifactType string
		ArtifactData  map[string]any
		ModelName     string
	},
) ([]*FusionModelPrediction, error) {
	type result struct {
		index int
		pred  *FusionModelPrediction
		err   error
	}

	results := make([]*FusionModelPrediction, len(requests))
	resultChan := make(chan result, len(requests))

	// Process all requests concurrently
	for i, req := range requests {
		go func(idx int, r struct {
			ArtifactType string
			ArtifactData  map[string]any
			ModelName     string
		}) {
			pred, err := mff.predictWithLocalAI(ctx, r.ArtifactType, r.ArtifactData, r.ModelName)
			resultChan <- result{index: idx, pred: pred, err: err}
		}(i, req)
	}

	// Collect results
	var lastErr error
	for i := 0; i < len(requests); i++ {
		res := <-resultChan
		if res.err != nil {
			lastErr = res.err
			results[res.index] = nil
		} else {
			results[res.index] = res.pred
		}
	}

	if lastErr != nil && len(results) == 0 {
		return nil, lastErr
	}

	return results, nil
}

// predictWithRelationalTransformer generates prediction using RelationalTransformer.
func (mff *ModelFusionFramework) predictWithRelationalTransformer(
	ctx context.Context,
	artifactType string,
	artifactData map[string]any,
) (*FusionModelPrediction, error) {
	// Call embedding script
	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py",
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

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed_sap_rpt.py",
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

// predictWithLocalAI generates prediction using LocalAI models with retry logic.
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

	// Call LocalAI with retry logic
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

	// Get retry configuration from environment
	maxRetries := 3
	if val := os.Getenv("LOCALAI_RETRY_MAX_ATTEMPTS"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxRetries = parsed
		}
	}

	initialBackoff := 100 * time.Millisecond
	maxBackoff := 2 * time.Second

	var resp *localai.ChatResponse
		retryErr := RetryWithExponentialBackoff(ctx, mff.logger, maxRetries, initialBackoff, maxBackoff, func() error {
		var err error
		resp, err = mff.localaiClient.ChatCompletion(ctx, req)
		if err != nil {
			// Only retry on retryable errors
			if IsRetryableError(err) {
				return err
			}
			// Non-retryable errors should be returned immediately
			return fmt.Errorf("non-retryable error: %w", err)
		}
		return nil
	})

	if retryErr != nil {
		return nil, fmt.Errorf("LocalAI chat completion: %w", retryErr)
	}

	if resp == nil {
		return nil, fmt.Errorf("empty response from LocalAI")
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
		return utils.CosineSimilarity(pred1.Embedding, pred2.Embedding)
	}

	// Otherwise, compare predictions directly
	if pred1.Prediction == pred2.Prediction {
		return 1.0
	}

	return 0.0
}

// Helper functions
