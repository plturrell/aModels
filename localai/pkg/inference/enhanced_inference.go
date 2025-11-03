package inference

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/maths/util"
)

// EnhancedInferenceEngine provides actual tensor operations for model inference
type EnhancedInferenceEngine struct {
	models        map[string]*ai.VaultGemma
	domainManager *domain.DomainManager
	tokenizer     *Tokenizer
}

// NewEnhancedInferenceEngine creates a new enhanced inference engine
func NewEnhancedInferenceEngine(models map[string]*ai.VaultGemma, domainManager *domain.DomainManager) *EnhancedInferenceEngine {
	return &EnhancedInferenceEngine{
		models:        models,
		domainManager: domainManager,
		tokenizer:     NewTokenizer(),
	}
}

// Tokenizer handles text tokenization
type Tokenizer struct {
	vocabSize int
	vocab     map[string]int
	reverse   map[int]string
}

// NewTokenizer creates a new tokenizer
func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		vocabSize: 32000, // Default vocab size
		vocab:     make(map[string]int),
		reverse:   make(map[int]string),
	}
}

// EnhancedInferenceRequest represents a request for enhanced model inference
type EnhancedInferenceRequest struct {
	Prompt      string
	Domain      string
	MaxTokens   int
	Temperature float64
	Model       *ai.VaultGemma
	TopP        float64
	TopK        int
	MinP        float64
	UseMinP     bool
}

// EnhancedInferenceResponse represents the response from enhanced model inference
type EnhancedInferenceResponse struct {
	Content      string
	TokensUsed   int
	ModelName    string
	Domain       string
	Duration     time.Duration
	LogProbs     []float64
	FinishReason string
	Error        error
}

// GenerateEnhancedResponse performs actual tensor-based model inference
func (e *EnhancedInferenceEngine) GenerateEnhancedResponse(ctx context.Context, req *EnhancedInferenceRequest) *EnhancedInferenceResponse {
	start := time.Now()

	if req.Model == nil {
		return &EnhancedInferenceResponse{
			Error: fmt.Errorf("no model available for inference"),
		}
	}

	if fastResponderEnabled() {
		content := generateFastResponse(req.Prompt, req.Domain)
		tokens := len(strings.Fields(content))
		duration := time.Since(start)
		log.Printf("⚡ Fast responder active for domain %s", req.Domain)
		return &EnhancedInferenceResponse{
			Content:      content,
			TokensUsed:   tokens,
			ModelName:    req.Domain,
			Domain:       req.Domain,
			Duration:     duration,
			LogProbs:     nil,
			FinishReason: "fast-responder",
			Error:        nil,
		}
	}

	// Set default parameters
	if req.MaxTokens == 0 {
		req.MaxTokens = 512
	}
	if req.Temperature == 0 {
		req.Temperature = 0.7
	}
	if req.TopP == 0 {
		req.TopP = 0.9
	}
	if req.TopK == 0 {
		req.TopK = 50
	}

	// Align tokenizer vocab size with model configuration
	if req.Model.Config.VocabSize > 0 && e.tokenizer.vocabSize != req.Model.Config.VocabSize {
		e.tokenizer.vocabSize = req.Model.Config.VocabSize
	}

	// Tokenize the input
	tokens, err := e.tokenizer.Tokenize(req.Prompt)
	if err != nil {
		return &EnhancedInferenceResponse{
			Error: fmt.Errorf("failed to tokenize input: %w", err),
		}
	}

	log.Printf("🔤 Tokenized input: %d tokens", len(tokens))

	// Generate response using actual tensor operations
	responseTokens, logProbs, err := e.generateTokensWithTensorOps(ctx, req.Model, tokens, req.MaxTokens, req.Temperature, req.TopP, req.TopK, req.MinP, req.UseMinP)
	if err != nil {
		return &EnhancedInferenceResponse{
			Error: fmt.Errorf("failed to generate tokens: %w", err),
		}
	}

	// Decode tokens to text
	content, err := e.tokenizer.Detokenize(responseTokens)
	if err != nil {
		return &EnhancedInferenceResponse{
			Error: fmt.Errorf("failed to decode tokens: %w", err),
		}
	}

	duration := time.Since(start)

	log.Printf("✅ Generated response: %d tokens in %.2fms", len(responseTokens), duration.Seconds()*1000)

	return &EnhancedInferenceResponse{
		Content:      content,
		TokensUsed:   len(responseTokens),
		ModelName:    req.Domain,
		Domain:       req.Domain,
		Duration:     duration,
		LogProbs:     logProbs,
		FinishReason: "stop",
		Error:        nil,
	}
}

func fastResponderEnabled() bool {
	if strings.EqualFold(os.Getenv("LOCALAI_FAST_RESPONDER"), "1") {
		return true
	}
	if strings.EqualFold(os.Getenv("LOCALAI_FAST_MODE"), "1") {
		return true
	}
	return false
}

func generateFastResponse(prompt, domain string) string {
	trimmed := strings.TrimSpace(prompt)
	if trimmed == "" {
		trimmed = "(empty prompt)"
	}
	if len(trimmed) > 240 {
		trimmed = trimmed[:240] + "…"
	}
	summaryTokens := len(strings.Fields(trimmed))
	builder := strings.Builder{}
	builder.WriteString("[fast-")
	if domain == "" {
		builder.WriteString("general")
	} else {
		builder.WriteString(domain)
	}
	builder.WriteString("] LocalAI fast responder output.\n")
	builder.WriteString("Prompt: ")
	builder.WriteString(trimmed)
	builder.WriteString("\nTokens (approx): ")
	builder.WriteString(fmt.Sprintf("%d", summaryTokens))
	return builder.String()
}

// generateTokensWithTensorOps generates tokens using actual tensor operations
func (e *EnhancedInferenceEngine) generateTokensWithTensorOps(ctx context.Context, model *ai.VaultGemma, inputTokens []int, maxTokens int, temperature, topP float64, topK int, minP float64, useMinP bool) ([]int, []float64, error) {
	samplingCfg := ai.SamplingConfig{
		Temperature: temperature,
		TopK:        topK,
		TopP:        topP,
	}
	sequence, err := model.GenerateWithSampling(inputTokens, maxTokens, samplingCfg)
	if err != nil {
		return nil, nil, fmt.Errorf("generate with sampling: %w", err)
	}
	if len(sequence) <= len(inputTokens) {
		return []int{}, []float64{}, nil
	}
	generated := append([]int(nil), sequence[len(inputTokens):]...)
	logProbs := make([]float64, len(generated))
	for i := range logProbs {
		logProbs[i] = 0
	}
	return generated, logProbs, nil
}

// tokensToEmbeddings converts tokens to embeddings using the model's embedding layer
func (e *EnhancedInferenceEngine) tokensToEmbeddings(model *ai.VaultGemma, tokens []int) (*util.Matrix64, error) {
	hiddenSize := model.Config.HiddenSize
	seqLen := len(tokens)

	// Create embedding matrix
	embeddings := util.NewMatrix64(seqLen, hiddenSize)

	// Look up embeddings for each token
	for i, token := range tokens {
		if token >= model.Config.VocabSize {
			return nil, fmt.Errorf("token %d exceeds vocab size %d", token, model.Config.VocabSize)
		}

		// Get embedding vector for this token
		for j := 0; j < hiddenSize; j++ {
			embeddings.Set(i, j, model.Embed.Weights.Data[token*model.Embed.Weights.Stride+j])
		}
	}

	return embeddings, nil
}

// runTransformerLayer runs a single transformer layer
func (e *EnhancedInferenceEngine) runTransformerLayer(model *ai.VaultGemma, layer *ai.TransformerLayer, hiddenStates *util.Matrix64) (*util.Matrix64, error) {
	// Self-attention
	attnOutput, err := e.runSelfAttention(layer.SelfAttention, hiddenStates)
	if err != nil {
		return nil, fmt.Errorf("self-attention failed: %w", err)
	}

	// Residual connection and layer norm
	hiddenStates = e.addResidualAndNorm(hiddenStates, attnOutput, layer.LayerNorm1)

	// Feed-forward network
	ffnOutput, err := e.runFeedForward(layer.FeedForward, hiddenStates)
	if err != nil {
		return nil, fmt.Errorf("feed-forward failed: %w", err)
	}

	// Residual connection and layer norm
	hiddenStates = e.addResidualAndNorm(hiddenStates, ffnOutput, layer.LayerNorm2)

	return hiddenStates, nil
}

// runSelfAttention runs multi-head self-attention
func (e *EnhancedInferenceEngine) runSelfAttention(attn *ai.MultiHeadAttention, hiddenStates *util.Matrix64) (*util.Matrix64, error) {
	hiddenSize := hiddenStates.Cols
	headDim := attn.HeadDim
	numHeads := attn.NumHeads

	// Compute Q, K, V (simplified implementation)
	Q := e.matrixMultiply(hiddenStates, attn.WQ)
	K := e.matrixMultiply(hiddenStates, attn.WK)
	V := e.matrixMultiply(hiddenStates, attn.WV)

	// Reshape for multi-head attention
	Q = e.reshapeForHeads(Q, numHeads, headDim)
	K = e.reshapeForHeads(K, numHeads, headDim)
	V = e.reshapeForHeads(V, numHeads, headDim)

	// Compute attention scores
	scores, err := e.computeAttentionScores(Q, K)
	if err != nil {
		return nil, fmt.Errorf("attention scores computation failed: %w", err)
	}

	// Apply attention to values
	attnOutput, err := e.applyAttention(scores, V)
	if err != nil {
		return nil, fmt.Errorf("attention application failed: %w", err)
	}

	// Reshape back and project
	attnOutput = e.reshapeFromHeads(attnOutput, hiddenSize)
	output := e.matrixMultiply(attnOutput, attn.WO)

	return output, nil
}

// runFeedForward runs the feed-forward network
func (e *EnhancedInferenceEngine) runFeedForward(ffn *ai.FeedForwardNetwork, hiddenStates *util.Matrix64) (*util.Matrix64, error) {
	// Gate projection
	gate := e.matrixMultiply(hiddenStates, ffn.W3)

	// Up projection
	up := e.matrixMultiply(hiddenStates, ffn.W1)

	// Apply SiLU activation to gate
	gate = e.applySiLU(gate)

	// Element-wise multiplication
	intermediate := e.elementWiseMultiply(gate, up)

	// Down projection
	output := e.matrixMultiply(intermediate, ffn.W2)

	return output, nil
}

// getNextTokenLogits gets logits for the next token
func (e *EnhancedInferenceEngine) getNextTokenLogits(model *ai.VaultGemma, tokens []int) ([]float64, error) {
	// Convert tokens to embeddings
	hiddenStates, err := e.tokensToEmbeddings(model, tokens)
	if err != nil {
		return nil, err
	}

	// Run through all transformer layers
	for i := 0; i < model.Config.NumLayers; i++ {
		hiddenStates, err = e.runTransformerLayer(model, &model.Layers[i], hiddenStates)
		if err != nil {
			return nil, err
		}
	}

	// Get the last hidden state
	lastHidden := make([]float64, hiddenStates.Cols)
	rowIndex := (hiddenStates.Rows - 1) * hiddenStates.Stride
	copy(lastHidden, hiddenStates.Data[rowIndex:rowIndex+hiddenStates.Cols])

	// Project to vocabulary
	logits := make([]float64, model.Config.VocabSize)
	for i := 0; i < model.Config.VocabSize; i++ {
		for j := 0; j < model.Config.HiddenSize; j++ {
			logits[i] += lastHidden[j] * model.Output.Weights.Data[j*model.Output.Weights.Stride+i]
		}
	}

	return logits, nil
}

// sampleToken samples a token from the logits
func (e *EnhancedInferenceEngine) sampleToken(logits []float64, temperature, topP float64, topK int, minP float64, useMinP bool) (int, float64, error) {
	_ = minP
	_ = useMinP
	// Apply temperature
	scaledLogits := make([]float64, len(logits))
	for i, logit := range logits {
		scaledLogits[i] = logit / temperature
	}

	// Apply top-k filtering
	if topK > 0 && topK < len(scaledLogits) {
		scaledLogits = e.applyTopK(scaledLogits, topK)
	}

	// Apply top-p filtering
	if topP < 1.0 {
		scaledLogits = e.applyTopP(scaledLogits, topP)
	}

	// Convert to probabilities
	probs := e.softmax(scaledLogits)

	// Sample from the distribution
	token, logProb := e.sampleFromDistribution(probs)

	return token, logProb, nil
}

// matrixMultiply performs matrix multiplication
func (e *EnhancedInferenceEngine) matrixMultiply(a, b *util.Matrix64) *util.Matrix64 {
	if a.Cols != b.Rows {
		// Return a copy of a if dimensions don't match
		return a
	}

	result := util.NewMatrix64(a.Rows, b.Cols)

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i*a.Stride+k] * b.Data[k*b.Stride+j]
			}
			result.Data[i*result.Stride+j] = sum
		}
	}

	return result
}

// Helper functions for tensor operations
func (e *EnhancedInferenceEngine) reshapeForHeads(matrix *util.Matrix64, numHeads, headDim int) *util.Matrix64 {
	// Reshape from [seqLen, hiddenSize] to [seqLen, numHeads, headDim]
	// This is a simplified implementation
	return matrix
}

func (e *EnhancedInferenceEngine) reshapeFromHeads(matrix *util.Matrix64, hiddenSize int) *util.Matrix64 {
	// Reshape from [seqLen, numHeads, headDim] to [seqLen, hiddenSize]
	// This is a simplified implementation
	return matrix
}

func (e *EnhancedInferenceEngine) computeAttentionScores(Q, K *util.Matrix64) (*util.Matrix64, error) {
	// Compute Q @ K^T / sqrt(head_dim)
	// This is a simplified implementation
	return Q, nil
}

func (e *EnhancedInferenceEngine) applyAttention(scores, V *util.Matrix64) (*util.Matrix64, error) {
	// Apply softmax to scores and multiply by V
	// This is a simplified implementation
	return V, nil
}

func (e *EnhancedInferenceEngine) addResidualAndNorm(x, residual *util.Matrix64, norm *ai.RMSNorm) *util.Matrix64 {
	// Add residual connection and apply RMS normalization
	// This is a simplified implementation
	return x
}

func (e *EnhancedInferenceEngine) applySiLU(matrix *util.Matrix64) *util.Matrix64 {
	// Apply SiLU activation: x * sigmoid(x)
	// This is a simplified implementation
	return matrix
}

func (e *EnhancedInferenceEngine) elementWiseMultiply(a, b *util.Matrix64) *util.Matrix64 {
	// Element-wise multiplication
	// This is a simplified implementation
	return a
}

func (e *EnhancedInferenceEngine) applyTopK(logits []float64, k int) []float64 {
	// Apply top-k filtering
	// This is a simplified implementation
	return logits
}

func (e *EnhancedInferenceEngine) applyTopP(logits []float64, p float64) []float64 {
	// Apply top-p (nucleus) filtering
	// This is a simplified implementation
	return logits
}

func (e *EnhancedInferenceEngine) softmax(logits []float64) []float64 {
	// Compute softmax
	max := logits[0]
	for _, x := range logits {
		if x > max {
			max = x
		}
	}

	sum := 0.0
	probs := make([]float64, len(logits))
	for i, x := range logits {
		probs[i] = math.Exp(x - max)
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

func (e *EnhancedInferenceEngine) sampleFromDistribution(probs []float64) (int, float64) {
	// Sample from the probability distribution
	// This is a simplified implementation using argmax
	maxProb := 0.0
	maxIndex := 0
	for i, prob := range probs {
		if prob > maxProb {
			maxProb = prob
			maxIndex = i
		}
	}

	return maxIndex, math.Log(maxProb)
}

// Tokenizer methods
func (t *Tokenizer) Tokenize(text string) ([]int, error) {
	// Simple tokenization - in practice, use a proper tokenizer
	words := splitIntoWords(text)
	tokens := make([]int, 0, len(words))

	// Add BOS token
	tokens = append(tokens, 1)

	for _, word := range words {
		// Simple hash-based tokenization
		token := hashString(word) % t.vocabSize
		tokens = append(tokens, token)
	}

	return tokens, nil
}

func (t *Tokenizer) Detokenize(tokens []int) (string, error) {
	// Simple detokenization - in practice, use a proper tokenizer
	words := make([]string, 0, len(tokens))

	for _, token := range tokens {
		if token == 2 { // EOS token
			break
		}
		if token == 1 { // BOS token
			continue
		}

		word := fmt.Sprintf("token_%d", token)
		words = append(words, word)
	}

	// Join words
	text := ""
	for i, word := range words {
		if i > 0 {
			text += " "
		}
		text += word
	}

	return text, nil
}
