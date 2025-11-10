package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// TerminologyLNN represents a hierarchical Liquid Neural Network for terminology learning.
type TerminologyLNN struct {
	universalLNN *UniversalTerminologyLNN
	domainLNNs   map[string]*DomainTerminologyLNN
	namingLNN    *NamingConventionLNN
	roleLNN      *BusinessRoleLNN
	mu           sync.RWMutex
	logger       *log.Logger
}

// NewTerminologyLNN creates a new hierarchical terminology LNN.
func NewTerminologyLNN(logger *log.Logger) *TerminologyLNN {
	// Initialize global word embedding service if not already initialized
	if globalWordEmbeddingService == nil {
		globalWordEmbeddingService = NewWordEmbeddingService(logger)
		if logger != nil {
			if globalWordEmbeddingService.enabled {
				logger.Printf("Word embedding service enabled (model_type=%s, model_path=%s)", 
					globalWordEmbeddingService.modelType, globalWordEmbeddingService.modelPath)
			} else {
				logger.Printf("Word embedding service disabled (using hash-based fallback)")
			}
		}
	}
	
	return &TerminologyLNN{
		universalLNN: NewUniversalTerminologyLNN(logger),
		domainLNNs:   make(map[string]*DomainTerminologyLNN),
		namingLNN:    NewNamingConventionLNN(logger),
		roleLNN:      NewBusinessRoleLNN(logger),
		logger:       logger,
	}
}

// SparseVocabulary represents a memory-efficient vocabulary with pruning support.
type SparseVocabulary struct {
	terms      map[string]float32
	maxSize    int
	accessCount map[string]int64 // Track access frequency for pruning
	mu         sync.RWMutex
}

// NewSparseVocabulary creates a new sparse vocabulary.
func NewSparseVocabulary(maxSize int) *SparseVocabulary {
	if maxSize <= 0 {
		maxSize = 10000 // Default max size
	}
	return &SparseVocabulary{
		terms:       make(map[string]float32),
		maxSize:     maxSize,
		accessCount: make(map[string]int64),
	}
}

// Get retrieves a term value, incrementing access count.
func (sv *SparseVocabulary) Get(term string) (float32, bool) {
	sv.mu.Lock()
	defer sv.mu.Unlock()
	
	sv.accessCount[term]++
	val, ok := sv.terms[term]
	return val, ok
}

// Set sets a term value.
func (sv *SparseVocabulary) Set(term string, value float32) {
	sv.mu.Lock()
	defer sv.mu.Unlock()
	
	// Prune if at capacity
	if len(sv.terms) >= sv.maxSize && sv.terms[term] == 0 {
		sv.pruneLeastUsed()
	}
	
	sv.terms[term] = value
	sv.accessCount[term]++
}

// pruneLeastUsed removes least frequently accessed terms.
func (sv *SparseVocabulary) pruneLeastUsed() {
	if len(sv.terms) < sv.maxSize {
		return
	}
	
	// Find terms with lowest access count
	type termAccess struct {
		term  string
		count int64
	}
	accessList := make([]termAccess, 0, len(sv.terms))
	for term, count := range sv.accessCount {
		if _, exists := sv.terms[term]; exists {
			accessList = append(accessList, termAccess{term: term, count: count})
		}
	}
	
	// Sort by access count (ascending)
	for i := 0; i < len(accessList)-1; i++ {
		for j := i + 1; j < len(accessList); j++ {
			if accessList[i].count > accessList[j].count {
				accessList[i], accessList[j] = accessList[j], accessList[i]
			}
		}
	}
	
	// Remove bottom 10% of least used terms
	pruneCount := len(accessList) / 10
	if pruneCount == 0 {
		pruneCount = 1
	}
	
	for i := 0; i < pruneCount && i < len(accessList); i++ {
		delete(sv.terms, accessList[i].term)
		delete(sv.accessCount, accessList[i].term)
	}
}

// Size returns the current vocabulary size.
func (sv *SparseVocabulary) Size() int {
	sv.mu.RLock()
	defer sv.mu.RUnlock()
	return len(sv.terms)
}

// UniversalTerminologyLNN learns universal patterns common across all domains.
type UniversalTerminologyLNN struct {
	lnnLayer   *LiquidLayer
	vocabulary *SparseVocabulary // Use sparse vocabulary for memory efficiency
	mu         sync.RWMutex
	logger     *log.Logger
}

// NewUniversalTerminologyLNN creates a new universal terminology LNN.
func NewUniversalTerminologyLNN(logger *log.Logger) *UniversalTerminologyLNN {
	maxVocabSize := 10000 // Default
	if maxVocabEnv := os.Getenv("LNN_MAX_VOCAB_SIZE"); maxVocabEnv != "" {
		var parsedMax int
		if _, err := fmt.Sscanf(maxVocabEnv, "%d", &parsedMax); err == nil && parsedMax > 0 {
			maxVocabSize = parsedMax
		}
	}
	
	return &UniversalTerminologyLNN{
		lnnLayer:   NewLiquidLayer(256, 128, 256), // input, hidden, output
		vocabulary: NewSparseVocabulary(maxVocabSize),
		logger:     logger,
	}
}

// DomainTerminologyLNN learns domain-specific terminology patterns.
type DomainTerminologyLNN struct {
	domainName   string
	lnnLayer     *LiquidLayer
	universalLNN *UniversalTerminologyLNN // Reference to universal layer
	vocabulary   *SparseVocabulary // Use sparse vocabulary for memory efficiency
	mu           sync.RWMutex
	logger       *log.Logger
}

// NewDomainTerminologyLNN creates a new domain-specific terminology LNN.
func NewDomainTerminologyLNN(domainName string, universalLNN *UniversalTerminologyLNN, logger *log.Logger) *DomainTerminologyLNN {
	maxVocabSize := 10000 // Default
	if maxVocabEnv := os.Getenv("LNN_MAX_VOCAB_SIZE"); maxVocabEnv != "" {
		var parsedMax int
		if _, err := fmt.Sscanf(maxVocabEnv, "%d", &parsedMax); err == nil && parsedMax > 0 {
			maxVocabSize = parsedMax
		}
	}
	
	return &DomainTerminologyLNN{
		domainName:   domainName,
		lnnLayer:     NewLiquidLayer(256, 128, 256), // Takes universal output as input
		universalLNN: universalLNN,
		vocabulary:   NewSparseVocabulary(maxVocabSize),
		logger:       logger,
	}
}

// NamingConventionLNN learns naming convention patterns.
type NamingConventionLNN struct {
	lnnLayer *LiquidLayer
	patterns map[string]float32
	mu       sync.RWMutex
	logger   *log.Logger
}

// NewNamingConventionLNN creates a new naming convention LNN.
func NewNamingConventionLNN(logger *log.Logger) *NamingConventionLNN {
	return &NamingConventionLNN{
		lnnLayer: NewLiquidLayer(128, 64, 128),
		patterns: make(map[string]float32),
		logger:   logger,
	}
}

// BusinessRoleLNN learns business role classification patterns.
type BusinessRoleLNN struct {
	lnnLayer *LiquidLayer
	roles    map[string]float32
	mu       sync.RWMutex
	logger   *log.Logger
}

// NewBusinessRoleLNN creates a new business role LNN.
func NewBusinessRoleLNN(logger *log.Logger) *BusinessRoleLNN {
	return &BusinessRoleLNN{
		lnnLayer: NewLiquidLayer(256, 128, 64), // Output: role embeddings
		roles:    make(map[string]float32),
		logger:   logger,
	}
}

// AdamOptimizer implements the Adam optimization algorithm.
type AdamOptimizer struct {
	learningRate float32
	beta1         float32
	beta2         float32
	epsilon       float32
	momentum      [][]float32 // First moment estimates
	velocity      [][]float32 // Second moment estimates
	step          int64       // Step counter for bias correction
}

// NewAdamOptimizer creates a new Adam optimizer.
func NewAdamOptimizer(learningRate, beta1, beta2, epsilon float32, weightShape [][]float32) *AdamOptimizer {
	momentum := make([][]float32, len(weightShape))
	velocity := make([][]float32, len(weightShape))
	
	for i := range weightShape {
		momentum[i] = make([]float32, len(weightShape[i]))
		velocity[i] = make([]float32, len(weightShape[i]))
	}
	
	return &AdamOptimizer{
		learningRate: learningRate,
		beta1:         beta1,
		beta2:         beta2,
		epsilon:       epsilon,
		momentum:      momentum,
		velocity:      velocity,
		step:          0,
	}
}

// Update performs Adam update on weights.
func (a *AdamOptimizer) Update(weights [][]float32, gradients [][]float32) {
	a.step++
	
	// Bias correction factors
	beta1T := float32(math.Pow(float64(a.beta1), float64(a.step)))
	beta2T := float32(math.Pow(float64(a.beta2), float64(a.step)))
	
	for i := range weights {
		for j := range weights[i] {
			if j < len(gradients[i]) {
				grad := gradients[i][j]
				
				// Update biased first moment estimate
				a.momentum[i][j] = a.beta1*a.momentum[i][j] + (1.0-a.beta1)*grad
				
				// Update biased second raw moment estimate
				a.velocity[i][j] = a.beta2*a.velocity[i][j] + (1.0-a.beta2)*grad*grad
				
				// Compute bias-corrected first moment estimate
				mHat := a.momentum[i][j] / (1.0 - beta1T)
				
				// Compute bias-corrected second raw moment estimate
				vHat := a.velocity[i][j] / (1.0 - beta2T)
				
				// Update weights
				weights[i][j] -= a.learningRate * mHat / (float32(math.Sqrt(float64(vHat))) + a.epsilon)
			}
		}
	}
}

// AttentionLayer implements scaled dot-product attention for LNN.
type AttentionLayer struct {
	queryWeights [][]float32
	keyWeights   [][]float32
	valueWeights [][]float32
	scale        float32
}

// NewAttentionLayer creates a new attention layer.
func NewAttentionLayer(inputDim, hiddenDim int, randSource *rand.Rand) *AttentionLayer {
	scale := float32(1.0 / math.Sqrt(float64(hiddenDim)))
	
	// Initialize query, key, value projection matrices
	queryWeights := make([][]float32, hiddenDim)
	keyWeights := make([][]float32, hiddenDim)
	valueWeights := make([][]float32, hiddenDim)
	
	initScale := float32(math.Sqrt(2.0 / float64(inputDim+hiddenDim)))
	for i := range queryWeights {
		queryWeights[i] = make([]float32, inputDim)
		keyWeights[i] = make([]float32, inputDim)
		valueWeights[i] = make([]float32, inputDim)
		for j := range queryWeights[i] {
			queryWeights[i][j] = (float32(randSource.Float64()) - 0.5) * initScale
			keyWeights[i][j] = (float32(randSource.Float64()) - 0.5) * initScale
			valueWeights[i][j] = (float32(randSource.Float64()) - 0.5) * initScale
		}
	}
	
	return &AttentionLayer{
		queryWeights: queryWeights,
		keyWeights:   keyWeights,
		valueWeights: valueWeights,
		scale:        scale,
	}
}

// Apply applies attention mechanism to input sequence.
func (a *AttentionLayer) Apply(inputs [][]float32) []float32 {
	if len(inputs) == 0 {
		return nil
	}
	
	inputDim := len(inputs[0])
	hiddenDim := len(a.queryWeights)
	
	// Project inputs to query, key, value
	queries := make([][]float32, len(inputs))
	keys := make([][]float32, len(inputs))
	values := make([][]float32, len(inputs))
	
	for i, input := range inputs {
		queries[i] = make([]float32, hiddenDim)
		keys[i] = make([]float32, hiddenDim)
		values[i] = make([]float32, hiddenDim)
		
		for j := 0; j < hiddenDim; j++ {
			for k := 0; k < inputDim && k < len(input); k++ {
				queries[i][j] += a.queryWeights[j][k] * input[k]
				keys[i][j] += a.keyWeights[j][k] * input[k]
				values[i][j] += a.valueWeights[j][k] * input[k]
			}
		}
	}
	
	// Compute attention scores (scaled dot-product)
	attentionScores := make([][]float32, len(inputs))
	for i := range inputs {
		attentionScores[i] = make([]float32, len(inputs))
		for j := range inputs {
			score := float32(0.0)
			for k := 0; k < hiddenDim; k++ {
				score += queries[i][k] * keys[j][k]
			}
			attentionScores[i][j] = score * a.scale
		}
	}
	
	// Apply softmax to attention scores
	for i := range attentionScores {
		maxScore := attentionScores[i][0]
		for j := 1; j < len(attentionScores[i]); j++ {
			if attentionScores[i][j] > maxScore {
				maxScore = attentionScores[i][j]
			}
		}
		
		sum := float32(0.0)
		for j := range attentionScores[i] {
			attentionScores[i][j] = float32(math.Exp(float64(attentionScores[i][j] - maxScore)))
			sum += attentionScores[i][j]
		}
		
		for j := range attentionScores[i] {
			attentionScores[i][j] /= sum
		}
	}
	
	// Apply attention to values
	output := make([]float32, hiddenDim)
	for i := range inputs {
		for j := range inputs {
			for k := 0; k < hiddenDim; k++ {
				output[k] += attentionScores[i][j] * values[j][k]
			}
		}
	}
	
	return output
}

// LiquidLayer implements a liquid neural network layer with temporal dynamics.
type LiquidLayer struct {
	inputSize    int
	hiddenSize   int
	outputSize   int
	weights      [][]float32
	timeConstant float64
	state        []float32
	mu           sync.RWMutex
	randSource   *rand.Rand // Seeded random source for reproducibility
	optimizer    *AdamOptimizer // Adam optimizer for weight updates
	batchSize    int            // Batch size for gradient accumulation
	batchGrads   [][]float32    // Accumulated gradients for batch learning
	batchCount   int            // Current batch count
	attention    *AttentionLayer // Attention mechanism (optional)
	useAttention bool            // Whether to use attention
}

// NewLiquidLayer creates a new liquid layer.
func NewLiquidLayer(inputSize, hiddenSize, outputSize int) *LiquidLayer {
	// Get seed from environment or use default
	seed := int64(42) // Default seed for reproducibility
	if seedEnv := os.Getenv("LNN_RANDOM_SEED"); seedEnv != "" {
		var parsedSeed int64
		if _, err := fmt.Sscanf(seedEnv, "%d", &parsedSeed); err == nil {
			seed = parsedSeed
		}
	}
	
	// Create seeded random source
	randSource := rand.New(rand.NewSource(seed))
	
	// Get batch size from environment
	batchSize := 32 // Default batch size
	if batchEnv := os.Getenv("LNN_BATCH_SIZE"); batchEnv != "" {
		var parsedBatch int
		if _, err := fmt.Sscanf(batchEnv, "%d", &parsedBatch); err == nil && parsedBatch > 0 {
			batchSize = parsedBatch
		}
	}
	
	// Check if attention is enabled
	useAttention := os.Getenv("LNN_USE_ATTENTION") != "false" // Default true
	
	layer := &LiquidLayer{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		timeConstant: 1.0,
		state:        make([]float32, hiddenSize),
		weights:      make([][]float32, hiddenSize),
		randSource:   randSource,
		batchSize:    batchSize,
		batchGrads:   make([][]float32, hiddenSize),
		batchCount:   0,
		useAttention: useAttention,
	}

	// Initialize weights with Xavier initialization using seeded random source
	scale := float32(math.Sqrt(2.0 / float64(inputSize+hiddenSize)))
	for i := range layer.weights {
		layer.weights[i] = make([]float32, inputSize)
		layer.batchGrads[i] = make([]float32, inputSize)
		for j := range layer.weights[i] {
			// Use seeded random source instead of time-based random
			layer.weights[i][j] = (float32(layer.randSource.Float64()) - 0.5) * scale
		}
	}
	
	// Initialize Adam optimizer
	// Get optimizer parameters from environment
	learningRate := float32(0.001) // Default learning rate
	if lrEnv := os.Getenv("LNN_ADAM_LR"); lrEnv != "" {
		var parsedLR float64
		if _, err := fmt.Sscanf(lrEnv, "%f", &parsedLR); err == nil {
			learningRate = float32(parsedLR)
		}
	}
	beta1 := float32(0.9)
	if beta1Env := os.Getenv("LNN_ADAM_BETA1"); beta1Env != "" {
		var parsedBeta1 float64
		if _, err := fmt.Sscanf(beta1Env, "%f", &parsedBeta1); err == nil {
			beta1 = float32(parsedBeta1)
		}
	}
	beta2 := float32(0.999)
	if beta2Env := os.Getenv("LNN_ADAM_BETA2"); beta2Env != "" {
		var parsedBeta2 float64
		if _, err := fmt.Sscanf(beta2Env, "%f", &parsedBeta2); err == nil {
			beta2 = float32(parsedBeta2)
		}
	}
	epsilon := float32(1e-8)
	
	layer.optimizer = NewAdamOptimizer(learningRate, beta1, beta2, epsilon, layer.weights)
	
	// Initialize attention layer if enabled
	if useAttention {
		layer.attention = NewAttentionLayer(inputSize, hiddenSize, randSource)
	}

	return layer
}

// Transform applies the liquid neural network transformation with temporal dynamics.
func (l *LiquidLayer) Transform(input []float32, timestamp time.Time) []float32 {
	l.mu.Lock()
	defer l.mu.Unlock()

	if len(input) != l.inputSize {
		// Pad or truncate input to match inputSize
		normalized := make([]float32, l.inputSize)
		copy(normalized, input)
		if len(input) < l.inputSize {
			for i := len(input); i < l.inputSize; i++ {
				normalized[i] = 0.0
			}
		}
		input = normalized
	}

	// Compute time delta (normalized to [0, 1])
	timeDelta := float32(timestamp.Unix()%86400) / 86400.0 // Daily cycle

	// Update hidden state with temporal dynamics
	// LNN: dx/dt = -x/τ + f(input, t)
	for i := range l.state {
		activation := float32(0.0)
		for j := range input {
			activation += l.weights[i][j] * input[j]
		}
		activation = tanh(activation + timeDelta) // Add temporal signal

		// Leaky integration
		l.state[i] = l.state[i]*(1.0-float32(l.timeConstant)) + activation*float32(l.timeConstant)
	}

	// Apply attention if enabled
	if l.useAttention && l.attention != nil {
		// Create input sequence for attention (current input + historical state)
		attentionInputs := [][]float32{input}
		// Add state as context
		stateVec := make([]float32, len(l.state))
		copy(stateVec, l.state)
		attentionInputs = append(attentionInputs, stateVec)
		
		// Apply attention
		attended := l.attention.Apply(attentionInputs)
		
		// Project hidden state back to output space with attention
		output := make([]float32, l.outputSize)
		for i := range output {
			if i < len(attended) {
				output[i] = input[i%len(input)] + 0.1*attended[i] + 0.05*l.state[i%len(l.state)]
			} else if i < len(l.state) {
				output[i] = input[i%len(input)] + 0.1*l.state[i%len(l.state)]
			} else {
				output[i] = input[i%len(input)]
			}
		}
		return output
	}
	
	// Project hidden state back to output space (without attention)
	output := make([]float32, l.outputSize)
	for i := range output {
		if i < len(l.state) {
			output[i] = input[i%len(input)] + 0.1*l.state[i%len(l.state)] // Residual connection
		} else {
			output[i] = input[i%len(input)]
		}
	}

	return output
}

// UpdateWeights updates LNN weights based on learning signal using Adam optimizer.
func (l *LiquidLayer) UpdateWeights(input []float32, target []float32, learningRate float32) {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Compute error
	error := make([]float32, l.outputSize)
	for i := range error {
		if i < len(target) && i < len(input) {
			error[i] = target[i] - input[i]
		}
	}

	// Compute gradients
	gradients := make([][]float32, len(l.weights))
	for i := range l.weights {
		gradients[i] = make([]float32, len(l.weights[i]))
		for j := range l.weights[i] {
			if j < len(error) && j < len(input) {
				// Gradient = error * input (simplified)
				gradients[i][j] = error[j%len(error)] * input[j%len(input)]
			}
		}
	}

	// Accumulate gradients for batch learning
	for i := range gradients {
		for j := range gradients[i] {
			l.batchGrads[i][j] += gradients[i][j]
		}
	}
	l.batchCount++

	// Apply update when batch is complete
	if l.batchCount >= l.batchSize {
		// Average gradients over batch
		for i := range l.batchGrads {
			for j := range l.batchGrads[i] {
				l.batchGrads[i][j] /= float32(l.batchCount)
			}
		}
		
		// Use Adam optimizer to update weights
		if l.optimizer != nil {
			l.optimizer.Update(l.weights, l.batchGrads)
		} else {
			// Fallback to simple SGD if optimizer not initialized
			for i := range l.weights {
				for j := range l.weights[i] {
					l.weights[i][j] += learningRate * l.batchGrads[i][j]
				}
			}
		}
		
		// Reset batch accumulation
		for i := range l.batchGrads {
			for j := range l.batchGrads[i] {
				l.batchGrads[i][j] = 0.0
			}
		}
		l.batchCount = 0
	}
}

// FlushBatch applies accumulated gradients even if batch is not complete.
func (l *LiquidLayer) FlushBatch() {
	l.mu.Lock()
	defer l.mu.Unlock()
	
	if l.batchCount == 0 {
		return
	}
	
	// Average gradients over batch
	for i := range l.batchGrads {
		for j := range l.batchGrads[i] {
			l.batchGrads[i][j] /= float32(l.batchCount)
		}
	}
	
	// Use Adam optimizer to update weights
	if l.optimizer != nil {
		l.optimizer.Update(l.weights, l.batchGrads)
	}
	
	// Reset batch accumulation
	for i := range l.batchGrads {
		for j := range l.batchGrads[i] {
			l.batchGrads[i][j] = 0.0
		}
	}
	l.batchCount = 0
}

// InferDomain infers domain using hierarchical LNN processing.
func (tnn *TerminologyLNN) InferDomain(
	ctx context.Context,
	columnName string,
	tableName string,
	context map[string]any,
) (string, float64) {
	tnn.mu.RLock()
	defer tnn.mu.RUnlock()

	// Step 1: Universal layer processing
	text := columnName + " " + tableName
	textEmbedding := generateTextEmbedding(text)
	universalOutput := tnn.universalLNN.Process(ctx, textEmbedding, time.Now())

	// Step 2: Try each domain-specific LNN
	maxConfidence := 0.0
	bestDomain := "unknown"

	for domainName, domainLNN := range tnn.domainLNNs {
		// Hierarchical: domain LNN takes universal output as input
		domainOutput := domainLNN.Process(ctx, universalOutput, time.Now())
		confidence := calculateConfidence(domainOutput)

		if confidence > maxConfidence {
			maxConfidence = confidence
			bestDomain = domainName
		}
	}

	// If no domain matches well, create a new one
	if maxConfidence < 0.5 && len(tnn.domainLNNs) < 20 { // Limit to prevent explosion
		// Auto-discover new domain
		detectedDomain := detectDomainFromContext(context)
		if detectedDomain != "" {
			tnn.mu.RUnlock()
			tnn.mu.Lock()
			if _, exists := tnn.domainLNNs[detectedDomain]; !exists {
				tnn.domainLNNs[detectedDomain] = NewDomainTerminologyLNN(detectedDomain, tnn.universalLNN, tnn.logger)
				tnn.logger.Printf("Auto-discovered new domain: %s", detectedDomain)
			}
			tnn.mu.Unlock()
			tnn.mu.RLock()
			bestDomain = detectedDomain
			maxConfidence = 0.6 // Initial confidence for new domain
		}
	}

	return bestDomain, maxConfidence
}

// InferRole infers business role using LNN.
func (tnn *TerminologyLNN) InferRole(
	ctx context.Context,
	columnName string,
	columnType string,
	tableName string,
	context map[string]any,
) (string, float64) {
	tnn.mu.RLock()
	defer tnn.mu.RUnlock()

	text := columnName + " " + tableName
	textEmbedding := generateTextEmbedding(text)
	roleOutput := tnn.roleLNN.Process(ctx, textEmbedding, time.Now())

	// Map output to role
	role := mapRoleOutput(roleOutput)
	confidence := calculateConfidence(roleOutput)

	return role, confidence
}

// AnalyzeNamingConvention analyzes naming patterns using LNN.
func (tnn *TerminologyLNN) AnalyzeNamingConvention(
	ctx context.Context,
	columnName string,
) []string {
	tnn.mu.RLock()
	defer tnn.mu.RUnlock()

	textEmbedding := generateTextEmbedding(columnName)
	patternOutput := tnn.namingLNN.Process(ctx, textEmbedding, time.Now())

	// Extract patterns from output
	patterns := extractPatternsFromOutput(patternOutput)

	return patterns
}

// Process processes input through universal LNN.
func (utn *UniversalTerminologyLNN) Process(
	ctx context.Context,
	input []float32,
	timestamp time.Time,
) []float32 {
	utn.mu.RLock()
	defer utn.mu.RUnlock()

	return utn.lnnLayer.Transform(input, timestamp)
}

// Process processes input through domain-specific LNN (hierarchical: universal → domain).
func (dtn *DomainTerminologyLNN) Process(
	ctx context.Context,
	universalOutput []float32,
	timestamp time.Time,
) []float32 {
	dtn.mu.RLock()
	defer dtn.mu.RUnlock()

	// Domain LNN takes universal output as input (hierarchical)
	return dtn.lnnLayer.Transform(universalOutput, timestamp)
}

// Process processes input through naming convention LNN.
func (ncn *NamingConventionLNN) Process(
	ctx context.Context,
	input []float32,
	timestamp time.Time,
) []float32 {
	ncn.mu.RLock()
	defer ncn.mu.RUnlock()

	return ncn.lnnLayer.Transform(input, timestamp)
}

// Process processes input through business role LNN.
func (brn *BusinessRoleLNN) Process(
	ctx context.Context,
	input []float32,
	timestamp time.Time,
) []float32 {
	brn.mu.RLock()
	defer brn.mu.RUnlock()

	return brn.lnnLayer.Transform(input, timestamp)
}

// LearnDomain learns from a domain example.
func (tnn *TerminologyLNN) LearnDomain(
	ctx context.Context,
	text string,
	domain string,
	timestamp time.Time,
) error {
	tnn.mu.Lock()
	defer tnn.mu.Unlock()

	// Get or create domain LNN
	domainLNN, exists := tnn.domainLNNs[domain]
	if !exists {
		domainLNN = NewDomainTerminologyLNN(domain, tnn.universalLNN, tnn.logger)
		tnn.domainLNNs[domain] = domainLNN
	}

	// Process through universal layer
	textEmbedding := generateTextEmbedding(text)
	universalOutput := tnn.universalLNN.Process(ctx, textEmbedding, timestamp)

	// Process through domain layer
	domainOutput := domainLNN.Process(ctx, universalOutput, timestamp)

	// Update weights with learning signal
	target := make([]float32, len(domainOutput))
	for i := range target {
		target[i] = 1.0 // Positive signal for correct domain
	}

	domainLNN.lnnLayer.UpdateWeights(universalOutput, target, 0.01)
	tnn.universalLNN.lnnLayer.UpdateWeights(textEmbedding, universalOutput, 0.005)

	return nil
}

// LearnRole learns from a role example.
func (tnn *TerminologyLNN) LearnRole(
	ctx context.Context,
	text string,
	role string,
	timestamp time.Time,
) error {
	tnn.mu.Lock()
	defer tnn.mu.Unlock()

	textEmbedding := generateTextEmbedding(text)
	_ = tnn.roleLNN.Process(ctx, textEmbedding, timestamp)

	// Update weights
	target := generateRoleTarget(role)
	tnn.roleLNN.lnnLayer.UpdateWeights(textEmbedding, target, 0.01)

	return nil
}

// SerializableAttentionLayerData represents serializable attention layer data.
type SerializableAttentionLayerData struct {
	QueryWeights [][]float32 `json:"query_weights"`
	KeyWeights   [][]float32 `json:"key_weights"`
	ValueWeights [][]float32 `json:"value_weights"`
	Scale        float32     `json:"scale"`
}

// SerializableLiquidLayerData represents serializable liquid layer data.
type SerializableLiquidLayerData struct {
	InputSize      int                          `json:"input_size"`
	HiddenSize     int                          `json:"hidden_size"`
	OutputSize     int                          `json:"output_size"`
	Weights        [][]float32                  `json:"weights"`
	TimeConstant   float64                      `json:"time_constant"`
	State          []float32                    `json:"state"`
	BatchSize      int                          `json:"batch_size"`
	UseAttention   bool                         `json:"use_attention"`
	AttentionState *SerializableAttentionLayerData `json:"attention_state,omitempty"`
	OptimizerState *AdamOptimizerState          `json:"optimizer_state,omitempty"`
}

// AdamOptimizerState represents serializable Adam optimizer state.
type AdamOptimizerState struct {
	LearningRate float32     `json:"learning_rate"`
	Beta1        float32     `json:"beta1"`
	Beta2        float32     `json:"beta2"`
	Epsilon      float32     `json:"epsilon"`
	Momentum     [][]float32 `json:"momentum"`
	Velocity     [][]float32 `json:"velocity"`
	Step         int64       `json:"step"`
}

// SerializableUniversalLNNData represents serializable universal LNN data.
type SerializableUniversalLNNData struct {
	LayerData  SerializableLiquidLayerData `json:"layer_data"`
	Vocabulary map[string]float32          `json:"vocabulary"` // Converted from sparse vocabulary
}

// SerializableDomainLNNData represents serializable domain LNN data.
type SerializableDomainLNNData struct {
	DomainName string                     `json:"domain_name"`
	LayerData  SerializableLiquidLayerData `json:"layer_data"`
	Vocabulary map[string]float32         `json:"vocabulary"`
}

// SerializableNamingLNNData represents serializable naming convention LNN data.
type SerializableNamingLNNData struct {
	LayerData SerializableLiquidLayerData `json:"layer_data"`
	Patterns  map[string]float32           `json:"patterns"`
}

// SerializableRoleLNNData represents serializable business role LNN data.
type SerializableRoleLNNData struct {
	LayerData SerializableLiquidLayerData `json:"layer_data"`
	Roles     map[string]float32          `json:"roles"`
}

// LNNModelData represents serializable model data for persistence.
type LNNModelData struct {
	Version      string                              `json:"version"`
	SavedAt      time.Time                           `json:"saved_at"`
	UniversalLNN SerializableUniversalLNNData        `json:"universal_lnn"`
	DomainLNNs   map[string]SerializableDomainLNNData `json:"domain_lnns"`
	NamingLNN    SerializableNamingLNNData           `json:"naming_lnn"`
	RoleLNN      SerializableRoleLNNData             `json:"role_lnn"`
}

// serializeLiquidLayer converts LiquidLayer to serializable format.
func serializeLiquidLayer(layer *LiquidLayer) SerializableLiquidLayerData {
	layer.mu.RLock()
	defer layer.mu.RUnlock()
	
	data := SerializableLiquidLayerData{
		InputSize:    layer.inputSize,
		HiddenSize:    layer.hiddenSize,
		OutputSize:    layer.outputSize,
		Weights:       make([][]float32, len(layer.weights)),
		TimeConstant:  layer.timeConstant,
		State:         make([]float32, len(layer.state)),
		BatchSize:     layer.batchSize,
	}
	
	// Copy weights
	for i := range layer.weights {
		data.Weights[i] = make([]float32, len(layer.weights[i]))
		copy(data.Weights[i], layer.weights[i])
	}
	
	// Copy state
	copy(data.State, layer.state)
	
	// Serialize attention state if available
	if layer.useAttention && layer.attention != nil {
		attState := SerializableAttentionLayerData{
			QueryWeights: make([][]float32, len(layer.attention.queryWeights)),
			KeyWeights:   make([][]float32, len(layer.attention.keyWeights)),
			ValueWeights: make([][]float32, len(layer.attention.valueWeights)),
			Scale:        layer.attention.scale,
		}
		
		for i := range layer.attention.queryWeights {
			attState.QueryWeights[i] = make([]float32, len(layer.attention.queryWeights[i]))
			copy(attState.QueryWeights[i], layer.attention.queryWeights[i])
			attState.KeyWeights[i] = make([]float32, len(layer.attention.keyWeights[i]))
			copy(attState.KeyWeights[i], layer.attention.keyWeights[i])
			attState.ValueWeights[i] = make([]float32, len(layer.attention.valueWeights[i]))
			copy(attState.ValueWeights[i], layer.attention.valueWeights[i])
		}
		
		data.AttentionState = &attState
		data.UseAttention = true
	}
	
	// Serialize optimizer state if available
	if layer.optimizer != nil {
		optState := AdamOptimizerState{
			LearningRate: layer.optimizer.learningRate,
			Beta1:         layer.optimizer.beta1,
			Beta2:         layer.optimizer.beta2,
			Epsilon:       layer.optimizer.epsilon,
			Momentum:      make([][]float32, len(layer.optimizer.momentum)),
			Velocity:      make([][]float32, len(layer.optimizer.velocity)),
			Step:          layer.optimizer.step,
		}
		
		for i := range layer.optimizer.momentum {
			optState.Momentum[i] = make([]float32, len(layer.optimizer.momentum[i]))
			copy(optState.Momentum[i], layer.optimizer.momentum[i])
			optState.Velocity[i] = make([]float32, len(layer.optimizer.velocity[i]))
			copy(optState.Velocity[i], layer.optimizer.velocity[i])
		}
		
		data.OptimizerState = &optState
	}
	
	return data
}

// deserializeLiquidLayer creates LiquidLayer from serializable format.
func deserializeLiquidLayer(data SerializableLiquidLayerData, logger *log.Logger) *LiquidLayer {
	layer := &LiquidLayer{
		inputSize:    data.InputSize,
		hiddenSize:   data.HiddenSize,
		outputSize:   data.OutputSize,
		weights:      make([][]float32, len(data.Weights)),
		timeConstant: data.TimeConstant,
		state:        make([]float32, len(data.State)),
		batchSize:    data.BatchSize,
		batchGrads:   make([][]float32, data.HiddenSize),
		batchCount:   0,
	}
	
	// Copy weights
	for i := range data.Weights {
		layer.weights[i] = make([]float32, len(data.Weights[i]))
		copy(layer.weights[i], data.Weights[i])
		layer.batchGrads[i] = make([]float32, data.InputSize)
	}
	
	// Copy state
	copy(layer.state, data.State)
	
	// Restore random source (will be recreated with same seed)
	seed := int64(42)
	if seedEnv := os.Getenv("LNN_RANDOM_SEED"); seedEnv != "" {
		var parsedSeed int64
		if _, err := fmt.Sscanf(seedEnv, "%d", &parsedSeed); err == nil {
			seed = parsedSeed
		}
	}
	layer.randSource = rand.New(rand.NewSource(seed))
	
	// Restore attention layer if state available
	if data.UseAttention && data.AttentionState != nil {
		layer.useAttention = true
		layer.attention = &AttentionLayer{
			queryWeights: data.AttentionState.QueryWeights,
			keyWeights:   data.AttentionState.KeyWeights,
			valueWeights: data.AttentionState.ValueWeights,
			scale:        data.AttentionState.Scale,
		}
	}
	
	// Restore optimizer if state available
	if data.OptimizerState != nil {
		layer.optimizer = &AdamOptimizer{
			learningRate: data.OptimizerState.LearningRate,
			beta1:         data.OptimizerState.Beta1,
			beta2:         data.OptimizerState.Beta2,
			epsilon:       data.OptimizerState.Epsilon,
			momentum:      data.OptimizerState.Momentum,
			velocity:      data.OptimizerState.Velocity,
			step:          data.OptimizerState.Step,
		}
	} else {
		// Initialize new optimizer
		learningRate := float32(0.001)
		if lrEnv := os.Getenv("LNN_ADAM_LR"); lrEnv != "" {
			var parsedLR float64
			if _, err := fmt.Sscanf(lrEnv, "%f", &parsedLR); err == nil {
				learningRate = float32(parsedLR)
			}
		}
		beta1 := float32(0.9)
		beta2 := float32(0.999)
		epsilon := float32(1e-8)
		layer.optimizer = NewAdamOptimizer(learningRate, beta1, beta2, epsilon, layer.weights)
	}
	
	return layer
}

// SaveModel saves the LNN model to disk.
func (tnn *TerminologyLNN) SaveModel(path string) error {
	tnn.mu.RLock()
	defer tnn.mu.RUnlock()
	
	// Create directory if it doesn't exist
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}
	
	// Serialize universal LNN
	utnData := SerializableUniversalLNNData{
		LayerData:  serializeLiquidLayer(tnn.universalLNN.lnnLayer),
		Vocabulary: make(map[string]float32),
	}
	tnn.universalLNN.mu.RLock()
	// Convert sparse vocabulary to regular map for serialization
	if tnn.universalLNN.vocabulary != nil {
		tnn.universalLNN.vocabulary.mu.RLock()
		for k, v := range tnn.universalLNN.vocabulary.terms {
			utnData.Vocabulary[k] = v
		}
		tnn.universalLNN.vocabulary.mu.RUnlock()
	}
	tnn.universalLNN.mu.RUnlock()
	
	// Serialize domain LNNs
	domainData := make(map[string]SerializableDomainLNNData)
	for domain, dlnn := range tnn.domainLNNs {
		dlnn.mu.RLock()
		domainData[domain] = SerializableDomainLNNData{
			DomainName: domain,
			LayerData:  serializeLiquidLayer(dlnn.lnnLayer),
			Vocabulary: make(map[string]float32),
		}
		if dlnn.vocabulary != nil {
			dlnn.vocabulary.mu.RLock()
			for k, v := range dlnn.vocabulary.terms {
				domainData[domain].Vocabulary[k] = v
			}
			dlnn.vocabulary.mu.RUnlock()
		}
		dlnn.mu.RUnlock()
	}
	
	// Serialize naming LNN
	ncnData := SerializableNamingLNNData{
		LayerData: serializeLiquidLayer(tnn.namingLNN.lnnLayer),
		Patterns:  make(map[string]float32),
	}
	tnn.namingLNN.mu.RLock()
	for k, v := range tnn.namingLNN.patterns {
		ncnData.Patterns[k] = v
	}
	tnn.namingLNN.mu.RUnlock()
	
	// Serialize role LNN
	brnData := SerializableRoleLNNData{
		LayerData: serializeLiquidLayer(tnn.roleLNN.lnnLayer),
		Roles:     make(map[string]float32),
	}
	tnn.roleLNN.mu.RLock()
	for k, v := range tnn.roleLNN.roles {
		brnData.Roles[k] = v
	}
	tnn.roleLNN.mu.RUnlock()
	
	// Create model data structure
	modelData := LNNModelData{
		Version:      "1.0",
		SavedAt:      time.Now(),
		UniversalLNN: utnData,
		DomainLNNs:   domainData,
		NamingLNN:    ncnData,
		RoleLNN:      brnData,
	}
	
	// Save as JSON for human readability and versioning
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(modelData); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}
	
	if tnn.logger != nil {
		tnn.logger.Printf("LNN model saved to %s", path)
	}
	
	return nil
}

// LoadModel loads the LNN model from disk.
func (tnn *TerminologyLNN) LoadModel(path string) error {
	tnn.mu.Lock()
	defer tnn.mu.Unlock()
	
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()
	
	var modelData LNNModelData
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&modelData); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}
	
	// Validate version
	if modelData.Version != "1.0" {
		return fmt.Errorf("unsupported model version: %s", modelData.Version)
	}
	
	// Restore universal LNN
	maxVocabSize := 10000
	if maxVocabEnv := os.Getenv("LNN_MAX_VOCAB_SIZE"); maxVocabEnv != "" {
		var parsedMax int
		if _, err := fmt.Sscanf(maxVocabEnv, "%d", &parsedMax); err == nil && parsedMax > 0 {
			maxVocabSize = parsedMax
		}
	}
	sparseVocab := NewSparseVocabulary(maxVocabSize)
	for k, v := range modelData.UniversalLNN.Vocabulary {
		sparseVocab.Set(k, v)
	}
	tnn.universalLNN = &UniversalTerminologyLNN{
		lnnLayer:   deserializeLiquidLayer(modelData.UniversalLNN.LayerData, tnn.logger),
		vocabulary: sparseVocab,
		logger:     tnn.logger,
	}
	
	// Restore domain LNNs
	tnn.domainLNNs = make(map[string]*DomainTerminologyLNN)
	for domain, data := range modelData.DomainLNNs {
		domainVocab := NewSparseVocabulary(maxVocabSize)
		for k, v := range data.Vocabulary {
			domainVocab.Set(k, v)
		}
		tnn.domainLNNs[domain] = &DomainTerminologyLNN{
			domainName:   domain,
			lnnLayer:     deserializeLiquidLayer(data.LayerData, tnn.logger),
			universalLNN: tnn.universalLNN,
			vocabulary:   domainVocab,
			logger:       tnn.logger,
		}
	}
	
	// Restore naming LNN
	tnn.namingLNN = &NamingConventionLNN{
		lnnLayer: deserializeLiquidLayer(modelData.NamingLNN.LayerData, tnn.logger),
		patterns: modelData.NamingLNN.Patterns,
		logger:   tnn.logger,
	}
	
	// Restore role LNN
	tnn.roleLNN = &BusinessRoleLNN{
		lnnLayer: deserializeLiquidLayer(modelData.RoleLNN.LayerData, tnn.logger),
		roles:    modelData.RoleLNN.Roles,
		logger:   tnn.logger,
	}
	
	if tnn.logger != nil {
		tnn.logger.Printf("LNN model loaded from %s (version: %s, saved at: %s)", 
			path, modelData.Version, modelData.SavedAt.Format(time.RFC3339))
	}
	
	return nil
}

// Helper functions

// WordEmbeddingService provides word embeddings using Word2Vec/FastText with fallback to hash-based
type WordEmbeddingService struct {
	enabled      bool
	modelType    string // "word2vec", "fasttext", or "hash"
	modelPath    string
	logger       *log.Logger
	fallbackHash bool // Whether to fallback to hash-based if embedding fails
}

// NewWordEmbeddingService creates a new word embedding service
func NewWordEmbeddingService(logger *log.Logger) *WordEmbeddingService {
	enabled := os.Getenv("LNN_USE_WORD_EMBEDDINGS") == "true"
	modelType := os.Getenv("LNN_EMBEDDING_MODEL_TYPE")
	if modelType == "" {
		modelType = "word2vec" // Default to Word2Vec
	}
	modelPath := os.Getenv("LNN_EMBEDDING_MODEL_PATH")
	fallbackHash := os.Getenv("LNN_EMBEDDING_FALLBACK_HASH") != "false" // Default true
	
	return &WordEmbeddingService{
		enabled:      enabled,
		modelType:    modelType,
		modelPath:    modelPath,
		logger:       logger,
		fallbackHash: fallbackHash,
	}
}

// GenerateEmbedding generates embedding for text using Word2Vec/FastText or hash-based fallback
func (wes *WordEmbeddingService) GenerateEmbedding(text string) []float32 {
	// Try Word2Vec/FastText if enabled
	if wes.enabled && wes.modelPath != "" {
		embedding, err := wes.generateWordEmbedding(text)
		if err == nil && len(embedding) > 0 {
			// Ensure embedding is 256 dimensions (pad or truncate)
			return wes.normalizeEmbedding(embedding, 256)
		}
		if wes.logger != nil {
			wes.logger.Printf("Word embedding failed for text '%s': %v, falling back to hash-based", text[:min(len(text), 50)], err)
		}
	}
	
	// Fallback to hash-based embedding
	if wes.fallbackHash {
		return generateHashEmbedding(text)
	}
	
	// If no fallback, return zero embedding
	return make([]float32, 256)
}

// generateWordEmbedding generates embedding using Word2Vec/FastText via Python script
func (wes *WordEmbeddingService) generateWordEmbedding(text string) ([]float32, error) {
	// Use Python script for Word2Vec/FastText embeddings (similar to other embedding scripts)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	scriptPath := "./scripts/embed_word2vec.py"
	if wes.modelType == "fasttext" {
		scriptPath = "./scripts/embed_fasttext.py"
	}
	
	cmd := exec.CommandContext(ctx, "python3", scriptPath,
		"--text", text,
		"--model-path", wes.modelPath,
		"--model-type", wes.modelType,
	)
	
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate word embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate word embedding: %w", err)
	}
	
	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal word embedding: %w", err)
	}
	
	return embedding, nil
}

// normalizeEmbedding normalizes embedding to target dimension
func (wes *WordEmbeddingService) normalizeEmbedding(embedding []float32, targetDim int) []float32 {
	result := make([]float32, targetDim)
	
	if len(embedding) == targetDim {
		copy(result, embedding)
	} else if len(embedding) > targetDim {
		// Truncate
		copy(result, embedding[:targetDim])
	} else {
		// Pad with zeros
		copy(result, embedding)
		for i := len(embedding); i < targetDim; i++ {
			result[i] = 0.0
		}
	}
	
	return result
}

// generateHashEmbedding generates hash-based embedding (fallback)
func generateHashEmbedding(text string) []float32 {
	embedding := make([]float32, 256)
	hash := 0
	for _, char := range text {
		hash = hash*31 + int(char)
	}
	for i := range embedding {
		embedding[i] = float32((hash+i*17)%1000) / 1000.0
	}
	return embedding
}

// Global word embedding service instance
var globalWordEmbeddingService *WordEmbeddingService

// GetWordEmbeddingService returns the global word embedding service
func GetWordEmbeddingService(logger *log.Logger) *WordEmbeddingService {
	if globalWordEmbeddingService == nil {
		globalWordEmbeddingService = NewWordEmbeddingService(logger)
	}
	return globalWordEmbeddingService
}

// generateTextEmbedding generates text embedding using Word2Vec/FastText with hash fallback
func generateTextEmbedding(text string) []float32 {
	// Use global word embedding service if available
	if globalWordEmbeddingService != nil {
		return globalWordEmbeddingService.GenerateEmbedding(text)
	}
	
	// Fallback to hash-based embedding
	return generateHashEmbedding(text)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func calculateConfidence(output []float32) float64 {
	// Calculate confidence from output magnitude
	sum := float64(0.0)
	for _, v := range output {
		sum += float64(v * v)
	}
	return math.Sqrt(sum / float64(len(output)))
}

func mapRoleOutput(output []float32) string {
	// Map output vector to role (simplified)
	roles := []string{"identifier", "amount", "date", "status", "name", "email", "phone", "address", "quantity"}
	idx := int(math.Abs(float64(output[0]*100))) % len(roles)
	return roles[idx]
}

func extractPatternsFromOutput(output []float32) []string {
	patterns := []string{}
	patternNames := []string{"snake_case", "camelCase", "PascalCase", "UPPER_SNAKE", "has_id_suffix", "has_id_prefix", "has_date_suffix", "has_ts_suffix"}

	for i, val := range output {
		if i < len(patternNames) && math.Abs(float64(val)) > 0.3 {
			patterns = append(patterns, patternNames[i])
		}
	}

	return patterns
}

func detectDomainFromContext(context map[string]any) string {
	// Try to detect domain from context
	if context == nil {
		return ""
	}

	// Check for domain hints in context
	if domain, ok := context["domain"].(string); ok && domain != "" {
		return domain
	}

	return ""
}

func generateRoleTarget(role string) []float32 {
	// Generate target vector for role
	roles := []string{"identifier", "amount", "date", "status", "name", "email", "phone", "address", "quantity"}
	target := make([]float32, 64)

	for i, r := range roles {
		if r == role {
			target[i%len(target)] = 1.0
		}
	}

	return target
}

// randomFloat32 generates a random float32 using the layer's seeded random source.
// This function is kept for backward compatibility but should use layer.randSource instead.
func randomFloat32() float32 {
	// Fallback: use time-based random if no seeded source available
	// This should not be used in production - always use layer.randSource
	return float32(math.Sin(float64(time.Now().UnixNano())) * 0.5)
}

func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}
