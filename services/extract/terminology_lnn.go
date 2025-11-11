package main

import (
	"context"
	"log"
	"math"
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
	return &TerminologyLNN{
		universalLNN: NewUniversalTerminologyLNN(logger),
		domainLNNs:   make(map[string]*DomainTerminologyLNN),
		namingLNN:    NewNamingConventionLNN(logger),
		roleLNN:      NewBusinessRoleLNN(logger),
		logger:       logger,
	}
}

// UniversalTerminologyLNN learns universal patterns common across all domains.
type UniversalTerminologyLNN struct {
	lnnLayer   *LiquidLayer
	vocabulary map[string]float32
	mu         sync.RWMutex
	logger     *log.Logger
}

// NewUniversalTerminologyLNN creates a new universal terminology LNN.
func NewUniversalTerminologyLNN(logger *log.Logger) *UniversalTerminologyLNN {
	return &UniversalTerminologyLNN{
		lnnLayer:   NewLiquidLayer(256, 128, 256), // input, hidden, output
		vocabulary: make(map[string]float32),
		logger:     logger,
	}
}

// DomainTerminologyLNN learns domain-specific terminology patterns.
type DomainTerminologyLNN struct {
	domainName   string
	lnnLayer     *LiquidLayer
	universalLNN *UniversalTerminologyLNN // Reference to universal layer
	vocabulary   map[string]float32
	mu           sync.RWMutex
	logger       *log.Logger
}

// NewDomainTerminologyLNN creates a new domain-specific terminology LNN.
func NewDomainTerminologyLNN(domainName string, universalLNN *UniversalTerminologyLNN, logger *log.Logger) *DomainTerminologyLNN {
	return &DomainTerminologyLNN{
		domainName:   domainName,
		lnnLayer:     NewLiquidLayer(256, 128, 256), // Takes universal output as input
		universalLNN: universalLNN,
		vocabulary:   make(map[string]float32),
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

// LiquidLayer implements a liquid neural network layer with temporal dynamics.
type LiquidLayer struct {
	inputSize    int
	hiddenSize   int
	outputSize   int
	weights      [][]float32
	timeConstant float64
	state        []float32
	mu           sync.RWMutex
}

// NewLiquidLayer creates a new liquid layer.
func NewLiquidLayer(inputSize, hiddenSize, outputSize int) *LiquidLayer {
	layer := &LiquidLayer{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		outputSize:   outputSize,
		timeConstant: 1.0,
		state:        make([]float32, hiddenSize),
		weights:      make([][]float32, hiddenSize),
	}

	// Initialize weights with Xavier initialization
	scale := float32(math.Sqrt(2.0 / float64(inputSize+hiddenSize)))
	for i := range layer.weights {
		layer.weights[i] = make([]float32, inputSize)
		for j := range layer.weights[i] {
			layer.weights[i][j] = (randomFloat32() - 0.5) * scale
		}
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

	// Project hidden state back to output space
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

// UpdateWeights updates LNN weights based on learning signal.
func (l *LiquidLayer) UpdateWeights(input []float32, target []float32, learningRate float32) {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Simplified gradient update
	error := make([]float32, l.outputSize)
	for i := range error {
		if i < len(target) && i < len(input) {
			error[i] = target[i] - input[i]
		}
	}

	// Update weights based on error
	for i := range l.weights {
		for j := range l.weights[i] {
			if j < len(error) {
				l.weights[i][j] += learningRate * error[j%len(error)] * input[j%len(input)]
			}
		}
	}
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

// Helper functions

func generateTextEmbedding(text string) []float32 {
	// Simple hash-based embedding (in production, would use proper embedding)
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

func randomFloat32() float32 {
	return float32(math.Sin(float64(time.Now().UnixNano())) * 0.5)
}

func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}
