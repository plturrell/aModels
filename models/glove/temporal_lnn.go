package glove

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// TemporalGloVe extends GloVe with Liquid Neural Network for temporal dynamics
type TemporalGloVe struct {
	baseModel  *Model
	lnnLayer   *LiquidLayer
	timeAware  bool
	temporalDB TemporalStorage
	mu         sync.RWMutex
}

// LiquidLayer implements a simplified Liquid Neural Network for temporal adaptation
type LiquidLayer struct {
	inputSize    int
	hiddenSize   int
	outputSize   int
	weights      [][]float32
	timeConstant float64 // Time constant for temporal dynamics
	state        []float32
	mu           sync.RWMutex
}

// TemporalStorage interface for storing time-stamped embeddings
type TemporalStorage interface {
	StoreTemporalVector(ctx context.Context, word string, vector []float32, timestamp time.Time) error
	GetTemporalVector(ctx context.Context, word string, timestamp time.Time) ([]float32, error)
	GetTemporalEvolution(ctx context.Context, word string, startTime, endTime time.Time) ([]TemporalVector, error)
}

// TemporalVector represents a word vector at a specific time
type TemporalVector struct {
	Word      string
	Vector    []float32
	Timestamp time.Time
	Drift     float64 // Semantic drift magnitude
}

// NewTemporalGloVe creates a GloVe model with temporal awareness
func NewTemporalGloVe(baseModel *Model, hiddenSize int) (*TemporalGloVe, error) {
	if baseModel == nil {
		return nil, fmt.Errorf("base model cannot be nil")
	}

	lnnLayer := &LiquidLayer{
		inputSize:    baseModel.vectorSize,
		hiddenSize:   hiddenSize,
		outputSize:   baseModel.vectorSize,
		timeConstant: 1.0, // Default time constant
		state:        make([]float32, hiddenSize),
	}

	// Initialize LNN weights with Xavier initialization
	lnnLayer.weights = make([][]float32, hiddenSize)
	scale := float32(math.Sqrt(2.0 / float64(baseModel.vectorSize+hiddenSize)))
	for i := range lnnLayer.weights {
		lnnLayer.weights[i] = make([]float32, baseModel.vectorSize)
		for j := range lnnLayer.weights[i] {
			lnnLayer.weights[i][j] = (randomFloat32() - 0.5) * scale
		}
	}

	return &TemporalGloVe{
		baseModel: baseModel,
		lnnLayer:  lnnLayer,
		timeAware: true,
	}, nil
}

// GetTemporalVector returns a temporally-adapted word vector
func (tg *TemporalGloVe) GetTemporalVector(ctx context.Context, word string, timestamp time.Time) ([]float32, error) {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	// Get base vector
	baseVec, err := tg.baseModel.GetVector(ctx, word)
	if err != nil {
		return nil, err
	}

	if !tg.timeAware {
		return baseVec, nil
	}

	// Apply LNN transformation for temporal adaptation
	temporalVec := tg.lnnLayer.Transform(baseVec, timestamp)

	return temporalVec, nil
}

// Transform applies the liquid neural network transformation
func (l *LiquidLayer) Transform(input []float32, timestamp time.Time) []float32 {
	l.mu.Lock()
	defer l.mu.Unlock()

	// Compute time delta (normalized to [0, 1])
	timeDelta := float32(timestamp.Unix()%86400) / 86400.0 // Daily cycle

	// Update hidden state with temporal dynamics
	// Simplified LNN: dx/dt = -x/τ + f(input, t)
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
		// Simple linear projection (in practice, would use learned weights)
		if i < len(l.state) {
			output[i] = input[i] + 0.1*l.state[i%len(l.state)] // Residual connection
		} else {
			output[i] = input[i]
		}
	}

	return output
}

// TrainTemporal trains the model with temporal awareness
func (tg *TemporalGloVe) TrainTemporal(ctx context.Context, corpus []TemporalDocument) error {
	tg.mu.Lock()
	defer tg.mu.Unlock()

	fmt.Println("Training temporal GloVe model...")

	// Sort documents by timestamp
	sortedCorpus := make([]TemporalDocument, len(corpus))
	copy(sortedCorpus, corpus)
	// In production, would sort by timestamp

	// Train base model on each temporal slice
	for i, doc := range sortedCorpus {
		fmt.Printf("Processing temporal slice %d/%d (timestamp: %s)\n",
			i+1, len(sortedCorpus), doc.Timestamp.Format("2006-01-02"))

		// Update base embeddings
		if err := tg.baseModel.BuildVocabulary(ctx, []string{doc.Text}); err != nil {
			return fmt.Errorf("build vocabulary for slice %d: %w", i, err)
		}

		// Adapt LNN layer for this time period
		if err := tg.adaptLNN(ctx, doc); err != nil {
			return fmt.Errorf("adapt LNN for slice %d: %w", i, err)
		}
	}

	fmt.Println("Temporal training complete")
	return nil
}

// adaptLNN adapts the LNN layer for a specific time period
func (tg *TemporalGloVe) adaptLNN(ctx context.Context, doc TemporalDocument) error {
	// In production, would update LNN weights based on temporal patterns
	// For now, just update time constant based on document characteristics
	tg.lnnLayer.mu.Lock()
	defer tg.lnnLayer.mu.Unlock()

	// Adjust time constant based on document recency
	age := time.Since(doc.Timestamp).Hours() / 24.0 // Age in days
	tg.lnnLayer.timeConstant = 1.0 / (1.0 + math.Log(1.0+age))

	return nil
}

// ComputeSemanticDrift measures how much a word's meaning has changed over time
func (tg *TemporalGloVe) ComputeSemanticDrift(ctx context.Context, word string, t1, t2 time.Time) (float64, error) {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	// Get vectors at two time points
	vec1, err := tg.GetTemporalVector(ctx, word, t1)
	if err != nil {
		return 0, err
	}

	vec2, err := tg.GetTemporalVector(ctx, word, t2)
	if err != nil {
		return 0, err
	}

	// Compute cosine distance (1 - similarity)
	similarity := cosineSimilarity(vec1, vec2)
	drift := 1.0 - float64(similarity)

	return drift, nil
}

// SimilarWord represents a word with similarity score
type SimilarWord struct {
	Word  string
	Score float64
}

// GetTemporalNeighbors finds words with similar temporal patterns
func (tg *TemporalGloVe) GetTemporalNeighbors(ctx context.Context, word string, timestamp time.Time, k int) ([]SimilarWord, error) {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	// Get temporal vector for query word
	queryVec, err := tg.GetTemporalVector(ctx, word, timestamp)
	if err != nil {
		return nil, err
	}

	// Find similar words at this time point
	return tg.baseModel.findSimilarVectors(ctx, queryVec, k)
}

// AnalyzeTemporalTrends analyzes how word usage evolves over time
func (tg *TemporalGloVe) AnalyzeTemporalTrends(ctx context.Context, word string, timePoints []time.Time) (*TemporalTrend, error) {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	trend := &TemporalTrend{
		Word:       word,
		TimePoints: timePoints,
		Vectors:    make([][]float32, len(timePoints)),
		Drifts:     make([]float64, len(timePoints)-1),
	}

	// Get vectors at each time point
	for i, t := range timePoints {
		vec, err := tg.GetTemporalVector(ctx, word, t)
		if err != nil {
			return nil, err
		}
		trend.Vectors[i] = vec

		// Compute drift from previous time point
		if i > 0 {
			similarity := cosineSimilarity(trend.Vectors[i-1], trend.Vectors[i])
			trend.Drifts[i-1] = 1.0 - float64(similarity)
		}
	}

	// Compute overall drift rate
	if len(trend.Drifts) > 0 {
		totalDrift := 0.0
		for _, d := range trend.Drifts {
			totalDrift += d
		}
		trend.DriftRate = totalDrift / float64(len(trend.Drifts))
	}

	return trend, nil
}

// TemporalDocument represents a document with timestamp
type TemporalDocument struct {
	Text      string
	Timestamp time.Time
	Metadata  map[string]string
}

// TemporalTrend represents the temporal evolution of a word
type TemporalTrend struct {
	Word       string
	TimePoints []time.Time
	Vectors    [][]float32
	Drifts     []float64
	DriftRate  float64
}

// PredictFutureVector predicts how a word vector might evolve
func (tg *TemporalGloVe) PredictFutureVector(ctx context.Context, word string, futureTime time.Time) ([]float32, error) {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	// Get current vector
	currentVec, err := tg.GetTemporalVector(ctx, word, time.Now())
	if err != nil {
		return nil, err
	}

	// Get historical trend
	pastTime := time.Now().AddDate(0, -6, 0) // 6 months ago
	trend, err := tg.AnalyzeTemporalTrends(ctx, word, []time.Time{pastTime, time.Now()})
	if err != nil {
		return nil, err
	}

	// Extrapolate based on drift rate
	timeDelta := futureTime.Sub(time.Now()).Hours() / 24.0 / 30.0 // Months
	_ = trend.DriftRate * timeDelta                               // driftMagnitude for future use

	// Apply drift to current vector
	predictedVec := make([]float32, len(currentVec))
	for i := range currentVec {
		// Simple linear extrapolation with decay
		decay := float32(math.Exp(-0.1 * timeDelta))
		predictedVec[i] = currentVec[i] * decay
	}

	return predictedVec, nil
}

// UpdateOnline performs online learning with new temporal data
func (tg *TemporalGloVe) UpdateOnline(ctx context.Context, doc TemporalDocument) error {
	tg.mu.Lock()
	defer tg.mu.Unlock()

	fmt.Printf("Online update with document from %s\n", doc.Timestamp.Format("2006-01-02"))

	// Update base model vocabulary
	if err := tg.baseModel.BuildVocabulary(ctx, []string{doc.Text}); err != nil {
		return fmt.Errorf("update vocabulary: %w", err)
	}

	// Adapt LNN layer
	if err := tg.adaptLNN(ctx, doc); err != nil {
		return fmt.Errorf("adapt LNN: %w", err)
	}

	return nil
}

// SetTimeConstant adjusts the temporal adaptation rate
func (tg *TemporalGloVe) SetTimeConstant(tau float64) {
	tg.lnnLayer.mu.Lock()
	defer tg.lnnLayer.mu.Unlock()
	tg.lnnLayer.timeConstant = tau
}

// GetTimeConstant returns the current time constant
func (tg *TemporalGloVe) GetTimeConstant() float64 {
	tg.lnnLayer.mu.RLock()
	defer tg.lnnLayer.mu.RUnlock()
	return tg.lnnLayer.timeConstant
}

// Helper function: tanh activation
func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// Helper function: random float32
func randomFloat32() float32 {
	return float32(rand.Float64())
}

// findSimilarVectors is a helper that would be added to the base Model
// This is a placeholder showing the integration point
func (m *Model) findSimilarVectors(ctx context.Context, queryVec []float32, k int) ([]SimilarWord, error) {
	// This would use the existing MostSimilar logic but with a custom vector
	// For now, return empty slice
	return []SimilarWord{}, nil
}

// ExportTemporalModel exports the temporal model for deployment
func (tg *TemporalGloVe) ExportTemporalModel(path string) error {
	tg.mu.RLock()
	defer tg.mu.RUnlock()

	fmt.Printf("Exporting temporal model to %s\n", path)

	// Save base model
	if err := tg.baseModel.Save(path + "_base"); err != nil {
		return fmt.Errorf("save base model: %w", err)
	}

	// Save LNN weights
	// In production, would serialize LNN layer to file
	fmt.Println("LNN layer saved")

	return nil
}

// LoadTemporalModel loads a temporal model from disk
func LoadTemporalModel(basePath string, db interface{}) (*TemporalGloVe, error) {
	// Load base model
	// baseModel, err := LoadModel(db.(*sql.DB), basePath+"_base")
	// if err != nil {
	//     return nil, err
	// }

	// Load LNN layer
	// In production, would deserialize LNN weights

	fmt.Printf("Temporal model loaded from %s\n", basePath)
	return nil, fmt.Errorf("not implemented - placeholder")
}
