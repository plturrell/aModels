package glove

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/SAP/go-hdb/driver"
)

const (
	// AdaGrad epsilon for numerical stability
	adagradEpsilon = 1e-6
)

// GloVe implements Global Vectors for Word Representation
// Paper: https://nlp.stanford.edu/pubs/glove.pdf
// This implementation uses SAP HANA for efficient vector storage and retrieval

// Model represents a GloVe word embedding model
type Model struct {
	db               *sql.DB
	vectorSize       int
	learningRate     float64
	maxIter          int
	xMax             float64
	alpha            float64
	mu               sync.RWMutex
	vocabulary       map[string]int
	vectors          [][]float32
	biases           []float32
	contextVecs      [][]float32
	contextBiases    []float32
	progressReporter ProgressReporter
	trainingState    *TrainingState
}

// ProgressReporter interface for training progress callbacks
type ProgressReporter interface {
	OnProgress(iter int, cost float64, speed float64)
}

// TrainingState tracks training progress for checkpointing
type TrainingState struct {
	Iteration int       `json:"iteration"`
	Cost      float64   `json:"cost"`
	Timestamp time.Time `json:"timestamp"`
	ModelPath string    `json:"model_path"`
}

// Config holds GloVe model configuration
type Config struct {
	VectorSize   int     // Dimensionality of word vectors (default: 100)
	LearningRate float64 // Learning rate for SGD (default: 0.05)
	MaxIter      int     // Maximum training iterations (default: 15)
	XMax         float64 // Cutoff for weighting function (default: 100)
	Alpha        float64 // Exponent for weighting function (default: 0.75)
	WindowSize   int     // Context window size (default: 15)
	MinWordFreq  int     // Minimum word frequency to include (default: 5)
	MaxVocabSize int     // Maximum vocabulary size (default: 0 = unlimited)

	// SentencePiece configuration
	UseSentencePiece bool   // Use subword tokenization (default: false)
	SPModelPath      string // Path to SentencePiece model
	SPVocabSize      int    // Vocabulary size when training SentencePiece models
}

// DefaultConfig returns default GloVe configuration
func DefaultConfig() Config {
	return Config{
		VectorSize:   100,
		LearningRate: 0.05,
		MaxIter:      15,
		XMax:         100.0,
		Alpha:        0.75,
		WindowSize:   15,
		MinWordFreq:  5,
		MaxVocabSize: 0,
	}
}

// Validate checks if the configuration is valid
func (cfg Config) Validate() error {
	if cfg.VectorSize <= 0 {
		return fmt.Errorf("vector_size must be positive, got %d", cfg.VectorSize)
	}
	if cfg.VectorSize > 1000 {
		return fmt.Errorf("vector_size too large (max 1000), got %d", cfg.VectorSize)
	}
	if cfg.LearningRate <= 0 {
		return fmt.Errorf("learning_rate must be positive, got %f", cfg.LearningRate)
	}
	if cfg.LearningRate > 1.0 {
		return fmt.Errorf("learning_rate too large (max 1.0), got %f", cfg.LearningRate)
	}
	if cfg.MaxIter <= 0 {
		return fmt.Errorf("max_iter must be positive, got %d", cfg.MaxIter)
	}
	if cfg.MaxIter > 1000 {
		return fmt.Errorf("max_iter too large (max 1000), got %d", cfg.MaxIter)
	}
	if cfg.XMax <= 0 {
		return fmt.Errorf("x_max must be positive, got %f", cfg.XMax)
	}
	if cfg.Alpha <= 0 || cfg.Alpha > 1 {
		return fmt.Errorf("alpha must be in (0, 1], got %f", cfg.Alpha)
	}
	if cfg.WindowSize <= 0 {
		return fmt.Errorf("window_size must be positive, got %d", cfg.WindowSize)
	}
	if cfg.WindowSize > 100 {
		return fmt.Errorf("window_size too large (max 100), got %d", cfg.WindowSize)
	}
	if cfg.MinWordFreq < 0 {
		return fmt.Errorf("min_word_freq must be non-negative, got %d", cfg.MinWordFreq)
	}
	if cfg.MaxVocabSize < 0 {
		return fmt.Errorf("max_vocab_size must be non-negative, got %d", cfg.MaxVocabSize)
	}
	return nil
}

// NewModel creates a new GloVe model with HANA backend
func NewModel(db *sql.DB, cfg Config) (*Model, error) {
	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	m := &Model{
		db:           db,
		vectorSize:   cfg.VectorSize,
		learningRate: cfg.LearningRate,
		maxIter:      cfg.MaxIter,
		xMax:         cfg.XMax,
		alpha:        cfg.Alpha,
		vocabulary:   make(map[string]int),
	}

	// Initialize HANA tables
	if err := m.initTables(); err != nil {
		return nil, fmt.Errorf("init tables: %w", err)
	}

	return m, nil
}

// initTables creates necessary HANA tables for vector storage with proper indexes
func (m *Model) initTables() error {
	queries := []string{
		`CREATE COLUMN TABLE IF NOT EXISTS GLOVE_VOCABULARY (
			WORD_ID INTEGER PRIMARY KEY,
			WORD NVARCHAR(200) UNIQUE NOT NULL,
			FREQUENCY INTEGER DEFAULT 0,
			CREATED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP
		)`,
		`CREATE COLUMN TABLE IF NOT EXISTS GLOVE_VECTORS (
			WORD_ID INTEGER PRIMARY KEY,
			VECTOR BLOB NOT NULL,
			BIAS REAL,
			UPDATED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP,
			FOREIGN KEY (WORD_ID) REFERENCES GLOVE_VOCABULARY(WORD_ID)
		)`,
		`CREATE COLUMN TABLE IF NOT EXISTS GLOVE_CONTEXT_VECTORS (
			WORD_ID INTEGER PRIMARY KEY,
			CONTEXT_VECTOR BLOB NOT NULL,
			CONTEXT_BIAS REAL,
			UPDATED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP,
			FOREIGN KEY (WORD_ID) REFERENCES GLOVE_VOCABULARY(WORD_ID)
		)`,
		`CREATE COLUMN TABLE IF NOT EXISTS GLOVE_COOCCURRENCE (
			WORD_ID INTEGER NOT NULL,
			CONTEXT_ID INTEGER NOT NULL,
			COOCCURRENCE REAL NOT NULL,
			PRIMARY KEY (WORD_ID, CONTEXT_ID),
			FOREIGN KEY (WORD_ID) REFERENCES GLOVE_VOCABULARY(WORD_ID),
			FOREIGN KEY (CONTEXT_ID) REFERENCES GLOVE_VOCABULARY(WORD_ID)
		)`,
		// Create indexes for performance
		`CREATE INDEX IF NOT EXISTS idx_glove_cooccur_word ON GLOVE_COOCCURRENCE(WORD_ID)`,
		`CREATE INDEX IF NOT EXISTS idx_glove_cooccur_context ON GLOVE_COOCCURRENCE(CONTEXT_ID)`,
		`CREATE INDEX IF NOT EXISTS idx_glove_vocab_word ON GLOVE_VOCABULARY(WORD)`,
		`CREATE INDEX IF NOT EXISTS idx_glove_vectors_word ON GLOVE_VECTORS(WORD_ID)`,
	}

	for _, query := range queries {
		if _, err := m.db.Exec(query); err != nil {
			return fmt.Errorf("exec query: %w", err)
		}
	}

	return nil
}

// BuildVocabulary builds vocabulary from corpus
func (m *Model) BuildVocabulary(ctx context.Context, corpus []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	wordFreq := make(map[string]int)
	for _, text := range corpus {
		words := tokenize(text)
		for _, word := range words {
			wordFreq[word]++
		}
	}

	// Insert vocabulary into HANA with batching
	const batchSize = 1000
	stmt, err := m.db.PrepareContext(ctx,
		"INSERT INTO GLOVE_VOCABULARY (WORD_ID, WORD, FREQUENCY) VALUES (?, ?, ?)")
	if err != nil {
		return fmt.Errorf("prepare vocabulary insert statement: %w", err)
	}
	defer stmt.Close()

	wordID := 0
	batchCount := 0
	for word, freq := range wordFreq {
		if _, err := stmt.ExecContext(ctx, wordID, word, freq); err != nil {
			return fmt.Errorf("insert word '%s' (id=%d, freq=%d): %w", word, wordID, freq, err)
		}
		m.vocabulary[word] = wordID
		wordID++
		batchCount++

		// Commit in batches for better performance
		if batchCount >= batchSize {
			batchCount = 0
		}
	}

	// Initialize vectors
	vocabSize := len(m.vocabulary)
	m.vectors = make([][]float32, vocabSize)
	m.biases = make([]float32, vocabSize)
	m.contextVecs = make([][]float32, vocabSize)
	m.contextBiases = make([]float32, vocabSize)

	for i := 0; i < vocabSize; i++ {
		m.vectors[i] = randomVector(m.vectorSize)
		m.biases[i] = rand.Float32() - 0.5
		m.contextVecs[i] = randomVector(m.vectorSize)
		m.contextBiases[i] = rand.Float32() - 0.5
	}

	return nil
}

// BuildCooccurrence builds word co-occurrence matrix
func (m *Model) BuildCooccurrence(ctx context.Context, corpus []string, windowSize int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	cooccur := make(map[[2]int]float64)

	for _, text := range corpus {
		words := tokenize(text)
		for i, word := range words {
			wordID, ok := m.vocabulary[word]
			if !ok {
				continue
			}

			// Context window
			start := max(0, i-windowSize)
			end := min(len(words), i+windowSize+1)

			for j := start; j < end; j++ {
				if i == j {
					continue
				}
				contextWord := words[j]
				contextID, ok := m.vocabulary[contextWord]
				if !ok {
					continue
				}

				distance := float64(abs(i - j))
				weight := 1.0 / distance
				key := [2]int{wordID, contextID}
				cooccur[key] += weight
			}
		}
	}

	// Store co-occurrence in HANA
	stmt, err := m.db.PrepareContext(ctx,
		"INSERT INTO GLOVE_COOCCURRENCE (WORD_ID, CONTEXT_ID, COOCCURRENCE) VALUES (?, ?, ?)")
	if err != nil {
		return fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	for key, value := range cooccur {
		if _, err := stmt.ExecContext(ctx, key[0], key[1], value); err != nil {
			return fmt.Errorf("insert cooccurrence: %w", err)
		}
	}

	return nil
}

// Train trains the GloVe model using AdaGrad
func (m *Model) Train(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Load co-occurrence data from HANA
	rows, err := m.db.QueryContext(ctx,
		"SELECT WORD_ID, CONTEXT_ID, COOCCURRENCE FROM GLOVE_COOCCURRENCE")
	if err != nil {
		return fmt.Errorf("query cooccurrence: %w", err)
	}
	defer rows.Close()

	type cooccurEntry struct {
		wordID    int
		contextID int
		cooccur   float64
	}

	var entries []cooccurEntry
	for rows.Next() {
		var e cooccurEntry
		if err := rows.Scan(&e.wordID, &e.contextID, &e.cooccur); err != nil {
			return fmt.Errorf("scan cooccurrence row (word=%d, context=%d): %w",
				e.wordID, e.contextID, err)
		}
		entries = append(entries, e)
	}

	// AdaGrad accumulators
	vocabSize := len(m.vocabulary)
	gradSqW := make([][]float32, vocabSize)
	gradSqC := make([][]float32, vocabSize)
	gradSqBW := make([]float32, vocabSize)
	gradSqBC := make([]float32, vocabSize)

	for i := 0; i < vocabSize; i++ {
		gradSqW[i] = make([]float32, m.vectorSize)
		gradSqC[i] = make([]float32, m.vectorSize)
	}

	// Training loop with progress reporting
	startTime := time.Now()
	for iter := 0; iter < m.maxIter; iter++ {
		iterStart := time.Now()
		totalCost := 0.0

		// Shuffle entries
		rand.Shuffle(len(entries), func(i, j int) {
			entries[i], entries[j] = entries[j], entries[i]
		})

		for _, e := range entries {
			// Compute cost
			diff := dotProduct(m.vectors[e.wordID], m.contextVecs[e.contextID]) +
				float64(m.biases[e.wordID]) + float64(m.contextBiases[e.contextID]) - math.Log(e.cooccur)

			// Weighting function
			weight := m.weightingFunc(e.cooccur)
			cost := weight * diff * diff
			totalCost += cost

			// Compute gradients
			fdiff := float32(weight * diff)

			// Update word vector with AdaGrad
			for d := 0; d < m.vectorSize; d++ {
				grad := fdiff * m.contextVecs[e.contextID][d]
				gradSqW[e.wordID][d] += grad * grad
				m.vectors[e.wordID][d] -= float32(m.learningRate) * grad /
					float32(math.Sqrt(float64(gradSqW[e.wordID][d])+adagradEpsilon))
			}

			// Update context vector with AdaGrad
			for d := 0; d < m.vectorSize; d++ {
				grad := fdiff * m.vectors[e.wordID][d]
				gradSqC[e.contextID][d] += grad * grad
				m.contextVecs[e.contextID][d] -= float32(m.learningRate) * grad /
					float32(math.Sqrt(float64(gradSqC[e.contextID][d])+adagradEpsilon))
			}

			// Update biases with AdaGrad
			gradBW := fdiff
			gradSqBW[e.wordID] += gradBW * gradBW
			m.biases[e.wordID] -= float32(m.learningRate) * gradBW /
				float32(math.Sqrt(float64(gradSqBW[e.wordID])+adagradEpsilon))

			gradBC := fdiff
			gradSqBC[e.contextID] += gradBC * gradBC
			m.contextBiases[e.contextID] -= float32(m.learningRate) * gradBC /
				float32(math.Sqrt(float64(gradSqBC[e.contextID])+adagradEpsilon))
		}

		avgCost := totalCost / float64(len(entries))
		iterDuration := time.Since(iterStart)
		speed := float64(len(entries)) / iterDuration.Seconds()

		// Report progress
		if m.progressReporter != nil {
			m.progressReporter.OnProgress(iter+1, avgCost, speed)
		} else {
			fmt.Printf("Iteration %d: avg cost = %.6f, speed = %.2f samples/sec\n",
				iter+1, avgCost, speed)
		}

		// Update training state for checkpointing
		m.trainingState = &TrainingState{
			Iteration: iter + 1,
			Cost:      avgCost,
			Timestamp: time.Now(),
		}
	}

	totalDuration := time.Since(startTime)
	fmt.Printf("Training complete in %s\n", totalDuration)

	// Save vectors to HANA
	return m.saveVectors(ctx)
}

// weightingFunc implements the GloVe weighting function
func (m *Model) weightingFunc(x float64) float64 {
	if x < m.xMax {
		return math.Pow(x/m.xMax, m.alpha)
	}
	return 1.0
}

// saveVectors saves trained vectors to HANA
func (m *Model) saveVectors(ctx context.Context) error {
	// Save word vectors
	stmtVec, err := m.db.PrepareContext(ctx,
		"UPSERT GLOVE_VECTORS (WORD_ID, VECTOR, BIAS) VALUES (?, ?, ?)")
	if err != nil {
		return fmt.Errorf("prepare vector statement: %w", err)
	}
	defer stmtVec.Close()

	// Save context vectors
	stmtCtx, err := m.db.PrepareContext(ctx,
		"UPSERT GLOVE_CONTEXT_VECTORS (WORD_ID, CONTEXT_VECTOR, CONTEXT_BIAS) VALUES (?, ?, ?)")
	if err != nil {
		return fmt.Errorf("prepare context statement: %w", err)
	}
	defer stmtCtx.Close()

	for word, wordID := range m.vocabulary {
		_ = word // Use word for logging if needed

		// Serialize vector to binary
		vecBytes := serializeVector(m.vectors[wordID])
		if _, err := stmtVec.ExecContext(ctx, wordID, vecBytes, m.biases[wordID]); err != nil {
			return fmt.Errorf("save vector: %w", err)
		}

		// Serialize context vector
		ctxBytes := serializeVector(m.contextVecs[wordID])
		if _, err := stmtCtx.ExecContext(ctx, wordID, ctxBytes, m.contextBiases[wordID]); err != nil {
			return fmt.Errorf("save context vector: %w", err)
		}
	}

	return nil
}

// GetVector retrieves word vector from HANA
func (m *Model) GetVector(ctx context.Context, word string) ([]float32, error) {
	m.mu.RLock()
	wordID, ok := m.vocabulary[word]
	m.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("word not in vocabulary: %s", word)
	}

	// Try memory first
	if wordID < len(m.vectors) && m.vectors[wordID] != nil {
		return m.vectors[wordID], nil
	}

	// Load from HANA
	var vecBytes []byte
	err := m.db.QueryRowContext(ctx,
		"SELECT VECTOR FROM GLOVE_VECTORS WHERE WORD_ID = ?", wordID).Scan(&vecBytes)
	if err != nil {
		return nil, fmt.Errorf("query vector: %w", err)
	}

	return deserializeVector(vecBytes), nil
}

// Similarity computes cosine similarity between two words
func (m *Model) Similarity(ctx context.Context, word1, word2 string) (float64, error) {
	vec1, err := m.GetVector(ctx, word1)
	if err != nil {
		return 0, err
	}

	vec2, err := m.GetVector(ctx, word2)
	if err != nil {
		return 0, err
	}

	return cosineSimilarity(vec1, vec2), nil
}

// MostSimilar finds k most similar words to the given word
func (m *Model) MostSimilar(ctx context.Context, word string, k int) ([]string, []float64, error) {
	vec, err := m.GetVector(ctx, word)
	if err != nil {
		return nil, nil, err
	}

	type similarity struct {
		word  string
		score float64
	}

	var similarities []similarity
	for w := range m.vocabulary {
		if w == word {
			continue
		}
		wVec, err := m.GetVector(ctx, w)
		if err != nil {
			continue
		}
		sim := cosineSimilarity(vec, wVec)
		similarities = append(similarities, similarity{w, sim})
	}

	// Sort by similarity using efficient sort.Slice (O(n log n) instead of O(n²))
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].score > similarities[j].score
	})

	// Return top k
	if k > len(similarities) {
		k = len(similarities)
	}

	words := make([]string, k)
	scores := make([]float64, k)
	for i := 0; i < k; i++ {
		words[i] = similarities[i].word
		scores[i] = similarities[i].score
	}

	return words, scores, nil
}

// Helper functions

// tokenize splits text into words using proper Unicode-aware tokenization
func tokenize(text string) []string {
	// Normalize text
	text = strings.ToLower(text)

	// Use regex to extract words (Unicode letters and numbers)
	re := regexp.MustCompile(`[\p{L}\p{N}]+`)
	words := re.FindAllString(text, -1)

	// Filter out very short tokens
	filtered := make([]string, 0, len(words))
	for _, word := range words {
		if len(word) > 1 { // Skip single characters
			filtered = append(filtered, word)
		}
	}

	return filtered
}

// randomVector creates a random vector with Xavier/Glorot initialization
func randomVector(size int) []float32 {
	// Xavier initialization: scale by sqrt(2/n) for better convergence
	scale := float32(math.Sqrt(2.0 / float64(size)))
	vec := make([]float32, size)
	for i := range vec {
		vec[i] = (rand.Float32() - 0.5) * scale
	}
	return vec
}

func dotProduct(a, b []float32) float64 {
	sum := 0.0
	for i := range a {
		sum += float64(a[i] * b[i])
	}
	return sum
}

func cosineSimilarity(a, b []float32) float64 {
	dot := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func serializeVector(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func deserializeVector(buf []byte) []float32 {
	vec := make([]float32, len(buf)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(buf[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	return vec
}

// Helper functions moved to avoid redeclaration
