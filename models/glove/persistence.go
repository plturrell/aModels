package glove

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
)

// Save saves the model to disk for persistence
func (m *Model) Save(modelPath string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Create model metadata
	metadata := struct {
		VectorSize   int            `json:"vector_size"`
		VocabSize    int            `json:"vocab_size"`
		Vocabulary   map[string]int `json:"vocabulary"`
		LearningRate float64        `json:"learning_rate"`
		MaxIter      int            `json:"max_iter"`
		XMax         float64        `json:"x_max"`
		Alpha        float64        `json:"alpha"`
	}{
		VectorSize:   m.vectorSize,
		VocabSize:    len(m.vocabulary),
		Vocabulary:   m.vocabulary,
		LearningRate: m.learningRate,
		MaxIter:      m.maxIter,
		XMax:         m.xMax,
		Alpha:        m.alpha,
	}

	// Save metadata
	metadataFile, err := os.Create(modelPath + ".meta.json")
	if err != nil {
		return fmt.Errorf("create metadata file: %w", err)
	}
	defer metadataFile.Close()

	if err := json.NewEncoder(metadataFile).Encode(metadata); err != nil {
		return fmt.Errorf("encode metadata: %w", err)
	}

	// Vectors are already in HANA, so we just need to save metadata
	// For full offline capability, we could also export vectors to file
	return nil
}

// LoadModel loads a model from disk and HANA
func LoadModel(db *sql.DB, modelPath string) (*Model, error) {
	// Load metadata
	metadataFile, err := os.Open(modelPath + ".meta.json")
	if err != nil {
		return nil, fmt.Errorf("open metadata file: %w", err)
	}
	defer metadataFile.Close()

	var metadata struct {
		VectorSize   int            `json:"vector_size"`
		VocabSize    int            `json:"vocab_size"`
		Vocabulary   map[string]int `json:"vocabulary"`
		LearningRate float64        `json:"learning_rate"`
		MaxIter      int            `json:"max_iter"`
		XMax         float64        `json:"x_max"`
		Alpha        float64        `json:"alpha"`
	}

	if err := json.NewDecoder(metadataFile).Decode(&metadata); err != nil {
		return nil, fmt.Errorf("decode metadata: %w", err)
	}

	// Create model
	cfg := Config{
		VectorSize:   metadata.VectorSize,
		LearningRate: metadata.LearningRate,
		MaxIter:      metadata.MaxIter,
		XMax:         metadata.XMax,
		Alpha:        metadata.Alpha,
	}

	model, err := NewModel(db, cfg)
	if err != nil {
		return nil, fmt.Errorf("create model: %w", err)
	}

	// Restore vocabulary
	model.vocabulary = metadata.Vocabulary

	// Vectors will be loaded from HANA on demand
	return model, nil
}

// SaveCheckpoint saves training state for resumption
func (m *Model) SaveCheckpoint(ctx context.Context, checkpointPath string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.trainingState == nil {
		return fmt.Errorf("no training state to save")
	}

	// Update checkpoint path
	m.trainingState.ModelPath = checkpointPath

	// Save checkpoint
	file, err := os.Create(checkpointPath)
	if err != nil {
		return fmt.Errorf("create checkpoint file: %w", err)
	}
	defer file.Close()

	if err := json.NewEncoder(file).Encode(m.trainingState); err != nil {
		return fmt.Errorf("encode checkpoint: %w", err)
	}

	// Save current vectors to HANA
	if err := m.saveVectors(ctx); err != nil {
		return fmt.Errorf("save vectors: %w", err)
	}

	return nil
}

// LoadCheckpoint loads training state for resumption
func (m *Model) LoadCheckpoint(checkpointPath string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	file, err := os.Open(checkpointPath)
	if err != nil {
		return fmt.Errorf("open checkpoint file: %w", err)
	}
	defer file.Close()

	var state TrainingState
	if err := json.NewDecoder(file).Decode(&state); err != nil {
		return fmt.Errorf("decode checkpoint: %w", err)
	}

	m.trainingState = &state
	return nil
}

// SetProgressReporter sets a callback for training progress
func (m *Model) SetProgressReporter(reporter ProgressReporter) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.progressReporter = reporter
}

// DefaultProgressReporter provides console output for training progress
type DefaultProgressReporter struct{}

func (r *DefaultProgressReporter) OnProgress(iter int, cost float64, speed float64) {
	fmt.Printf("Iteration %d: avg cost = %.6f, speed = %.2f samples/sec\n",
		iter, cost, speed)
}
