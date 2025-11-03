package models

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
)

// ModelLoader handles loading models from various formats
type ModelLoader struct {
	modelPaths map[string]string
}

// NewModelLoader creates a new model loader
func NewModelLoader() *ModelLoader {
	return &ModelLoader{
		modelPaths: make(map[string]string),
	}
}

// LoadModelFromSafeTensors loads a model from SafeTensors format
func (ml *ModelLoader) LoadModelFromSafeTensors(modelPath, domain string) (*ai.VaultGemma, error) {
	log.Printf("🔄 Loading model from SafeTensors: %s", modelPath)

	// Check if model path exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model path does not exist: %s", modelPath)
	}

	// Check for required files
	configPath := filepath.Join(modelPath, "config.json")
	safetensorsPath := filepath.Join(modelPath, "model.safetensors")
	indexPath := filepath.Join(modelPath, "model.safetensors.index.json")

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config.json not found in model path: %s", modelPath)
	}

	if _, err := os.Stat(safetensorsPath); os.IsNotExist(err) {
		if _, indexErr := os.Stat(indexPath); os.IsNotExist(indexErr) {
			return nil, fmt.Errorf("model.safetensors not found in model path: %s", modelPath)
		}
	}

	// Load model using the SafeTensors loader
	model, err := ai.LoadVaultGemmaFromSafetensors(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load model from SafeTensors: %w", err)
	}

	log.Printf("✅ Model loaded successfully: %s", domain)
	return model, nil
}

// LoadModelFromConfig loads a model from a configuration file
func (ml *ModelLoader) LoadModelFromConfig(configPath, domain string) (*ai.VaultGemma, error) {
	log.Printf("🔄 Loading model from config: %s", configPath)

	// Check if config file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("config file does not exist: %s", configPath)
	}

	// Load model using the basic loader
	model, err := ai.NewVaultGemma(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load model from config: %w", err)
	}

	log.Printf("✅ Model loaded successfully: %s", domain)
	return model, nil
}

// LoadModelsFromDirectory loads all models from a directory
func (ml *ModelLoader) LoadModelsFromDirectory(basePath string) (map[string]*ai.VaultGemma, error) {
	log.Printf("🔄 Loading models from directory: %s", basePath)

	models := make(map[string]*ai.VaultGemma)

	// Check if base path exists
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("base path does not exist: %s", basePath)
	}

	// Read directory contents
	entries, err := os.ReadDir(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %w", err)
	}

	// Load each model directory
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelPath := filepath.Join(basePath, entry.Name())
		domain := entry.Name()

		// Try to load as SafeTensors first
		model, err := ml.LoadModelFromSafeTensors(modelPath, domain)
		if err != nil {
			log.Printf("⚠️ Failed to load %s as SafeTensors: %v", domain, err)

			// Try to load as config
			configPath := filepath.Join(modelPath, "config.json")
			model, err = ml.LoadModelFromConfig(configPath, domain)
			if err != nil {
				log.Printf("⚠️ Failed to load %s as config: %v", domain, err)
				continue
			}
		}

		models[domain] = model
		log.Printf("✅ Loaded model: %s", domain)
	}

	return models, nil
}

// LoadModelFromPath loads a model from a specific path, auto-detecting format
func (ml *ModelLoader) LoadModelFromPath(modelPath, domain string) (*ai.VaultGemma, error) {
	log.Printf("🔄 Loading model from path: %s", modelPath)

	// Check if model path exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model path does not exist: %s", modelPath)
	}

	// Check for SafeTensors format first
	safetensorsPath := filepath.Join(modelPath, "model.safetensors")
	if _, err := os.Stat(safetensorsPath); err == nil {
		return ml.LoadModelFromSafeTensors(modelPath, domain)
	}

	// Check for config format
	configPath := filepath.Join(modelPath, "config.json")
	if _, err := os.Stat(configPath); err == nil {
		return ml.LoadModelFromConfig(configPath, domain)
	}

	// Check if the path itself is a config file
	if filepath.Ext(modelPath) == ".json" {
		return ml.LoadModelFromConfig(modelPath, domain)
	}

	return nil, fmt.Errorf("no supported model format found in: %s", modelPath)
}

// GetModelInfo returns information about a loaded model
func (ml *ModelLoader) GetModelInfo(model *ai.VaultGemma) map[string]interface{} {
	return map[string]interface{}{
		"hidden_size":       model.Config.HiddenSize,
		"num_layers":        model.Config.NumLayers,
		"num_heads":         model.Config.NumHeads,
		"vocab_size":        model.Config.VocabSize,
		"max_position_embs": model.Config.MaxPositionEmbs,
		"intermediate_size": model.Config.IntermediateSize,
		"head_dim":          model.Config.HeadDim,
		"rms_norm_eps":      model.Config.RMSNormEps,
	}
}

// ValidateModel validates that a model is properly loaded
func (ml *ModelLoader) ValidateModel(model *ai.VaultGemma) error {
	if model == nil {
		return fmt.Errorf("model is nil")
	}

	if model.Config.HiddenSize <= 0 {
		return fmt.Errorf("invalid hidden size: %d", model.Config.HiddenSize)
	}

	if model.Config.NumLayers <= 0 {
		return fmt.Errorf("invalid number of layers: %d", model.Config.NumLayers)
	}

	if model.Config.VocabSize <= 0 {
		return fmt.Errorf("invalid vocab size: %d", model.Config.VocabSize)
	}

	if model.Embed == nil {
		return fmt.Errorf("embedding layer is nil")
	}

	if model.Output == nil {
		return fmt.Errorf("output layer is nil")
	}

	if len(model.Layers) != model.Config.NumLayers {
		return fmt.Errorf("number of layers mismatch: expected %d, got %d", model.Config.NumLayers, len(model.Layers))
	}

	return nil
}

// GetSupportedFormats returns the supported model formats
func (ml *ModelLoader) GetSupportedFormats() []string {
	return []string{
		"SafeTensors",
		"JSON Config",
	}
}

// GetModelSize estimates the memory usage of a model
func (ml *ModelLoader) GetModelSize(model *ai.VaultGemma) (int64, error) {
	if model == nil {
		return 0, fmt.Errorf("model is nil")
	}

	// Estimate memory usage based on model parameters
	hiddenSize := model.Config.HiddenSize
	numLayers := model.Config.NumLayers
	vocabSize := model.Config.VocabSize
	intermediateSize := model.Config.IntermediateSize

	// Embedding layer: vocab_size * hidden_size * 4 bytes (float32)
	embeddingSize := int64(vocabSize * hiddenSize * 4)

	// Output layer: hidden_size * vocab_size * 4 bytes
	outputSize := int64(hiddenSize * vocabSize * 4)

	// Each transformer layer
	layerSize := int64(0)

	// Self-attention: 4 * hidden_size * hidden_size * 4 bytes (Q, K, V, O)
	attentionSize := int64(4 * hiddenSize * hiddenSize * 4)

	// Feed-forward: 3 * hidden_size * intermediate_size * 4 bytes (W1, W2, W3)
	feedForwardSize := int64(3 * hiddenSize * intermediateSize * 4)

	// Layer norms: 2 * hidden_size * 4 bytes
	layerNormSize := int64(2 * hiddenSize * 4)

	layerSize = attentionSize + feedForwardSize + layerNormSize
	totalLayerSize := layerSize * int64(numLayers)

	totalSize := embeddingSize + outputSize + totalLayerSize

	return totalSize, nil
}
