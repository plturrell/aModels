package server

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ModelGPURequirements defines GPU requirements for a model
type ModelGPURequirements struct {
	ModelType    string // "gemma-7b", "gemma-2b", "vaultgemma-1b", etc.
	MinMemoryMB  int64  // Minimum GPU memory needed
	RequiredGPUs int    // Number of GPUs (1 for most)
	Dedicated    bool   // Whether model needs dedicated GPU
	Priority     int    // Allocation priority (higher = more important)
}

// ModelRegistry manages model metadata and GPU requirements
type ModelRegistry struct {
	requirements map[string]*ModelGPURequirements
}

// NewModelRegistry creates a new model registry with default requirements
func NewModelRegistry() *ModelRegistry {
	registry := &ModelRegistry{
		requirements: make(map[string]*ModelGPURequirements),
	}
	
	// Register known models with their GPU requirements
	registry.registerDefaultModels()
	
	return registry
}

// registerDefaultModels registers GPU requirements for known models
func (mr *ModelRegistry) registerDefaultModels() {
	// Large models (7B+) - need dedicated GPU
	mr.requirements["gemma-7b"] = &ModelGPURequirements{
		ModelType:    "gemma-7b",
		MinMemoryMB:  16384, // 16GB
		RequiredGPUs: 1,
		Dedicated:    true,
		Priority:     8,
	}
	mr.requirements["gemma-7b-it"] = &ModelGPURequirements{
		ModelType:    "gemma-7b-it",
		MinMemoryMB:  16384,
		RequiredGPUs: 1,
		Dedicated:    true,
		Priority:     8,
	}
	
	// Medium models (2B-3B) - can share GPU
	mr.requirements["gemma-2b"] = &ModelGPURequirements{
		ModelType:    "gemma-2b",
		MinMemoryMB:  4096, // 4GB
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     7,
	}
	mr.requirements["gemma-2b-it"] = &ModelGPURequirements{
		ModelType:    "gemma-2b-it",
		MinMemoryMB:  4096,
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     7,
	}
	
	// Small models (1B-2B) - can share GPU
	mr.requirements["vaultgemma-1b"] = &ModelGPURequirements{
		ModelType:    "vaultgemma-1b",
		MinMemoryMB:  2048, // 2GB
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     6,
	}
	mr.requirements["vaultgemma"] = &ModelGPURequirements{
		ModelType:    "vaultgemma-1b",
		MinMemoryMB:  2048,
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     6,
	}
	
	// Phi-3.5-mini - can share GPU
	mr.requirements["phi-3.5-mini"] = &ModelGPURequirements{
		ModelType:    "phi-3.5-mini",
		MinMemoryMB:  4096, // 4GB
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     7,
	}
	mr.requirements["phi-3.5-mini-instruct"] = &ModelGPURequirements{
		ModelType:    "phi-3.5-mini-instruct",
		MinMemoryMB:  4096,
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     7,
	}
	
	// Granite 4.0 - can share GPU
	mr.requirements["granite-4.0"] = &ModelGPURequirements{
		ModelType:    "granite-4.0",
		MinMemoryMB:  4096, // 4GB
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     7,
	}
	mr.requirements["granite-4.0-h-micro"] = &ModelGPURequirements{
		ModelType:    "granite-4.0-h-micro",
		MinMemoryMB:  4096,
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     7,
	}
}

// GetRequirements returns GPU requirements for a model
func (mr *ModelRegistry) GetRequirements(modelPath, modelName string) *ModelGPURequirements {
	// Try to match by model name first
	if req, ok := mr.requirements[strings.ToLower(modelName)]; ok {
		return req
	}
	
	// Try to detect from model path
	detectedType := mr.detectModelType(modelPath)
	if req, ok := mr.requirements[detectedType]; ok {
		return req
	}
	
	// Default requirements for unknown models
	return mr.getDefaultRequirements(modelPath)
}

// detectModelType detects model type from path
func (mr *ModelRegistry) detectModelType(modelPath string) string {
	pathLower := strings.ToLower(modelPath)
	
	// Check for known patterns
	if strings.Contains(pathLower, "gemma-7b") || strings.Contains(pathLower, "7b") {
		return "gemma-7b"
	}
	if strings.Contains(pathLower, "gemma-2b") || strings.Contains(pathLower, "2b") {
		return "gemma-2b"
	}
	if strings.Contains(pathLower, "vaultgemma") || strings.Contains(pathLower, "vault-gemma") {
		return "vaultgemma-1b"
	}
	if strings.Contains(pathLower, "phi-3.5") || strings.Contains(pathLower, "phi3.5") {
		return "phi-3.5-mini"
	}
	if strings.Contains(pathLower, "granite-4.0") || strings.Contains(pathLower, "granite4.0") {
		return "granite-4.0"
	}
	
	return ""
}

// getDefaultRequirements returns default GPU requirements based on model size estimation
func (mr *ModelRegistry) getDefaultRequirements(modelPath string) *ModelGPURequirements {
	// Try to estimate from path or use conservative defaults
	pathLower := strings.ToLower(modelPath)
	
	// Check if it's a large model (7B+)
	if strings.Contains(pathLower, "7b") || strings.Contains(pathLower, "8b") || 
	   strings.Contains(pathLower, "13b") || strings.Contains(pathLower, "70b") {
		return &ModelGPURequirements{
			ModelType:    "unknown-large",
			MinMemoryMB:  16384,
			RequiredGPUs: 1,
			Dedicated:    true,
			Priority:     7,
		}
	}
	
	// Default to small model that can share
	return &ModelGPURequirements{
		ModelType:    "unknown-small",
		MinMemoryMB:  4096,
		RequiredGPUs: 1,
		Dedicated:    false,
		Priority:     6,
	}
}

// EstimateModelSize estimates model size in billions of parameters from config
func (mr *ModelRegistry) EstimateModelSize(numLayers, hiddenSize, vocabSize int) float64 {
	// Rough estimation: embedding + layers + output
	// This is a simplified calculation
	embeddingParams := float64(vocabSize * hiddenSize)
	layerParams := float64(numLayers * (4*hiddenSize*hiddenSize + 3*hiddenSize*hiddenSize*4)) // attention + FFN
	outputParams := float64(hiddenSize * vocabSize)
	
	totalParams := (embeddingParams + layerParams + outputParams) / 1e9 // Convert to billions
	return totalParams
}

// IsLargeModel determines if a model is considered "large" (needs dedicated GPU)
func (mr *ModelRegistry) IsLargeModel(modelPath, modelName string, numLayers, hiddenSize, vocabSize int) bool {
	// Check explicit requirements first
	req := mr.GetRequirements(modelPath, modelName)
	if req != nil && req.Dedicated {
		return true
	}
	
	// Estimate from parameters
	modelSizeB := mr.EstimateModelSize(numLayers, hiddenSize, vocabSize)
	
	// Get threshold from environment or use default (3B)
	threshold := 3.0
	if envThreshold := os.Getenv("GPU_MIN_MODEL_SIZE_FOR_DEDICATED"); envThreshold != "" {
		if parsed, err := fmt.Sscanf(envThreshold, "%f", &threshold); err == nil && parsed == 1 {
			// threshold already set
		}
	}
	
	return modelSizeB >= threshold
}

// GetModelTypeFromPath extracts model type identifier from path
func (mr *ModelRegistry) GetModelTypeFromPath(modelPath string) string {
	// Extract directory name or filename
	base := filepath.Base(modelPath)
	dir := filepath.Dir(modelPath)
	parentDir := filepath.Base(dir)
	
	// Try parent directory name first (e.g., "gemma-7b-it-tensorrt")
	if detected := mr.detectModelType(parentDir); detected != "" {
		return detected
	}
	
	// Try base name
	if detected := mr.detectModelType(base); detected != "" {
		return detected
	}
	
	// Try full path
	return mr.detectModelType(modelPath)
}

