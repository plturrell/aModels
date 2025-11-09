package server

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/gguf"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/transformers"
	llama "github.com/go-skynet/go-llama.cpp"
)

// ModelCache manages lazy loading and caching of models
type ModelCache struct {
	// Safetensors models
	safetensorModels map[string]*ai.VaultGemma
	safetensorPaths  map[string]string
	safetensorMu     sync.RWMutex

	// GGUF models
	ggufModels map[string]*gguf.Model
	ggufPaths  map[string]string
	ggufMu     sync.RWMutex

	// Transformers clients
	transformerClients map[string]*transformers.Client
	transformerConfigs map[string]*domain.TransformersConfig
	transformerMu      sync.RWMutex

	// Cache management
	maxMemoryMB      int64
	currentMemoryMB  int64
	currentMemoryMu  sync.RWMutex
	accessTimes      map[string]time.Time
	accessTimesMu    sync.RWMutex
	loadingInProgress map[string]chan struct{}
	loadingMu        sync.Mutex
	modelMemoryMB    map[string]int64 // Track memory per domain
	modelMemoryMu    sync.RWMutex
	loadingTimes     map[string]time.Duration // Track loading times per domain
	loadingTimesMu   sync.RWMutex

	// Domain manager for configuration
	domainManager *domain.DomainManager
}

// NewModelCache creates a new model cache with lazy loading support
func NewModelCache(domainManager *domain.DomainManager, maxMemoryMB int64) *ModelCache {
	if maxMemoryMB <= 0 {
		maxMemoryMB = 8192 // Default 8GB
	}

	return &ModelCache{
		safetensorModels:   make(map[string]*ai.VaultGemma),
		safetensorPaths:    make(map[string]string),
		ggufModels:         make(map[string]*gguf.Model),
		ggufPaths:          make(map[string]string),
		transformerClients: make(map[string]*transformers.Client),
		transformerConfigs: make(map[string]*domain.TransformersConfig),
		maxMemoryMB:        maxMemoryMB,
		currentMemoryMB:   0,
		accessTimes:        make(map[string]time.Time),
		loadingInProgress:  make(map[string]chan struct{}),
		modelMemoryMB:      make(map[string]int64),
		loadingTimes:       make(map[string]time.Duration),
		domainManager:      domainManager,
	}
}

// RegisterSafetensorModel registers a safetensors model path for lazy loading
func (mc *ModelCache) RegisterSafetensorModel(domain, modelPath string) {
	mc.safetensorMu.Lock()
	defer mc.safetensorMu.Unlock()
	mc.safetensorPaths[domain] = modelPath
}

// RegisterGGUFModel registers a GGUF model path for lazy loading
func (mc *ModelCache) RegisterGGUFModel(domain, modelPath string) {
	mc.ggufMu.Lock()
	defer mc.ggufMu.Unlock()
	mc.ggufPaths[domain] = modelPath
}

// RegisterTransformerClient registers a transformers client configuration
func (mc *ModelCache) RegisterTransformerClient(domain string, config *domain.TransformersConfig) {
	mc.transformerMu.Lock()
	defer mc.transformerMu.Unlock()
	mc.transformerConfigs[domain] = config
	if config != nil {
		timeout := time.Duration(config.TimeoutSeconds) * time.Second
		mc.transformerClients[domain] = transformers.NewClient(config.Endpoint, config.ModelName, timeout)
	}
}

// GetSafetensorModel gets a safetensors model, loading it lazily if needed
func (mc *ModelCache) GetSafetensorModel(ctx context.Context, domain string) (*ai.VaultGemma, error) {
	mc.safetensorMu.RLock()
	model, exists := mc.safetensorModels[domain]
	path := mc.safetensorPaths[domain]
	mc.safetensorMu.RUnlock()

	if exists && model != nil {
		mc.updateAccessTime(domain)
		return model, nil
	}

	if path == "" {
		return nil, fmt.Errorf("no model path registered for domain %s", domain)
	}

	// Check if loading is in progress
	mc.loadingMu.Lock()
	loadingChan, loading := mc.loadingInProgress[domain]
	if loading {
		mc.loadingMu.Unlock()
		// Wait for loading to complete
		select {
		case <-loadingChan:
			mc.safetensorMu.RLock()
			model = mc.safetensorModels[domain]
			mc.safetensorMu.RUnlock()
			if model != nil {
				mc.updateAccessTime(domain)
				return model, nil
			}
			return nil, fmt.Errorf("model loading failed for domain %s", domain)
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Start loading
	loadingChan = make(chan struct{})
	mc.loadingInProgress[domain] = loadingChan
	mc.loadingMu.Unlock()

	// Load model in background
	go func() {
		defer close(loadingChan)
		defer func() {
			mc.loadingMu.Lock()
			delete(mc.loadingInProgress, domain)
			mc.loadingMu.Unlock()
		}()

		log.Printf("📥 Lazy loading safetensors model for domain %s from %s...", domain, path)
		start := time.Now()

		loadedModel, err := ai.LoadVaultGemmaFromSafetensors(path)
		if err != nil {
			log.Printf("❌ Failed to lazy load model for domain %s: %v", domain, err)
			return
		}

		duration := time.Since(start)
		log.Printf("✅ Model loaded for domain %s in %.2fs", domain, duration.Seconds())

		// Track loading time
		mc.loadingTimesMu.Lock()
		mc.loadingTimes[domain] = duration
		mc.loadingTimesMu.Unlock()

		// Estimate memory usage
		memoryMB := mc.estimateSafetensorMemory(loadedModel)
		
		mc.safetensorMu.Lock()
		mc.safetensorModels[domain] = loadedModel
		mc.safetensorMu.Unlock()

		// Update memory tracking
		mc.addModelMemory(domain, memoryMB)

		mc.updateAccessTime(domain)
	}()

	// Wait for loading to complete
	select {
	case <-loadingChan:
		mc.safetensorMu.RLock()
		model = mc.safetensorModels[domain]
		mc.safetensorMu.RUnlock()
		if model != nil {
			mc.updateAccessTime(domain)
			return model, nil
		}
		return nil, fmt.Errorf("model loading failed for domain %s", domain)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// GetGGUFModel gets a GGUF model, loading it lazily if needed
func (mc *ModelCache) GetGGUFModel(ctx context.Context, domain string) (*gguf.Model, error) {
	mc.ggufMu.RLock()
	model, exists := mc.ggufModels[domain]
	path := mc.ggufPaths[domain]
	mc.ggufMu.RUnlock()

	if exists && model != nil {
		mc.updateAccessTime(domain)
		return model, nil
	}

	if path == "" {
		return nil, fmt.Errorf("no GGUF model path registered for domain %s", domain)
	}

	// Check if loading is in progress
	mc.loadingMu.Lock()
	loadingChan, loading := mc.loadingInProgress[domain]
	if loading {
		mc.loadingMu.Unlock()
		select {
		case <-loadingChan:
			mc.ggufMu.RLock()
			model = mc.ggufModels[domain]
			mc.ggufMu.RUnlock()
			if model != nil {
				mc.updateAccessTime(domain)
				return model, nil
			}
			return nil, fmt.Errorf("GGUF model loading failed for domain %s", domain)
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	loadingChan = make(chan struct{})
	mc.loadingInProgress[domain] = loadingChan
	mc.loadingMu.Unlock()

	// Load model in background
	go func() {
		defer close(loadingChan)
		defer func() {
			mc.loadingMu.Lock()
			delete(mc.loadingInProgress, domain)
			mc.loadingMu.Unlock()
		}()

		log.Printf("📥 Lazy loading GGUF model for domain %s from %s...", domain, path)
		start := time.Now()

		// Determine GPU layers configuration
		gpuLayers := mc.getGPULayersForDomain(domain)

		var loadedModel *gguf.Model
		var err error
		if gpuLayers != 0 {
			loadedModel, err = gguf.Load(path, llama.SetGPULayers(gpuLayers))
			if err != nil {
				// Fallback to CPU if GPU loading fails
				log.Printf("⚠️  GPU loading failed for domain %s, falling back to CPU: %v", domain, err)
				loadedModel, err = gguf.Load(path)
				gpuLayers = 0
			}
		} else {
			loadedModel, err = gguf.Load(path)
		}

		if err != nil {
			log.Printf("❌ Failed to lazy load GGUF model for domain %s: %v", domain, err)
			return
		}

		duration := time.Since(start)
		if gpuLayers != 0 {
			log.Printf("✅ GGUF model loaded for domain %s with GPU in %.2fs", domain, duration.Seconds())
		} else {
			log.Printf("✅ GGUF model loaded for domain %s (CPU) in %.2fs", domain, duration.Seconds())
		}

		// Track loading time
		mc.loadingTimesMu.Lock()
		mc.loadingTimes[domain] = duration
		mc.loadingTimesMu.Unlock()

		// Estimate memory usage (GGUF models are typically smaller due to quantization)
		memoryMB := mc.estimateGGUFMemory(path)

		mc.ggufMu.Lock()
		mc.ggufModels[domain] = loadedModel
		mc.ggufMu.Unlock()

		// Update memory tracking
		mc.addModelMemory(domain, memoryMB)

		mc.updateAccessTime(domain)
	}()

	select {
	case <-loadingChan:
		mc.ggufMu.RLock()
		model = mc.ggufModels[domain]
		mc.ggufMu.RUnlock()
		if model != nil {
			mc.updateAccessTime(domain)
			return model, nil
		}
		return nil, fmt.Errorf("GGUF model loading failed for domain %s", domain)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// GetTransformerClient gets a transformers client
func (mc *ModelCache) GetTransformerClient(domain string) (*transformers.Client, bool) {
	mc.transformerMu.RLock()
	defer mc.transformerMu.RUnlock()
	client, exists := mc.transformerClients[domain]
	return client, exists
}

// updateAccessTime updates the last access time for a model
func (mc *ModelCache) updateAccessTime(domain string) {
	mc.accessTimesMu.Lock()
	defer mc.accessTimesMu.Unlock()
	mc.accessTimes[domain] = time.Now()
}

// StartBackgroundCleanup starts a background goroutine to unload unused models
func (mc *ModelCache) StartBackgroundCleanup(interval, maxAge time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			mc.UnloadUnusedModels(maxAge)
		}
	}()
}

// UnloadUnusedModels unloads models that haven't been accessed recently
func (mc *ModelCache) UnloadUnusedModels(maxAge time.Duration) {
	now := time.Now()
	cutoff := now.Add(-maxAge)

	mc.accessTimesMu.Lock()
	domainsToUnload := make([]string, 0)
	for domain, accessTime := range mc.accessTimes {
		if accessTime.Before(cutoff) {
			domainsToUnload = append(domainsToUnload, domain)
		}
	}
	mc.accessTimesMu.Unlock()

	// Unload safetensors models
	mc.safetensorMu.Lock()
	for _, domain := range domainsToUnload {
		if _, exists := mc.safetensorModels[domain]; exists {
			delete(mc.safetensorModels, domain)
			mc.removeModelMemory(domain)
			mc.accessTimesMu.Lock()
			delete(mc.accessTimes, domain)
			mc.accessTimesMu.Unlock()
			log.Printf("🗑️  Unloaded unused safetensors model for domain %s", domain)
		}
	}
	mc.safetensorMu.Unlock()

	// Unload GGUF models
	mc.ggufMu.Lock()
	for _, domain := range domainsToUnload {
		if _, exists := mc.ggufModels[domain]; exists {
			delete(mc.ggufModels, domain)
			mc.removeModelMemory(domain)
			mc.accessTimesMu.Lock()
			delete(mc.accessTimes, domain)
			mc.accessTimesMu.Unlock()
			log.Printf("🗑️  Unloaded unused GGUF model for domain %s", domain)
		}
	}
	mc.ggufMu.Unlock()
}

// EvictModelsByMemory evicts least recently used models when memory limit is exceeded
func (mc *ModelCache) EvictModelsByMemory() {
	mc.currentMemoryMu.RLock()
	currentMB := mc.currentMemoryMB
	maxMB := mc.maxMemoryMB
	mc.currentMemoryMu.RUnlock()

	if currentMB <= maxMB {
		return // No eviction needed
	}

	// Sort domains by access time (LRU first)
	type domainAccess struct {
		domain     string
		accessTime time.Time
		memoryMB   int64
	}

	mc.accessTimesMu.RLock()
	domains := make([]domainAccess, 0, len(mc.accessTimes))
	for domain, accessTime := range mc.accessTimes {
		mc.modelMemoryMu.RLock()
		memoryMB := mc.modelMemoryMB[domain]
		mc.modelMemoryMu.RUnlock()
		domains = append(domains, domainAccess{
			domain:     domain,
			accessTime: accessTime,
			memoryMB:   memoryMB,
		})
	}
	mc.accessTimesMu.RUnlock()

	// Sort by access time (oldest first)
	for i := 0; i < len(domains)-1; i++ {
		for j := i + 1; j < len(domains); j++ {
			if domains[i].accessTime.After(domains[j].accessTime) {
				domains[i], domains[j] = domains[j], domains[i]
			}
		}
	}

	// Evict until we're under the limit
	evictedMB := int64(0)
	for _, da := range domains {
		if currentMB-evictedMB <= maxMB {
			break
		}

		// Try to unload safetensors model
		mc.safetensorMu.Lock()
		if _, exists := mc.safetensorModels[da.domain]; exists {
			delete(mc.safetensorModels, da.domain)
			mc.safetensorMu.Unlock()
			mc.removeModelMemory(da.domain)
			evictedMB += da.memoryMB
			mc.accessTimesMu.Lock()
			delete(mc.accessTimes, da.domain)
			mc.accessTimesMu.Unlock()
			log.Printf("🗑️  Evicted safetensors model for domain %s (freed %d MB)", da.domain, da.memoryMB)
			continue
		}
		mc.safetensorMu.Unlock()

		// Try to unload GGUF model
		mc.ggufMu.Lock()
		if _, exists := mc.ggufModels[da.domain]; exists {
			delete(mc.ggufModels, da.domain)
			mc.ggufMu.Unlock()
			mc.removeModelMemory(da.domain)
			evictedMB += da.memoryMB
			mc.accessTimesMu.Lock()
			delete(mc.accessTimes, da.domain)
			mc.accessTimesMu.Unlock()
			log.Printf("🗑️  Evicted GGUF model for domain %s (freed %d MB)", da.domain, da.memoryMB)
			continue
		}
		mc.ggufMu.Unlock()
	}
}

// estimateSafetensorMemory estimates memory usage for a safetensors model in MB
func (mc *ModelCache) estimateSafetensorMemory(model *ai.VaultGemma) int64 {
	if model == nil {
		return 0
	}

	// Estimate based on model config
	hiddenSize := int64(model.Config.HiddenSize)
	numLayers := int64(model.Config.NumLayers)
	vocabSize := int64(model.Config.VocabSize)
	intermediateSize := int64(model.Config.IntermediateSize)

	// Embedding layer: vocab_size * hidden_size * 4 bytes (float32)
	embeddingSize := vocabSize * hiddenSize * 4

	// Output layer: hidden_size * vocab_size * 4 bytes
	outputSize := hiddenSize * vocabSize * 4

	// Each transformer layer
	// Self-attention: 4 * hidden_size * hidden_size * 4 bytes (Q, K, V, O)
	attentionSize := 4 * hiddenSize * hiddenSize * 4

	// Feed-forward: 3 * hidden_size * intermediate_size * 4 bytes (W1, W2, W3)
	feedForwardSize := 3 * hiddenSize * intermediateSize * 4

	// Layer norms: 2 * hidden_size * 4 bytes
	layerNormSize := 2 * hiddenSize * 4

	layerSize := attentionSize + feedForwardSize + layerNormSize
	totalLayerSize := layerSize * numLayers

	totalSize := embeddingSize + outputSize + totalLayerSize

	// Convert bytes to MB
	return totalSize / (1024 * 1024)
}

// estimateGGUFMemory estimates memory usage for a GGUF model in MB
func (mc *ModelCache) estimateGGUFMemory(modelPath string) int64 {
	// Try to get file size as a rough estimate
	// GGUF models are quantized, so they're typically 1/4 to 1/8 of original size
	info, err := os.Stat(modelPath)
	if err != nil {
		// Default estimate for GGUF models (conservative)
		return 512 // Assume ~512MB for quantized models
	}

	// File size in MB, add 20% overhead for runtime
	fileSizeMB := info.Size() / (1024 * 1024)
	estimatedMB := fileSizeMB + (fileSizeMB * 20 / 100)

	return estimatedMB
}

// getGPULayersForDomain determines the number of GPU layers to use for a domain
func (mc *ModelCache) getGPULayersForDomain(domain string) int {
	// Check if GPU is explicitly disabled
	if os.Getenv("DISABLE_GGUF_GPU") == "1" {
		return 0
	}

	// Check for domain-specific configuration
	if mc.domainManager != nil {
		domainConfig, _ := mc.domainManager.GetDomainConfig(domain)
		if domainConfig != nil && domainConfig.GPULayers != nil {
			gpuLayers := *domainConfig.GPULayers
			if gpuLayers >= 0 {
				log.Printf("🔧 Using domain-specific GPU layers: %d for domain %s", gpuLayers, domain)
				return gpuLayers
			}
		}
	}

	// Check environment variable for global GPU layers setting
	envGPULayers := os.Getenv("GGUF_GPU_LAYERS")
	if envGPULayers != "" {
		var gpuLayers int
		if _, err := fmt.Sscanf(envGPULayers, "%d", &gpuLayers); err == nil {
			if gpuLayers >= 0 {
				log.Printf("🔧 Using GGUF_GPU_LAYERS=%d", gpuLayers)
				return gpuLayers
			}
		}
	}

	// Smart GPU detection
	hasGPU := mc.detectGPUAvailability()
	if !hasGPU {
		log.Printf("ℹ️  No GPU detected, using CPU for GGUF model")
		return 0
	}

	// Default: offload all layers if GPU is available
	// This can be overridden by GGUF_GPU_LAYERS or domain config
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "" || os.Getenv("GGML_CUDA") != "" {
		log.Printf("🚀 GPU acceleration enabled for GGUF model (offloading all layers)")
		return -1
	}

	// Auto-detect: try to use GPU if available
	log.Printf("🚀 GPU acceleration enabled for GGUF model (auto-detected)")
	return -1
}

// detectGPUAvailability checks if GPU is actually available
func (mc *ModelCache) detectGPUAvailability() bool {
	// Check for CUDA environment variables
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "" {
		return true
	}

	// Check for GGML_CUDA
	if os.Getenv("GGML_CUDA") != "" {
		return true
	}

	// Try to detect GPU by checking if we can load a test model
	// This is a simple heuristic - in production, you might want to use
	// a more sophisticated detection mechanism
	// For now, we'll be conservative and only enable if explicitly set
	return false
}

// addModelMemory adds memory usage for a model
func (mc *ModelCache) addModelMemory(domain string, memoryMB int64) {
	mc.modelMemoryMu.Lock()
	oldMB := mc.modelMemoryMB[domain]
	mc.modelMemoryMB[domain] = memoryMB
	mc.modelMemoryMu.Unlock()

	mc.currentMemoryMu.Lock()
	mc.currentMemoryMB = mc.currentMemoryMB - oldMB + memoryMB
	currentMB := mc.currentMemoryMB
	maxMB := mc.maxMemoryMB
	mc.currentMemoryMu.Unlock()

	log.Printf("📊 Memory usage: %d MB / %d MB (domain: %s, added: %d MB)", currentMB, maxMB, domain, memoryMB)

	// Check if we need to evict
	if currentMB > maxMB {
		log.Printf("⚠️  Memory limit exceeded (%d MB > %d MB), evicting models...", currentMB, maxMB)
		mc.EvictModelsByMemory()
	}
}

// removeModelMemory removes memory usage for a model
func (mc *ModelCache) removeModelMemory(domain string) {
	mc.modelMemoryMu.Lock()
	memoryMB := mc.modelMemoryMB[domain]
	delete(mc.modelMemoryMB, domain)
	mc.modelMemoryMu.Unlock()

	mc.currentMemoryMu.Lock()
	mc.currentMemoryMB -= memoryMB
	if mc.currentMemoryMB < 0 {
		mc.currentMemoryMB = 0
	}
	currentMB := mc.currentMemoryMB
	mc.currentMemoryMu.Unlock()

	log.Printf("📊 Memory usage: %d MB / %d MB (domain: %s, freed: %d MB)", currentMB, mc.maxMemoryMB, domain, memoryMB)
}

// PreloadModel preloads a model in the background (with context)
func (mc *ModelCache) PreloadModel(ctx context.Context, domain string) {
	// Check if already loaded
	mc.safetensorMu.RLock()
	_, safetensorExists := mc.safetensorModels[domain]
	path := mc.safetensorPaths[domain]
	mc.safetensorMu.RUnlock()

	if safetensorExists {
		return
	}

	if path != "" {
		go func() {
			_, _ = mc.GetSafetensorModel(ctx, domain)
		}()
		return
	}

	// Try GGUF
	mc.ggufMu.RLock()
	_, ggufExists := mc.ggufModels[domain]
	ggufPath := mc.ggufPaths[domain]
	mc.ggufMu.RUnlock()

	if !ggufExists && ggufPath != "" {
		go func() {
			_, _ = mc.GetGGUFModel(ctx, domain)
		}()
	}
}

// GetStats returns cache statistics
func (mc *ModelCache) GetStats() map[string]interface{} {
	mc.safetensorMu.RLock()
	safetensorCount := len(mc.safetensorModels)
	mc.safetensorMu.RUnlock()

	mc.ggufMu.RLock()
	ggufCount := len(mc.ggufModels)
	mc.ggufMu.RUnlock()

	mc.transformerMu.RLock()
	transformerCount := len(mc.transformerClients)
	mc.transformerMu.RUnlock()

	mc.currentMemoryMu.RLock()
	currentMB := mc.currentMemoryMB
	mc.currentMemoryMu.RUnlock()

	mc.modelMemoryMu.RLock()
	totalTrackedMB := int64(0)
	for _, mb := range mc.modelMemoryMB {
		totalTrackedMB += mb
	}
	mc.modelMemoryMu.RUnlock()

	// Calculate loading time statistics
	mc.loadingTimesMu.RLock()
	loadingTimes := make(map[string]float64)
	var totalLoadingTime time.Duration
	var maxLoadingTime time.Duration
	var minLoadingTime time.Duration = time.Hour // Large initial value
	for domain, duration := range mc.loadingTimes {
		loadingTimes[domain] = duration.Seconds()
		totalLoadingTime += duration
		if duration > maxLoadingTime {
			maxLoadingTime = duration
		}
		if duration < minLoadingTime {
			minLoadingTime = duration
		}
	}
	avgLoadingTime := time.Duration(0)
	if len(mc.loadingTimes) > 0 {
		avgLoadingTime = totalLoadingTime / time.Duration(len(mc.loadingTimes))
	}
	mc.loadingTimesMu.RUnlock()

	return map[string]interface{}{
		"safetensor_models":  safetensorCount,
		"gguf_models":        ggufCount,
		"transformer_clients": transformerCount,
		"max_memory_mb":      mc.maxMemoryMB,
		"current_memory_mb":  currentMB,
		"tracked_memory_mb":  totalTrackedMB,
		"memory_usage_pct":   float64(currentMB) / float64(mc.maxMemoryMB) * 100,
		"loading_times":      loadingTimes,
		"avg_loading_time_s": avgLoadingTime.Seconds(),
		"max_loading_time_s": maxLoadingTime.Seconds(),
		"min_loading_time_s": minLoadingTime.Seconds(),
		"total_models_loaded": len(mc.loadingTimes),
	}
}

