package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/gpu"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/gguf"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/storage"
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

	// GPU orchestrator integration
	gpuRouter           *gpu.GPURouter
	modelGPUAllocations map[string]string // domain -> allocation ID
	sharedGPUGroup      map[string][]string // GPU ID -> list of domains sharing it
	gpuAllocMu          sync.RWMutex
	modelRegistry       *ModelRegistry

	// Domain manager for configuration
	domainManager *domain.DomainManager
	
	// Postgres cache store (Phase 1)
	postgresCache *storage.PostgresCacheStore
}

// NewModelCache creates a new model cache with lazy loading support
func NewModelCache(domainManager *domain.DomainManager, maxMemoryMB int64) *ModelCache {
	if maxMemoryMB <= 0 {
		maxMemoryMB = 8192 // Default 8GB
	}

	// Initialize GPU router if orchestrator URL is configured
	var gpuRouter *gpu.GPURouter
	if orchestratorURL := os.Getenv("GPU_ORCHESTRATOR_URL"); orchestratorURL != "" {
		gpuRouter = gpu.NewGPURouter(orchestratorURL, nil)
		log.Printf("🔌 GPU orchestrator integration enabled: %s", orchestratorURL)
	}

	// Initialize Postgres cache store if DSN is configured (Phase 1)
	var postgresCache *storage.PostgresCacheStore
	if postgresDSN := os.Getenv("POSTGRES_DSN"); postgresDSN != "" {
		if cacheStore, err := storage.NewPostgresCacheStore(postgresDSN); err == nil {
			postgresCache = cacheStore
			log.Printf("✅ Postgres cache store initialized for ModelCache")
		} else {
			log.Printf("⚠️  Failed to initialize Postgres cache store: %v", err)
		}
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
		gpuRouter:          gpuRouter,
		modelGPUAllocations: make(map[string]string),
		sharedGPUGroup:      make(map[string][]string),
		modelRegistry:       NewModelRegistry(),
		domainManager:       domainManager,
		postgresCache:       postgresCache,
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

		// Allocate GPU if needed (SafeTensors are CPU-only, but check anyway)
		// This is a placeholder - SafeTensors models don't use GPU
		// But we check in case the model config indicates GPU usage
		_ = mc.allocateGPUForModel(context.Background(), domain, path, "safetensors", nil)

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
		
		// Persist cache state to Postgres (Phase 1)
		if mc.postgresCache != nil {
			go func() {
				state := &storage.CacheState{
					Domain:      domain,
					ModelType:   "safetensors",
					ModelPath:   path,
					LoadedAt:    time.Now(),
					MemoryMB:    memoryMB,
					AccessCount: 1,
					LastAccess:  time.Now(),
					CacheData:   make(map[string]interface{}),
				}
				if err := mc.postgresCache.SaveCacheState(context.Background(), state); err != nil {
					log.Printf("⚠️  Failed to persist cache state for domain %s: %v", domain, err)
				}
			}()
		}
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
		
		// Allocate GPU if needed (before loading)
		if gpuLayers != 0 {
			// Try to load config first to get accurate model size
			// GGUF models might have config.json in the same directory
			var modelConfig *ai.VaultGemmaConfig
			configPath := filepath.Join(filepath.Dir(path), "config.json")
			if configData, err := os.ReadFile(configPath); err == nil {
				var config ai.VaultGemmaConfig
				if err := json.Unmarshal(configData, &config); err == nil {
					modelConfig = &config
					log.Printf("📋 Loaded model config for GPU allocation: %d layers, %d hidden size", config.NumLayers, config.HiddenSize)
				}
			}
			
			// If no config file, estimate from path/model name
			if modelConfig == nil {
				// Estimate based on model path/name
				modelType := mc.modelRegistry.GetModelTypeFromPath(path)
				req := mc.modelRegistry.GetRequirements(path, modelType)
				if req != nil {
					// Use registry defaults for estimation
					modelConfig = &ai.VaultGemmaConfig{
						NumLayers:  18, // Default estimate
						HiddenSize: 2048,
						VocabSize:  256000,
					}
					// Adjust based on model type
					if strings.Contains(strings.ToLower(modelType), "7b") {
						modelConfig.NumLayers = 28
						modelConfig.HiddenSize = 3072
					} else if strings.Contains(strings.ToLower(modelType), "2b") {
						modelConfig.NumLayers = 18
						modelConfig.HiddenSize = 2048
					}
				}
			}
			
			if err := mc.allocateGPUForModel(context.Background(), domain, path, "gguf", modelConfig); err != nil {
				log.Printf("⚠️  GPU allocation failed, continuing with CPU: %v", err)
				gpuLayers = 0 // Fallback to CPU
			}
		}

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
		
		// Persist cache state to Postgres (Phase 1)
		if mc.postgresCache != nil {
			go func() {
				state := &storage.CacheState{
					Domain:      domain,
					ModelType:   "gguf",
					ModelPath:   path,
					LoadedAt:    time.Now(),
					MemoryMB:    memoryMB,
					AccessCount: 1,
					LastAccess:  time.Now(),
					CacheData:   make(map[string]interface{}),
				}
				if err := mc.postgresCache.SaveCacheState(context.Background(), state); err != nil {
					log.Printf("⚠️  Failed to persist cache state for domain %s: %v", domain, err)
				}
			}()
		}
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

	ctx := context.Background()

	// Unload safetensors models
	mc.safetensorMu.Lock()
	for _, domain := range domainsToUnload {
		if _, exists := mc.safetensorModels[domain]; exists {
			// Release GPU allocation before unloading
			mc.releaseGPUForModel(ctx, domain)
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
			// Release GPU allocation before unloading
			mc.releaseGPUForModel(ctx, domain)
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

		ctx := context.Background()

		// Try to unload safetensors model
		mc.safetensorMu.Lock()
		if _, exists := mc.safetensorModels[da.domain]; exists {
			// Release GPU allocation before evicting
			mc.releaseGPUForModel(ctx, da.domain)
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
			// Release GPU allocation before evicting
			mc.releaseGPUForModel(ctx, da.domain)
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

// allocateGPUForModel allocates GPU for a model using hybrid strategy
func (mc *ModelCache) allocateGPUForModel(ctx context.Context, domain, modelPath, backendType string, modelConfig *ai.VaultGemmaConfig) error {
	if mc.gpuRouter == nil {
		return nil // No GPU orchestrator configured
	}

	// Get model requirements
	modelType := mc.modelRegistry.GetModelTypeFromPath(modelPath)
	req := mc.modelRegistry.GetRequirements(modelPath, modelType)
	
	// Determine if model needs GPU
	needsGPU := false
	if backendType == "gguf" {
		// GGUF models may use GPU
		gpuLayers := mc.getGPULayersForDomain(domain)
		needsGPU = gpuLayers != 0
	} else if backendType == "safetensors" {
		// SafeTensors models are CPU-only, no GPU allocation needed
		return nil
	} else if backendType == "transformers" {
		// Transformers service handles its own GPU allocation
		return nil
	}

	if !needsGPU {
		return nil
	}

	// Determine allocation strategy
	strategy := os.Getenv("GPU_ALLOCATION_STRATEGY")
	if strategy == "" {
		strategy = "hybrid"
	}

	// Check if model is large (needs dedicated GPU)
	isLarge := false
	if modelConfig != nil {
		isLarge = mc.modelRegistry.IsLargeModel(modelPath, modelType, modelConfig.NumLayers, modelConfig.HiddenSize, modelConfig.VocabSize)
	} else {
		isLarge = req != nil && req.Dedicated
	}

	// Build workload data
	workloadData := make(map[string]interface{})
	workloadData["model_name"] = modelType
	workloadData["model_path"] = modelPath
	workloadData["backend_type"] = backendType
	workloadData["domain"] = domain
	
	if modelConfig != nil {
		workloadData["num_layers"] = modelConfig.NumLayers
		workloadData["hidden_size"] = modelConfig.HiddenSize
		workloadData["vocab_size"] = modelConfig.VocabSize
		modelSizeB := mc.modelRegistry.EstimateModelSize(modelConfig.NumLayers, modelConfig.HiddenSize, modelConfig.VocabSize)
		workloadData["model_size_b"] = modelSizeB
		
		if modelSizeB < 3.0 {
			workloadData["model_size"] = "small"
		} else if modelSizeB < 7.0 {
			workloadData["model_size"] = "medium"
		} else {
			workloadData["model_size"] = "large"
		}
	}
	
	if req != nil {
		workloadData["min_memory_mb"] = req.MinMemoryMB
		workloadData["priority"] = req.Priority
	}

	// Determine required GPUs and memory
	requiredGPUs := 1
	minMemoryMB := int64(4096) // Default 4GB
	if req != nil {
		requiredGPUs = req.RequiredGPUs
		minMemoryMB = req.MinMemoryMB
	}

	// For hybrid strategy: large models get dedicated, small models can share
	if strategy == "hybrid" {
		if isLarge {
			workloadData["dedicated"] = true
			log.Printf("🎯 Allocating dedicated GPU for large model: %s (domain: %s)", modelType, domain)
		} else {
			workloadData["dedicated"] = false
			workloadData["allow_sharing"] = true
			log.Printf("🎯 Allocating shared GPU for small model: %s (domain: %s)", modelType, domain)
		}
	} else if strategy == "dedicated" {
		workloadData["dedicated"] = true
	} else {
		workloadData["dedicated"] = false
		workloadData["allow_sharing"] = true
	}

	// Request GPU allocation
	allocationID, allocatedGPUs, err := mc.gpuRouter.AllocateGPUsWithWorkload(ctx, requiredGPUs, workloadData)
	if err != nil {
		log.Printf("⚠️  Failed to allocate GPU for domain %s: %v (continuing with CPU fallback)", domain, err)
		return err // Return error but allow fallback
	}

	// Track allocation
	mc.gpuAllocMu.Lock()
	if allocationID == "" {
		// Fallback: generate a simple allocation ID for tracking
		allocationID = fmt.Sprintf("%s-%d", domain, time.Now().UnixNano())
	}
	mc.modelGPUAllocations[domain] = allocationID
	
	// Track shared GPU groups if not dedicated
	if !isLarge && strategy == "hybrid" && len(allocatedGPUs) > 0 {
		gpuID := fmt.Sprintf("%d", allocatedGPUs[0])
		if mc.sharedGPUGroup[gpuID] == nil {
			mc.sharedGPUGroup[gpuID] = []string{}
		}
		mc.sharedGPUGroup[gpuID] = append(mc.sharedGPUGroup[gpuID], domain)
	}
	mc.gpuAllocMu.Unlock()

	log.Printf("✅ GPU allocated for domain %s (allocation ID: %s)", domain, allocationID)
	return nil
}

// releaseGPUForModel releases GPU allocation for a model
func (mc *ModelCache) releaseGPUForModel(ctx context.Context, domain string) error {
	if mc.gpuRouter == nil {
		return nil
	}

	mc.gpuAllocMu.Lock()
	allocationID, exists := mc.modelGPUAllocations[domain]
	if !exists {
		mc.gpuAllocMu.Unlock()
		return nil // No allocation to release
	}
	delete(mc.modelGPUAllocations, domain)
	
	// Remove from shared GPU groups
	for gpuID, domains := range mc.sharedGPUGroup {
		for i, d := range domains {
			if d == domain {
				mc.sharedGPUGroup[gpuID] = append(domains[:i], domains[i+1:]...)
				if len(mc.sharedGPUGroup[gpuID]) == 0 {
					delete(mc.sharedGPUGroup, gpuID)
				}
				break
			}
		}
	}
	mc.gpuAllocMu.Unlock()

	// Release via router (if it's the last model using that allocation)
	// For now, we'll just log - actual release happens when all models are unloaded
	log.Printf("🔄 Released GPU allocation tracking for domain %s (ID: %s)", domain, allocationID)
	
	// If this was the only model using the GPU, we could release it
	// But for simplicity, we'll keep the allocation until explicitly released
	return nil
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

