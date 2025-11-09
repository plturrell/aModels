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
	accessTimes      map[string]time.Time
	accessTimesMu    sync.RWMutex
	loadingInProgress map[string]chan struct{}
	loadingMu        sync.Mutex

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
		accessTimes:        make(map[string]time.Time),
		loadingInProgress:  make(map[string]chan struct{}),
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

		mc.safetensorMu.Lock()
		mc.safetensorModels[domain] = loadedModel
		mc.safetensorMu.Unlock()

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

		// Enable GPU layers if available
		gpuLayers := 0
		if os.Getenv("CUDA_VISIBLE_DEVICES") != "" || os.Getenv("GGML_CUDA") != "" {
			gpuLayers = -1
			log.Printf("🚀 GPU acceleration enabled for GGUF model (offloading all layers)")
		} else if os.Getenv("DISABLE_GGUF_GPU") == "" {
			gpuLayers = -1
			log.Printf("🚀 GPU acceleration enabled for GGUF model (auto-detected)")
		}

		var loadedModel *gguf.Model
		var err error
		if gpuLayers != 0 {
			loadedModel, err = gguf.Load(path, llama.SetGPULayers(gpuLayers))
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

		mc.ggufMu.Lock()
		mc.ggufModels[domain] = loadedModel
		mc.ggufMu.Unlock()

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
			mc.accessTimesMu.Lock()
			delete(mc.accessTimes, domain)
			mc.accessTimesMu.Unlock()
			log.Printf("🗑️  Unloaded unused GGUF model for domain %s", domain)
		}
	}
	mc.ggufMu.Unlock()
}

// PreloadModel preloads a model in the background
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

	return map[string]interface{}{
		"safetensor_models": safetensorCount,
		"gguf_models":       ggufCount,
		"transformer_clients": transformerCount,
		"max_memory_mb":     mc.maxMemoryMB,
		"current_memory_mb": mc.currentMemoryMB,
	}
}

