# Model Loading: Discovery, Loading, and Caching

This document explains how models are discovered, loaded, cached, and managed in the LocalAI service, with detailed code references.

## Overview

The LocalAI service supports multiple model formats and loading strategies:
- **SafeTensors**: Native Go implementation (CPU)
- **GGUF**: Quantized models via llama.cpp (CPU/GPU)
- **Transformers**: External Python service (GPU)
- **Lazy Loading**: Models loaded on first use to reduce startup time
- **Caching**: In-memory and persistent caching of loaded models

## Model Discovery

### Configuration Source

Models are discovered from domain configuration files:

**File**: `services/localai/config/domains.json`

**Structure**:
```json
{
  "domains": {
    "general": {
      "name": "general",
      "backend_type": "vaultgemma",
      "model_path": "../../models/vaultgemma-1b-transformers",
      "keywords": ["general", "default"],
      "max_tokens": 500,
      "temperature": 0.7
    },
    "sql": {
      "name": "sql",
      "backend_type": "gguf",
      "model_path": "../../models/phi-3.5-mini-instruct.gguf",
      "keywords": ["sql", "query", "database"],
      "gpu_layers": 20
    },
    "vision": {
      "name": "vision",
      "backend_type": "hf-transformers",
      "transformers_config": {
        "endpoint": "http://transformers-service:8081",
        "model_name": "deepseek-ocr"
      },
      "keywords": ["image", "ocr", "vision"]
    }
  },
  "default_domain": "general"
}
```

### Configuration Loading

**File**: `services/localai/cmd/vaultgemma-server/main.go`

**Function**: Domain configuration loading (line 81)

```go
// Load domain configurations (Redis or file-based)
domainManager := domain.NewDomainManager()
configLoader, err := domain.NewConfigLoader()
if err != nil {
    log.Printf("⚠️  Failed to create config loader: %v", err)
    log.Printf("⚠️  Falling back to file-based config")
    configLoader = domain.NewFileConfigLoader(*configPath)
}

if err := configLoader.LoadDomainConfigs(context.Background(), domainManager); err != nil {
    log.Printf("⚠️  Failed to load domain configs: %v", err)
    log.Printf("⚠️  Continuing with single model mode")
} else {
    log.Printf("✅ Loaded domain configs from %s", domain.GetConfigSource())
}
```

**File**: `services/localai/pkg/domain/domain_config.go`

**Function**: `LoadDomainConfigs()` (line 134)

```go
func (dm *DomainManager) LoadDomainConfigs(configPath string) error {
    data, err := os.ReadFile(configPath)
    if err != nil {
        return fmt.Errorf("read config file: %w", err)
    }

    var config DomainsConfig
    if err := json.Unmarshal(data, &config); err != nil {
        return fmt.Errorf("parse config: %w", err)
    }

    // Filter enabled domains
    filtered := make(map[string]*DomainConfig)
    for name, cfg := range config.Domains {
        if !isDomainEnabled(cfg.EnabledEnvVar) {
            continue
        }
        if err := cfg.Validate(); err != nil {
            return fmt.Errorf("domain %s invalid: %w", name, err)
        }
        filtered[name] = cfg
    }

    dm.domains = filtered
    dm.defaultDomain = config.DefaultDomain
    return nil
}
```

## Model Registration

### Startup Registration

**File**: `services/localai/cmd/vaultgemma-server/main.go`

**Function**: Model registration (line 106)

```go
// Phase 4: Register models for lazy loading instead of loading them
log.Printf("\n🎯 Registering models for lazy loading...")

if domainManager != nil {
    configs := domainManager.ListDomainConfigs()
    for name, cfg := range configs {
        cfgModelPath := strings.TrimSpace(cfg.ModelPath)
        lowerPath := strings.ToLower(cfgModelPath)

        // Register Transformers backend
        if strings.EqualFold(cfg.BackendType, "hf-transformers") {
            if cfg.TransformersConfig == nil {
                continue
            }
            timeout := time.Duration(cfg.TransformersConfig.TimeoutSeconds) * time.Second
            client := transformers.NewClient(cfg.TransformersConfig.Endpoint, cfg.TransformersConfig.ModelName, timeout)
            transformerClients[name] = client
            log.Printf("✅ Transformers backend ready for domain %s -> %s", name, cfg.TransformersConfig.ModelName)
            continue
        }

        // Register GGUF model for lazy loading
        if strings.HasSuffix(lowerPath, ".gguf") {
            log.Printf("📝 Registered GGUF model for domain %s: %s (will load on first use)", name, cfgModelPath)
            continue
        }

        // Register safetensors model for lazy loading
        log.Printf("📝 Registered safetensors model for domain %s: %s (will load on first use)", name, cfgModelPath)
    }
}
```

### Model Cache Registration

**File**: `services/localai/cmd/vaultgemma-server/main.go`

**Function**: Cache registration (line 177)

```go
// Phase 4: Register models in cache for lazy loading
if enableLazyLoading && vgServer.modelCache != nil {
    log.Printf("\n📝 Registering models in cache for lazy loading...")
    
    // Register default VaultGemma model if not disabled
    if !disableFallback && *modelPath != "" {
        vgServer.modelCache.RegisterSafetensorModel("general", *modelPath)
        vgServer.modelCache.RegisterSafetensorModel("vaultgemma", *modelPath)
        log.Printf("✅ Registered default model: %s", *modelPath)
    }

    // Register domain-specific models
    if domainManager != nil {
        configs := domainManager.ListDomainConfigs()
        for name, cfg := range configs {
            cfgModelPath := strings.TrimSpace(cfg.ModelPath)
            if cfgModelPath == "" {
                continue
            }

            lowerPath := strings.ToLower(cfgModelPath)

            // Register GGUF models
            if strings.HasSuffix(lowerPath, ".gguf") {
                vgServer.modelCache.RegisterGGUFModel(name, cfgModelPath)
                log.Printf("✅ Registered GGUF model for domain %s: %s", name, cfgModelPath)
                continue
            }

            // Register safetensors models
            if cfgModelPath != *modelPath && !strings.EqualFold(cfg.BackendType, "hf-transformers") {
                vgServer.modelCache.RegisterSafetensorModel(name, cfgModelPath)
                log.Printf("✅ Registered safetensors model for domain %s: %s", name, cfgModelPath)
            }

            // Register transformers clients
            if strings.EqualFold(cfg.BackendType, "hf-transformers") && cfg.TransformersConfig != nil {
                vgServer.modelCache.RegisterTransformerClient(name, cfg.TransformersConfig)
            }
        }
    }
}
```

## Lazy Loading

### Lazy Loading Strategy

Lazy loading is enabled by default to reduce startup time. Models are registered at startup but only loaded when first requested.

**File**: `services/localai/cmd/vaultgemma-server/main.go`

**Configuration** (line 47):
```go
enableLazyLoading := os.Getenv("ENABLE_LAZY_LOADING") != "0" // Default to enabled

if !enableLazyLoading {
    // Legacy behavior: load model at startup
    loadedModel, err := ai.LoadVaultGemmaFromSafetensors(*modelPath)
    // ...
} else {
    log.Printf("\n🚀 Lazy loading enabled - models will be loaded on first use")
}
```

### Model Resolution with Lazy Loading

**File**: `services/localai/pkg/server/chat_helpers.go`

**Function**: `resolveModelForDomain()` (line 128)

```go
func (s *VaultGemmaServer) resolveModelForDomain(
    ctx context.Context,
    domain string,
    domainConfig *domain.DomainConfig,
    preferredBackend string,
) (model *ai.VaultGemma, modelKey string, fallbackUsed bool, fallbackKey string, err error) {
    // Phase 4: Try lazy loading from cache first
    if s.modelCache != nil {
        // Try to get safetensors model from cache (lazy loading)
        cachedModel, err := s.modelCache.GetSafetensorModel(ctx, domain)
        if err == nil && cachedModel != nil {
            return cachedModel, domain, false, "", nil
        }

        // Try fallback model from cache
        if domainConfig != nil && domainConfig.FallbackModel != "" {
            fallbackModel, err := s.modelCache.GetSafetensorModel(ctx, domainConfig.FallbackModel)
            if err == nil && fallbackModel != nil {
                return fallbackModel, domainConfig.FallbackModel, true, domainConfig.FallbackModel, nil
            }
        }

        // Try default domain from cache
        defaultDomain := s.domainManager.GetDefaultDomain()
        if defaultDomain != "" {
            defaultModel, err := s.modelCache.GetSafetensorModel(ctx, defaultDomain)
            if err == nil && defaultModel != nil {
                return defaultModel, defaultDomain, false, "", nil
            }
        }
    }

    // Fallback to pre-loaded models (for backward compatibility)
    // ...
}
```

### SafeTensors Lazy Loading

**File**: `services/localai/pkg/server/model_cache.go`

**Function**: `GetSafetensorModel()` (lazy loading implementation)

```go
func (mc *ModelCache) GetSafetensorModel(ctx context.Context, domain string) (*ai.VaultGemma, error) {
    // Check if already loaded
    mc.safetensorMu.RLock()
    if model, exists := mc.safetensorModels[domain]; exists {
        mc.safetensorMu.RUnlock()
        mc.updateAccessTime(domain)
        return model, nil
    }
    mc.safetensorMu.RUnlock()

    // Check if path is registered
    mc.pathsMu.RLock()
    path, exists := mc.safetensorPaths[domain]
    mc.pathsMu.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("no model path registered for domain: %s", domain)
    }

    // Check if loading in progress
    mc.loadingMu.Lock()
    if loadingChan, loading := mc.loadingInProgress[domain]; loading {
        mc.loadingMu.Unlock()
        // Wait for loading to complete
        select {
        case <-loadingChan:
            mc.safetensorMu.RLock()
            model := mc.safetensorModels[domain]
            mc.safetensorMu.RUnlock()
            return model, nil
        case <-ctx.Done():
            return nil, ctx.Err()
        }
    }

    // Start loading
    loadingChan := make(chan struct{})
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

        // Store in cache
        mc.safetensorMu.Lock()
        mc.safetensorModels[domain] = loadedModel
        mc.safetensorMu.Unlock()

        // Update memory tracking
        memoryMB := mc.estimateSafetensorMemory(loadedModel)
        mc.addModelMemory(domain, memoryMB)
        mc.updateAccessTime(domain)
    }()

    // Wait for loading to complete
    select {
    case <-loadingChan:
        mc.safetensorMu.RLock()
        model := mc.safetensorModels[domain]
        mc.safetensorMu.RUnlock()
        return model, nil
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}
```

### GGUF Lazy Loading

**File**: `services/localai/pkg/server/model_cache.go`

**Function**: `GetGGUFModel()` (similar pattern)

```go
func (mc *ModelCache) GetGGUFModel(ctx context.Context, domain string) (*gguf.Model, error) {
    // Check if already loaded
    mc.ggufMu.RLock()
    if model, exists := mc.ggufModels[domain]; exists {
        mc.ggufMu.RUnlock()
        mc.updateAccessTime(domain)
        return model, nil
    }
    mc.ggufMu.RUnlock()

    // Get path and start loading
    // ... similar to SafeTensors loading
    
    // Load GGUF model
    log.Printf("📥 Lazy loading GGUF model for domain %s from %s...", domain, path)
    
    // Determine GPU layers configuration
    gpuLayers := mc.getGPULayersForDomain(domain)
    
    // Allocate GPU if needed
    if gpuLayers != 0 {
        if err := mc.allocateGPUForModel(ctx, domain, path, "gguf", modelConfig); err != nil {
            log.Printf("⚠️  GPU allocation failed, continuing with CPU: %v", err)
            gpuLayers = 0
        }
    }

    // Load model with llama.cpp
    loadedModel, err := gguf.LoadModel(path, gpuLayers)
    // ... store in cache
}
```

## Model Loading Functions

### SafeTensors Loading

**File**: `services/localai/pkg/models/model_loader.go`

**Function**: `LoadModelFromSafeTensors()` (line 25)

```go
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
```

### GGUF Loading

**File**: `services/localai/pkg/models/gguf/model.go`

GGUF models are loaded using the llama.cpp library:

```go
// Load GGUF model with GPU layers
model, err := gguf.LoadModel(modelPath, gpuLayers)
```

**GPU Layers Configuration**:
- `gpuLayers = -1`: All layers on GPU
- `gpuLayers = 0`: All layers on CPU
- `gpuLayers = N`: First N layers on GPU, rest on CPU

### Transformers Loading

**File**: `services/localai/pkg/transformers/client.go`

Transformers models use an external Python service:

```go
client := transformers.NewClient(
    endpoint,      // "http://transformers-service:8081"
    modelName,     // "deepseek-ocr"
    timeout,       // 30 * time.Second
)
```

The client makes HTTP requests to the Python service, which handles model loading internally.

## Caching Mechanisms

### In-Memory Cache

**File**: `services/localai/pkg/server/model_cache.go`

**Type**: `ModelCache` struct

```go
type ModelCache struct {
    safetensorModels map[string]*ai.VaultGemma
    ggufModels       map[string]*gguf.Model
    transformerClients map[string]*transformers.Client
    safetensorPaths   map[string]string
    ggufPaths         map[string]string
    loadingInProgress map[string]chan struct{}
    // ... mutexes and tracking
}
```

**Cache Operations**:
- **Get**: Retrieve model from cache
- **Set**: Store model in cache after loading
- **Update Access Time**: Track last access for LRU eviction
- **Memory Tracking**: Monitor memory usage per model

### Persistent Cache (PostgreSQL)

**File**: `services/localai/pkg/storage/hana_cache.go`

Cache state can be persisted to PostgreSQL:

```go
type CacheState struct {
    Domain      string
    ModelType   string  // "safetensors", "gguf", "transformers"
    ModelPath   string
    LoadedAt    time.Time
    MemoryMB    int64
    AccessCount int64
    LastAccess  time.Time
    CacheData   map[string]interface{}
}
```

**Benefits**:
- Survives server restarts
- Shared across multiple instances
- Tracks model usage statistics

## Backend Selection

### Backend Types

**File**: `services/localai/pkg/server/vaultgemma_server.go`

**Backend Selection Logic**:

```go
// Determine backend type
backendType := ""
if domainConfig != nil {
    backendType = strings.TrimSpace(domainConfig.BackendType)
}
if backendType == "" {
    backendType = pickPreferredBackend()
}

// Route to appropriate backend
if strings.EqualFold(backendType, BackendTypeDeepSeekOCR) {
    return s.processDeepSeekOCR(...)
}

if strings.EqualFold(backendType, BackendTypeTransformers) {
    return s.processTransformersBackend(...)
}

// Try GGUF backend
if ggufModel := s.ggufModels[modelKey]; ggufModel != nil {
    // Generate with GGUF
}

// Generate with SafeTensors
if model != nil {
    generated, tokensUsed, err := s.inferenceEngine.Generate(...)
}
```

### Backend Priority

1. **Domain Configuration**: Use `backend_type` from domain config
2. **Model Format**: Auto-detect from file extension (.gguf)
3. **Default**: SafeTensors (CPU) if no other backend available

## GPU Allocation

### GPU Requirements

**File**: `services/localai/pkg/server/model_registry.go`

**Function**: `GetRequirements()` (line 120)

```go
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
```

**Registered Models**:
- `gemma-7b`: 16GB, dedicated GPU
- `gemma-2b`: 4GB, can share GPU
- `vaultgemma-1b`: 2GB, can share GPU
- `phi-3.5-mini`: 4GB, can share GPU

### GPU Allocation Flow

**File**: `services/localai/pkg/server/model_cache.go`

```go
func (mc *ModelCache) allocateGPUForModel(ctx context.Context, domain, path, modelType string, config *ai.VaultGemmaConfig) error {
    // Get GPU requirements
    req := mc.modelRegistry.GetRequirements(path, domain)
    
    // Allocate via GPU orchestrator
    if mc.gpuOrchestrator != nil {
        allocation, err := mc.gpuOrchestrator.AllocateGPUsWithWorkload(ctx, req.RequiredGPUs, workloadData)
        // ... store allocation
    }
    
    return nil
}
```

## Preloading

### Preload Configuration

**File**: `services/localai/cmd/vaultgemma-server/main.go`

**Environment Variables**:
- `PRELOAD_MODELS`: Comma-separated list of domains to preload
- `PRELOAD_DEFAULT_MODELS`: Set to "1" to preload default models

```go
// Preload frequently used models if configured
preloadEnv := os.Getenv("PRELOAD_MODELS")
if preloadEnv != "" {
    envDomains := strings.Split(preloadEnv, ",")
    for _, d := range envDomains {
        d = strings.TrimSpace(d)
        if d != "" {
            preloadDomains = append(preloadDomains, d)
        }
    }
}

// Preload models in background
if len(preloadDomains) > 0 {
    log.Printf("\n🚀 Preloading %d models in background: %v", len(preloadDomains), preloadDomains)
    ctx := context.Background()
    for _, domain := range preloadDomains {
        vgServer.modelCache.PreloadModel(ctx, domain)
    }
}
```

## Model Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Server Startup                                           │
│    - Load domains.json                                      │
│    - Register model paths                                   │
│    - Initialize model cache                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. Model Registration                                       │
│    - SafeTensors: Register path                             │
│    - GGUF: Register path                                    │
│    - Transformers: Create HTTP client                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. First Request                                            │
│    - Domain detection                                       │
│    - Model resolution                                       │
│    - Check cache (not found)                                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 4. Lazy Loading                                             │
│    - Check loading in progress                              │
│    - Start background loading                               │
│    - Load from filesystem                                   │
│    - Store in cache                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. Subsequent Requests                                     │
│    - Domain detection                                       │
│    - Model resolution                                       │
│    - Retrieve from cache                                    │
│    - Update access time                                     │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling

### Model Loading Errors

1. **File Not Found**: Model path doesn't exist
2. **Invalid Format**: Missing required files (config.json, model.safetensors)
3. **Memory Error**: Insufficient memory for model
4. **GPU Error**: GPU allocation failed (falls back to CPU)

### Fallback Strategy

1. Try domain-specific model
2. Try fallback model (if configured)
3. Try default "general" domain
4. Return error if no model available

## Performance Considerations

1. **Lazy Loading**: Reduces startup time from minutes to seconds
2. **Caching**: Avoids reloading models on every request
3. **Memory Management**: Tracks memory usage, supports eviction
4. **GPU Sharing**: Multiple small models can share GPU
5. **Preloading**: Preload frequently used models in background

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture overview
- [ABSTRACTION_LAYERS.md](./ABSTRACTION_LAYERS.md) - Abstraction layer details
- [DATA_FLOW.md](./DATA_FLOW.md) - Request/response flow
- [DIAGRAMS.md](./DIAGRAMS.md) - Visual architecture diagrams

