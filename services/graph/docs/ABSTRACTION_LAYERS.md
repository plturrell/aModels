# Abstraction Layers: Detailed Breakdown

This document provides a detailed explanation of each abstraction layer in the Graph-LocalAI-Models architecture, with specific code references and interface definitions.

## Layer 1: Graph Service → LocalAI Abstraction

### Overview

The Graph service uses LocalAI as an external LLM provider through an HTTP REST API. This abstraction allows the Graph service to remain agnostic about how LocalAI manages models internally.

### Implementation

**File**: `services/graph/pkg/workflows/orchestration_processor.go`

**Function**: `createOrchestrationChain()` (line 321)
```go
func createOrchestrationChain(chainName, localAIURL string, headers map[string]string) (chains.Chain, error) {
    // Create LocalAI LLM instance using the orchestration framework
    opts := []localai.Option{localai.WithBaseURL(localAIURL)}
    if headers != nil && len(headers) > 0 {
        opts = append(opts, localai.WithHeaders(headers))
    }
    llm, err := localai.New(opts...)
    if err != nil {
        return nil, fmt.Errorf("create LocalAI LLM: %w", err)
    }
    // ... create chain with LLM
}
```

**Function**: `RunOrchestrationChainNode()` (line 48)
```go
func RunOrchestrationChainNode(localAIURL string) stategraph.NodeFunc {
    return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
        // Extract chain configuration from state
        // Create orchestration chain
        chain, err := createOrchestrationChain(chainName, localAIURL, headers)
        // Execute chain
        result, execErr = chains.Call(ctx, chain, chainInputs)
        // ... return results
    })
}
```

### Client Implementation

**File**: `infrastructure/third_party/orchestration/llms/localai/localai.go`

**Type**: `LLM` struct (line 21)
```go
type LLM struct {
    CallbacksHandler callbacks.Handler
    client           *http.Client
    baseURL          string
    model            string
    temperature      float64
    maxTokens        int
    domains          []string
    autoRouting      bool
    headers          map[string]string // Workflow context headers
}
```

**Function**: `GenerateContent()` (line 161)
```go
func (l *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
    // Convert messages to LocalAI chat format
    chatMessages := make([]chatMessage, 0, len(messages))
    // ... build request
    
    reqBody := chatCompletionRequest{
        Model:       l.model,
        Messages:    chatMessages,
        Temperature: l.temperature,
        MaxTokens:   l.maxTokens,
        Domains:     l.domains,
    }
    
    // Make HTTP POST request
    req, err := http.NewRequestWithContext(ctx, "POST", l.baseURL+"/v1/chat/completions", bytes.NewBuffer(jsonData))
    // ... send request and parse response
}
```

### Abstraction Characteristics

1. **Protocol**: HTTP REST (OpenAI-compatible)
2. **Request Format**: JSON with `model`, `messages`, `temperature`, `max_tokens`
3. **Response Format**: OpenAI-compatible chat completion response
4. **Error Handling**: HTTP status codes and error messages
5. **Workflow Context**: Custom headers (X-Workflow-ID, X-Workflow-Priority)

### Benefits

- Graph service doesn't need to know about model formats or loading
- LocalAI can change internal implementation without affecting Graph service
- Standard OpenAI-compatible interface allows easy swapping of providers
- Workflow context can be passed through headers

---

## Layer 2: LocalAI Service Internal Abstractions

### 2.1 Domain Management Abstraction

**File**: `services/localai/pkg/domain/domain_config.go`

**Type**: `DomainConfig` struct (line 12)
```go
type DomainConfig struct {
    Name               string
    Layer              string              // layer1, layer2, layer3
    Team               string              // DataTeam, FoundationTeam, etc.
    BackendType        string              // vaultgemma, openai, ollama, mock
    ModelPath          string              // For local models
    ModelName          string              // For API models
    APIKey             string              // For API backends
    BaseURL            string              // For custom API endpoints
    AgentID            string              // Single agent hex ID
    AttentionWeights   map[string]float32
    MaxTokens          int
    Temperature        float32
    TopP               float32
    TopK               int
    DomainTags         []string
    Keywords           []string            // For domain detection
    FallbackModel      string
    EnabledEnvVar      string
    GPULayers          *int
    VisionConfig       *VisionConfig
    TransformersConfig *TransformersConfig
}
```

**Type**: `DomainManager` struct (line 109)
```go
type DomainManager struct {
    domains       map[string]*DomainConfig
    defaultDomain string
    mu            sync.RWMutex
}
```

**Function**: `DetectDomain()` (line 226)
```go
func (dm *DomainManager) DetectDomain(prompt string, userDomains []string) string {
    promptLower := strings.ToLower(prompt)
    
    // Score each domain based on keyword matches
    bestScore := 0
    bestDomain := dm.defaultDomain
    
    for domainName, config := range dm.domains {
        score := 0
        for _, keyword := range config.Keywords {
            if strings.Contains(promptLower, strings.ToLower(keyword)) {
                score++
            }
        }
        if score > bestScore {
            bestScore = score
            bestDomain = domainName
        }
    }
    
    return bestDomain
}
```

**Abstraction Purpose**: Separates domain configuration from domain detection logic, allowing easy addition of new domains without code changes.

### 2.2 Model Provider Interface

**File**: `services/localai/pkg/server/interfaces.go`

**Interface**: `ModelProvider` (line 17)
```go
type ModelProvider interface {
    // GetSafetensorsModel returns a safetensors model by key
    GetSafetensorsModel(key string) (*ai.VaultGemma, bool)
    // GetGGUFModel returns a GGUF model by key
    GetGGUFModel(key string) (*gguf.Model, bool)
    // GetTransformerClient returns a transformers client by key
    GetTransformerClient(key string) (*transformers.Client, bool)
    // HasModel checks if a model exists for the given key
    HasModel(key string) bool
}
```

**Abstraction Purpose**: Provides a unified interface for accessing models regardless of their format (SafeTensors, GGUF, or Transformers). This allows the request processor to work with any model type without knowing the implementation details.

### 2.3 Backend Provider Interface

**File**: `services/localai/pkg/server/interfaces.go`

**Interface**: `BackendProvider` (line 29)
```go
type BackendProvider interface {
    // GetBackendType returns the backend type for a domain
    GetBackendType(domain string) string
    // IsBackendAvailable checks if a backend is available
    IsBackendAvailable(backendType string) bool
    // GetBackendEndpoint returns the endpoint for a backend
    GetBackendEndpoint(backendType, domain string) string
}
```

**Abstraction Purpose**: Abstracts backend selection logic, allowing the system to support multiple backends (vaultgemma, GGUF, transformers, OCR) and route requests appropriately.

### 2.4 Request Processor Interface

**File**: `services/localai/pkg/server/interfaces.go`

**Interface**: `RequestProcessor` (line 39)
```go
type RequestProcessor interface {
    // ProcessChatRequest processes a chat completion request
    ProcessChatRequest(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
    // ProcessStreamingRequest processes a streaming chat request
    ProcessStreamingRequest(ctx context.Context, req *ChatRequest, writer *StreamWriter) error
    // ProcessFunctionCallingRequest processes a function calling request
    ProcessFunctionCallingRequest(ctx context.Context, req *FunctionCallingRequest) (*FunctionCallingResponse, error)
}
```

**Abstraction Purpose**: Defines the contract for processing different types of requests (standard, streaming, function calling) without coupling to specific implementations.

### Implementation: VaultGemmaServer

**File**: `services/localai/pkg/server/vaultgemma_server.go`

**Type**: `VaultGemmaServer` struct (line 77)
```go
type VaultGemmaServer struct {
    models             map[string]*ai.VaultGemma
    ggufModels         map[string]*gguf.Model
    transformerClients map[string]*transformers.Client
    domainManager      *domain.DomainManager
    limiter            *rate.Limiter
    inferenceEngine    *inference.InferenceEngine
    enhancedEngine     *inference.EnhancedInferenceEngine
    hanaPool           *hanapool.Pool
    hanaLogger         *storage.HANALogger
    hanaCache          hanaCacheStore
    semanticCache      semanticCacheStore
    // ... additional fields
}
```

**Function**: `HandleChat()` (line 197)
```go
func (s *VaultGemmaServer) HandleChat(w http.ResponseWriter, r *http.Request) {
    // Validate and decode request
    req, err := validateChatRequest(r)
    
    // Build prompt from messages
    prompt := buildPromptFromMessages(req.Messages)
    
    // Detect or use specified domain
    domain := req.Model
    if domain == DomainAuto || domain == "" {
        domain = s.domainManager.DetectDomain(prompt, req.Domains)
    }
    
    // Retrieve domain configuration
    domainConfig, _ := s.domainManager.GetDomainConfig(domain)
    
    // Resolve model with fallback
    model, modelKey, fallbackUsed, fallbackKey, err := s.resolveModelForDomain(ctx, domain, domainConfig, preferredBackend)
    
    // Process chat request
    result, err := s.processChatRequest(ctx, req, domain, domainConfig, prompt, maxTokens, topP, topK, requestID, userID, sessionID)
    
    // Return response
    // ...
}
```

---

## Layer 3: Model Loading Abstraction

### 3.1 Model Loader Interface

**File**: `services/localai/pkg/models/model_loader.go`

**Type**: `ModelLoader` struct (line 13)
```go
type ModelLoader struct {
    modelPaths map[string]string
}
```

**Function**: `LoadModelFromSafeTensors()` (line 25)
```go
func (ml *ModelLoader) LoadModelFromSafeTensors(modelPath, domain string) (*ai.VaultGemma, error) {
    // Check if model path exists
    if _, err := os.Stat(modelPath); os.IsNotExist(err) {
        return nil, fmt.Errorf("model path does not exist: %s", modelPath)
    }
    
    // Check for required files
    configPath := filepath.Join(modelPath, "config.json")
    safetensorsPath := filepath.Join(modelPath, "model.safetensors")
    
    // Load model using the SafeTensors loader
    model, err := ai.LoadVaultGemmaFromSafetensors(modelPath)
    if err != nil {
        return nil, fmt.Errorf("failed to load model from SafeTensors: %w", err)
    }
    
    return model, nil
}
```

**Function**: `LoadModelFromPath()` (line 125)
```go
func (ml *ModelLoader) LoadModelFromPath(modelPath, domain string) (*ai.VaultGemma, error) {
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
    
    return nil, fmt.Errorf("no supported model format found in: %s", modelPath)
}
```

**Abstraction Purpose**: Provides a unified interface for loading models from different formats, automatically detecting the format and using the appropriate loader.

### 3.2 Model Registry

**File**: `services/localai/pkg/server/model_registry.go`

**Type**: `ModelRegistry` struct (line 20)
```go
type ModelRegistry struct {
    requirements map[string]*ModelGPURequirements
}
```

**Type**: `ModelGPURequirements` struct (line 11)
```go
type ModelGPURequirements struct {
    ModelType    string // "gemma-7b", "gemma-2b", "vaultgemma-1b", etc.
    MinMemoryMB  int64  // Minimum GPU memory needed
    RequiredGPUs int    // Number of GPUs (1 for most)
    Dedicated    bool   // Whether model needs dedicated GPU
    Priority     int    // Allocation priority (higher = more important)
}
```

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

**Abstraction Purpose**: Abstracts GPU resource requirements, allowing the system to make intelligent decisions about model placement and GPU allocation without hardcoding requirements.

---

## Abstraction Benefits Summary

### 1. Graph Service → LocalAI
- **Decoupling**: Graph service doesn't need to know about model internals
- **Standardization**: OpenAI-compatible interface allows provider swapping
- **Context Passing**: Workflow metadata can be passed through headers

### 2. LocalAI Internal Abstractions
- **Domain Management**: Easy addition of new domains via configuration
- **Model Provider**: Support for multiple model formats through unified interface
- **Backend Provider**: Flexible backend selection (CPU, GPU, external services)
- **Request Processing**: Consistent handling of different request types

### 3. Model Loading
- **Format Agnostic**: Automatic detection and loading of different formats
- **Resource Management**: Intelligent GPU allocation based on model requirements
- **Lazy Loading**: Models loaded on-demand to reduce startup time

## Code References

### Graph Service
- `services/graph/pkg/workflows/orchestration_processor.go:48` - `RunOrchestrationChainNode()`
- `services/graph/pkg/workflows/orchestration_processor.go:321` - `createOrchestrationChain()`
- `infrastructure/third_party/orchestration/llms/localai/localai.go:21` - `LLM` struct
- `infrastructure/third_party/orchestration/llms/localai/localai.go:161` - `GenerateContent()`

### LocalAI Service
- `services/localai/pkg/server/vaultgemma_server.go:77` - `VaultGemmaServer` struct
- `services/localai/pkg/server/vaultgemma_server.go:197` - `HandleChat()`
- `services/localai/pkg/domain/domain_config.go:12` - `DomainConfig` struct
- `services/localai/pkg/domain/domain_config.go:109` - `DomainManager` struct
- `services/localai/pkg/domain/domain_config.go:226` - `DetectDomain()`
- `services/localai/pkg/server/interfaces.go:17` - `ModelProvider` interface
- `services/localai/pkg/server/interfaces.go:29` - `BackendProvider` interface
- `services/localai/pkg/server/interfaces.go:39` - `RequestProcessor` interface

### Model Layer
- `services/localai/pkg/models/model_loader.go:13` - `ModelLoader` struct
- `services/localai/pkg/models/model_loader.go:25` - `LoadModelFromSafeTensors()`
- `services/localai/pkg/models/model_loader.go:125` - `LoadModelFromPath()`
- `services/localai/pkg/server/model_registry.go:20` - `ModelRegistry` struct
- `services/localai/pkg/server/model_registry.go:120` - `GetRequirements()`

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - High-level system overview
- [DATA_FLOW.md](./DATA_FLOW.md) - Request/response flow
- [MODEL_LOADING.md](./MODEL_LOADING.md) - Model discovery and loading
- [DIAGRAMS.md](./DIAGRAMS.md) - Visual architecture diagrams

