# Data Flow: Request/Response Tracing

This document traces the complete flow of a request from the Graph service through LocalAI to the underlying models, with specific code references at each step.

## Overview

A typical request flows through these stages:
1. **Graph Workflow** - Workflow execution triggers orchestration chain
2. **Orchestration Processor** - Creates LocalAI client and prepares request
3. **LocalAI Client** - Sends HTTP request to LocalAI service
4. **LocalAI Server** - Receives, validates, and routes request
5. **Domain Detection** - Determines appropriate domain/model
6. **Model Resolution** - Loads or retrieves model instance
7. **Inference** - Executes model inference
8. **Response** - Formats and returns result

## Detailed Flow

### Step 1: Graph Workflow Execution

**Location**: `services/graph/pkg/workflows/unified_processor.go`

**Function**: `ProcessUnifiedWorkflowNode()` (line 84)

```go
func ProcessUnifiedWorkflowNode(opts UnifiedProcessorOptions) stategraph.NodeFunc {
    return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
        // Extract orchestration request from state
        if unifiedReq.OrchestrationRequest != nil {
            orchState := map[string]any{
                "orchestration_request": map[string]any{
                    "chain_name": unifiedReq.OrchestrationRequest.ChainName,
                    "inputs":      unifiedReq.OrchestrationRequest.Inputs,
                },
            }
            // Execute orchestration chain
            orchNode := RunOrchestrationChainNode(opts.LocalAIURL)
            orchResult, err := orchNode(ctx, orchState)
            // ...
        }
    })
}
```

**State Input**:
- `orchestration_request.chain_name`: Chain type (e.g., "llm_chain", "qa")
- `orchestration_request.inputs`: Chain input parameters
- `knowledge_graph`: Optional KG context for enrichment

**State Output**:
- `orchestration_result`: Chain execution results
- `orchestration_text`: Extracted text output

---

### Step 2: Orchestration Chain Creation

**Location**: `services/graph/pkg/workflows/orchestration_processor.go`

**Function**: `RunOrchestrationChainNode()` (line 48)

```go
func RunOrchestrationChainNode(localAIURL string) stategraph.NodeFunc {
    return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
        // Extract chain configuration
        chainName := extractChainName(state)
        chainInputs := extractChainInputs(state)
        
        // Create orchestration chain
        chain, err := createOrchestrationChain(chainName, localAIURL, headers)
        if err != nil {
            return nil, fmt.Errorf("create chain %s: %w", chainName, err)
        }
        
        // Execute chain
        result, execErr = chains.Call(ctx, chain, chainInputs)
        // ...
    })
}
```

**Function**: `createOrchestrationChain()` (line 321)

```go
func createOrchestrationChain(chainName, localAIURL string, headers map[string]string) (chains.Chain, error) {
    // Create LocalAI LLM instance
    opts := []localai.Option{localai.WithBaseURL(localAIURL)}
    if headers != nil && len(headers) > 0 {
        opts = append(opts, localai.WithHeaders(headers))
    }
    llm, err := localai.New(opts...)
    if err != nil {
        return nil, fmt.Errorf("create LocalAI LLM: %w", err)
    }
    
    // Create chain based on chain name
    switch chainName {
    case "llm_chain", "default":
        promptTemplate := prompts.NewPromptTemplate(
            "Answer the following question or task:\n\n{{.input}}",
            []string{"input"},
        )
        return chains.NewLLMChain(llm, promptTemplate), nil
    // ... other chain types
    }
}
```

**Key Actions**:
- Creates LocalAI client with base URL
- Adds workflow context headers (X-Workflow-ID, X-Workflow-Priority)
- Creates prompt template for chain type
- Wraps LLM in chain structure

---

### Step 3: LocalAI Client Request

**Location**: `infrastructure/third_party/orchestration/llms/localai/localai.go`

**Function**: `GenerateContent()` (line 161)

```go
func (l *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
    // Convert messages to LocalAI chat format
    chatMessages := make([]chatMessage, 0, len(messages))
    for _, msg := range messages {
        chatMessages = append(chatMessages, chatMessage{
            Role:    convertRole(msg.Role),
            Content: extractContent(msg.Parts),
        })
    }
    
    // Build request
    reqBody := chatCompletionRequest{
        Model:       l.model,        // "auto" for domain detection
        Messages:    chatMessages,
        Temperature: l.temperature,
        MaxTokens:   l.maxTokens,
        Domains:     l.domains,
    }
    
    jsonData, err := json.Marshal(reqBody)
    
    // Create HTTP request
    req, err := http.NewRequestWithContext(ctx, "POST", l.baseURL+"/v1/chat/completions", bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    
    // Add custom headers (workflow context)
    for k, v := range l.headers {
        req.Header.Set(k, v)
    }
    
    // Send request
    resp, err := l.client.Do(req)
    // Parse response...
}
```

**HTTP Request**:
```json
POST /v1/chat/completions
Content-Type: application/json
X-Workflow-ID: workflow-123
X-Workflow-Priority: 5

{
    "model": "auto",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
}
```

---

### Step 4: LocalAI Server Request Handling

**Location**: `services/localai/pkg/server/vaultgemma_server.go`

**Function**: `HandleChat()` (line 197)

```go
func (s *VaultGemmaServer) HandleChat(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), RequestTimeoutDefault)
    defer cancel()
    
    // Validate and decode request
    req, err := validateChatRequest(r)
    if err != nil {
        handleChatError(w, err, http.StatusBadRequest)
        return
    }
    
    // Build prompt from messages
    prompt := buildPromptFromMessages(req.Messages)
    prompt = s.enrichPromptWithAgentCatalog(prompt)
    
    // Detect or use specified domain
    domain := req.Model
    if domain == DomainAuto || domain == "" {
        domain = s.domainManager.DetectDomain(prompt, req.Domains)
        log.Printf("Auto-detected domain: %s", domain)
    }
    
    // Retrieve domain configuration
    domainConfig, _ := s.domainManager.GetDomainConfig(domain)
    
    // Resolve model with fallback
    model, modelKey, fallbackUsed, fallbackKey, err := s.resolveModelForDomain(ctx, domain, domainConfig, preferredBackend)
    
    // Process chat request
    result, err := s.processChatRequest(ctx, req, domain, domainConfig, model, modelKey, prompt, maxTokens, topP, topK, requestID, userID, sessionID)
    
    // Build and send response
    // ...
}
```

**Key Actions**:
- Validates HTTP request and JSON body
- Extracts prompt from messages
- Detects domain (if "auto" or empty)
- Retrieves domain configuration
- Resolves model instance

---

### Step 5: Domain Detection

**Location**: `services/localai/pkg/domain/domain_config.go`

**Function**: `DetectDomain()` (line 226)

```go
func (dm *DomainManager) DetectDomain(prompt string, userDomains []string) string {
    promptLower := strings.ToLower(prompt)
    
    // Score each domain based on keyword matches
    bestScore := 0
    bestDomain := dm.defaultDomain
    
    for domainName, config := range dm.domains {
        // Skip if user doesn't have access
        if len(userDomains) > 0 && !userDomainMap[domainName] {
            continue
        }
        
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

**Example**:
- Prompt: "Analyze SQL query performance"
- Keywords matched: "sql" → SQL domain
- Result: Domain "sql" selected

---

### Step 6: Model Resolution

**Location**: `services/localai/pkg/server/chat_helpers.go`

**Function**: `resolveModelForDomain()` (line 128)

```go
func (s *VaultGemmaServer) resolveModelForDomain(
    ctx context.Context,
    domain string,
    domainConfig *domain.DomainConfig,
    preferredBackend string,
) (model *ai.VaultGemma, modelKey string, fallbackUsed bool, fallbackKey string, err error) {
    // Determine if we need a safetensors model
    requireModel := true
    if domainConfig != nil {
        if strings.EqualFold(domainConfig.BackendType, BackendTypeTransformers) {
            requireModel = false  // Uses external service
        } else if strings.HasSuffix(domainConfig.ModelPath, ".gguf") {
            requireModel = false  // Uses GGUF backend
        }
    }
    
    if !requireModel {
        return nil, domain, false, "", nil
    }
    
    // Try to get model from cache
    if model, ok := s.models[domain]; ok {
        return model, domain, false, "", nil
    }
    
    // Try fallback model
    if domainConfig != nil && domainConfig.FallbackModel != "" {
        if fallbackModel, ok := s.models[domainConfig.FallbackModel]; ok {
            return fallbackModel, domainConfig.FallbackModel, true, domainConfig.FallbackModel, nil
        }
    }
    
    // Try default "general" domain
    if generalModel, ok := s.models["general"]; ok {
        return generalModel, "general", true, "general", nil
    }
    
    return nil, domain, false, "", fmt.Errorf("no model available for domain: %s", domain)
}
```

**Model Resolution Strategy**:
1. Check if backend requires SafeTensors model
2. Try domain-specific model from cache
3. Try fallback model (if configured)
4. Try default "general" domain
5. Return error if no model available

---

### Step 7: Chat Request Processing

**Location**: `services/localai/pkg/server/chat_helpers.go`

**Function**: `processChatRequest()` (line 231)

```go
func (s *VaultGemmaServer) processChatRequest(
    ctx context.Context,
    req *ChatRequestInternal,
    domain string,
    domainConfig *domain.DomainConfig,
    model *ai.VaultGemma,
    modelKey string,
    prompt string,
    maxTokens int,
    topP float64,
    topK int,
    requestID string,
    userID string,
    sessionID string,
) (*ChatProcessingResult, error) {
    result := &ChatProcessingResult{
        ModelKey:    modelKey,
        Domain:      domain,
        DomainConfig: domainConfig,
    }
    
    // Determine backend type
    backendType := ""
    if domainConfig != nil {
        backendType = strings.TrimSpace(domainConfig.BackendType)
    }
    
    // Handle DeepSeek OCR backend
    if strings.EqualFold(backendType, BackendTypeDeepSeekOCR) {
        return s.processDeepSeekOCR(ctx, req, domain, domainConfig, prompt, requestID, userID, sessionID, result)
    }
    
    // Handle Transformers backend
    if strings.EqualFold(backendType, BackendTypeTransformers) {
        return s.processTransformersBackend(ctx, req, domain, domainConfig, maxTokens, topP, prompt, requestID, userID, sessionID, result)
    }
    
    // Check cache first
    if s.hanaCache != nil {
        cacheResult, err := s.checkCache(ctx, prompt, modelKey, domain, req.Temperature, maxTokens, topP, topK, requestID, userID, sessionID)
        if err == nil && cacheResult != nil {
            result.Content = cacheResult.Content
            result.TokensUsed = cacheResult.TokensUsed
            result.CacheHit = cacheResult.CacheHit
            return result, nil
        }
    }
    
    // Try GGUF backend
    if ggufModel := s.ggufModels[modelKey]; ggufModel != nil {
        // Generate with GGUF model
        // ...
    }
    
    // Generate with SafeTensors model
    if model != nil {
        generated, tokensUsed, err := s.inferenceEngine.Generate(ctx, model, prompt, maxTokens, topP, topK, req.Temperature)
        result.Content = generated
        result.TokensUsed = tokensUsed
        return result, err
    }
    
    return nil, fmt.Errorf("no available backend for domain: %s", domain)
}
```

**Backend Selection**:
1. **DeepSeek OCR**: For vision/image processing
2. **Transformers**: External Python service
3. **Cache**: Check HANA cache first
4. **GGUF**: Quantized models via llama.cpp
5. **SafeTensors**: Native Go implementation

---

### Step 8: Model Inference

**Location**: `services/localai/pkg/inference/inference.go` and `services/localai/pkg/models/ai/vaultgemma.go`

**Inference Request Structure**:
```go
// InferenceRequest contains what the agent queries
type InferenceRequest struct {
    Prompt      string        // The actual text prompt to process
    Domain      string        // Domain name (e.g., "general", "sql")
    MaxTokens   int           // Maximum tokens to generate
    Temperature float64       // Sampling temperature (0.0-2.0)
    TopP        float64       // Nucleus sampling threshold (0.0-1.0)
    TopK        int           // Top-k sampling (number of candidates)
    Model       *ai.VaultGemma // The actual model instance
}
```

**Detailed Inference Process**:

#### 8.1 Tokenization

**Location**: `services/localai/pkg/inference/inference.go:117`

```go
func (e *InferenceEngine) tokenizeInput(prompt string, model *ai.VaultGemma) ([]int, error) {
    // Convert text prompt to token IDs
    // Example: "What is the capital of France?" → [1, 1234, 567, 890, 123, 456, 789, 2]
    // Where 1 = BOS token, 2 = EOS token
    
    tokens := make([]int, 0, len(prompt)/4)
    words := splitIntoWords(prompt)
    for _, word := range words {
        token := hashString(word) % vocabSize
        tokens = append(tokens, token)
    }
    tokens = append([]int{bosID}, tokens...) // Add BOS token
    return tokens, nil
}
```

**What the agent queries**: The prompt text is converted to a sequence of integer token IDs that the model can process.

#### 8.2 Forward Pass Through Model

**Location**: `services/localai/pkg/models/ai/vaultgemma.go:480`

```go
func (vg *VaultGemma) GenerateWithSampling(prompt []int, maxTokens int, cfg SamplingConfig) ([]int, error) {
    // Step 1: Initial forward pass on prompt tokens
    logits, err := vg.Forward(prompt)
    // logits shape: [seqLen, vocabSize] - probability distribution over vocabulary
    
    total := append([]int(nil), prompt...)
    
    // Step 2: Autoregressive generation loop
    for step := 0; step < maxTokens; step++ {
        // Extract logits for last position
        lastLogits := logits.Data[(logits.Rows-1)*logits.Cols : logits.Rows*logits.Cols]
        
        // Sample next token from logits
        nextToken := sampleLogits(lastLogits, effectiveCfg, rng)
        total = append(total, nextToken)
        
        if nextToken == vg.Config.EOSTokenID {
            break // End of sequence
        }
        
        // Step 3: Process single new token through model
        // Embed the new token
        nextHidden := vg.embedTokens([]int{nextToken})
        
        // Run through all transformer layers
        for layerIndex := range vg.Layers {
            layer := &vg.Layers[layerIndex]
            
            // Layer norm 1
            normed := vg.rmsNorm(nextHidden, layer.LayerNorm1)
            
            // Multi-head self-attention (with KV cache)
            attnOut := vg.multiHeadAttentionWithCache(normed, layer.SelfAttention, &cache[layerIndex])
            
            // Residual connection
            residual := vg.addResidual(nextHidden, attnOut)
            
            // Layer norm 2
            normed = vg.rmsNorm(residual, layer.LayerNorm2)
            
            // Feed-forward network
            ffnOut := vg.feedForward(normed, layer.FeedForward)
            
            // Residual connection
            nextHidden = vg.addResidual(residual, ffnOut)
        }
        
        // Step 4: Project to vocabulary logits
        logits = vg.projectOutput(nextHidden)
        // logits shape: [1, vocabSize] - probabilities for next token
    }
    
    return total, nil
}
```

**What the agent queries at each step**:

1. **Initial Forward Pass**: 
   - Input: Tokenized prompt `[1, 1234, 567, 890, ...]`
   - Process: Embeddings → Transformer layers → Output projection
   - Output: Logits `[seqLen, vocabSize]` - probability distribution over vocabulary

2. **Autoregressive Generation Loop** (for each new token):
   - Input: Single new token ID (e.g., `123`)
   - Process:
     - **Embedding**: Token ID → Embedding vector `[hiddenSize]`
     - **Transformer Layers** (for each of N layers):
       - **Self-Attention**: Query, Key, Value matrices → Attention scores → Contextualized representation
       - **Feed-Forward**: Two linear projections with SiLU activation
       - **Residual Connections**: Add input to output at each step
       - **Layer Normalization**: RMSNorm normalization
     - **Output Projection**: Hidden state → Vocabulary logits `[vocabSize]`
   - Output: Logits for next token prediction

3. **Token Sampling**:
   - Input: Logits `[vocabSize]` (raw scores)
   - Process:
     - Apply temperature scaling: `logits / temperature`
     - Apply top-k filtering: Keep only top K tokens
     - Apply top-p (nucleus) filtering: Keep tokens until cumulative probability > p
     - Apply softmax: Convert to probabilities
     - Sample: Randomly select token based on probability distribution
   - Output: Next token ID

#### 8.3 Model Architecture Query

**What the model actually processes**:

```
Input Prompt: "What is the capital of France?"
    ↓
Tokenization: [1, 1234, 567, 890, 123, 456, 789]
    ↓
Embedding Layer:
    Token IDs → Embedding vectors [seqLen, hiddenSize]
    Example: token 1234 → [0.1, -0.3, 0.5, ..., 0.2] (hiddenSize dimensions)
    ↓
Transformer Layer 1:
    Self-Attention:
        Q = hiddenStates @ WQ  [seqLen, hiddenSize]
        K = hiddenStates @ WK  [seqLen, hiddenSize]
        V = hiddenStates @ WV  [seqLen, hiddenSize]
        Attention = softmax(Q @ K^T / sqrt(headDim)) @ V
    Feed-Forward:
        gate = SiLU(hiddenStates @ W3)
        up = hiddenStates @ W1
        output = (gate * up) @ W2
    ↓
Transformer Layer 2-N: (same process)
    ↓
Output Layer:
    Hidden states → Logits [seqLen, vocabSize]
    Example: [0.01, 0.05, 0.001, ..., 0.03] (probabilities for each vocab token)
    ↓
Sampling:
    Apply temperature/top-p/top-k → Sample next token
    Example: token 2345 selected (represents "Paris")
    ↓
Repeat for maxTokens iterations...
```

#### 8.4 Enhanced Inference with KV Cache

**Location**: `services/localai/pkg/inference/enhanced_inference.go:227`

```go
func (e *EnhancedInferenceEngine) generateTokensWithTensorOps(
    ctx context.Context,
    model *ai.VaultGemma,
    inputTokens []int,
    maxTokens int,
    temperature, topP float64,
    topK int,
    minP float64,
    useMinP bool,
) ([]int, []float64, error) {
    // Check for cached prompt prefix (KV cache optimization)
    cachedPrefix, prefixLen := e.findLongestMatchingPrefix(inputTokens)
    
    if cachedPrefix != nil && prefixLen > 0 {
        // Reuse KV cache from previous computation
        // This avoids recomputing attention for common prefixes
        kvCache = e.cloneKVCache(cachedPrefix.KVCache)
        remainingTokens = inputTokens[prefixLen:]
    }
    
    // Generate with KV cache reuse
    sequence, err := e.generateWithKVCache(ctx, model, remainingTokens, maxTokens, samplingCfg, kvCache)
    
    // Cache the full prompt for future reuse
    e.cachePromptPrefix(inputTokens, kvCache)
    
    return generated, logProbs, nil
}
```

**What the agent queries with KV cache**:
- **KV Cache**: Stores Key and Value matrices from attention layers
- **Purpose**: Avoid recomputing attention for prompt prefixes that have been seen before
- **Benefit**: Faster inference when prompts share common prefixes (e.g., system prompts, few-shot examples)

#### 8.5 Actual Model Query Example

**Complete query flow for a single generation step**:

```go
// 1. Input: Token sequence [1, 1234, 567, 890]
// 2. Embedding lookup:
embeddings = model.Embed.Weights[token_ids]  // [4, hiddenSize]

// 3. For each transformer layer:
for layer in model.Layers:
    // Self-attention
    Q = embeddings @ layer.SelfAttention.WQ  // [4, hiddenSize]
    K = embeddings @ layer.SelfAttention.WK  // [4, hiddenSize]
    V = embeddings @ layer.SelfAttention.WV  // [4, hiddenSize]
    
    // Attention computation
    scores = Q @ K^T / sqrt(headDim)  // [4, 4]
    attention = softmax(scores) @ V    // [4, hiddenSize]
    
    // Feed-forward
    gate = SiLU(embeddings @ layer.FeedForward.W3)
    up = embeddings @ layer.FeedForward.W1
    ffn = (gate * up) @ layer.FeedForward.W2
    
    // Residual + norm
    embeddings = RMSNorm(embeddings + attention + ffn)

// 4. Output projection
logits = embeddings[-1] @ model.Output.Weights  // [vocabSize]

// 5. Sampling
nextToken = sample(logits, temperature, topP, topK)  // e.g., 2345
```

**What the agent queries**: 
- **Model weights**: Embedding matrix, attention weights (WQ, WK, WV, WO), feed-forward weights (W1, W2, W3), layer norms, output projection
- **Computation**: Matrix multiplications, attention scores, activation functions (SiLU), normalization (RMSNorm)
- **Output**: Probability distribution over vocabulary, sampled token ID

#### 8.6 Backend-Specific Queries

**SafeTensors Backend** (`services/localai/pkg/inference/inference.go:55`):
```go
response := s.inferenceEngine.GenerateResponse(ctx, &inference.InferenceRequest{
    Prompt:      prompt,      // Text string
    Domain:      domain,      // Domain name
    MaxTokens:   maxTokens,   // Generation limit
    Temperature: temperature, // Sampling temperature
    TopP:        topP,        // Nucleus sampling
    TopK:        topK,        // Top-k sampling
    Model:       model,       // VaultGemma model instance
})
```

**GGUF Backend** (via llama.cpp):
- Queries the GGUF model file directly
- Uses llama.cpp's optimized inference engine
- Supports GPU acceleration via CUDA/Metal

**Transformers Backend** (`services/localai/services/transformers_cpu_server.py:194`):
```python
# Python service receives:
inputs = tokenizer(req.prompt, return_tensors="pt")
output_ids = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    do_sample=temperature > 0,
)
text = tokenizer.decode(output_ids[0, input_ids.shape[1]:])
```

**What the agent queries in Transformers backend**:
- HTTP POST to Python service with prompt text
- Python service uses HuggingFace transformers library
- Model runs forward pass using PyTorch
- Returns generated text

---

### Step 9: Response Formatting

**Location**: `services/localai/pkg/server/vaultgemma_server.go`

**Function**: `HandleChat()` (after processing)

```go
// Build OpenAI-compatible response
response := map[string]interface{}{
    "id":      requestID,
    "object":  "chat.completion",
    "created": time.Now().Unix(),
    "model":   modelKey,
    "choices": []map[string]interface{}{
        {
            "index": 0,
            "message": map[string]interface{}{
                "role":    "assistant",
                "content": result.Content,
            },
            "finish_reason": "stop",
        },
    },
    "usage": map[string]interface{}{
        "prompt_tokens":     len(prompt) / 4, // Estimate
        "completion_tokens": result.TokensUsed,
        "total_tokens":      len(prompt)/4 + result.TokensUsed,
    },
}

w.Header().Set("Content-Type", "application/json")
json.NewEncoder(w).Encode(response)
```

**HTTP Response**:
```json
{
    "id": "req_1234567890",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "general",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}
```

---

### Step 10: Client Response Handling

**Location**: `infrastructure/third_party/orchestration/llms/localai/localai.go`

**Function**: `GenerateContent()` (response parsing)

```go
var chatResp chatCompletionResponse
if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
    return nil, fmt.Errorf("decode response: %w", err)
}

if len(chatResp.Choices) == 0 {
    return nil, fmt.Errorf("no choices in response")
}

text := chatResp.Choices[0].Message.Content

response := &llms.ContentResponse{
    Choices: []*llms.ContentChoice{
        {
            Content: text,
            GenerationInfo: map[string]any{
                "model":         chatResp.Model,
                "finish_reason": chatResp.Choices[0].FinishReason,
            },
        },
    },
}

return response, nil
```

---

### Step 11: Chain Result Extraction

**Location**: `services/graph/pkg/workflows/orchestration_processor.go`

**Function**: `RunOrchestrationChainNode()` (result handling)

```go
// Execute chain
result, execErr = chains.Call(ctx, chain, chainInputs)

// Store results in state
newState := make(map[string]any, len(state)+5)
for k, v := range state {
    newState[k] = v
}
newState["orchestration_result"] = result
newState["orchestration_chain_name"] = chainName
newState["orchestration_success"] = true
newState["orchestration_executed_at"] = time.Now().Format(time.RFC3339)

// Extract text output
if text, ok := result["text"].(string); ok {
    newState["orchestration_text"] = text
}

return newState, nil
```

---

## Complete Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Graph Workflow                                           │
│    ProcessUnifiedWorkflowNode()                             │
│    → Extracts orchestration_request from state              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. Orchestration Processor                                  │
│    RunOrchestrationChainNode()                              │
│    → Creates orchestration chain                            │
│    → Calls createOrchestrationChain()                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. LocalAI Client                                           │
│    localai.New() → LLM struct                              │
│    GenerateContent()                                        │
│    → Builds HTTP request                                    │
│    → POST /v1/chat/completions                              │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP POST
┌────────────────────▼────────────────────────────────────────┐
│ 4. LocalAI Server                                           │
│    HandleChat()                                             │
│    → Validates request                                      │
│    → Builds prompt from messages                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. Domain Detection                                         │
│    DomainManager.DetectDomain()                             │
│    → Scores domains by keywords                             │
│    → Returns best matching domain                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 6. Model Resolution                                         │
│    resolveModelForDomain()                                  │
│    → Checks cache                                           │
│    → Tries fallback                                         │
│    → Returns model instance                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 7. Request Processing                                       │
│    processChatRequest()                                     │
│    → Selects backend (SafeTensors/GGUF/Transformers)       │
│    → Checks cache                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 8. Model Inference                                          │
│    InferenceEngine.Generate()                              │
│    → Tokenizes prompt                                       │
│    → Runs through model                                     │
│    → Samples tokens                                         │
│    → Decodes to text                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 9. Response Formatting                                      │
│    Build OpenAI-compatible response                         │
│    → JSON with choices, usage, metadata                     │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP 200 OK
┌────────────────────▼────────────────────────────────────────┐
│ 10. Client Response                                         │
│     Parse JSON response                                     │
│     → Extract text content                                  │
│     → Return ContentResponse                                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 11. Chain Result                                            │
│     Extract text from result                                │
│     → Update workflow state                                 │
│     → Return to Graph workflow                              │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling Flow

At each step, errors are handled and propagated:

1. **Graph Service**: Errors wrapped with context, returned in state
2. **LocalAI Client**: HTTP errors converted to Go errors
3. **LocalAI Server**: HTTP status codes (400, 500, 502)
4. **Model Loading**: File system errors, validation errors
5. **Inference**: Model execution errors, timeout errors

## Performance Considerations

1. **Caching**: HANA cache checked before inference
2. **Lazy Loading**: Models loaded on first use
3. **Connection Pooling**: HTTP client with connection reuse
4. **Timeout Management**: Context timeouts at each layer
5. **Resource Management**: GPU allocation and release

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture overview
- [ABSTRACTION_LAYERS.md](./ABSTRACTION_LAYERS.md) - Abstraction layer details
- [MODEL_LOADING.md](./MODEL_LOADING.md) - Model loading mechanisms
- [DIAGRAMS.md](./DIAGRAMS.md) - Visual flow diagrams

