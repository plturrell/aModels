# Architecture Diagrams: Graph-LocalAI-Models Abstraction

This document contains Mermaid diagrams illustrating the abstraction layers between the Graph service, LocalAI service, and underlying models.

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Graph Service Layer"
        GW[Graph Workflow<br/>LangGraph StateGraph]
        OP[Orchestration Processor<br/>orchestration_processor.go]
        LLM[LocalAI Client<br/>llms/localai/localai.go]
    end
    
    subgraph "LocalAI Service Layer"
        HTTP[HTTP Server<br/>vaultgemma_server.go]
        DM[Domain Manager<br/>domain_config.go]
        MP[Model Provider<br/>interfaces.go]
        RP[Request Processor<br/>vaultgemma_server.go]
    end
    
    subgraph "Model Layer"
        ML[Model Loader<br/>model_loader.go]
        MR[Model Registry<br/>model_registry.go]
        ST[SafeTensors Models]
        GGUF[GGUF Models]
        HF[Transformers Models]
    end
    
    GW -->|"State Graph Execution"| OP
    OP -->|"createOrchestrationChain()"| LLM
    LLM -->|"HTTP POST /v1/chat/completions"| HTTP
    HTTP -->|"DetectDomain()"| DM
    HTTP -->|"ProcessChatRequest()"| RP
    RP -->|"GetSafetensorsModel()<br/>GetGGUFModel()<br/>GetTransformerClient()"| MP
    MP -->|"LoadModelFromPath()"| ML
    ML -->|"Register & Load"| MR
    MR -->|"Physical Files"| ST
    MR -->|"Physical Files"| GGUF
    MR -->|"External Service"| HF
    
    style GW fill:#e1f5ff
    style OP fill:#e1f5ff
    style LLM fill:#e1f5ff
    style HTTP fill:#fff4e1
    style DM fill:#fff4e1
    style MP fill:#fff4e1
    style RP fill:#fff4e1
    style ML fill:#e8f5e9
    style MR fill:#e8f5e9
    style ST fill:#e8f5e9
    style GGUF fill:#e8f5e9
    style HF fill:#e8f5e9
```

## Abstraction Layer Diagram

```mermaid
graph LR
    subgraph "Layer 1: Graph Service"
        A1[LangGraph Workflow]
        A2[Orchestration Chain]
        A3[LocalAI Client Wrapper]
    end
    
    subgraph "Layer 2: API Abstraction"
        B1[HTTP REST API<br/>OpenAI-Compatible]
        B2[Request/Response Format]
    end
    
    subgraph "Layer 3: LocalAI Service"
        C1[Domain Router]
        C2[Model Provider Interface]
        C3[Backend Provider Interface]
        C4[Request Processor Interface]
    end
    
    subgraph "Layer 4: Model Abstraction"
        D1[ModelLoader Interface]
        D2[Model Registry]
        D3[Backend Adapters]
    end
    
    subgraph "Layer 5: Physical Models"
        E1[SafeTensors Files]
        E2[GGUF Files]
        E3[Transformers Service]
    end
    
    A1 --> A2
    A2 --> A3
    A3 -->|"HTTP Client"| B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> E1
    D3 --> E2
    D3 --> E3
    
    style A1 fill:#bbdefb
    style A2 fill:#bbdefb
    style A3 fill:#bbdefb
    style B1 fill:#c8e6c9
    style B2 fill:#c8e6c9
    style C1 fill:#fff9c4
    style C2 fill:#fff9c4
    style C3 fill:#fff9c4
    style C4 fill:#fff9c4
    style D1 fill:#ffccbc
    style D2 fill:#ffccbc
    style D3 fill:#ffccbc
    style E1 fill:#f8bbd0
    style E2 fill:#f8bbd0
    style E3 fill:#f8bbd0
```

## Request Flow Diagram

```mermaid
sequenceDiagram
    participant GW as Graph Workflow
    participant OP as Orchestration Processor
    participant LC as LocalAI Client
    participant HTTP as LocalAI HTTP Server
    participant DM as Domain Manager
    participant RP as Request Processor
    participant MP as Model Provider
    participant ML as Model Loader
    participant Model as Physical Model
    
    GW->>OP: Execute workflow node
    OP->>OP: Extract chain configuration
    OP->>OP: createOrchestrationChain()
    OP->>LC: New(localai.WithBaseURL())
    LC->>HTTP: POST /v1/chat/completions
    Note over LC,HTTP: OpenAI-compatible JSON request
    
    HTTP->>HTTP: Validate request
    HTTP->>HTTP: buildPromptFromMessages()
    HTTP->>DM: DetectDomain(prompt)
    DM->>DM: Score domains by keywords
    DM-->>HTTP: Selected domain
    
    HTTP->>RP: ProcessChatRequest()
    RP->>MP: GetSafetensorsModel(domain)
    alt Model in cache
        MP-->>RP: Cached model
    else Model not loaded
        MP->>ML: LoadModelFromPath()
        ML->>Model: Load from filesystem
        Model-->>ML: Model instance
        ML-->>MP: Loaded model
        MP-->>RP: Model instance
    end
    
    RP->>Model: Generate(prompt, params)
    Model-->>RP: Generated text
    RP-->>HTTP: ChatResponse
    HTTP-->>LC: JSON response
    LC-->>OP: ContentResponse
    OP->>OP: Extract text output
    OP-->>GW: Updated state
```

## Model Loading Flow Diagram

```mermaid
flowchart TD
    Start([Server Startup]) --> LoadConfig[Load domains.json]
    LoadConfig --> InitDM[Initialize DomainManager]
    InitDM --> ParseConfig[Parse domain configurations]
    
    ParseConfig --> CheckBackend{Backend Type?}
    
    CheckBackend -->|vaultgemma| SafeTensors[Register SafeTensors Path]
    CheckBackend -->|gguf| GGUF[Register GGUF Path]
    CheckBackend -->|hf-transformers| Transformers[Register Transformers Client]
    CheckBackend -->|deepseek-ocr| OCR[Register OCR Service]
    
    SafeTensors --> LazyLoad{Enable Lazy Loading?}
    GGUF --> LazyLoad
    Transformers --> RegisterClient[Create HTTP Client]
    OCR --> RegisterOCR[Create OCR Service]
    
    LazyLoad -->|Yes| RegisterOnly[Register model path only]
    LazyLoad -->|No| LoadNow[Load model immediately]
    
    RegisterOnly --> WaitRequest[Wait for first request]
    LoadNow --> ValidateModel[Validate model structure]
    RegisterClient --> Ready[Service Ready]
    RegisterOCR --> Ready
    ValidateModel --> Ready
    WaitRequest --> FirstRequest[First request arrives]
    
    FirstRequest --> CheckCache{Model in cache?}
    CheckCache -->|Yes| UseCached[Use cached model]
    CheckCache -->|No| LoadFromFS[Load from filesystem]
    
    LoadFromFS --> LoadSafeTensors{Format?}
    LoadSafeTensors -->|SafeTensors| LoadST[ai.LoadVaultGemmaFromSafetensors]
    LoadSafeTensors -->|GGUF| LoadGGUF[gguf.LoadModel]
    LoadSafeTensors -->|Config| LoadConfig[ai.NewVaultGemma]
    
    LoadST --> CacheModel[Cache model instance]
    LoadGGUF --> CacheModel
    LoadConfig --> CacheModel
    CacheModel --> UseCached
    UseCached --> Ready
    
    style Start fill:#e1f5ff
    style Ready fill:#c8e6c9
    style LoadNow fill:#fff9c4
    style RegisterOnly fill:#fff9c4
    style CacheModel fill:#ffccbc
```

## Component Interaction Diagram

```mermaid
graph TB
    subgraph "Graph Service Components"
        G1[UnifiedProcessorWorkflow]
        G2[RunOrchestrationChainNode]
        G3[createOrchestrationChain]
        G4[LocalAI LLM Client]
    end
    
    subgraph "LocalAI Service Components"
        L1[VaultGemmaServer]
        L2[HandleChat]
        L3[DomainManager]
        L4[ModelProvider]
        L5[BackendProvider]
    end
    
    subgraph "Model Management"
        M1[ModelLoader]
        M2[ModelRegistry]
        M3[ModelCache]
    end
    
    subgraph "Backend Implementations"
        B1[SafeTensors Backend]
        B2[GGUF Backend]
        B3[Transformers Backend]
        B4[OCR Backend]
    end
    
    G1 -->|"Invoke workflow"| G2
    G2 -->|"Create chain"| G3
    G3 -->|"New() with options"| G4
    G4 -->|"HTTP POST"| L1
    L1 -->|"Route request"| L2
    L2 -->|"DetectDomain()"| L3
    L2 -->|"GetModel()"| L4
    L4 -->|"GetBackendType()"| L5
    L4 -->|"Load if needed"| M1
    M1 -->|"Check registry"| M2
    M1 -->|"Check cache"| M3
    M2 -->|"Select backend"| B1
    M2 -->|"Select backend"| B2
    M2 -->|"Select backend"| B3
    M2 -->|"Select backend"| B4
    
    style G1 fill:#e3f2fd
    style G2 fill:#e3f2fd
    style G3 fill:#e3f2fd
    style G4 fill:#e3f2fd
    style L1 fill:#fff3e0
    style L2 fill:#fff3e0
    style L3 fill:#fff3e0
    style L4 fill:#fff3e0
    style L5 fill:#fff3e0
    style M1 fill:#f1f8e9
    style M2 fill:#f1f8e9
    style M3 fill:#f1f8e9
    style B1 fill:#fce4ec
    style B2 fill:#fce4ec
    style B3 fill:#fce4ec
    style B4 fill:#fce4ec
```

## Code Reference Map

### Graph Service → LocalAI Interface

**File**: `services/graph/pkg/workflows/orchestration_processor.go`
- **Function**: `createOrchestrationChain()` (line 321)
- **Function**: `RunOrchestrationChainNode()` (line 48)

**File**: `infrastructure/third_party/orchestration/llms/localai/localai.go`
- **Type**: `LLM` struct (line 21)
- **Function**: `New()` (line 98)
- **Function**: `GenerateContent()` (line 161)

### LocalAI Service Internal

**File**: `services/localai/pkg/server/vaultgemma_server.go`
- **Type**: `VaultGemmaServer` struct (line 77)
- **Function**: `HandleChat()` (line 197)

**File**: `services/localai/pkg/domain/domain_config.go`
- **Type**: `DomainManager` struct (line 109)
- **Function**: `DetectDomain()` (line 226)

**File**: `services/localai/pkg/server/interfaces.go`
- **Interface**: `ModelProvider` (line 17)
- **Interface**: `BackendProvider` (line 29)
- **Interface**: `RequestProcessor` (line 39)

**File**: `services/localai/pkg/models/model_loader.go`
- **Type**: `ModelLoader` struct (line 13)
- **Function**: `LoadModelFromSafeTensors()` (line 25)
- **Function**: `LoadModelFromPath()` (line 125)

**File**: `services/localai/pkg/server/model_registry.go`
- **Type**: `ModelRegistry` struct (line 20)
- **Function**: `GetRequirements()` (line 120)

