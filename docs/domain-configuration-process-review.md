# Domain Configuration Process Review

## Executive Summary

This document provides a comprehensive end-to-end review of the domain configuration process in the aModels platform, covering how domains are established, identified during extraction, trained, and finally mounted/deployed.

## Table of Contents

1. [Domain Definition and Configuration](#1-domain-definition-and-configuration)
2. [Domain Loading and Configuration Sources](#2-domain-loading-and-configuration-sources)
3. [Domain Identification During Extraction](#3-domain-identification-during-extraction)
4. [Domain Training Process](#4-domain-training-process)
5. [Domain Deployment and Mounting](#5-domain-deployment-and-mounting)
6. [Domain Routing and Runtime](#6-domain-routing-and-runtime)
7. [Complete Lifecycle Flow](#7-complete-lifecycle-flow)
8. [Gaps and Issues Identified](#8-gaps-and-issues-identified)

---

## 1. Domain Definition and Configuration

### 1.1 Domain Schema

Domains are defined in `services/localai/config/domains.json` using a JSON schema. Each domain represents a specialized AI agent with its own model configuration.

**Key Components:**

- **Domain ID**: Unique identifier (e.g., `"0x3579-VectorProcessingAgent"`)
- **Domain Configuration**: Full configuration object with all parameters
- **Layer Mapping**: Hierarchical structure organizing domains by layer and team

### 1.2 Domain Configuration Structure

**Location**: `services/localai/config/domains.json`

```json
{
  "domains": {
    "0x3579-VectorProcessingAgent": {
      "name": "Vector Processing Agent",
      "layer": "layer1",
      "team": "DataTeam",
      "backend_type": "hf-transformers",
      "model_name": "phi-3.5-mini",
      "model_path": "",
      "transformers_config": {
        "endpoint": "http://transformers-service:9090/v1/chat/completions",
        "model_name": "phi-3.5-mini",
        "timeout_seconds": 60
      },
      "agent_id": "0x3579",
      "attention_weights": {
        "vector_operations": 0.95,
        "embeddings": 0.9,
        "similarity": 0.85
      },
      "max_tokens": 256,
      "temperature": 0.3,
      "tags": ["vector", "embedding", "similarity", "data"],
      "keywords": ["vector", "embedding", "similarity", "cosine", "dimension", "tensor"]
    }
  },
  "default_domain": "general",
  "layer_mapping": {
    "layer1": {
      "DataTeam": ["0x3579-VectorProcessingAgent", ...],
      "FoundationTeam": [...]
    }
  }
}
```

### 1.3 Domain Configuration Fields

**File**: `services/localai/pkg/domain/domain_config.go`

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `name` | string | Human-readable domain name | Yes |
| `layer` | string | Architectural layer (layer1, layer2, layer3, layer4) | Yes |
| `team` | string | Team ownership (DataTeam, FoundationTeam, etc.) | Yes |
| `agent_id` | string | Unique hex identifier (e.g., "0x3579") | Yes |
| `backend_type` | string | Backend: `gguf`, `hf-transformers`, `vaultgemma`, `deepseek-ocr` | Conditional |
| `model_path` | string | Path to model file (GGUF) or directory (safetensors) | Conditional |
| `model_name` | string | Model identifier for API backends | Conditional |
| `transformers_config` | object | Configuration for hf-transformers backend | If backend_type=hf-transformers |
| `vision_config` | object | Configuration for deepseek-ocr backend | If backend_type=deepseek-ocr |
| `keywords` | array[string] | Keywords for domain detection/routing | Recommended |
| `tags` | array[string] | Categorization tags | Optional |
| `attention_weights` | object | Task-specific attention weights | Optional |
| `max_tokens` | integer | Maximum tokens per response | Yes |
| `temperature` | float | Sampling temperature (0.0-2.0) | Yes |
| `enabled_env_var` | string | Environment variable toggle | Optional |

### 1.4 Layer Mapping Structure

The `layer_mapping` provides hierarchical organization:

```
layer_mapping: {
  "layer1": {
    "DataTeam": ["0x3579-VectorProcessingAgent", "0x5678-SQLAgent", ...],
    "FoundationTeam": [...]
  },
  "layer2": {
    "QualityControl": [...]
  },
  "layer3": {
    "FinanceOperations": [...]
  },
  "layer4": {
    "BrowserTeam": [...],
    "VisionTeam": [],
    "FoundationTeam": [...]
  }
}
```

### 1.5 Domain Validation

**File**: `services/localai/pkg/domain/domain_config.go` - `Validate()` method

Validation rules:
- Domain name must not be empty
- Either `model_path` or `model_name` required (unless `backend_type=deepseek-ocr`)
- `hf-transformers` backend requires `transformers_config` with `endpoint` and `model_name`
- `deepseek-ocr` backend requires `vision_config` with `endpoint` or `script_path`
- `max_tokens` must be positive
- `temperature` must be between 0.0 and 2.0
- `top_p` must be between 0.0 and 1.0
- `top_k` must be non-negative

### 1.6 Environment Variable Toggles

Domains can be conditionally enabled via `enabled_env_var`:

**File**: `services/localai/pkg/domain/domain_config.go` - `isDomainEnabled()`

```go
func isDomainEnabled(envVar string) bool {
    if envVar == "" {
        return true  // Enabled by default
    }
    value, exists := os.LookupEnv(envVar)
    if !exists {
        return false  // Not enabled if env var doesn't exist
    }
    // Check if value is truthy (not "0", "false", "no", "off")
    normalized := strings.TrimSpace(strings.ToLower(value))
    return normalized != "" && normalized != "0" && normalized != "false" && 
           normalized != "no" && normalized != "off"
}
```

**Example**: `"enabled_env_var": "ENABLE_GEMMA7B"` requires `ENABLE_GEMMA7B=1` in environment.

---

## 2. Domain Loading and Configuration Sources

### 2.1 Configuration Source Priority

**File**: `services/localai/pkg/domain/config_loader.go` - `NewConfigLoader()`

Configuration sources are checked in this order:

1. **Redis** (preferred for production)
   - Checked via `REDIS_URL` environment variable
   - Key: `REDIS_DOMAIN_CONFIG_KEY` (default: `localai:domains:config`)
2. **File** (fallback)
   - Path: `DOMAIN_CONFIG_PATH` environment variable (default: `config/domains.json`)

### 2.2 Redis Configuration Loading

**File**: `services/localai/pkg/domain/redis_config.go`

**Process:**

1. **Connection**: `NewRedisConfigLoader()` creates Redis client and tests connection
2. **Loading**: `LoadDomainConfigs()` retrieves JSON from Redis key
3. **Parsing**: JSON is unmarshaled into `DomainsConfig` structure
4. **Filtering**: Domains are filtered by:
   - `enabled_env_var` check
   - Validation
5. **Default Domain**: Default domain is determined from config or falls back to "general"
6. **Hot Reloading**: `WatchConfig()` periodically reloads config from Redis

**Redis Key Format**: 
- Key: `localai:domains:config` (or `REDIS_DOMAIN_CONFIG_KEY`)
- Value: Complete `domains.json` structure as JSON string

### 2.3 PostgreSQL Configuration Store

**File**: `services/localai/pkg/domain/postgres_config.go`

**Table Schema**: `domain_configs`

```sql
CREATE TABLE domain_configs (
    id SERIAL PRIMARY KEY,
    domain_name VARCHAR(255) UNIQUE NOT NULL,
    config_json JSONB NOT NULL,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    training_run_id VARCHAR(255),
    model_version VARCHAR(255),
    performance_metrics JSONB
);
```

**Key Methods:**
- `GetAllDomainConfigs()`: Retrieves all enabled domain configs
- `SaveDomainConfig()`: Saves/updates domain config with training metadata
- `SyncToRedis()`: Syncs PostgreSQL configs to Redis for fast access

**Training Integration**: PostgreSQL stores domain configs linked to training runs via `training_run_id` and `model_version`.

### 2.4 File-Based Configuration

**File**: `services/localai/pkg/domain/config_loader.go` - `FileConfigLoader`

**Process:**
1. Reads JSON file from path
2. Parses into `DomainsConfig` structure
3. Validates and filters domains
4. Loads into `DomainManager`

**Entrypoint Handling**: `services/localai/entrypoint.sh` handles path resolution if `domains.json` is mounted as a directory.

### 2.5 Server Startup Process

**File**: `services/localai/cmd/vaultgemma-server/main.go`

**Startup Sequence:**

```go
// 1. Create domain manager
domainManager := domain.NewDomainManager()

// 2. Create config loader (Redis or File)
configLoader, err := domain.NewConfigLoader()
if err != nil {
    // Fallback to file-based config
    configLoader = &domain.FileConfigLoader{path: *configPath}
}

// 3. Load domain configs
if err := configLoader.LoadDomainConfigs(context.Background(), domainManager); err != nil {
    log.Printf("⚠️  Failed to load domain configs: %v")
    // Continue with single model mode
} else {
    log.Printf("✅ Loaded domain configs from %s", domain.GetConfigSource())
}
```

---

## 3. Domain Identification During Extraction

### 3.1 Current State

**Finding**: The extraction service (`services/extract`) does **NOT** currently associate extracted data with specific domains or agent IDs.

**Files Reviewed:**
- `services/extract/main.go`
- `services/extract/extract_logic.go`
- `services/extract/advanced_extraction.go`
- `services/extract/catalog.go`

### 3.2 Extraction Process

**Extract Service** processes:
- JSON tables (`json_with_changes.json`)
- Hive DDL files (`.hql`)
- SQL queries
- Control-M XML files

**Output:**
- Knowledge graph (nodes and edges) in Neo4j
- CSV files for training
- Glean batches for catalog export

**Missing Link**: Extracted data does not include:
- Domain assignments
- Agent ID associations
- Domain-specific metadata

### 3.3 Training Data Output

**Training Output Location**: `data/training/extracts` (configurable via `TRAINING_OUTPUT_DIR`)

**Format**: CSV files with structured data:
- `table_columns.csv`
- `table_relationships.csv`
- `view_dependencies.csv`

**Gap**: Training data is not linked to specific domains at extraction time.

### 3.4 Potential Domain Association Points

**Hypothetical Integration Points:**

1. **Keyword Matching**: Match extracted patterns to domain keywords
   - Example: SQL queries → `0x5678-SQLAgent`
   - Example: Vector operations → `0x3579-VectorProcessingAgent`

2. **Agent ID in Metadata**: Store `agent_id` in Neo4j node properties
   - Nodes: `properties_json` field could include `agent_id`
   - Edges: Relationship metadata could include domain assignments

3. **Training Pipeline**: Associate domains during training data preparation
   - `services/training/pipeline.py` could match patterns to domains
   - Use domain keywords to tag training data

---

## 4. Domain Training Process

### 4.1 Training Pipeline Overview

**File**: `services/training/pipeline.py` - `TrainingPipeline.run_full_pipeline()`

**Steps:**

1. **Extract Knowledge Graph** from source data
2. **Query Glean Catalog** for historical patterns (optional)
3. **Learn Patterns** from knowledge graph and Glean data
4. **Generate Training Features**
5. **Prepare Training Dataset**

### 4.2 Training Data Sources

**Sources:**
- Extract service knowledge graph
- Glean Catalog historical data
- Pattern learning algorithms
- Temporal analysis from change history

**No Direct Domain Link**: Training pipeline does not currently filter or organize data by domain.

### 4.3 Pattern Learning

**File**: `services/training/pattern_learning.py`

**Pattern Types:**
- Column type patterns
- Relationship patterns
- Metadata entropy patterns
- Workflow patterns (from Petri nets)
- Temporal patterns (from change history)

**Domain Association**: Patterns are learned generically, not per-domain.

### 4.4 Training Model Output

**Training Scripts:**
- `tools/scripts/train_relational_transformer.py`
- Model checkpoints saved to `./checkpoints/`

**Output:**
- Trained model files (`.pt`)
- Training metrics
- Evaluation results

### 4.5 Loading Trained Models into Domains

**File**: `services/localai/scripts/load_domains_from_training.py`

**Process:**

```python
load_domain_config_from_training(
    postgres_dsn=postgres_dsn,
    domain_name="general",
    config=domain_config,
    training_run_id="training_run_001",
    model_version="phi-3.5-mini-v1",
    performance_metrics={
        "accuracy": 0.85,
        "latency_ms": 120,
        "tokens_per_second": 45.2,
        "training_loss": 0.023,
        "validation_loss": 0.028
    }
)
```

**PostgreSQL Integration:**
- Saves domain config to `domain_configs` table
- Links to training run via `training_run_id`
- Stores performance metrics in `performance_metrics` JSONB field
- Updates version number on conflict

**Redis Sync**: PostgreSQL configs can be synced to Redis via `PostgresConfigStore.SyncToRedis()`.

### 4.6 Training-to-Domain Flow

**Current Flow:**
```
Training Pipeline → Model Checkpoint → load_domains_from_training.py → PostgreSQL domain_configs → Redis → LocalAI Server
```

**Gap**: No automatic association during training - manual script execution required.

---

## 5. Domain Deployment and Mounting

### 5.1 Model Loading Process

**File**: `services/localai/cmd/vaultgemma-server/main.go` - Model loading loop

**Process:**

1. **Iterate Domain Configs**: Loop through all loaded domain configurations
2. **Backend Type Resolution**: Check `backend_type` field
3. **Model Loading**:
   - **GGUF**: Load via `gguf.Load(modelPath)` if path ends with `.gguf`
   - **hf-transformers**: Create `transformers.Client` with endpoint configuration
   - **VaultGemma**: Load safetensors from directory path
   - **deepseek-ocr**: Configure vision endpoint/script
4. **Model Registration**: Store in appropriate registry map

### 5.2 Backend Type Handling

**GGUF Models** (`backend_type: "gguf"`):
```go
if strings.HasSuffix(lowerPath, ".gguf") {
    gm, err := gguf.Load(cfgModelPath)
    ggufModels[name] = gm
}
```

**Transformers** (`backend_type: "hf-transformers"`):
```go
if strings.EqualFold(cfg.BackendType, "hf-transformers") {
    client := transformers.NewClient(
        cfg.TransformersConfig.Endpoint,
        cfg.TransformersConfig.ModelName,
        timeout
    )
    transformerClients[name] = client
}
```

**VaultGemma** (default):
```go
loadedModel, err := ai.LoadVaultGemmaFromSafetensors(cfgModelPath)
models[name] = loadedModel
```

### 5.3 Docker Volume Mounting

**File**: `infrastructure/docker/brev/docker-compose.yml`

**Model Volumes:**
```yaml
localai:
  volumes:
    - ../../models:/models:ro
    - ../../services/localai/config:/workspace/config:ro
```

**Model Path Resolution:**
- GGUF models: `/models/gemma-2b-q4_k_m.gguf` (absolute Docker path)
- Safetensors: `/models/vaultgemma-transformers-1b-v1/` (directory path)
- Transformers: HTTP endpoint (no local file)

### 5.4 Model Path Configuration

**Model Path Formats:**
- **GGUF**: Absolute path to `.gguf` file
  - Example: `"/models/gemma-2b-q4_k_m.gguf"`
- **Safetensors**: Directory path containing model files
  - Example: `"../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1"`
- **Transformers**: Not used (endpoint in `transformers_config`)

### 5.5 Server Model Registries

**File**: `services/localai/cmd/vaultgemma-server/main.go`

**Registry Types:**
- `models map[string]*ai.VaultGemma` - VaultGemma safetensor models
- `ggufModels map[string]*gguf.Model` - GGUF quantized models
- `transformerClients map[string]*transformers.Client` - Transformers HTTP clients

**Domain-to-Model Mapping:**
- Domain name (e.g., `"0x5678-SQLAgent"`) → Model instance
- Multiple domains can share the same model instance (deduplication)

---

## 6. Domain Routing and Runtime

### 6.1 Domain Detection

**File**: `services/localai/pkg/domain/domain_manager.go` - `DetectDomain()`

**Algorithm:**
1. Lowercase prompt text
2. Score each domain by keyword matches
3. Select domain with highest score
4. Fallback to default domain if no matches

**Code:**
```go
func (dm *DomainManager) DetectDomain(prompt string, userDomains []string) string {
    promptLower := strings.ToLower(prompt)
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

### 6.2 Intelligent Routing

**File**: `services/localai/pkg/routing/intelligent_router.go`

**Features:**
- Query complexity analysis
- Model capability matching
- Performance-based routing
- Fallback domain selection
- Alternative route suggestions

**Routing Decision Factors:**
- **Keyword Matching** (40% weight)
- **Domain Expertise** (30% weight)
- **Technical Level Matching** (20% weight)
- **Performance Considerations** (10% weight)

**Query Complexity Analysis:**
- Token count estimation
- Domain specificity detection
- Technical level assessment
- Reasoning requirements

### 6.3 Runtime Domain Selection

**Process:**
1. User query arrives
2. `IntelligentRouter.RouteQuery()` analyzes query
3. Domain scores calculated for available domains
4. Best domain selected (or fallback)
5. Model instance retrieved from registry
6. Request routed to appropriate backend

**Model Selection:**
- GGUF → `ggufModels[domainName]`
- Transformers → `transformerClients[domainName]`
- VaultGemma → `models[domainName]`

---

## 7. Complete Lifecycle Flow

### 7.1 Domain Lifecycle Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DOMAIN DEFINITION                                            │
│    domains.json → DomainConfig struct                            │
│    - Schema validation                                           │
│    - Environment variable toggles                               │
│    - Layer mapping structure                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CONFIGURATION LOADING                                         │
│    Redis (preferred) → File (fallback)                          │
│    - RedisConfigLoader.LoadDomainConfigs()                       │
│    - FileConfigLoader.LoadDomainConfigs()                        │
│    - PostgreSQL domain_configs table (optional)                 │
│    - DomainManager initialization                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. EXTRACTION (Gap: No domain association)                      │
│    Source Data → Extract Service → Knowledge Graph              │
│    - JSON tables, DDLs, SQL, Control-M                          │
│    - Neo4j nodes/edges                                           │
│    - CSV training data                                           │
│    ⚠️  Missing: Agent ID association, domain tagging            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. TRAINING                                                     │
│    Knowledge Graph → Training Pipeline → Model Checkpoint       │
│    - Pattern learning                                            │
│    - Glean Catalog integration                                   │
│    - Model training                                              │
│    - Performance metrics collection                              │
│    ⚠️  Missing: Domain-specific training data filtering          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. DOMAIN CONFIG UPDATE                                         │
│    Training Results → load_domains_from_training.py              │
│    → PostgreSQL domain_configs → Redis sync                     │
│    - Training run ID linking                                     │
│    - Model version tracking                                      │
│    - Performance metrics storage                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. DEPLOYMENT/MOUNTING                                          │
│    Domain Config → Model Loading → Server Registration          │
│    - GGUF model loading (gguf.Load())                           │
│    - Transformers client creation                                │
│    - VaultGemma safetensor loading                               │
│    - Model registry population                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. RUNTIME ROUTING                                              │
│    User Query → Domain Detection → Model Selection → Response    │
│    - Keyword-based domain detection                             │
│    - Intelligent routing (query complexity)                      │
│    - Model capability matching                                   │
│    - Fallback handling                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Configuration Source Chain

```
┌─────────────────┐
│ domains.json    │  (Source of truth)
└────────┬────────┘
         │
         ├──→ Redis (localai:domains:config)  ← Preferred for production
         │
         ├──→ PostgreSQL (domain_configs table)  ← Training integration
         │     └──→ SyncToRedis() → Redis
         │
         └──→ File (DOMAIN_CONFIG_PATH)  ← Fallback
                 │
                 └──→ DomainManager (in-memory)
                         │
                         └──→ LocalAI Server (runtime)
```

### 7.3 Data Flow: Extraction → Training → Deployment

**Current Flow:**
```
Source Data
    ↓
Extract Service (no domain tagging)
    ↓
Knowledge Graph (Neo4j)
    ↓
Training Pipeline (generic pattern learning)
    ↓
Model Training (domain-agnostic)
    ↓
Model Checkpoint
    ↓
Manual: load_domains_from_training.py
    ↓
PostgreSQL domain_configs
    ↓
Redis sync
    ↓
LocalAI Server (domain config loaded)
    ↓
Model mounting (per domain)
    ↓
Runtime routing
```

**Missing Connections:**
- Extraction → Domain association
- Training → Domain-specific filtering
- Automatic domain config updates

---

## 8. Gaps and Issues Identified

### 8.1 Critical Gaps

#### Gap 1: No Domain Association During Extraction

**Impact**: HIGH

**Issue**: Extract service does not associate extracted patterns with domains or agent IDs.

**Current State:**
- Extraction produces knowledge graph nodes/edges
- No domain metadata in extracted data
- No agent_id linking

**Recommendation:**
- Add domain keyword matching during extraction
- Store `agent_id` in Neo4j node properties
- Tag training data with domain assignments

#### Gap 2: No Domain-Specific Training Data Filtering

**Impact**: MEDIUM

**Issue**: Training pipeline processes all data generically, not per-domain.

**Current State:**
- Pattern learning is domain-agnostic
- Training data not organized by domain
- No domain-specific model training

**Recommendation:**
- Filter training data by domain keywords
- Create domain-specific training datasets
- Train domain-specific models

#### Gap 3: Manual Domain Config Updates from Training

**Impact**: MEDIUM

**Issue**: Training results must be manually loaded into domain configs via script.

**Current State:**
- `load_domains_from_training.py` requires manual execution
- No automatic integration after training completes

**Recommendation:**
- Integrate domain config updates into training pipeline
- Automatically update PostgreSQL after training
- Trigger Redis sync automatically

### 8.2 Configuration Issues

#### Issue 1: File Mount Path Conflict

**Location**: `infrastructure/docker/brev/docker-compose.yml`

**Issue**: `domains.json` mounted as directory instead of file.

**Current**: Volume mount creates directory at `/workspace/config/domains.json`

**Workaround**: Entrypoint script handles directory case, but Redis is preferred.

#### Issue 2: Redis Config Loading Not Working

**Issue**: Redis config loader fails silently, falls back to file.

**Root Cause**: Likely Redis connection or validation issues.

**Recommendation**: Add better error logging and connection diagnostics.

### 8.3 Integration Gaps

#### Gap 4: No Direct Link Between Training and Domain Deployment

**Impact**: LOW

**Issue**: Trained models must be manually associated with domain configurations.

**Recommendation**: 
- Automatic model path updates in domain config after training
- Version tracking for model updates
- Automatic deployment triggers

#### Gap 5: No Domain-Specific Metrics Collection

**Impact**: LOW

**Issue**: Performance metrics are collected generically, not per-domain.

**Recommendation**:
- Track domain-specific performance metrics
- Store in PostgreSQL `performance_metrics` field
- Use for routing decisions

### 8.4 Summary of Gaps

| Gap | Impact | Priority | Status |
|-----|--------|----------|--------|
| No domain association during extraction | HIGH | Critical | Identified |
| No domain-specific training filtering | MEDIUM | High | Identified |
| Manual domain config updates | MEDIUM | High | Identified |
| File mount path conflict | LOW | Medium | Workaround exists |
| Redis config loading issues | MEDIUM | Medium | Needs investigation |
| No automatic training→deployment | LOW | Low | Enhancement |

---

## 9. Recommendations

### 9.1 Immediate Actions

1. **Fix Redis Configuration Loading**
   - Add connection diagnostics
   - Improve error logging
   - Verify Redis connectivity

2. **Add Domain Association to Extraction**
   - Implement keyword matching during extraction
   - Store `agent_id` in Neo4j properties
   - Tag training data with domain assignments

### 9.2 Short-Term Improvements

1. **Domain-Specific Training**
   - Filter training data by domain keywords
   - Create domain-specific training datasets
   - Train domain-specific models

2. **Automatic Domain Config Updates**
   - Integrate `load_domains_from_training.py` into training pipeline
   - Automatic PostgreSQL updates after training
   - Automatic Redis sync

### 9.3 Long-Term Enhancements

1. **Domain Metrics Collection**
   - Track domain-specific performance
   - Store in PostgreSQL
   - Use for intelligent routing

2. **Automatic Deployment**
   - Model version tracking
   - Automatic domain config updates
   - Deployment triggers

---

## 10. Review Questions Answered

### Q1: How are domains initially created/defined?

**Answer**: Domains are defined in `services/localai/config/domains.json` with a JSON schema including:
- Domain ID (unique identifier)
- Layer and team assignment
- Backend type and model configuration
- Keywords for routing
- Attention weights and other parameters

### Q2: How are domains identified during data extraction?

**Answer**: Currently, domains are **NOT** identified during extraction. This is a critical gap. Extraction produces generic knowledge graphs without domain associations.

### Q3: How does training link to specific domains?

**Answer**: Training does **NOT** currently link to specific domains. Training is domain-agnostic, processing all data generically. Manual script (`load_domains_from_training.py`) links training results to domains after training completes.

### Q4: How are trained models associated with domain configurations?

**Answer**: After training completes, `load_domains_from_training.py` script:
1. Takes domain config and training results
2. Saves to PostgreSQL `domain_configs` table
3. Links via `training_run_id` and `model_version`
4. Stores performance metrics
5. Can sync to Redis for fast access

### Q5: How are domains loaded and mounted at runtime?

**Answer**: 
1. LocalAI server starts
2. Config loader (Redis or File) loads domain configs
3. DomainManager initialized with configs
4. Server iterates domain configs
5. Models loaded based on `backend_type`:
   - GGUF: `gguf.Load(modelPath)`
   - Transformers: `transformers.NewClient(endpoint, modelName)`
   - VaultGemma: `ai.LoadVaultGemmaFromSafetensors(path)`
6. Models registered in appropriate registry maps

### Q6: What is the complete data flow from extraction → training → deployment?

**Answer**: See Section 7.3 for complete flow. Key points:
- Extraction: Generic (no domain tagging)
- Training: Generic (no domain filtering)
- Deployment: Manual script execution required
- Runtime: Domain-based routing works

### Q7: Are there any gaps or missing connections in the process?

**Answer**: Yes, see Section 8 for complete gap analysis. Critical gaps:
1. No domain association during extraction
2. No domain-specific training filtering
3. Manual domain config updates required

---

## Conclusion

The domain configuration system has a solid foundation with:
- ✅ Well-defined domain schema
- ✅ Multiple configuration sources (Redis, PostgreSQL, File)
- ✅ Robust validation and loading
- ✅ Intelligent runtime routing

**Phase 1 Implementation Status** (2025-11-04):
- ✅ Domain association during extraction (with Neo4j storage)
- ✅ Domain-specific training data filtering (with differential privacy)
- ✅ Automated domain config updates (with differential privacy)
- ✅ Privacy budget tracking and management

**Overall Assessment**: The system is now fully integrated end-to-end with domain awareness at extraction, training, and deployment stages. Differential privacy is integrated throughout to protect sensitive information.

**See**: `docs/domain-configuration-phase1-dp-integration.md` for Phase 1 implementation details.

---

**Document Version**: 2.0  
**Last Updated**: 2025-11-04  
**Author**: Domain Configuration Review  
**Phase 1**: ✅ Complete with Differential Privacy Integration

