# Phases 7-9: Domain Integration Plan (Adjusted Based on Actual Code)

## Overview

After reviewing the actual Phase 7-9 implementations, this document provides **adjusted integration points** to add domain awareness to existing code without duplicating functionality.

## Current State Analysis

### ✅ Phase 7-9 Code Already Implemented

**Phase 7 - Pattern Learning**:
- ✅ `pattern_learning_gnn.py` - GNN implementation (no domain awareness)
- ✅ `meta_pattern_learner.py` - Meta-patterns (has optional `domains` param, minimal integration)
- ✅ `sequence_pattern_transformer.py` - Sequence patterns (no domain awareness)
- ✅ `active_pattern_learner.py` - Active learning (no domain awareness)

**Phase 8 - Extraction & Intelligence**:
- ✅ `semantic_schema_analyzer.go` - Semantic analysis (has domain inference, not integrated with domain system)
- ✅ `cross_system_extractor.go` - Cross-system extraction (no domain awareness)
- ✅ `model_fusion.go` - Model fusion (no domain awareness)
- ✅ `pattern_transfer.py` - Pattern transfer (no domain awareness)

**Phase 9 - Automation**:
- ✅ `auto_tuner.py` - Auto-tuning (no domain awareness)
- ✅ `self_healing.go` - Self-healing (no domain awareness)
- ✅ `auto_pipeline.go` - Auto-pipeline (no domain awareness)
- ✅ `agent_coordinator.go` - Agent coordination (no domain awareness)
- ✅ `predictive_analytics.go` - Predictive analytics (no domain awareness)
- ✅ `recommendation_engine.go` - Recommendation engine (no domain awareness)

### 🔍 Integration Opportunities

1. **Domain System Integration**: All components need domain manager integration
2. **SAP RPT Enhancement**: Use SAP RPT embeddings for domain-aware semantic analysis
3. **Cross-Domain Learning**: Integrate with Phase 4 cross-domain capabilities
4. **Health Scoring**: Use domain health metrics for self-healing decisions
5. **Templates**: Use domain templates for configuration

---

## Integration Plan: Add Domain Awareness to Existing Code

### Phase 7: Domain-Aware Pattern Learning

#### 7.1: Enhance GNN Pattern Learning (`pattern_learning_gnn.py`)

**Current State**: Generic GNN implementation
**Integration**: Add domain-specific GNN models

**Changes Needed**:

```python
# Add to GNNRelationshipPatternLearner.__init__
def __init__(
    self,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    use_gat: bool = False,
    domain_manager=None  # NEW: Add domain manager
):
    # ... existing init ...
    self.domain_manager = domain_manager
    self.domain_models = {}  # domain_id -> GNN model

# Add new method
def learn_domain_patterns(
    self,
    domain_id: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Learn patterns specific to a domain."""
    # Get or create domain-specific model
    if domain_id not in self.domain_models:
        self.domain_models[domain_id] = self._create_domain_model(domain_id)
    
    # Convert to PyG with domain context
    data = self.convert_graph_to_pyg_data_with_domain(nodes, edges, domain_id)
    
    # Learn patterns
    patterns = self.learn_patterns_from_data(data, domain_id)
    
    return patterns

def convert_graph_to_pyg_data_with_domain(
    self,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    domain_id: str
) -> Optional[Data]:
    """Convert graph with domain context."""
    # Use domain config for node/edge features
    domain_config = self.domain_manager.get_domain_config(domain_id) if self.domain_manager else None
    
    # Enhance node features with domain keywords
    for node in nodes:
        features = self._extract_node_features(node, {})
        if domain_config:
            # Add domain-specific features
            domain_features = self._extract_domain_features(node, domain_config)
            features = np.concatenate([features, domain_features])
        node["domain_features"] = features
    
    return self.convert_graph_to_pyg_data(nodes, edges)
```

**Integration Points**:
- Import `DomainManager` from domain system
- Use domain config for feature extraction
- Store patterns per domain

---

#### 7.2: Enhance Meta-Pattern Learning (`meta_pattern_learner.py`)

**Current State**: Has optional `domains` parameter but minimal integration
**Integration**: Full domain-aware meta-pattern learning

**Changes Needed**:

```python
# Enhance learn_meta_patterns to use domain system
def learn_meta_patterns(
    self,
    learned_patterns: Dict[str, Any],
    domains: Optional[List[str]] = None,
    domain_manager=None  # NEW: Add domain manager
) -> Dict[str, Any]:
    """Learn meta-patterns with domain awareness."""
    # ... existing code ...
    
    # NEW: Use domain manager if available
    if domain_manager:
        # Group patterns by domain layer/team
        domain_grouped = self._group_patterns_by_domain_layer(
            learned_patterns, domain_manager, domains
        )
        
        # Learn layer-specific meta-patterns
        layer_patterns = self._learn_layer_meta_patterns(domain_grouped)
        
        # Learn team-specific meta-patterns
        team_patterns = self._learn_team_meta_patterns(domain_grouped)
        
        meta_patterns["layer_patterns"] = layer_patterns
        meta_patterns["team_patterns"] = team_patterns
    
    # Enhanced cross-domain patterns
    if domains and len(domains) > 1:
        # Use Phase 4 cross-domain learning
        from .routing_optimizer import RoutingOptimizer
        optimizer = RoutingOptimizer()
        
        # Find similar domains
        similar_domains = optimizer.find_similar_domains(domains[0])
        
        # Transfer patterns between similar domains
        cross_domain_patterns = self._transfer_patterns_across_similar_domains(
            learned_patterns, similar_domains
        )
        
        meta_patterns["cross_domain_patterns"] = cross_domain_patterns
    
    return meta_patterns
```

**Integration Points**:
- Use domain manager for layer/team grouping
- Integrate with Phase 4 cross-domain learning
- Store meta-patterns per domain/layer/team

---

#### 7.3: Enhance Sequence Pattern Transformer (`sequence_pattern_transformer.py`)

**Current State**: Generic sequence learning
**Integration**: Domain-conditioned sequences

**Changes Needed**:

```python
# Add domain conditioning to sequence model
class DomainSequencePatternTransformer:
    """Domain-aware sequence pattern transformer."""
    
    def __init__(self, domain_manager=None):
        self.domain_manager = domain_manager
        self.domain_embeddings = {}  # domain_id -> embedding
    
    def learn_domain_sequences(
        self,
        domain_id: str,
        sequences: List[List[str]]
    ) -> Dict[str, Any]:
        """Learn sequences specific to a domain."""
        # Get domain embedding for conditioning
        domain_embedding = self._get_domain_embedding(domain_id)
        
        # Learn sequences with domain context
        patterns = self._learn_sequences_with_context(
            sequences, domain_embedding
        )
        
        return patterns
    
    def _get_domain_embedding(self, domain_id: str) -> np.ndarray:
        """Get domain embedding for conditioning."""
        if domain_id in self.domain_embeddings:
            return self.domain_embeddings[domain_id]
        
        # Use SAP RPT for domain embedding
        if self.domain_manager:
            domain_config = self.domain_manager.get_domain_config(domain_id)
            if domain_config:
                # Embed domain keywords using SAP RPT
                keywords = " ".join(domain_config.get("keywords", []))
                # Call SAP RPT embedding service
                embedding = self._embed_with_sap_rpt(keywords)
                self.domain_embeddings[domain_id] = embedding
                return embedding
        
        return np.zeros(384)  # Default SAP RPT dimension
```

**Integration Points**:
- Domain conditioning for sequence models
- SAP RPT embeddings for domain context
- Domain-specific sequence patterns

---

#### 7.4: Enhance Active Pattern Learning (`active_pattern_learner.py`)

**Current State**: Generic active learning
**Integration**: Domain-aware active learning

**Changes Needed**:

```python
# Add domain awareness to active learning
def discover_patterns(
    self,
    learned_patterns: Dict[str, Any],
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    domain_id: Optional[str] = None,  # NEW
    domain_manager=None  # NEW
) -> Dict[str, Any]:
    """Discover patterns with domain awareness."""
    # ... existing discovery ...
    
    # NEW: Filter by domain if specified
    if domain_id and domain_manager:
        domain_config = domain_manager.get_domain_config(domain_id)
        if domain_config:
            # Filter nodes/edges by domain
            domain_nodes = self._filter_by_domain(graph_nodes, domain_id)
            domain_edges = self._filter_by_domain(graph_edges, domain_id)
            
            # Discover domain-specific patterns
            domain_patterns = self._discover_new_patterns(
                domain_nodes, domain_edges, learned_patterns
            )
            
            # Use domain keywords for pattern validation
            domain_keywords = domain_config.get("keywords", [])
            validated = self._validate_with_domain_keywords(
                domain_patterns, domain_keywords
            )
            
            result["domain_patterns"] = validated
            result["domain_id"] = domain_id
    
    return result
```

**Integration Points**:
- Domain filtering for pattern discovery
- Domain keyword validation
- Domain-specific pattern taxonomy

---

### Phase 8: Domain-Aware Extraction & Intelligence

#### 8.1: Integrate Semantic Schema Analyzer (`semantic_schema_analyzer.go`)

**Current State**: Has domain inference but not integrated with domain system
**Integration**: Connect to domain manager and use domain configs

**Changes Needed**:

```go
// Add domain manager to SemanticSchemaAnalyzer
type SemanticSchemaAnalyzer struct {
    logger              *log.Logger
    useSAPRPTEmbeddings bool
    useGloveEmbeddings  bool
    extractServiceURL   string
    domainManager       *domain.DomainManager  // NEW
}

// Enhance AnalyzeColumnSemantics
func (ssa *SemanticSchemaAnalyzer) AnalyzeColumnSemantics(
    ctx context.Context,
    columnName string,
    columnType string,
    tableName string,
    tableContext map[string]any,
    domainID string,  // NEW: Add domain ID parameter
) (*SemanticColumnAnalysis, error) {
    // ... existing analysis ...
    
    // NEW: Use domain manager if available
    if ssa.domainManager != nil && domainID != "" {
        domainConfig, exists := ssa.domainManager.GetDomainConfig(domainID)
        if exists {
            // Use domain keywords for enhanced inference
            domainKeywords := domainConfig.Keywords
            enhancedDomain, enhancedConf := ssa.inferDomainWithKeywords(
                columnName, tableName, domainKeywords,
            )
            
            if enhancedConf > analysis.DomainConfidence {
                analysis.InferredDomain = enhancedDomain
                analysis.DomainConfidence = enhancedConf
            }
            
            // Use domain tags for semantic similarity
            domainTags := domainConfig.DomainTags
            if ssa.useSAPRPTEmbeddings {
                similarities := ssa.calculateSemanticSimilarityWithDomain(
                    ctx, columnName, tableName, domainTags,
                )
                analysis.SemanticSimilarity = similarities
            }
        }
    }
    
    return analysis, nil
}
```

**Integration Points**:
- Domain manager integration
- Domain keywords for enhanced inference
- SAP RPT embeddings with domain context

---

#### 8.2: Enhance Model Fusion (`model_fusion.go`)

**Current State**: Generic model fusion
**Integration**: Domain-optimized fusion

**Changes Needed**:

```go
// Add domain awareness to ModelFusionFramework
type ModelFusionFramework struct {
    logger            *log.Logger
    useRelationalTransformer bool
    useSAPRPT         bool
    useGlove          bool
    weights           ModelWeights
    domainManager     *domain.DomainManager  // NEW
    domainWeights     map[string]ModelWeights  // domain_id -> weights
}

// Enhance FusePredictions with domain awareness
func (mff *ModelFusionFramework) FusePredictions(
    ctx context.Context,
    predictions []ModelPrediction,
    fusionMethod string,
    domainID string,  // NEW: Add domain ID
) (*FusedPrediction, error) {
    // ... existing fusion ...
    
    // NEW: Use domain-specific weights if available
    if mff.domainManager != nil && domainID != "" {
        if domainWeights, exists := mff.domainWeights[domainID]; exists {
            // Use domain-specific weights
            mff.weights = domainWeights
        } else {
            // Optimize weights for domain
            domainWeights = mff.optimizeWeightsForDomain(domainID, predictions)
            mff.domainWeights[domainID] = domainWeights
            mff.weights = domainWeights
        }
    }
    
    // Perform fusion with domain-optimized weights
    return mff.weightedAverageFusion(predictions)
}

func (mff *ModelFusionFramework) optimizeWeightsForDomain(
    domainID string,
    predictions []ModelPrediction,
) ModelWeights {
    // Use Phase 4 routing optimizer for weight optimization
    // Get domain-specific performance metrics
    // Optimize weights based on domain characteristics
    return ModelWeights{
        RelationalTransformer: 0.4,
        SAPRPT:               0.5,  // Higher for semantic-rich domains
        Glove:                0.1,
    }
}
```

**Integration Points**:
- Domain-specific model weights
- Phase 4 routing optimizer integration
- Domain performance metrics

---

#### 8.3: Enhance Cross-System Extractor (`cross_system_extractor.go`)

**Current State**: Generic cross-system extraction
**Integration**: Domain-normalized extraction

**Changes Needed**:

```go
// Add domain manager to CrossSystemExtractor
type CrossSystemExtractor struct {
    logger        *log.Logger
    domainManager *domain.DomainManager  // NEW
    systemAdapters map[string]SystemAdapter
}

// Enhance ExtractCrossSystem with domain normalization
func (cse *CrossSystemExtractor) ExtractCrossSystem(
    ctx context.Context,
    sourceSystem string,
    targetSystem string,
    domainID string,  // NEW: Add domain ID
) (*CrossSystemPattern, error) {
    // ... existing extraction ...
    
    // NEW: Normalize using domain configuration
    if cse.domainManager != nil && domainID != "" {
        domainConfig, exists := cse.domainManager.GetDomainConfig(domainID)
        if exists {
            // Apply domain-specific normalization
            normalized = cse.normalizeWithDomainConfig(
                pattern, domainConfig,
            )
            
            // Use domain keywords for pattern matching
            domainKeywords := domainConfig.Keywords
            matched = cse.matchPatternsWithKeywords(
                sourcePattern, targetPattern, domainKeywords,
            )
        }
    }
    
    return normalized, nil
}
```

**Integration Points**:
- Domain configuration for normalization
- Domain keywords for pattern matching
- Domain-specific system templates

---

#### 8.4: Enhance Pattern Transfer (`pattern_transfer.py`)

**Current State**: Generic pattern transfer
**Integration**: Domain-aware transfer with similarity

**Changes Needed**:

```python
# Add domain manager integration
class DomainAwarePatternTransfer:
    """Domain-aware pattern transfer."""
    
    def __init__(self, domain_manager=None):
        self.domain_manager = domain_manager
        # Use Phase 4 cross-domain learning
        from .routing_optimizer import RoutingOptimizer
        self.routing_optimizer = RoutingOptimizer()
    
    def transfer_patterns(
        self,
        source_domain: str,
        target_domain: str,
        patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transfer patterns with domain awareness."""
        # Use Phase 4 cross-domain learning for similarity
        similar_domains = self.routing_optimizer.find_similar_domains(
            source_domain
        )
        
        # Check if domains are similar
        is_similar = any(
            d["domain_id"] == target_domain 
            for d in similar_domains
            if d.get("similarity", 0) > 0.7
        )
        
        if not is_similar:
            return {
                "transferred": False,
                "reason": "domains_not_similar",
                "similarity": 0.0
            }
        
        # Get domain configurations
        source_config = self.domain_manager.get_domain_config(source_domain)
        target_config = self.domain_manager.get_domain_config(target_domain)
        
        # Adapt patterns
        adapted = self._adapt_patterns(
            patterns, source_config, target_config
        )
        
        return {
            "transferred": True,
            "patterns": adapted,
            "similarity": next(
                d["similarity"] for d in similar_domains
                if d["domain_id"] == target_domain
            )
        }
```

**Integration Points**:
- Phase 4 cross-domain learning for similarity
- Domain manager for config access
- Domain-aware pattern adaptation

---

### Phase 9: Domain-Aware Automation

#### 9.1: Enhance Auto-Tuner (`auto_tuner.py`)

**Current State**: Generic auto-tuning
**Integration**: Domain-specific hyperparameter optimization

**Changes Needed**:

```python
# Add domain awareness to AutoTuner
class DomainAwareAutoTuner(AutoTuner):
    """Domain-aware auto-tuning."""
    
    def __init__(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        n_trials: int = 50,
        domain_manager=None  # NEW
    ):
        super().__init__(study_name, storage, n_trials)
        self.domain_manager = domain_manager
        self.domain_studies = {}  # domain_id -> Optuna study
    
    def optimize_for_domain(
        self,
        domain_id: str,
        objective_func,
        training_data_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific domain."""
        # Create or get domain-specific study
        if domain_id not in self.domain_studies:
            study_name = f"{self.study_name}_{domain_id}"
            self.domain_studies[domain_id] = optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                direction="maximize",
                load_if_exists=True
            )
        
        study = self.domain_studies[domain_id]
        
        # Get domain configuration for constraints
        domain_config = None
        if self.domain_manager:
            domain_config = self.domain_manager.get_domain_config(domain_id)
        
        # Optimize with domain constraints
        def domain_objective(trial):
            # Suggest hyperparameters with domain constraints
            params = self.suggest_hyperparameters_with_domain(
                trial, domain_config, training_data_stats
            )
            return objective_func(params)
        
        study.optimize(domain_objective, n_trials=self.n_trials)
        
        return {
            "domain_id": domain_id,
            "best_hyperparameters": study.best_params,
            "best_score": study.best_value,
        }
```

**Integration Points**:
- Domain-specific Optuna studies
- Domain configuration constraints
- Domain-specific architecture selection

---

#### 9.2: Enhance Self-Healing (`self_healing.go`)

**Current State**: Generic self-healing
**Integration**: Domain-aware health monitoring

**Changes Needed**:

```go
// Add domain health monitoring to SelfHealingSystem
type SelfHealingSystem struct {
    logger           *log.Logger
    retryConfig      *RetryConfig
    circuitBreakers  map[string]*CircuitBreaker
    fallbackHandlers map[string]FallbackHandler
    healthMonitors   map[string]*HealthMonitor
    domainManager    *domain.DomainManager  // NEW
    domainHealthMonitor *DomainHealthMonitor  // NEW: Phase 4
    mu               sync.RWMutex
}

// Enhance ExecuteWithCircuitBreaker with domain health
func (shs *SelfHealingSystem) ExecuteWithCircuitBreaker(
    ctx context.Context,
    serviceName string,
    operation func() (any, error),
    domainID string,  // NEW: Add domain ID
) (any, error) {
    // ... existing circuit breaker logic ...
    
    // NEW: Check domain health
    if shs.domainHealthMonitor != nil && domainID != "" {
        health := shs.domainHealthMonitor.GetDomainHealth(domainID)
        
        if health.Score < 0.5 {
            // Domain unhealthy, use fallback immediately
            if fallback, exists := shs.fallbackHandlers[serviceName]; exists {
                shs.logger.Printf(
                    "Domain %s health low (%.2f), using fallback",
                    domainID, health.Score,
                )
                return fallback(ctx, fmt.Errorf("domain_unhealthy"))
            }
        }
    }
    
    // ... rest of circuit breaker logic ...
}
```

**Integration Points**:
- Phase 4 domain health monitoring
- Domain-aware circuit breakers
- Domain-specific fallback handlers

---

#### 9.3: Enhance Auto-Pipeline (`auto_pipeline.go`)

**Current State**: Generic auto-pipeline
**Integration**: Domain-orchestrated automation

**Changes Needed**:

```go
// Add domain awareness to AutoPipelineOrchestrator
type AutoPipelineOrchestrator struct {
    logger            *log.Logger
    extractServiceURL string
    trainingServiceURL string
    gleanClient      *GleanClient
    modelRegistry    *ModelRegistry
    abTestManager    *ABTestManager
    domainManager    *domain.DomainManager  // NEW
    domainTrainer    *DomainTrainer  // NEW: Phase 2
}

// Enhance TriggerTrainingOnNewData with domain awareness
func (apo *AutoPipelineOrchestrator) TriggerTrainingOnNewData(
    ctx context.Context,
    projectID string,
    systemID string,
    domainID string,  // NEW: Add domain ID
) error {
    // ... existing logic ...
    
    // NEW: Use domain-specific training
    if apo.domainTrainer != nil && domainID != "" {
        // Check if domain training is enabled
        domainConfig, exists := apo.domainManager.GetDomainConfig(domainID)
        if exists {
            // Run domain-specific training
            trainingResult, err := apo.domainTrainer.TrainDomainModel(
                domainID, trainingDataPath, fine_tune=True,
            )
            
            if err == nil && trainingResult.ShouldDeploy {
                // Deploy domain model
                return apo.deployDomainModel(ctx, domainID, trainingResult)
            }
        }
    }
    
    // Fallback to generic training
    return apo.runTrainingPipeline(ctx, projectID, systemID)
}
```

**Integration Points**:
- Phase 2 domain trainer
- Domain-specific A/B testing
- Domain-aware deployment

---

#### 9.4: Enhance Predictive Analytics (`predictive_analytics.go`)

**Current State**: Generic predictive analytics
**Integration**: Domain-intelligent predictions

**Changes Needed**:

```go
// Add domain awareness to PredictiveAnalytics
type PredictiveAnalytics struct {
    logger        *log.Logger
    domainManager *domain.DomainManager  // NEW
    domainMetricsCollector *DomainMetricsCollector  // NEW: Phase 2
}

// Enhance PredictIssues with domain context
func (pa *PredictiveAnalytics) PredictIssues(
    ctx context.Context,
    domainID string,  // NEW: Add domain ID
    timeHorizonDays int,
) (*Predictions, error) {
    // ... existing prediction logic ...
    
    // NEW: Use domain metrics for predictions
    if pa.domainMetricsCollector != nil {
        domainMetrics := pa.domainMetricsCollector.CollectDomainMetrics(
            domainID, time_window_days=30,
        )
        
        // Predict with domain context
        predictions.DataQualityIssues = pa.predictDataQualityWithDomain(
            domainMetrics, domainID,
        )
        
        predictions.PerformanceDegradation = pa.predictPerformanceWithDomain(
            domainMetrics, domainID,
        )
    }
    
    // Get domain configuration for recommendations
    if pa.domainManager != nil {
        domainConfig, _ := pa.domainManager.GetDomainConfig(domainID)
        if domainConfig {
            predictions.Recommendations = pa.generateDomainRecommendations(
                predictions, domainConfig,
            )
        }
    }
    
    return predictions, nil
}
```

**Integration Points**:
- Phase 2 domain metrics collector
- Domain-specific predictions
- Domain-aware recommendations

---

## Implementation Strategy

### Step 1: Add Domain Manager Integration

**Pattern**: Add `domain_manager` parameter to all constructors

```python
# Python pattern
def __init__(self, ..., domain_manager=None):
    self.domain_manager = domain_manager

# Go pattern
type Component struct {
    // ... existing fields ...
    domainManager *domain.DomainManager
}
```

### Step 2: Add Domain ID Parameters

**Pattern**: Add `domain_id` parameter to key methods

```python
def method(self, ..., domain_id: Optional[str] = None):
    if domain_id and self.domain_manager:
        domain_config = self.domain_manager.get_domain_config(domain_id)
        # Use domain config
```

### Step 3: Integrate Phase 4 Features

- Use `RoutingOptimizer` for domain similarity
- Use `DomainMetricsCollector` for domain metrics
- Use `DomainHealthMonitor` for health checks
- Use `DomainTrainer` for domain-specific training

### Step 4: SAP RPT Integration

- Use SAP RPT embeddings with domain keywords
- Domain-specific SAP RPT classification
- SAP RPT as ensemble component (not sole focus)

---

## Files to Modify (Not Create)

### Python Files
1. `services/training/pattern_learning_gnn.py` - Add domain awareness
2. `services/training/meta_pattern_learner.py` - Enhance domain integration
3. `services/training/sequence_pattern_transformer.py` - Add domain conditioning
4. `services/training/active_pattern_learner.py` - Add domain filtering
5. `services/training/pattern_transfer.py` - Add domain similarity
6. `services/training/auto_tuner.py` - Add domain-specific optimization

### Go Files
1. `services/extract/semantic_schema_analyzer.go` - Integrate domain manager
2. `services/extract/model_fusion.go` - Add domain-optimized weights
3. `services/extract/cross_system_extractor.go` - Add domain normalization
4. `services/extract/workflow_converter_advanced.go` - Add domain routing
5. `services/extract/self_healing.go` - Add domain health monitoring
6. `services/orchestration/auto_pipeline.go` - Add domain orchestration
7. `services/orchestration/agent_coordinator.go` - Add domain-aware coordination
8. `services/analytics/predictive_analytics.go` - Add domain predictions
9. `services/analytics/recommendation_engine.go` - Add domain recommendations

---

## Integration Checklist

### Phase 7
- [ ] Add domain manager to GNN learner
- [ ] Enhance meta-pattern learner domain integration
- [ ] Add domain conditioning to sequence transformer
- [ ] Add domain filtering to active learner

### Phase 8
- [ ] Integrate domain manager in semantic analyzer
- [ ] Add domain-optimized weights to model fusion
- [ ] Add domain normalization to cross-system extractor
- [ ] Enhance pattern transfer with domain similarity

### Phase 9
- [ ] Add domain-specific optimization to auto-tuner
- [ ] Add domain health monitoring to self-healing
- [ ] Add domain orchestration to auto-pipeline
- [ ] Add domain predictions to analytics

### SAP RPT Integration
- [ ] Use SAP RPT embeddings with domain keywords
- [ ] Domain-specific SAP RPT classification
- [ ] SAP RPT in domain-optimized ensemble

---

## Success Metrics

### Integration Completeness
- ✅ All Phase 7-9 components have domain manager integration
- ✅ Domain ID parameters added to key methods
- ✅ Phase 4 features integrated (cross-domain, health, metrics)
- ✅ SAP RPT used where appropriate (not exclusively)

### Performance Improvements
- Domain-specific pattern learning: +10-15% accuracy
- Domain-optimized model fusion: +5-8% ensemble accuracy
- Domain-aware automation: -80% manual intervention
- Domain health monitoring: +15% reliability

---

**Document Version**: 2.0  
**Updated**: 2025-11-04  
**Status**: Adjusted Based on Actual Code Review  
**Approach**: Enhance Existing Code (Not Duplicate)

