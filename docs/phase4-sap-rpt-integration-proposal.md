# Phase 4: SAP RPT OSS Integration with Domain Intelligence

## Overview

This document outlines how Phase 4 features (Advanced Statistical Analysis, Cross-Domain Learning, Domain Health Scoring, Model Ensemble, Configuration Templates) will integrate with the existing SAP RPT OSS extraction capabilities.

## Current SAP RPT OSS Integration

### Existing Capabilities ✅

1. **Semantic Embeddings** (`services/extract/scripts/embed_sap_rpt.py`)
   - ZMQ-based sentence embedding server
   - Table and column embeddings (384-dim)
   - Uses `sentence-transformers/all-MiniLM-L6-v2`
   - Stored in vector stores with `embedding_type: "sap_rpt_semantic"`

2. **Table Classification** (`services/extract/scripts/classify_table_sap_rpt.py`)
   - ML-based classification (transaction vs reference)
   - Feature extraction for tabular data
   - Falls back to pattern matching

3. **Advanced Multi-Task Learning** (`services/extract/scripts/sap_rpt_advanced.py`)
   - Classification + Regression tasks
   - Training data integration
   - Quality scoring

4. **Integration Points**:
   - Extract service: Dual embedding generation (RelationalTransformer + sap-rpt-1-oss)
   - Training service: Semantic embeddings for training data
   - Search: Hybrid search with semantic embeddings

### Current Limitations ⚠️

- **No Domain Association**: SAP RPT embeddings not linked to domains
- **No Statistical Analysis**: No advanced analysis of SAP RPT performance
- **No Cross-Domain Learning**: Can't learn patterns across domains
- **No Health Scoring**: No metrics for SAP RPT extraction quality
- **No Ensemble**: Not used in model selection/ensemble
- **No Templates**: No reusable SAP RPT configurations per domain

## Phase 4 Integration Strategy

### 1. Advanced Statistical Analysis for SAP RPT Extraction ✅

**Goal**: Analyze SAP RPT extraction quality and performance statistically

**Integration Points**:
- **Extraction Metrics**: Track SAP RPT classification accuracy, embedding quality
- **Domain-Specific Analysis**: Per-domain SAP RPT performance statistics
- **Statistical Tests**: Compare SAP RPT vs RelationalTransformer embeddings
- **A/B Testing**: Test SAP RPT configurations across domains

**Implementation**:
```python
# services/training/sap_rpt_statistics.py
class SAPRPTStatisticalAnalyzer:
    """Statistical analysis for SAP RPT extraction."""
    
    def analyze_extraction_quality(
        self,
        domain_id: str,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Analyze SAP RPT extraction quality for a domain."""
        # Query extraction metrics
        metrics = self._get_extraction_metrics(domain_id, time_range)
        
        # Statistical analysis
        analysis = {
            "classification_accuracy": self._calculate_classification_accuracy(metrics),
            "embedding_quality": self._analyze_embedding_quality(metrics),
            "sap_rpt_vs_relational": self._compare_embeddings(metrics),
            "statistical_significance": self._test_significance(metrics),
        }
        
        return analysis
    
    def compare_extraction_methods(
        self,
        domain_id: str,
        method_a: str,  # "sap_rpt_semantic"
        method_b: str   # "relational_transformer"
    ) -> Dict[str, Any]:
        """Compare extraction methods statistically."""
        # T-test for embedding quality
        # Chi-square for classification accuracy
        # Effect size calculation
        pass
```

**Database Schema**:
```sql
CREATE TABLE sap_rpt_extraction_metrics (
    id SERIAL PRIMARY KEY,
    domain_id VARCHAR(255) NOT NULL,
    extraction_id VARCHAR(255) NOT NULL,
    classification_accuracy FLOAT,
    embedding_quality_score FLOAT,
    method VARCHAR(50),  -- 'sap_rpt_semantic' or 'relational_transformer'
    table_name VARCHAR(255),
    columns_count INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_sap_rpt_metrics_domain_time 
ON sap_rpt_extraction_metrics(domain_id, created_at);
```

### 2. Cross-Domain Learning for SAP RPT ✅

**Goal**: Learn extraction patterns across similar domains using SAP RPT

**Integration Points**:
- **Domain Similarity**: Use SAP RPT embeddings to find similar domains
- **Knowledge Transfer**: Transfer SAP RPT extraction patterns between domains
- **Shared Embeddings**: Share learned embeddings across domains
- **Pattern Recognition**: Identify common table structures across domains

**Implementation**:
```python
# services/training/sap_rpt_cross_domain.py
class SAPRPTCrossDomainLearner:
    """Cross-domain learning for SAP RPT extraction."""
    
    def find_similar_domains(
        self,
        domain_id: str,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find domains with similar table structures using SAP RPT embeddings."""
        # Get domain's SAP RPT embeddings
        domain_embeddings = self._get_domain_embeddings(domain_id)
        
        # Find similar domains
        similar = []
        for other_domain in self._get_all_domains():
            if other_domain == domain_id:
                continue
            
            other_embeddings = self._get_domain_embeddings(other_domain)
            similarity = self._cosine_similarity(domain_embeddings, other_embeddings)
            
            if similarity >= similarity_threshold:
                similar.append({
                    "domain_id": other_domain,
                    "similarity": similarity,
                    "shared_patterns": self._identify_shared_patterns(
                        domain_id, other_domain
                    )
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)
    
    def transfer_extraction_patterns(
        self,
        source_domain: str,
        target_domain: str
    ) -> Dict[str, Any]:
        """Transfer SAP RPT extraction patterns between domains."""
        # Get source domain patterns
        source_patterns = self._extract_patterns(source_domain)
        
        # Adapt to target domain
        adapted_patterns = self._adapt_patterns(source_patterns, target_domain)
        
        # Apply to target domain
        result = self._apply_patterns(target_domain, adapted_patterns)
        
        return {
            "transferred_patterns": len(adapted_patterns),
            "success_rate": result["success_rate"],
            "improvement": result["improvement"]
        }
```

**Integration with Domain System**:
- Extend `DomainFilter` to use SAP RPT embeddings for domain similarity
- Use SAP RPT embeddings in cross-domain knowledge transfer
- Store cross-domain patterns in PostgreSQL

### 3. Domain Health Scoring for SAP RPT Extraction ✅

**Goal**: Score domain health based on SAP RPT extraction quality

**Integration Points**:
- **Extraction Quality**: SAP RPT classification accuracy, embedding quality
- **SLA Tracking**: Extraction latency, success rate
- **Health Components**: Classification accuracy, embedding quality, extraction speed
- **Domain Ranking**: Rank domains by SAP RPT extraction health

**Implementation**:
```python
# services/training/sap_rpt_health.py
class SAPRPTHealthMonitor:
    """Health monitoring for SAP RPT extraction."""
    
    def calculate_extraction_health_score(
        self,
        domain_id: str
    ) -> Dict[str, Any]:
        """Calculate health score for SAP RPT extraction."""
        metrics = self._get_extraction_metrics(domain_id, time_window_days=7)
        
        # Component scores
        classification_score = self._score_classification(
            metrics.get("classification_accuracy", 0)
        )
        
        embedding_score = self._score_embeddings(
            metrics.get("embedding_quality", 0)
        )
        
        speed_score = self._score_speed(
            metrics.get("avg_extraction_time_ms", 0)
        )
        
        # Composite score
        health_score = (
            classification_score * 0.4 +
            embedding_score * 0.4 +
            speed_score * 0.2
        )
        
        return {
            "domain_id": domain_id,
            "health_score": health_score,
            "component_scores": {
                "classification": classification_score,
                "embedding_quality": embedding_score,
                "speed": speed_score,
            },
            "sla_compliance": self._check_sla_compliance(domain_id),
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def check_sla_compliance(
        self,
        domain_id: str,
        sla_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check SLA compliance for SAP RPT extraction."""
        metrics = self._get_recent_metrics(domain_id)
        
        compliance = {
            "classification_accuracy": {
                "target": sla_config.get("min_classification_accuracy", 0.9),
                "actual": metrics.get("classification_accuracy", 0),
                "compliant": metrics.get("classification_accuracy", 0) >= 
                           sla_config.get("min_classification_accuracy", 0.9)
            },
            "extraction_latency": {
                "target": sla_config.get("max_latency_ms", 1000),
                "actual": metrics.get("avg_extraction_time_ms", 0),
                "compliant": metrics.get("avg_extraction_time_ms", 0) <= 
                           sla_config.get("max_latency_ms", 1000)
            },
            "success_rate": {
                "target": sla_config.get("min_success_rate", 0.95),
                "actual": metrics.get("success_rate", 0),
                "compliant": metrics.get("success_rate", 0) >= 
                           sla_config.get("min_success_rate", 0.95)
            }
        }
        
        return compliance
```

**Database Schema**:
```sql
CREATE TABLE sap_rpt_health_scores (
    domain_id VARCHAR(255) PRIMARY KEY,
    health_score FLOAT NOT NULL,
    classification_score FLOAT,
    embedding_score FLOAT,
    speed_score FLOAT,
    sla_compliance JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 4. Intelligent Model Selection & Ensemble for SAP RPT ✅

**Goal**: Automatically select best extraction method or ensemble multiple methods

**Integration Points**:
- **Method Selection**: Choose SAP RPT vs RelationalTransformer based on query
- **Ensemble Embeddings**: Combine SAP RPT and RelationalTransformer embeddings
- **Confidence Weighting**: Weight embeddings based on classification confidence
- **Query-Specific Routing**: Route to best method based on table characteristics

**Implementation**:
```python
# services/training/sap_rpt_ensemble.py
class SAPRPTEnsemble:
    """Ensemble selection for SAP RPT extraction."""
    
    def select_extraction_method(
        self,
        table_name: str,
        columns: List[Dict[str, Any]],
        domain_id: str
    ) -> Dict[str, Any]:
        """Select best extraction method for a table."""
        # Analyze table characteristics
        characteristics = self._analyze_table(table_name, columns)
        
        # Get domain preferences
        domain_config = self._get_domain_config(domain_id)
        
        # Select method
        if characteristics["semantic_richness"] > 0.7:
            method = "sap_rpt_semantic"
            confidence = 0.9
        elif characteristics["relational_complexity"] > 0.7:
            method = "relational_transformer"
            confidence = 0.9
        else:
            method = "ensemble"  # Use both
            confidence = 0.85
        
        return {
            "method": method,
            "confidence": confidence,
            "reasoning": self._generate_reasoning(characteristics, domain_config)
        }
    
    def ensemble_embeddings(
        self,
        sap_rpt_embedding: List[float],
        relational_embedding: List[float],
        weights: Tuple[float, float] = (0.6, 0.4)
    ) -> List[float]:
        """Ensemble SAP RPT and RelationalTransformer embeddings."""
        # Weighted combination
        sap_weight, rel_weight = weights
        
        # Normalize dimensions (SAP RPT: 384, Relational: 768)
        sap_normalized = self._normalize_embedding(sap_rpt_embedding, target_dim=768)
        
        # Combine
        ensemble = [
            sap_weight * sap_normalized[i] + rel_weight * relational_embedding[i]
            for i in range(len(relational_embedding))
        ]
        
        return ensemble
```

**Integration with Routing**:
- Extend `RoutingOptimizer` to consider SAP RPT extraction quality
- Use SAP RPT classification confidence in routing decisions
- Ensemble embeddings for better search quality

### 5. Domain Configuration Templates for SAP RPT ✅

**Goal**: Reusable SAP RPT configurations per domain type

**Integration Points**:
- **Template Library**: SAP RPT configurations for common domain types
- **Auto-Configuration**: Auto-configure SAP RPT based on domain characteristics
- **Best Practices**: Templates include best practices for SAP RPT usage
- **Version Control**: Version-controlled templates

**Implementation**:
```python
# services/localai/pkg/domain/sap_rpt_templates.go
type SAPRPTTemplate struct {
    Name        string
    Description string
    DomainType  string  // "transactional", "reference", "analytical"
    Config      SAPRPTConfig
    Variables   []string
}

type SAPRPTConfig struct {
    UseSemanticEmbeddings bool
    UseClassification     bool
    UseMultiTaskLearning  bool
    EmbeddingDimension    int
    ClassificationModel   string
    TrainingDataPath      string
    ZMQPort              int
    BatchSize            int
    CacheEnabled         bool
}

func CreateDomainWithSAPRPTTemplate(
    templateID string,
    domainID string,
    variables map[string]string
) (*DomainConfig, error) {
    // Create domain with SAP RPT template
}
```

**Template Examples**:
```json
{
  "templates": {
    "transactional_table": {
      "name": "Transactional Table Domain",
      "sap_rpt_config": {
        "use_semantic_embeddings": true,
        "use_classification": true,
        "classification_model": "transactional",
        "batch_size": 10,
        "cache_enabled": true
      }
    },
    "reference_table": {
      "name": "Reference Table Domain",
      "sap_rpt_config": {
        "use_semantic_embeddings": true,
        "use_classification": true,
        "classification_model": "reference",
        "batch_size": 20,
        "cache_enabled": true
      }
    }
  }
}
```

## Integration Architecture

### Data Flow
```
Extraction Request
    ↓
Domain Detection (Phase 1)
    ↓
SAP RPT Configuration Selection (Phase 4 Templates)
    ↓
Method Selection (Phase 4 Ensemble)
    ↓
SAP RPT Extraction (Current)
    ↓
Metrics Collection (Phase 4 Statistics)
    ↓
Health Scoring (Phase 4 Health)
    ↓
Cross-Domain Learning (Phase 4 Cross-Domain)
    ↓
Domain Config Update
```

### Component Integration

1. **Extract Service**:
   - Use SAP RPT templates for configuration
   - Use ensemble for method selection
   - Collect metrics for statistics

2. **Training Service**:
   - Statistical analysis of SAP RPT performance
   - Cross-domain learning with SAP RPT embeddings
   - Health scoring integration

3. **LocalAI Service**:
   - SAP RPT templates in domain configs
   - Template-based domain creation

## Database Schema Extensions

```sql
-- SAP RPT extraction metrics
CREATE TABLE sap_rpt_extraction_metrics (
    id SERIAL PRIMARY KEY,
    domain_id VARCHAR(255) NOT NULL,
    extraction_id VARCHAR(255) NOT NULL,
    classification_accuracy FLOAT,
    embedding_quality_score FLOAT,
    method VARCHAR(50),
    table_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- SAP RPT health scores
CREATE TABLE sap_rpt_health_scores (
    domain_id VARCHAR(255) PRIMARY KEY,
    health_score FLOAT NOT NULL,
    component_scores JSONB,
    sla_compliance JSONB,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- SAP RPT templates
CREATE TABLE sap_rpt_templates (
    template_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    domain_type VARCHAR(100),
    config JSONB NOT NULL,
    variables JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Cross-domain SAP RPT patterns
CREATE TABLE sap_rpt_cross_domain_patterns (
    id SERIAL PRIMARY KEY,
    source_domain VARCHAR(255) NOT NULL,
    target_domain VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100),
    pattern_data JSONB,
    similarity FLOAT,
    transferred_at TIMESTAMP DEFAULT NOW()
);
```

## Implementation Priority

### Phase 4.1 (High Priority)
1. **Statistical Analysis** (1 week)
   - Extract metrics collection
   - Statistical tests
   - A/B testing integration

2. **Cross-Domain Learning** (1 week)
   - Domain similarity detection
   - Pattern transfer
   - Knowledge sharing

3. **Health Scoring** (1 week)
   - Health score calculation
   - SLA tracking
   - Alerting

### Phase 4.2 (Medium Priority)
4. **Model Ensemble** (1-2 weeks)
   - Method selection
   - Embedding ensemble
   - Query routing

5. **Configuration Templates** (1 week)
   - Template library
   - Auto-configuration
   - Template management

## Success Metrics

- ✅ SAP RPT extraction quality tracked statistically
- ✅ Cross-domain learning reduces extraction time by 30%
- ✅ Health scores > 90 for all domains using SAP RPT
- ✅ Ensemble embeddings improve search quality by 10%
- ✅ Template-based domain creation reduces setup time by 70%

---

**Document Version**: 1.0  
**Created**: 2025-11-04  
**Status**: Integration Proposal  
**Phase**: Phase 4 with SAP RPT OSS Integration

