# Codebase Understanding Summary

## Overview

This document provides comprehensive answers to questions about the aModels codebase, covering SAP BDC integration, training pipeline, unimplemented agents, and ETL/lineage mapping status.

---

## 1. SAP BDC Integration (`services/extract/sap_bdc_integration.go`)

### How It Works

The SAP BDC integration extracts data from SAP Business Data Cloud and converts it to a knowledge graph format:

1. **HTTP Client**: Makes POST requests to SAP BDC service (default: `http://localhost:8083`)
2. **Request**: Sends formation_id, source_system, and optional filters
3. **Response**: Receives schema data (tables, views, columns, foreign keys)
4. **Conversion**: Transforms schema into graph nodes and edges

### Current Architecture

- **Nodes Created**: database, table, view, column
- **Edges Created**: CONTAINS, HAS_COLUMN, REFERENCES
- **Metadata**: Propagates SAP metadata to graph nodes

### Areas for Contribution

1. **Incremental Extraction**: Add change detection and timestamp-based filtering
2. **View Dependency Parsing**: Parse SQL to extract underlying table dependencies
3. **Schema Versioning**: Track schema changes over time
4. **Metadata Enrichment**: Integrate with SAP Business Glossary
5. **Performance Optimization**: Add batching and parallel processing

**See**: [SAP BDC Integration Guide](./SAP_BDC_INTEGRATION.md) for detailed documentation

---

## 2. Training Pipeline (`services/training/pipeline.py`)

### Current Implementation

The training pipeline orchestrates end-to-end ML training:

1. **Extract Knowledge Graph**: Calls extract service
2. **Glean Integration**: Queries historical patterns (optional)
3. **Pattern Learning**: Learns patterns from graph structure
4. **Temporal Analysis**: Analyzes schema evolution (optional)
5. **Feature Generation**: Creates training features
6. **Domain Filtering**: Applies domain-specific filtering (optional)
7. **Dataset Preparation**: Saves features to files

### GNN Integration Opportunities

The codebase structure is well-suited for Graph Neural Network integration. Five key use cases:

#### 1. Node Classification
- **Purpose**: Classify nodes (tables, columns) by type, domain, quality
- **Integration**: After Step 1 (Extract), before Step 3 (Pattern Learning)
- **Value**: Automatic domain detection, better schema organization

#### 2. Link Prediction
- **Purpose**: Predict missing relationships or suggest new mappings
- **Integration**: Step 3 (Pattern Learning) - enhance relationship discovery
- **Value**: Automatic mapping discovery, reduced manual effort

#### 3. Graph Embeddings
- **Purpose**: Generate embeddings for similarity search and pattern matching
- **Integration**: Step 4 (Feature Generation) - replace manual features
- **Value**: Better feature representation, semantic similarity matching

#### 4. Anomaly Detection
- **Purpose**: Detect structural anomalies in graph patterns
- **Integration**: New step after Step 3 (Pattern Learning)
- **Value**: Automatic anomaly detection, structural quality assessment

#### 5. Schema Matching
- **Purpose**: Use GNNs for cross-system schema alignment
- **Integration**: New service for cross-system mapping
- **Value**: Automated schema alignment, reduced manual mapping work

**See**: [Training Pipeline GNN Opportunities](./TRAINING_PIPELINE_GNN_OPPORTUNITIES.md) for detailed implementation guidance

---

## 3. Unimplemented Agents

### Implemented Agents ✅

1. **Data Ingestion Agent**: Autonomous data ingestion from source systems
2. **Mapping Rule Agent**: Automatic mapping rule learning and updates
3. **Anomaly Detection Agent**: Automatic anomaly detection in data streams
4. **Test Generation Agent**: Generates and runs test scenarios
5. **Test Scenario Generator**: Creates test scenarios from schemas

### Planned but Not Implemented Agents ❌

1. **Schema Evolution Agent**
   - **Purpose**: Track and analyze schema changes over time
   - **Value**: Critical for understanding schema evolution and predicting impacts
   - **Integration**: Would use `TemporalPatternLearner` from training pipeline

2. **Quality Monitoring Agent**
   - **Purpose**: Continuous data quality monitoring and alerting
   - **Value**: Essential for maintaining data quality across systems
   - **Integration**: Would use quality SLOs from `config/quality-slos.yaml`

3. **Lineage Discovery Agent**
   - **Purpose**: Automatically discover and map data lineage
   - **Value**: Critical for understanding data flow and impact analysis
   - **Integration**: Would use `CrossSystemExtractor` and lineage mappings

4. **ETL Orchestration Agent**
   - **Purpose**: Automatically configure and manage ETL pipelines
   - **Value**: Essential for automating ETL operations
   - **Integration**: Would read pipeline configs from `config/pipelines/*.yaml`

5. **Cross-System Mapping Agent**
   - **Purpose**: Automatically learn and apply cross-system mappings
   - **Value**: Critical for cross-system integration and automation
   - **Integration**: Would use mapping rule agent and schema matching (GNN)

**See**: [Agents Status](./AGENTS_STATUS.md) for detailed documentation

---

## 4. ETL Pipelines and Cross-System Lineage Mappings

### The Discrepancy Explained

**What the Documentation Said**: "Not yet implemented"

**What Actually Exists**:
- ✅ ETL pipeline definitions (`config/pipelines/*.yaml`) - **6 pipelines created**
- ✅ Schema mappings with field-level mappings (`config/mappings/*.yaml`) - **8+ mappings created**
- ✅ Lineage mappings (`config/lineage-mappings.yaml`) - **Complete flow defined**

**What's Actually Missing**:
- ❌ **Runtime execution engine** to execute ETL pipeline definitions
- ❌ **ETL orchestration agent** to manage ETL jobs
- ❌ **Automatic lineage graph creation** from mapping configs
- ❌ **Field-level lineage tracking service** (entity-level exists, but field-level not automated)

### Clarification

**Configuration Files**: ✅ **Created and Complete**
- All ETL pipeline definitions exist
- All schema mappings with field-level details exist
- Complete lineage flow documented

**Runtime Execution**: ❌ **Not Implemented**
- No service reads YAML files and creates ETL jobs
- No service creates lineage edges in knowledge graph from mappings
- No agent orchestrates ETL operations

**Field-Level Lineage**:
- Field-level **schema mappings** exist (how fields map between systems)
- Entity-level **lineage mappings** exist (which entities connect)
- But **field-level lineage edges** are not automatically created in the knowledge graph

### Updated Documentation

The `config/README.md` has been updated to clarify:
- ✅ Configuration files exist and are complete
- ❌ Runtime execution engines are not implemented
- ❌ Automatic lineage graph creation is not implemented

**See**: [Config README](../config/README.md) for updated status

---

## Summary of Contributions Opportunities

### High-Value Contributions

1. **GNN Integration for Training Pipeline**
   - Start with graph embeddings (foundation)
   - Add node classification
   - Implement link prediction
   - Build schema matcher
   - Add anomaly detection

2. **SAP BDC Integration Enhancements**
   - Incremental extraction support
   - View dependency parsing
   - Schema versioning
   - Metadata enrichment

3. **Unimplemented Agents**
   - ETL Orchestration Agent (highest priority)
   - Lineage Discovery Agent (high priority)
   - Quality Monitoring Agent (high priority)
   - Cross-System Mapping Agent (medium priority)
   - Schema Evolution Agent (medium priority)

4. **Runtime Execution Engines**
   - ETL pipeline execution engine
   - Lineage graph creation service
   - Field-level lineage tracking

---

## Related Documentation

- [SAP BDC Integration Guide](./SAP_BDC_INTEGRATION.md)
- [Training Pipeline GNN Opportunities](./TRAINING_PIPELINE_GNN_OPPORTUNITIES.md)
- [Agents Status](./AGENTS_STATUS.md)
- [Config README](../config/README.md)
- [Orchestration Service Integration](../services/orchestration/INTEGRATION.md)

---

## Next Steps

1. **Review Documentation**: Read the detailed guides for each area
2. **Choose Contribution Area**: Select based on your expertise and interests
3. **Understand Integration Points**: Review how components connect
4. **Start Small**: Begin with foundational work (e.g., graph embeddings)
5. **Iterate**: Build incrementally, test, and integrate

---

## Questions?

For specific questions about:
- **SAP BDC**: See [SAP BDC Integration Guide](./SAP_BDC_INTEGRATION.md)
- **GNN Integration**: See [Training Pipeline GNN Opportunities](./TRAINING_PIPELINE_GNN_OPPORTUNITIES.md)
- **Agents**: See [Agents Status](./AGENTS_STATUS.md)
- **ETL/Lineage**: See [Config README](../config/README.md)

