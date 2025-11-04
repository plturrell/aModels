# SGMI Extraction Quality & Information Metrics

## Overview

This document summarizes the quality and information metrics for the SGMI data extraction process.

## Quick Metrics Report

Run the quality metrics script:
```bash
./scripts/run_quality_metrics.sh
```

## Metrics Explained

### 1. Data Completeness

- **Total Nodes**: 31,979
- **Total Edges**: 29,010
- **Node-to-Edge Ratio**: ~1.10

**Interpretation**: 
- Every node has approximately 1 edge on average
- This indicates good graph connectivity
- Ratio > 1 suggests most nodes are connected

### 2. Data Quality

- **Missing Labels**: 0 ❌
- **Missing IDs**: 0 ❌
- **Missing Properties**: ~8,232 ⚠️
- **Orphan Edges**: 0 ✅

**Interpretation**:
- All nodes have required identifiers (labels, IDs)
- Some nodes (columns) may have minimal properties (empty JSON)
- No broken relationships (orphan edges)

### 3. Information Metrics

#### Column Type Entropy: ~1.92 bits

**What it means**:
- Measures diversity of column data types
- Higher entropy = more diverse types
- 1.92 bits indicates moderate diversity
- Max possible: ~3.3 bits (if all types equally distributed)

**Calculation**:
```
Entropy = -Σ(p(x) * log2(p(x)))
where p(x) is the probability of each column type
```

#### KL Divergence: ~0.42 bits

**What it means**:
- Measures how different the actual distribution is from ideal
- Lower = closer to ideal distribution
- 0.42 bits indicates moderate deviation
- 0.0 = perfect match to ideal

**Ideal Distribution** (used by extract service):
- String: 40%
- Number: 30%
- Date: 10%
- Array: 10%
- Object: 5%
- Boolean: 5%

### 4. Node Type Distribution

- **Columns**: 31,653 (98.98%)
- **Tables**: 323 (1.01%)
- **Metadata**: 3 (project, system, information-system)

### 5. Edge Type Distribution

- **HAS_COLUMN**: 23,458 (80.86%)
- **DATA_FLOW**: 5,549 (19.13%)
- **CONTAINS**: 3 (0.01%)

### 6. Table Statistics

- **Total Tables**: 323
- **Tables with Columns**: 323 (100%)
- **Average Columns per Table**: ~72.6
- **Max Columns in Single Table**: 1,026

### 7. Neo4j Sync Status

- **Nodes**: ✅ Synced (31,979 in both)
- **Edges**: ✅ Synced (29,010 in both)

## Quality Assessment

### ✅ Strengths

1. **Complete Data**: All nodes have required identifiers
2. **No Orphan Edges**: All relationships are valid
3. **Good Connectivity**: Every table has columns
4. **Sync Status**: Neo4j and Postgres are in sync
5. **Information Rich**: Moderate entropy indicates diverse data types

### ⚠️ Areas for Improvement

1. **Missing Properties**: ~8,232 nodes have empty properties
   - Most are likely columns with minimal metadata
   - Consider enriching with additional schema information

2. **Type Normalization**: Mixed case types (string vs STRING)
   - Consider normalizing to lowercase for consistency
   - Would improve entropy calculation accuracy

3. **KL Divergence**: 0.42 bits deviation from ideal
   - Slightly higher than optimal
   - Indicates potential bias toward certain types

## Validation Tools

### 1. Extract Validation Tool

The extract service includes a validation tool:
```bash
cd services/extract
go run ./cmd/extract-validate -timeout 10s
```

This checks:
- Postgres connectivity
- Node count
- Edge count
- Minimum thresholds

### 2. Quality Metrics Script

Run comprehensive metrics:
```bash
./scripts/run_quality_metrics.sh
```

### 3. Manual Validation

Check specific metrics:
```bash
# Count nodes by type
docker exec postgres psql -U postgres -d amodels -c "SELECT kind, COUNT(*) FROM glean_nodes GROUP BY kind;"

# Check for orphan edges
docker exec postgres psql -U postgres -d amodels -c "SELECT COUNT(*) FROM glean_edges e WHERE NOT EXISTS (SELECT 1 FROM glean_nodes n WHERE n.id = e.source_id);"

# Verify Neo4j sync
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);"
```

## Recommendations

### Before Training

1. ✅ **Data is ready**: Quality metrics are acceptable
2. ✅ **No critical issues**: All required fields present
3. ⚠️ **Consider property enrichment**: Add more metadata if available
4. ✅ **Proceed with training**: Metrics indicate good data quality

### For Production

1. **Normalize types**: Standardize column type names
2. **Enrich properties**: Add more schema metadata
3. **Monitor metrics**: Track entropy and KL divergence over time
4. **Validate regularly**: Run quality checks after each extraction

## Next Steps

1. ✅ Quality metrics checked
2. ✅ Information metrics calculated
3. ✅ Validation completed
4. ✅ Ready for training

Proceed with Relational Transformer training using the extracted data.

