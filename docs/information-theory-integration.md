# Information Theory Metrics Integration

This document describes how information theory metrics (metadata entropy and KL divergence) are integrated throughout the Extract service's code parsing and Glean export pipeline.

## Overview

Information theory metrics are calculated during graph processing and flow through multiple stages:

1. **Code Parsing** → Extracts schema/data types from various sources
2. **Graph Normalization** → Aggregates nodes and edges
3. **Metric Calculation** → Computes entropy and KL divergence
4. **Graph Storage** → Stores metrics in root node properties
5. **Glean Export** → Includes metrics in export manifest
6. **Telemetry** → Records metrics for monitoring

---

## Integration Flow

### 1. Code Parsing Phase

During the `/graph` endpoint processing, various parsers extract schema information:

- **JSON Tables** (`extractSchemaFromJSON`): Extracts column types from JSON data
- **Hive DDLs** (`parseHiveDDL`): Parses DDL statements to extract table/column schemas
- **SQL Queries** (`parseSQL`): Extracts lineage but also infers column types from queries
- **Control-M Files** (`parseControlMXML`): Extracts job definitions and dependencies

**Column Type Extraction:**
- Each parser identifies column data types (string, number, boolean, date, array, object)
- Types are stored in node `Props["type"]` for column nodes
- This type information is the foundation for metric calculation

### 2. Graph Normalization Phase

After all parsing is complete:

```go
normResult := normalizeGraph(normalizationInput{
    Nodes:               nodes,
    Edges:               edges,
    ProjectID:           req.ProjectID,
    SystemID:            req.SystemID,
    InformationSystemID: req.InformationSystemID,
    Catalog:             s.catalog,
})
```

- Deduplicates nodes and edges
- Adds catalog entries (project/system/information-system)
- Selects root node for the graph

### 3. Metric Calculation Phase

**After normalization**, metrics are calculated from the final graph:

```go
// Collect all column data types from the normalized graph
columnDtypes := make([]string, 0)
for _, node := range nodes {
    if node.Type != "column" || node.Props == nil {
        continue
    }
    if dtype, ok := node.Props["type"].(string); ok && dtype != "" {
        columnDtypes = append(columnDtypes, dtype)
    }
}

// Calculate metadata entropy (diversity of data types)
metadataEntropy := calculateEntropy(columnDtypes)

// Calculate actual distribution
actualDistribution := make(map[string]float64)
totalColumns := float64(len(columnDtypes))
for _, dtype := range columnDtypes {
    actualDistribution[dtype]++
}
if totalColumns > 0 {
    for dtype, count := range actualDistribution {
        actualDistribution[dtype] = count / totalColumns
    }
}

// Calculate KL divergence vs ideal distribution
idealDistribution := req.IdealDistribution
if idealDistribution == nil {
    idealDistribution = map[string]float64{
        "string":  0.4,  // 40%
        "number":  0.4,  // 40%
        "boolean": 0.1,  // 10%
        "date":    0.05, // 5%
        "array":   0.03, // 3%
        "object":  0.02  // 2%
    }
}
klDivergence := calculateKLDivergence(actualDistribution, idealDistribution)
```

### 4. Metrics Storage in Root Node

Metrics are stored in the root node properties for persistence:

```go
if rootID != "" {
    for i := range nodes {
        if nodes[i].ID == rootID {
            if nodes[i].Props == nil {
                nodes[i].Props = make(map[string]any)
            }
            nodes[i].Props["metadata_entropy"] = metadataEntropy
            nodes[i].Props["kl_divergence"] = klDivergence
            nodes[i].Props["actual_distribution"] = actualDistribution
            nodes[i].Props["ideal_distribution"] = idealDistribution
            nodes[i].Props["column_count"] = len(columnDtypes)
            nodes[i].Props["metrics_calculated_at"] = time.Now().UTC().Format(time.RFC3339Nano)
            break
        }
    }
}
```

**Storage Locations:**
- **Neo4j**: Root node properties can be queried via Cypher
- **Glean**: Root node properties are exported as part of the graph
- **HANA/Postgres**: Root node properties stored in `glean_nodes` table

### 5. Glean Export Integration

When `SaveGraph` is called on `GleanPersistence`, metrics are extracted from the root node and included in the export manifest:

```go
func (g *GleanPersistence) buildBatch(nodes []Node, edges []Edge) ([]gleanPredicate, error) {
    // ... node and edge fact creation ...
    
    // Extract metrics from root node
    var metadataEntropy, klDivergence float64
    var actualDistribution, idealDistribution map[string]any
    var columnCount int
    var metricsCalculatedAt string
    
    for _, node := range nodes {
        if node.Props != nil {
            if val, ok := node.Props["metadata_entropy"].(float64); ok {
                metadataEntropy = val
            }
            // ... extract other metrics ...
            if node.Type == "project" || node.Type == "system" || node.Type == "information-system" {
                if metadataEntropy > 0 || klDivergence > 0 {
                    break
                }
            }
        }
    }
    
    // Add metrics to export manifest
    exportMeta := gleanFact{
        Key: map[string]any{
            "exported_at": time.Now().UTC().Format(time.RFC3339Nano),
            "node_count":  len(nodeFacts),
            "edge_count":  len(edgeFacts),
        },
    }
    
    if metadataEntropy > 0 || klDivergence > 0 {
        if metaMap, ok := exportMeta.Key.(map[string]any); ok {
            metaMap["metadata_entropy"] = metadataEntropy
            metaMap["kl_divergence"] = klDivergence
            metaMap["column_count"] = columnCount
            metaMap["actual_distribution"] = actualDistribution
            metaMap["ideal_distribution"] = idealDistribution
            metaMap["metrics_calculated_at"] = metricsCalculatedAt
        }
    }
    
    predicates = append(predicates, gleanPredicate{
        Predicate: g.predicate(manifestPredicateName),
        Facts:     []gleanFact{exportMeta},
    })
}
```

**Glean Export Manifest Structure:**
```json
{
  "predicate": "agenticAiETH.ETL.ExportManifest.1",
  "facts": [
    {
      "key": {
        "exported_at": "2024-01-15T10:30:45.123456789Z",
        "node_count": 150,
        "edge_count": 200,
        "metadata_entropy": 2.345,
        "kl_divergence": 0.123,
        "column_count": 45,
        "actual_distribution": {
          "string": 0.42,
          "number": 0.38,
          "boolean": 0.12,
          "date": 0.05,
          "array": 0.02,
          "object": 0.01
        },
        "ideal_distribution": {
          "string": 0.4,
          "number": 0.4,
          "boolean": 0.1,
          "date": 0.05,
          "array": 0.03,
          "object": 0.02
        },
        "metrics_calculated_at": "2024-01-15T10:30:45.123456789Z"
      }
    }
  ]
}
```

### 6. Telemetry Integration

Metrics are also recorded in telemetry for monitoring:

```go
telemetryRecord := telemetryRecord{
    LibraryType:  "layer4_extract",
    Operation:    "graph_processing",
    Input: map[string]any{
        "json_tables_count":    len(req.JSONTables),
        "hive_ddls_count":      len(req.HiveDDLs),
        "sql_queries_count":    len(req.SqlQueries),
        "control_m_files_count": len(req.ControlMFiles),
    },
    Output: map[string]any{
        "nodes_count":        len(nodes),
        "edges_count":        len(edges),
        "metadata_entropy":   metadataEntropy,
        "kl_divergence":      klDivergence,
        "actual_distribution": actualDistribution,
        "ideal_distribution":  idealDistribution,
        "column_count":        len(columnDtypes),
    },
    StartedAt:    started,
    CompletedAt:  time.Now(),
    Latency:      time.Since(started),
}
```

---

## Usage in Glean Queries

Once exported to Glean, metrics can be queried:

### Query Export Manifests by Metrics

```angle
agenticAiETH.ETL.ExportManifest.1 {
  exported_at,
  metadata_entropy,
  kl_divergence,
  column_count
}
where metadata_entropy < 1.0  -- Find low diversity schemas
```

### Query Root Nodes with Metrics

```angle
agenticAiETH.ETL.Node.1 {
  id,
  kind,
  label,
  properties_json
}
where kind = "project" or kind = "system"
```

Then parse `properties_json` to extract:
- `metadata_entropy`
- `kl_divergence`
- `actual_distribution`
- `ideal_distribution`
- `column_count`

### Track Metrics Over Time

```angle
agenticAiETH.ETL.ExportManifest.1 {
  exported_at,
  metadata_entropy,
  kl_divergence
}
| order by exported_at desc
```

---

## Monitoring and Alerting

### Log-Based Alerts

The service logs warnings/errors when metrics indicate issues:

- **Low entropy (<1.0)**: `WARNING: Low metadata entropy - schema may have low diversity`
- **High entropy (>4.0)**: `INFO: High metadata entropy - schema has high diversity`
- **High KL divergence (>0.5)**: `WARNING: High KL divergence - data type distribution deviates significantly from ideal`
- **Very high KL divergence (>1.0)**: `ERROR: Very high KL divergence - data type distribution is highly abnormal`

### Telemetry-Based Monitoring

Query telemetry service to track metrics over time:

```sql
SELECT 
    output->>'metadata_entropy' as entropy,
    output->>'kl_divergence' as kl_div,
    created_at
FROM lang_operations
WHERE library_type = 'layer4_extract'
  AND operation = 'graph_processing'
ORDER BY created_at DESC
LIMIT 100;
```

---

## Design Decisions

### Why Calculate After Normalization?

Metrics are calculated **after** all parsing and normalization because:

1. **Complete Picture**: We need the full normalized graph to accurately assess schema diversity
2. **Deduplication**: Normalization deduplicates nodes, giving accurate counts
3. **Root Node**: Normalization selects/creates the root node where metrics are stored
4. **Single Source of Truth**: One calculation point ensures consistency

### Why Store in Root Node?

Storing metrics in the root node:

1. **Graph Queries**: Easy to query in Neo4j: `MATCH (n:project) RETURN n.metadata_entropy`
2. **Glean Export**: Root nodes are always exported, ensuring metrics are available
3. **Hierarchical Context**: Metrics belong to the entire graph, not individual nodes
4. **Versioning**: Each export has metrics for that version of the graph

### Why Include in Glean Manifest?

Including metrics in the export manifest:

1. **Batch-Level Metrics**: Each export batch has aggregate metrics
2. **Query Efficiency**: Can query manifests without parsing node properties
3. **Trend Analysis**: Easy to track metrics across multiple exports
4. **Data Quality Dashboard**: Can build dashboards from manifest queries

---

## Future Enhancements

### Per-Source Metrics

Currently, metrics are calculated for the entire graph. Future enhancements could include:

- **Per JSON Table**: Calculate metrics for each JSON table separately
- **Per DDL**: Calculate metrics for each Hive DDL schema
- **Per SQL Query**: Calculate metrics for columns referenced in each query

### Real-Time Metrics

During parsing, we could:

- **Track metrics incrementally** as each source is parsed
- **Log warnings early** if a source has abnormal distributions
- **Skip processing** if metrics indicate data quality issues

### Metric-Based Routing

Use metrics to:

- **Route to different processors** based on schema complexity (entropy)
- **Adjust processing strategies** based on distribution patterns
- **Trigger validation** if KL divergence exceeds thresholds

---

## References

- [Information Theory Metrics Documentation](./extract-metrics.md)
- [Extract Service Architecture](./architecture.md)
- [Glean Export Format](https://github.com/plturrell/aModels/services/extract/glean_persistence.go)

