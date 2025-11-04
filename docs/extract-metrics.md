# Extract Service Metrics

This document describes the metrics used in the `agenticAiETH_layer4_Extract` service for processing and analyzing data extraction and graph generation tasks.

## Overview

The Extract service tracks multiple categories of metrics:
1. **Telemetry Metrics** - Operational metrics for monitoring
2. **Information Theory Metrics** - Data quality and distribution metrics
3. **Graph Normalization Metrics** - Graph processing statistics
4. **Schema Metrics** - Column and data type distribution metrics

---

## 1. Telemetry Metrics

Telemetry metrics are collected for every extract operation and logged to the Postgres telemetry service.

### Input Metrics (`telemetryInputFromRequest`)

**Request-level metrics:**
- `prompt_description` - The prompt description used for extraction
- `model_id` - Model identifier (e.g., "gemini-2.5-flash")
- `documents_count` - Number of documents in the request
- `text_or_documents_present` - Boolean indicating if text/documents were provided
- `text_or_documents_bytes` - Size of text/documents in bytes
- `examples_count` - Number of example extractions provided

**Document-level metrics:**
- `document_preview` - First 200 characters of the document (for single document)
- `document_chars` - Character count of the document
- `document_hash` - SHA-256 hash of the document content
- `documents_preview` - Array of previews (first 3 documents, 120 chars each)
- `documents_hashes` - Array of SHA-256 hashes for all documents
- `documents_total_bytes` - Total byte size of all documents

**Example-level metrics:**
- `examples_classes` - Sorted list of unique extraction classes from examples

### Output Metrics (`telemetryOutputFromResponse`)

**Response-level metrics:**
- `extraction_count` - Total number of extractions returned
- `entities` - Flattened entity map (grouped by entity type)
- `extractions` - Array of extraction summaries, each containing:
  - `class` - Extraction class name
  - `text_preview` - First 160 characters of extracted text
  - `text_chars` - Character count of extracted text
  - `text_hash` - SHA-256 hash of extracted text
  - `attributes` - Additional attributes (if present)
  - `start_index` - Start position in source document (if available)
  - `end_index` - End position in source document (if available)

### Operational Metrics

**Timing metrics:**
- `latency` - Request processing duration (milliseconds)
- `started_at` - Timestamp when request started
- `completed_at` - Timestamp when request completed

**Status metrics:**
- `status` - Operation status (SUCCESS or ERROR)
- `error_message` - Error message if operation failed
- `session_id` - Session identifier for request tracking
- `user_id_hash` - Hashed user identifier
- `privacy_level` - Privacy level classification

---

## 2. Information Theory Metrics

These metrics use information theory to assess data quality and distribution.

### Metadata Entropy (`calculateEntropy`)

**Purpose:** Measures the diversity/unpredictability of column data types in extracted schemas.

**Formula:**
```
H(X) = -Σ p(x) * log₂(p(x))
```

Where:
- `H(X)` is the entropy
- `p(x)` is the probability of data type `x`
- Sum is over all unique data types

**Calculation:**
1. Collects all column data types from graph nodes
2. Counts frequency of each type
3. Calculates probability distribution
4. Computes Shannon entropy using base-2 logarithm

**Usage:**
- Higher entropy = more diverse data types (good for heterogeneous schemas)
- Lower entropy = more uniform data types (good for structured schemas)
- Returned in graph response as `metadata_entropy`

### KL Divergence (`calculateKLDivergence`)

**Purpose:** Measures how much the actual data type distribution differs from an ideal/reference distribution.

**Formula:**
```
D_KL(P||Q) = Σ P(x) * log₂(P(x) / Q(x))
```

Where:
- `P(x)` is the actual distribution
- `Q(x)` is the ideal distribution
- Uses add-one smoothing (1e-10) for missing values

**Default Ideal Distribution:**
```go
{
  "string":  0.4,  // 40% strings
  "number":  0.4,  // 40% numbers
  "boolean": 0.1,  // 10% booleans
  "date":    0.05, // 5% dates
  "array":   0.03, // 3% arrays
  "object":  0.02  // 2% objects
}
```

**Usage:**
- Lower KL divergence = distribution closer to ideal
- Higher KL divergence = distribution deviates from ideal
- Can be customized via `ideal_distribution` in graph request
- Returned in graph response as `kl_divergence`

**Interpretation:**
- **0.0** = Perfect match to ideal distribution
- **< 0.1** = Very close to ideal
- **0.1 - 0.5** = Moderate deviation
- **> 0.5** = Significant deviation

---

## 3. Graph Normalization Metrics

These metrics track the graph processing and normalization pipeline.

### Node Metrics

- `original_node_count` - Number of nodes before normalization
- `unique_node_count` - Number of unique nodes after deduplication
- `dropped_nodes` - Number of nodes dropped (empty IDs, invalid data)
- `catalog_nodes_added` - Number of catalog nodes added (project/system/information-system)

### Edge Metrics

- `original_edge_count` - Number of edges before normalization
- `unique_edge_count` - Number of unique edges after deduplication
- `dropped_edges` - Number of edges dropped (missing source/target nodes, invalid references)

### Catalog Metrics

- `catalog_entries_added` - Number of catalog entries updated/added
- `root_node_id` - Identifier of the selected root node

### Normalization Warnings

Array of warning messages generated during normalization:
- `"dropped node with empty id"`
- `"dropped edge with missing source or target"`
- `"dropped edge X->Y (missing source node)"`
- `"dropped edge X->Y (missing target node)"`
- `"graph produced no root node; catalog containment edges skipped"`
- `"failed to persist catalog update: <error>"`

---

## 4. Schema Metrics

Additional metrics calculated during schema extraction from JSON tables.

### Column Profile Metrics (`profileJSONColumns`)

For each column in JSON tables:
- `type` - Inferred data type (string, number, boolean, object, array, mixed, unknown)
- `types` - Array of all types found (for mixed-type columns)
- `nullable` - Boolean indicating if column contains null values
- `presence_ratio` - Fraction of rows where column is present (0.0 to 1.0)
- `examples` - Up to 3 example values from the data

### Distribution Metrics

**Actual Distribution:**
- Calculated from column data types found in the graph
- Normalized to probabilities (sums to 1.0)
- Used for KL divergence calculation

**Type Frequencies:**
- Count of each data type across all columns
- Used to compute entropy and distribution metrics

---

## 5. Usage in Graph Processing

### Graph Request/Response Flow

1. **Input Processing:**
   - JSON tables → Schema extraction
   - Hive DDLs → DDL parsing
   - SQL queries → Lineage extraction
   - Control-M files → Job parsing

2. **Graph Construction:**
   - Nodes: tables, columns, jobs, calendars, conditions
   - Edges: relationships, data flows, schedules, dependencies

3. **Normalization:**
   - Deduplication of nodes and edges
   - Catalog integration
   - Root node selection
   - Statistics collection

4. **Metric Calculation:**
   - Column data types extracted
   - Metadata entropy calculated
   - Actual distribution computed
   - KL divergence calculated vs. ideal distribution

5. **Response:**
   ```json
   {
     "nodes": [...],
     "edges": [...],
     "metadata_entropy": 2.345,
     "kl_divergence": 0.123,
     "root_node_id": "project:abc123",
     "normalization": {
       "root_node_id": "project:abc123",
       "stats": {
         "original_node_count": 150,
         "unique_node_count": 145,
         "dropped_nodes": 5,
         "catalog_nodes_added": 3,
         "catalog_entries_added": 2,
         "original_edge_count": 200,
         "unique_edge_count": 195,
         "dropped_edges": 5
       },
       "warnings": [...]
     }
   }
   ```

---

## 6. Persistence and Storage

Metrics are stored in multiple persistence layers:

### Graph Persistence
- **Neo4j** - Graph database for nodes and edges
- **Glean** - Knowledge graph export (if enabled)

### Vector Persistence
- **Redis** - SQL query embeddings stored as vectors

### Document Persistence
- **File System** - Original JSON documents

### Table Persistence
- **SQLite** - Extracted table data

### Telemetry Persistence
- **Postgres** - All telemetry records via gRPC service

---

## 7. Performance Considerations

### Metric Collection Overhead

- **Telemetry:** Asynchronous, non-blocking (3s timeout)
- **Entropy/KL:** O(n) where n = number of columns
- **Normalization:** O(n + m) where n = nodes, m = edges
- **Persistence:** Background operations, don't block response

### Optimization

- Entropy calculation uses efficient counting maps
- KL divergence uses add-one smoothing to avoid division by zero
- Normalization uses hash maps for O(1) lookups
- Telemetry uses timeouts to prevent blocking

---

## 8. Monitoring and Alerting

### Key Metrics to Monitor

1. **High Error Rate:** `status = ERROR` in telemetry
2. **High Latency:** `latency` > threshold
3. **High KL Divergence:** Indicates unexpected data type distributions
4. **High Dropped Nodes/Edges:** Indicates data quality issues
5. **Low Metadata Entropy:** May indicate insufficient schema diversity

### Recommended Thresholds

- **Latency:** Alert if > 30 seconds
- **Error Rate:** Alert if > 5% of requests
- **KL Divergence:** Alert if > 1.0 (significant deviation)
- **Dropped Nodes:** Alert if > 10% of original nodes
- **Metadata Entropy:** Monitor trends, alert on sudden drops

---

## 9. Example Usage

### Extracting Metrics from Response

```go
response := graphResponse{
    MetadataEntropy: 2.5,
    KLDivergence:    0.15,
    Normalization: NormalizationStats{
        Stats: map[string]any{
            "original_node_count": 100,
            "unique_node_count":   95,
            "dropped_nodes":       5,
        },
    },
}

// Use metadata entropy to assess schema diversity
if response.MetadataEntropy < 1.0 {
    log.Printf("Low schema diversity: entropy=%.2f", response.MetadataEntropy)
}

// Use KL divergence to assess data quality
if response.KLDivergence > 0.5 {
    log.Printf("Significant deviation from ideal: KL=%.2f", response.KLDivergence)
}
```

---

## References

- **Information Theory:** Shannon entropy (1948)
- **KL Divergence:** Kullback-Leibler divergence for distribution comparison
- **Telemetry:** Postgres Lang Service gRPC API
- **Graph Processing:** Neo4j, Glean knowledge graphs

