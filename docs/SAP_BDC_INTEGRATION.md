# SAP BDC Integration Guide

## Overview

The SAP BDC (Business Data Cloud) integration extracts data from SAP Business Data Cloud and converts it to a knowledge graph format for use in the aModels system.

**File**: `services/extract/sap_bdc_integration.go`

---

## Architecture

### Components

1. **SAPBDCIntegration Struct**
   - Manages HTTP client connection to SAP BDC service
   - Handles request/response processing
   - Converts SAP schema to graph format

2. **ExtractFromSAPBDC() Function**
   - Main extraction entry point
   - Makes POST request to SAP BDC `/extract` endpoint
   - Processes response and converts to graph

3. **convertSAPBDCSchemaToGraph() Function**
   - Transforms SAP BDC schema into nodes and edges
   - Creates hierarchical structure: database → tables/views → columns
   - Handles foreign key relationships

### Data Flow

```
SAP BDC Service → Extract Request → Schema Response → Graph Conversion → Nodes & Edges
```

---

## Current Implementation Details

### Request Format

The integration sends POST requests to the SAP BDC service with:

```go
{
  "formation_id": "formation-id",
  "source_system": "system-name",
  "include_views": true,
  "data_product_id": "optional",
  "space_id": "optional",
  "database": "optional"
}
```

### Response Processing

The service returns:
- **Schema**: Database, schema, tables, views, columns
- **Metadata**: Additional properties for enrichment
- **Foreign Keys**: Relationship information

### Graph Conversion

**Node Types Created:**
- `database`: Database/schema container
- `table`: Table nodes with metadata
- `view`: View nodes with definitions
- `column`: Column nodes with data types and constraints

**Edge Types Created:**
- `CONTAINS`: Database → Table/View
- `HAS_COLUMN`: Table/View → Column
- `REFERENCES`: Table → Table (via foreign keys)

**Node Properties:**
- Source system identification (`source: "sap_bdc"`)
- Project and system IDs for multi-tenancy
- Column metadata (data type, nullable, default, comments)
- Table/view metadata from SAP

---

## Current Limitations

### 1. Error Handling
- **Current**: Basic error handling with HTTP status checks
- **Opportunity**: Add retry logic with exponential backoff
- **Impact**: More resilient to transient network issues

### 2. Incremental Extraction
- **Current**: Always performs full extraction
- **Opportunity**: Add change detection and timestamp-based filtering
- **Impact**: Faster updates, reduced load on SAP BDC

### 3. Performance
- **Current**: Sequential processing of tables/views
- **Opportunity**: Add batching and parallel processing for large schemas
- **Impact**: Better performance for enterprise-scale schemas

### 4. Metadata Enrichment
- **Current**: Basic metadata propagation
- **Opportunity**: Enhance with business glossary mappings, data quality metrics
- **Impact**: Richer context for downstream processing

### 5. View Dependencies
- **Current**: View definitions stored but not parsed
- **Opportunity**: Parse SQL to extract underlying table dependencies
- **Impact**: More complete lineage tracking

### 6. Schema Evolution
- **Current**: No tracking of schema changes over time
- **Opportunity**: Add schema versioning and change tracking
- **Impact**: Better understanding of schema evolution patterns

---

## Potential Contributions

### High-Value Enhancements

1. **Incremental Extraction Support**
   - Track last extraction timestamp
   - Filter by modification date
   - Only extract changed objects

2. **View Dependency Parsing**
   - Parse view SQL definitions
   - Extract referenced tables/views
   - Create explicit dependency edges

3. **Schema Versioning**
   - Track schema versions over time
   - Detect schema changes
   - Store change history in graph

4. **Metadata Enrichment**
   - Integrate with SAP Business Glossary
   - Add data quality metrics
   - Include business context

5. **Performance Optimizations**
   - Parallel table processing
   - Batch column processing
   - Caching of schema metadata

### Code Locations

- **Main File**: `services/extract/sap_bdc_integration.go`
- **Integration Point**: Called from extract service main handler
- **Graph Storage**: Uses Node/Edge structures defined in extract service

---

## Usage Example

```go
// Create integration instance
logger := log.New(os.Stdout, "", log.LstdFlags)
sapBDC := NewSAPBDCIntegration(logger)

// Extract data
nodes, edges, err := sapBDC.ExtractFromSAPBDC(
    ctx,
    "formation-123",
    "sap-system",
    "data-product-456",
    "space-789",
    "database-name",
    "project-001",
    "system-002",
)

if err != nil {
    log.Fatal(err)
}

// Use nodes and edges in knowledge graph
// ...
```

---

## Configuration

### Environment Variables

- `SAP_BDC_URL`: SAP BDC service URL (default: `http://localhost:8083`)

### Timeout Settings

- HTTP client timeout: 30 seconds (configurable)

---

## Integration with Other Services

### Extract Service
- SAP BDC integration is part of the extract service
- Called when SAP BDC source type is specified
- Results stored in Neo4j knowledge graph

### Knowledge Graph
- Nodes and edges created follow standard graph schema
- Compatible with graph query and analysis tools
- Supports cross-system pattern matching

---

## Testing

### Unit Tests
- Test schema conversion logic
- Test error handling
- Test edge cases (empty schemas, missing fields)

### Integration Tests
- Test with mock SAP BDC service
- Verify graph structure correctness
- Validate metadata propagation

---

## Future Enhancements

1. **Real-time Updates**: Webhook support for schema changes
2. **Multi-tenant Support**: Enhanced project/system isolation
3. **Custom Transformations**: User-defined mapping rules
4. **Performance Monitoring**: Metrics and observability
5. **Caching Layer**: Reduce redundant API calls

---

## Related Documentation

- [Extract Service Overview](../services/extract/README.md)
- [Knowledge Graph Schema](../services/graph/README.md)
- [Cross-System Extraction](../services/extract/cross_system_extractor.go)

