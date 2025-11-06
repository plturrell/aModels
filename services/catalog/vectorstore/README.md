# HANA Cloud Vector Store Integration

## Overview

This package provides integration with SAP HANA Cloud vector store for storing and searching public information. It's designed as a general-purpose solution that can be used across different systems (Murex, SAP, BCRS, etc.) and is not limited to any specific customer.

## Features

- ✅ **Vector Storage**: Store embeddings in HANA Cloud REAL_VECTOR format
- ✅ **Semantic Search**: Cosine similarity search across stored vectors
- ✅ **Public Information Management**: Store and search public knowledge
- ✅ **Multi-Type Support**: Break patterns, regulatory rules, best practices, knowledge base
- ✅ **System-Agnostic**: Works with any system (Murex, SAP, BCRS, RCO, AxiomSL, or "general")
- ✅ **Filtering**: Filter by type, system, category, tags, public status
- ✅ **Vector Indexing**: Optional vector indexing for performance

## Usage

### 1. Initialize HANA Cloud Vector Store

```go
import "github.com/plturrell/aModels/services/catalog/vectorstore"

// Create HANA Cloud connection
config := &vectorstore.HANAConfig{
    ConnectionString: "hdb://user:password@host:port",
    Schema:           "PUBLIC",
    TableName:        "PUBLIC_VECTORS",
    VectorDimension:  1536, // OpenAI embedding dimension
    EnableIndexing:   true,
}

store, err := vectorstore.NewHANACloudVectorStore(
    "hdb://user:password@host:port",
    config,
    logger,
)
if err != nil {
    log.Fatal(err)
}
defer store.Close()
```

### 2. Store Public Information

```go
// Store break pattern
info := &vectorstore.PublicInformation{
    ID:       "pattern-123",
    Type:     "break_pattern",
    System:   "sap_fioneer", // or "murex", "bcrs", "general", etc.
    Category: "finance",
    Title:    "Reconciliation Break Pattern",
    Content:  "This pattern occurs when...",
    Vector:   embeddingVector, // []float32 of dimension 1536
    Metadata: map[string]interface{}{
        "frequency": 10,
        "resolution": "Check journal entries",
    },
    Tags:      []string{"reconciliation", "finance", "critical"},
    IsPublic:  true,
    CreatedAt: time.Now(),
    UpdatedAt: time.Now(),
}

err := store.StorePublicInformation(ctx, info)
```

### 3. Semantic Search

```go
// Search for similar break patterns
queryVector := getEmbedding("reconciliation break in finance system")

options := &vectorstore.SearchOptions{
    Type:      "break_pattern",
    System:    "sap_fioneer", // or "general" for all systems
    Category:  "finance",
    IsPublic:  &[]bool{true}[0],
    Limit:     10,
    Threshold: 0.7, // Minimum similarity
}

results, err := store.SearchPublicInformation(ctx, queryVector, options)
```

### 4. Store Break Patterns

```go
breakPatternStore := vectorstore.NewHANABreakPatternStore(store)

pattern := &vectorstore.BreakPattern{
    Description: "Missing journal entries in reconciliation",
    Frequency:   10,
    Resolution:  "Verify source system data",
    Prevention:  "Add data validation checks",
    Tags:        []string{"missing_entry", "reconciliation"},
}

err := breakPatternStore.StoreBreakPattern(
    ctx,
    breakdetection.SystemSAPFioneer,
    breakdetection.DetectionTypeFinance,
    breakdetection.BreakTypeMissingEntry,
    pattern,
    embeddingVector,
)
```

### 5. Store Regulatory Rules

```go
regulatoryStore := vectorstore.NewHANARegulatoryRuleStore(store)

rule := &vectorstore.RegulatoryRule{
    Regulation:    "Basel III",
    Title:         "Capital Ratio Requirements",
    Description:   "Banks must maintain minimum capital ratios...",
    Requirement:   "Tier 1 capital ratio >= 6%",
    Compliance:    "Regular monitoring and reporting",
    EffectiveDate: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
    Tags:          []string{"basel", "capital", "regulatory"},
}

err := regulatoryStore.StoreRegulatoryRule(ctx, rule, embeddingVector)
```

### 6. Store Best Practices

```go
practiceStore := vectorstore.NewHANABestPracticeStore(store)

practice := &vectorstore.BestPractice{
    System:      "general", // or specific system
    Category:    "break_detection",
    Title:       "Automated Baseline Comparison",
    Description: "Use automated baseline comparison to detect breaks...",
    Application: "Implement baseline snapshots before migrations",
    Benefits:    []string{"Reduces manual work", "Faster detection"},
    Tags:        []string{"baseline", "automation", "best_practice"},
}

err := practiceStore.StoreBestPractice(ctx, practice, embeddingVector)
```

## Integration with Break Detection

The vector store can be integrated with the break detection system to:

1. **Store Break Patterns**: Automatically store anonymized break patterns for future reference
2. **Search Similar Breaks**: Find similar historical breaks across systems
3. **Regulatory Compliance**: Store and search regulatory rules for compliance checks
4. **Best Practices**: Share best practices across systems

### Example Integration

```go
// In break detection service
func (s *BreakDetectionService) DetectBreaks(ctx context.Context, req *DetectionRequest) (*DetectionResult, error) {
    // ... perform break detection ...
    
    // Store break patterns in HANA Cloud (if enabled)
    if s.hanaVectorStore != nil && s.embeddingService != nil {
        for _, breakRecord := range result.Breaks {
            // Generate embedding
            embedding, err := s.embeddingService.GenerateEmbedding(ctx, buildBreakContent(breakRecord))
            if err == nil {
                // Store as public pattern (anonymized)
                vectorstore.StoreBreakForPublicKnowledge(ctx, s.hanaVectorStore, breakRecord, embedding, s.logger)
            }
        }
    }
    
    return result, nil
}
```

## Information Types

### 1. Break Patterns (`break_pattern`)
- Historical break patterns
- System-specific or general
- Includes resolution and prevention strategies

### 2. Regulatory Rules (`regulatory_rule`)
- Regulatory requirements (Basel III, IFRS 9, etc.)
- Compliance guidelines
- Public information for all systems

### 3. Best Practices (`best_practice`)
- Best practices for break detection
- System-specific or general
- Application guidelines

### 4. Knowledge Base (`knowledge_base`)
- General knowledge entries
- System documentation
- Custom information

## Configuration

### Environment Variables

```bash
HANA_CLOUD_CONNECTION_STRING="hdb://user:password@host:port"
HANA_CLOUD_SCHEMA="PUBLIC"
HANA_CLOUD_TABLE_NAME="PUBLIC_VECTORS"
HANA_CLOUD_VECTOR_DIMENSION=1536
HANA_CLOUD_ENABLE_INDEXING=true
```

### Table Schema

The vector store automatically creates the following table:

```sql
CREATE COLUMN TABLE PUBLIC.PUBLIC_VECTORS (
    ID NVARCHAR(255) PRIMARY KEY,
    TYPE NVARCHAR(100) NOT NULL,
    SYSTEM NVARCHAR(100),
    CATEGORY NVARCHAR(100),
    TITLE NVARCHAR(500),
    CONTENT NCLOB,
    VECTOR REAL_VECTOR(1536),
    METADATA NCLOB,
    TAGS NVARCHAR(500),
    IS_PUBLIC BOOLEAN DEFAULT true,
    CREATED_AT TIMESTAMP NOT NULL,
    UPDATED_AT TIMESTAMP NOT NULL,
    CREATED_BY NVARCHAR(100)
)
```

## Performance

- **Vector Indexing**: Enable for fast similarity search (COSINE_SIMILARITY)
- **Indexes**: Created on TYPE, SYSTEM, CATEGORY, IS_PUBLIC, CREATED_AT
- **Batch Operations**: Support for batch inserts (future enhancement)

## Security

- **Public vs Private**: Use `IsPublic` flag to control visibility
- **System Filtering**: Filter by system to restrict access
- **SQL Injection Prevention**: All inputs are escaped

## Multi-System Support

The vector store is designed to work across systems:

- **Murex**: Store Murex-specific break patterns
- **SAP Fioneer**: Store SAP-specific patterns
- **BCRS**: Store capital calculation patterns
- **RCO**: Store liquidity patterns
- **AxiomSL**: Store regulatory reporting patterns
- **General**: Store system-agnostic information

## Examples

See `examples/` directory for complete examples:
- Storing break patterns
- Searching regulatory rules
- Managing best practices
- Integration with break detection service

## Future Enhancements

- [ ] Batch operations for bulk inserts
- [ ] Real-time synchronization
- [ ] Versioning for information updates
- [ ] Access control and permissions
- [ ] Analytics and usage statistics

