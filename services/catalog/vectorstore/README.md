# HANA Cloud Vector Store Integration

## Overview

This package provides integration with SAP HANA Cloud vector store for storing and searching public information. It's designed as a general-purpose solution that can be used across different systems (Murex, SAP, BCRS, etc.) and is not limited to any specific customer.

## Features

- ✅ **Semantic Search**: Cosine similarity search across stored vectors (READ-ONLY)
- ✅ **Public Information Access**: Read-only access to public knowledge in HANA Cloud
- ✅ **Multi-Type Support**: Search break patterns, regulatory rules, best practices, knowledge base
- ✅ **System-Agnostic**: Works with any system (Murex, SAP, BCRS, RCO, AxiomSL, or "general")
- ✅ **Filtering**: Filter by type, system, category, tags, public status
- 🔒 **Security**: Write operations disabled to protect confidential information

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

**⚠️ DISABLED FOR SECURITY**: Write operations are disabled to protect confidential information. This service is read-only.

If you need to store public information in HANA Cloud, you must do so through a separate, secure process that ensures data is properly anonymized and approved before storage.

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

**⚠️ DISABLED FOR SECURITY**: Automatic storage of break patterns is disabled to protect confidential information.

Break patterns are not automatically stored in HANA Cloud. If you need to share patterns publicly, they must be manually reviewed, anonymized, and approved before storage through a separate secure process.

### 5. Store Regulatory Rules

**⚠️ DISABLED FOR SECURITY**: Storage operations are disabled to protect confidential information.

Regulatory rules in HANA Cloud are managed separately and are read-only from this service.

### 6. Store Best Practices

**⚠️ DISABLED FOR SECURITY**: Storage operations are disabled to protect confidential information.

Best practices in HANA Cloud are managed separately and are read-only from this service.

## Integration with Break Detection

The vector store is integrated with the break detection system for **READ-ONLY** operations:

1. **❌ Store Break Patterns**: DISABLED - Automatic storage disabled to protect confidential information
2. **✅ Search Similar Breaks**: Find similar historical breaks across systems (read-only)
3. **✅ Regulatory Compliance**: Search regulatory rules for compliance checks (read-only)
4. **✅ Best Practices**: Search best practices across systems (read-only)

**Security Note**: All write operations are disabled. This service can only read from HANA Cloud, not write to it.

### Example Integration (Read-Only)

```go
// In break detection service - SEARCH ONLY
func (s *BreakDetectionService) DetectBreaks(ctx context.Context, req *DetectionRequest) (*DetectionResult, error) {
    // ... perform break detection ...
    
    // Search for similar breaks in HANA Cloud (read-only)
    if s.hanaVectorStore != nil && s.embeddingService != nil {
        for _, breakRecord := range result.Breaks {
            // Generate embedding for search
            content := vectorstore.BuildBreakContent(breakRecord)
            embedding, err := s.embeddingService.GenerateEmbedding(ctx, content)
            if err == nil {
                // Search for similar historical breaks
                similar, err := s.hanaVectorStore.SearchPublicInformation(ctx, embedding, &vectorstore.SearchOptions{
                    Type: "break_pattern",
                    System: "general",
                    Limit: 5,
                    Threshold: 0.7,
                })
                if err == nil && len(similar) > 0 {
                    // Use similar breaks for recommendations
                    breakRecord.SimilarBreaks = convertToSimilarBreaks(similar)
                }
            }
        }
    }
    
    return result, nil
}

// Note: StoreBreakForPublicKnowledge is DISABLED - no automatic storage
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

