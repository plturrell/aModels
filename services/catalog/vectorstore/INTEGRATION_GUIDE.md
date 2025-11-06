# HANA Cloud Vector Store Integration Guide

## Overview

The HANA Cloud Vector Store integration provides a general-purpose solution for storing and searching public information across all systems (Murex, SAP, BCRS, RCO, AxiomSL, or general). This enables knowledge sharing, pattern recognition, and best practice dissemination.

## Architecture

```
┌─────────────────┐
│ Break Detection │
│    Service      │
└────────┬────────┘
         │
         │ Stores break patterns
         │
┌────────▼────────────────────────┐
│   HANA Cloud Vector Store       │
│                                 │
│  - Break Patterns               │
│  - Regulatory Rules             │
│  - Best Practices               │
│  - Knowledge Base               │
└────────┬────────────────────────┘
         │
         │ Semantic Search
         │
┌────────▼────────┐
│  All Systems    │
│  (Murex, SAP,   │
│   BCRS, etc.)   │
└─────────────────┘
```

## Use Cases

### 1. Cross-System Break Pattern Sharing

**Problem**: Each system discovers break patterns independently, leading to duplicated effort.

**Solution**: Store break patterns in HANA Cloud, accessible to all systems.

**Example**:
- Murex detects a reconciliation break pattern
- Stored in HANA Cloud as public knowledge
- SAP Fioneer can search and find similar patterns
- BCRS can learn from Murex's experience

### 2. Regulatory Compliance Knowledge Base

**Problem**: Regulatory rules are scattered across systems and documents.

**Solution**: Centralized regulatory rules repository in HANA Cloud.

**Example**:
- Store Basel III capital requirements
- Store IFRS 9 reporting requirements
- All systems can search and reference the same rules
- Updates propagate to all systems

### 3. Best Practice Sharing

**Problem**: Best practices discovered in one system don't benefit others.

**Solution**: Store best practices in HANA Cloud for cross-system access.

**Example**:
- Murex discovers best practice for baseline management
- Stored as "general" (available to all systems)
- SAP, BCRS, RCO can all benefit from the same practice

### 4. Historical Break Pattern Analysis

**Problem**: Similar breaks occur across systems but aren't connected.

**Solution**: Semantic search across all historical breaks to find patterns.

**Example**:
- New break detected in SAP Fioneer
- Search HANA Cloud for similar breaks across all systems
- Find resolution from Murex's experience
- Apply same resolution approach

## Integration Steps

### Step 1: Configure HANA Cloud Connection

```go
import "github.com/plturrell/aModels/services/catalog/vectorstore"

config := &vectorstore.HANAConfig{
    ConnectionString: os.Getenv("HANA_CLOUD_CONNECTION_STRING"),
    Schema:           "PUBLIC",
    TableName:        "PUBLIC_VECTORS",
    VectorDimension:  1536, // OpenAI embeddings
    EnableIndexing:   true,
}

store, err := vectorstore.NewHANACloudVectorStore(
    os.Getenv("HANA_CLOUD_CONNECTION_STRING"),
    config,
    logger,
)
```

### Step 2: Initialize Embedding Service

```go
embeddingService := vectorstore.NewEmbeddingService(
    os.Getenv("LOCALAI_URL"), // or OpenAI API
    logger,
)
```

### Step 3: Integrate with Break Detection

```go
// In break detection service initialization
hanaVectorStore, _ := vectorstore.NewHANACloudVectorStore(...)
embeddingService := vectorstore.NewEmbeddingService(...)

breakDetectionService := breakdetection.NewBreakDetectionService(
    // ... other services ...
)

// Store break patterns automatically
if hanaVectorStore != nil && embeddingService != nil {
    // Store patterns after detection
}
```

### Step 4: Enable API Endpoints

```go
// In main.go
if hanaVectorStore != nil {
    handler := vectorstore.NewHANAVectorStoreHandler(
        hanaVectorStore,
        embeddingService,
        logger,
    )
    
    mux.HandleFunc("/vectorstore/store", handler.HandleStoreInformation)
    mux.HandleFunc("/vectorstore/search", handler.HandleSearchInformation)
    mux.HandleFunc("/vectorstore/", handler.HandleGetInformation)
}
```

## Information Types

### 1. Break Patterns (`break_pattern`)

**Purpose**: Store historical break patterns for pattern recognition.

**Example**:
```json
{
  "type": "break_pattern",
  "system": "sap_fioneer",
  "category": "finance",
  "title": "Reconciliation Break Pattern",
  "content": "Reconciliation breaks occur when...",
  "metadata": {
    "frequency": 10,
    "resolution": "Check ETL pipeline",
    "prevention": "Add validation checks"
  },
  "tags": ["reconciliation", "finance", "critical"]
}
```

### 2. Regulatory Rules (`regulatory_rule`)

**Purpose**: Store regulatory requirements for compliance.

**Example**:
```json
{
  "type": "regulatory_rule",
  "system": "general",
  "category": "Basel III",
  "title": "Capital Ratio Requirements",
  "content": "Banks must maintain minimum capital ratios...",
  "metadata": {
    "regulation": "Basel III",
    "requirement": "Tier 1 >= 6%",
    "effective_date": "2024-01-01"
  }
}
```

### 3. Best Practices (`best_practice`)

**Purpose**: Share best practices across systems.

**Example**:
```json
{
  "type": "best_practice",
  "system": "general",
  "category": "break_detection",
  "title": "Automated Baseline Comparison",
  "content": "Use automated baselines to...",
  "metadata": {
    "application": "Create baseline before migration",
    "benefits": ["Reduces manual work", "Faster detection"]
  }
}
```

### 4. Knowledge Base (`knowledge_base`)

**Purpose**: General knowledge entries.

**Example**:
```json
{
  "type": "knowledge_base",
  "system": "murex",
  "category": "documentation",
  "title": "Murex Version Migration Guide",
  "content": "Step-by-step guide for...",
  "tags": ["migration", "murex", "guide"]
}
```

## API Endpoints

### ⚠️ Store Information (DISABLED FOR SECURITY)

**POST /vectorstore/store** - **DISABLED**

Write operations are disabled to protect confidential information. This service is read-only.

If you need to store public information in HANA Cloud, you must do so through a separate, secure process that ensures data is properly anonymized and approved before storage.

### Search Information

```bash
POST /vectorstore/search
Content-Type: application/json

{
  "query": "reconciliation break in finance system",
  "type": "break_pattern",
  "system": "general",
  "limit": 10,
  "threshold": 0.7
}
```

### Get Information by ID

```bash
GET /vectorstore/{id}
```

### List Public Information

```bash
GET /vectorstore?type={type}&system={system}&limit={limit}&offset={offset}
```

**Query Parameters:**
- `type` - Filter by type (break_pattern, regulatory_rule, best_practice, knowledge_base)
- `system` - Filter by system (murex, sap_fioneer, bcrs, rco, axiomsl, general)
- `category` - Filter by category
- `tags` - Filter by tags (comma-separated)
- `is_public` - Filter by public status (true/false/1/0)
- `limit` - Max results (default: 100)
- `offset` - Pagination offset (default: 0)
- `order_by` - Order by field (created_at, updated_at, title)
- `order_desc` - Order descending (true/false/1/0)

**Examples:**

```bash
# List all break patterns for Murex
curl "http://localhost:8084/vectorstore?type=break_pattern&system=murex&limit=20"

# List all public regulatory rules
curl "http://localhost:8084/vectorstore?type=regulatory_rule&is_public=true"

# List best practices, ordered by creation date
curl "http://localhost:8084/vectorstore?type=best_practice&order_by=created_at&order_desc=true"

# List all public information for a category
curl "http://localhost:8084/vectorstore?category=finance&is_public=true&limit=50"

# Paginated listing
curl "http://localhost:8084/vectorstore?limit=10&offset=0"  # First page
curl "http://localhost:8084/vectorstore?limit=10&offset=10" # Second page
```

## Multi-System Support

The vector store supports multiple systems:

- **System-Specific**: Store information for specific systems (e.g., `"system": "murex"`)
- **General**: Store information available to all systems (e.g., `"system": "general"`)
- **Cross-System Search**: Search across all systems or filter by specific system

### Example: Murex-Specific Pattern

```go
info := &PublicInformation{
    System: "murex", // Murex-specific
    Type:   "break_pattern",
    // ...
}
```

### Example: General Best Practice

```go
info := &PublicInformation{
    System: "general", // Available to all systems
    Type:   "best_practice",
    // ...
}
```

### Example: Cross-System Search

```go
options := &SearchOptions{
    System: "general", // Search general knowledge
    // OR
    System: "", // Search all systems
}
```

## Benefits

1. **Knowledge Sharing**: Break patterns, best practices, and solutions shared across systems
2. **Reduced Duplication**: Don't solve the same problem multiple times
3. **Regulatory Compliance**: Centralized regulatory rules repository
4. **Pattern Recognition**: Find similar breaks across systems
5. **Best Practice Dissemination**: Share successful approaches
6. **System-Agnostic**: Works with any system, not just Murex

## Security Considerations

- **🔒 Read-Only Access**: This service is **READ-ONLY** to protect confidential information
- **🔒 Write Operations Disabled**: POST /vectorstore/store is disabled
- **🔒 No Automatic Storage**: Break patterns are NOT automatically stored
- **✅ Search Only**: Only search and list operations are available
- **✅ Public Information Only**: Can only access public information already in HANA Cloud
- **System Filtering**: Filter by system to restrict access to specific systems

## Performance

- **Vector Indexing**: Enable for fast similarity search
- **Batch Operations**: Support for bulk inserts (future)
- **Caching**: Consider caching frequently accessed information

## Future Enhancements

- [ ] Real-time synchronization
- [ ] Versioning for information updates
- [ ] Access control and permissions
- [ ] Analytics and usage statistics
- [ ] Multi-tenant support
- [ ] Backup and disaster recovery

