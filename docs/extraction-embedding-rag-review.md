# Extraction Process: Embedding and RAG Review

## Current State

### ✅ Embedding Generation (Partially Implemented)

**What exists:**
- **SQL Query Embeddings**: The extraction process generates embeddings for SQL queries using a Relational Transformer model
- **Implementation**: `services/extract/embedding.go` calls `scripts/embed_sql.py`
- **Model**: Uses `RelationalTransformer` from the training project
- **Embedding Type**: Multi-modal embeddings (value, schema, temporal, role embeddings)
- **Trigger**: Automatically generated for each SQL query during extraction

**Code Location:**
```go
// In main.go handleGraph function (lines 801-815)
embedding, err := generateEmbedding(ctx, sql)
if err != nil {
    s.logger.Printf("failed to generate embedding for sql query %q: %v", sql, err)
    continue
}

if s.vectorPersistence != nil {
    h := sha256.New()
    h.Write([]byte(sql))
    key := fmt.Sprintf("sql:%x", h.Sum(nil))
    
    if err := s.vectorPersistence.SaveVector(key, embedding); err != nil {
        s.logger.Printf("failed to save vector for sql query %q: %v", sql, err)
    }
}
```

### ✅ Vector Storage (Partially Implemented)

**What exists:**
- **Redis Vector Storage**: Embeds are stored in Redis using `RedisPersistence.SaveVector()`
- **Storage Format**: Vectors are serialized to JSON and stored in Redis hash sets
- **Key Format**: `sql:{sha256_hash}` where hash is derived from SQL query text

**Code Location:**
```go
// In redis.go
func (p *RedisPersistence) SaveVector(key string, vector []float32) error {
    jsonVector, err := json.Marshal(vector)
    if err != nil {
        return fmt.Errorf("failed to marshal vector: %w", err)
    }
    
    err = p.client.HSet(context.Background(), key, "vector", jsonVector).Err()
    if err != nil {
        return fmt.Errorf("failed to save vector: %w", err)
    }
    
    return nil
}
```

### ❌ RAG/Search Capabilities (Missing)

**What's missing:**
1. **Vector Retrieval**: No methods to retrieve embeddings from Redis
2. **Similarity Search**: No cosine similarity or distance-based search
3. **RAG API**: No endpoint to search embeddings and retrieve similar SQL queries
4. **Integration**: No integration with pgvector (Postgres) or OpenSearch (both configured but not used)
5. **Metadata Storage**: Embeddings stored without metadata (table names, column lineage, etc.)

### ❌ Expanded Embedding Coverage (Missing)

**What's missing:**
- **Table Embeddings**: No embeddings generated for table schemas
- **Column Embeddings**: No embeddings for column definitions or metadata
- **Control-M Job Embeddings**: No embeddings for job definitions
- **Process Sequence Embeddings**: No embeddings for table process sequences
- **Workflow Embeddings**: No embeddings for Petri nets or workflows

## Current Limitations

### 1. SQL-Only Embeddings
- Only SQL queries are embedded
- Tables, columns, jobs, and workflows are not embedded
- This limits semantic search capabilities

### 2. No Retrieval API
- Vectors are stored but cannot be searched
- No similarity search endpoint
- No RAG functionality for querying ETL metadata

### 3. Basic Storage
- Redis is used as a key-value store, not optimized for vector search
- No vector indexing (no HNSW, IVF, or similar)
- No support for approximate nearest neighbor (ANN) search

### 4. Missing Metadata
- Embeddings stored without associated metadata
- No link between embeddings and graph nodes
- Cannot retrieve context about what an embedding represents

## Infrastructure Available (Not Integrated)

### Postgres with pgvector
- **Status**: Configured in `docker/compose.yml`
- **Capability**: Native vector similarity search with pgvector extension
- **Not Used**: Extract service doesn't use pgvector for vector storage

### OpenSearch
- **Status**: Configured and accessible via gateway
- **Capability**: Vector search, semantic search, hybrid search
- **Not Used**: Extract service doesn't use OpenSearch for vector storage

## Recommended Enhancements

### Priority 1: Vector Retrieval and Search

**Add to `redis.go` or create new `vector_search.go`:**
```go
// VectorPersistence interface extension
type VectorPersistence interface {
    SaveVector(key string, vector []float32, metadata map[string]any) error
    GetVector(key string) ([]float32, map[string]any, error)
    SearchSimilar(queryVector []float32, limit int, threshold float32) ([]VectorSearchResult, error)
}

type VectorSearchResult struct {
    Key      string
    Vector   []float32
    Metadata map[string]any
    Score    float32 // Similarity score
}
```

**Implementation Options:**
1. **Redis with RediSearch**: Add RediSearch module for vector similarity search
2. **Postgres pgvector**: Migrate to pgvector for native vector search
3. **OpenSearch**: Use OpenSearch for semantic and hybrid search

### Priority 2: RAG API Endpoints

**Add to `main.go`:**
```go
// POST /knowledge-graph/search
// Search for similar SQL queries, tables, or workflows
func (s *extractServer) handleVectorSearch(w http.ResponseWriter, r *http.Request)

// POST /knowledge-graph/embed
// Generate embeddings for arbitrary text (SQL, table names, etc.)
func (s *extractServer) handleGenerateEmbedding(w http.ResponseWriter, r *http.Request)

// GET /knowledge-graph/embed/{key}
// Retrieve embedding and metadata by key
func (s *extractServer) handleGetEmbedding(w http.ResponseWriter, r *http.Request)
```

### Priority 3: Expand Embedding Coverage

**Generate embeddings for:**
1. **Table Schemas**: Embed table definitions with column metadata
2. **Column Definitions**: Embed column names, types, and constraints
3. **Control-M Jobs**: Embed job definitions, commands, and dependencies
4. **Process Sequences**: Embed table processing sequences
5. **Petri Nets**: Embed workflow representations

**Code Location to Add:**
```go
// In handleGraph function, after normalization:
// Generate embeddings for tables
for _, node := range nodes {
    if node.Type == "table" {
        embedding, err := generateTableEmbedding(ctx, node)
        // Store with metadata linking to node
    }
}

// Generate embeddings for Control-M jobs
for _, job := range allControlMJobs {
    embedding, err := generateJobEmbedding(ctx, job)
    // Store with metadata linking to job node
}
```

### Priority 4: Integration with pgvector

**Benefits:**
- Native vector similarity search
- SQL-based queries with vector operations
- Integration with existing Postgres schema replication
- Better performance for large-scale vector search

**Implementation:**
```go
// Create pgvector persistence layer
type PgVectorPersistence struct {
    db *sql.DB
}

func (p *PgVectorPersistence) CreateVectorTable() error {
    // CREATE TABLE IF NOT EXISTS embeddings (
    //     id TEXT PRIMARY KEY,
    //     embedding vector(768),
    //     metadata JSONB,
    //     created_at TIMESTAMP
    // )
}

func (p *PgVectorPersistence) SearchSimilar(queryVector []float32, limit int) ([]VectorSearchResult, error) {
    // SELECT id, metadata, embedding <-> $1::vector AS distance
    // FROM embeddings
    // ORDER BY distance
    // LIMIT $2
}
```

### Priority 5: Integration with OpenSearch

**Benefits:**
- Semantic search capabilities
- Hybrid search (keyword + vector)
- Full-text search with vector similarity
- Better for complex queries

**Implementation:**
```go
// Create OpenSearch persistence layer
type OpenSearchPersistence struct {
    client *http.Client
    url    string
}

func (p *OpenSearchPersistence) IndexVector(key string, vector []float32, metadata map[string]any) error {
    // POST /embeddings/_doc/{key}
    // {
    //   "vector": [...],
    //   "metadata": {...},
    //   "text": metadata["sql"] or metadata["table_name"]
    // }
}

func (p *OpenSearchPersistence) SearchSimilar(queryVector []float32, query string, limit int) ([]VectorSearchResult, error) {
    // POST /embeddings/_search
    // {
    //   "query": {
    //     "hybrid": {
    //       "queries": [
    //         {"knn": {"vector": {...}}},
    //         {"match": {"text": query}}
    //       ]
    //     }
    //   }
    // }
}
```

## Rating: 45/100

### Breakdown
- **Embedding Generation**: 60/100 - Only SQL queries, limited coverage
- **Vector Storage**: 50/100 - Basic Redis storage, no indexing
- **Retrieval/Search**: 0/100 - No search capabilities
- **RAG Integration**: 0/100 - No RAG API or endpoints
- **Infrastructure Utilization**: 30/100 - pgvector and OpenSearch available but not used
- **Metadata Association**: 40/100 - Embeddings stored without rich metadata

## Current Workflow

```
SQL Query → generateEmbedding() → RelationalTransformer → Embedding Vector
    ↓
SHA256 Hash → Key: "sql:{hash}"
    ↓
Redis HSet(key, "vector", JSON(vector))
    ↓
[STOP - No retrieval/search available]
```

## Target Workflow

```
SQL Query/Table/Job → generateEmbedding() → Embedding Vector
    ↓
Store with metadata (table names, lineage, etc.)
    ↓
pgvector/OpenSearch with vector indexing
    ↓
RAG API: POST /knowledge-graph/search
    ↓
Similarity Search → Retrieve similar queries/tables
    ↓
Return results with metadata and similarity scores
```

## Conclusion

The extraction process has **basic embedding generation and storage** but lacks **retrieval, search, and RAG capabilities**. The infrastructure (pgvector, OpenSearch) is available but not integrated. To achieve a 10/10 embedding and RAG system, we need to:

1. ✅ Expand embedding coverage beyond SQL queries
2. ✅ Add vector retrieval and similarity search
3. ✅ Create RAG API endpoints
4. ✅ Integrate with pgvector or OpenSearch for production-grade vector search
5. ✅ Associate embeddings with rich metadata from the knowledge graph

**Recommendation**: Implement Priority 1-3 first to enable basic RAG functionality, then add Priority 4-5 for production-grade vector search.

