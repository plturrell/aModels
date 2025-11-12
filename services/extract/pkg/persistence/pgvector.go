package persistence

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// PgVectorPersistence is the persistence layer for pgvector.
type PgVectorPersistence struct {
	dsn    string
	logger *log.Logger
	mu     sync.Mutex
	db     *sql.DB
}

// NewPgVectorPersistence creates a new pgvector persistence layer.
func NewPgVectorPersistence(dsn string, logger *log.Logger) (*PgVectorPersistence, error) {
	p := &PgVectorPersistence{
		dsn:    dsn,
		logger: logger,
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open postgres connection: %w", err)
	}

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping postgres: %w", err)
	}

	p.db = db

	// Ensure vector table exists
	if err := p.EnsureVectorTable(ctx); err != nil {
		p.db.Close()
		return nil, fmt.Errorf("failed to ensure vector table: %w", err)
	}

	return p, nil
}

// EnsureVectorTable creates the embeddings table with pgvector extension if it doesn't exist.
func (p *PgVectorPersistence) EnsureVectorTable(ctx context.Context) error {
	queries := []string{
		// Enable pgvector extension
		"CREATE EXTENSION IF NOT EXISTS vector",
		// Create embeddings table
		`CREATE TABLE IF NOT EXISTS embeddings (
			id TEXT PRIMARY KEY,
			artifact_type TEXT NOT NULL,
			artifact_id TEXT NOT NULL,
			embedding vector(768),
			metadata JSONB,
			created_at TIMESTAMP DEFAULT NOW(),
			updated_at TIMESTAMP DEFAULT NOW()
		)`,
		// Create index on artifact_type for filtering
		"CREATE INDEX IF NOT EXISTS idx_embeddings_artifact_type ON embeddings(artifact_type)",
		// Create index on artifact_id for lookups
		"CREATE INDEX IF NOT EXISTS idx_embeddings_artifact_id ON embeddings(artifact_id)",
		// Create vector index for similarity search (IVFFlat with cosine distance)
		// Note: IVFFlat requires some data to be present, so we create it with IF NOT EXISTS
		"CREATE INDEX IF NOT EXISTS embedding_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
	}

	for _, query := range queries {
		if _, err := p.db.ExecContext(ctx, query); err != nil {
			// Ignore error for index creation if it already exists or table is empty
			if strings.Contains(err.Error(), "must have at least") {
				// IVFFlat index requires data - will be created later
				p.logger.Printf("Note: IVFFlat index will be created when data is available")
				continue
			}
			if !strings.Contains(err.Error(), "already exists") {
				return fmt.Errorf("failed to execute query: %w", err)
			}
		}
	}

	return nil
}

// SaveVector saves a vector to pgvector with metadata.
func (p *PgVectorPersistence) SaveVector(key string, vector []float32, metadata map[string]any) error {
	ctx := context.Background()

	// Extract artifact type and ID from metadata
	artifactType := ""
	artifactID := key
	if metadata != nil {
		if at, ok := metadata["artifact_type"].(string); ok {
			artifactType = at
		}
		if aid, ok := metadata["artifact_id"].(string); ok {
			artifactID = aid
		}
	}

	// Convert vector to PostgreSQL vector format
	// pgvector expects format: [1.0,2.0,3.0]
	vectorStr := formatVectorForPostgres(vector)

	// Serialize metadata to JSON
	metadataJSON := "{}"
	if metadata != nil {
		metadataBytes, err := json.Marshal(metadata)
		if err != nil {
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}
		metadataJSON = string(metadataBytes)
	}

	// Upsert vector
	query := `
		INSERT INTO embeddings (id, artifact_type, artifact_id, embedding, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4::vector, $5::jsonb, NOW(), NOW())
		ON CONFLICT (id) DO UPDATE SET
			embedding = $4::vector,
			metadata = $5::jsonb,
			updated_at = NOW()
	`

	_, err := p.db.ExecContext(ctx, query, key, artifactType, artifactID, vectorStr, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to save vector: %w", err)
	}

	return nil
}

// GetVector retrieves a vector and metadata from pgvector.
func (p *PgVectorPersistence) GetVector(key string) ([]float32, map[string]any, error) {
	ctx := context.Background()

	query := `
		SELECT embedding, metadata
		FROM embeddings
		WHERE id = $1
	`

	var vectorStr string
	var metadataJSON sql.NullString

	err := p.db.QueryRowContext(ctx, query, key).Scan(&vectorStr, &metadataJSON)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil, fmt.Errorf("vector not found: %s", key)
		}
		return nil, nil, fmt.Errorf("failed to get vector: %w", err)
	}

	// Parse vector from PostgreSQL format
	vector := parseVectorFromPostgres(vectorStr)

	// Parse metadata
	metadata := make(map[string]any)
	if metadataJSON.Valid && metadataJSON.String != "" {
		if err := json.Unmarshal([]byte(metadataJSON.String), &metadata); err != nil {
			// Non-fatal: return empty metadata if unmarshal fails
			metadata = make(map[string]any)
		}
	}

	return vector, metadata, nil
}

// SearchSimilar performs similarity search using pgvector.
func (p *PgVectorPersistence) SearchSimilar(queryVector []float32, artifactType string, limit int, threshold float32) ([]VectorSearchResult, error) {
	ctx := context.Background()

	// Convert vector to PostgreSQL format
	vectorStr := formatVectorForPostgres(queryVector)

	// Build query with optional artifact type filter
	query := `
		SELECT id, artifact_type, artifact_id, metadata, 
			   1 - (embedding <=> $1::vector) AS similarity
		FROM embeddings
		WHERE 1 - (embedding <=> $1::vector) >= $3
	`
	args := []any{vectorStr, limit, threshold}

	if artifactType != "" {
		query += " AND artifact_type = $4"
		args = append(args, artifactType)
		// Adjust threshold parameter position
		query = strings.Replace(query, "$3", "$3", 1)
	}

	query += " ORDER BY embedding <=> $1::vector LIMIT $2"

	rows, err := p.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to search vectors: %w", err)
	}
	defer rows.Close()

	results := []VectorSearchResult{}
	for rows.Next() {
		var id, at, aid string
		var metadataJSON sql.NullString
		var similarity float64

		if err := rows.Scan(&id, &at, &aid, &metadataJSON, &similarity); err != nil {
			continue
		}

		// Get full vector (for completeness)
		vector, metadata, err := p.GetVector(id)
		if err != nil {
			continue
		}

		text := ""
		if metadata != nil {
			if t, ok := metadata["label"].(string); ok {
				text = t
			}
		}

		results = append(results, VectorSearchResult{
			Key:          id,
			ArtifactType: at,
			ArtifactID:   aid,
			Vector:       vector,
			Metadata:     metadata,
			Score:        float32(similarity),
			Text:         text,
		})
	}

	return results, rows.Err()
}

// SearchByText performs text-based search using metadata JSONB queries.
func (p *PgVectorPersistence) SearchByText(query string, artifactType string, limit int) ([]VectorSearchResult, error) {
	ctx := context.Background()

	// Build query with JSONB text search
	searchQuery := `
		SELECT id, artifact_type, artifact_id, metadata
		FROM embeddings
		WHERE metadata::text ILIKE $1
	`
	args := []any{"%" + query + "%"}

	if artifactType != "" {
		searchQuery += " AND artifact_type = $2"
		args = append(args, artifactType)
	}

	searchQuery += " LIMIT $3"
	if artifactType != "" {
		args = append(args, limit)
	} else {
		args = []any{"%" + query + "%", limit}
	}

	rows, err := p.db.QueryContext(ctx, searchQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to search by text: %w", err)
	}
	defer rows.Close()

	results := []VectorSearchResult{}
	for rows.Next() {
		var id, at, aid string
		var metadataJSON sql.NullString

		if err := rows.Scan(&id, &at, &aid, &metadataJSON); err != nil {
			continue
		}

		// Get full vector
		vector, metadata, err := p.GetVector(id)
		if err != nil {
			continue
		}

		text := ""
		if metadata != nil {
			if t, ok := metadata["label"].(string); ok {
				text = t
			}
		}

		results = append(results, VectorSearchResult{
			Key:          id,
			ArtifactType: at,
			ArtifactID:   aid,
			Vector:       vector,
			Metadata:     metadata,
			Score:        1.0, // Text match doesn't have similarity score
			Text:         text,
		})
	}

	return results, rows.Err()
}

// Close closes the database connection.
func (p *PgVectorPersistence) Close() error {
	if p.db != nil {
		return p.db.Close()
	}
	return nil
}

// formatVectorForPostgres converts a []float32 to PostgreSQL vector format.
func formatVectorForPostgres(vector []float32) string {
	if len(vector) == 0 {
		return "[]"
	}

	parts := make([]string, len(vector))
	for i, v := range vector {
		parts[i] = fmt.Sprintf("%.6f", v)
	}

	return "[" + strings.Join(parts, ",") + "]"
}

// parseVectorFromPostgres parses a PostgreSQL vector string to []float32.
func parseVectorFromPostgres(vectorStr string) []float32 {
	// Remove brackets
	vectorStr = strings.Trim(vectorStr, "[]")
	if vectorStr == "" {
		return []float32{}
	}

	// Split by comma
	parts := strings.Split(vectorStr, ",")
	vector := make([]float32, len(parts))

	for i, part := range parts {
		var v float64
		fmt.Sscanf(strings.TrimSpace(part), "%f", &v)
		vector[i] = float32(v)
	}

	return vector
}

