package vectorstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	_ "github.com/SAP/go-hdb/driver" // SAP HANA driver
)

// HANACloudVectorStore provides vector storage using SAP HANA Cloud
type HANACloudVectorStore struct {
	db     *sql.DB
	logger *log.Logger
	config *HANAConfig
}

// HANAConfig contains HANA Cloud connection configuration
type HANAConfig struct {
	ConnectionString string // HANA Cloud connection string
	Schema           string // Schema name (default: PUBLIC)
	TableName        string // Vector table name (default: PUBLIC_VECTORS)
	VectorDimension  int    // Vector dimension (default: 1536 for OpenAI embeddings)
	EnableIndexing   bool   // Enable vector indexing for performance
}

// DefaultHANAConfig returns default HANA Cloud configuration
func DefaultHANAConfig() *HANAConfig {
	return &HANAConfig{
		Schema:          "PUBLIC",
		TableName:       "PUBLIC_VECTORS",
		VectorDimension: 1536,
		EnableIndexing:  true,
	}
}

// PublicInformation represents public information stored in HANA Cloud
type PublicInformation struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`        // "break_pattern", "regulatory_rule", "best_practice", "knowledge_base"
	System      string                 `json:"system"`      // Optional: system name (e.g., "murex", "sap_fioneer") or "general"
	Category    string                 `json:"category"`   // Optional: category for organization
	Title       string                 `json:"title"`
	Content     string                 `json:"content"`    // Main content to be vectorized
	Vector      []float32              `json:"-"`          // Vector embedding (not in JSON)
	Metadata    map[string]interface{} `json:"metadata"`   // Additional metadata
	Tags        []string               `json:"tags"`       // Tags for filtering
	IsPublic    bool                   `json:"is_public"`  // Whether this is public information
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	CreatedBy   string                 `json:"created_by,omitempty"`
}

// NewHANACloudVectorStore creates a new HANA Cloud vector store
func NewHANACloudVectorStore(connectionString string, config *HANAConfig, logger *log.Logger) (*HANACloudVectorStore, error) {
	if config == nil {
		config = DefaultHANAConfig()
	}

	// Connect to HANA Cloud
	db, err := sql.Open("hdb", connectionString)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to HANA Cloud: %w", err)
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping HANA Cloud: %w", err)
	}

	store := &HANACloudVectorStore{
		db:     db,
		logger: logger,
		config: config,
	}

	// Initialize schema and tables
	if err := store.initializeSchema(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return store, nil
}

// initializeSchema creates necessary tables and indexes in HANA Cloud
func (h *HANACloudVectorStore) initializeSchema(ctx context.Context) error {
	// Create vector table if it doesn't exist
	createTableSQL := fmt.Sprintf(`
		CREATE COLUMN TABLE IF NOT EXISTS %s.%s (
			ID NVARCHAR(255) PRIMARY KEY,
			TYPE NVARCHAR(100) NOT NULL,
			SYSTEM NVARCHAR(100),
			CATEGORY NVARCHAR(100),
			TITLE NVARCHAR(500),
			CONTENT NCLOB,
			VECTOR REAL_VECTOR(%d),
			METADATA NCLOB,
			TAGS NVARCHAR(500),
			IS_PUBLIC BOOLEAN DEFAULT true,
			CREATED_AT TIMESTAMP NOT NULL,
			UPDATED_AT TIMESTAMP NOT NULL,
			CREATED_BY NVARCHAR(100)
		)
	`, h.config.Schema, h.config.TableName, h.config.VectorDimension)

	if _, err := h.db.ExecContext(ctx, createTableSQL); err != nil {
		return fmt.Errorf("failed to create vector table: %w", err)
	}

	// Create indexes for performance
	indexSQL := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS IDX_TYPE ON %s.%s (TYPE);
		CREATE INDEX IF NOT EXISTS IDX_SYSTEM ON %s.%s (SYSTEM);
		CREATE INDEX IF NOT EXISTS IDX_CATEGORY ON %s.%s (CATEGORY);
		CREATE INDEX IF NOT EXISTS IDX_IS_PUBLIC ON %s.%s (IS_PUBLIC);
		CREATE INDEX IF NOT EXISTS IDX_CREATED_AT ON %s.%s (CREATED_AT);
	`, h.config.Schema, h.config.TableName, h.config.Schema, h.config.TableName,
		h.config.Schema, h.config.TableName, h.config.Schema, h.config.TableName,
		h.config.Schema, h.config.TableName)

	if h.config.EnableIndexing {
		// Create vector index for similarity search
		vectorIndexSQL := fmt.Sprintf(`
			CREATE VECTOR INDEX IF NOT EXISTS IDX_VECTOR ON %s.%s (VECTOR)
			TYPE COSINE_SIMILARITY
		`, h.config.Schema, h.config.TableName)

		if _, err := h.db.ExecContext(ctx, vectorIndexSQL); err != nil {
			if h.logger != nil {
				h.logger.Printf("Warning: Failed to create vector index: %v", err)
			}
		}
	}

	if _, err := h.db.ExecContext(ctx, indexSQL); err != nil {
		if h.logger != nil {
			h.logger.Printf("Warning: Failed to create indexes: %v", err)
		}
	}

	if h.logger != nil {
		h.logger.Printf("HANA Cloud vector store initialized: schema=%s, table=%s", h.config.Schema, h.config.TableName)
	}

	return nil
}

// StorePublicInformation stores public information in HANA Cloud
func (h *HANACloudVectorStore) StorePublicInformation(ctx context.Context, info *PublicInformation) error {
	if info == nil {
		return fmt.Errorf("information cannot be nil")
	}

	if info.ID == "" {
		info.ID = fmt.Sprintf("info-%d", time.Now().UnixNano())
	}

	if info.Vector == nil || len(info.Vector) != h.config.VectorDimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", 
			h.config.VectorDimension, len(info.Vector))
	}

	// Serialize metadata
	metadataJSON, err := json.Marshal(info.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// Serialize tags
	tagsJSON := strings.Join(info.Tags, ",")

	// Format vector for HANA (REAL_VECTOR format)
	vectorStr := formatVectorForHANA(info.Vector)

	now := time.Now()
	if info.CreatedAt.IsZero() {
		info.CreatedAt = now
	}
	info.UpdatedAt = now

	// Insert or update
	upsertSQL := fmt.Sprintf(`
		UPSERT %s.%s (ID, TYPE, SYSTEM, CATEGORY, TITLE, CONTENT, VECTOR, METADATA, TAGS, IS_PUBLIC, CREATED_AT, UPDATED_AT, CREATED_BY)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, h.config.Schema, h.config.TableName)

	_, err = h.db.ExecContext(ctx, upsertSQL,
		info.ID,
		info.Type,
		info.System,
		info.Category,
		info.Title,
		info.Content,
		vectorStr,
		string(metadataJSON),
		tagsJSON,
		info.IsPublic,
		info.CreatedAt,
		info.UpdatedAt,
		info.CreatedBy,
	)

	if err != nil {
		return fmt.Errorf("failed to store information: %w", err)
	}

	if h.logger != nil {
		h.logger.Printf("Stored public information: id=%s, type=%s, system=%s", info.ID, info.Type, info.System)
	}

	return nil
}

// SearchPublicInformation performs semantic search on public information
func (h *HANACloudVectorStore) SearchPublicInformation(ctx context.Context, 
	queryVector []float32,
	options *SearchOptions) ([]*PublicInformation, error) {

	if options == nil {
		options = &SearchOptions{}
	}

	if len(queryVector) != h.config.VectorDimension {
		return nil, fmt.Errorf("query vector dimension mismatch: expected %d, got %d",
			h.config.VectorDimension, len(queryVector))
	}

	// Build search query
	query := h.buildSearchQuery(queryVector, options)

	rows, err := h.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to search information: %w", err)
	}
	defer rows.Close()

	var results []*PublicInformation
	for rows.Next() {
		var info PublicInformation
		var vectorStr string
		var metadataJSON sql.NullString
		var tagsJSON sql.NullString
		var similarity sql.NullFloat64

		err := rows.Scan(
			&info.ID,
			&info.Type,
			&info.System,
			&info.Category,
			&info.Title,
			&info.Content,
			&vectorStr,
			&metadataJSON,
			&tagsJSON,
			&info.IsPublic,
			&info.CreatedAt,
			&info.UpdatedAt,
			&info.CreatedBy,
			&similarity,
		)
		if err != nil {
			if h.logger != nil {
				h.logger.Printf("Warning: Failed to scan result: %v", err)
			}
			continue
		}

		// Parse vector
		info.Vector = parseVectorFromHANA(vectorStr)

		// Parse metadata
		if metadataJSON.Valid {
			if err := json.Unmarshal([]byte(metadataJSON.String), &info.Metadata); err != nil {
				if h.logger != nil {
					h.logger.Printf("Warning: Failed to unmarshal metadata: %v", err)
				}
			}
		}

		// Parse tags
		if tagsJSON.Valid {
			info.Tags = strings.Split(tagsJSON.String, ",")
		}

		results = append(results, &info)
	}

	if h.logger != nil {
		h.logger.Printf("Search completed: found %d results", len(results))
	}

	return results, nil
}

// SearchOptions defines search options
type SearchOptions struct {
	Type        string   // Filter by type
	System      string   // Filter by system (or "general" for public)
	Category    string   // Filter by category
	Tags        []string // Filter by tags
	IsPublic    *bool    // Filter by public status (nil = all)
	Limit       int      // Max results (default: 10)
	Threshold   float64  // Minimum similarity threshold (default: 0.7)
	MinDate     *time.Time // Minimum creation date
	MaxDate     *time.Time // Maximum creation date
}

// buildSearchQuery builds a HANA SQL query for vector similarity search
func (h *HANACloudVectorStore) buildSearchQuery(queryVector []float32, options *SearchOptions) string {
	vectorStr := formatVectorForHANA(queryVector)

	// Set default limit
	if options.Limit <= 0 {
		options.Limit = 10
	}
	if options.Threshold <= 0 {
		options.Threshold = 0.7
	}

	// Start with similarity search
	// HANA Cloud uses COSINE_SIMILARITY function for vector similarity
	query := fmt.Sprintf(`
		SELECT TOP %d
			ID, TYPE, SYSTEM, CATEGORY, TITLE, CONTENT, VECTOR, METADATA, TAGS,
			IS_PUBLIC, CREATED_AT, UPDATED_AT, CREATED_BY,
			COSINE_SIMILARITY(VECTOR, %s) AS SIMILARITY
		FROM %s.%s
		WHERE COSINE_SIMILARITY(VECTOR, %s) >= %f
	`, options.Limit, vectorStr, h.config.Schema, h.config.TableName, vectorStr, options.Threshold)

	// Add filters
	var filters []string

	if options.Type != "" {
		filters = append(filters, fmt.Sprintf("TYPE = '%s'", escapeSQL(options.Type)))
	}

	if options.System != "" {
		filters = append(filters, fmt.Sprintf("SYSTEM = '%s'", escapeSQL(options.System)))
	}

	if options.Category != "" {
		filters = append(filters, fmt.Sprintf("CATEGORY = '%s'", escapeSQL(options.Category)))
	}

	if options.IsPublic != nil {
		if *options.IsPublic {
			filters = append(filters, "IS_PUBLIC = true")
		} else {
			filters = append(filters, "IS_PUBLIC = false")
		}
	}

	if len(options.Tags) > 0 {
		tagConditions := make([]string, len(options.Tags))
		for i, tag := range options.Tags {
			tagConditions[i] = fmt.Sprintf("TAGS LIKE '%%%s%%'", escapeSQL(tag))
		}
		filters = append(filters, fmt.Sprintf("(%s)", strings.Join(tagConditions, " OR ")))
	}

	if options.MinDate != nil {
		filters = append(filters, fmt.Sprintf("CREATED_AT >= '%s'", options.MinDate.Format("2006-01-02 15:04:05")))
	}

	if options.MaxDate != nil {
		filters = append(filters, fmt.Sprintf("CREATED_AT <= '%s'", options.MaxDate.Format("2006-01-02 15:04:05")))
	}

	if len(filters) > 0 {
		query += " AND " + strings.Join(filters, " AND ")
	}

	// Order by similarity
	query += " ORDER BY SIMILARITY DESC"

	return query
}

// GetPublicInformation retrieves public information by ID
func (h *HANACloudVectorStore) GetPublicInformation(ctx context.Context, id string) (*PublicInformation, error) {
	query := fmt.Sprintf(`
		SELECT ID, TYPE, SYSTEM, CATEGORY, TITLE, CONTENT, VECTOR, METADATA, TAGS,
		       IS_PUBLIC, CREATED_AT, UPDATED_AT, CREATED_BY
		FROM %s.%s
		WHERE ID = ?
	`, h.config.Schema, h.config.TableName)

	var info PublicInformation
	var vectorStr string
	var metadataJSON sql.NullString
	var tagsJSON sql.NullString

	err := h.db.QueryRowContext(ctx, query, id).Scan(
		&info.ID,
		&info.Type,
		&info.System,
		&info.Category,
		&info.Title,
		&info.Content,
		&vectorStr,
		&metadataJSON,
		&tagsJSON,
		&info.IsPublic,
		&info.CreatedAt,
		&info.UpdatedAt,
		&info.CreatedBy,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("information not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get information: %w", err)
	}

	// Parse vector
	info.Vector = parseVectorFromHANA(vectorStr)

	// Parse metadata
	if metadataJSON.Valid {
		if err := json.Unmarshal([]byte(metadataJSON.String), &info.Metadata); err != nil {
			if h.logger != nil {
				h.logger.Printf("Warning: Failed to unmarshal metadata: %v", err)
			}
		}
	}

	// Parse tags
	if tagsJSON.Valid {
		info.Tags = strings.Split(tagsJSON.String, ",")
	}

	return &info, nil
}

// DeletePublicInformation deletes public information by ID
func (h *HANACloudVectorStore) DeletePublicInformation(ctx context.Context, id string) error {
	query := fmt.Sprintf(`
		DELETE FROM %s.%s WHERE ID = ?
	`, h.config.Schema, h.config.TableName)

	result, err := h.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete information: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("information not found: %s", id)
	}

	if h.logger != nil {
		h.logger.Printf("Deleted public information: id=%s", id)
	}

	return nil
}

// ListPublicInformation lists public information with filters
func (h *HANACloudVectorStore) ListPublicInformation(ctx context.Context, 
	options *ListOptions) ([]*PublicInformation, error) {

	if options == nil {
		options = &ListOptions{}
	}

	query := h.buildListQuery(options)

	rows, err := h.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list information: %w", err)
	}
	defer rows.Close()

	var results []*PublicInformation
	for rows.Next() {
		var info PublicInformation
		var vectorStr string
		var metadataJSON sql.NullString
		var tagsJSON sql.NullString

		err := rows.Scan(
			&info.ID,
			&info.Type,
			&info.System,
			&info.Category,
			&info.Title,
			&info.Content,
			&vectorStr,
			&metadataJSON,
			&tagsJSON,
			&info.IsPublic,
			&info.CreatedAt,
			&info.UpdatedAt,
			&info.CreatedBy,
		)
		if err != nil {
			if h.logger != nil {
				h.logger.Printf("Warning: Failed to scan result: %v", err)
			}
			continue
		}

		// Parse metadata and tags (skip vector for list)
		if metadataJSON.Valid {
			json.Unmarshal([]byte(metadataJSON.String), &info.Metadata)
		}
		if tagsJSON.Valid {
			info.Tags = strings.Split(tagsJSON.String, ",")
		}

		results = append(results, &info)
	}

	return results, nil
}

// ListOptions defines list options
type ListOptions struct {
	Type      string
	System    string
	Category  string
	Tags      []string
	IsPublic  *bool
	Limit     int
	Offset    int
	OrderBy   string // "created_at", "updated_at", "title"
	OrderDesc bool
}

func (h *HANACloudVectorStore) buildListQuery(options *ListOptions) string {
	query := fmt.Sprintf(`
		SELECT ID, TYPE, SYSTEM, CATEGORY, TITLE, CONTENT, VECTOR, METADATA, TAGS,
		       IS_PUBLIC, CREATED_AT, UPDATED_AT, CREATED_BY
		FROM %s.%s
	`, h.config.Schema, h.config.TableName)

	var filters []string

	if options.Type != "" {
		filters = append(filters, fmt.Sprintf("TYPE = '%s'", escapeSQL(options.Type)))
	}
	if options.System != "" {
		filters = append(filters, fmt.Sprintf("SYSTEM = '%s'", escapeSQL(options.System)))
	}
	if options.Category != "" {
		filters = append(filters, fmt.Sprintf("CATEGORY = '%s'", escapeSQL(options.Category)))
	}
	if options.IsPublic != nil {
		if *options.IsPublic {
			filters = append(filters, "IS_PUBLIC = true")
		} else {
			filters = append(filters, "IS_PUBLIC = false")
		}
	}

	if len(filters) > 0 {
		query += " WHERE " + strings.Join(filters, " AND ")
	}

	// Order by
	orderBy := "CREATED_AT"
	if options.OrderBy != "" {
		orderBy = strings.ToUpper(options.OrderBy)
	}
	orderDir := "ASC"
	if options.OrderDesc {
		orderDir = "DESC"
	}
	query += fmt.Sprintf(" ORDER BY %s %s", orderBy, orderDir)

	// Limit and offset
	if options.Limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", options.Limit)
	}
	if options.Offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", options.Offset)
	}

	return query
}

// Close closes the database connection
func (h *HANACloudVectorStore) Close() error {
	if h.db != nil {
		return h.db.Close()
	}
	return nil
}

// Helper functions

func formatVectorForHANA(vector []float32) string {
	// HANA REAL_VECTOR format: [1.0, 2.0, 3.0, ...]
	parts := make([]string, len(vector))
	for i, v := range vector {
		parts[i] = fmt.Sprintf("%.6f", v)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func parseVectorFromHANA(vectorStr string) []float32 {
	// Remove brackets and parse
	vectorStr = strings.Trim(vectorStr, "[]")
	parts := strings.Split(vectorStr, ",")
	vector := make([]float32, len(parts))
	for i, part := range parts {
		var v float32
		fmt.Sscanf(strings.TrimSpace(part), "%f", &v)
		vector[i] = v
	}
	return vector
}

func escapeSQL(s string) string {
	// Basic SQL injection prevention
	return strings.ReplaceAll(strings.ReplaceAll(s, "'", "''"), ";", "")
}

