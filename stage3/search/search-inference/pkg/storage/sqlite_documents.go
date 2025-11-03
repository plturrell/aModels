//go:build !hana

package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
	"time"
)

// SQLiteDocumentStore provides document storage using SQLite (replaces HANA)
type SQLiteDocumentStore struct {
	db            *sql.DB
	privacyConfig *PrivacyConfig
	dbPath        string
}

// NewSQLiteDocumentStore creates a new SQLite document store
func NewSQLiteDocumentStore(dsn string, privacyConfig *PrivacyConfig) (*SQLiteDocumentStore, error) {
	if dsn == "" {
		dsn = "file:documents.db?cache=shared&mode=rwc"
	}
	
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to SQLite: %w", err)
	}

	store := &SQLiteDocumentStore{
		db:            db,
		privacyConfig: privacyConfig,
		dbPath:        dsn,
	}

	// Create tables if they don't exist
	if err := store.createTables(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to create tables: %w", err)
	}

	return store, nil
}

// createTables creates the necessary SQLite tables
func (s *SQLiteDocumentStore) createTables(ctx context.Context) error {
	createDocuments := `
CREATE TABLE IF NOT EXISTS documents (
	id TEXT PRIMARY KEY,
	content TEXT,
	metadata TEXT,
	privacy_level TEXT DEFAULT 'medium',
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	access_count INTEGER DEFAULT 0,
	last_accessed TIMESTAMP
)`
	if _, err := s.db.ExecContext(ctx, createDocuments); err != nil {
		return fmt.Errorf("failed to create documents table: %w", err)
	}

	createVersions := `
CREATE TABLE IF NOT EXISTS document_versions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	document_id TEXT NOT NULL,
	content TEXT,
	metadata TEXT,
	privacy_level TEXT,
	version_number INTEGER DEFAULT 1,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	created_by TEXT,
	change_reason TEXT
)`
	if _, err := s.db.ExecContext(ctx, createVersions); err != nil {
		return fmt.Errorf("failed to create document_versions table: %w", err)
	}

	createAccess := `
CREATE TABLE IF NOT EXISTS document_access_log (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	document_id TEXT NOT NULL,
	user_id_hash TEXT,
	session_id TEXT,
	access_type TEXT,
	privacy_budget_used REAL DEFAULT 0,
	timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	ip_hash TEXT
)`
	if _, err := s.db.ExecContext(ctx, createAccess); err != nil {
		return fmt.Errorf("failed to create document_access_log table: %w", err)
	}

	indexes := []string{
		"CREATE INDEX IF NOT EXISTS idx_documents_privacy_level ON documents (privacy_level)",
		"CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at)",
		"CREATE INDEX IF NOT EXISTS idx_document_versions_doc_id ON document_versions (document_id)",
		"CREATE INDEX IF NOT EXISTS idx_document_access_doc_id ON document_access_log (document_id)",
	}

	for _, stmt := range indexes {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			fmt.Printf("⚠️  Index creation failed: %v\n", err)
		}
	}

	return nil
}

// StoreDocument stores a document with privacy controls
func (s *SQLiteDocumentStore) StoreDocument(ctx context.Context, doc *Document) error {
	cost := PrivacyBudgetCosts.DocumentAdd
	if !s.privacyConfig.CanPerformOperation(cost) {
		return fmt.Errorf("privacy budget would be exceeded")
	}

	metadataJSON, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to serialize metadata: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT OR REPLACE INTO documents (id, content, metadata, privacy_level, created_at, updated_at, access_count, last_accessed)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, doc.ID, doc.Content, string(metadataJSON), doc.PrivacyLevel, doc.CreatedAt, doc.UpdatedAt, doc.AccessCount, doc.LastAccessed)

	if err != nil {
		return fmt.Errorf("failed to store document: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO document_versions (document_id, content, metadata, privacy_level, created_by, change_reason)
		VALUES (?, ?, ?, ?, ?, ?)
	`, doc.ID, doc.Content, string(metadataJSON), doc.PrivacyLevel, "system", "document_created")

	if err != nil {
		return fmt.Errorf("failed to store document version: %w", err)
	}

	s.privacyConfig.ConsumeBudget(cost)
	return nil
}

// GetDocument retrieves a document by ID
func (s *SQLiteDocumentStore) GetDocument(ctx context.Context, docID string, userID, sessionID string) (*Document, error) {
	cost := PrivacyBudgetCosts.DocumentAdd * 0.1
	if !s.privacyConfig.CanPerformOperation(cost) {
		return nil, fmt.Errorf("privacy budget would be exceeded")
	}

	row := s.db.QueryRowContext(ctx, `
		SELECT id, content, metadata, privacy_level, created_at, updated_at, access_count, last_accessed
		FROM documents
		WHERE id = ?
	`, docID)

	var doc Document
	var metadataJSON string
	var lastAccessed sql.NullTime

	err := row.Scan(&doc.ID, &doc.Content, &metadataJSON, &doc.PrivacyLevel, &doc.CreatedAt, &doc.UpdatedAt, &doc.AccessCount, &lastAccessed)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("document not found: %s", docID)
		}
		return nil, fmt.Errorf("failed to get document: %w", err)
	}

	if metadataJSON != "" {
		_ = json.Unmarshal([]byte(metadataJSON), &doc.Metadata)
	}

	if lastAccessed.Valid {
		doc.LastAccessed = lastAccessed.Time
	}

	_, err = s.db.ExecContext(ctx, `
		UPDATE documents 
		SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
		WHERE id = ?
	`, docID)
	if err != nil {
		fmt.Printf("Warning: failed to update access count: %v\n", err)
	}

	s.logDocumentAccess(ctx, docID, userID, sessionID, "read", cost)
	s.privacyConfig.ConsumeBudget(cost)

	return &doc, nil
}

// UpdateDocument updates a document
func (s *SQLiteDocumentStore) UpdateDocument(ctx context.Context, doc *Document, userID, changeReason string) error {
	cost := PrivacyBudgetCosts.DocumentAdd
	if !s.privacyConfig.CanPerformOperation(cost) {
		return fmt.Errorf("privacy budget would be exceeded")
	}

	metadataJSON, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to serialize metadata: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		UPDATE documents 
		SET content = ?, metadata = ?, privacy_level = ?, updated_at = CURRENT_TIMESTAMP
		WHERE id = ?
	`, doc.Content, string(metadataJSON), doc.PrivacyLevel, doc.ID)

	if err != nil {
		return fmt.Errorf("failed to update document: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO document_versions (document_id, content, metadata, privacy_level, created_by, change_reason)
		VALUES (?, ?, ?, ?, ?, ?)
	`, doc.ID, doc.Content, string(metadataJSON), doc.PrivacyLevel, userID, changeReason)

	if err != nil {
		return fmt.Errorf("failed to store document version: %w", err)
	}

	s.logDocumentAccess(ctx, doc.ID, userID, "", "update", cost)
	s.privacyConfig.ConsumeBudget(cost)

	return nil
}

// DeleteDocument deletes a document
func (s *SQLiteDocumentStore) DeleteDocument(ctx context.Context, docID string, userID string) error {
	cost := PrivacyBudgetCosts.DocumentAdd * 0.5
	if !s.privacyConfig.CanPerformOperation(cost) {
		return fmt.Errorf("privacy budget would be exceeded")
	}

	_, err := s.db.ExecContext(ctx, `
		UPDATE documents 
		SET privacy_level = 'deleted', updated_at = CURRENT_TIMESTAMP
		WHERE id = ?
	`, docID)

	if err != nil {
		return fmt.Errorf("failed to delete document: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO document_versions (document_id, content, metadata, privacy_level, created_by, change_reason)
		VALUES (?, '', '{}', 'deleted', ?, 'document_deleted')
	`, docID, userID)

	if err != nil {
		return fmt.Errorf("failed to store document version: %w", err)
	}

	s.logDocumentAccess(ctx, docID, userID, "", "delete", cost)
	s.privacyConfig.ConsumeBudget(cost)

	return nil
}

// ListDocuments lists documents with pagination
func (s *SQLiteDocumentStore) ListDocuments(ctx context.Context, privacyLevel string, offset, limit int) ([]*Document, error) {
	var query string
	var args []interface{}

	if privacyLevel != "" {
		query = `
			SELECT id, content, metadata, privacy_level, created_at, updated_at, access_count, last_accessed
			FROM documents
			WHERE privacy_level = ?
			ORDER BY created_at DESC
			LIMIT ? OFFSET ?
		`
		args = []interface{}{privacyLevel, limit, offset}
	} else {
		query = `
			SELECT id, content, metadata, privacy_level, created_at, updated_at, access_count, last_accessed
			FROM documents
			WHERE privacy_level != 'deleted'
			ORDER BY created_at DESC
			LIMIT ? OFFSET ?
		`
		args = []interface{}{limit, offset}
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list documents: %w", err)
	}
	defer rows.Close()

	var documents []*Document
	for rows.Next() {
		var doc Document
		var metadataJSON string
		var lastAccessed sql.NullTime

		if err := rows.Scan(&doc.ID, &doc.Content, &metadataJSON, &doc.PrivacyLevel, &doc.CreatedAt, &doc.UpdatedAt, &doc.AccessCount, &lastAccessed); err != nil {
			continue
		}

		if metadataJSON != "" {
			_ = json.Unmarshal([]byte(metadataJSON), &doc.Metadata)
		}

		if lastAccessed.Valid {
			doc.LastAccessed = lastAccessed.Time
		}

		documents = append(documents, &doc)
	}

	return documents, nil
}

// SearchDocuments searches documents by content
func (s *SQLiteDocumentStore) SearchDocuments(ctx context.Context, query string, privacyLevel string, limit int) ([]*Document, error) {
	cost := PrivacyBudgetCosts.SearchQuery
	if !s.privacyConfig.CanPerformOperation(cost) {
		return nil, fmt.Errorf("privacy budget would be exceeded")
	}

	var sqlQuery string
	var args []interface{}

	if privacyLevel != "" {
		sqlQuery = `
			SELECT id, content, metadata, privacy_level, created_at, updated_at, access_count, last_accessed
			FROM documents
			WHERE privacy_level = ? AND content LIKE ?
			ORDER BY created_at DESC
			LIMIT ?
		`
		args = []interface{}{privacyLevel, "%" + query + "%", limit}
	} else {
		sqlQuery = `
			SELECT id, content, metadata, privacy_level, created_at, updated_at, access_count, last_accessed
			FROM documents
			WHERE privacy_level != 'deleted' AND content LIKE ?
			ORDER BY created_at DESC
			LIMIT ?
		`
		args = []interface{}{"%" + query + "%", limit}
	}

	rows, err := s.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to search documents: %w", err)
	}
	defer rows.Close()

	var documents []*Document
	for rows.Next() {
		var doc Document
		var metadataJSON string
		var lastAccessed sql.NullTime

		if err := rows.Scan(&doc.ID, &doc.Content, &metadataJSON, &doc.PrivacyLevel, &doc.CreatedAt, &doc.UpdatedAt, &doc.AccessCount, &lastAccessed); err != nil {
			continue
		}

		if metadataJSON != "" {
			_ = json.Unmarshal([]byte(metadataJSON), &doc.Metadata)
		}

		if lastAccessed.Valid {
			doc.LastAccessed = lastAccessed.Time
		}

		documents = append(documents, &doc)
	}

	s.privacyConfig.ConsumeBudget(cost)
	return documents, nil
}

// GetDocumentStats returns document statistics
func (s *SQLiteDocumentStore) GetDocumentStats(ctx context.Context) (map[string]interface{}, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT 
			privacy_level,
			COUNT(*) as count,
			AVG(LENGTH(content)) as avg_content_length,
			SUM(access_count) as total_accesses
		FROM documents
		WHERE privacy_level != 'deleted'
		GROUP BY privacy_level
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to get document stats: %w", err)
	}
	defer rows.Close()

	stats := map[string]interface{}{
		"by_privacy_level": make(map[string]interface{}),
	}

	for rows.Next() {
		var privacyLevel string
		var count, avgLength, totalAccesses sql.NullFloat64

		err := rows.Scan(&privacyLevel, &count, &avgLength, &totalAccesses)
		if err != nil {
			continue
		}

		stats["by_privacy_level"].(map[string]interface{})[privacyLevel] = map[string]interface{}{
			"count":              count.Float64,
			"avg_content_length": avgLength.Float64,
			"total_accesses":     totalAccesses.Float64,
		}
	}

	return stats, nil
}

// CleanupOldDocuments removes old documents
func (s *SQLiteDocumentStore) CleanupOldDocuments(ctx context.Context) error {
	cutoffDate := time.Now().AddDate(0, 0, -s.privacyConfig.RetentionDays)

	_, err := s.db.ExecContext(ctx, `
		DELETE FROM document_versions WHERE created_at < ?
	`, cutoffDate)
	if err != nil {
		return fmt.Errorf("failed to cleanup document versions: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		DELETE FROM document_access_log WHERE timestamp < ?
	`, cutoffDate)
	if err != nil {
		return fmt.Errorf("failed to cleanup access logs: %w", err)
	}

	return nil
}

func (s *SQLiteDocumentStore) logDocumentAccess(ctx context.Context, docID, userID, sessionID, accessType string, privacyCost float64) {
	userIDHash := AnonymizeString(userID)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO document_access_log (document_id, user_id_hash, session_id, access_type, privacy_budget_used)
		VALUES (?, ?, ?, ?, ?)
	`, docID, userIDHash, sessionID, accessType, privacyCost)

	if err != nil {
		fmt.Printf("Warning: failed to log document access: %v\n", err)
	}
}

// Close closes the database connection
func (s *SQLiteDocumentStore) Close() error {
	return s.db.Close()
}

// SQLiteDocumentStore implements all HANA DocumentStore methods

