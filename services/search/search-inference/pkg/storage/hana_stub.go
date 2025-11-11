//go:build !hana

package storage

// Use SQLite instead of HANA when HANA build tag is not set
// All HANA functions are redirected to SQLite implementations

// NewHANASearchIndex creates a SQLite-based search index
// Returns SQLiteDocumentStore which implements the same interface
func NewHANASearchIndex(dsn string, privacyConfig *PrivacyConfig) (*SQLiteDocumentStore, error) {
	if dsn == "" {
		dsn = "file:search_index.db?cache=shared&mode=rwc"
	}
	return NewSQLiteDocumentStore(dsn, privacyConfig)
}

// NewHANASearchLogger creates a SQLite-based logger
func NewHANASearchLogger(dsn string, privacyConfig *PrivacyConfig) (*SQLiteSearchLogger, error) {
	if dsn == "" {
		dsn = "file:search_logs.db?cache=shared&mode=rwc"
	}
	return NewSQLiteSearchLogger(dsn, privacyConfig)
}

// NewHANADocumentStore creates a SQLite-based document store
func NewHANADocumentStore(dsn string, privacyConfig *PrivacyConfig) (*SQLiteDocumentStore, error) {
	if dsn == "" {
		dsn = "file:documents.db?cache=shared&mode=rwc"
	}
	return NewSQLiteDocumentStore(dsn, privacyConfig)
}

// Type aliases for compatibility - SQLite implementations are in sqlite_documents.go and sqlite_search_log.go
type HANASearchLogger = SQLiteSearchLogger
type HANADocumentStore = SQLiteDocumentStore
