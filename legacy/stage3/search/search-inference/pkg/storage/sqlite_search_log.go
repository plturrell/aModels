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

// SQLiteSearchLogger provides search logging using SQLite
type SQLiteSearchLogger struct {
	db            *sql.DB
	privacyConfig *PrivacyConfig
}

// NewSQLiteSearchLogger creates a new SQLite search logger
func NewSQLiteSearchLogger(dsn string, privacyConfig *PrivacyConfig) (*SQLiteSearchLogger, error) {
	if dsn == "" {
		dsn = "file:search_logs.db?cache=shared&mode=rwc"
	}
	
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to SQLite: %w", err)
	}

	logger := &SQLiteSearchLogger{
		db:            db,
		privacyConfig: privacyConfig,
	}

	if err := logger.createTables(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to create tables: %w", err)
	}

	return logger, nil
}

func (s *SQLiteSearchLogger) createTables(ctx context.Context) error {
	createLogs := `
CREATE TABLE IF NOT EXISTS search_logs (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	query_hash TEXT NOT NULL,
	query_embedding BLOB,
	result_count INTEGER NOT NULL,
	top_result_id TEXT,
	latency_ms INTEGER NOT NULL,
	user_id_hash TEXT,
	session_id TEXT,
	timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	privacy_budget_used REAL NOT NULL
)`
	if _, err := s.db.ExecContext(ctx, createLogs); err != nil {
		return fmt.Errorf("failed to create search_logs table: %w", err)
	}

	createClicks := `
CREATE TABLE IF NOT EXISTS search_clicks (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	query_hash TEXT NOT NULL,
	document_id TEXT NOT NULL,
	position INTEGER DEFAULT 0,
	timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	session_id TEXT
)`
	if _, err := s.db.ExecContext(ctx, createClicks); err != nil {
		return fmt.Errorf("failed to create search_clicks table: %w", err)
	}

	indexes := []string{
		"CREATE INDEX IF NOT EXISTS idx_search_logs_query_hash ON search_logs (query_hash)",
		"CREATE INDEX IF NOT EXISTS idx_search_logs_timestamp ON search_logs (timestamp)",
		"CREATE INDEX IF NOT EXISTS idx_search_clicks_query_hash ON search_clicks (query_hash)",
	}

	for _, stmt := range indexes {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			fmt.Printf("⚠️  Index creation failed: %v\n", err)
		}
	}

	return nil
}

func (s *SQLiteSearchLogger) LogSearch(ctx context.Context, searchLog *SearchLog) error {
	if s.privacyConfig == nil {
		return fmt.Errorf("privacy configuration not set")
	}
	
	cost := PrivacyBudgetCosts.SearchLog
	if !s.privacyConfig.CanPerformOperation(cost) {
		return fmt.Errorf("privacy budget would be exceeded")
	}

	embeddingJSON, _ := json.Marshal(searchLog.QueryEmbedding)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO search_logs (query_hash, query_embedding, result_count, top_result_id, latency_ms, user_id_hash, session_id, timestamp, privacy_budget_used)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, searchLog.QueryHash, embeddingJSON, searchLog.ResultCount, searchLog.TopResultID, searchLog.LatencyMs, searchLog.UserIDHash, searchLog.SessionID, searchLog.Timestamp, searchLog.PrivacyBudgetUsed)

	if err != nil {
		return fmt.Errorf("failed to log search: %w", err)
	}

	s.privacyConfig.ConsumeBudget(cost)
	return nil
}

func (s *SQLiteSearchLogger) LogClick(ctx context.Context, queryHash, documentID string, position int, sessionID string) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO search_clicks (query_hash, document_id, position, timestamp, session_id)
		VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
	`, queryHash, documentID, position, sessionID)

	if err != nil {
		return fmt.Errorf("failed to log click: %w", err)
	}

	return nil
}

func (s *SQLiteSearchLogger) GetSearchAnalytics(ctx context.Context, startTime, endTime time.Time) (map[string]interface{}, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT 
			COUNT(*) as total_searches,
			AVG(latency_ms) as avg_latency,
			AVG(result_count) as avg_results,
			SUM(privacy_budget_used) as total_privacy_budget
		FROM search_logs
		WHERE timestamp >= ? AND timestamp <= ?
	`, startTime, endTime)
	
	if err != nil {
		return nil, fmt.Errorf("failed to get analytics: %w", err)
	}
	defer rows.Close()

	analytics := make(map[string]interface{})
	if rows.Next() {
		var totalSearches, avgLatency, avgResults, totalBudget sql.NullFloat64
		if err := rows.Scan(&totalSearches, &avgLatency, &avgResults, &totalBudget); err == nil {
			analytics["total_searches"] = totalSearches.Float64
			analytics["avg_latency_ms"] = avgLatency.Float64
			analytics["avg_results"] = avgResults.Float64
			analytics["total_privacy_budget"] = totalBudget.Float64
		}
	}

	return analytics, nil
}

func (s *SQLiteSearchLogger) Close() error {
	return s.db.Close()
}

