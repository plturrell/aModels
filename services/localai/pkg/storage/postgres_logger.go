package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// PostgresLogger logs inference requests and responses to PostgreSQL
type PostgresLogger struct {
	db          *sql.DB
	batchBuffer []InferenceLogEntry
	batchMu     sync.Mutex
	batchSize   int
	flushInterval time.Duration
	stopChan    chan struct{}
	wg          sync.WaitGroup
}

// InferenceLogEntry represents a single inference log entry
type InferenceLogEntry struct {
	RequestID      string
	Domain         string
	ModelName      string
	PromptLength   int
	ResponseLength int
	LatencyMS      int64
	TokensGenerated int
	TokensPrompt   int
	CreatedAt      time.Time
	WorkflowID     string
	UserID         string
	SessionID      string
	Metadata       map[string]interface{}
}

// NewPostgresLogger creates a new Postgres logger
func NewPostgresLogger(dsn string) (*PostgresLogger, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("postgres ping failed: %w", err)
	}

	logger := &PostgresLogger{
		db:            db,
		batchBuffer:   make([]InferenceLogEntry, 0, 100),
		batchSize:     100,
		flushInterval: 5 * time.Second,
		stopChan:      make(chan struct{}),
	}

	// Create tables
	if err := logger.createTables(context.Background()); err != nil {
		return nil, fmt.Errorf("create tables: %w", err)
	}

	// Start background flush goroutine
	logger.wg.Add(1)
	go logger.flushLoop()

	return logger, nil
}

// createTables creates the necessary tables for inference logging
func (l *PostgresLogger) createTables(ctx context.Context) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS inference_logs (
			id SERIAL PRIMARY KEY,
			request_id VARCHAR(255) NOT NULL,
			domain VARCHAR(255),
			model_name VARCHAR(255),
			prompt_length INTEGER,
			response_length INTEGER,
			latency_ms BIGINT,
			tokens_generated INTEGER,
			tokens_prompt INTEGER,
			created_at TIMESTAMP DEFAULT NOW(),
			workflow_id VARCHAR(255),
			user_id VARCHAR(255),
			session_id VARCHAR(255),
			metadata JSONB
		)`,
		`CREATE INDEX IF NOT EXISTS idx_inference_logs_domain ON inference_logs(domain)`,
		`CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at ON inference_logs(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_inference_logs_workflow_id ON inference_logs(workflow_id)`,
		`CREATE INDEX IF NOT EXISTS idx_inference_logs_user_id ON inference_logs(user_id)`,
		`CREATE TABLE IF NOT EXISTS model_performance_metrics (
			id SERIAL PRIMARY KEY,
			domain VARCHAR(255) NOT NULL,
			model_name VARCHAR(255),
			loading_time_ms INTEGER,
			memory_usage_mb INTEGER,
			access_count BIGINT DEFAULT 0,
			last_access TIMESTAMP,
			avg_latency_ms INTEGER,
			created_at TIMESTAMP DEFAULT NOW(),
			updated_at TIMESTAMP DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS idx_model_performance_domain ON model_performance_metrics(domain)`,
		`CREATE INDEX IF NOT EXISTS idx_model_performance_last_access ON model_performance_metrics(last_access)`,
		`CREATE TABLE IF NOT EXISTS model_cache_state (
			domain VARCHAR(255) PRIMARY KEY,
			model_type VARCHAR(50),
			model_path TEXT,
			loaded_at TIMESTAMP,
			memory_mb INTEGER,
			access_count BIGINT DEFAULT 0,
			last_access TIMESTAMP,
			cache_data JSONB,
			updated_at TIMESTAMP DEFAULT NOW()
		)`,
	}

	for _, query := range queries {
		if _, err := l.db.ExecContext(ctx, query); err != nil {
			return fmt.Errorf("execute query: %w", err)
		}
	}

	return nil
}

// LogInference logs an inference request (batched)
func (l *PostgresLogger) LogInference(entry InferenceLogEntry) {
	l.batchMu.Lock()
	defer l.batchMu.Unlock()

	l.batchBuffer = append(l.batchBuffer, entry)

	// Flush if batch is full
	if len(l.batchBuffer) >= l.batchSize {
		l.flushLocked()
	}
}

// flushLocked flushes the batch buffer (must be called with batchMu locked)
func (l *PostgresLogger) flushLocked() {
	if len(l.batchBuffer) == 0 {
		return
	}

	// Copy buffer and clear
	batch := make([]InferenceLogEntry, len(l.batchBuffer))
	copy(batch, l.batchBuffer)
	l.batchBuffer = l.batchBuffer[:0]

	// Flush in background
	go l.flushBatch(batch)
}

// flushBatch flushes a batch of entries
func (l *PostgresLogger) flushBatch(batch []InferenceLogEntry) {
	if len(batch) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	query := `
		INSERT INTO inference_logs (
			request_id, domain, model_name, prompt_length, response_length,
			latency_ms, tokens_generated, tokens_prompt, created_at,
			workflow_id, user_id, session_id, metadata
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
	`

	tx, err := l.db.BeginTx(ctx, nil)
	if err != nil {
		log.Printf("⚠️  Failed to begin transaction for inference logs: %v", err)
		return
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, query)
	if err != nil {
		log.Printf("⚠️  Failed to prepare statement: %v", err)
		return
	}
	defer stmt.Close()

	for _, entry := range batch {
		metadataJSON, _ := json.Marshal(entry.Metadata)
		createdAt := entry.CreatedAt
		if createdAt.IsZero() {
			createdAt = time.Now()
		}

		_, err := stmt.ExecContext(ctx,
			entry.RequestID,
			entry.Domain,
			entry.ModelName,
			entry.PromptLength,
			entry.ResponseLength,
			entry.LatencyMS,
			entry.TokensGenerated,
			entry.TokensPrompt,
			createdAt,
			entry.WorkflowID,
			entry.UserID,
			entry.SessionID,
			metadataJSON,
		)
		if err != nil {
			log.Printf("⚠️  Failed to insert inference log: %v", err)
			continue
		}
	}

	if err := tx.Commit(); err != nil {
		log.Printf("⚠️  Failed to commit inference logs: %v", err)
	} else {
		log.Printf("✅ Flushed %d inference logs to Postgres", len(batch))
	}
}

// flushLoop periodically flushes the batch buffer
func (l *PostgresLogger) flushLoop() {
	defer l.wg.Done()

	ticker := time.NewTicker(l.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			l.batchMu.Lock()
			l.flushLocked()
			l.batchMu.Unlock()
		case <-l.stopChan:
			// Final flush on shutdown
			l.batchMu.Lock()
			l.flushLocked()
			l.batchMu.Unlock()
			return
		}
	}
}

// SaveModelCacheState saves model cache state to Postgres
func (l *PostgresLogger) SaveModelCacheState(ctx context.Context, domain, modelType, modelPath string, memoryMB int64, accessCount int64, lastAccess time.Time, cacheData map[string]interface{}) error {
	cacheDataJSON, _ := json.Marshal(cacheData)

	query := `
		INSERT INTO model_cache_state (
			domain, model_type, model_path, loaded_at, memory_mb,
			access_count, last_access, cache_data, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
		ON CONFLICT (domain) DO UPDATE SET
			model_type = EXCLUDED.model_type,
			model_path = EXCLUDED.model_path,
			loaded_at = EXCLUDED.loaded_at,
			memory_mb = EXCLUDED.memory_mb,
			access_count = EXCLUDED.access_count,
			last_access = EXCLUDED.last_access,
			cache_data = EXCLUDED.cache_data,
			updated_at = NOW()
	`

	_, err := l.db.ExecContext(ctx, query,
		domain, modelType, modelPath, time.Now(), memoryMB,
		accessCount, lastAccess, cacheDataJSON,
	)
	return err
}

// LoadModelCacheState loads model cache state from Postgres
func (l *PostgresLogger) LoadModelCacheState(ctx context.Context) (map[string]ModelCacheState, error) {
	query := `
		SELECT domain, model_type, model_path, loaded_at, memory_mb,
		       access_count, last_access, cache_data
		FROM model_cache_state
		ORDER BY domain
	`

	rows, err := l.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query model cache state: %w", err)
	}
	defer rows.Close()

	states := make(map[string]ModelCacheState)
	for rows.Next() {
		var state ModelCacheState
		var cacheDataJSON []byte
		var loadedAt, lastAccess sql.NullTime

		if err := rows.Scan(
			&state.Domain,
			&state.ModelType,
			&state.ModelPath,
			&loadedAt,
			&state.MemoryMB,
			&state.AccessCount,
			&lastAccess,
			&cacheDataJSON,
		); err != nil {
			log.Printf("⚠️  Error scanning model cache state: %v", err)
			continue
		}

		if loadedAt.Valid {
			state.LoadedAt = loadedAt.Time
		}
		if lastAccess.Valid {
			state.LastAccess = lastAccess.Time
		}
		if len(cacheDataJSON) > 0 {
			json.Unmarshal(cacheDataJSON, &state.CacheData)
		}

		states[state.Domain] = state
	}

	return states, rows.Err()
}

// ModelCacheState represents cached model state
type ModelCacheState struct {
	Domain      string
	ModelType   string
	ModelPath   string
	LoadedAt    time.Time
	MemoryMB    int64
	AccessCount int64
	LastAccess  time.Time
	CacheData   map[string]interface{}
}

// UpdateModelPerformanceMetrics updates model performance metrics
func (l *PostgresLogger) UpdateModelPerformanceMetrics(ctx context.Context, domain, modelName string, loadingTimeMS, memoryMB, avgLatencyMS int, accessCount int64, lastAccess time.Time) error {
	query := `
		INSERT INTO model_performance_metrics (
			domain, model_name, loading_time_ms, memory_usage_mb,
			avg_latency_ms, access_count, last_access, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
		ON CONFLICT (domain) DO UPDATE SET
			model_name = EXCLUDED.model_name,
			loading_time_ms = EXCLUDED.loading_time_ms,
			memory_usage_mb = EXCLUDED.memory_usage_mb,
			avg_latency_ms = EXCLUDED.avg_latency_ms,
			access_count = EXCLUDED.access_count,
			last_access = EXCLUDED.last_access,
			updated_at = NOW()
	`

	_, err := l.db.ExecContext(ctx, query,
		domain, modelName, loadingTimeMS, memoryMB,
		avgLatencyMS, accessCount, lastAccess,
	)
	return err
}

// Close closes the logger and flushes remaining entries
func (l *PostgresLogger) Close() error {
	close(l.stopChan)
	l.wg.Wait()

	// Final flush
	l.batchMu.Lock()
	l.flushLocked()
	l.batchMu.Unlock()

	return l.db.Close()
}

