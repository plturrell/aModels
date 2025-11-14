package storage

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
)

// PostgresInferenceLogger logs inference requests to PostgreSQL
type PostgresInferenceLogger struct {
	db *sql.DB
}

// PostgresInferenceLog represents a single inference request log entry for PostgreSQL
type PostgresInferenceLog struct {
	Domain          string
	ModelName       string
	PromptLength    int
	ResponseLength  int
	LatencyMS       int
	TokensGenerated int
	TokensPrompt    int
	WorkflowID      string
	UserID          string
	CreatedAt       time.Time
}

// NewPostgresInferenceLogger creates a new PostgreSQL inference logger
func NewPostgresInferenceLogger(dsn string) (*PostgresInferenceLogger, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("postgres ping failed: %w", err)
	}

	logger := &PostgresInferenceLogger{db: db}

	// Create table if it doesn't exist
	if err := logger.createTable(context.Background()); err != nil {
		return nil, fmt.Errorf("create inference_logs table: %w", err)
	}

	return logger, nil
}

// createTable creates the inference_logs table if it doesn't exist
func (p *PostgresInferenceLogger) createTable(ctx context.Context) error {
	query := `
	CREATE TABLE IF NOT EXISTS inference_logs (
		id SERIAL PRIMARY KEY,
		domain VARCHAR(255) NOT NULL,
		model_name VARCHAR(255),
		prompt_length INTEGER,
		response_length INTEGER,
		latency_ms INTEGER,
		tokens_generated INTEGER,
		tokens_prompt INTEGER,
		workflow_id VARCHAR(255),
		user_id VARCHAR(255),
		created_at TIMESTAMP DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_inference_logs_domain ON inference_logs(domain);
	CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at ON inference_logs(created_at);
	CREATE INDEX IF NOT EXISTS idx_inference_logs_workflow_id ON inference_logs(workflow_id);
	CREATE INDEX IF NOT EXISTS idx_inference_logs_user_id ON inference_logs(user_id);
	`

	_, err := p.db.ExecContext(ctx, query)
	return err
}

// GetDB returns the underlying database connection (for function calling queries)
func (p *PostgresInferenceLogger) GetDB() *sql.DB {
	return p.db
}

// LogInference logs an inference request (non-blocking, batched)
func (p *PostgresInferenceLogger) LogInference(ctx context.Context, logEntry *PostgresInferenceLog) error {
	if logEntry == nil {
		return fmt.Errorf("log entry is nil")
	}

	query := `
		INSERT INTO inference_logs (
			domain, model_name, prompt_length, response_length, 
			latency_ms, tokens_generated, tokens_prompt, 
			workflow_id, user_id, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`

	_, err := p.db.ExecContext(ctx, query,
		logEntry.Domain,
		logEntry.ModelName,
		logEntry.PromptLength,
		logEntry.ResponseLength,
		logEntry.LatencyMS,
		logEntry.TokensGenerated,
		logEntry.TokensPrompt,
		logEntry.WorkflowID,
		logEntry.UserID,
		logEntry.CreatedAt,
	)

	return err
}

// BatchLogInference logs multiple inference requests in a single transaction
func (p *PostgresInferenceLogger) BatchLogInference(ctx context.Context, logs []*PostgresInferenceLog) error {
	if len(logs) == 0 {
		return nil
	}

	tx, err := p.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO inference_logs (
			domain, model_name, prompt_length, response_length, 
			latency_ms, tokens_generated, tokens_prompt, 
			workflow_id, user_id, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`)
	if err != nil {
		return fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, logEntry := range logs {
		_, err := stmt.ExecContext(ctx,
			logEntry.Domain,
			logEntry.ModelName,
			logEntry.PromptLength,
			logEntry.ResponseLength,
			logEntry.LatencyMS,
			logEntry.TokensGenerated,
			logEntry.TokensPrompt,
			logEntry.WorkflowID,
			logEntry.UserID,
			logEntry.CreatedAt,
		)
		if err != nil {
			return fmt.Errorf("execute batch insert: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}

	return nil
}

// Close closes the database connection
func (p *PostgresInferenceLogger) Close() error {
	if p.db != nil {
		return p.db.Close()
	}
	return nil
}

