package sqlite

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/langchain-ai/langgraph-go/pkg/checkpoint"
	_ "github.com/mattn/go-sqlite3"
)

const defaultTableName = "checkpoints"

type storeConfig struct {
	tableName string
}

// Option configures optional store behaviour (currently only the table name).
type Option func(*storeConfig)

// WithTableName overrides the checkpoints table.
func WithTableName(name string) Option {
	return func(cfg *storeConfig) {
		if name != "" {
			cfg.tableName = name
		}
	}
}

type Store struct {
	db        *sql.DB
	tableName string
}

// New initialises a SQLite store using the provided database path.
func New(ctx context.Context, dbPath string, opts ...Option) (*Store, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite database: %w", err)
	}

	cfg := storeConfig{tableName: defaultTableName}
	for _, opt := range opts {
		opt(&cfg)
	}

	if err := ensureTable(ctx, db, cfg.tableName); err != nil {
		return nil, err
	}

	return &Store{db: db, tableName: cfg.tableName}, nil
}

// NewStore creates a Store from an existing *sql.DB.
func NewStore(db *sql.DB, opts ...Option) (*Store, error) {
	if db == nil {
		return nil, fmt.Errorf("nil db provided")
	}
	cfg := storeConfig{tableName: defaultTableName}
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := ensureTable(context.Background(), db, cfg.tableName); err != nil {
		return nil, err
	}
	return &Store{db: db, tableName: cfg.tableName}, nil
}

func ensureTable(ctx context.Context, db *sql.DB, tableName string) error {
	createStmt := fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS %s (
	key TEXT PRIMARY KEY,
	payload BLOB
);`, tableName)
	if _, err := db.ExecContext(ctx, createStmt); err != nil {
		return fmt.Errorf("failed to create checkpoints table: %w", err)
	}
	return nil
}

func (s *Store) Save(ctx context.Context, key string, payload []byte) error {
	stmt := fmt.Sprintf("INSERT OR REPLACE INTO %s (key, payload) VALUES (?, ?)", s.tableName)
	_, err := s.db.ExecContext(ctx, stmt, key, payload)
	if err != nil {
		return fmt.Errorf("failed to save checkpoint: %w", err)
	}
	return nil
}

func (s *Store) Load(ctx context.Context, key string) ([]byte, error) {
	stmt := fmt.Sprintf("SELECT payload FROM %s WHERE key = ?", s.tableName)
	var payload []byte
	err := s.db.QueryRowContext(ctx, stmt, key).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, checkpoint.ErrNotFound
		}
		return nil, fmt.Errorf("failed to load checkpoint: %w", err)
	}
	return payload, nil
}

func (s *Store) Delete(ctx context.Context, key string) error {
	stmt := fmt.Sprintf("DELETE FROM %s WHERE key = ?", s.tableName)
	_, err := s.db.ExecContext(ctx, stmt, key)
	if err != nil {
		return fmt.Errorf("failed to delete checkpoint: %w", err)
	}
	return nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}
