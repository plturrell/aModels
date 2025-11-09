package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	_ "github.com/lib/pq"
)

// PostgresCacheStore persists model cache state to PostgreSQL
type PostgresCacheStore struct {
	db *sql.DB
}

// CacheState represents the state of a cached model
type CacheState struct {
	Domain      string
	ModelType   string // safetensors, gguf, transformers
	ModelPath   string
	LoadedAt    time.Time
	MemoryMB    int64
	AccessCount  int64
	LastAccess  time.Time
	CacheData   map[string]interface{} // Additional metadata
}

// NewPostgresCacheStore creates a new PostgreSQL cache store
func NewPostgresCacheStore(dsn string) (*PostgresCacheStore, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("postgres ping failed: %w", err)
	}

	store := &PostgresCacheStore{db: db}

	// Create table if it doesn't exist
	if err := store.createTable(context.Background()); err != nil {
		return nil, fmt.Errorf("create model_cache_state table: %w", err)
	}

	return store, nil
}

// createTable creates the model_cache_state table if it doesn't exist
func (p *PostgresCacheStore) createTable(ctx context.Context) error {
	query := `
	CREATE TABLE IF NOT EXISTS model_cache_state (
		domain VARCHAR(255) PRIMARY KEY,
		model_type VARCHAR(50),
		model_path TEXT,
		loaded_at TIMESTAMP,
		memory_mb INTEGER,
		access_count BIGINT DEFAULT 0,
		last_access TIMESTAMP,
		cache_data JSONB,
		updated_at TIMESTAMP DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_model_cache_state_last_access ON model_cache_state(last_access);
	CREATE INDEX IF NOT EXISTS idx_model_cache_state_model_type ON model_cache_state(model_type);
	`

	_, err := p.db.ExecContext(ctx, query)
	return err
}

// SaveCacheState saves or updates a model cache state
func (p *PostgresCacheStore) SaveCacheState(ctx context.Context, state *CacheState) error {
	if state == nil {
		return fmt.Errorf("cache state is nil")
	}

	cacheDataJSON, _ := json.Marshal(state.CacheData)

	query := `
		INSERT INTO model_cache_state (
			domain, model_type, model_path, loaded_at, 
			memory_mb, access_count, last_access, cache_data, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
		ON CONFLICT (domain) 
		DO UPDATE SET 
			model_type = EXCLUDED.model_type,
			model_path = EXCLUDED.model_path,
			loaded_at = EXCLUDED.loaded_at,
			memory_mb = EXCLUDED.memory_mb,
			access_count = EXCLUDED.access_count,
			last_access = EXCLUDED.last_access,
			cache_data = EXCLUDED.cache_data,
			updated_at = NOW()
	`

	_, err := p.db.ExecContext(ctx, query,
		state.Domain,
		state.ModelType,
		state.ModelPath,
		state.LoadedAt,
		state.MemoryMB,
		state.AccessCount,
		state.LastAccess,
		cacheDataJSON,
	)

	return err
}

// LoadCacheState loads a model cache state by domain
func (p *PostgresCacheStore) LoadCacheState(ctx context.Context, domain string) (*CacheState, error) {
	query := `
		SELECT domain, model_type, model_path, loaded_at, 
		       memory_mb, access_count, last_access, cache_data
		FROM model_cache_state
		WHERE domain = $1
	`

	var state CacheState
	var cacheDataJSON []byte
	var loadedAt, lastAccess sql.NullTime

	err := p.db.QueryRowContext(ctx, query, domain).Scan(
		&state.Domain,
		&state.ModelType,
		&state.ModelPath,
		&loadedAt,
		&state.MemoryMB,
		&state.AccessCount,
		&lastAccess,
		&cacheDataJSON,
	)

	if err == sql.ErrNoRows {
		return nil, nil // Not found, not an error
	}
	if err != nil {
		return nil, fmt.Errorf("query cache state: %w", err)
	}

	if loadedAt.Valid {
		state.LoadedAt = loadedAt.Time
	}
	if lastAccess.Valid {
		state.LastAccess = lastAccess.Time
	}

	if len(cacheDataJSON) > 0 {
		if err := json.Unmarshal(cacheDataJSON, &state.CacheData); err != nil {
			log.Printf("⚠️  Failed to unmarshal cache_data for domain %s: %v", domain, err)
			state.CacheData = make(map[string]interface{})
		}
	} else {
		state.CacheData = make(map[string]interface{})
	}

	return &state, nil
}

// LoadAllCacheStates loads all model cache states
func (p *PostgresCacheStore) LoadAllCacheStates(ctx context.Context) (map[string]*CacheState, error) {
	query := `
		SELECT domain, model_type, model_path, loaded_at, 
		       memory_mb, access_count, last_access, cache_data
		FROM model_cache_state
		ORDER BY last_access DESC
	`

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("query all cache states: %w", err)
	}
	defer rows.Close()

	states := make(map[string]*CacheState)
	for rows.Next() {
		var state CacheState
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
			log.Printf("⚠️  Error scanning cache state: %v", err)
			continue
		}

		if loadedAt.Valid {
			state.LoadedAt = loadedAt.Time
		}
		if lastAccess.Valid {
			state.LastAccess = lastAccess.Time
		}

		if len(cacheDataJSON) > 0 {
			if err := json.Unmarshal(cacheDataJSON, &state.CacheData); err != nil {
				log.Printf("⚠️  Failed to unmarshal cache_data for domain %s: %v", state.Domain, err)
				state.CacheData = make(map[string]interface{})
			}
		} else {
			state.CacheData = make(map[string]interface{})
		}

		states[state.Domain] = &state
	}

	return states, rows.Err()
}

// DeleteCacheState deletes a model cache state
func (p *PostgresCacheStore) DeleteCacheState(ctx context.Context, domain string) error {
	query := `DELETE FROM model_cache_state WHERE domain = $1`
	_, err := p.db.ExecContext(ctx, query, domain)
	return err
}

// Close closes the database connection
func (p *PostgresCacheStore) Close() error {
	if p.db != nil {
		return p.db.Close()
	}
	return nil
}

