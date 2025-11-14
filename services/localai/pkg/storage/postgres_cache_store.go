package storage

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"strings"
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

// createTable creates the model_cache_state and response_cache tables if they don't exist
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

	CREATE TABLE IF NOT EXISTS response_cache (
		cache_key VARCHAR(255) PRIMARY KEY,
		prompt_hash VARCHAR(255),
		model VARCHAR(255),
		domain VARCHAR(255),
		response TEXT,
		tokens_used INTEGER,
		temperature FLOAT,
		max_tokens INTEGER,
		created_at TIMESTAMP DEFAULT NOW(),
		expires_at TIMESTAMP,
		access_count INTEGER DEFAULT 0,
		last_accessed TIMESTAMP DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_response_cache_model_domain ON response_cache(model, domain);
	CREATE INDEX IF NOT EXISTS idx_response_cache_expires_at ON response_cache(expires_at);
	CREATE INDEX IF NOT EXISTS idx_response_cache_last_accessed ON response_cache(last_accessed);
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

// GenerateCacheKey generates a cache key for a response
func (p *PostgresCacheStore) GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string {
	// Use the same algorithm as HANACache stub
	normalized := strings.ToLower(strings.TrimSpace(prompt))
	base := sha256.Sum256([]byte(normalized))
	keyData := fmt.Sprintf("%s:%s:%s:%.2f:%d:%.3f:%d", hex.EncodeToString(base[:]), model, domain, temperature, maxTokens, topP, topK)
	key := sha256.Sum256([]byte(keyData))
	return hex.EncodeToString(key[:])
}

// Get retrieves a cached response entry
func (p *PostgresCacheStore) Get(ctx context.Context, key string) (*CacheEntry, error) {
	query := `
		SELECT cache_key, prompt_hash, model, domain, response, tokens_used,
		       temperature, max_tokens, created_at, expires_at, access_count, last_accessed
		FROM response_cache
		WHERE cache_key = $1 AND (expires_at IS NULL OR expires_at > NOW())
	`

	var entry CacheEntry
	var expiresAt sql.NullTime
	var createdAt, lastAccessed sql.NullTime

	err := p.db.QueryRowContext(ctx, query, key).Scan(
		&entry.CacheKey,
		&entry.PromptHash,
		&entry.Model,
		&entry.Domain,
		&entry.Response,
		&entry.TokensUsed,
		&entry.Temperature,
		&entry.MaxTokens,
		&createdAt,
		&expiresAt,
		&entry.AccessCount,
		&lastAccessed,
	)

	if err == sql.ErrNoRows {
		return nil, nil // Not found, not an error
	}
	if err != nil {
		return nil, fmt.Errorf("query cache entry: %w", err)
	}

	if createdAt.Valid {
		entry.CreatedAt = createdAt.Time
	}
	if expiresAt.Valid {
		entry.ExpiresAt = expiresAt.Time
	}
	if lastAccessed.Valid {
		entry.LastAccessed = lastAccessed.Time
	}

	// Update access count and last accessed time
	go p.updateAccessStats(context.Background(), key)

	return &entry, nil
}

// Set stores a response entry in cache
func (p *PostgresCacheStore) Set(ctx context.Context, entry *CacheEntry) error {
	if entry == nil {
		return fmt.Errorf("cache entry is nil")
	}

	// Set default expiration if not provided
	expiresAt := entry.ExpiresAt
	if expiresAt.IsZero() {
		expiresAt = time.Now().Add(24 * time.Hour)
	}

	query := `
		INSERT INTO response_cache (
			cache_key, prompt_hash, model, domain, response, tokens_used,
			temperature, max_tokens, created_at, expires_at, access_count, last_accessed
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		ON CONFLICT (cache_key)
		DO UPDATE SET
			response = EXCLUDED.response,
			tokens_used = EXCLUDED.tokens_used,
			temperature = EXCLUDED.temperature,
			max_tokens = EXCLUDED.max_tokens,
			expires_at = EXCLUDED.expires_at,
			last_accessed = NOW()
	`

	createdAt := entry.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now()
	}

	_, err := p.db.ExecContext(ctx, query,
		entry.CacheKey,
		entry.PromptHash,
		entry.Model,
		entry.Domain,
		entry.Response,
		entry.TokensUsed,
		entry.Temperature,
		entry.MaxTokens,
		createdAt,
		expiresAt,
		entry.AccessCount,
		time.Now(),
	)

	return err
}

// updateAccessStats updates access statistics for a cache entry
func (p *PostgresCacheStore) updateAccessStats(ctx context.Context, key string) {
	query := `
		UPDATE response_cache
		SET access_count = access_count + 1,
		    last_accessed = NOW()
		WHERE cache_key = $1
	`
	_, _ = p.db.ExecContext(ctx, query, key)
}

