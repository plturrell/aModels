//go:build hana

package storage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"log"
	"strings"
	"time"

	hdbdriver "github.com/SAP/go-hdb/driver"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/hanapool"
)

// CacheEntry represents a cached inference response
type CacheEntry struct {
	ID           int64     `json:"id"`
	CacheKey     string    `json:"cache_key"`
	PromptHash   string    `json:"prompt_hash"`
	Model        string    `json:"model"`
	Domain       string    `json:"domain"`
	Response     string    `json:"response"`
	TokensUsed   int       `json:"tokens_used"`
	Temperature  float64   `json:"temperature"`
	MaxTokens    int       `json:"max_tokens"`
	CreatedAt    time.Time `json:"created_at"`
	ExpiresAt    time.Time `json:"expires_at"`
	AccessCount  int       `json:"access_count"`
	LastAccessed time.Time `json:"last_accessed"`
}

// CacheStats tracks cache performance
type CacheStats struct {
	TotalEntries   int64   `json:"total_entries"`
	HitCount       int64   `json:"hit_count"`
	MissCount      int64   `json:"miss_count"`
	HitRate        float64 `json:"hit_rate"`
	AvgAccessCount float64 `json:"avg_access_count"`
	ExpiredEntries int64   `json:"expired_entries"`
}

// HANACache provides semantic caching for LocalAI inference
type HANACache struct {
	pool *hanapool.Pool
}

// NewHANACache creates a new HANA cache
func NewHANACache(pool *hanapool.Pool) *HANACache {
	return &HANACache{pool: pool}
}

func (c *HANACache) poolAvailable() bool {
	if c == nil || c.pool == nil {
		return false
	}
	return c.pool.GetDB() != nil
}

// GenerateCacheKey creates a cache key from prompt and parameters
func (c *HANACache) GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string {
	// Normalize prompt for better cache hits
	normalizedPrompt := strings.ToLower(strings.TrimSpace(prompt))

	// Create hash of normalized prompt
	hash := sha256.Sum256([]byte(normalizedPrompt))
	promptHash := hex.EncodeToString(hash[:])

	// Create cache key with parameters
	keyData := fmt.Sprintf("%s:%s:%s:%.2f:%d:%.3f:%d", promptHash, model, domain, temperature, maxTokens, topP, topK)
	keyHash := sha256.Sum256([]byte(keyData))

	return hex.EncodeToString(keyHash[:])
}

// Get retrieves a cached response
func (c *HANACache) Get(ctx context.Context, cacheKey string) (*CacheEntry, error) {
	if !c.poolAvailable() {
		return nil, fmt.Errorf("hana pool not configured")
	}

	query := `
SELECT id, cache_key, prompt_hash, model, domain, response,
       tokens_used, temperature, max_tokens, created_at,
       expires_at, access_count, last_accessed
FROM cache_entries
WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
`

	row := c.pool.QueryRow(ctx, query, cacheKey)

	entry := &CacheEntry{}
	var responseBuf bytes.Buffer
	responseLob := hdbdriver.NewLob(nil, &responseBuf)
	err := row.Scan(
		&entry.ID,
		&entry.CacheKey,
		&entry.PromptHash,
		&entry.Model,
		&entry.Domain,
		responseLob,
		&entry.TokensUsed,
		&entry.Temperature,
		&entry.MaxTokens,
		&entry.CreatedAt,
		&entry.ExpiresAt,
		&entry.AccessCount,
		&entry.LastAccessed,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Cache miss
		}
		return nil, fmt.Errorf("failed to get cache entry: %w", err)
	}

	entry.Response = responseBuf.String()

	// Update access statistics
	go c.updateAccessStats(context.Background(), entry.ID)

	return entry, nil
}

// Set stores a response in cache
func (c *HANACache) Set(ctx context.Context, entry *CacheEntry) error {
	if !c.poolAvailable() {
		return fmt.Errorf("hana pool not configured")
	}

	// Set default expiration (24 hours)
	if entry.ExpiresAt.IsZero() {
		entry.ExpiresAt = time.Now().Add(24 * time.Hour)
	}

	query := `
UPSERT cache_entries (
	cache_key, prompt_hash, model, domain, response,
	tokens_used, temperature, max_tokens, expires_at, last_accessed, access_count
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, access_count + 1)
WITH PRIMARY KEY
`

	_, err := c.pool.Execute(ctx, query,
		entry.CacheKey,
		entry.PromptHash,
		entry.Model,
		entry.Domain,
		entry.Response,
		entry.TokensUsed,
		entry.Temperature,
		entry.MaxTokens,
		entry.ExpiresAt,
	)

	if err != nil {
		return fmt.Errorf("failed to set cache entry: %w", err)
	}

	return nil
}

// FindSimilar finds semantically similar cached responses
func (c *HANACache) FindSimilar(ctx context.Context, prompt, model, domain string, threshold float64) ([]*CacheEntry, error) {
	if !c.poolAvailable() {
		return nil, fmt.Errorf("hana pool not configured")
	}
	// Normalize prompt
	normalizedPrompt := strings.ToLower(strings.TrimSpace(prompt))

	// For now, use simple text similarity
	// In production, you'd use vector similarity search
	query := `
		SELECT id, cache_key, prompt_hash, model, domain, response,
		       tokens_used, temperature, max_tokens, created_at,
		       expires_at, access_count, last_accessed
		FROM cache_entries 
		WHERE model = ? AND domain = ? 
		  AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
		  AND (
			LOWER(SUBSTRING(response, 1, 100)) LIKE ? OR
			LOWER(SUBSTRING(response, 1, 200)) LIKE ?
		  )
		ORDER BY access_count DESC, created_at DESC
		LIMIT 10
	`

	// Create search patterns
	searchPattern1 := "%" + strings.Join(strings.Fields(normalizedPrompt)[:min(3, len(strings.Fields(normalizedPrompt)))], "%") + "%"
	searchPattern2 := "%" + strings.Join(strings.Fields(normalizedPrompt)[:min(5, len(strings.Fields(normalizedPrompt)))], "%") + "%"

	rows, err := c.pool.Query(ctx, query, model, domain, searchPattern1, searchPattern2)
	if err != nil {
		return nil, fmt.Errorf("failed to find similar cache entries: %w", err)
	}
	defer rows.Close()

	var entries []*CacheEntry
	for rows.Next() {
		entry := &CacheEntry{}
		var responseBuf bytes.Buffer
		responseLob := hdbdriver.NewLob(nil, &responseBuf)
		err := rows.Scan(
			&entry.ID,
			&entry.CacheKey,
			&entry.PromptHash,
			&entry.Model,
			&entry.Domain,
			responseLob,
			&entry.TokensUsed,
			&entry.Temperature,
			&entry.MaxTokens,
			&entry.CreatedAt,
			&entry.ExpiresAt,
			&entry.AccessCount,
			&entry.LastAccessed,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan cache entry: %w", err)
		}

		entry.Response = responseBuf.String()

		entries = append(entries, entry)
	}

	return entries, nil
}

// GetStats retrieves cache performance statistics
func (c *HANACache) GetStats(ctx context.Context) (*CacheStats, error) {
	if !c.poolAvailable() {
		return nil, fmt.Errorf("hana pool not configured")
	}

	query := `
		SELECT 
			COUNT(*) as total_entries,
			SUM(access_count) as total_accesses,
			AVG(access_count) as avg_access_count,
			COUNT(CASE WHEN expires_at < CURRENT_TIMESTAMP THEN 1 END) as expired_entries
		FROM cache_entries
	`

	row := c.pool.QueryRow(ctx, query)

	stats := &CacheStats{}
	err := row.Scan(
		&stats.TotalEntries,
		&stats.HitCount,
		&stats.AvgAccessCount,
		&stats.ExpiredEntries,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to get cache stats: %w", err)
	}

	// Calculate hit rate (simplified - in production you'd track this separately)
	if stats.TotalEntries > 0 {
		stats.HitRate = float64(stats.HitCount) / float64(stats.TotalEntries)
	}

	return stats, nil
}

// CleanupExpired removes expired cache entries
func (c *HANACache) CleanupExpired(ctx context.Context) error {
	if !c.poolAvailable() {
		return fmt.Errorf("hana pool not configured")
	}

	query := `
		DELETE FROM cache_entries 
		WHERE expires_at < CURRENT_TIMESTAMP
	`

	result, err := c.pool.Execute(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to cleanup expired entries: %w", err)
	}

	rowsAffected, _ := result.RowsAffected()
	log.Printf("🧹 Cleaned up %d expired cache entries", rowsAffected)

	return nil
}

// CleanupOldEntries removes old cache entries based on access count
func (c *HANACache) CleanupOldEntries(ctx context.Context, olderThanDays int, minAccessCount int) error {
	if !c.poolAvailable() {
		return fmt.Errorf("hana pool not configured")
	}

	query := `
		DELETE FROM cache_entries 
		WHERE created_at < DATEADD(day, -?, CURRENT_TIMESTAMP)
		  AND access_count < ?
	`

	result, err := c.pool.Execute(ctx, query, olderThanDays, minAccessCount)
	if err != nil {
		return fmt.Errorf("failed to cleanup old entries: %w", err)
	}

	rowsAffected, _ := result.RowsAffected()
	log.Printf("🧹 Cleaned up %d old cache entries", rowsAffected)

	return nil
}

// updateAccessStats updates access statistics for a cache entry
func (c *HANACache) updateAccessStats(ctx context.Context, entryID int64) {
	if !c.poolAvailable() {
		return
	}

	query := `
		UPDATE cache_entries 
		SET access_count = access_count + 1,
		    last_accessed = CURRENT_TIMESTAMP
		WHERE id = ?
	`

	_, err := c.pool.Execute(ctx, query, entryID)
	if err != nil {
		log.Printf("Failed to update access stats: %v", err)
	}
}

// GetTopEntries retrieves the most accessed cache entries
func (c *HANACache) GetTopEntries(ctx context.Context, limit int) ([]*CacheEntry, error) {
	if !c.poolAvailable() {
		return nil, fmt.Errorf("hana pool not configured")
	}

	query := `
		SELECT id, cache_key, prompt_hash, model, domain, response,
		       tokens_used, temperature, max_tokens, created_at,
		       expires_at, access_count, last_accessed
		FROM cache_entries 
		WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
		ORDER BY access_count DESC, last_accessed DESC
		LIMIT ?
	`

	rows, err := c.pool.Query(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get top entries: %w", err)
	}
	defer rows.Close()

	var entries []*CacheEntry
	for rows.Next() {
		entry := &CacheEntry{}
		var responseBuf bytes.Buffer
		responseLob := hdbdriver.NewLob(nil, &responseBuf)
		err := rows.Scan(
			&entry.ID,
			&entry.CacheKey,
			&entry.PromptHash,
			&entry.Model,
			&entry.Domain,
			responseLob,
			&entry.TokensUsed,
			&entry.Temperature,
			&entry.MaxTokens,
			&entry.CreatedAt,
			&entry.ExpiresAt,
			&entry.AccessCount,
			&entry.LastAccessed,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan cache entry: %w", err)
		}

		entry.Response = responseBuf.String()

		entries = append(entries, entry)
	}

	return entries, nil
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
