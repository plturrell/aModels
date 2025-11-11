//go:build !hana

package storage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/hanapool"
)

var errHANANotEnabled = errors.New("hana integration requires build tag 'hana'")

// Re-exported structs for non-HANA builds so the package type-checks without the tag.
type CacheEntry struct {
	ID           int64
	CacheKey     string
	PromptHash   string
	Model        string
	Domain       string
	Response     string
	TokensUsed   int
	Temperature  float64
	MaxTokens    int
	CreatedAt    time.Time
	ExpiresAt    time.Time
	AccessCount  int
	LastAccessed time.Time
}

type CacheStats struct {
	TotalEntries   int64
	HitCount       int64
	MissCount      int64
	HitRate        float64
	AvgAccessCount float64
	ExpiredEntries int64
}

func NewHANACache(*hanapool.Pool) *HANACache { return &HANACache{} }

type HANACache struct{}

func (*HANACache) poolAvailable() bool                { return false }
func (*HANACache) CreateTables(context.Context) error { return errHANANotEnabled }
func (*HANACache) GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string {
	normalized := strings.ToLower(strings.TrimSpace(prompt))
	base := sha256.Sum256([]byte(normalized))
	keyData := fmt.Sprintf("%s:%s:%s:%.2f:%d:%.3f:%d", hex.EncodeToString(base[:]), model, domain, temperature, maxTokens, topP, topK)
	key := sha256.Sum256([]byte(keyData))
	return hex.EncodeToString(key[:])
}
func (*HANACache) Get(context.Context, string) (*CacheEntry, error) { return nil, errHANANotEnabled }
func (*HANACache) Set(context.Context, *CacheEntry) error           { return errHANANotEnabled }
func (*HANACache) FindSimilar(context.Context, string, string, string, float64) ([]*CacheEntry, error) {
	return nil, errHANANotEnabled
}
func (*HANACache) GetStats(context.Context) (*CacheStats, error)     { return nil, errHANANotEnabled }
func (*HANACache) CleanupExpired(context.Context) error              { return errHANANotEnabled }
func (*HANACache) CleanupOldEntries(context.Context, int, int) error { return errHANANotEnabled }
func (*HANACache) GetTopEntries(context.Context, int) ([]*CacheEntry, error) {
	return nil, errHANANotEnabled
}

// Semantic cache stubs.
type SemanticCacheEntry struct {
	ID              int64
	CacheKey        string
	PromptHash      string
	SemanticHash    string
	Model           string
	Domain          string
	Prompt          string
	Response        string
	TokensUsed      int
	Temperature     float64
	MaxTokens       int
	SimilarityScore float64
	CreatedAt       time.Time
	ExpiresAt       time.Time
	AccessCount     int
	LastAccessed    time.Time
	Metadata        map[string]string
	Tags            []string
}

type SemanticCacheConfig struct {
	DefaultTTL          time.Duration
	SimilarityThreshold float64
	MaxEntries          int
	CleanupInterval     time.Duration
	EnableVectorSearch  bool
	EnableFuzzyMatching bool
}

type SemanticCacheStats struct {
	TotalEntries       int64
	HitCount           int64
	MissCount          int64
	SemanticHitCount   int64
	HitRate            float64
	SemanticHitRate    float64
	AvgSimilarityScore float64
	ExpiredEntries     int64
	ByModel            map[string]int64
	ByDomain           map[string]int64
	TopTags            map[string]int64
}

type SemanticCache struct{}

func NewSemanticCache(*hanapool.Pool, *SemanticCacheConfig) *SemanticCache { return &SemanticCache{} }
func (*SemanticCache) CreateTables(context.Context) error                  { return errHANANotEnabled }
func (*SemanticCache) Get(context.Context, string) (*SemanticCacheEntry, error) {
	return nil, errHANANotEnabled
}
func (*SemanticCache) Set(context.Context, *SemanticCacheEntry) error { return errHANANotEnabled }
func (*SemanticCache) FindSemanticSimilar(context.Context, string, string, string, float64, int) ([]*SemanticCacheEntry, error) {
	return nil, errHANANotEnabled
}
func (*SemanticCache) FindSimilar(context.Context, string, string, string) ([]*SemanticCacheEntry, error) {
	return nil, errHANANotEnabled
}
func (*SemanticCache) GetStats(context.Context) (*SemanticCacheStats, error) {
	return nil, errHANANotEnabled
}
func (*SemanticCache) CleanupExpired(context.Context) error              { return errHANANotEnabled }
func (*SemanticCache) CleanupOldEntries(context.Context, int, int) error { return errHANANotEnabled }
func (*SemanticCache) GetTopEntries(context.Context, int) ([]*SemanticCacheEntry, error) {
	return nil, errHANANotEnabled
}
func (*SemanticCache) GetByTags(context.Context, []string, int) ([]*SemanticCacheEntry, error) {
	return nil, errHANANotEnabled
}
func (*SemanticCache) GenerateCacheKey(prompt, model, domain string, temperature float64, maxTokens int, topP float64, topK int) string {
	normalized := strings.ToLower(strings.TrimSpace(prompt))
	base := sha256.Sum256([]byte(normalized))
	keyData := fmt.Sprintf("%s:%s:%s:%.2f:%d:%.3f:%d", hex.EncodeToString(base[:]), model, domain, temperature, maxTokens, topP, topK)
	sum := sha256.Sum256([]byte(keyData))
	return hex.EncodeToString(sum[:])
}

func (*SemanticCache) GenerateSemanticHash(prompt string) string {
	normalized := strings.ToLower(strings.TrimSpace(prompt))
	sum := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(sum[:])
}

func (*SemanticCache) VectorSearchEnabled() bool { return false }

// Logger stubs.
type InferenceLog struct {
	RequestID    string
	Model        string
	Domain       string
	Prompt       string
	Response     string
	TokensUsed   int
	LatencyMs    int64
	Temperature  float64
	MaxTokens    int
	CacheHit     bool
	UserID       string
	SessionID    string
	RequestTime  time.Time
	ResponseTime time.Time
	Error        string
	Metadata     map[string]interface{}
}

type ModelMetrics struct {
	Model         string
	TotalRequests int64
	TotalTokens   int64
	AvgLatencyMs  float64
	CacheHitRate  float64
	ErrorRate     float64
	LastUpdated   time.Time
}

type HANALogger struct{}

func NewHANALogger(*hanapool.Pool) *HANALogger                        { return &HANALogger{} }
func (*HANALogger) CreateTables(context.Context) error                { return errHANANotEnabled }
func (*HANALogger) LogInference(context.Context, *InferenceLog) error { return errHANANotEnabled }
func (*HANALogger) GetModelMetrics(context.Context, string) (*ModelMetrics, error) {
	return nil, errHANANotEnabled
}
func (*HANALogger) GetRecentInferences(context.Context, int) ([]InferenceLog, error) {
	return nil, errHANANotEnabled
}
func (*HANALogger) CleanupOldLogs(context.Context, int) error { return errHANANotEnabled }
func (*HANALogger) Close() error                              { return nil }
