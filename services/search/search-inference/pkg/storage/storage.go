package storage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"time"
)

type PrivacyLevel string

const (
	PrivacyLevelLow    PrivacyLevel = "low"
	PrivacyLevelMedium PrivacyLevel = "medium"
	PrivacyLevelHigh   PrivacyLevel = "high"
)

type PrivacyConfig struct {
	PrivacyLevel PrivacyLevel
}

func DefaultPrivacyConfig() *PrivacyConfig {
	return &PrivacyConfig{PrivacyLevel: PrivacyLevelMedium}
}

func NewPrivacyConfig(level PrivacyLevel) *PrivacyConfig {
	if level == "" {
		level = PrivacyLevelMedium
	}
	return &PrivacyConfig{PrivacyLevel: level}
}

type Document struct {
	ID           string
	Content      string
	Metadata     map[string]interface{}
	PrivacyLevel string
	CreatedAt    time.Time
	UpdatedAt    time.Time
}

type SearchLog struct {
	QueryHash         string
	ResultCount       int
	TopResultID       string
	LatencyMs         int64
	UserIDHash        string
	SessionID         string
	Timestamp         time.Time
	PrivacyBudgetUsed float64
}

var PrivacyBudgetCosts = struct {
	SearchQuery float64
}{
	SearchQuery: 0,
}

func AnonymizeString(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:8])
}

type HANADocumentStore struct{}

func NewHANADocumentStore(dsn string, cfg *PrivacyConfig) (*HANADocumentStore, error) {
	return &HANADocumentStore{}, nil
}

func (h *HANADocumentStore) Close() error { return nil }

func (h *HANADocumentStore) GetDocument(ctx context.Context, id, user, purpose string) (*Document, error) {
	return nil, nil
}

func (h *HANADocumentStore) StoreDocument(ctx context.Context, doc *Document) error { return nil }

func (h *HANADocumentStore) DeleteDocument(ctx context.Context, id string) error { return nil }

type HANASearchLogger struct{}

func NewHANASearchLogger(dsn string, cfg *PrivacyConfig) (*HANASearchLogger, error) {
	return &HANASearchLogger{}, nil
}

func (l *HANASearchLogger) Close() error { return nil }

func (l *HANASearchLogger) LogSearch(ctx context.Context, log *SearchLog) error { return nil }
