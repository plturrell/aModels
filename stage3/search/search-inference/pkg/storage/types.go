package storage

import "time"

// Document represents a stored document with metadata.
type Document struct {
	ID           string                 `json:"id"`
	Content      string                 `json:"content"`
	Metadata     map[string]interface{} `json:"metadata"`
	PrivacyLevel string                 `json:"privacy_level"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	AccessCount  int64                  `json:"access_count"`
	LastAccessed time.Time              `json:"last_accessed"`
}

// SearchEmbedding represents a document embedding with privacy controls.
type SearchEmbedding struct {
	ID           int64                  `json:"id"`
	DocumentID   string                 `json:"document_id"`
	Embedding    []float64              `json:"embedding"`
	Content      string                 `json:"content"`
	Metadata     map[string]interface{} `json:"metadata"`
	PrivacyLevel string                 `json:"privacy_level"`
	CreatedAt    time.Time              `json:"created_at"`
}

// SearchLog represents a search query log with privacy information.
type SearchLog struct {
	ID                int64     `json:"id"`
	QueryHash         string    `json:"query_hash"`
	QueryEmbedding    []float64 `json:"query_embedding"`
	ResultCount       int       `json:"result_count"`
	TopResultID       string    `json:"top_result_id"`
	LatencyMs         int64     `json:"latency_ms"`
	UserIDHash        string    `json:"user_id_hash"`
	SessionID         string    `json:"session_id"`
	Timestamp         time.Time `json:"timestamp"`
	PrivacyBudgetUsed float64   `json:"privacy_budget_used"`
}

// SearchClick represents a click-through event captured for analytics.
type SearchClick struct {
	ID         int64     `json:"id"`
	QueryHash  string    `json:"query_hash"`
	DocumentID string    `json:"document_id"`
	Position   int       `json:"position"`
	Timestamp  time.Time `json:"timestamp"`
	SessionID  string    `json:"session_id"`
}
