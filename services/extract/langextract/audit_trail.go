package langextract

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// AuditTrail represents an audit trail entry for LangExtract operations.
type AuditTrail struct {
	ID                string                 `json:"id"`
	ExtractionID      string                 `json:"extraction_id"`
	Timestamp         time.Time              `json:"timestamp"`
	User              string                 `json:"user"`
	Context           string                 `json:"context,omitempty"` // e.g., "regulatory", "mas_610", "bcbs239"
	Operation         string                 `json:"operation"`          // "extract", "validate", "store"
	Request           ExtractionRequest      `json:"request"`
	Response          ExtractionResponse     `json:"response"`
	ProcessingTime    time.Duration          `json:"processing_time_ms"`
	Confidence        float64                `json:"confidence,omitempty"`
	SchemaVersion     string                 `json:"schema_version,omitempty"`
	QualityMetrics    QualityMetrics         `json:"quality_metrics,omitempty"`
	ResourceUsage     ResourceUsage          `json:"resource_usage,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

// ExtractionRequest represents the request that was audited.
type ExtractionRequest struct {
	Document          string                 `json:"document,omitempty"`
	Documents         []string               `json:"documents,omitempty"`
	PromptDescription string                 `json:"prompt_description"`
	ModelID           string                 `json:"model_id"`
	Examples          []interface{}           `json:"examples,omitempty"`
	Parameters        map[string]interface{} `json:"parameters,omitempty"`
}

// ExtractionResponse represents the response that was audited.
type ExtractionResponse struct {
	Extractions       []ExtractionResult     `json:"extractions"`
	Error             string                 `json:"error,omitempty"`
	ProcessingTime    time.Duration          `json:"processing_time_ms"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

// ExtractionResult represents a single extraction result.
type ExtractionResult struct {
	ExtractionClass string                 `json:"extraction_class"`
	ExtractionText  string                 `json:"extraction_text"`
	Attributes      map[string]interface{} `json:"attributes,omitempty"`
	StartIndex      *int                   `json:"start_index,omitempty"`
	EndIndex        *int                   `json:"end_index,omitempty"`
	Confidence      float64                `json:"confidence,omitempty"`
}

// QualityMetrics represents quality metrics for an extraction.
type QualityMetrics struct {
	Completeness  float64 `json:"completeness"`
	Accuracy      float64 `json:"accuracy"`
	Consistency   float64 `json:"consistency"`
	OverallScore  float64 `json:"overall_score"`
}

// ResourceUsage represents resource usage for an extraction.
type ResourceUsage struct {
	TokensUsed      int     `json:"tokens_used"`
	ProcessingTime  float64 `json:"processing_time_seconds"`
	MemoryUsed      int64   `json:"memory_used_bytes,omitempty"`
}

// AuditLogger logs audit trail entries.
type AuditLogger struct {
	store  AuditStore
	logger *log.Logger
}

// NewAuditLogger creates a new audit logger.
func NewAuditLogger(store AuditStore, logger *log.Logger) *AuditLogger {
	return &AuditLogger{
		store:  store,
		logger: logger,
	}
}

// LogExtraction logs an extraction operation to the audit trail.
func (al *AuditLogger) LogExtraction(ctx context.Context, auditEntry *AuditTrail) error {
	if al.store == nil {
		// No audit store configured, just log
		if al.logger != nil {
			al.logger.Printf("Audit log (no store): %s - %s by %s", auditEntry.Operation, auditEntry.ExtractionID, auditEntry.User)
		}
		return nil
	}

	if err := al.store.SaveAuditEntry(ctx, auditEntry); err != nil {
		if al.logger != nil {
			al.logger.Printf("Failed to save audit entry: %v", err)
		}
		return fmt.Errorf("failed to save audit entry: %w", err)
	}

	if al.logger != nil {
		al.logger.Printf("Audit log saved: %s - %s by %s", auditEntry.Operation, auditEntry.ExtractionID, auditEntry.User)
	}

	return nil
}

// CreateAuditEntry creates a new audit trail entry from extraction request/response.
func CreateAuditEntry(
	extractionID string,
	user string,
	context string,
	operation string,
	request ExtractionRequest,
	response ExtractionResponse,
	processingTime time.Duration,
) *AuditTrail {
	return &AuditTrail{
		ID:             fmt.Sprintf("audit-%s-%d", extractionID, time.Now().UnixNano()),
		ExtractionID:   extractionID,
		Timestamp:      time.Now(),
		User:           user,
		Context:        context,
		Operation:      operation,
		Request:        request,
		Response:       response,
		ProcessingTime: processingTime,
	}
}

// EnrichWithMetadata enriches an audit entry with additional metadata.
func (at *AuditTrail) EnrichWithMetadata(metadata map[string]interface{}) {
	if at.Metadata == nil {
		at.Metadata = make(map[string]interface{})
	}
	for k, v := range metadata {
		at.Metadata[k] = v
	}
}

// ToJSON converts an audit trail entry to JSON.
func (at *AuditTrail) ToJSON() ([]byte, error) {
	return json.MarshalIndent(at, "", "  ")
}

// FromJSON creates an audit trail entry from JSON.
func FromJSON(data []byte) (*AuditTrail, error) {
	var at AuditTrail
	if err := json.Unmarshal(data, &at); err != nil {
		return nil, fmt.Errorf("failed to unmarshal audit trail: %w", err)
	}
	return &at, nil
}

