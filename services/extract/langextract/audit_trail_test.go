package langextract

import (
	"context"
	"log"
	"os"
	"testing"
	"time"
)

func TestAuditLogger_LogExtraction_NoStore(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	auditLogger := NewAuditLogger(nil, logger)

	ctx := context.Background()
	auditEntry := CreateAuditEntry(
		"test-extraction-1",
		"test-user",
		"test-context",
		"extract",
		ExtractionRequest{PromptDescription: "Test", ModelID: "test-model"},
		ExtractionResponse{Extractions: []ExtractionResult{}, ProcessingTime: 50 * time.Millisecond},
		50 * time.Millisecond,
	)

	if err := auditLogger.LogExtraction(ctx, auditEntry); err != nil {
		t.Errorf("LogExtraction() with nil store should not error, got: %v", err)
	}
}

func TestCreateAuditEntry(t *testing.T) {
	entry := CreateAuditEntry(
		"test-1",
		"user",
		"context",
		"extract",
		ExtractionRequest{PromptDescription: "Test", ModelID: "model"},
		ExtractionResponse{ProcessingTime: 50 * time.Millisecond},
		50 * time.Millisecond,
	)

	if entry.ExtractionID != "test-1" {
		t.Errorf("ExtractionID = %v, want test-1", entry.ExtractionID)
	}
}

func TestAuditTrail_JSONSerialization(t *testing.T) {
	entry := CreateAuditEntry(
		"test-1",
		"user",
		"context",
		"extract",
		ExtractionRequest{PromptDescription: "Test", ModelID: "model"},
		ExtractionResponse{ProcessingTime: 50 * time.Millisecond},
		50 * time.Millisecond,
	)

	jsonData, err := entry.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON() error = %v", err)
	}

	deserialized, err := FromJSON(jsonData)
	if err != nil {
		t.Fatalf("FromJSON() error = %v", err)
	}

	if deserialized.ExtractionID != entry.ExtractionID {
		t.Errorf("FromJSON() ExtractionID = %v, want %v", deserialized.ExtractionID, entry.ExtractionID)
	}
}
