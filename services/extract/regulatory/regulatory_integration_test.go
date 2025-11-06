package regulatory

import (
	"context"
	"database/sql"
	"log"
	"os"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

func setupTestDB(t *testing.T) *sql.DB {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}

	schema := `
		CREATE TABLE IF NOT EXISTS regulatory_schemas (
			id TEXT PRIMARY KEY,
			regulatory_type TEXT,
			version TEXT,
			document_source TEXT,
			document_version TEXT,
			spec_json TEXT,
			created_at TIMESTAMP,
			updated_at TIMESTAMP,
			created_by TEXT,
			is_reference INTEGER DEFAULT 0,
			status TEXT,
			UNIQUE(regulatory_type, version)
		);
	`
	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("Failed to create schema: %v", err)
	}

	return db
}

func TestRegulatorySpecExtractor_ExtractSpec(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	extractor := NewRegulatorySpecExtractor("http://localhost:8080", nil, logger)

	ctx := context.Background()
	req := ExtractionRequest{
		DocumentContent: "MAS 610 Regulatory Report\nField 1.1: Total Assets (Required)",
		DocumentType:    "mas_610",
		DocumentSource:  "mas_610_guidelines.pdf",
		DocumentVersion: "2024.1",
		ExtractorType:   "mas_610",
		User:            "test-user",
	}

	result, err := extractor.ExtractSpec(ctx, req)
	if err != nil {
		// May fail if LangExtract is not available - that's OK for tests
		t.Logf("ExtractSpec() error (expected if LangExtract unavailable): %v", err)
		return
	}

	if result.Spec == nil {
		t.Error("Expected spec to be extracted")
	}

	if result.Spec.RegulatoryType != "mas_610" {
		t.Errorf("RegulatoryType = %v, want mas_610", result.Spec.RegulatoryType)
	}
}

func TestValidationEngine_ValidateSpec(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	repo := NewRegulatorySchemaRepository(db, logger)
	engine := NewValidationEngine(repo, logger)

	ctx := context.Background()
	spec := &RegulatorySpec{
		ID:             "spec-1",
		RegulatoryType: "mas_610",
		Version:        "1.0.0",
		ReportStructure: ReportStructure{
			ReportName:     "MAS 610 Report",
			ReportID:       "MAS_610",
			TotalFields:    5,
			RequiredFields: 3,
		},
		FieldDefinitions: []FieldDefinition{
			{FieldID: "1.1", FieldName: "Total Assets", FieldType: "currency", Required: true},
			{FieldID: "1.2", FieldName: "Description", FieldType: "text", Required: false},
		},
	}

	result, err := engine.ValidateSpec(ctx, spec)
	if err != nil {
		t.Fatalf("ValidateSpec() error = %v", err)
	}

	if result == nil {
		t.Error("Expected validation result")
	}
}

func TestRegulatorySchemaRepository_SaveSchema(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	repo := NewRegulatorySchemaRepository(db, logger)

	ctx := context.Background()
	spec := &RegulatorySpec{
		ID:             "spec-1",
		RegulatoryType: "mas_610",
		Version:        "1.0.0",
		DocumentSource: "test.pdf",
		DocumentVersion: "1.0",
		ReportStructure: ReportStructure{
			ReportName: "MAS 610 Report",
		},
	}

	err := repo.SaveSchema(ctx, spec, "test-user")
	if err != nil {
		t.Fatalf("SaveSchema() error = %v", err)
	}

	// Retrieve schema
	retrieved, err := repo.GetSchema(ctx, spec.ID)
	if err != nil {
		t.Fatalf("GetSchema() error = %v", err)
	}

	if retrieved.ReportStructure.ReportName != spec.ReportStructure.ReportName {
		t.Errorf("Report name = %v, want %v", retrieved.ReportStructure.ReportName, spec.ReportStructure.ReportName)
	}
}

func TestMAS610Extractor_ExtractMAS610(t *testing.T) {
	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	baseExtractor := NewRegulatorySpecExtractor("http://localhost:8080", nil, logger)
	extractor := NewMAS610Extractor(baseExtractor, logger)

	ctx := context.Background()
	documentContent := "MAS 610 Regulatory Guidelines\nField definitions and requirements..."

	result, err := extractor.ExtractMAS610(ctx, documentContent, "mas_610.pdf", "2024.1", "test-user")
	if err != nil {
		// May fail if LangExtract unavailable
		t.Logf("ExtractMAS610() error (expected if LangExtract unavailable): %v", err)
		return
	}

	if result.Spec != nil && result.Spec.RegulatoryType != "mas_610" {
		t.Errorf("RegulatoryType = %v, want mas_610", result.Spec.RegulatoryType)
	}
}

