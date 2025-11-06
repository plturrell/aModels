package regulatory

import (
	"context"
	"database/sql"
	"fmt"
	"log"
)

// RegulatorySpecSystem integrates all regulatory spec components.
type RegulatorySpecSystem struct {
	extractor        *RegulatorySpecExtractor
	mas610Extractor  *MAS610Extractor
	bcbs239Extractor *BCBS239Extractor
	validationEngine *ValidationEngine
	schemaRepo       *RegulatorySchemaRepository
	logger           *log.Logger
}

// NewRegulatorySpecSystem creates a new regulatory spec system.
func NewRegulatorySpecSystem(
	langextractURL string,
	auditLogger interface{}, // *langextract.AuditLogger
	db *sql.DB,
	logger *log.Logger,
) *RegulatorySpecSystem {
	// Create base extractor
	extractor := NewRegulatorySpecExtractor(langextractURL, nil, logger) // Would pass auditLogger

	// Create specialized extractors
	mas610Extractor := NewMAS610Extractor(extractor, logger)
	bcbs239Extractor := NewBCBS239Extractor(extractor, logger)

	// Create schema repository
	schemaRepo := NewRegulatorySchemaRepository(db, logger)

	// Create validation engine
	validationEngine := NewValidationEngine(schemaRepo, logger)

	return &RegulatorySpecSystem{
		extractor:        extractor,
		mas610Extractor:  mas610Extractor,
		bcbs239Extractor: bcbs239Extractor,
		validationEngine: validationEngine,
		schemaRepo:       schemaRepo,
		logger:           logger,
	}
}

// ExtractMAS610 extracts MAS 610 specifications.
func (rss *RegulatorySpecSystem) ExtractMAS610(ctx context.Context, documentContent string, documentSource string, documentVersion string, user string) (*ExtractionResult, error) {
	return rss.mas610Extractor.ExtractMAS610(ctx, documentContent, documentSource, documentVersion, user)
}

// ExtractBCBS239 extracts BCBS 239 specifications.
func (rss *RegulatorySpecSystem) ExtractBCBS239(ctx context.Context, documentContent string, documentSource string, documentVersion string, user string) (*ExtractionResult, error) {
	return rss.bcbs239Extractor.ExtractBCBS239(ctx, documentContent, documentSource, documentVersion, user)
}

// ValidateAndSave validates a specification and saves it to the repository.
func (rss *RegulatorySpecSystem) ValidateAndSave(ctx context.Context, spec *RegulatorySpec, user string) (*ValidationResult, error) {
	// Validate
	validationResult, err := rss.validationEngine.ValidateSpec(ctx, spec)
	if err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	// Save if valid
	if validationResult.Valid {
		if err := rss.schemaRepo.SaveSchema(ctx, spec, user); err != nil {
			return nil, fmt.Errorf("failed to save schema: %w", err)
		}
	}

	return validationResult, nil
}

// GetSchema retrieves a schema.
func (rss *RegulatorySpecSystem) GetSchema(ctx context.Context, id string) (*RegulatorySpec, error) {
	return rss.schemaRepo.GetSchema(ctx, id)
}

// CompareVersions compares two schema versions.
func (rss *RegulatorySpecSystem) CompareVersions(ctx context.Context, fromID string, toID string) ([]SchemaChange, error) {
	return rss.schemaRepo.CompareSchemas(ctx, fromID, toID)
}

