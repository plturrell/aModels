package regulatory

import (
	"context"
	"fmt"
	"log"
)

// MultiRegulatoryExtractor supports extraction from multiple regulatory frameworks.
type MultiRegulatoryExtractor struct {
	extractors map[string]RegulatoryExtractor
	base       *RegulatorySpecExtractor
	logger     *log.Logger
}

// RegulatoryExtractor interface for different regulatory extractors.
type RegulatoryExtractor interface {
	Extract(ctx context.Context, documentContent, documentSource, documentVersion, user string) (*ExtractionResult, error)
	GetRegulatoryType() string
}

// NewMultiRegulatoryExtractor creates a new multi-regulatory extractor.
func NewMultiRegulatoryExtractor(baseExtractor *RegulatorySpecExtractor, logger *log.Logger) *MultiRegulatoryExtractor {
	extractors := make(map[string]RegulatoryExtractor)
	
	// Register default extractors
	mas610Extractor := NewMAS610Extractor(baseExtractor, logger)
	bcbs239Extractor := NewBCBS239Extractor(baseExtractor, logger)
	
	extractors["mas_610"] = mas610Extractor
	extractors["bcbs_239"] = bcbs239Extractor

	return &MultiRegulatoryExtractor{
		extractors: extractors,
		base:       baseExtractor,
		logger:     logger,
	}
}

// RegisterExtractor registers a new regulatory extractor.
func (mre *MultiRegulatoryExtractor) RegisterExtractor(extractor RegulatoryExtractor) {
	mre.extractors[extractor.GetRegulatoryType()] = extractor
	if mre.logger != nil {
		mre.logger.Printf("Registered regulatory extractor: %s", extractor.GetRegulatoryType())
	}
}

// Extract extracts from a regulatory document using the appropriate extractor.
func (mre *MultiRegulatoryExtractor) Extract(ctx context.Context, regulatoryType, documentContent, documentSource, documentVersion, user string) (*ExtractionResult, error) {
	extractor, exists := mre.extractors[regulatoryType]
	if !exists {
		return nil, fmt.Errorf("unsupported regulatory type: %s", regulatoryType)
	}

	return extractor.Extract(ctx, documentContent, documentSource, documentVersion, user)
}

// ListSupportedRegulatoryTypes returns a list of supported regulatory types.
func (mre *MultiRegulatoryExtractor) ListSupportedRegulatoryTypes() []string {
	types := make([]string, 0, len(mre.extractors))
	for t := range mre.extractors {
		types = append(types, t)
	}
	return types
}

// GenericRegulatoryExtractor is a generic extractor for regulatory types without specialized extractors.
type GenericRegulatoryExtractor struct {
	regulatoryType string
	base           *RegulatorySpecExtractor
	logger         *log.Logger
}

// NewGenericRegulatoryExtractor creates a new generic regulatory extractor.
func NewGenericRegulatoryExtractor(regulatoryType string, baseExtractor *RegulatorySpecExtractor, logger *log.Logger) *GenericRegulatoryExtractor {
	return &GenericRegulatoryExtractor{
		regulatoryType: regulatoryType,
		base:           baseExtractor,
		logger:         logger,
	}
}

// GetRegulatoryType returns the regulatory type.
func (gre *GenericRegulatoryExtractor) GetRegulatoryType() string {
	return gre.regulatoryType
}

// Extract extracts from a regulatory document using the base extractor.
func (gre *GenericRegulatoryExtractor) Extract(ctx context.Context, documentContent, documentSource, documentVersion, user string) (*ExtractionResult, error) {
	req := ExtractionRequest{
		DocumentContent: documentContent,
		DocumentType:    gre.regulatoryType,
		DocumentSource:  documentSource,
		DocumentVersion: documentVersion,
		ExtractorType:   "generic",
		User:            user,
	}

	return gre.base.ExtractSpec(ctx, req)
}

