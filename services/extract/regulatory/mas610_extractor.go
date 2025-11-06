package regulatory

import (
	"context"
	"fmt"
	"log"
)

// MAS610Extractor provides specialized extraction for MAS 610.
type MAS610Extractor struct {
	baseExtractor *RegulatorySpecExtractor
	logger        *log.Logger
}

// NewMAS610Extractor creates a new MAS 610 extractor.
func NewMAS610Extractor(baseExtractor *RegulatorySpecExtractor, logger *log.Logger) *MAS610Extractor {
	return &MAS610Extractor{
		baseExtractor: baseExtractor,
		logger:        logger,
	}
}

// ExtractMAS610 extracts MAS 610 specifications.
func (me *MAS610Extractor) ExtractMAS610(ctx context.Context, documentContent string, documentSource string, documentVersion string, user string) (*ExtractionResult, error) {
	req := ExtractionRequest{
		DocumentContent: documentContent,
		DocumentType:    "mas_610",
		DocumentSource:  documentSource,
		DocumentVersion: documentVersion,
		ExtractorType:   "mas_610",
		User:            user,
	}

	result, err := me.baseExtractor.ExtractSpec(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("MAS 610 extraction failed: %w", err)
	}

	// Enhance with MAS 610 specific processing
	if result.Spec != nil {
		me.enhanceMAS610Spec(result.Spec)
	}

	return result, nil
}

// enhanceMAS610Spec enhances spec with MAS 610 specific fields.
func (me *MAS610Extractor) enhanceMAS610Spec(spec *RegulatorySpec) {
	// MAS 610 specific enhancements
	if spec.ReportStructure.ReportName == "" {
		spec.ReportStructure.ReportName = "MAS 610 Regulatory Report"
	}
	if spec.ReportStructure.ReportID == "" {
		spec.ReportStructure.ReportID = "MAS_610"
	}

	// Add MAS 610 specific metadata
	if spec.Metadata == nil {
		spec.Metadata = make(map[string]interface{})
	}
	spec.Metadata["regulatory_authority"] = "Monetary Authority of Singapore"
	spec.Metadata["regulation_type"] = "banking_regulation"
}

