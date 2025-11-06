package regulatory

import (
	"context"
	"fmt"
	"log"
)

// BCBS239Extractor provides specialized extraction for BCBS 239.
type BCBS239Extractor struct {
	baseExtractor *RegulatorySpecExtractor
	logger        *log.Logger
}

// NewBCBS239Extractor creates a new BCBS 239 extractor.
func NewBCBS239Extractor(baseExtractor *RegulatorySpecExtractor, logger *log.Logger) *BCBS239Extractor {
	return &BCBS239Extractor{
		baseExtractor: baseExtractor,
		logger:        logger,
	}
}

// ExtractBCBS239 extracts BCBS 239 specifications.
func (be *BCBS239Extractor) ExtractBCBS239(ctx context.Context, documentContent string, documentSource string, documentVersion string, user string) (*ExtractionResult, error) {
	req := ExtractionRequest{
		DocumentContent: documentContent,
		DocumentType:    "bcbs_239",
		DocumentSource:  documentSource,
		DocumentVersion: documentVersion,
		ExtractorType:   "bcbs_239",
		User:            user,
	}

	result, err := be.baseExtractor.ExtractSpec(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("BCBS 239 extraction failed: %w", err)
	}

	// Enhance with BCBS 239 specific processing
	if result.Spec != nil {
		be.enhanceBCBS239Spec(result.Spec)
	}

	return result, nil
}

// enhanceBCBS239Spec enhances spec with BCBS 239 specific fields.
func (be *BCBS239Extractor) enhanceBCBS239Spec(spec *RegulatorySpec) {
	// BCBS 239 specific enhancements
	if spec.ReportStructure.ReportName == "" {
		spec.ReportStructure.ReportName = "BCBS 239 Risk Data Aggregation Report"
	}
	if spec.ReportStructure.ReportID == "" {
		spec.ReportStructure.ReportID = "BCBS_239"
	}

	// Add BCBS 239 specific metadata
	if spec.Metadata == nil {
		spec.Metadata = make(map[string]interface{})
	}
	spec.Metadata["regulatory_authority"] = "Basel Committee on Banking Supervision"
	spec.Metadata["regulation_type"] = "risk_data_aggregation"
}

