package regulatory

import (
	"context"
	// "encoding/json" // unused - removed to fix compilation
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/extract/langextract"
)

// RegulatorySpecExtractor extracts specifications from regulatory documents.
type RegulatorySpecExtractor struct {
	langextractURL string
	auditLogger    *langextract.AuditLogger
	httpClient     *http.Client
	logger         *log.Logger
}

// RegulatorySpec represents an extracted regulatory specification.
type RegulatorySpec struct {
	ID                string
	RegulatoryType    string // "mas_610", "bcbs_239", "generic"
	Version           string
	DocumentSource    string
	DocumentVersion   string
	ExtractedAt       time.Time
	ExtractedBy       string
	Confidence        float64
	ReportStructure   ReportStructure
	FieldDefinitions  []FieldDefinition
	ValidationRules   []ValidationRule
	CalculationRules  []CalculationRule
	SubmissionRules   SubmissionRules
	Metadata          map[string]interface{}
}

// ReportStructure defines the structure of a regulatory report.
type ReportStructure struct {
	ReportName        string
	ReportID          string
	ReportType        string
	Sections          []ReportSection
	TotalFields       int
	RequiredFields    int
}

// ReportSection represents a section in a regulatory report.
type ReportSection struct {
	SectionID       string
	SectionName     string
	SectionNumber   string
	Description     string
	Fields          []string
	Required        bool
	Subsections     []ReportSection
}

// FieldDefinition defines a field in a regulatory report.
type FieldDefinition struct {
	FieldID          string
	FieldName        string
	FieldType        string // "text", "number", "date", "currency", "percentage"
	Required         bool
	ValidationRules  []string
	Description      string
	Source           string // "regulatory_document", "derived", "calculated"
	DataMapping      string // Reference to data product field
	RegulatoryRef    string // Reference to regulatory document section
}

// ValidationRule defines a validation rule for regulatory data.
type ValidationRule struct {
	RuleID          string
	RuleName        string
	RuleType        string // "range", "format", "calculation", "cross_field"
	Description     string
	Expression      string
	ErrorMessage    string
	Severity        string // "error", "warning"
	RegulatoryRef   string
}

// CalculationRule defines a calculation rule for regulatory reporting.
type CalculationRule struct {
	RuleID          string
	RuleName        string
	Formula         string
	Description     string
	InputFields     []string
	OutputField     string
	RegulatoryRef   string
}

// SubmissionRules defines rules for submitting regulatory reports.
type SubmissionRules struct {
	Frequency        string // "daily", "weekly", "monthly", "quarterly", "annual"
	Deadline         string
	Format           string // "xbrl", "csv", "xml", "json"
	ValidationLevel  string // "strict", "moderate", "lenient"
	RequiredAttachments []string
}

// ExtractionRequest represents a request to extract regulatory specs.
type ExtractionRequest struct {
	DocumentContent  string
	DocumentType     string // "mas_610", "bcbs_239"
	DocumentSource   string
	DocumentVersion  string
	ExtractorType    string
	User             string
}

// ExtractionResult represents the result of an extraction.
type ExtractionResult struct {
	Spec            *RegulatorySpec
	ExtractionID    string
	Confidence      float64
	Warnings        []string
	Errors          []string
	ProcessingTime  time.Duration
}

// NewRegulatorySpecExtractor creates a new regulatory spec extractor.
func NewRegulatorySpecExtractor(
	langextractURL string,
	auditLogger *langextract.AuditLogger,
	logger *log.Logger,
) *RegulatorySpecExtractor {
	return &RegulatorySpecExtractor{
		langextractURL: langextractURL,
		auditLogger:    auditLogger,
		httpClient:     &http.Client{Timeout: 120 * time.Second},
		logger:         logger,
	}
}

// ExtractSpec extracts a regulatory specification from a document.
func (rse *RegulatorySpecExtractor) ExtractSpec(ctx context.Context, req ExtractionRequest) (*ExtractionResult, error) {
	startTime := time.Now()
	extractionID := fmt.Sprintf("reg-extract-%d", time.Now().UnixNano())

	if rse.logger != nil {
		rse.logger.Printf("Extracting regulatory spec from %s document", req.DocumentType)
	}

	// Build LangExtract prompt based on regulatory type
	prompt := rse.buildExtractionPrompt(req.DocumentType)
	
	// Build LangExtract request
	langextractReq := langextract.ExtractionRequest{
		Document:          req.DocumentContent,
		PromptDescription: prompt,
		ModelID:           "sap-rpt-1-oss-main", // Domain-specific model
		Examples:          rse.getExamples(req.DocumentType),
		Parameters: map[string]interface{}{
			"temperature": 0.1, // Low temperature for accuracy
			"max_tokens":  4000,
		},
	}

	// Call LangExtract
	response, err := rse.callLangExtract(ctx, langextractReq)
	if err != nil {
		return nil, fmt.Errorf("LangExtract extraction failed: %w", err)
	}

	// Parse extraction results into structured spec
	spec, err := rse.parseExtractionResponse(response, req)
	if err != nil {
		return nil, fmt.Errorf("failed to parse extraction: %w", err)
	}

	spec.ID = fmt.Sprintf("spec-%s-%s-%d", req.DocumentType, req.DocumentVersion, time.Now().UnixNano())
	spec.ExtractedAt = time.Now()
	spec.ExtractedBy = req.User

	// Calculate confidence
	confidence := rse.calculateConfidence(spec, response)

	processingTime := time.Since(startTime)

	// Log to audit trail
	if rse.auditLogger != nil {
		auditEntry := langextract.CreateAuditEntry(
			extractionID,
			req.User,
			req.DocumentType,
			"regulatory_spec_extraction",
			langextract.ExtractionRequest{
				PromptDescription: prompt,
				ModelID:           langextractReq.ModelID,
			},
			langextract.ExtractionResponse{
				Extractions:    convertToExtractionResults(response),
				ProcessingTime:  processingTime,
			},
			processingTime,
		)
		auditEntry.Confidence = confidence
		auditEntry.Metadata = map[string]interface{}{
			"regulatory_type": req.DocumentType,
			"document_source": req.DocumentSource,
			"document_version": req.DocumentVersion,
		}

		rse.auditLogger.LogExtraction(ctx, auditEntry)
	}

	result := &ExtractionResult{
		Spec:           spec,
		ExtractionID:   extractionID,
		Confidence:     confidence,
		ProcessingTime: processingTime,
	}

	return result, nil
}

// buildExtractionPrompt builds a domain-specific extraction prompt.
func (rse *RegulatorySpecExtractor) buildExtractionPrompt(regulatoryType string) string {
	switch regulatoryType {
	case "mas_610":
		return rse.buildMAS610Prompt()
	case "bcbs_239":
		return rse.buildBCBS239Prompt()
	default:
		return rse.buildGenericPrompt()
	}
}

// buildMAS610Prompt builds the extraction prompt for MAS 610.
func (rse *RegulatorySpecExtractor) buildMAS610Prompt() string {
	return `Extract regulatory reporting specifications from this MAS 610 (Monetary Authority of Singapore) regulatory document.

Extract the following structured information:
1. Report Structure: Report name, sections, subsections, field organization
2. Field Definitions: For each field, extract:
   - Field ID/Code
   - Field Name
   - Data Type (text, number, date, currency, percentage)
   - Required/Optional status
   - Description
   - Validation rules
   - Regulatory reference
3. Validation Rules: Extract all validation rules including:
   - Range validations
   - Format validations
   - Cross-field validations
   - Calculation validations
4. Calculation Rules: Extract formulas and calculations
5. Submission Rules: Frequency, deadlines, format requirements

Return the extracted information as a structured JSON object matching the RegulatorySpec schema.
Be precise and accurate - this is for regulatory compliance.`
}

// buildBCBS239Prompt builds the extraction prompt for BCBS 239.
func (rse *RegulatorySpecExtractor) buildBCBS239Prompt() string {
	return `Extract regulatory reporting specifications from this BCBS 239 (Basel Committee on Banking Supervision - Principles for Effective Risk Data Aggregation) regulatory document.

Extract the following structured information:
1. Report Structure: Report name, sections, subsections, field organization
2. Field Definitions: For each field, extract:
   - Field ID/Code
   - Field Name
   - Data Type (text, number, date, currency, percentage)
   - Required/Optional status
   - Description
   - Validation rules
   - Regulatory reference
3. Validation Rules: Extract all validation rules including:
   - Data quality requirements
   - Completeness requirements
   - Accuracy requirements
   - Timeliness requirements
4. Calculation Rules: Extract formulas and calculations
5. Submission Rules: Frequency, deadlines, format requirements

Return the extracted information as a structured JSON object matching the RegulatorySpec schema.
Be precise and accurate - this is for regulatory compliance.`
}

// buildGenericPrompt builds a generic extraction prompt.
func (rse *RegulatorySpecExtractor) buildGenericPrompt() string {
	return `Extract regulatory reporting specifications from this regulatory document.

Extract the following structured information:
1. Report Structure: Report name, sections, subsections
2. Field Definitions: Field IDs, names, types, requirements
3. Validation Rules: All validation requirements
4. Calculation Rules: Formulas and calculations
5. Submission Rules: Frequency, deadlines, formats

Return as structured JSON matching the RegulatorySpec schema.`
}

// getExamples returns example extractions for few-shot learning.
func (rse *RegulatorySpecExtractor) getExamples(regulatoryType string) []interface{} {
	// Return example extraction patterns for few-shot learning
	return []interface{}{
		map[string]interface{}{
			"text": "Field 1.1: Total Assets (Required) - Report total assets as of the reporting date. Format: Currency, Decimal places: 2",
			"extractions": []map[string]interface{}{
				{
					"extraction_class": "field_definition",
					"extraction_text": "Field 1.1: Total Assets",
					"attributes": map[string]interface{}{
						"field_id": "1.1",
						"field_name": "Total Assets",
						"field_type": "currency",
						"required": true,
						"format": "decimal_places:2",
					},
				},
			},
		},
	}
}

// callLangExtract calls the LangExtract service.
func (rse *RegulatorySpecExtractor) callLangExtract(ctx context.Context, req langextract.ExtractionRequest) (*langextract.ExtractionResponse, error) {
	// In production, would make actual HTTP call to LangExtract
	// For now, return a structured response
	
	// This would be the actual implementation:
	// body, _ := json.Marshal(req)
	// httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, rse.langextractURL+"/extract", bytes.NewReader(body))
	// resp, err := rse.httpClient.Do(httpReq)
	// ... parse response
	
	// For now, return a mock response structure
	return &langextract.ExtractionResponse{
		Extractions: []langextract.ExtractionResult{},
		ProcessingTime: 2 * time.Second,
	}, nil
}

// parseExtractionResponse parses LangExtract response into RegulatorySpec.
func (rse *RegulatorySpecExtractor) parseExtractionResponse(response *langextract.ExtractionResponse, req ExtractionRequest) (*RegulatorySpec, error) {
	// Parse extraction results into structured spec
	// In production, would parse JSON from LangExtract response
	
	spec := &RegulatorySpec{
		RegulatoryType:  req.DocumentType,
		Version:          "1.0.0",
		DocumentSource:   req.DocumentSource,
		DocumentVersion:  req.DocumentVersion,
		ReportStructure: ReportStructure{
			ReportName:     fmt.Sprintf("%s Report", req.DocumentType),
			ReportID:       req.DocumentType,
			ReportType:     "regulatory",
			Sections:       []ReportSection{},
			TotalFields:    0,
			RequiredFields: 0,
		},
		FieldDefinitions: []FieldDefinition{},
		ValidationRules:  []ValidationRule{},
		CalculationRules: []CalculationRule{},
		SubmissionRules: SubmissionRules{
			Frequency:       "monthly",
			Format:          "xbrl",
			ValidationLevel: "strict",
		},
		Metadata: map[string]interface{}{
			"extracted_from": req.DocumentSource,
		},
	}

	// Parse extractions into structured fields
	for _, extraction := range response.Extractions {
		if extraction.ExtractionClass == "field_definition" {
			field := rse.parseFieldDefinition(extraction)
			if field != nil {
				spec.FieldDefinitions = append(spec.FieldDefinitions, *field)
				spec.ReportStructure.TotalFields++
				if field.Required {
					spec.ReportStructure.RequiredFields++
				}
			}
		} else if extraction.ExtractionClass == "validation_rule" {
			rule := rse.parseValidationRule(extraction)
			if rule != nil {
				spec.ValidationRules = append(spec.ValidationRules, *rule)
			}
		}
	}

	return spec, nil
}

// parseFieldDefinition parses a field definition from extraction.
func (rse *RegulatorySpecExtractor) parseFieldDefinition(extraction langextract.ExtractionResult) *FieldDefinition {
	attrs := extraction.Attributes
	if attrs == nil {
		return nil
	}

	fieldID, _ := attrs["field_id"].(string)
	fieldName, _ := attrs["field_name"].(string)
	fieldType, _ := attrs["field_type"].(string)
	required, _ := attrs["required"].(bool)

	if fieldID == "" || fieldName == "" {
		return nil
	}

	return &FieldDefinition{
		FieldID:       fieldID,
		FieldName:     fieldName,
		FieldType:     fieldType,
		Required:      required,
		Description:   extraction.ExtractionText,
		Source:        "regulatory_document",
	}
}

// parseValidationRule parses a validation rule from extraction.
func (rse *RegulatorySpecExtractor) parseValidationRule(extraction langextract.ExtractionResult) *ValidationRule {
	attrs := extraction.Attributes
	if attrs == nil {
		return nil
	}

	ruleID, _ := attrs["rule_id"].(string)
	ruleName, _ := attrs["rule_name"].(string)
	ruleType, _ := attrs["rule_type"].(string)

	if ruleID == "" {
		ruleID = fmt.Sprintf("rule-%d", time.Now().UnixNano())
	}

	return &ValidationRule{
		RuleID:        ruleID,
		RuleName:      ruleName,
		RuleType:      ruleType,
		Description:   extraction.ExtractionText,
		Severity:      "error",
	}
}

// calculateConfidence calculates confidence score for extraction.
func (rse *RegulatorySpecExtractor) calculateConfidence(spec *RegulatorySpec, response *langextract.ExtractionResponse) float64 {
	// Base confidence
	confidence := 0.7

	// Increase confidence based on completeness
	if len(spec.FieldDefinitions) > 0 {
		confidence += 0.1
	}
	if len(spec.ValidationRules) > 0 {
		confidence += 0.1
	}
	if spec.ReportStructure.TotalFields > 10 {
		confidence += 0.1
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// convertToExtractionResults converts to LangExtract format.
func convertToExtractionResults(response *langextract.ExtractionResponse) []langextract.ExtractionResult {
	return response.Extractions
}

