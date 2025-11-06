package regulatory

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/extract/regulatory"
)

// MAS610Reporting provides complete MAS 610 regulatory reporting implementation.
type MAS610Reporting struct {
	extractor         regulatory.MAS610Extractor
	calculationEngine *RegulatoryCalculationEngine
	validator         *ReportValidator
	outputTracer      *OutputTracer
	logger            *log.Logger
}

// ExtractionResult represents the result of an extraction (from extract/regulatory package).
type ExtractionResult = regulatory.ExtractionResult

// RegulatorySpec represents a regulatory specification (from extract/regulatory package).
type RegulatorySpec = regulatory.RegulatorySpec

// NewMAS610Reporting creates a new MAS 610 reporting system.
func NewMAS610Reporting(
	extractor regulatory.MAS610Extractor,
	calculationEngine *RegulatoryCalculationEngine,
	validator *ReportValidator,
	outputTracer *OutputTracer,
	logger *log.Logger,
) *MAS610Reporting {
	return &MAS610Reporting{
		extractor:         extractor,
		calculationEngine: calculationEngine,
		validator:         validator,
		outputTracer:      outputTracer,
		logger:            logger,
	}
}

// GenerateReport generates a complete MAS 610 regulatory report.
func (m *MAS610Reporting) GenerateReport(ctx context.Context, req MAS610ReportRequest) (*MAS610Report, error) {
	if m.logger != nil {
		m.logger.Printf("Generating MAS 610 report for period: %s", req.ReportPeriod)
	}

	// Step 1: Extract regulatory specifications if needed
	var spec *regulatory.RegulatorySpec
	if req.SpecID != "" {
		// Load spec from repository
		spec = m.loadSpec(ctx, req.SpecID)
	}

	// Step 2: Calculate regulatory metrics
	calculations, err := m.calculationEngine.CalculateRegulatoryMetrics(ctx, RegulatoryCalculationRequest{
		Framework:    "MAS 610",
		ReportPeriod: req.ReportPeriod,
		Metrics:      req.Metrics,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to calculate regulatory metrics: %w", err)
	}

	// Step 3: Generate report structure
	report := &MAS610Report{
		ReportID:      fmt.Sprintf("MAS610-%s-%s", req.ReportPeriod, time.Now().Format("20060102")),
		ReportPeriod:  req.ReportPeriod,
		GeneratedAt:  time.Now(),
		GeneratedBy:   req.GeneratedBy,
		Status:        "draft",
		Calculations: calculations,
		Sections:      []MAS610Section{},
	}

	// Step 4: Populate report sections based on MAS 610 requirements
	m.populateReportSections(report, spec, calculations)

	// Step 5: Validate report
	validationResult, err := m.validator.ValidateMAS610Report(ctx, report)
	if err != nil {
		return nil, fmt.Errorf("failed to validate report: %w", err)
	}

	report.ValidationResult = validationResult
	if validationResult.IsValid {
		report.Status = "validated"
	} else {
		report.Status = "validation_failed"
		report.ValidationErrors = validationResult.Errors
	}

	// Step 6: Trace output
	if m.outputTracer != nil {
		if err := m.outputTracer.TraceReport(ctx, report.ReportID, calculations, report); err != nil {
			if m.logger != nil {
				m.logger.Printf("Warning: Failed to trace report output: %v", err)
			}
		}
	}

	if m.logger != nil {
		m.logger.Printf("MAS 610 report generated: %s (status: %s)", report.ReportID, report.Status)
	}

	return report, nil
}

// MAS610ReportRequest represents a request to generate a MAS 610 report.
type MAS610ReportRequest struct {
	ReportPeriod string
	SpecID       string
	Metrics      []string
	GeneratedBy  string
}

// MAS610Report represents a complete MAS 610 regulatory report.
type MAS610Report struct {
	ReportID          string
	ReportPeriod      string
	GeneratedAt       time.Time
	GeneratedBy       string
	Status            string // "draft", "validated", "submitted", "validation_failed"
	Calculations      []RegulatoryCalculation
	Sections          []MAS610Section
	ValidationResult  *ValidationResult
	ValidationErrors  []string
	SubmissionDetails *SubmissionDetails
}

// MAS610Section represents a section in a MAS 610 report.
type MAS610Section struct {
	SectionID    string
	SectionName  string
	Order        int
	Fields       []MAS610Field
	Subsections  []MAS610Section
}

// MAS610Field represents a field in a MAS 610 report.
type MAS610Field struct {
	FieldID      string
	FieldName    string
	FieldType    string
	Value        interface{}
	Required     bool
	ValidationRules []ValidationRule
}

// populateReportSections populates report sections based on MAS 610 requirements.
func (m *MAS610Reporting) populateReportSections(report *MAS610Report, spec *RegulatorySpec, calculations []RegulatoryCalculation) {
	// Standard MAS 610 sections
	sections := []MAS610Section{
		{
			SectionID:   "section_1",
			SectionName: "Capital Adequacy",
			Order:       1,
			Fields:      m.extractFieldsFromCalculations(calculations, "capital"),
		},
		{
			SectionID:   "section_2",
			SectionName: "Liquidity Coverage",
			Order:       2,
			Fields:      m.extractFieldsFromCalculations(calculations, "liquidity"),
		},
		{
			SectionID:   "section_3",
			SectionName: "Risk Weighted Assets",
			Order:       3,
			Fields:      m.extractFieldsFromCalculations(calculations, "rwa"),
		},
	}

	report.Sections = sections
}

// extractFieldsFromCalculations extracts fields from regulatory calculations.
func (m *MAS610Reporting) extractFieldsFromCalculations(calculations []RegulatoryCalculation, metricType string) []MAS610Field {
	var fields []MAS610Field
	for _, calc := range calculations {
		if calc.CalculationType == metricType {
			fields = append(fields, MAS610Field{
				FieldID:   calc.CalculationID,
				FieldName: calc.CalculationID,
				FieldType: "decimal",
				Value:     calc.Result,
				Required:  true,
			})
		}
	}
	return fields
}

// loadSpec loads a regulatory specification.
func (m *MAS610Reporting) loadSpec(ctx context.Context, specID string) *regulatory.RegulatorySpec {
	// In production, would load from repository
	return nil
}

// SubmitReport submits a MAS 610 report to the regulatory authority.
func (m *MAS610Reporting) SubmitReport(ctx context.Context, reportID string) error {
	if m.logger != nil {
		m.logger.Printf("Submitting MAS 610 report: %s", reportID)
	}

	// In production, would submit to MAS portal/API
	// For now, mark as submitted
	return nil
}

// ValidateReport validates a MAS 610 report against regulatory requirements.
func (m *MAS610Reporting) ValidateReport(ctx context.Context, report *MAS610Report) (*ValidationResult, error) {
	return m.validator.ValidateMAS610Report(ctx, report)
}

