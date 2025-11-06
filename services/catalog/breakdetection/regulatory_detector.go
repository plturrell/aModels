package breakdetection

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// RegulatoryDetector detects breaks in AxiomSL for regulatory reporting
type RegulatoryDetector struct {
	axiomSLURL  string
	httpClient  *http.Client
	logger      *log.Logger
}

// NewRegulatoryDetector creates a new regulatory break detector
func NewRegulatoryDetector(axiomSLURL string, logger *log.Logger) *RegulatoryDetector {
	if axiomSLURL == "" {
		axiomSLURL = "http://localhost:8080" // Default AxiomSL URL
	}
	return &RegulatoryDetector{
		axiomSLURL:  axiomSLURL,
		httpClient:  &http.Client{Timeout: 60 * time.Second},
		logger:      logger,
	}
}

// AxiomSLRegulatoryReport represents a regulatory report from AxiomSL
type AxiomSLRegulatoryReport struct {
	ReportID          string    `json:"report_id"`
	ReportType        string    `json:"report_type"` // "FR Y-14", "CCAR", "Basel III", etc.
	ReportingDate     time.Time `json:"reporting_date"`
	SubmissionDate    *time.Time `json:"submission_date,omitempty"`
	Status            string    `json:"status"` // "draft", "submitted", "approved"
	RegulatoryFramework string  `json:"regulatory_framework"`
	TotalAmount       float64   `json:"total_amount"`
	ValidationStatus  string    `json:"validation_status"` // "passed", "failed", "warnings"
}

// AxiomSLRegulatoryCalculation represents a regulatory calculation from AxiomSL
type AxiomSLRegulatoryCalculation struct {
	CalculationID     string    `json:"calculation_id"`
	ReportID          string    `json:"report_id"`
	CalculationType   string    `json:"calculation_type"`
	InputValue         float64   `json:"input_value"`
	OutputValue        float64   `json:"output_value"`
	ExpectedValue     *float64  `json:"expected_value,omitempty"`
	Formula           string    `json:"formula,omitempty"`
	ValidationPassed  bool      `json:"validation_passed"`
}

// DetectBreaks detects breaks in AxiomSL for regulatory reporting
func (rd *RegulatoryDetector) DetectBreaks(ctx context.Context, baseline *Baseline, config map[string]interface{}) ([]*Break, error) {
	if rd.logger != nil {
		rd.logger.Printf("Starting regulatory break detection for system: %s", baseline.SystemName)
	}

	// Parse baseline snapshot data
	var baselineData map[string]interface{}
	if err := json.Unmarshal(baseline.SnapshotData, &baselineData); err != nil {
		return nil, fmt.Errorf("failed to parse baseline data: %w", err)
	}

	// Extract baseline regulatory reports and calculations
	baselineReports, err := rd.extractBaselineRegulatoryReports(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline regulatory reports: %w", err)
	}

	baselineCalculations, err := rd.extractBaselineRegulatoryCalculations(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline regulatory calculations: %w", err)
	}

	// Fetch current AxiomSL data
	currentReports, err := rd.fetchCurrentRegulatoryReports(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current regulatory reports: %w", err)
	}

	currentCalculations, err := rd.fetchCurrentRegulatoryCalculations(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current regulatory calculations: %w", err)
	}

	// Detect breaks
	var breaks []*Break

	// 1. Detect compliance violations
	complianceBreaks := rd.detectComplianceViolations(baselineReports, currentReports)
	breaks = append(breaks, complianceBreaks...)

	// 2. Detect reporting breaks
	reportingBreaks := rd.detectReportingBreaks(baselineReports, currentReports)
	breaks = append(breaks, reportingBreaks...)

	// 3. Detect calculation errors
	calculationBreaks := rd.detectCalculationErrors(baselineCalculations, currentCalculations)
	breaks = append(breaks, calculationBreaks...)

	if rd.logger != nil {
		rd.logger.Printf("Regulatory break detection completed: %d breaks detected", len(breaks))
	}

	return breaks, nil
}

// extractBaselineRegulatoryReports extracts regulatory reports from baseline data
func (rd *RegulatoryDetector) extractBaselineRegulatoryReports(baselineData map[string]interface{}) (map[string]*AxiomSLRegulatoryReport, error) {
	reports := make(map[string]*AxiomSLRegulatoryReport)

	if reportsData, ok := baselineData["regulatory_reports"].([]interface{}); ok {
		for _, reportData := range reportsData {
			if reportMap, ok := reportData.(map[string]interface{}); ok {
				report := rd.parseRegulatoryReport(reportMap)
				if report != nil {
					reports[report.ReportID] = report
				}
			}
		}
	}

	return reports, nil
}

// extractBaselineRegulatoryCalculations extracts regulatory calculations from baseline data
func (rd *RegulatoryDetector) extractBaselineRegulatoryCalculations(baselineData map[string]interface{}) (map[string]*AxiomSLRegulatoryCalculation, error) {
	calculations := make(map[string]*AxiomSLRegulatoryCalculation)

	if calculationsData, ok := baselineData["regulatory_calculations"].([]interface{}); ok {
		for _, calcData := range calculationsData {
			if calcMap, ok := calcData.(map[string]interface{}); ok {
				calculation := rd.parseRegulatoryCalculation(calcMap)
				if calculation != nil {
					calculations[calculation.CalculationID] = calculation
				}
			}
		}
	}

	return calculations, nil
}

// parseRegulatoryReport parses a regulatory report from a map
func (rd *RegulatoryDetector) parseRegulatoryReport(data map[string]interface{}) *AxiomSLRegulatoryReport {
	report := &AxiomSLRegulatoryReport{}

	if reportID, ok := data["report_id"].(string); ok {
		report.ReportID = reportID
	} else {
		return nil // Report ID is required
	}

	if reportType, ok := data["report_type"].(string); ok {
		report.ReportType = reportType
	}

	if totalAmount, ok := data["total_amount"].(float64); ok {
		report.TotalAmount = totalAmount
	}

	if status, ok := data["status"].(string); ok {
		report.Status = status
	}

	if validationStatus, ok := data["validation_status"].(string); ok {
		report.ValidationStatus = validationStatus
	}

	if framework, ok := data["regulatory_framework"].(string); ok {
		report.RegulatoryFramework = framework
	}

	return report
}

// parseRegulatoryCalculation parses a regulatory calculation from a map
func (rd *RegulatoryDetector) parseRegulatoryCalculation(data map[string]interface{}) *AxiomSLRegulatoryCalculation {
	calculation := &AxiomSLRegulatoryCalculation{}

	if calculationID, ok := data["calculation_id"].(string); ok {
		calculation.CalculationID = calculationID
	} else {
		return nil // Calculation ID is required
	}

	if reportID, ok := data["report_id"].(string); ok {
		calculation.ReportID = reportID
	}

	if inputValue, ok := data["input_value"].(float64); ok {
		calculation.InputValue = inputValue
	}

	if outputValue, ok := data["output_value"].(float64); ok {
		calculation.OutputValue = outputValue
	}

	if expectedValue, ok := data["expected_value"].(float64); ok {
		calculation.ExpectedValue = &expectedValue
	}

	if validationPassed, ok := data["validation_passed"].(bool); ok {
		calculation.ValidationPassed = validationPassed
	}

	if formula, ok := data["formula"].(string); ok {
		calculation.Formula = formula
	}

	return calculation
}

// fetchCurrentRegulatoryReports fetches current regulatory reports from AxiomSL
func (rd *RegulatoryDetector) fetchCurrentRegulatoryReports(ctx context.Context) (map[string]*AxiomSLRegulatoryReport, error) {
	// TODO: Implement actual API call to AxiomSL
	reports := make(map[string]*AxiomSLRegulatoryReport)

	// Placeholder: In production, this would call AxiomSL API
	// url := fmt.Sprintf("%s/api/regulatory-reports", rd.axiomSLURL)
	// resp, err := rd.httpClient.Get(url)
	// ... parse response ...

	return reports, nil
}

// fetchCurrentRegulatoryCalculations fetches current regulatory calculations from AxiomSL
func (rd *RegulatoryDetector) fetchCurrentRegulatoryCalculations(ctx context.Context) (map[string]*AxiomSLRegulatoryCalculation, error) {
	// TODO: Implement actual API call to AxiomSL
	calculations := make(map[string]*AxiomSLRegulatoryCalculation)

	// Placeholder: In production, this would call AxiomSL API
	// url := fmt.Sprintf("%s/api/regulatory-calculations", rd.axiomSLURL)
	// resp, err := rd.httpClient.Get(url)
	// ... parse response ...

	return calculations, nil
}

// detectComplianceViolations detects compliance violations
func (rd *RegulatoryDetector) detectComplianceViolations(baseline, current map[string]*AxiomSLRegulatoryReport) []*Break {
	var breaks []*Break

	for reportID, currentReport := range current {
		// Check validation status
		if currentReport.ValidationStatus == "failed" {
			break := &Break{
				BreakID:        fmt.Sprintf("break-compliance-violation-%s", reportID),
				SystemName:      SystemAxiomSL,
				DetectionType:   DetectionTypeRegulatory,
				BreakType:       BreakTypeComplianceViolation,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    rd.regulatoryReportToMap(currentReport),
				BaselineValue:   rd.regulatoryReportToMap(baseline[reportID]),
				Difference: map[string]interface{}{
					"validation_status": currentReport.ValidationStatus,
					"report_type":       currentReport.ReportType,
				},
				AffectedEntities: []string{reportID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}
	}

	return breaks
}

// detectReportingBreaks detects reporting breaks
func (rd *RegulatoryDetector) detectReportingBreaks(baseline, current map[string]*AxiomSLRegulatoryReport) []*Break {
	var breaks []*Break
	tolerance := 0.01

	for reportID, baselineReport := range baseline {
		currentReport, exists := current[reportID]
		if !exists {
			// Missing report
			break := &Break{
				BreakID:        fmt.Sprintf("break-missing-report-%s", reportID),
				SystemName:      SystemAxiomSL,
				DetectionType:   DetectionTypeRegulatory,
				BreakType:       BreakTypeReportingBreak,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    nil,
				BaselineValue:   rd.regulatoryReportToMap(baselineReport),
				Difference:      map[string]interface{}{"missing": true},
				AffectedEntities: []string{reportID},
				DetectedAt:      time.Now(),
				CreatedAt:       time.Now(),
				UpdatedAt:       time.Now(),
			}
			breaks = append(breaks, break)
			continue
		}

		// Check total amount mismatch
		amountDiff := currentReport.TotalAmount - baselineReport.TotalAmount
		if amountDiff < 0 {
			amountDiff = -amountDiff
		}
		if amountDiff > tolerance {
			break := &Break{
				BreakID:        fmt.Sprintf("break-report-amount-mismatch-%s", reportID),
				SystemName:      SystemAxiomSL,
				DetectionType:   DetectionTypeRegulatory,
				BreakType:       BreakTypeReportingBreak,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    rd.regulatoryReportToMap(currentReport),
				BaselineValue:   rd.regulatoryReportToMap(baselineReport),
				Difference: map[string]interface{}{
					"field":      "total_amount",
					"baseline":   baselineReport.TotalAmount,
					"current":   currentReport.TotalAmount,
					"difference": amountDiff,
				},
				AffectedEntities: []string{reportID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}
	}

	return breaks
}

// detectCalculationErrors detects regulatory calculation errors
func (rd *RegulatoryDetector) detectCalculationErrors(baseline, current map[string]*AxiomSLRegulatoryCalculation) []*Break {
	var breaks []*Break
	tolerance := 0.01

	for calculationID, currentCalculation := range current {
		// Check if validation failed
		if !currentCalculation.ValidationPassed {
			break := &Break{
				BreakID:        fmt.Sprintf("break-calculation-validation-failed-%s", calculationID),
				SystemName:      SystemAxiomSL,
				DetectionType:   DetectionTypeRegulatory,
				BreakType:       BreakTypeCalculationError,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    rd.regulatoryCalculationToMap(currentCalculation),
				BaselineValue:   rd.regulatoryCalculationToMap(baseline[calculationID]),
				Difference: map[string]interface{}{
					"validation_passed": currentCalculation.ValidationPassed,
				},
				AffectedEntities: []string{calculationID, currentCalculation.ReportID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}

		// Check if expected value differs from output
		if currentCalculation.ExpectedValue != nil {
			diff := currentCalculation.OutputValue - *currentCalculation.ExpectedValue
			if diff < 0 {
				diff = -diff
			}
			if diff > tolerance {
				break := &Break{
					BreakID:        fmt.Sprintf("break-calculation-mismatch-%s", calculationID),
					SystemName:      SystemAxiomSL,
					DetectionType:   DetectionTypeRegulatory,
					BreakType:       BreakTypeCalculationError,
					Severity:        SeverityHigh,
					Status:          BreakStatusOpen,
					CurrentValue:    rd.regulatoryCalculationToMap(currentCalculation),
					BaselineValue:   rd.regulatoryCalculationToMap(baseline[calculationID]),
					Difference: map[string]interface{}{
						"expected_value": *currentCalculation.ExpectedValue,
						"output_value":   currentCalculation.OutputValue,
						"difference":     diff,
					},
					AffectedEntities: []string{calculationID, currentCalculation.ReportID},
					DetectedAt:       time.Now(),
					CreatedAt:        time.Now(),
					UpdatedAt:        time.Now(),
				}
				breaks = append(breaks, break)
			}
		}
	}

	return breaks
}

// Helper functions
func (rd *RegulatoryDetector) regulatoryReportToMap(report *AxiomSLRegulatoryReport) map[string]interface{} {
	if report == nil {
		return nil
	}
	return map[string]interface{}{
		"report_id":           report.ReportID,
		"report_type":         report.ReportType,
		"total_amount":        report.TotalAmount,
		"status":              report.Status,
		"validation_status":   report.ValidationStatus,
		"regulatory_framework": report.RegulatoryFramework,
	}
}

func (rd *RegulatoryDetector) regulatoryCalculationToMap(calculation *AxiomSLRegulatoryCalculation) map[string]interface{} {
	if calculation == nil {
		return nil
	}
	m := map[string]interface{}{
		"calculation_id":  calculation.CalculationID,
		"report_id":       calculation.ReportID,
		"input_value":     calculation.InputValue,
		"output_value":    calculation.OutputValue,
		"validation_passed": calculation.ValidationPassed,
	}
	if calculation.ExpectedValue != nil {
		m["expected_value"] = *calculation.ExpectedValue
	}
	if calculation.Formula != "" {
		m["formula"] = calculation.Formula
	}
	return m
}

