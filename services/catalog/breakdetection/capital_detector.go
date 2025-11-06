package breakdetection

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// CapitalDetector detects breaks in BCRS (Banking Credit Risk System)
type CapitalDetector struct {
	bcrsURL    string
	httpClient *http.Client
	logger     *log.Logger
}

// NewCapitalDetector creates a new capital break detector
func NewCapitalDetector(bcrsURL string, logger *log.Logger) *CapitalDetector {
	if bcrsURL == "" {
		bcrsURL = "http://localhost:8080" // Default BCRS URL
	}
	return &CapitalDetector{
		bcrsURL:    bcrsURL,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		logger:     logger,
	}
}

// BCRSCreditExposure represents a credit exposure from BCRS
type BCRSCreditExposure struct {
	ExposureID          string    `json:"exposure_id"`
	CounterpartyID     string    `json:"counterparty_id"`
	CounterpartyName   string    `json:"counterparty_name,omitempty"`
	ExposureAmount     float64   `json:"exposure_amount"`
	ExposureDate       time.Time `json:"exposure_date"`
	RiskWeight         float64   `json:"risk_weight"`
	EffectiveRiskWeight *float64 `json:"effective_risk_weight,omitempty"`
	RiskWeightedAmount float64   `json:"risk_weighted_amount"`
	Rating             string    `json:"rating,omitempty"`
	RatingAgency       string    `json:"rating_agency,omitempty"`
}

// BCRSCapitalRatio represents a capital ratio from BCRS
type BCRSCapitalRatio struct {
	RatioType      string    `json:"ratio_type"` // "CET1", "Tier1", "Total"
	RatioValue     float64   `json:"ratio_value"`
	MinimumRequired float64  `json:"minimum_required"`
	AsOfDate       time.Time `json:"as_of_date"`
	Status         string    `json:"status"` // "compliant", "below_minimum"
}

// DetectBreaks detects breaks in BCRS for capital calculations
func (cd *CapitalDetector) DetectBreaks(ctx context.Context, baseline *Baseline, config map[string]interface{}) ([]*Break, error) {
	if cd.logger != nil {
		cd.logger.Printf("Starting capital break detection for system: %s", baseline.SystemName)
	}

	// Parse baseline snapshot data
	var baselineData map[string]interface{}
	if err := json.Unmarshal(baseline.SnapshotData, &baselineData); err != nil {
		return nil, fmt.Errorf("failed to parse baseline data: %w", err)
	}

	// Extract baseline credit exposures and capital ratios
	baselineExposures, err := cd.extractBaselineCreditExposures(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline credit exposures: %w", err)
	}

	baselineRatios, err := cd.extractBaselineCapitalRatios(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline capital ratios: %w", err)
	}

	// Fetch current BCRS data
	currentExposures, err := cd.fetchCurrentCreditExposures(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current credit exposures: %w", err)
	}

	currentRatios, err := cd.fetchCurrentCapitalRatios(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current capital ratios: %w", err)
	}

	// Detect breaks
	var breaks []*Break

	// 1. Detect capital ratio violations
	ratioBreaks := cd.detectCapitalRatioViolations(baselineRatios, currentRatios)
	breaks = append(breaks, ratioBreaks...)

	// 2. Detect RWA calculation errors
	rwaBreaks := cd.detectRWAErrors(baselineExposures, currentExposures)
	breaks = append(breaks, rwaBreaks...)

	// 3. Detect exposure mismatches
	exposureBreaks := cd.detectExposureMismatches(baselineExposures, currentExposures)
	breaks = append(breaks, exposureBreaks...)

	if cd.logger != nil {
		cd.logger.Printf("Capital break detection completed: %d breaks detected", len(breaks))
	}

	return breaks, nil
}

// extractBaselineCreditExposures extracts credit exposures from baseline data
func (cd *CapitalDetector) extractBaselineCreditExposures(baselineData map[string]interface{}) (map[string]*BCRSCreditExposure, error) {
	exposures := make(map[string]*BCRSCreditExposure)

	if exposuresData, ok := baselineData["credit_exposures"].([]interface{}); ok {
		for _, expData := range exposuresData {
			if expMap, ok := expData.(map[string]interface{}); ok {
				exposure := cd.parseCreditExposure(expMap)
				if exposure != nil {
					exposures[exposure.ExposureID] = exposure
				}
			}
		}
	}

	return exposures, nil
}

// extractBaselineCapitalRatios extracts capital ratios from baseline data
func (cd *CapitalDetector) extractBaselineCapitalRatios(baselineData map[string]interface{}) (map[string]*BCRSCapitalRatio, error) {
	ratios := make(map[string]*BCRSCapitalRatio)

	if ratiosData, ok := baselineData["capital_ratios"].([]interface{}); ok {
		for _, ratioData := range ratiosData {
			if ratioMap, ok := ratioData.(map[string]interface{}); ok {
				ratio := cd.parseCapitalRatio(ratioMap)
				if ratio != nil {
					ratios[ratio.RatioType] = ratio
				}
			}
		}
	}

	return ratios, nil
}

// parseCreditExposure parses a credit exposure from a map
func (cd *CapitalDetector) parseCreditExposure(data map[string]interface{}) *BCRSCreditExposure {
	exposure := &BCRSCreditExposure{}

	if exposureID, ok := data["exposure_id"].(string); ok {
		exposure.ExposureID = exposureID
	} else {
		return nil // Exposure ID is required
	}

	if counterpartyID, ok := data["counterparty_id"].(string); ok {
		exposure.CounterpartyID = counterpartyID
	}

	if exposureAmount, ok := data["exposure_amount"].(float64); ok {
		exposure.ExposureAmount = exposureAmount
	}

	if riskWeight, ok := data["risk_weight"].(float64); ok {
		exposure.RiskWeight = riskWeight
	}

	if riskWeightedAmount, ok := data["risk_weighted_amount"].(float64); ok {
		exposure.RiskWeightedAmount = riskWeightedAmount
	}

	// Validate RWA calculation: risk_weighted_amount should equal exposure_amount * risk_weight
	expectedRWA := exposure.ExposureAmount * exposure.RiskWeight
	if exposure.RiskWeightedAmount > 0 && expectedRWA > 0 {
		diff := exposure.RiskWeightedAmount - expectedRWA
		if diff < 0 {
			diff = -diff
		}
		// Allow small rounding differences
		if diff > 0.01 {
			// RWA calculation error - but we'll detect this separately
		}
	}

	return exposure
}

// parseCapitalRatio parses a capital ratio from a map
func (cd *CapitalDetector) parseCapitalRatio(data map[string]interface{}) *BCRSCapitalRatio {
	ratio := &BCRSCapitalRatio{}

	if ratioType, ok := data["ratio_type"].(string); ok {
		ratio.RatioType = ratioType
	} else {
		return nil // Ratio type is required
	}

	if ratioValue, ok := data["ratio_value"].(float64); ok {
		ratio.RatioValue = ratioValue
	}

	if minimumRequired, ok := data["minimum_required"].(float64); ok {
		ratio.MinimumRequired = minimumRequired
	}

	if status, ok := data["status"].(string); ok {
		ratio.Status = status
	}

	return ratio
}

// fetchCurrentCreditExposures fetches current credit exposures from BCRS
func (cd *CapitalDetector) fetchCurrentCreditExposures(ctx context.Context) (map[string]*BCRSCreditExposure, error) {
	// TODO: Implement actual API call to BCRS
	exposures := make(map[string]*BCRSCreditExposure)

	// Placeholder: In production, this would call BCRS API
	// url := fmt.Sprintf("%s/api/credit-exposures", cd.bcrsURL)
	// resp, err := cd.httpClient.Get(url)
	// ... parse response ...

	return exposures, nil
}

// fetchCurrentCapitalRatios fetches current capital ratios from BCRS
func (cd *CapitalDetector) fetchCurrentCapitalRatios(ctx context.Context) (map[string]*BCRSCapitalRatio, error) {
	// TODO: Implement actual API call to BCRS
	ratios := make(map[string]*BCRSCapitalRatio)

	// Placeholder: In production, this would call BCRS API
	// url := fmt.Sprintf("%s/api/capital-ratios", cd.bcrsURL)
	// resp, err := cd.httpClient.Get(url)
	// ... parse response ...

	return ratios, nil
}

// detectCapitalRatioViolations detects capital ratio violations
func (cd *CapitalDetector) detectCapitalRatioViolations(baseline, current map[string]*BCRSCapitalRatio) []*Break {
	var breaks []*Break

	for ratioType, baselineRatio := range baseline {
		currentRatio, exists := current[ratioType]
		if !exists {
			// Missing ratio
			break := &Break{
				BreakID:        fmt.Sprintf("break-missing-ratio-%s", ratioType),
				SystemName:      SystemBCRS,
				DetectionType:   DetectionTypeCapital,
				BreakType:       BreakTypeCapitalRatioViolation,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    nil,
				BaselineValue:   cd.capitalRatioToMap(baselineRatio),
				Difference:      map[string]interface{}{"missing": true},
				AffectedEntities: []string{ratioType},
				DetectedAt:      time.Now(),
				CreatedAt:       time.Now(),
				UpdatedAt:       time.Now(),
			}
			breaks = append(breaks, break)
			continue
		}

		// Check if ratio is below minimum
		if currentRatio.RatioValue < currentRatio.MinimumRequired {
			break := &Break{
				BreakID:        fmt.Sprintf("break-ratio-violation-%s", ratioType),
				SystemName:      SystemBCRS,
				DetectionType:   DetectionTypeCapital,
				BreakType:       BreakTypeCapitalRatioViolation,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    cd.capitalRatioToMap(currentRatio),
				BaselineValue:   cd.capitalRatioToMap(baselineRatio),
				Difference: map[string]interface{}{
					"ratio_value":     currentRatio.RatioValue,
					"minimum_required": currentRatio.MinimumRequired,
					"deficit":         currentRatio.MinimumRequired - currentRatio.RatioValue,
				},
				AffectedEntities: []string{ratioType},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}

		// Check for significant ratio changes
		ratioChange := currentRatio.RatioValue - baselineRatio.RatioValue
		if ratioChange < 0 {
			ratioChange = -ratioChange
		}
		// Significant change threshold: 0.5% (0.005)
		if ratioChange > 0.005 {
			break := &Break{
				BreakID:        fmt.Sprintf("break-ratio-change-%s", ratioType),
				SystemName:      SystemBCRS,
				DetectionType:   DetectionTypeCapital,
				BreakType:       BreakTypeCapitalRatioViolation,
				Severity:        SeverityHigh,
				Status:          BreakStatusOpen,
				CurrentValue:    cd.capitalRatioToMap(currentRatio),
				BaselineValue:   cd.capitalRatioToMap(baselineRatio),
				Difference: map[string]interface{}{
					"baseline_ratio": baselineRatio.RatioValue,
					"current_ratio":  currentRatio.RatioValue,
					"change":         currentRatio.RatioValue - baselineRatio.RatioValue,
				},
				AffectedEntities: []string{ratioType},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}
	}

	return breaks
}

// detectRWAErrors detects Risk-Weighted Asset calculation errors
func (cd *CapitalDetector) detectRWAErrors(baseline, current map[string]*BCRSCreditExposure) []*Break {
	var breaks []*Break
	tolerance := 0.01

	for exposureID, baselineExposure := range baseline {
		currentExposure, exists := current[exposureID]
		if !exists {
			continue // Already handled by exposure mismatch detection
		}

		// Check RWA calculation: risk_weighted_amount should equal exposure_amount * risk_weight
		expectedRWA := currentExposure.ExposureAmount * currentExposure.RiskWeight
		actualRWA := currentExposure.RiskWeightedAmount

		diff := actualRWA - expectedRWA
		if diff < 0 {
			diff = -diff
		}

		if diff > tolerance {
			break := &Break{
				BreakID:        fmt.Sprintf("break-rwa-error-%s", exposureID),
				SystemName:      SystemBCRS,
				DetectionType:   DetectionTypeCapital,
				BreakType:       BreakTypeRWAError,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    cd.creditExposureToMap(currentExposure),
				BaselineValue:   cd.creditExposureToMap(baselineExposure),
				Difference: map[string]interface{}{
					"expected_rwa": expectedRWA,
					"actual_rwa":   actualRWA,
					"difference":   diff,
				},
				AffectedEntities: []string{exposureID, currentExposure.CounterpartyID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}
	}

	return breaks
}

// detectExposureMismatches detects credit exposure mismatches
func (cd *CapitalDetector) detectExposureMismatches(baseline, current map[string]*BCRSCreditExposure) []*Break {
	var breaks []*Break
	tolerance := 0.01

	for exposureID, baselineExposure := range baseline {
		currentExposure, exists := current[exposureID]
		if !exists {
			// Missing exposure
			break := &Break{
				BreakID:        fmt.Sprintf("break-missing-exposure-%s", exposureID),
				SystemName:      SystemBCRS,
				DetectionType:   DetectionTypeCapital,
				BreakType:       BreakTypeExposureMismatch,
				Severity:        SeverityHigh,
				Status:          BreakStatusOpen,
				CurrentValue:    nil,
				BaselineValue:   cd.creditExposureToMap(baselineExposure),
				Difference:      map[string]interface{}{"missing": true},
				AffectedEntities: []string{exposureID, baselineExposure.CounterpartyID},
				DetectedAt:      time.Now(),
				CreatedAt:       time.Now(),
				UpdatedAt:       time.Now(),
			}
			breaks = append(breaks, break)
			continue
		}

		// Check exposure amount mismatch
		amountDiff := currentExposure.ExposureAmount - baselineExposure.ExposureAmount
		if amountDiff < 0 {
			amountDiff = -amountDiff
		}
		if amountDiff > tolerance {
			break := &Break{
				BreakID:        fmt.Sprintf("break-exposure-amount-mismatch-%s", exposureID),
				SystemName:      SystemBCRS,
				DetectionType:   DetectionTypeCapital,
				BreakType:       BreakTypeExposureMismatch,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    cd.creditExposureToMap(currentExposure),
				BaselineValue:   cd.creditExposureToMap(baselineExposure),
				Difference: map[string]interface{}{
					"field":      "exposure_amount",
					"baseline":   baselineExposure.ExposureAmount,
					"current":   currentExposure.ExposureAmount,
					"difference": amountDiff,
				},
				AffectedEntities: []string{exposureID, currentExposure.CounterpartyID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}

		// Check risk weight mismatch
		weightDiff := currentExposure.RiskWeight - baselineExposure.RiskWeight
		if weightDiff < 0 {
			weightDiff = -weightDiff
		}
		if weightDiff > 0.001 { // 0.1% tolerance for risk weights
			break := &Break{
				BreakID:        fmt.Sprintf("break-risk-weight-mismatch-%s", exposureID),
				SystemName:      SystemBCRS,
				DetectionType:   DetectionTypeCapital,
				BreakType:       BreakTypeExposureMismatch,
				Severity:        SeverityHigh,
				Status:          BreakStatusOpen,
				CurrentValue:    cd.creditExposureToMap(currentExposure),
				BaselineValue:   cd.creditExposureToMap(baselineExposure),
				Difference: map[string]interface{}{
					"field":      "risk_weight",
					"baseline":   baselineExposure.RiskWeight,
					"current":   currentExposure.RiskWeight,
					"difference": weightDiff,
				},
				AffectedEntities: []string{exposureID, currentExposure.CounterpartyID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, break)
		}
	}

	return breaks
}

// Helper functions
func (cd *CapitalDetector) creditExposureToMap(exposure *BCRSCreditExposure) map[string]interface{} {
	return map[string]interface{}{
		"exposure_id":        exposure.ExposureID,
		"counterparty_id":    exposure.CounterpartyID,
		"exposure_amount":    exposure.ExposureAmount,
		"risk_weight":        exposure.RiskWeight,
		"risk_weighted_amount": exposure.RiskWeightedAmount,
	}
}

func (cd *CapitalDetector) capitalRatioToMap(ratio *BCRSCapitalRatio) map[string]interface{} {
	return map[string]interface{}{
		"ratio_type":      ratio.RatioType,
		"ratio_value":     ratio.RatioValue,
		"minimum_required": ratio.MinimumRequired,
		"status":          ratio.Status,
	}
}

