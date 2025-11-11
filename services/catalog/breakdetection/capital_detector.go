package breakdetection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
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
	baseURL := strings.TrimSuffix(cd.bcrsURL, "/")
	endpoint := fmt.Sprintf("%s/api/credit-exposures", baseURL)

	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if apiKey := os.Getenv("BCRS_API_KEY"); apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}
	req.Header.Set("Accept", "application/json")

	if cd.logger != nil {
		cd.logger.Printf("Fetching credit exposures from BCRS: %s", endpoint)
	}

	resp, err := cd.makeHTTPRequestWithRetry(req, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch credit exposures: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("BCRS API returned error status %d: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Exposures []*BCRSCreditExposure `json:"exposures,omitempty"`
		Data      []*BCRSCreditExposure `json:"data,omitempty"`
	}

	var exposuresArray []*BCRSCreditExposure
	if err := json.Unmarshal(body, &exposuresArray); err == nil {
		exposures := make(map[string]*BCRSCreditExposure, len(exposuresArray))
		for _, exposure := range exposuresArray {
			exposures[exposure.ExposureID] = exposure
		}
		return exposures, nil
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	exposures := make(map[string]*BCRSCreditExposure)
	if len(response.Exposures) > 0 {
		for _, exposure := range response.Exposures {
			exposures[exposure.ExposureID] = exposure
		}
	} else if len(response.Data) > 0 {
		for _, exposure := range response.Data {
			exposures[exposure.ExposureID] = exposure
		}
	}

	if cd.logger != nil {
		cd.logger.Printf("Fetched %d credit exposures from BCRS", len(exposures))
	}

	return exposures, nil
}

// fetchCurrentCapitalRatios fetches current capital ratios from BCRS
func (cd *CapitalDetector) fetchCurrentCapitalRatios(ctx context.Context) (map[string]*BCRSCapitalRatio, error) {
	baseURL := strings.TrimSuffix(cd.bcrsURL, "/")
	endpoint := fmt.Sprintf("%s/api/capital-ratios", baseURL)

	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if apiKey := os.Getenv("BCRS_API_KEY"); apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}
	req.Header.Set("Accept", "application/json")

	if cd.logger != nil {
		cd.logger.Printf("Fetching capital ratios from BCRS: %s", endpoint)
	}

	resp, err := cd.makeHTTPRequestWithRetry(req, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch capital ratios: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("BCRS API returned error status %d: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Ratios []*BCRSCapitalRatio `json:"ratios,omitempty"`
		Data   []*BCRSCapitalRatio `json:"data,omitempty"`
	}

	var ratiosArray []*BCRSCapitalRatio
	if err := json.Unmarshal(body, &ratiosArray); err == nil {
		ratios := make(map[string]*BCRSCapitalRatio, len(ratiosArray))
		for _, ratio := range ratiosArray {
			ratios[ratio.RatioType] = ratio
		}
		return ratios, nil
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	ratios := make(map[string]*BCRSCapitalRatio)
	if len(response.Ratios) > 0 {
		for _, ratio := range response.Ratios {
			ratios[ratio.RatioType] = ratio
		}
	} else if len(response.Data) > 0 {
		for _, ratio := range response.Data {
			ratios[ratio.RatioType] = ratio
		}
	}

	if cd.logger != nil {
		cd.logger.Printf("Fetched %d capital ratios from BCRS", len(ratios))
	}

	return ratios, nil
}

// makeHTTPRequestWithRetry makes an HTTP request with retry logic and exponential backoff
func (cd *CapitalDetector) makeHTTPRequestWithRetry(req *http.Request, maxRetries int) (*http.Response, error) {
	var resp *http.Response
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			waitTime := time.Duration(1<<uint(attempt-1)) * time.Second
			if cd.logger != nil {
				cd.logger.Printf("Retrying HTTP request (attempt %d/%d) after %v", attempt+1, maxRetries, waitTime)
			}
			select {
			case <-time.After(waitTime):
			case <-req.Context().Done():
				return nil, fmt.Errorf("request cancelled: %w", req.Context().Err())
			}
		}

		resp, err = cd.httpClient.Do(req)
		if err == nil {
			return resp, nil
		}

		if req.Context().Err() != nil {
			return nil, fmt.Errorf("request cancelled or timed out: %w", req.Context().Err())
		}

		if cd.logger != nil && attempt < maxRetries-1 {
			cd.logger.Printf("HTTP request failed (attempt %d/%d): %v", attempt+1, maxRetries, err)
		}
	}

	return nil, fmt.Errorf("failed after %d attempts: %w", maxRetries, err)
}

// detectCapitalRatioViolations detects capital ratio violations
func (cd *CapitalDetector) detectCapitalRatioViolations(baseline, current map[string]*BCRSCapitalRatio) []*Break {
	var breaks []*Break

	for ratioType, baselineRatio := range baseline {
		currentRatio, exists := current[ratioType]
		if !exists {
			// Missing ratio
			br := &Break{
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
			breaks = append(breaks, br)
			continue
		}

		// Check if ratio is below minimum
		if currentRatio.RatioValue < currentRatio.MinimumRequired {
			br := &Break{
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
			breaks = append(breaks, br)
		}

		// Check for significant ratio changes
		ratioChange := currentRatio.RatioValue - baselineRatio.RatioValue
		if ratioChange < 0 {
			ratioChange = -ratioChange
		}
		// Significant change threshold: 0.5% (0.005)
		if ratioChange > 0.005 {
			br := &Break{
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
			breaks = append(breaks, br)
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
			br := &Break{
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
			breaks = append(breaks, br)
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
			br := &Break{
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
			breaks = append(breaks, br)
			continue
		}

		// Check exposure amount mismatch
		amountDiff := currentExposure.ExposureAmount - baselineExposure.ExposureAmount
		if amountDiff < 0 {
			amountDiff = -amountDiff
		}
		if amountDiff > tolerance {
			br := &Break{
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
			breaks = append(breaks, br)
		}

		// Check risk weight mismatch
		weightDiff := currentExposure.RiskWeight - baselineExposure.RiskWeight
		if weightDiff < 0 {
			weightDiff = -weightDiff
		}
		if weightDiff > 0.001 { // 0.1% tolerance for risk weights
			br := &Break{
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
			breaks = append(breaks, br)
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

