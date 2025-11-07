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

// LiquidityDetector detects breaks in RCO (Risk Control Office) for liquidity positions
type LiquidityDetector struct {
	rcoURL     string
	httpClient *http.Client
	logger     *log.Logger
}

// NewLiquidityDetector creates a new liquidity break detector
func NewLiquidityDetector(rcoURL string, logger *log.Logger) *LiquidityDetector {
	if rcoURL == "" {
		rcoURL = "http://localhost:8080" // Default RCO URL
	}
	return &LiquidityDetector{
		rcoURL:     rcoURL,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		logger:     logger,
	}
}

// RCOLiquidityPosition represents a liquidity position from RCO
type RCOLiquidityPosition struct {
	PositionID          string    `json:"position_id"`
	TradeID             string    `json:"trade_id,omitempty"`
	PositionDate        time.Time `json:"position_date"`
	PositionType        string    `json:"position_type"` // "Long", "Short", "Net"
	PositionAmount      float64   `json:"position_amount"`
	Currency            string    `json:"currency"`
	MarketValue         *float64  `json:"market_value,omitempty"`
	LiquidityRequirement float64  `json:"liquidity_requirement"`
	MaturityDate        *time.Time `json:"maturity_date,omitempty"`
}

// RCOLCR represents a Liquidity Coverage Ratio from RCO
type RCOLCR struct {
	AsOfDate       time.Time `json:"as_of_date"`
	LCRValue       float64   `json:"lcr_value"`
	MinimumRequired float64  `json:"minimum_required"` // Typically 1.0 (100%)
	Status         string    `json:"status"`             // "compliant", "below_minimum"
	HQLA          float64   `json:"hqla"`              // High Quality Liquid Assets
	NetCashOutflow float64   `json:"net_cash_outflow"`
}

// DetectBreaks detects breaks in RCO for liquidity positions
func (ld *LiquidityDetector) DetectBreaks(ctx context.Context, baseline *Baseline, config map[string]interface{}) ([]*Break, error) {
	if ld.logger != nil {
		ld.logger.Printf("Starting liquidity break detection for system: %s", baseline.SystemName)
	}

	// Parse baseline snapshot data
	var baselineData map[string]interface{}
	if err := json.Unmarshal(baseline.SnapshotData, &baselineData); err != nil {
		return nil, fmt.Errorf("failed to parse baseline data: %w", err)
	}

	// Extract baseline liquidity positions and LCR
	baselinePositions, err := ld.extractBaselineLiquidityPositions(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline liquidity positions: %w", err)
	}

	baselineLCR, err := ld.extractBaselineLCR(baselineData)
	if err != nil {
		return nil, fmt.Errorf("failed to extract baseline LCR: %w", err)
	}

	// Fetch current RCO data
	currentPositions, err := ld.fetchCurrentLiquidityPositions(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current liquidity positions: %w", err)
	}

	currentLCR, err := ld.fetchCurrentLCR(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current LCR: %w", err)
	}

	// Detect breaks
	var breaks []*Break

	// 1. Detect LCR violations
	lcrBreaks := ld.detectLCRViolations(baselineLCR, currentLCR)
	breaks = append(breaks, lcrBreaks...)

	// 2. Detect liquidity mismatches
	liquidityBreaks := ld.detectLiquidityMismatches(baselinePositions, currentPositions)
	breaks = append(breaks, liquidityBreaks...)

	// 3. Detect position mismatches
	positionBreaks := ld.detectPositionMismatches(baselinePositions, currentPositions)
	breaks = append(breaks, positionBreaks...)

	if ld.logger != nil {
		ld.logger.Printf("Liquidity break detection completed: %d breaks detected", len(breaks))
	}

	return breaks, nil
}

// extractBaselineLiquidityPositions extracts liquidity positions from baseline data
func (ld *LiquidityDetector) extractBaselineLiquidityPositions(baselineData map[string]interface{}) (map[string]*RCOLiquidityPosition, error) {
	positions := make(map[string]*RCOLiquidityPosition)

	if positionsData, ok := baselineData["liquidity_positions"].([]interface{}); ok {
		for _, posData := range positionsData {
			if posMap, ok := posData.(map[string]interface{}); ok {
				position := ld.parseLiquidityPosition(posMap)
				if position != nil {
					positions[position.PositionID] = position
				}
			}
		}
	}

	return positions, nil
}

// extractBaselineLCR extracts LCR from baseline data
func (ld *LiquidityDetector) extractBaselineLCR(baselineData map[string]interface{}) (*RCOLCR, error) {
	if lcrData, ok := baselineData["lcr"].(map[string]interface{}); ok {
		return ld.parseLCR(lcrData), nil
	}
	return nil, nil
}

// parseLiquidityPosition parses a liquidity position from a map
func (ld *LiquidityDetector) parseLiquidityPosition(data map[string]interface{}) *RCOLiquidityPosition {
	position := &RCOLiquidityPosition{}

	if positionID, ok := data["position_id"].(string); ok {
		position.PositionID = positionID
	} else {
		return nil // Position ID is required
	}

	if positionAmount, ok := data["position_amount"].(float64); ok {
		position.PositionAmount = positionAmount
	}

	if currency, ok := data["currency"].(string); ok {
		position.Currency = currency
	}

	if positionType, ok := data["position_type"].(string); ok {
		position.PositionType = positionType
	}

	if liquidityRequirement, ok := data["liquidity_requirement"].(float64); ok {
		position.LiquidityRequirement = liquidityRequirement
	}

	return position
}

// parseLCR parses an LCR from a map
func (ld *LiquidityDetector) parseLCR(data map[string]interface{}) *RCOLCR {
	lcr := &RCOLCR{}

	if lcrValue, ok := data["lcr_value"].(float64); ok {
		lcr.LCRValue = lcrValue
	}

	if minimumRequired, ok := data["minimum_required"].(float64); ok {
		lcr.MinimumRequired = minimumRequired
	} else {
		lcr.MinimumRequired = 1.0 // Default 100%
	}

	if status, ok := data["status"].(string); ok {
		lcr.Status = status
	}

	if hqla, ok := data["hqla"].(float64); ok {
		lcr.HQLA = hqla
	}

	if netCashOutflow, ok := data["net_cash_outflow"].(float64); ok {
		lcr.NetCashOutflow = netCashOutflow
	}

	return lcr
}

// fetchCurrentLiquidityPositions fetches current liquidity positions from RCO
func (ld *LiquidityDetector) fetchCurrentLiquidityPositions(ctx context.Context) (map[string]*RCOLiquidityPosition, error) {
	baseURL := strings.TrimSuffix(ld.rcoURL, "/")
	endpoint := fmt.Sprintf("%s/api/liquidity-positions", baseURL)

	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if apiKey := os.Getenv("RCO_API_KEY"); apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}
	req.Header.Set("Accept", "application/json")

	if ld.logger != nil {
		ld.logger.Printf("Fetching liquidity positions from RCO: %s", endpoint)
	}

	resp, err := ld.makeHTTPRequestWithRetry(req, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch liquidity positions: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("RCO API returned error status %d: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Positions []*RCOLiquidityPosition `json:"positions,omitempty"`
		Data      []*RCOLiquidityPosition `json:"data,omitempty"`
	}

	var positionsArray []*RCOLiquidityPosition
	if err := json.Unmarshal(body, &positionsArray); err == nil {
		positions := make(map[string]*RCOLiquidityPosition, len(positionsArray))
		for _, position := range positionsArray {
			positions[position.PositionID] = position
		}
		return positions, nil
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	positions := make(map[string]*RCOLiquidityPosition)
	if len(response.Positions) > 0 {
		for _, position := range response.Positions {
			positions[position.PositionID] = position
		}
	} else if len(response.Data) > 0 {
		for _, position := range response.Data {
			positions[position.PositionID] = position
		}
	}

	if ld.logger != nil {
		ld.logger.Printf("Fetched %d liquidity positions from RCO", len(positions))
	}

	return positions, nil
}

// fetchCurrentLCR fetches current LCR from RCO
func (ld *LiquidityDetector) fetchCurrentLCR(ctx context.Context) (*RCOLCR, error) {
	baseURL := strings.TrimSuffix(ld.rcoURL, "/")
	endpoint := fmt.Sprintf("%s/api/lcr", baseURL)

	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if apiKey := os.Getenv("RCO_API_KEY"); apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}
	req.Header.Set("Accept", "application/json")

	if ld.logger != nil {
		ld.logger.Printf("Fetching LCR from RCO: %s", endpoint)
	}

	resp, err := ld.makeHTTPRequestWithRetry(req, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch LCR: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("RCO API returned error status %d: %s", resp.StatusCode, string(body))
	}

	var response struct {
		LCR  *RCOLCR `json:"lcr,omitempty"`
		Data *RCOLCR `json:"data,omitempty"`
	}

	// Try parsing as direct object first
	var lcr *RCOLCR
	if err := json.Unmarshal(body, &lcr); err == nil && lcr != nil {
		return lcr, nil
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if response.LCR != nil {
		return response.LCR, nil
	}
	if response.Data != nil {
		return response.Data, nil
	}

	return nil, fmt.Errorf("no LCR data found in response")
}

// makeHTTPRequestWithRetry makes an HTTP request with retry logic and exponential backoff
func (ld *LiquidityDetector) makeHTTPRequestWithRetry(req *http.Request, maxRetries int) (*http.Response, error) {
	var resp *http.Response
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			waitTime := time.Duration(1<<uint(attempt-1)) * time.Second
			if ld.logger != nil {
				ld.logger.Printf("Retrying HTTP request (attempt %d/%d) after %v", attempt+1, maxRetries, waitTime)
			}
			select {
			case <-time.After(waitTime):
			case <-req.Context().Done():
				return nil, fmt.Errorf("request cancelled: %w", req.Context().Err())
			}
		}

		resp, err = ld.httpClient.Do(req)
		if err == nil {
			return resp, nil
		}

		if req.Context().Err() != nil {
			return nil, fmt.Errorf("request cancelled or timed out: %w", req.Context().Err())
		}

		if ld.logger != nil && attempt < maxRetries-1 {
			ld.logger.Printf("HTTP request failed (attempt %d/%d): %v", attempt+1, maxRetries, err)
		}
	}

	return nil, fmt.Errorf("failed after %d attempts: %w", maxRetries, err)
}

// detectLCRViolations detects LCR violations
func (ld *LiquidityDetector) detectLCRViolations(baseline, current *RCOLCR) []*Break {
	var breaks []*Break

	if current == nil {
		// Missing LCR
		if baseline != nil {
			br := &Break{
				BreakID:        fmt.Sprintf("break-missing-lcr-%d", time.Now().Unix()),
				SystemName:      SystemRCO,
				DetectionType:   DetectionTypeLiquidity,
				BreakType:       BreakTypeLCRViolation,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    nil,
				BaselineValue:   ld.lcrToMap(baseline),
				Difference:      map[string]interface{}{"missing": true},
				AffectedEntities: []string{"lcr"},
				DetectedAt:      time.Now(),
				CreatedAt:       time.Now(),
				UpdatedAt:       time.Now(),
			}
			breaks = append(breaks, br)
		}
		return breaks
	}

	// Check if LCR is below minimum
	if current.LCRValue < current.MinimumRequired {
		br := &Break{
			BreakID:        fmt.Sprintf("break-lcr-violation-%d", time.Now().Unix()),
			SystemName:      SystemRCO,
			DetectionType:   DetectionTypeLiquidity,
			BreakType:       BreakTypeLCRViolation,
			Severity:        SeverityCritical,
			Status:          BreakStatusOpen,
			CurrentValue:    ld.lcrToMap(current),
			BaselineValue:   ld.lcrToMap(baseline),
			Difference: map[string]interface{}{
				"lcr_value":       current.LCRValue,
				"minimum_required": current.MinimumRequired,
				"deficit":         current.MinimumRequired - current.LCRValue,
			},
			AffectedEntities: []string{"lcr"},
			DetectedAt:       time.Now(),
			CreatedAt:        time.Now(),
			UpdatedAt:        time.Now(),
		}
		breaks = append(breaks, br)
	}

	// Check for significant LCR changes
	if baseline != nil {
		lcrChange := current.LCRValue - baseline.LCRValue
		if lcrChange < 0 {
			lcrChange = -lcrChange
		}
		// Significant change threshold: 5% (0.05)
		if lcrChange > 0.05 {
			br := &Break{
				BreakID:        fmt.Sprintf("break-lcr-change-%d", time.Now().Unix()),
				SystemName:      SystemRCO,
				DetectionType:   DetectionTypeLiquidity,
				BreakType:       BreakTypeLCRViolation,
				Severity:        SeverityHigh,
				Status:          BreakStatusOpen,
				CurrentValue:    ld.lcrToMap(current),
				BaselineValue:   ld.lcrToMap(baseline),
				Difference: map[string]interface{}{
					"baseline_lcr": baseline.LCRValue,
					"current_lcr":  current.LCRValue,
					"change":       current.LCRValue - baseline.LCRValue,
				},
				AffectedEntities: []string{"lcr"},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, br)
		}
	}

	return breaks
}

// detectLiquidityMismatches detects liquidity requirement mismatches
func (ld *LiquidityDetector) detectLiquidityMismatches(baseline, current map[string]*RCOLiquidityPosition) []*Break {
	var breaks []*Break
	tolerance := 0.01

	for positionID, baselinePosition := range baseline {
		currentPosition, exists := current[positionID]
		if !exists {
			continue // Already handled by position mismatch detection
		}

		// Check liquidity requirement mismatch
		requirementDiff := currentPosition.LiquidityRequirement - baselinePosition.LiquidityRequirement
		if requirementDiff < 0 {
			requirementDiff = -requirementDiff
		}
		if requirementDiff > tolerance {
			br := &Break{
				BreakID:        fmt.Sprintf("break-liquidity-requirement-mismatch-%s", positionID),
				SystemName:      SystemRCO,
				DetectionType:   DetectionTypeLiquidity,
				BreakType:       BreakTypeLiquidityMismatch,
				Severity:        SeverityHigh,
				Status:          BreakStatusOpen,
				CurrentValue:    ld.liquidityPositionToMap(currentPosition),
				BaselineValue:   ld.liquidityPositionToMap(baselinePosition),
				Difference: map[string]interface{}{
					"field":      "liquidity_requirement",
					"baseline":   baselinePosition.LiquidityRequirement,
					"current":   currentPosition.LiquidityRequirement,
					"difference": requirementDiff,
				},
				AffectedEntities: []string{positionID},
				DetectedAt:       time.Now(),
				CreatedAt:        time.Now(),
				UpdatedAt:        time.Now(),
			}
			breaks = append(breaks, br)
		}
	}

	return breaks
}

// detectPositionMismatches detects liquidity position mismatches
func (ld *LiquidityDetector) detectPositionMismatches(baseline, current map[string]*RCOLiquidityPosition) []*Break {
	var breaks []*Break
	tolerance := 0.01

	for positionID, baselinePosition := range baseline {
		currentPosition, exists := current[positionID]
		if !exists {
			// Missing position
			br := &Break{
				BreakID:        fmt.Sprintf("break-missing-position-%s", positionID),
				SystemName:      SystemRCO,
				DetectionType:   DetectionTypeLiquidity,
				BreakType:       BreakTypePositionMismatch,
				Severity:        SeverityHigh,
				Status:          BreakStatusOpen,
				CurrentValue:    nil,
				BaselineValue:   ld.liquidityPositionToMap(baselinePosition),
				Difference:      map[string]interface{}{"missing": true},
				AffectedEntities: []string{positionID},
				DetectedAt:      time.Now(),
				CreatedAt:       time.Now(),
				UpdatedAt:       time.Now(),
			}
			breaks = append(breaks, br)
			continue
		}

		// Check position amount mismatch
		amountDiff := currentPosition.PositionAmount - baselinePosition.PositionAmount
		if amountDiff < 0 {
			amountDiff = -amountDiff
		}
		if amountDiff > tolerance {
			br := &Break{
				BreakID:        fmt.Sprintf("break-position-amount-mismatch-%s", positionID),
				SystemName:      SystemRCO,
				DetectionType:   DetectionTypeLiquidity,
				BreakType:       BreakTypePositionMismatch,
				Severity:        SeverityCritical,
				Status:          BreakStatusOpen,
				CurrentValue:    ld.liquidityPositionToMap(currentPosition),
				BaselineValue:   ld.liquidityPositionToMap(baselinePosition),
				Difference: map[string]interface{}{
					"field":      "position_amount",
					"baseline":   baselinePosition.PositionAmount,
					"current":   currentPosition.PositionAmount,
					"difference": amountDiff,
				},
				AffectedEntities: []string{positionID},
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
func (ld *LiquidityDetector) liquidityPositionToMap(position *RCOLiquidityPosition) map[string]interface{} {
	return map[string]interface{}{
		"position_id":          position.PositionID,
		"position_amount":      position.PositionAmount,
		"currency":             position.Currency,
		"position_type":        position.PositionType,
		"liquidity_requirement": position.LiquidityRequirement,
	}
}

func (ld *LiquidityDetector) lcrToMap(lcr *RCOLCR) map[string]interface{} {
	if lcr == nil {
		return nil
	}
	return map[string]interface{}{
		"lcr_value":       lcr.LCRValue,
		"minimum_required": lcr.MinimumRequired,
		"status":          lcr.Status,
		"hqla":           lcr.HQLA,
		"net_cash_outflow": lcr.NetCashOutflow,
	}
}

