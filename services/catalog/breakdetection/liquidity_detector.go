package breakdetection

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
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
	// TODO: Implement actual API call to RCO
	positions := make(map[string]*RCOLiquidityPosition)

	// Placeholder: In production, this would call RCO API
	// url := fmt.Sprintf("%s/api/liquidity-positions", ld.rcoURL)
	// resp, err := ld.httpClient.Get(url)
	// ... parse response ...

	return positions, nil
}

// fetchCurrentLCR fetches current LCR from RCO
func (ld *LiquidityDetector) fetchCurrentLCR(ctx context.Context) (*RCOLCR, error) {
	// TODO: Implement actual API call to RCO
	// Placeholder: In production, this would call RCO API
	// url := fmt.Sprintf("%s/api/lcr", ld.rcoURL)
	// resp, err := ld.httpClient.Get(url)
	// ... parse response ...

	return nil, nil
}

// detectLCRViolations detects LCR violations
func (ld *LiquidityDetector) detectLCRViolations(baseline, current *RCOLCR) []*Break {
	var breaks []*Break

	if current == nil {
		// Missing LCR
		if baseline != nil {
			break := &Break{
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
			breaks = append(breaks, break)
		}
		return breaks
	}

	// Check if LCR is below minimum
	if current.LCRValue < current.MinimumRequired {
		break := &Break{
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
		breaks = append(breaks, break)
	}

	// Check for significant LCR changes
	if baseline != nil {
		lcrChange := current.LCRValue - baseline.LCRValue
		if lcrChange < 0 {
			lcrChange = -lcrChange
		}
		// Significant change threshold: 5% (0.05)
		if lcrChange > 0.05 {
			break := &Break{
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
			breaks = append(breaks, break)
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
			break := &Break{
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
			breaks = append(breaks, break)
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
			break := &Break{
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
			breaks = append(breaks, break)
			continue
		}

		// Check position amount mismatch
		amountDiff := currentPosition.PositionAmount - baselinePosition.PositionAmount
		if amountDiff < 0 {
			amountDiff = -amountDiff
		}
		if amountDiff > tolerance {
			break := &Break{
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
			breaks = append(breaks, break)
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

