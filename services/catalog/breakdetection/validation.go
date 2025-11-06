package breakdetection

import (
	"fmt"
	"time"
)

// ValidationConfig defines validation rules for break detection
type ValidationConfig struct {
	MinTolerance         float64       // Minimum tolerance for numeric comparisons
	MaxTolerance         float64       // Maximum tolerance for numeric comparisons
	PercentTolerance     float64       // Percentage tolerance (e.g., 0.01 for 1%)
	TimeWindow          time.Duration // Time window for temporal comparisons
	MaxAffectedEntities int           // Maximum number of affected entities per break
	RequireTimestamp    bool          // Whether timestamps are required
}

// DefaultValidationConfig returns default validation configuration
func DefaultValidationConfig() *ValidationConfig {
	return &ValidationConfig{
		MinTolerance:         0.01,
		MaxTolerance:         1000000.0,
		PercentTolerance:     0.0001, // 0.01%
		TimeWindow:           24 * time.Hour,
		MaxAffectedEntities: 1000,
		RequireTimestamp:    true,
	}
}

// ValidateBreak validates a detected break before storing
func ValidateBreak(b *Break, config *ValidationConfig) error {
	if config == nil {
		config = DefaultValidationConfig()
	}

	// Validate required fields
	if b.BreakID == "" {
		return fmt.Errorf("break_id is required")
	}
	if b.SystemName == "" {
		return fmt.Errorf("system_name is required")
	}
	if b.DetectionType == "" {
		return fmt.Errorf("detection_type is required")
	}
	if b.BreakType == "" {
		return fmt.Errorf("break_type is required")
	}
	if b.Severity == "" {
		return fmt.Errorf("severity is required")
	}
	if b.Status == "" {
		return fmt.Errorf("status is required")
	}

	// Validate timestamp
	if config.RequireTimestamp && b.DetectedAt.IsZero() {
		return fmt.Errorf("detected_at timestamp is required")
	}

	// Validate affected entities
	if len(b.AffectedEntities) > config.MaxAffectedEntities {
		return fmt.Errorf("too many affected entities: %d (max: %d)", 
			len(b.AffectedEntities), config.MaxAffectedEntities)
	}

	// Validate severity is valid
	if !isValidSeverity(b.Severity) {
		return fmt.Errorf("invalid severity: %s", b.Severity)
	}

	// Validate break type is valid
	if !isValidBreakType(b.BreakType) {
		return fmt.Errorf("invalid break_type: %s", b.BreakType)
	}

	// Validate difference calculation
	if b.Difference != nil {
		if err := validateDifference(b.Difference, config); err != nil {
			return fmt.Errorf("invalid difference: %w", err)
		}
	}

	return nil
}

// ValidateDetectionRequest validates a detection request
func ValidateDetectionRequest(req *DetectionRequest) error {
	if req == nil {
		return fmt.Errorf("detection request is required")
	}

	if req.SystemName == "" {
		return fmt.Errorf("system_name is required")
	}

	if req.BaselineID == "" {
		return fmt.Errorf("baseline_id is required")
	}

	if req.DetectionType == "" {
		return fmt.Errorf("detection_type is required")
	}

	// Validate detection type
	validTypes := []DetectionType{
		DetectionTypeFinance,
		DetectionTypeCapital,
		DetectionTypeLiquidity,
		DetectionTypeRegulatory,
	}
	
	isValid := false
	for _, validType := range validTypes {
		if req.DetectionType == validType {
			isValid = true
			break
		}
	}
	
	if !isValid {
		return fmt.Errorf("invalid detection_type: %s (valid: finance, capital, liquidity, regulatory)", req.DetectionType)
	}

	return nil
}

// ValidateBaseline validates a baseline before storage
func ValidateBaseline(b *Baseline) error {
	if b == nil {
		return fmt.Errorf("baseline is required")
	}

	if b.BaselineID == "" {
		return fmt.Errorf("baseline_id is required")
	}

	if b.SystemName == "" {
		return fmt.Errorf("system_name is required")
	}

	if b.Version == "" {
		return fmt.Errorf("version is required")
	}

	if len(b.SnapshotData) == 0 {
		return fmt.Errorf("snapshot_data cannot be empty")
	}

	// Validate snapshot data size (max 100MB)
	maxSize := 100 * 1024 * 1024 // 100MB
	if len(b.SnapshotData) > maxSize {
		return fmt.Errorf("snapshot_data too large: %d bytes (max: %d bytes)", 
			len(b.SnapshotData), maxSize)
	}

	return nil
}

// Helper functions
func isValidSeverity(s Severity) bool {
	validSeverities := []Severity{
		SeverityCritical,
		SeverityHigh,
		SeverityMedium,
		SeverityLow,
	}
	for _, valid := range validSeverities {
		if s == valid {
			return true
		}
	}
	return false
}

func isValidBreakType(bt BreakType) bool {
	validTypes := []BreakType{
		// Finance
		BreakTypeMissingEntry,
		BreakTypeAmountMismatch,
		BreakTypeBalanceBreak,
		BreakTypeReconciliationBreak,
		BreakTypeAccountMismatch,
		// Capital
		BreakTypeCapitalRatioViolation,
		BreakTypeRWAError,
		BreakTypeExposureMismatch,
		// Liquidity
		BreakTypeLCRViolation,
		BreakTypeLiquidityMismatch,
		BreakTypePositionMismatch,
		// Regulatory
		BreakTypeComplianceViolation,
		BreakTypeReportingBreak,
		BreakTypeCalculationError,
	}
	for _, valid := range validTypes {
		if bt == valid {
			return true
		}
	}
	return false
}

func validateDifference(diff map[string]interface{}, config *ValidationConfig) error {
	// Validate numeric differences
	if diffValue, ok := diff["difference"].(float64); ok {
		if diffValue < 0 {
			diffValue = -diffValue
		}
		if diffValue > config.MaxTolerance {
			return fmt.Errorf("difference exceeds maximum tolerance: %f (max: %f)", 
				diffValue, config.MaxTolerance)
		}
	}

	// Validate percentage differences
	if percentDiff, ok := diff["percent_difference"].(float64); ok {
		if percentDiff < 0 {
			percentDiff = -percentDiff
		}
		if percentDiff > config.PercentTolerance*100 {
			return fmt.Errorf("percentage difference exceeds tolerance: %.2f%% (max: %.2f%%)", 
				percentDiff, config.PercentTolerance*100)
		}
	}

	return nil
}

// CalculateSeverity calculates severity based on difference magnitude
func CalculateSeverity(diff map[string]interface{}, breakType BreakType, config *ValidationConfig) Severity {
	if diff == nil {
		return SeverityMedium
	}

	// Extract difference value
	var diffValue float64
	if d, ok := diff["difference"].(float64); ok {
		if d < 0 {
			d = -d
		}
		diffValue = d
	} else if percentDiff, ok := diff["percent_difference"].(float64); ok {
		if percentDiff < 0 {
			percentDiff = -percentDiff
		}
		// Convert percentage to absolute if we have baseline
		if baseline, ok := diff["baseline"].(float64); ok && baseline != 0 {
			diffValue = baseline * percentDiff / 100.0
		} else {
			// Use percentage as proxy
			diffValue = percentDiff
		}
	}

	// Critical: Reconciliation breaks, large amount mismatches
	if breakType == BreakTypeReconciliationBreak {
		return SeverityCritical
	}
	if diffValue > config.MaxTolerance*0.1 { // > 10% of max tolerance
		return SeverityCritical
	}

	// High: Balance breaks, medium amount mismatches
	if breakType == BreakTypeBalanceBreak {
		return SeverityHigh
	}
	if diffValue > config.MaxTolerance*0.01 { // > 1% of max tolerance
		return SeverityHigh
	}

	// Medium: Account mismatches, small amount mismatches
	if breakType == BreakTypeAccountMismatch {
		return SeverityMedium
	}
	if diffValue > config.MinTolerance {
		return SeverityMedium
	}

	// Low: Minor differences
	return SeverityLow
}

