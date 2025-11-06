package breakdetection

import (
	"encoding/json"
	"time"
)

// BreakType represents the type of break detected
type BreakType string

const (
	// Finance break types
	BreakTypeMissingEntry        BreakType = "missing_entry"
	BreakTypeAmountMismatch      BreakType = "amount_mismatch"
	BreakTypeBalanceBreak        BreakType = "balance_break"
	BreakTypeReconciliationBreak BreakType = "reconciliation_break"
	BreakTypeAccountMismatch     BreakType = "account_mismatch"
	
	// Capital break types
	BreakTypeCapitalRatioViolation BreakType = "capital_ratio_violation"
	BreakTypeRWAError              BreakType = "rwa_error"
	BreakTypeExposureMismatch      BreakType = "exposure_mismatch"
	
	// Liquidity break types
	BreakTypeLCRViolation        BreakType = "lcr_violation"
	BreakTypeLiquidityMismatch   BreakType = "liquidity_mismatch"
	BreakTypePositionMismatch    BreakType = "position_mismatch"
	
	// Regulatory break types
	BreakTypeComplianceViolation BreakType = "compliance_violation"
	BreakTypeReportingBreak       BreakType = "reporting_break"
	BreakTypeCalculationError     BreakType = "calculation_error"
)

// DetectionType represents the type of system being checked
type DetectionType string

const (
	DetectionTypeFinance    DetectionType = "finance"
	DetectionTypeCapital    DetectionType = "capital"
	DetectionTypeLiquidity  DetectionType = "liquidity"
	DetectionTypeRegulatory DetectionType = "regulatory"
)

// SystemName represents the system being monitored
type SystemName string

const (
	SystemSAPFioneer SystemName = "sap_fioneer"
	SystemBCRS       SystemName = "bcrs"
	SystemRCO        SystemName = "rco"
	SystemAxiomSL    SystemName = "axiomsl"
)

// Severity represents the severity of a break
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
)

// BreakStatus represents the status of a break
type BreakStatus string

const (
	BreakStatusOpen          BreakStatus = "open"
	BreakStatusInvestigating BreakStatus = "investigating"
	BreakStatusResolved      BreakStatus = "resolved"
	BreakStatusFalsePositive BreakStatus = "false_positive"
)

// RunStatus represents the status of a detection run
type RunStatus string

const (
	RunStatusRunning   RunStatus = "running"
	RunStatusCompleted RunStatus = "completed"
	RunStatusFailed    RunStatus = "failed"
	RunStatusCancelled RunStatus = "cancelled"
)

// Baseline represents a baseline snapshot for comparison
type Baseline struct {
	ID           string                 `json:"id"`
	BaselineID   string                 `json:"baseline_id"`
	SystemName   SystemName             `json:"system_name"`
	Version      string                 `json:"version"`
	SnapshotType string                 `json:"snapshot_type"` // "full", "incremental", "point_in_time"
	SnapshotData json.RawMessage        `json:"snapshot_data"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
	CreatedBy    string                 `json:"created_by,omitempty"`
	ExpiresAt    *time.Time             `json:"expires_at,omitempty"`
	IsActive     bool                   `json:"is_active"`
}

// Break represents a detected break
type Break struct {
	ID                string                 `json:"id"`
	BreakID           string                 `json:"break_id"`
	RunID             string                 `json:"run_id"`
	SystemName        SystemName             `json:"system_name"`
	DetectionType     DetectionType           `json:"detection_type"`
	BreakType         BreakType               `json:"break_type"`
	Severity          Severity                `json:"severity"`
	Status            BreakStatus             `json:"status"`
	
	// Break details
	CurrentValue      map[string]interface{} `json:"current_value"`
	BaselineValue     map[string]interface{} `json:"baseline_value"`
	Difference        map[string]interface{} `json:"difference"`
	AffectedEntities  []string               `json:"affected_entities"`
	
	// Analysis (from Deep Research)
	RootCauseAnalysis string                 `json:"root_cause_analysis,omitempty"`
	SemanticEnrichment map[string]interface{} `json:"semantic_enrichment,omitempty"`
	Recommendations    []string               `json:"recommendations,omitempty"`
	
	// AI Analysis (from LocalAI)
	AIDescription     string                 `json:"ai_description,omitempty"`
	AICategory        string                 `json:"ai_category,omitempty"`
	AIPriorityScore   float64                `json:"ai_priority_score,omitempty"`
	
	// Search integration
	SimilarBreaks     []SimilarBreak         `json:"similar_breaks,omitempty"`
	
	// Metadata
	DetectedAt        time.Time              `json:"detected_at"`
	ResolvedAt        *time.Time             `json:"resolved_at,omitempty"`
	ResolvedBy        string                 `json:"resolved_by,omitempty"`
	ResolutionNotes   string                 `json:"resolution_notes,omitempty"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
}

// SimilarBreak represents a similar historical break found via search
type SimilarBreak struct {
	BreakID       string  `json:"break_id"`
	Similarity    float64 `json:"similarity"` // 0.0 to 1.0
	Description   string  `json:"description"`
	Resolution    string  `json:"resolution,omitempty"`
	DetectedAt    time.Time `json:"detected_at"`
}

// DetectionRun represents a break detection execution
type DetectionRun struct {
	ID                  string                 `json:"id"`
	RunID               string                 `json:"run_id"`
	SystemName          SystemName             `json:"system_name"`
	BaselineID          string                 `json:"baseline_id"`
	DetectionType       DetectionType           `json:"detection_type"`
	Status              RunStatus              `json:"status"`
	StartedAt           time.Time              `json:"started_at"`
	CompletedAt         *time.Time             `json:"completed_at,omitempty"`
	TotalBreaksDetected int                    `json:"total_breaks_detected"`
	TotalRecordsChecked int                    `json:"total_records_checked"`
	Configuration       map[string]interface{} `json:"configuration,omitempty"`
	ResultSummary       map[string]interface{} `json:"result_summary,omitempty"`
	ErrorMessage        string                 `json:"error_message,omitempty"`
	CreatedBy           string                 `json:"created_by,omitempty"`
	WorkflowInstanceID  string                 `json:"workflow_instance_id,omitempty"`
	
	// Breaks detected in this run
	Breaks               []*Break              `json:"breaks,omitempty"`
}

// DetectionRequest represents a request to detect breaks
type DetectionRequest struct {
	SystemName      SystemName             `json:"system_name"`
	BaselineID      string                 `json:"baseline_id"`
	DetectionType   DetectionType           `json:"detection_type"`
	Configuration   map[string]interface{} `json:"configuration,omitempty"`
	WorkflowInstanceID string              `json:"workflow_instance_id,omitempty"`
	CreatedBy       string                 `json:"created_by,omitempty"`
}

// DetectionResult represents the result of a break detection
type DetectionResult struct {
	RunID               string                 `json:"run_id"`
	Status              RunStatus              `json:"status"`
	TotalBreaksDetected int                    `json:"total_breaks_detected"`
	TotalRecordsChecked int                    `json:"total_records_checked"`
	Breaks              []*Break               `json:"breaks"`
	ResultSummary       map[string]interface{} `json:"result_summary,omitempty"`
	ErrorMessage        string                 `json:"error_message,omitempty"`
}

// BreakComparison represents the comparison between current and baseline
type BreakComparison struct {
	CurrentData  map[string]interface{} `json:"current_data"`
	BaselineData map[string]interface{} `json:"baseline_data"`
	Differences  []Difference           `json:"differences"`
	Match        bool                   `json:"match"`
}

// Difference represents a single difference between current and baseline
type Difference struct {
	Field         string      `json:"field"`
	CurrentValue  interface{} `json:"current_value"`
	BaselineValue interface{} `json:"baseline_value"`
	Difference    interface{} `json:"difference,omitempty"`
	IsBreak       bool        `json:"is_break"` // True if this difference constitutes a break
}

// BaselineRequest represents a request to create a baseline
type BaselineRequest struct {
	SystemName    SystemName             `json:"system_name"`
	Version       string                 `json:"version"`
	SnapshotType string                 `json:"snapshot_type"`
	SnapshotData  json.RawMessage        `json:"snapshot_data"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	ExpiresAt     *time.Time             `json:"expires_at,omitempty"`
	CreatedBy     string                 `json:"created_by,omitempty"`
}

