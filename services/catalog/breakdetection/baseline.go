package breakdetection

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
)

// BaselineManager manages baseline snapshots for break detection
type BaselineManager struct {
	db     *sql.DB
	logger *log.Logger
}

// NewBaselineManager creates a new baseline manager
func NewBaselineManager(db *sql.DB, logger *log.Logger) *BaselineManager {
	return &BaselineManager{
		db:     db,
		logger: logger,
	}
}

// CreateBaseline creates a new baseline snapshot
func (bm *BaselineManager) CreateBaseline(ctx context.Context, req *BaselineRequest) (*Baseline, error) {
	baselineID := fmt.Sprintf("baseline-%s-%s-%s", req.SystemName, req.Version, time.Now().Format("20060102-150405"))
	
	id := uuid.New().String()
	snapshotJSON, err := json.Marshal(req.SnapshotData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal snapshot data: %w", err)
	}
	
	metadataJSON, err := json.Marshal(req.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal metadata: %w", err)
	}
	
	var expiresAt *time.Time
	if req.ExpiresAt != nil {
		expiresAt = req.ExpiresAt
	}
	
	query := `
		INSERT INTO break_detection_baselines (
			id, baseline_id, system_name, version, snapshot_type,
			snapshot_data, metadata, created_by, expires_at, is_active
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING id, created_at
	`
	
	var createdAt time.Time
	err = bm.db.QueryRowContext(ctx, query,
		id, baselineID, string(req.SystemName), req.Version, req.SnapshotType,
		snapshotJSON, metadataJSON, req.CreatedBy, expiresAt, true,
	).Scan(&id, &createdAt)
	
	if err != nil {
		return nil, fmt.Errorf("failed to create baseline: %w", err)
	}
	
	if bm.logger != nil {
		bm.logger.Printf("Created baseline: %s for system: %s, version: %s", baselineID, req.SystemName, req.Version)
	}
	
	return &Baseline{
		ID:           id,
		BaselineID:   baselineID,
		SystemName:  req.SystemName,
		Version:      req.Version,
		SnapshotType: req.SnapshotType,
		SnapshotData: req.SnapshotData,
		Metadata:     req.Metadata,
		CreatedAt:    createdAt,
		CreatedBy:    req.CreatedBy,
		ExpiresAt:    expiresAt,
		IsActive:     true,
	}, nil
}

// GetBaseline retrieves a baseline by ID
func (bm *BaselineManager) GetBaseline(ctx context.Context, baselineID string) (*Baseline, error) {
	query := `
		SELECT id, baseline_id, system_name, version, snapshot_type,
		       snapshot_data, metadata, created_at, created_by, expires_at, is_active
		FROM break_detection_baselines
		WHERE baseline_id = $1 AND is_active = true
	`
	
	var b Baseline
	var snapshotJSON, metadataJSON []byte
	var expiresAt sql.NullTime
	
	err := bm.db.QueryRowContext(ctx, query, baselineID).Scan(
		&b.ID, &b.BaselineID, &b.SystemName, &b.Version, &b.SnapshotType,
		&snapshotJSON, &metadataJSON, &b.CreatedAt, &b.CreatedBy, &expiresAt, &b.IsActive,
	)
	
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("baseline not found: %s", baselineID)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get baseline: %w", err)
	}
	
	b.SnapshotData = snapshotJSON
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &b.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
	}
	if expiresAt.Valid {
		b.ExpiresAt = &expiresAt.Time
	}
	
	return &b, nil
}

// GetBaselineBySystemVersion retrieves a baseline by system and version
func (bm *BaselineManager) GetBaselineBySystemVersion(ctx context.Context, systemName SystemName, version string) (*Baseline, error) {
	query := `
		SELECT id, baseline_id, system_name, version, snapshot_type,
		       snapshot_data, metadata, created_at, created_by, expires_at, is_active
		FROM break_detection_baselines
		WHERE system_name = $1 AND version = $2 AND is_active = true
		ORDER BY created_at DESC
		LIMIT 1
	`
	
	var b Baseline
	var snapshotJSON, metadataJSON []byte
	var expiresAt sql.NullTime
	
	err := bm.db.QueryRowContext(ctx, query, string(systemName), version).Scan(
		&b.ID, &b.BaselineID, &b.SystemName, &b.Version, &b.SnapshotType,
		&snapshotJSON, &metadataJSON, &b.CreatedAt, &b.CreatedBy, &expiresAt, &b.IsActive,
	)
	
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("baseline not found for system: %s, version: %s", systemName, version)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get baseline: %w", err)
	}
	
	b.SnapshotData = snapshotJSON
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &b.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
	}
	if expiresAt.Valid {
		b.ExpiresAt = &expiresAt.Time
	}
	
	return &b, nil
}

// ListBaselines lists baselines for a system
func (bm *BaselineManager) ListBaselines(ctx context.Context, systemName SystemName, limit int) ([]*Baseline, error) {
	query := `
		SELECT id, baseline_id, system_name, version, snapshot_type,
		       snapshot_data, metadata, created_at, created_by, expires_at, is_active
		FROM break_detection_baselines
		WHERE system_name = $1 AND is_active = true
		ORDER BY created_at DESC
		LIMIT $2
	`
	
	rows, err := bm.db.QueryContext(ctx, query, string(systemName), limit)
	if err != nil {
		return nil, fmt.Errorf("failed to list baselines: %w", err)
	}
	defer rows.Close()
	
	var baselines []*Baseline
	for rows.Next() {
		var b Baseline
		var snapshotJSON, metadataJSON []byte
		var expiresAt sql.NullTime
		
		err := rows.Scan(
			&b.ID, &b.BaselineID, &b.SystemName, &b.Version, &b.SnapshotType,
			&snapshotJSON, &metadataJSON, &b.CreatedAt, &b.CreatedBy, &expiresAt, &b.IsActive,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan baseline: %w", err)
		}
		
		b.SnapshotData = snapshotJSON
		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &b.Metadata); err != nil {
				// Log but continue
				if bm.logger != nil {
					bm.logger.Printf("Warning: Failed to unmarshal metadata: %v", err)
				}
			}
		}
		if expiresAt.Valid {
			b.ExpiresAt = &expiresAt.Time
		}
		
		baselines = append(baselines, &b)
	}
	
	return baselines, nil
}

// DeactivateBaseline deactivates a baseline
func (bm *BaselineManager) DeactivateBaseline(ctx context.Context, baselineID string) error {
	query := `
		UPDATE break_detection_baselines
		SET is_active = false
		WHERE baseline_id = $1
	`
	
	result, err := bm.db.ExecContext(ctx, query, baselineID)
	if err != nil {
		return fmt.Errorf("failed to deactivate baseline: %w", err)
	}
	
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	
	if rowsAffected == 0 {
		return fmt.Errorf("baseline not found: %s", baselineID)
	}
	
	if bm.logger != nil {
		bm.logger.Printf("Deactivated baseline: %s", baselineID)
	}
	
	return nil
}

// CompareBaselines compares two baselines
func (bm *BaselineManager) CompareBaselines(ctx context.Context, baseline1ID, baseline2ID string) (*BreakComparison, error) {
	baseline1, err := bm.GetBaseline(ctx, baseline1ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get baseline 1: %w", err)
	}
	
	baseline2, err := bm.GetBaseline(ctx, baseline2ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get baseline 2: %w", err)
	}
	
	// Parse snapshot data
	var data1, data2 map[string]interface{}
	if err := json.Unmarshal(baseline1.SnapshotData, &data1); err != nil {
		return nil, fmt.Errorf("failed to unmarshal baseline 1 data: %w", err)
	}
	if err := json.Unmarshal(baseline2.SnapshotData, &data2); err != nil {
		return nil, fmt.Errorf("failed to unmarshal baseline 2 data: %w", err)
	}
	
	// Compare data
	comparison := &BreakComparison{
		CurrentData:  data1,
		BaselineData: data2,
		Differences:  []Difference{},
		Match:        true,
	}
	
	// Find differences (simplified - would need more sophisticated comparison)
	differences := findDifferences(data1, data2, "")
	comparison.Differences = differences
	
	// Check if any differences constitute breaks
	for _, diff := range differences {
		if diff.IsBreak {
			comparison.Match = false
			break
		}
	}
	
	return comparison, nil
}

// findDifferences recursively finds differences between two maps
func findDifferences(current, baseline map[string]interface{}, prefix string) []Difference {
	var differences []Difference
	
	// Check all keys in current
	for key, currentVal := range current {
		fieldPath := key
		if prefix != "" {
			fieldPath = prefix + "." + key
		}
		
		baselineVal, exists := baseline[key]
		if !exists {
			differences = append(differences, Difference{
				Field:         fieldPath,
				CurrentValue:  currentVal,
				BaselineValue: nil,
				IsBreak:       true, // Missing field is a break
			})
			continue
		}
		
		// Compare values
		if !valuesEqual(currentVal, baselineVal) {
			isBreak := false
			// Determine if this is a break based on type and magnitude
			if isSignificantDifference(currentVal, baselineVal) {
				isBreak = true
			}
			
			differences = append(differences, Difference{
				Field:         fieldPath,
				CurrentValue:  currentVal,
				BaselineValue: baselineVal,
				Difference:    calculateDifference(currentVal, baselineVal),
				IsBreak:       isBreak,
			})
		}
		
		// Recursively check nested maps
		if currentMap, ok := currentVal.(map[string]interface{}); ok {
			if baselineMap, ok := baselineVal.(map[string]interface{}); ok {
				nestedDiffs := findDifferences(currentMap, baselineMap, fieldPath)
				differences = append(differences, nestedDiffs...)
			}
		}
	}
	
	// Check for keys in baseline that don't exist in current
	for key, baselineVal := range baseline {
		if _, exists := current[key]; !exists {
			fieldPath := key
			if prefix != "" {
				fieldPath = prefix + "." + key
			}
			differences = append(differences, Difference{
				Field:         fieldPath,
				CurrentValue:  nil,
				BaselineValue: baselineVal,
				IsBreak:       true, // Missing field is a break
			})
		}
	}
	
	return differences
}

// valuesEqual checks if two values are equal
func valuesEqual(a, b interface{}) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a == b
}

// isSignificantDifference determines if a difference is significant enough to be a break
func isSignificantDifference(current, baseline interface{}) bool {
	// For numeric types, check if difference exceeds threshold
	if currentFloat, ok := current.(float64); ok {
		if baselineFloat, ok := baseline.(float64); ok {
			diff := currentFloat - baselineFloat
			if diff < 0 {
				diff = -diff
			}
			// Threshold: 0.01% or 0.01 absolute, whichever is larger
			threshold := 0.01
			if baselineFloat != 0 {
				percentThreshold := baselineFloat * 0.0001 // 0.01%
				if percentThreshold > threshold {
					threshold = percentThreshold
				}
			}
			return diff > threshold
		}
	}
	
	// For strings, any difference is significant
	if _, ok := current.(string); ok {
		return current != baseline
	}
	
	// For other types, check equality
	return current != baseline
}

// calculateDifference calculates the difference between two values
func calculateDifference(current, baseline interface{}) interface{} {
	if currentFloat, ok := current.(float64); ok {
		if baselineFloat, ok := baseline.(float64); ok {
			return currentFloat - baselineFloat
		}
	}
	return nil
}

