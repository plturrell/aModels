package agents

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
)

// PostgresMappingRuleStore implements MappingRuleStore using PostgreSQL.
type PostgresMappingRuleStore struct {
	db     *sql.DB
	logger *log.Logger
}

// NewPostgresMappingRuleStore creates a new PostgreSQL mapping rule store.
func NewPostgresMappingRuleStore(db *sql.DB, logger *log.Logger) *PostgresMappingRuleStore {
	return &PostgresMappingRuleStore{
		db:     db,
		logger: logger,
	}
}

// GetRules retrieves mapping rules.
func (prs *PostgresMappingRuleStore) GetRules(ctx context.Context, sourceType string, version string) (*MappingRules, error) {
	query := `
		SELECT rules_json, version, confidence
		FROM mapping_rules
		WHERE source_type = $1 AND version = $2
		ORDER BY created_at DESC
		LIMIT 1
	`

	var rulesJSON []byte
	var v string
	var confidence float64

	err := prs.db.QueryRowContext(ctx, query, sourceType, version).Scan(&rulesJSON, &v, &confidence)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("mapping rules not found for %s version %s", sourceType, version)
		}
		return nil, fmt.Errorf("failed to get rules: %w", err)
	}

	var rules MappingRules
	if err := json.Unmarshal(rulesJSON, &rules); err != nil {
		return nil, fmt.Errorf("failed to unmarshal rules: %w", err)
	}

	rules.Version = v
	rules.Confidence = confidence

	return &rules, nil
}

// SaveRules saves mapping rules.
func (prs *PostgresMappingRuleStore) SaveRules(ctx context.Context, rules *MappingRules) error {
	rulesJSON, err := json.Marshal(rules)
	if err != nil {
		return fmt.Errorf("failed to marshal rules: %w", err)
	}

	query := `
		INSERT INTO mapping_rules (source_type, version, rules_json, confidence, created_at)
		VALUES ($1, $2, $3, $4, NOW())
		ON CONFLICT (source_type, version) DO UPDATE
		SET rules_json = EXCLUDED.rules_json,
		    confidence = EXCLUDED.confidence,
		    updated_at = NOW()
	`

	_, err = prs.db.ExecContext(ctx, query, rules.SourceType(), rules.Version, rulesJSON, rules.Confidence)
	if err != nil {
		return fmt.Errorf("failed to save rules: %w", err)
	}

	return nil
}

// ListRules lists all rules for a source type.
func (prs *PostgresMappingRuleStore) ListRules(ctx context.Context, sourceType string) ([]*MappingRules, error) {
	query := `
		SELECT rules_json, version, confidence
		FROM mapping_rules
		WHERE source_type = $1
		ORDER BY created_at DESC
	`

	rows, err := prs.db.QueryContext(ctx, query, sourceType)
	if err != nil {
		return nil, fmt.Errorf("failed to query rules: %w", err)
	}
	defer rows.Close()

	var rulesList []*MappingRules
	for rows.Next() {
		var rulesJSON []byte
		var version string
		var confidence float64

		if err := rows.Scan(&rulesJSON, &version, &confidence); err != nil {
			return nil, fmt.Errorf("failed to scan rules: %w", err)
		}

		var rules MappingRules
		if err := json.Unmarshal(rulesJSON, &rules); err != nil {
			return nil, fmt.Errorf("failed to unmarshal rules: %w", err)
		}

		rules.Version = version
		rules.Confidence = confidence
		rulesList = append(rulesList, &rules)
	}

	return rulesList, nil
}

// Helper method for MappingRules to get source type
func (mr *MappingRules) SourceType() string {
	if len(mr.NodeMappings) > 0 {
		// Extract from first node mapping's source table
		return mr.NodeMappings[0].SourceTable
	}
	return "unknown"
}

