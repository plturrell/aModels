package breakdetection

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/catalog/research"
)

// RuleGeneratorService generates break detection rules from regulatory specifications using Deep Research
type RuleGeneratorService struct {
	deepResearchClient *research.DeepResearchClient
	db                 *sql.DB
	logger             *log.Logger
}

// NewRuleGeneratorService creates a new rule generator service
func NewRuleGeneratorService(deepResearchClient *research.DeepResearchClient, db *sql.DB, logger *log.Logger) *RuleGeneratorService {
	return &RuleGeneratorService{
		deepResearchClient: deepResearchClient,
		db:                 db,
		logger:             logger,
	}
}

// DetectionRule represents a generated or manual break detection rule
type DetectionRule struct {
	RuleID          string                 `json:"rule_id"`
	SystemName      SystemName             `json:"system_name"`
	DetectionType   DetectionType           `json:"detection_type"`
	RuleType        string                 `json:"rule_type"` // "auto_generated", "manual", "regulatory"
	RuleSource      string                 `json:"rule_source"` // "deep_research", "regulatory_spec", "manual"
	RuleName        string                 `json:"rule_name"`
	RuleDescription string                 `json:"rule_description"`
	RuleCondition   map[string]interface{} `json:"rule_condition"`
	RuleThreshold   map[string]interface{} `json:"rule_threshold,omitempty"`
	ValidationQuery string                 `json:"validation_query,omitempty"`
	IsActive        bool                   `json:"is_active"`
	Priority        int                    `json:"priority"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// GenerateRulesFromRegulatorySpec generates break detection rules from regulatory specifications
func (rgs *RuleGeneratorService) GenerateRulesFromRegulatorySpec(ctx context.Context, regulatorySpec string, systemName SystemName, detectionType DetectionType) ([]*DetectionRule, error) {
	if rgs.deepResearchClient == nil {
		return nil, fmt.Errorf("Deep Research client not initialized")
	}

	// Build query for rule generation
	query := rgs.buildRuleGenerationQuery(regulatorySpec, systemName, detectionType)

	context := map[string]interface{}{
		"regulatory_spec": regulatorySpec,
		"system_name":     string(systemName),
		"detection_type":  string(detectionType),
	}

	req := &research.ResearchRequest{
		Query:   query,
		Context: context,
		Tools:   []string{"regulatory_parser", "rule_extraction", "catalog_search"},
	}

	if rgs.logger != nil {
		rgs.logger.Printf("Generating rules from regulatory spec for system: %s, type: %s", systemName, detectionType)
	}

	report, err := rgs.deepResearchClient.Research(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to generate rules: %w", err)
	}

	if report.Status == "error" {
		return nil, fmt.Errorf("Deep Research returned error: %s", report.Error)
	}

	// Extract rules from report
	rules := rgs.extractRulesFromReport(report, systemName, detectionType, "regulatory_spec")

	// Store rules in database
	for _, rule := range rules {
		if err := rgs.storeRule(ctx, rule); err != nil {
			if rgs.logger != nil {
				rgs.logger.Printf("Warning: Failed to store rule %s: %v", rule.RuleID, err)
			}
		}
	}

	if rgs.logger != nil {
		rgs.logger.Printf("Generated %d rules from regulatory spec", len(rules))
	}

	return rules, nil
}

// buildRuleGenerationQuery builds a query for rule generation from regulatory specs
func (rgs *RuleGeneratorService) buildRuleGenerationQuery(regulatorySpec string, systemName SystemName, detectionType DetectionType) string {
	var queryBuilder strings.Builder

	queryBuilder.WriteString(fmt.Sprintf("Extract break detection rules from the following regulatory specification for %s system (%s detection type).\n\n", systemName, detectionType))
	queryBuilder.WriteString("Regulatory Specification:\n")
	queryBuilder.WriteString(regulatorySpec)
	queryBuilder.WriteString("\n\n")

	queryBuilder.WriteString("Extract the following:\n")
	queryBuilder.WriteString("1. Break detection conditions (what constitutes a break)\n")
	queryBuilder.WriteString("2. Threshold values (tolerances, limits)\n")
	queryBuilder.WriteString("3. Validation rules (what must be true/false)\n")
	queryBuilder.WriteString("4. Calculation rules (how to compute values for comparison)\n")
	queryBuilder.WriteString("5. Reconciliation requirements\n")
	queryBuilder.WriteString("6. Compliance checks\n\n")

	queryBuilder.WriteString("For each rule, provide:\n")
	queryBuilder.WriteString("- Rule name (descriptive)\n")
	queryBuilder.WriteString("- Rule description (what it checks)\n")
	queryBuilder.WriteString("- Rule condition (logical expression)\n")
	queryBuilder.WriteString("- Threshold values (if applicable)\n")
	queryBuilder.WriteString("- Validation query (SQL or query expression, if applicable)\n")
	queryBuilder.WriteString("- Priority (1-10, higher = more critical)\n")

	return queryBuilder.String()
}

// extractRulesFromReport extracts rules from Deep Research report
func (rgs *RuleGeneratorService) extractRulesFromReport(report *research.ResearchReport, systemName SystemName, detectionType DetectionType, source string) []*DetectionRule {
	var rules []*DetectionRule

	if report.Report == nil {
		return rules
	}

	// Parse rules from report sections
	ruleIDCounter := 1
	for _, section := range report.Report.Sections {
		sectionRules := rgs.parseRulesFromSection(section, systemName, detectionType, source, &ruleIDCounter)
		rules = append(rules, sectionRules...)
	}

	// If no rules found in sections, try to parse from summary
	if len(rules) == 0 {
		rules = rgs.parseRulesFromText(report.Report.Summary, systemName, detectionType, source, &ruleIDCounter)
	}

	return rules
}

// parseRulesFromSection parses rules from a report section
func (rgs *RuleGeneratorService) parseRulesFromSection(section research.ReportSection, systemName SystemName, detectionType DetectionType, source string, ruleIDCounter *int) []*DetectionRule {
	var rules []*DetectionRule

	// Look for rule-like content
	title := strings.ToLower(section.Title)
	if strings.Contains(title, "rule") || 
	   strings.Contains(title, "validation") ||
	   strings.Contains(title, "check") ||
	   strings.Contains(title, "requirement") {
		
		// Parse rules from content
		contentRules := rgs.parseRulesFromText(section.Content, systemName, detectionType, source, ruleIDCounter)
		rules = append(rules, contentRules...)
	}

	return rules
}

// parseRulesFromText parses rules from text content
func (rgs *RuleGeneratorService) parseRulesFromText(text string, systemName SystemName, detectionType DetectionType, source string, ruleIDCounter *int) []*DetectionRule {
	var rules []*DetectionRule

	// Split by lines and look for rule patterns
	lines := strings.Split(text, "\n")
	
	var currentRule *DetectionRule
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Check if this line starts a new rule
		if strings.HasPrefix(strings.ToLower(line), "rule") ||
		   strings.HasPrefix(strings.ToLower(line), "validation") ||
		   strings.HasPrefix(strings.ToLower(line), "check") {
			// Save previous rule if exists
			if currentRule != nil {
				rules = append(rules, currentRule)
			}
			
			// Start new rule
			ruleID := fmt.Sprintf("rule-%s-%s-%d", systemName, detectionType, *ruleIDCounter)
			*ruleIDCounter++
			
			currentRule = &DetectionRule{
				RuleID:        ruleID,
				SystemName:    systemName,
				DetectionType: detectionType,
				RuleType:      "auto_generated",
				RuleSource:    source,
				RuleName:      line,
				IsActive:      true,
				Priority:      5, // Default priority
				CreatedAt:     time.Now(),
				UpdatedAt:     time.Now(),
			}
		} else if currentRule != nil {
			// Add content to current rule
			lowerLine := strings.ToLower(line)
			if strings.Contains(lowerLine, "description") || strings.Contains(lowerLine, "desc") {
				currentRule.RuleDescription = strings.TrimSpace(strings.Split(line, ":")[1])
			} else if strings.Contains(lowerLine, "condition") {
				// Parse condition
				conditionStr := strings.TrimSpace(strings.Split(line, ":")[1])
				currentRule.RuleCondition = map[string]interface{}{
					"expression": conditionStr,
				}
			} else if strings.Contains(lowerLine, "threshold") {
				// Parse threshold
				thresholdStr := strings.TrimSpace(strings.Split(line, ":")[1])
				currentRule.RuleThreshold = map[string]interface{}{
					"value": thresholdStr,
				}
			} else if strings.Contains(lowerLine, "priority") {
				// Parse priority
				if parts := strings.Split(line, ":"); len(parts) > 1 {
					var priority int
					if _, err := fmt.Sscanf(strings.TrimSpace(parts[1]), "%d", &priority); err == nil {
						currentRule.Priority = priority
					}
				}
			} else if currentRule.RuleDescription == "" {
				// Use as description if no description yet
				currentRule.RuleDescription = line
			}
		}
	}

	// Add last rule
	if currentRule != nil {
		rules = append(rules, currentRule)
	}

	// If no structured rules found, create a single rule from the text
	if len(rules) == 0 && len(text) > 50 {
		ruleID := fmt.Sprintf("rule-%s-%s-%d", systemName, detectionType, *ruleIDCounter)
		*ruleIDCounter++
		
		rule := &DetectionRule{
			RuleID:          ruleID,
			SystemName:      systemName,
			DetectionType:    detectionType,
			RuleType:        "auto_generated",
			RuleSource:      source,
			RuleName:        fmt.Sprintf("Auto-generated rule for %s", systemName),
			RuleDescription: text,
			RuleCondition: map[string]interface{}{
				"description": "Auto-generated from regulatory specification",
			},
			IsActive:  true,
			Priority:  5,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		rules = append(rules, rule)
	}

	return rules
}

// storeRule stores a rule in the database
func (rgs *RuleGeneratorService) storeRule(ctx context.Context, rule *DetectionRule) error {
	if rgs.db == nil {
		return fmt.Errorf("database not initialized")
	}

	conditionJSON, _ := json.Marshal(rule.RuleCondition)
	thresholdJSON, _ := json.Marshal(rule.RuleThreshold)

	query := `
		INSERT INTO break_detection_rules (
			id, rule_id, system_name, detection_type, rule_type, rule_source,
			rule_name, rule_description, rule_condition, rule_threshold,
			validation_query, is_active, priority, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
		ON CONFLICT (rule_id) DO UPDATE SET
			rule_name = EXCLUDED.rule_name,
			rule_description = EXCLUDED.rule_description,
			rule_condition = EXCLUDED.rule_condition,
			rule_threshold = EXCLUDED.rule_threshold,
			validation_query = EXCLUDED.validation_query,
			is_active = EXCLUDED.is_active,
			priority = EXCLUDED.priority,
			updated_at = EXCLUDED.updated_at
	`

	_, err := rgs.db.ExecContext(ctx, query,
		rule.RuleID, rule.RuleID, string(rule.SystemName), string(rule.DetectionType),
		rule.RuleType, rule.RuleSource, rule.RuleName, rule.RuleDescription,
		conditionJSON, thresholdJSON, rule.ValidationQuery, rule.IsActive,
		rule.Priority, rule.CreatedAt, rule.UpdatedAt,
	)

	return err
}

// GetRulesForSystem retrieves active rules for a system
func (rgs *RuleGeneratorService) GetRulesForSystem(ctx context.Context, systemName SystemName, detectionType DetectionType) ([]*DetectionRule, error) {
	if rgs.db == nil {
		return nil, fmt.Errorf("database not initialized")
	}

	query := `
		SELECT rule_id, system_name, detection_type, rule_type, rule_source,
		       rule_name, rule_description, rule_condition, rule_threshold,
		       validation_query, is_active, priority, created_at, updated_at
		FROM break_detection_rules
		WHERE system_name = $1 AND detection_type = $2 AND is_active = true
		ORDER BY priority DESC, created_at DESC
	`

	rows, err := rgs.db.QueryContext(ctx, query, string(systemName), string(detectionType))
	if err != nil {
		return nil, fmt.Errorf("failed to query rules: %w", err)
	}
	defer rows.Close()

	var rules []*DetectionRule
	for rows.Next() {
		var rule DetectionRule
		var conditionJSON, thresholdJSON []byte

		err := rows.Scan(
			&rule.RuleID, &rule.SystemName, &rule.DetectionType, &rule.RuleType, &rule.RuleSource,
			&rule.RuleName, &rule.RuleDescription, &conditionJSON, &thresholdJSON,
			&rule.ValidationQuery, &rule.IsActive, &rule.Priority, &rule.CreatedAt, &rule.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan rule: %w", err)
		}

		if len(conditionJSON) > 0 {
			if err := json.Unmarshal(conditionJSON, &rule.RuleCondition); err != nil {
				if rgs.logger != nil {
					rgs.logger.Printf("Warning: Failed to unmarshal rule condition: %v", err)
				}
			}
		}

		if len(thresholdJSON) > 0 {
			if err := json.Unmarshal(thresholdJSON, &rule.RuleThreshold); err != nil {
				if rgs.logger != nil {
					rgs.logger.Printf("Warning: Failed to unmarshal rule threshold: %v", err)
				}
			}
		}

		rules = append(rules, &rule)
	}

	return rules, nil
}

