package agents

import (
	"context"
	"fmt"
	"log"
	"time"
)

// MappingRuleAgent learns and updates mapping rules automatically.
type MappingRuleAgent struct {
	ID            string
	RuleStore     MappingRuleStore
	Learner       RuleLearner
	logger        *log.Logger
	lastUpdate    time.Time
	stats         RuleStats
}

// MappingRuleStore stores and retrieves mapping rules.
type MappingRuleStore interface {
	GetRules(ctx context.Context, sourceType string, version string) (*MappingRules, error)
	SaveRules(ctx context.Context, rules *MappingRules) error
	ListRules(ctx context.Context, sourceType string) ([]*MappingRules, error)
}

// RuleLearner learns mapping rules from patterns.
type RuleLearner interface {
	LearnFromPattern(ctx context.Context, pattern MappingPattern) (*MappingRuleUpdate, error)
	CalculateConfidence(ctx context.Context, rules *MappingRules) (float64, error)
}

// MappingPattern represents a pattern for rule learning.
type MappingPattern struct {
	SourceTable    string
	SourceColumns  []string
	TargetLabel    string
	TargetProperties []string
	SuccessCount   int
	FailureCount   int
	Examples       []MappingExample
}

// MappingExample represents an example mapping.
type MappingExample struct {
	SourceData map[string]interface{}
	TargetNode GraphNode
	Success    bool
	Timestamp  time.Time
}

// MappingRuleUpdate represents an update to mapping rules.
type MappingRuleUpdate struct {
	Rules         *MappingRules
	Confidence    float64
	Reason        string
	Examples      []MappingExample
	Timestamp     time.Time
}

// RuleStats tracks rule learning statistics.
type RuleStats struct {
	TotalUpdates     int
	SuccessfulUpdates int
	FailedUpdates    int
	RulesLearned      int
	AverageConfidence float64
	LastUpdate        time.Time
}

// NewMappingRuleAgent creates a new mapping rule agent.
func NewMappingRuleAgent(
	id string,
	ruleStore MappingRuleStore,
	learner RuleLearner,
	logger *log.Logger,
) *MappingRuleAgent {
	return &MappingRuleAgent{
		ID:        id,
		RuleStore: ruleStore,
		Learner:   learner,
		logger:    logger,
		stats:     RuleStats{},
	}
}

// LearnAndUpdate learns new rules from patterns and updates existing rules.
func (agent *MappingRuleAgent) LearnAndUpdate(ctx context.Context, patterns []MappingPattern) error {
	agent.stats.TotalUpdates++

	if agent.logger != nil {
		agent.logger.Printf("Learning from %d patterns", len(patterns))
	}

	var updates []*MappingRuleUpdate

	// Learn from each pattern
	for _, pattern := range patterns {
		update, err := agent.Learner.LearnFromPattern(ctx, pattern)
		if err != nil {
			agent.logger.Printf("Warning: Failed to learn from pattern %s: %v", pattern.SourceTable, err)
			continue
		}

		updates = append(updates, update)
	}

	// Validate and apply updates
	for _, update := range updates {
		// Calculate confidence
		confidence, err := agent.Learner.CalculateConfidence(ctx, update.Rules)
		if err != nil {
			agent.logger.Printf("Warning: Failed to calculate confidence: %v", err)
			continue
		}
		update.Confidence = confidence

		// Only apply high-confidence updates
		if confidence >= 0.7 {
			if err := agent.RuleStore.SaveRules(ctx, update.Rules); err != nil {
				agent.stats.FailedUpdates++
				agent.logger.Printf("Failed to save rules: %v", err)
				continue
			}

			agent.stats.SuccessfulUpdates++
			agent.stats.RulesLearned++
			agent.lastUpdate = time.Now()

			if agent.logger != nil {
				agent.logger.Printf("Updated rules with confidence %.2f: %s", confidence, update.Reason)
			}
		} else {
			if agent.logger != nil {
				agent.logger.Printf("Skipping low-confidence update (%.2f): %s", confidence, update.Reason)
			}
		}
	}

	// Update statistics
	agent.updateStats(updates)

	return nil
}

// GetMappingRules retrieves mapping rules for a source type.
func (agent *MappingRuleAgent) GetMappingRules(ctx context.Context, sourceType string, version string) (*MappingRules, error) {
	return agent.RuleStore.GetRules(ctx, sourceType, version)
}

// GetStats returns rule learning statistics.
func (agent *MappingRuleAgent) GetStats() RuleStats {
	return agent.stats
}

// updateStats updates statistics from rule updates.
func (agent *MappingRuleAgent) updateStats(updates []*MappingRuleUpdate) {
	if len(updates) == 0 {
		return
	}

	totalConfidence := 0.0
	for _, update := range updates {
		totalConfidence += update.Confidence
	}
	agent.stats.AverageConfidence = totalConfidence / float64(len(updates))
	agent.stats.LastUpdate = time.Now()
}

// DefaultRuleLearner implements RuleLearner with basic learning algorithms.
type DefaultRuleLearner struct {
	logger *log.Logger
}

// NewDefaultRuleLearner creates a new default rule learner.
func NewDefaultRuleLearner(logger *log.Logger) *DefaultRuleLearner {
	return &DefaultRuleLearner{
		logger: logger,
	}
}

// LearnFromPattern learns mapping rules from a pattern.
func (rl *DefaultRuleLearner) LearnFromPattern(ctx context.Context, pattern MappingPattern) (*MappingRuleUpdate, error) {
	if rl.logger != nil {
		rl.logger.Printf("Learning rules from pattern: %s -> %s", pattern.SourceTable, pattern.TargetLabel)
	}

	// Create node mapping
	nodeMapping := NodeMapping{
		SourceTable:    pattern.SourceTable,
		TargetLabel:    pattern.TargetLabel,
		ColumnMappings: []ColumnMapping{},
	}

	// Map columns to properties
	for i, sourceCol := range pattern.SourceColumns {
		targetProp := pattern.TargetLabel
		if i < len(pattern.TargetProperties) {
			targetProp = pattern.TargetProperties[i]
		}

		nodeMapping.ColumnMappings = append(nodeMapping.ColumnMappings, ColumnMapping{
			SourceColumn:   sourceCol,
			TargetProperty: targetProp,
		})
	}

	rules := &MappingRules{
		NodeMappings:   []NodeMapping{nodeMapping},
		EdgeMappings:   []EdgeMapping{},
		Transformations: []Transformation{},
		Version:       "1.0.0",
		Confidence:    rl.calculatePatternConfidence(pattern),
	}

	update := &MappingRuleUpdate{
		Rules:      rules,
		Confidence:  rules.Confidence,
		Reason:     fmt.Sprintf("Learned from pattern: %s (success: %d, failure: %d)", pattern.SourceTable, pattern.SuccessCount, pattern.FailureCount),
		Examples:   pattern.Examples,
		Timestamp:  time.Now(),
	}

	return update, nil
}

// CalculateConfidence calculates confidence score for mapping rules.
func (rl *DefaultRuleLearner) CalculateConfidence(ctx context.Context, rules *MappingRules) (float64, error) {
	// Base confidence from rules
	confidence := rules.Confidence

	// Adjust based on number of mappings (more mappings = higher confidence)
	if len(rules.NodeMappings) > 0 {
		confidence += 0.1
	}
	if len(rules.EdgeMappings) > 0 {
		confidence += 0.1
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence, nil
}

// calculatePatternConfidence calculates confidence from a pattern.
func (rl *DefaultRuleLearner) calculatePatternConfidence(pattern MappingPattern) float64 {
	total := pattern.SuccessCount + pattern.FailureCount
	if total == 0 {
		return 0.5 // Default confidence
	}

	successRate := float64(pattern.SuccessCount) / float64(total)

	// Adjust based on number of examples
	exampleBonus := 0.0
	if len(pattern.Examples) >= 10 {
		exampleBonus = 0.1
	} else if len(pattern.Examples) >= 5 {
		exampleBonus = 0.05
	}

	confidence := successRate + exampleBonus
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

