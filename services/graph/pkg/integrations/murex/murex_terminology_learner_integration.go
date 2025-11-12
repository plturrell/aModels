package murex

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"
)

// TerminologyLearnerInterface defines the interface for terminology learning.
// This allows us to integrate with the extract service's TerminologyLearner
// without importing the extract package directly.
type TerminologyLearnerInterface interface {
	LearnFromExtraction(ctx context.Context, nodes []TerminologyNode, edges []TerminologyEdge) error
	LearnDomain(ctx context.Context, text, domain string, timestamp time.Time) error
	LearnRole(ctx context.Context, text, role string, timestamp time.Time) error
	InferDomain(ctx context.Context, columnName, tableName string, context map[string]interface{}) (string, float64)
	InferRole(ctx context.Context, columnName, columnType, tableName string, context map[string]interface{}) (string, float64)
	EnhanceEmbedding(ctx context.Context, text string, baseEmbedding []float32, embeddingType string) ([]float32, error)
}

// TerminologyNode represents a node for terminology learning (compatible with extract service Node).
type TerminologyNode struct {
	ID     string                 `json:"id"`
	Type   string                 `json:"type"`
	Label  string                 `json:"label,omitempty"`
	Props  map[string]interface{} `json:"props,omitempty"`
	Domain string                 `json:"domain,omitempty"` // For terminology learning
}

// TerminologyEdge represents an edge for terminology learning (compatible with extract service Edge).
type TerminologyEdge struct {
	SourceID string                 `json:"source_id"`
	TargetID string                 `json:"target_id"`
	Label    string                 `json:"label,omitempty"`
	Props    map[string]interface{} `json:"props,omitempty"`
}

// MurexTerminologyLearnerIntegration integrates Murex terminology extraction with TerminologyLearner.
type MurexTerminologyLearnerIntegration struct {
	extractor     *MurexTerminologyExtractor
	learner       TerminologyLearnerInterface
	logger        *log.Logger
}

// NewMurexTerminologyLearnerIntegration creates a new integration.
func NewMurexTerminologyLearnerIntegration(
	extractor *MurexTerminologyExtractor,
	learner TerminologyLearnerInterface,
	logger *log.Logger,
) *MurexTerminologyLearnerIntegration {
	return &MurexTerminologyLearnerIntegration{
		extractor: extractor,
		learner:   learner,
		logger:    logger,
	}
}

// TrainFromExtractedTerminology trains the terminology learner from extracted Murex terminology.
func (mti *MurexTerminologyLearnerIntegration) TrainFromExtractedTerminology(ctx context.Context) error {
	if mti.logger != nil {
		mti.logger.Printf("Training terminology learner from Murex extracted terminology")
	}

	terminology := mti.extractor.GetTerminology()
	trainingData := mti.extractor.GetTrainingData()

	// Convert terminology to nodes and edges for LearnFromExtraction
	nodes, edges := mti.convertToTerminologyNodesAndEdges(terminology, trainingData)

	// Learn from extraction format
	if err := mti.learner.LearnFromExtraction(ctx, nodes, edges); err != nil {
		return fmt.Errorf("failed to learn from extraction: %w", err)
	}

	// Also directly learn domains and roles with higher confidence
	now := time.Now()
	for domain, examples := range terminology.Domains {
		for _, example := range examples {
			if err := mti.learner.LearnDomain(ctx, example.Text, domain, example.Timestamp); err != nil {
				if mti.logger != nil {
					mti.logger.Printf("Warning: Failed to learn domain %s for %s: %v", domain, example.Text, err)
				}
			}
		}
	}

	for role, examples := range terminology.Roles {
		for _, example := range examples {
			if err := mti.learner.LearnRole(ctx, example.Text, role, example.Timestamp); err != nil {
				if mti.logger != nil {
					mti.logger.Printf("Warning: Failed to learn role %s for %s: %v", role, example.Text, err)
				}
			}
		}
	}

	if mti.logger != nil {
		mti.logger.Printf("Training completed: %d nodes, %d edges processed", len(nodes), len(edges))
	}

	return nil
}

// convertToTerminologyNodesAndEdges converts extracted terminology to Node/Edge format.
func (mti *MurexTerminologyLearnerIntegration) convertToTerminologyNodesAndEdges(
	terminology *ExtractedTerminology,
	trainingData *TrainingData,
) ([]TerminologyNode, []TerminologyEdge) {
	var nodes []TerminologyNode
	var edges []TerminologyEdge

	// Create nodes for domain terms
	for domain, examples := range terminology.Domains {
		for _, example := range examples {
			node := TerminologyNode{
				ID:     fmt.Sprintf("murex:domain:%s:%s", domain, example.Text),
				Type:   "terminology",
				Label:  example.Text,
				Domain: domain,
				Props: map[string]interface{}{
					"source":      "murex",
					"domain":       domain,
					"confidence":   example.Confidence,
					"source_type":  example.Source,
					"timestamp":    example.Timestamp.Format(time.RFC3339),
				},
			}
			if example.Context != nil {
				for k, v := range example.Context {
					node.Props[k] = v
				}
			}
			nodes = append(nodes, node)
		}
	}

	// Create nodes for role terms
	for role, examples := range terminology.Roles {
		for _, example := range examples {
			// Extract domain from context if available, otherwise use default
			domain := "finance" // Default domain
			if tableName, ok := example.Context["table"].(string); ok {
				// Try to infer domain from table context
				if strings.Contains(strings.ToLower(tableName), "trade") ||
					strings.Contains(strings.ToLower(tableName), "cashflow") ||
					strings.Contains(strings.ToLower(tableName), "position") {
					domain = "finance"
				}
			}
			
			node := TerminologyNode{
				ID:     fmt.Sprintf("murex:role:%s:%s", role, example.Text),
				Type:   "column", // Roles apply to columns
				Label:  example.Text,
				Domain: domain,
				Props: map[string]interface{}{
					"source":      "murex",
					"role":         role,
					"confidence":   example.Confidence,
					"source_type":  example.Source,
					"timestamp":    example.Timestamp.Format(time.RFC3339),
				},
			}
			if example.Context != nil {
				for k, v := range example.Context {
					node.Props[k] = v
				}
			}
			nodes = append(nodes, node)
		}
	}

	// Create nodes for schema examples (tables)
	for _, schemaExample := range trainingData.SchemaExamples {
		// Infer domain from table name or use default
		domain := "finance" // Default domain
		if strings.Contains(strings.ToLower(schemaExample.TableName), "trade") ||
			strings.Contains(strings.ToLower(schemaExample.TableName), "cashflow") ||
			strings.Contains(strings.ToLower(schemaExample.TableName), "position") {
			domain = "finance"
		}
		
		tableNode := TerminologyNode{
			ID:     fmt.Sprintf("murex:table:%s", schemaExample.TableName),
			Type:   "table",
			Label:  schemaExample.TableName,
			Domain: domain,
			Props: map[string]interface{}{
				"source":      "murex",
				"table_name":  schemaExample.TableName,
				"description": schemaExample.Description,
				"source_type": schemaExample.Source,
			},
		}
		nodes = append(nodes, tableNode)

		// Create column nodes and edges
		for _, column := range schemaExample.Columns {
			columnNode := TerminologyNode{
				ID:     fmt.Sprintf("murex:column:%s:%s", schemaExample.TableName, column.Name),
				Type:   "column",
				Label:  column.Name,
				Domain: domain, // Use same domain as table
				Props: map[string]interface{}{
					"source":      "murex",
					"table":       schemaExample.TableName,
					"data_type":   column.Type,
					"nullable":    column.Nullable,
					"description": column.Description,
				},
			}
			if len(column.Examples) > 0 {
				columnNode.Props["example_value"] = column.Examples[0]
			}
			nodes = append(nodes, columnNode)

			// Create edge from table to column
			edge := TerminologyEdge{
				SourceID: tableNode.ID,
				TargetID: columnNode.ID,
				Label:    "HAS_COLUMN",
				Props: map[string]interface{}{
					"source": "murex",
				},
			}
			edges = append(edges, edge)
		}

		// Create edges for foreign keys
		for _, fk := range schemaExample.ForeignKeys {
			edge := TerminologyEdge{
				SourceID: fmt.Sprintf("murex:table:%s", schemaExample.TableName),
				TargetID: fmt.Sprintf("murex:table:%s", fk.ReferencedTable),
				Label:    "REFERENCES",
				Props: map[string]interface{}{
					"source":           "murex",
					"columns":          fk.Columns,
					"referenced_table": fk.ReferencedTable,
					"referenced_cols": fk.ReferencedColumns,
				},
			}
			edges = append(edges, edge)
		}
	}

	// Create nodes for field examples
	for _, fieldExample := range trainingData.FieldExamples {
		node := TerminologyNode{
			ID:     fmt.Sprintf("murex:field:%s", fieldExample.FieldName),
			Type:   "column",
			Label:  fieldExample.FieldName,
			Domain: fieldExample.Domain,
			Props: map[string]interface{}{
				"source":      "murex",
				"field_name":  fieldExample.FieldName,
				"field_type":  fieldExample.FieldType,
				"domain":      fieldExample.Domain,
				"role":        fieldExample.Role,
				"pattern":     fieldExample.Pattern,
				"description": fieldExample.Description,
			},
		}
		if len(fieldExample.Examples) > 0 {
			node.Props["example_values"] = fieldExample.Examples[:min(5, len(fieldExample.Examples))]
		}
		nodes = append(nodes, node)
	}

	// Create edges for relationship examples
	for _, relExample := range trainingData.RelationshipExamples {
		sourceID := fmt.Sprintf("murex:table:%s", relExample.SourceType)
		targetID := fmt.Sprintf("murex:table:%s", relExample.TargetType)
		
		edge := TerminologyEdge{
			SourceID: sourceID,
			TargetID: targetID,
			Label:    relExample.Relationship,
			Props: map[string]interface{}{
				"source":      "murex",
				"description": relExample.Description,
			},
		}
		edges = append(edges, edge)
	}

	return nodes, edges
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TrainFromSchemaExamples trains the learner from schema examples.
func (mti *MurexTerminologyLearnerIntegration) TrainFromSchemaExamples(ctx context.Context) error {
	trainingData := mti.extractor.GetTrainingData()
	now := time.Now()

	for _, schemaExample := range trainingData.SchemaExamples {
		// Learn table name as domain term
		if err := mti.learner.LearnDomain(ctx, schemaExample.TableName, "finance", now); err != nil {
			if mti.logger != nil {
				mti.logger.Printf("Warning: Failed to learn table domain: %v", err)
			}
		}

		// Learn column roles
		for _, column := range schemaExample.Columns {
			// Infer role from column name and type
			role, _ := mti.learner.InferRole(ctx, column.Name, column.Type, schemaExample.TableName, map[string]interface{}{
				"source": "murex",
				"table":  schemaExample.TableName,
			})

			// Learn the role
			if err := mti.learner.LearnRole(ctx, column.Name, role, now); err != nil {
				if mti.logger != nil {
					mti.logger.Printf("Warning: Failed to learn column role: %v", err)
				}
			}
		}
	}

	return nil
}

// ExportTrainingData exports training data in a format suitable for ML model training.
func (mti *MurexTerminologyLearnerIntegration) ExportTrainingData() map[string]interface{} {
	terminology := mti.extractor.GetTerminology()
	trainingData := mti.extractor.GetTrainingData()

	return map[string]interface{}{
		"terminology": map[string]interface{}{
			"domains":  terminology.Domains,
			"roles":    terminology.Roles,
			"patterns": terminology.NamingPatterns,
		},
		"training_data": map[string]interface{}{
			"schema_examples":      trainingData.SchemaExamples,
			"field_examples":       trainingData.FieldExamples,
			"relationship_examples": trainingData.RelationshipExamples,
			"value_patterns":       trainingData.ValuePatterns,
		},
		"metadata": map[string]interface{}{
			"source":      "murex",
			"extracted_at": time.Now().Format(time.RFC3339),
		},
	}
}

