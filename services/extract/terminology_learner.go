package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// TerminologyLearner orchestrates all LNN layers for terminology learning.
type TerminologyLearner struct {
	terminologyLNN *TerminologyLNN
	store          TerminologyStore
	mu             sync.RWMutex
	logger         *log.Logger
}

// NewTerminologyLearner creates a new terminology learner.
func NewTerminologyLearner(store TerminologyStore, logger *log.Logger) *TerminologyLearner {
	return &TerminologyLearner{
		terminologyLNN: NewTerminologyLNN(logger),
		store:          store,
		logger:         logger,
	}
}

// LearnFromExtraction learns terminology from an extraction run.
func (tl *TerminologyLearner) LearnFromExtraction(
	ctx context.Context,
	nodes []Node,
	edges []Edge,
) error {
	tl.mu.Lock()
	defer tl.mu.Unlock()

	tl.logger.Println("Learning terminology from extraction run...")

	// Extract terminology from nodes
	for _, node := range nodes {
		// Learn domain patterns
		if nodeLabel := node.Label; nodeLabel != "" {
			// Try to infer domain from context
			domain, _ := tl.terminologyLNN.InferDomain(ctx, nodeLabel, nodeLabel, node.Props)
			
			// Learn from this example
			if err := tl.terminologyLNN.LearnDomain(ctx, nodeLabel, domain, time.Now()); err != nil {
				tl.logger.Printf("Failed to learn domain: %v", err)
			}

			// Learn role patterns if column node
			if node.Type == "column" {
				role, _ := tl.terminologyLNN.InferRole(ctx, nodeLabel, "", "", node.Props)
				if err := tl.terminologyLNN.LearnRole(ctx, nodeLabel, role, time.Now()); err != nil {
					tl.logger.Printf("Failed to learn role: %v", err)
				}
			}
		}
	}

	// Store learned terminology
	if err := tl.store.StoreTerminology(ctx, nodes, edges, time.Now()); err != nil {
		tl.logger.Printf("Failed to store terminology: %v", err)
	}

	return nil
}

// InferDomain infers domain using LNN.
func (tl *TerminologyLearner) InferDomain(
	ctx context.Context,
	columnName string,
	tableName string,
	context map[string]any,
) (string, float64) {
	tl.mu.RLock()
	defer tl.mu.RUnlock()

	return tl.terminologyLNN.InferDomain(ctx, columnName, tableName, context)
}

// InferRole infers business role using LNN.
func (tl *TerminologyLearner) InferRole(
	ctx context.Context,
	columnName string,
	columnType string,
	tableName string,
	context map[string]any,
) (string, float64) {
	tl.mu.RLock()
	defer tl.mu.RUnlock()

	return tl.terminologyLNN.InferRole(ctx, columnName, columnType, tableName, context)
}

// AnalyzeNamingConvention analyzes naming patterns using LNN.
func (tl *TerminologyLearner) AnalyzeNamingConvention(
	ctx context.Context,
	columnName string,
) []string {
	tl.mu.RLock()
	defer tl.mu.RUnlock()

	return tl.terminologyLNN.AnalyzeNamingConvention(ctx, columnName)
}

// GetLearnedDomains returns list of learned domains.
func (tl *TerminologyLearner) GetLearnedDomains(ctx context.Context) ([]string, error) {
	tl.mu.RLock()
	defer tl.mu.RUnlock()

	domains := []string{}
	for domain := range tl.terminologyLNN.domainLNNs {
		domains = append(domains, domain)
	}

	return domains, nil
}

// GetLearnedRoles returns list of learned roles.
func (tl *TerminologyLearner) GetLearnedRoles(ctx context.Context) ([]string, error) {
	tl.mu.RLock()
	defer tl.mu.RUnlock()

	// Return standard roles (can be extended)
	roles := []string{"identifier", "amount", "date", "status", "name", "email", "phone", "address", "quantity"}
	return roles, nil
}

// GetLearnedPatterns returns list of learned naming patterns.
func (tl *TerminologyLearner) GetLearnedPatterns(ctx context.Context) ([]string, error) {
	tl.mu.RLock()
	defer tl.mu.RUnlock()

	patterns := []string{"snake_case", "camelCase", "PascalCase", "UPPER_SNAKE", "has_id_suffix", "has_id_prefix", "has_date_suffix", "has_ts_suffix"}
	return patterns, nil
}

// EnhanceEmbedding enhances an embedding with terminology knowledge.
func (tl *TerminologyLearner) EnhanceEmbedding(
	ctx context.Context,
	text string,
	baseEmbedding []float32,
	embeddingType string,
) ([]float32, error) {
	tl.mu.RLock()
	defer tl.mu.RUnlock()

	// Generate terminology-enhanced embedding
	textEmbedding := generateTextEmbedding(text)
	
	// Apply universal LNN
	universalOutput := tl.terminologyLNN.universalLNN.Process(ctx, textEmbedding, time.Now())

	// Combine with base embedding
	enhanced := make([]float32, len(baseEmbedding))
	for i := range enhanced {
		if i < len(universalOutput) && i < len(baseEmbedding) {
			enhanced[i] = baseEmbedding[i] + 0.2*universalOutput[i] // Weighted combination
		} else if i < len(baseEmbedding) {
			enhanced[i] = baseEmbedding[i]
		}
	}

	return enhanced, nil
}

// LoadTerminology loads learned terminology from store.
func (tl *TerminologyLearner) LoadTerminology(ctx context.Context) error {
	tl.mu.Lock()
	defer tl.mu.Unlock()

	// Load terminology from store
	terminology, err := tl.store.LoadTerminology(ctx)
	if err != nil {
		return fmt.Errorf("failed to load terminology: %w", err)
	}

	// Apply loaded terminology to LNNs
	for domain, examples := range terminology.Domains {
		for _, example := range examples {
			if err := tl.terminologyLNN.LearnDomain(ctx, example.Text, domain, example.Timestamp); err != nil {
				tl.logger.Printf("Failed to apply loaded domain example: %v", err)
			}
		}
	}

	for role, examples := range terminology.Roles {
		for _, example := range examples {
			if err := tl.terminologyLNN.LearnRole(ctx, example.Text, role, example.Timestamp); err != nil {
				tl.logger.Printf("Failed to apply loaded role example: %v", err)
			}
		}
	}

	return nil
}

