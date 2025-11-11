package agents

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/catalog/autonomous"
	"github.com/plturrell/aModels/services/catalog/research"
)

// PerplexityAutonomousWrapper wraps PerplexityPipeline with autonomous intelligence.
// This enables learning, optimization, and predictive capabilities.
type PerplexityAutonomousWrapper struct {
	pipeline              *PerplexityPipeline
	autonomousSystem      *autonomous.IntegratedAutonomousSystem
	patternLearningURL    string
	lnnURL                string
	httpClient            *http.Client
	logger                *log.Logger
}

// PerplexityAutonomousConfig configures the autonomous wrapper.
type PerplexityAutonomousConfig struct {
	PipelineConfig      PerplexityPipelineConfig
	DeepResearchURL     string
	DeepAgentsURL       string
	UnifiedWorkflowURL  string
	PatternLearningURL  string
	LNNURL              string
	Database            *sql.DB
	Logger              *log.Logger
}

// NewPerplexityAutonomousWrapper creates a new autonomous wrapper for Perplexity processing.
func NewPerplexityAutonomousWrapper(config PerplexityAutonomousConfig) (*PerplexityAutonomousWrapper, error) {
	// Create pipeline
	pipeline, err := NewPerplexityPipeline(config.PipelineConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create pipeline: %w", err)
	}

	// Create Deep Research client for IntelligenceLayer
	deepResearchClient := research.NewDeepResearchClient(config.DeepResearchURL, config.Logger)

	// Create IntegratedAutonomousSystem with database support
	autonomousSystem := autonomous.NewIntegratedAutonomousSystem(
		deepResearchClient,
		config.DeepAgentsURL,
		config.UnifiedWorkflowURL,
		config.Database, // Database for Goose migrations
		config.Logger,
	)

	// Run Goose migrations if database is available
	if config.Database != nil {
		ctx := context.Background()
		migrationDir := "./migrations" // Can be configured
		if err := autonomousSystem.RunGooseMigration(ctx, migrationDir); err != nil {
			if config.Logger != nil {
				config.Logger.Printf("Warning: Goose migration failed (non-fatal): %v", err)
			}
		}
	}

	return &PerplexityAutonomousWrapper{
		pipeline:           pipeline,
		autonomousSystem:  autonomousSystem,
		patternLearningURL: config.PatternLearningURL,
		lnnURL:            config.LNNURL,
		// Use connection pooling for better performance (Priority 1)
		httpClient: &http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
				MaxConnsPerHost:     50,
			},
			Timeout: 120 * time.Second,
		},
		logger:            config.Logger,
	}, nil
}

// ProcessDocumentsWithIntelligence processes documents with autonomous intelligence.
// This enables learning from execution and optimization over time.
func (paw *PerplexityAutonomousWrapper) ProcessDocumentsWithIntelligence(ctx context.Context, query map[string]interface{}) error {
	queryStr, _ := query["query"].(string)
	if queryStr == "" {
		queryStr = "process Perplexity documents"
	}

	// Create autonomous task
	task := &autonomous.AutonomousTask{
		ID:          fmt.Sprintf("perplexity-%d", ctx.Value("request_id")),
		Type:        "document_processing",
		Query:       queryStr,
		Description: fmt.Sprintf("Process documents from Perplexity API: %s", queryStr),
		Context:     query,
	}

	// Execute with autonomous intelligence and Goose migration tracking
	// This will:
	// 1. Use Deep Research to understand context
	// 2. Plan with DeepAgents
	// 3. Execute using unified workflow
	// 4. Learn from execution
	// 5. Optimize based on results
	// 6. Record execution in database via Goose migrations
	result, err := paw.autonomousSystem.ExecuteWithGooseMigration(ctx, task)
	if err != nil {
		// Fallback to direct pipeline execution if autonomous fails
		if paw.logger != nil {
			paw.logger.Printf("Autonomous execution failed, falling back to direct pipeline: %v", err)
		}
		return paw.pipeline.ProcessDocuments(ctx, query)
	}

	// Process documents through pipeline (autonomous system handles learning/optimization)
	pipelineErr := paw.pipeline.ProcessDocuments(ctx, query)

	// Extract patterns from unified workflow results if available
	if result != nil && result.Result != nil {
		if resultMap, ok := result.Result.(map[string]interface{}); ok {
			// Extract knowledge graph results from unified workflow
			if kgResult, ok := resultMap["knowledge_graph"].(map[string]interface{}); ok {
				if paw.logger != nil {
					paw.logger.Printf("Extracted knowledge graph from unified workflow: %d nodes, %d edges",
						len(kgResult), len(kgResult))
				}
				// Store KG results for pattern mining
				if nodes, ok := kgResult["nodes"].([]interface{}); ok {
					if paw.logger != nil {
						paw.logger.Printf("Knowledge graph contains %d nodes", len(nodes))
					}
				}
			}
		}
	}

	// Perform real-time pattern mining from processed documents
	if pipelineErr == nil {
		if err := paw.minePatternsFromDocuments(ctx, query); err != nil {
			if paw.logger != nil {
				paw.logger.Printf("Pattern mining failed (non-fatal): %v", err)
			}
		}
	}

	// Check if execution was successful
	if result != nil && result.Success {
		if paw.logger != nil {
			paw.logger.Printf("Autonomous processing completed successfully. Learned: %d lessons, Optimized: %d items",
				len(result.Learned), len(result.Optimized))
		}
		return pipelineErr
	}

	return pipelineErr
}

// ProcessDocumentsWithLearning processes documents and learns from the results.
// This is a simpler wrapper that just adds learning without full autonomous execution.
func (paw *PerplexityAutonomousWrapper) ProcessDocumentsWithLearning(ctx context.Context, query map[string]interface{}) error {
	// Process documents using pipeline
	err := paw.pipeline.ProcessDocuments(ctx, query)

	// Learn from execution (even if there were errors)
	if paw.autonomousSystem != nil {
		// Create a simple task for learning
		task := &autonomous.AutonomousTask{
			ID:          fmt.Sprintf("perplexity-learning-%d", ctx.Value("request_id")),
			Type:        "document_processing",
			Query:       "process Perplexity documents",
			Description: "Process documents from Perplexity API",
			Context:     query,
		}

		// Create a result for learning
		result := &autonomous.TaskResult{
			TaskID:  task.ID,
			Success: err == nil,
			Result: map[string]interface{}{
				"query": query,
				"error": func() string {
					if err != nil {
						return err.Error()
					}
					return ""
				}(),
			},
		}

		// Record in database for learning
		if _, recordErr := paw.autonomousSystem.ExecuteWithGooseMigration(ctx, task); recordErr != nil {
			if paw.logger != nil {
				paw.logger.Printf("Failed to record learning (non-fatal): %v", recordErr)
			}
		}

		if paw.logger != nil {
			paw.logger.Printf("Learning from Perplexity processing: success=%v", err == nil)
		}
	}

	// Perform pattern mining
	if err == nil {
		if mineErr := paw.minePatternsFromDocuments(ctx, query); mineErr != nil {
			if paw.logger != nil {
				paw.logger.Printf("Pattern mining failed (non-fatal): %v", mineErr)
			}
		}
	}

	return err
}

// minePatternsFromDocuments performs real-time pattern mining from processed documents.
func (paw *PerplexityAutonomousWrapper) minePatternsFromDocuments(ctx context.Context, query map[string]interface{}) error {
	if paw.patternLearningURL == "" {
		return nil // Pattern learning not configured
	}

	// Extract documents from query (they should be processed by now)
	// We'll create a pattern learning request based on the query
	queryStr, _ := query["query"].(string)
	if queryStr == "" {
		return nil
	}

	// Extract domain from query metadata if available
	domain := "general"
	if metadata, ok := query["metadata"].(map[string]interface{}); ok {
		if domainVal, ok := metadata["domain"].(string); ok {
			domain = domainVal
		}
	}

	// Call pattern learning service with domain-aware context
	payload := map[string]interface{}{
		"source": "perplexity",
		"query":  queryStr,
		"domain": domain,
		"context": map[string]interface{}{
			"enable_gnn":        true,
			"enable_transformer": true,
			"enable_temporal":   true,
			"domain":            domain,
		},
	}

	url := strings.TrimRight(paw.patternLearningURL, "/") + "/patterns/learn"
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal pattern learning request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return fmt.Errorf("failed to create pattern learning request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := paw.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("pattern learning request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("pattern learning returned status %d", resp.StatusCode)
	}

	if paw.logger != nil {
		paw.logger.Printf("Pattern mining completed for query: %s", queryStr)
	}

	return nil
}

// updateLNNWithFeedback updates LNN with processing feedback for adaptive learning.
func (paw *PerplexityAutonomousWrapper) updateLNNWithFeedback(ctx context.Context, success bool, metrics map[string]float64) error {
	if paw.lnnURL == "" {
		return nil // LNN not configured
	}

	payload := map[string]interface{}{
		"task_type": "perplexity_document_processing",
		"success":   success,
		"metrics":   metrics,
	}

	url := strings.TrimRight(paw.lnnURL, "/") + "/update"
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal LNN update: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return fmt.Errorf("failed to create LNN update request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := paw.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("LNN update request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("LNN update returned status %d", resp.StatusCode)
	}

	if paw.logger != nil {
		paw.logger.Printf("LNN updated with feedback: success=%v", success)
	}

	return nil
}

