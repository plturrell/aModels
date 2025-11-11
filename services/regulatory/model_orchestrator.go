package regulatory

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

// ModelOrchestrator intelligently routes queries to appropriate models based on task type.
type ModelOrchestrator struct {
	models         []ModelAdapter
	routingRules   []RoutingRule
	fallbackChain  []string // Fallback order: ["GNN", "LocalAI", "Goose"]
	logger         *log.Logger
	mu             sync.RWMutex
	
	// Performance tracking
	modelPerformance map[string]*ModelPerformance
}

// ModelPerformance tracks performance metrics for each model.
type ModelPerformance struct {
	ModelType        string
	TotalQueries     int
	SuccessfulQueries int
	FailedQueries    int
	AvgProcessTime   time.Duration
	AvgConfidence    float64
	LastUsed         time.Time
}

// RoutingRule defines how to route queries to models.
type RoutingRule struct {
	Name         string
	Priority     int
	Condition    func(request ModelQueryRequest) bool
	ModelType    string
	Explanation  string
}

// NewModelOrchestrator creates a new model orchestrator.
func NewModelOrchestrator(logger *log.Logger) *ModelOrchestrator {
	return &ModelOrchestrator{
		models:           []ModelAdapter{},
		routingRules:     defaultRoutingRules(),
		fallbackChain:    []string{"GNN", "DeepResearch", "Goose", "LocalAI"},
		logger:           logger,
		modelPerformance: make(map[string]*ModelPerformance),
	}
}

// RegisterModel registers a model adapter.
func (o *ModelOrchestrator) RegisterModel(model ModelAdapter) {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	o.models = append(o.models, model)
	
	// Initialize performance tracking
	modelType := model.GetModelType()
	if _, exists := o.modelPerformance[modelType]; !exists {
		o.modelPerformance[modelType] = &ModelPerformance{
			ModelType: modelType,
		}
	}
	
	if o.logger != nil {
		o.logger.Printf("Registered model: %s (capabilities: %v)", 
			modelType, model.GetCapabilities())
	}
}

// RouteAndExecute routes a query to the most appropriate model and executes it.
func (o *ModelOrchestrator) RouteAndExecute(
	ctx context.Context,
	request ModelQueryRequest,
) (*ModelQueryResponse, error) {
	// Determine the best model for this request
	selectedModel, reason := o.selectModel(request)
	
	if selectedModel == nil {
		return nil, fmt.Errorf("no suitable model found for query type: %s", request.QueryType)
	}
	
	if o.logger != nil {
		o.logger.Printf("Routing query to %s: %s", selectedModel.GetModelType(), reason)
	}
	
	// Execute the query
	response, err := o.executeWithFallback(ctx, request, selectedModel)
	
	// Track performance
	o.trackPerformance(selectedModel.GetModelType(), response, err)
	
	return response, err
}

// selectModel selects the most appropriate model based on routing rules.
func (o *ModelOrchestrator) selectModel(request ModelQueryRequest) (ModelAdapter, string) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	// Apply routing rules in priority order
	for _, rule := range o.routingRules {
		if rule.Condition(request) {
			// Find model by type
			for _, model := range o.models {
				if model.GetModelType() == rule.ModelType {
					return model, rule.Explanation
				}
			}
		}
	}
	
	// Default: use first available model
	if len(o.models) > 0 {
		return o.models[0], "Default selection (first available model)"
	}
	
	return nil, "No models available"
}

// executeWithFallback executes query with fallback strategy.
func (o *ModelOrchestrator) executeWithFallback(
	ctx context.Context,
	request ModelQueryRequest,
	primaryModel ModelAdapter,
) (*ModelQueryResponse, error) {
	// Try primary model first
	response, err := primaryModel.Query(ctx, request)
	if err == nil && response != nil {
		return response, nil
	}
	
	// Log primary failure
	if o.logger != nil {
		o.logger.Printf("Primary model %s failed: %v, attempting fallback", 
			primaryModel.GetModelType(), err)
	}
	
	// Try fallback chain
	primaryType := primaryModel.GetModelType()
	for _, fallbackType := range o.fallbackChain {
		if fallbackType == primaryType {
			continue // Skip the failed primary model
		}
		
		// Find fallback model
		for _, model := range o.models {
			if model.GetModelType() == fallbackType {
				if o.logger != nil {
					o.logger.Printf("Trying fallback model: %s", fallbackType)
				}
				
				response, err := model.Query(ctx, request)
				if err == nil && response != nil {
					// Add fallback metadata
					if response.Metadata == nil {
						response.Metadata = make(map[string]interface{})
					}
					response.Metadata["fallback_from"] = primaryType
					response.Metadata["fallback_to"] = fallbackType
					return response, nil
				}
			}
		}
	}
	
	return nil, fmt.Errorf("all models failed for query, primary error: %w", err)
}

// trackPerformance tracks model performance metrics.
func (o *ModelOrchestrator) trackPerformance(modelType string, response *ModelQueryResponse, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	perf, exists := o.modelPerformance[modelType]
	if !exists {
		perf = &ModelPerformance{ModelType: modelType}
		o.modelPerformance[modelType] = perf
	}
	
	perf.TotalQueries++
	perf.LastUsed = time.Now()
	
	if err == nil && response != nil {
		perf.SuccessfulQueries++
		
		// Update running averages
		totalSuccess := float64(perf.SuccessfulQueries)
		perf.AvgProcessTime = time.Duration(
			(float64(perf.AvgProcessTime)*( totalSuccess-1) + float64(response.ProcessTime)) / totalSuccess,
		)
		perf.AvgConfidence = (perf.AvgConfidence*(totalSuccess-1) + response.Confidence) / totalSuccess
	} else {
		perf.FailedQueries++
	}
}

// GetPerformanceMetrics returns performance metrics for all models.
func (o *ModelOrchestrator) GetPerformanceMetrics() map[string]*ModelPerformance {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	// Return a copy
	metrics := make(map[string]*ModelPerformance)
	for k, v := range o.modelPerformance {
		metricsCopy := *v
		metrics[k] = &metricsCopy
	}
	return metrics
}

// defaultRoutingRules returns default routing rules for BCBS239 compliance queries.
func defaultRoutingRules() []RoutingRule {
	return []RoutingRule{
		{
			Name:     "Structural Analysis -> GNN",
			Priority: 100,
			Condition: func(req ModelQueryRequest) bool {
				queryLower := strings.ToLower(req.Question)
				return strings.Contains(queryLower, "pattern") ||
					strings.Contains(queryLower, "structure") ||
					strings.Contains(queryLower, "similar") ||
					strings.Contains(queryLower, "anomaly") ||
					strings.Contains(queryLower, "predict") ||
					req.QueryType == "structural" ||
					req.QueryType == "similarity"
			},
			ModelType:   "GNN",
			Explanation: "Structural analysis best handled by Graph Neural Networks",
		},
		{
			Name:     "Comprehensive Research -> DeepResearch",
			Priority: 90,
			Condition: func(req ModelQueryRequest) bool {
				queryLower := strings.ToLower(req.Question)
				return strings.Contains(queryLower, "research") ||
					strings.Contains(queryLower, "comprehensive") ||
					strings.Contains(queryLower, "detailed analysis") ||
					strings.Contains(queryLower, "regulatory document") ||
					strings.Contains(queryLower, "best practice") ||
					req.QueryType == "research"
			},
			ModelType:   "DeepResearch",
			Explanation: "Comprehensive research requiring deep analysis and multi-source validation",
		},
		{
			Name:     "Multi-Step Task -> Goose",
			Priority: 85,
			Condition: func(req ModelQueryRequest) bool {
				queryLower := strings.ToLower(req.Question)
				return strings.Contains(queryLower, "workflow") ||
					strings.Contains(queryLower, "automate") ||
					strings.Contains(queryLower, "generate code") ||
					strings.Contains(queryLower, "create") ||
					strings.Contains(queryLower, "build") ||
					req.QueryType == "workflow" ||
					req.QueryType == "automation"
			},
			ModelType:   "Goose",
			Explanation: "Multi-step autonomous task execution",
		},
		{
			Name:     "Critical Principles -> DeepResearch + GNN",
			Priority: 95,
			Condition: func(req ModelQueryRequest) bool {
				criticalPrinciples := []string{"P3", "P4", "P7", "P12"}
				for _, p := range criticalPrinciples {
					if req.PrincipleID == p {
						return true
					}
				}
				return false
			},
			ModelType:   "DeepResearch",
			Explanation: "Critical BCBS239 principles require deep research validation",
		},
		{
			Name:     "Lineage Analysis -> GNN",
			Priority: 88,
			Condition: func(req ModelQueryRequest) bool {
				queryLower := strings.ToLower(req.Question)
				return strings.Contains(queryLower, "lineage") ||
					strings.Contains(queryLower, "trace") ||
					strings.Contains(queryLower, "dependency") ||
					strings.Contains(queryLower, "impact") ||
					req.QueryType == "lineage"
			},
			ModelType:   "GNN",
			Explanation: "Lineage and dependency analysis using graph embeddings",
		},
		{
			Name:     "Default -> LocalAI",
			Priority: 0,
			Condition: func(req ModelQueryRequest) bool {
				return true // Always matches
			},
			ModelType:   "LocalAI",
			Explanation: "General compliance questions",
		},
	}
}

// HybridQuery executes a query across multiple models and combines results.
func (o *ModelOrchestrator) HybridQuery(
	ctx context.Context,
	request ModelQueryRequest,
	modelTypes []string,
) (*HybridQueryResponse, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	responses := make([]*ModelQueryResponse, 0, len(modelTypes))
	errors := make([]error, 0)
	
	for _, modelType := range modelTypes {
		for _, model := range o.models {
			if model.GetModelType() == modelType {
				resp, err := model.Query(ctx, request)
				if err != nil {
					errors = append(errors, fmt.Errorf("%s: %w", modelType, err))
					continue
				}
				responses = append(responses, resp)
				break
			}
		}
	}
	
	if len(responses) == 0 {
		return nil, fmt.Errorf("all models failed: %v", errors)
	}
	
	// Combine responses
	hybrid := &HybridQueryResponse{
		CombinedAnswer:    o.combineAnswers(responses),
		ModelResponses:    responses,
		AverageConfidence: o.calculateAvgConfidence(responses),
		Sources:           o.combineSources(responses),
		Metadata: map[string]interface{}{
			"models_used": modelTypes,
			"total_models": len(responses),
		},
	}
	
	return hybrid, nil
}

// HybridQueryResponse contains combined results from multiple models.
type HybridQueryResponse struct {
	CombinedAnswer    string
	ModelResponses    []*ModelQueryResponse
	AverageConfidence float64
	Sources           []string
	Metadata          map[string]interface{}
}

// combineAnswers combines answers from multiple models.
func (o *ModelOrchestrator) combineAnswers(responses []*ModelQueryResponse) string {
	if len(responses) == 0 {
		return ""
	}
	
	if len(responses) == 1 {
		return responses[0].Answer
	}
	
	// Combine with weights based on confidence
	combined := "# Multi-Model Compliance Analysis\n\n"
	
	for i, resp := range responses {
		combined += fmt.Sprintf("## Analysis from %s (Confidence: %.2f)\n%s\n\n",
			resp.ModelType, resp.Confidence, resp.Answer)
		
		if i < len(responses)-1 {
			combined += "---\n\n"
		}
	}
	
	return combined
}

// calculateAvgConfidence calculates average confidence across models.
func (o *ModelOrchestrator) calculateAvgConfidence(responses []*ModelQueryResponse) float64 {
	if len(responses) == 0 {
		return 0
	}
	
	total := 0.0
	for _, resp := range responses {
		total += resp.Confidence
	}
	return total / float64(len(responses))
}

// combineSources combines sources from all models.
func (o *ModelOrchestrator) combineSources(responses []*ModelQueryResponse) []string {
	sourceMap := make(map[string]bool)
	
	for _, resp := range responses {
		for _, source := range resp.Sources {
			sourceMap[source] = true
		}
	}
	
	sources := make([]string, 0, len(sourceMap))
	for source := range sourceMap {
		sources = append(sources, source)
	}
	return sources
}
