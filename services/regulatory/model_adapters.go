package regulatory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// ModelAdapter defines the interface for different AI model integrations.
type ModelAdapter interface {
	Query(ctx context.Context, request ModelQueryRequest) (*ModelQueryResponse, error)
	GetCapabilities() []string
	GetModelType() string
}

// ModelQueryRequest represents a generic query request to any model.
type ModelQueryRequest struct {
	QueryType   string                 `json:"query_type"` // "compliance", "lineage", "structural", "research"
	Question    string                 `json:"question"`
	Context     map[string]interface{} `json:"context,omitempty"`
	PrincipleID string                 `json:"principle_id,omitempty"`
	GraphData   *GraphContextData      `json:"graph_data,omitempty"`
}

// ModelQueryResponse represents the response from a model.
type ModelQueryResponse struct {
	Answer      string                 `json:"answer"`
	Confidence  float64                `json:"confidence"`
	Sources     []string               `json:"sources"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	ModelType   string                 `json:"model_type"`
	ProcessTime time.Duration          `json:"process_time"`
}

// GraphContextData contains graph data for model queries.
type GraphContextData struct {
	Nodes []map[string]interface{} `json:"nodes"`
	Edges []map[string]interface{} `json:"edges"`
	Facts []map[string]interface{} `json:"facts"`
}

// ==================== GNN Adapter ====================

// GNNAdapter integrates Graph Neural Network capabilities for structural compliance analysis.
type GNNAdapter struct {
	trainingServiceURL string
	httpClient         *http.Client
	logger             *log.Logger
}

// NewGNNAdapter creates a new GNN adapter.
func NewGNNAdapter(trainingServiceURL string, logger *log.Logger) *GNNAdapter {
	return &GNNAdapter{
		trainingServiceURL: trainingServiceURL,
		httpClient: &http.Client{
			Timeout: 90 * time.Second,
		},
		logger: logger,
	}
}

// Query executes a GNN-based structural analysis query.
func (a *GNNAdapter) Query(ctx context.Context, request ModelQueryRequest) (*ModelQueryResponse, error) {
	startTime := time.Now()
	
	// Prepare GNN query based on request type
	gnnRequest := map[string]interface{}{
		"query_type": a.mapToGNNQueryType(request.QueryType),
		"question":   request.Question,
	}
	
	// Include graph data if available
	if request.GraphData != nil {
		gnnRequest["nodes"] = request.GraphData.Nodes
		gnnRequest["edges"] = request.GraphData.Edges
	}
	
	// Add BCBS239-specific parameters
	if request.PrincipleID != "" {
		gnnRequest["params"] = map[string]interface{}{
			"principle_id": request.PrincipleID,
			"domain":       "regulatory_compliance",
			"framework":    "BCBS239",
		}
	}
	
	// Call GNN service
	result, err := a.callGNNService(ctx, gnnRequest)
	if err != nil {
		return nil, fmt.Errorf("GNN query failed: %w", err)
	}
	
	// Parse and return response
	response := &ModelQueryResponse{
		Answer:      a.extractAnswer(result),
		Confidence:  a.extractConfidence(result),
		Sources:     a.extractSources(result),
		Metadata:    result,
		ModelType:   "GNN",
		ProcessTime: time.Since(startTime),
	}
	
	if a.logger != nil {
		a.logger.Printf("GNN query completed in %v (confidence: %.2f)", response.ProcessTime, response.Confidence)
	}
	
	return response, nil
}

// GetCapabilities returns GNN adapter capabilities.
func (a *GNNAdapter) GetCapabilities() []string {
	return []string{
		"structural_analysis",
		"pattern_recognition",
		"anomaly_detection",
		"link_prediction",
		"node_classification",
		"graph_embeddings",
		"similarity_analysis",
	}
}

// GetModelType returns the model type.
func (a *GNNAdapter) GetModelType() string {
	return "GNN"
}

// callGNNService makes HTTP request to GNN training service.
func (a *GNNAdapter) callGNNService(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	url := a.trainingServiceURL + "/api/gnn/query"
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("GNN service returned %d: %s", resp.StatusCode, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	return result, nil
}

// mapToGNNQueryType maps generic query types to GNN-specific types.
func (a *GNNAdapter) mapToGNNQueryType(queryType string) string {
	mapping := map[string]string{
		"compliance":   "structural-insights",
		"lineage":      "embeddings",
		"structural":   "structural-insights",
		"similarity":   "embeddings",
		"prediction":   "predict-links",
		"classification": "classify",
	}
	
	if gnnType, ok := mapping[queryType]; ok {
		return gnnType
	}
	return "structural-insights"
}

// extractAnswer extracts the answer from GNN result.
func (a *GNNAdapter) extractAnswer(result map[string]interface{}) string {
	if insights, ok := result["insights"].([]interface{}); ok && len(insights) > 0 {
		if insight, ok := insights[0].(map[string]interface{}); ok {
			if desc, ok := insight["description"].(string); ok {
				return desc
			}
		}
	}
	
	if answer, ok := result["answer"].(string); ok {
		return answer
	}
	
	return "GNN structural analysis completed"
}

// extractConfidence extracts confidence score from GNN result.
func (a *GNNAdapter) extractConfidence(result map[string]interface{}) float64 {
	if conf, ok := result["confidence"].(float64); ok {
		return conf
	}
	if score, ok := result["score"].(float64); ok {
		return score
	}
	return 0.75 // Default confidence for GNN structural analysis
}

// extractSources extracts source references from GNN result.
func (a *GNNAdapter) extractSources(result map[string]interface{}) []string {
	sources := []string{"GNN:StructuralAnalysis"}
	
	if nodes, ok := result["nodes"].([]interface{}); ok {
		sources = append(sources, fmt.Sprintf("GNN:Nodes:%d", len(nodes)))
	}
	if edges, ok := result["edges"].([]interface{}); ok {
		sources = append(sources, fmt.Sprintf("GNN:Edges:%d", len(edges)))
	}
	
	return sources
}

// ==================== Goose Adapter ====================

// GooseAdapter integrates Goose AI agent for autonomous compliance task execution.
type GooseAdapter struct {
	gooseServerURL string
	httpClient     *http.Client
	logger         *log.Logger
}

// NewGooseAdapter creates a new Goose adapter.
func NewGooseAdapter(gooseServerURL string, logger *log.Logger) *GooseAdapter {
	return &GooseAdapter{
		gooseServerURL: gooseServerURL,
		httpClient: &http.Client{
			Timeout: 120 * time.Second, // Goose can take longer for complex tasks
		},
		logger: logger,
	}
}

// Query executes a Goose agent task for compliance.
func (a *GooseAdapter) Query(ctx context.Context, request ModelQueryRequest) (*ModelQueryResponse, error) {
	startTime := time.Now()
	
	// Create Goose task request
	gooseRequest := map[string]interface{}{
		"task":    request.Question,
		"context": a.buildGooseContext(request),
		"options": map[string]interface{}{
			"autonomous": true,
			"max_steps":  10,
			"domain":     "regulatory_compliance",
		},
	}
	
	// Execute Goose task
	result, err := a.callGooseAgent(ctx, gooseRequest)
	if err != nil {
		return nil, fmt.Errorf("Goose agent failed: %w", err)
	}
	
	response := &ModelQueryResponse{
		Answer:      a.extractGooseAnswer(result),
		Confidence:  a.extractGooseConfidence(result),
		Sources:     a.extractGooseSources(result),
		Metadata:    result,
		ModelType:   "Goose",
		ProcessTime: time.Since(startTime),
	}
	
	if a.logger != nil {
		a.logger.Printf("Goose task completed in %v", response.ProcessTime)
	}
	
	return response, nil
}

// GetCapabilities returns Goose adapter capabilities.
func (a *GooseAdapter) GetCapabilities() []string {
	return []string{
		"autonomous_task_execution",
		"code_generation",
		"workflow_orchestration",
		"multi_step_reasoning",
		"tool_integration",
		"mcp_server_interaction",
		"documentation_generation",
	}
}

// GetModelType returns the model type.
func (a *GooseAdapter) GetModelType() string {
	return "Goose"
}

// callGooseAgent makes HTTP request to Goose server.
func (a *GooseAdapter) callGooseAgent(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	url := a.gooseServerURL + "/api/v1/task"
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Goose server returned %d: %s", resp.StatusCode, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	return result, nil
}

// buildGooseContext constructs context for Goose agent.
func (a *GooseAdapter) buildGooseContext(request ModelQueryRequest) map[string]interface{} {
	context := make(map[string]interface{})
	
	context["framework"] = "BCBS239"
	context["principle_id"] = request.PrincipleID
	context["query_type"] = request.QueryType
	
	// Add graph context if available
	if request.GraphData != nil {
		context["graph_nodes"] = len(request.GraphData.Nodes)
		context["graph_edges"] = len(request.GraphData.Edges)
		context["graph_facts"] = request.GraphData.Facts
	}
	
	// Merge additional context
	for k, v := range request.Context {
		context[k] = v
	}
	
	return context
}

// extractGooseAnswer extracts answer from Goose result.
func (a *GooseAdapter) extractGooseAnswer(result map[string]interface{}) string {
	if output, ok := result["output"].(string); ok {
		return output
	}
	if result, ok := result["result"].(string); ok {
		return result
	}
	return "Goose task executed successfully"
}

// extractGooseConfidence extracts confidence from Goose result.
func (a *GooseAdapter) extractGooseConfidence(result map[string]interface{}) float64 {
	if status, ok := result["status"].(string); ok && status == "completed" {
		return 0.9
	}
	return 0.7
}

// extractGooseSources extracts sources from Goose result.
func (a *GooseAdapter) extractGooseSources(result map[string]interface{}) []string {
	sources := []string{"Goose:AutonomousAgent"}
	
	if steps, ok := result["steps"].([]interface{}); ok {
		sources = append(sources, fmt.Sprintf("Goose:Steps:%d", len(steps)))
	}
	
	return sources
}

// ==================== Deep Research Adapter ====================

// DeepResearchAdapter integrates Deep Research Agent for complex compliance research.
type DeepResearchAdapter struct {
	deepAgentsURL string
	httpClient    *http.Client
	logger        *log.Logger
}

// NewDeepResearchAdapter creates a new Deep Research adapter.
func NewDeepResearchAdapter(deepAgentsURL string, logger *log.Logger) *DeepResearchAdapter {
	return &DeepResearchAdapter{
		deepAgentsURL: deepAgentsURL,
		httpClient: &http.Client{
			Timeout: 180 * time.Second, // Deep research can take longer
		},
		logger: logger,
	}
}

// Query executes a deep research query for compliance.
func (a *DeepResearchAdapter) Query(ctx context.Context, request ModelQueryRequest) (*ModelQueryResponse, error) {
	startTime := time.Now()
	
	// Create deep research request
	researchRequest := map[string]interface{}{
		"query":     request.Question,
		"domain":    "regulatory_compliance",
		"framework": "BCBS239",
		"depth":     "comprehensive", // Can be "quick", "standard", "comprehensive"
		"context":   request.Context,
	}
	
	if request.PrincipleID != "" {
		researchRequest["principle_id"] = request.PrincipleID
	}
	
	// Execute deep research
	result, err := a.callDeepResearch(ctx, researchRequest)
	if err != nil {
		return nil, fmt.Errorf("Deep research failed: %w", err)
	}
	
	response := &ModelQueryResponse{
		Answer:      a.extractResearchAnswer(result),
		Confidence:  a.extractResearchConfidence(result),
		Sources:     a.extractResearchSources(result),
		Metadata:    result,
		ModelType:   "DeepResearch",
		ProcessTime: time.Since(startTime),
	}
	
	if a.logger != nil {
		a.logger.Printf("Deep research completed in %v (sources: %d)", 
			response.ProcessTime, len(response.Sources))
	}
	
	return response, nil
}

// GetCapabilities returns Deep Research adapter capabilities.
func (a *DeepResearchAdapter) GetCapabilities() []string {
	return []string{
		"comprehensive_research",
		"multi_source_analysis",
		"regulatory_document_analysis",
		"compliance_gap_identification",
		"best_practice_recommendations",
		"cross_reference_validation",
		"citation_tracking",
	}
}

// GetModelType returns the model type.
func (a *DeepResearchAdapter) GetModelType() string {
	return "DeepResearch"
}

// callDeepResearch makes HTTP request to Deep Research service.
func (a *DeepResearchAdapter) callDeepResearch(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	url := a.deepAgentsURL + "/api/research"
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Deep research service returned %d: %s", resp.StatusCode, string(body))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	return result, nil
}

// extractResearchAnswer extracts answer from research result.
func (a *DeepResearchAdapter) extractResearchAnswer(result map[string]interface{}) string {
	if summary, ok := result["summary"].(string); ok {
		return summary
	}
	if analysis, ok := result["analysis"].(string); ok {
		return analysis
	}
	return "Deep research analysis completed"
}

// extractResearchConfidence extracts confidence from research result.
func (a *DeepResearchAdapter) extractResearchConfidence(result map[string]interface{}) float64 {
	if conf, ok := result["confidence"].(float64); ok {
		return conf
	}
	
	// High confidence if multiple sources
	if sources, ok := result["sources"].([]interface{}); ok && len(sources) >= 3 {
		return 0.95
	}
	
	return 0.85
}

// extractResearchSources extracts sources from research result.
func (a *DeepResearchAdapter) extractResearchSources(result map[string]interface{}) []string {
	sources := []string{}
	
	if sourcesRaw, ok := result["sources"].([]interface{}); ok {
		for _, s := range sourcesRaw {
			if source, ok := s.(string); ok {
				sources = append(sources, source)
			} else if sourceMap, ok := s.(map[string]interface{}); ok {
				if title, ok := sourceMap["title"].(string); ok {
					sources = append(sources, title)
				}
			}
		}
	}
	
	if len(sources) == 0 {
		sources = append(sources, "DeepResearch:ComplianceAnalysis")
	}
	
	return sources
}
