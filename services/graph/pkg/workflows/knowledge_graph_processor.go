package workflows

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/graph"
	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
	"github.com/plturrell/aModels/services/graph/pkg/models"
)

// KnowledgeGraphProcessorOptions configures the knowledge graph processing workflow.
type KnowledgeGraphProcessorOptions struct {
	ExtractServiceURL string // URL to extract service (e.g., "http://extract-service:19080")
}

// KnowledgeGraphRequest represents a request to process a knowledge graph.
type KnowledgeGraphRequest struct {
	JSONTables          []string           `json:"json_tables"`
	HiveDDLs            []string           `json:"hive_ddls"`
	SqlQueries          []string           `json:"sql_queries"`
	ControlMFiles       []string           `json:"control_m_files"`
	IdealDistribution   map[string]float64 `json:"ideal_distribution,omitempty"`
	ProjectID           string             `json:"project_id,omitempty"`
	SystemID            string             `json:"system_id,omitempty"`
	InformationSystemID string             `json:"information_system_id,omitempty"`
}

// KnowledgeGraphResponse represents the response from knowledge graph processing.
type KnowledgeGraphResponse struct {
	Nodes            []Node   `json:"nodes"`
	Edges            []Edge   `json:"edges"`
	MetadataEntropy  float64  `json:"metadata_entropy"`
	KLDivergence     float64  `json:"kl_divergence"`
	Quality          Quality  `json:"quality"`
	RootNodeID       string   `json:"root_node_id"`
	Warnings         []string `json:"warnings,omitempty"`
}

// Node represents a node in the knowledge graph.
type Node struct {
	ID    string         `json:"id"`
	Type  string         `json:"type"`
	Label string         `json:"label"`
	Props map[string]any `json:"properties,omitempty"`
}

// Edge represents an edge in the knowledge graph.
type Edge struct {
	SourceID string         `json:"source"`
	TargetID string         `json:"target"`
	Label    string         `json:"label"`
	Props    map[string]any `json:"properties,omitempty"`
}

// Quality represents data quality metrics.
type Quality struct {
	Score            float64  `json:"score"`
	Level            string   `json:"level"`
	Issues           []string `json:"issues"`
	Recommendations  []string `json:"recommendations"`
	ProcessingStrategy string `json:"processing_strategy"`
}

var knowledgeGraphHTTPClient = &http.Client{
	Timeout: 60 * time.Second,
}

// ProcessKnowledgeGraphNode returns a node that processes a knowledge graph.
func ProcessKnowledgeGraphNode(extractServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract knowledge graph request from state
		var kgRequest KnowledgeGraphRequest
		
		// Try to get from state directly
		if req, ok := state["knowledge_graph_request"].(map[string]any); ok {
			// Convert map to KnowledgeGraphRequest
			if jsonTables, ok := req["json_tables"].([]any); ok {
				for _, v := range jsonTables {
					if s, ok := v.(string); ok {
						kgRequest.JSONTables = append(kgRequest.JSONTables, s)
					}
				}
			}
			if hiveDDLs, ok := req["hive_ddls"].([]any); ok {
				for _, v := range hiveDDLs {
					if s, ok := v.(string); ok {
						kgRequest.HiveDDLs = append(kgRequest.HiveDDLs, s)
					}
				}
			}
			if sqlQueries, ok := req["sql_queries"].([]any); ok {
				for _, v := range sqlQueries {
					if s, ok := v.(string); ok {
						kgRequest.SqlQueries = append(kgRequest.SqlQueries, s)
					}
				}
			}
			if controlMFiles, ok := req["control_m_files"].([]any); ok {
				for _, v := range controlMFiles {
					if s, ok := v.(string); ok {
						kgRequest.ControlMFiles = append(kgRequest.ControlMFiles, s)
					}
				}
			}
			if projectID, ok := req["project_id"].(string); ok {
				kgRequest.ProjectID = projectID
			}
			if systemID, ok := req["system_id"].(string); ok {
				kgRequest.SystemID = systemID
			}
		} else {
			// Try to construct from individual fields
			if jsonTables, ok := state["json_tables"].([]any); ok {
				for _, v := range jsonTables {
					if s, ok := v.(string); ok {
						kgRequest.JSONTables = append(kgRequest.JSONTables, s)
					}
				}
			}
			if hiveDDLs, ok := state["hive_ddls"].([]any); ok {
				for _, v := range hiveDDLs {
					if s, ok := v.(string); ok {
						kgRequest.HiveDDLs = append(kgRequest.HiveDDLs, s)
					}
				}
			}
			if sqlQueries, ok := state["sql_queries"].([]any); ok {
				for _, v := range sqlQueries {
					if s, ok := v.(string); ok {
						kgRequest.SqlQueries = append(kgRequest.SqlQueries, s)
					}
				}
			}
		}

		if len(kgRequest.JSONTables) == 0 && len(kgRequest.HiveDDLs) == 0 && 
		   len(kgRequest.SqlQueries) == 0 && len(kgRequest.ControlMFiles) == 0 {
			log.Println("No knowledge graph inputs provided; skipping processing")
			return state, nil
		}

		if extractServiceURL == "" || extractServiceURL == "offline" {
			extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
			if extractServiceURL == "" {
				extractServiceURL = "http://extract-service:19080"
			}
		}

		log.Printf("Processing knowledge graph with %d JSON tables, %d DDLs, %d SQL queries, %d Control-M files",
			len(kgRequest.JSONTables), len(kgRequest.HiveDDLs), 
			len(kgRequest.SqlQueries), len(kgRequest.ControlMFiles))

		// Call extract service knowledge graph endpoint
		endpoint := strings.TrimRight(extractServiceURL, "/") + "/knowledge-graph"
		body, err := json.Marshal(kgRequest)
		if err != nil {
			return nil, fmt.Errorf("marshal knowledge graph request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build knowledge graph request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := knowledgeGraphHTTPClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request knowledge graph processing: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			return nil, fmt.Errorf("knowledge graph processing failed with status %s: %s", 
				resp.Status, strings.TrimSpace(string(bodyBytes)))
		}

		var kgResp KnowledgeGraphResponse
		if err := json.NewDecoder(resp.Body).Decode(&kgResp); err != nil {
			return nil, fmt.Errorf("decode knowledge graph response: %w", err)
		}

		// Store results in state
		newState := make(map[string]any, len(state)+5)
		for k, v := range state {
			newState[k] = v
		}
		// Convert to unified GraphData format
		graphData := &models.GraphData{
			Nodes: convertKGResponseNodes(kgResp.Nodes),
			Edges: convertKGResponseEdges(kgResp.Edges),
			Metadata: models.Metadata{
				ProjectID:       kgRequest.ProjectID,
				SystemID:        kgRequest.SystemID,
				RootNodeID:      kgResp.RootNodeID,
				MetadataEntropy: kgResp.MetadataEntropy,
				KLDivergence:    kgResp.KLDivergence,
				Warnings:        kgResp.Warnings,
			},
			Quality: &models.Quality{
				Score:             kgResp.Quality.Score,
				Level:             kgResp.Quality.Level,
				Issues:            kgResp.Quality.Issues,
				Recommendations:   kgResp.Quality.Recommendations,
				ProcessingStrategy: kgResp.Quality.ProcessingStrategy,
			},
		}

		// Validate graph data
		if err := graphData.Validate(); err != nil {
			log.Printf("WARNING: Graph data validation failed: %v", err)
		}

		// Store in unified format
		newState["knowledge_graph"] = graphData.ToNeo4j() // For backward compatibility
		newState["graph_data"] = graphData                 // New unified format
		newState["knowledge_graph_quality"] = kgResp.Quality
		newState["knowledge_graph_nodes"] = kgResp.Nodes
		newState["knowledge_graph_edges"] = kgResp.Edges

		if len(kgResp.Warnings) > 0 {
			newState["warnings"] = kgResp.Warnings
			log.Printf("Knowledge graph processing warnings: %v", kgResp.Warnings)
		}

		log.Printf("Knowledge graph processed: %d nodes, %d edges, quality=%s (%.2f)",
			len(kgResp.Nodes), len(kgResp.Edges), kgResp.Quality.Level, kgResp.Quality.Score)

		return newState, nil
	})
}

// AnalyzeKnowledgeGraphQualityNode returns a node that analyzes knowledge graph quality and decides next steps.
func AnalyzeKnowledgeGraphQualityNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		kgIface, ok := state["knowledge_graph"]
		if !ok {
			return nil, fmt.Errorf("knowledge_graph not found in state")
		}

		kgMap, ok := kgIface.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("knowledge_graph is not a map")
		}

		qualityIface, ok := kgMap["quality"]
		if !ok {
			return nil, fmt.Errorf("quality not found in knowledge_graph")
		}

		qualityMap, ok := qualityIface.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("quality is not a map")
		}

		qualityLevel, _ := qualityMap["level"].(string)
		qualityScore, _ := qualityMap["score"].(float64)
		processingStrategy, _ := qualityMap["processing_strategy"].(string)

		log.Printf("Analyzing knowledge graph quality: level=%s, score=%.2f, strategy=%s",
			qualityLevel, qualityScore, processingStrategy)

		newState := make(map[string]any, len(state)+3)
		for k, v := range state {
			newState[k] = v
		}

		// Decide next steps based on quality
		var shouldProcess bool
		var shouldValidate bool
		var shouldReview bool

		switch qualityLevel {
		case "excellent", "good":
			shouldProcess = true
			shouldValidate = false
			shouldReview = false
		case "fair":
			shouldProcess = true
			shouldValidate = true
			shouldReview = false
		case "poor":
			shouldProcess = true
			shouldValidate = true
			shouldReview = true
		case "critical":
			shouldProcess = false
			shouldValidate = true
			shouldReview = true
		default:
			shouldProcess = true
			shouldValidate = true
			shouldReview = false
		}

		newState["should_process_kg"] = shouldProcess
		newState["should_validate_kg"] = shouldValidate
		newState["should_review_kg"] = shouldReview
		newState["processing_strategy"] = processingStrategy

		if !shouldProcess {
			log.Printf("WARNING: Knowledge graph quality is %s - processing may be skipped", qualityLevel)
		}

		return newState, nil
	})
}

// QueryKnowledgeGraphNode returns a node that queries a knowledge graph using Neo4j Cypher queries.
func QueryKnowledgeGraphNode(extractServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		query, ok := state["knowledge_graph_query"].(string)
		if !ok || query == "" {
			log.Println("No knowledge graph query provided; skipping")
			return state, nil
		}

		// Extract query parameters if provided
		var queryParams map[string]any
		if params, ok := state["knowledge_graph_query_params"].(map[string]any); ok {
			queryParams = params
		} else {
			queryParams = make(map[string]any)
		}

		if extractServiceURL == "" || extractServiceURL == "offline" {
			extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
			if extractServiceURL == "" {
				extractServiceURL = "http://extract-service:19080"
			}
		}

		log.Printf("Querying knowledge graph with Cypher: %s", query)

		// Call extract service Neo4j query endpoint
		endpoint := strings.TrimRight(extractServiceURL, "/") + "/knowledge-graph/query"
		requestBody := map[string]any{
			"query":  query,
			"params": queryParams,
		}

		body, err := json.Marshal(requestBody)
		if err != nil {
			return nil, fmt.Errorf("marshal query request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build query request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := knowledgeGraphHTTPClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request knowledge graph query: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			return nil, fmt.Errorf("knowledge graph query failed with status %s: %s",
				resp.Status, strings.TrimSpace(string(bodyBytes)))
		}

		var queryResult map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&queryResult); err != nil {
			return nil, fmt.Errorf("decode query response: %w", err)
		}

		// Convert to unified GraphData format
		graphData, err := models.FromNeo4j(queryResult)
		if err != nil {
			log.Printf("WARNING: Failed to convert Neo4j result to GraphData: %v", err)
			// Fallback to storing raw result
			graphData = nil
		}

		// Store results in state
		newState := make(map[string]any, len(state)+4)
		for k, v := range state {
			newState[k] = v
		}
		
		// Store in both formats for compatibility
		newState["knowledge_graph_query_results"] = queryResult
		if graphData != nil {
			newState["graph_data"] = graphData
			newState["knowledge_graph_query_results"] = graphData.ToNeo4j()
			log.Printf("Knowledge graph query returned %d nodes, %d edges (converted to GraphData)",
				len(graphData.Nodes), len(graphData.Edges))
		} else {
			// Fallback: extract columns and data if present
			if columns, ok := queryResult["columns"].([]any); ok {
				newState["knowledge_graph_query_columns"] = columns
			}
			if data, ok := queryResult["data"].([]any); ok {
				newState["knowledge_graph_query_results"] = data
				log.Printf("Knowledge graph query returned %d results", len(data))
			}
		}

		return newState, nil
	})
}

// QualityRoutingFunc determines the next node based on knowledge graph quality.
func QualityRoutingFunc(ctx context.Context, value any) ([]string, error) {
	state, ok := value.(map[string]any)
	if !ok {
		return []string{"reject"}, nil // Default to reject if state is invalid
	}

	// Check if processing should continue
	shouldProcess, _ := state["should_process_kg"].(bool)
	shouldValidate, _ := state["should_validate_kg"].(bool)
	shouldReview, _ := state["should_review_kg"].(bool)
	processingStrategy, _ := state["processing_strategy"].(string)

	// Route based on quality and processing strategy
	if !shouldProcess {
		return []string{"reject"}, nil
	}

	if processingStrategy == "skip" {
		return []string{"skip"}, nil
	}

	if shouldReview {
		return []string{"review"}, nil
	}

	if shouldValidate {
		return []string{"validate"}, nil
	}

	// Good quality - proceed to query
	return []string{"query"}, nil
}

// ValidateKnowledgeGraphNode returns a node that validates knowledge graph data.
func ValidateKnowledgeGraphNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		log.Println("Validating knowledge graph data...")
		
		kgIface, ok := state["knowledge_graph"]
		if !ok {
			return nil, fmt.Errorf("knowledge_graph not found in state")
		}

		kgMap, ok := kgIface.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("knowledge_graph is not a map")
		}

		// Validate nodes and edges exist
		nodes, _ := kgMap["nodes"].([]any)
		edges, _ := kgMap["edges"].([]any)

		validationResults := map[string]any{
			"valid":         len(nodes) > 0 && len(edges) > 0,
			"node_count":    len(nodes),
			"edge_count":    len(edges),
			"validated_at":  time.Now().Format(time.RFC3339),
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		newState["validation_results"] = validationResults

		log.Printf("Knowledge graph validation: valid=%v, nodes=%d, edges=%d",
			validationResults["valid"], len(nodes), len(edges))

		return newState, nil
	})
}

// ReviewKnowledgeGraphNode returns a node that flags knowledge graph for human review.
func ReviewKnowledgeGraphNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		log.Println("Flagging knowledge graph for human review...")
		
		kgIface, ok := state["knowledge_graph"]
		if !ok {
			return nil, fmt.Errorf("knowledge_graph not found in state")
		}

		reviewInfo := map[string]any{
			"flagged_for_review": true,
			"reason":            "Quality level requires human review",
			"flagged_at":        time.Now().Format(time.RFC3339),
		}

		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}
		newState["review_info"] = reviewInfo
		newState["knowledge_graph"] = kgIface // Preserve knowledge graph

		log.Println("Knowledge graph flagged for review")

		return newState, nil
	})
}

// RejectKnowledgeGraphNode returns a node that rejects low-quality knowledge graphs.
func RejectKnowledgeGraphNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		log.Println("Rejecting knowledge graph due to poor quality...")
		
		rejectionInfo := map[string]any{
			"rejected":   true,
			"reason":    "Knowledge graph quality is below acceptable threshold",
			"rejected_at": time.Now().Format(time.RFC3339),
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		newState["rejection_info"] = rejectionInfo

		log.Println("Knowledge graph rejected")

		return newState, nil
	})
}

// SkipKnowledgeGraphNode returns a node that skips processing for simplified strategy.
func SkipKnowledgeGraphNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		log.Println("Skipping knowledge graph processing (simplified strategy)...")
		
		skipInfo := map[string]any{
			"skipped":   true,
			"reason":   "Processing strategy is 'simplified' or 'skip'",
			"skipped_at": time.Now().Format(time.RFC3339),
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		newState["skip_info"] = skipInfo

		log.Println("Knowledge graph processing skipped")

		return newState, nil
	})
}

// NewKnowledgeGraphProcessorWorkflow creates a workflow that processes knowledge graphs using LangGraph.
// This workflow uses conditional edges for quality-based routing.
func NewKnowledgeGraphProcessorWorkflow(opts KnowledgeGraphProcessorOptions) (*stategraph.CompiledStateGraph, error) {
	extractServiceURL := opts.ExtractServiceURL
	if extractServiceURL == "" {
		extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
		if extractServiceURL == "" {
			extractServiceURL = "http://extract-service:19080"
		}
	}

	nodes := map[string]stategraph.NodeFunc{
		"process_kg":     ProcessKnowledgeGraphNode(extractServiceURL),
		"analyze_quality": AnalyzeKnowledgeGraphQualityNode(),
		"validate_kg":    ValidateKnowledgeGraphNode(),
		"review_kg":      ReviewKnowledgeGraphNode(),
		"reject_kg":      RejectKnowledgeGraphNode(),
		"skip_kg":        SkipKnowledgeGraphNode(),
		"query_kg":       QueryKnowledgeGraphNode(extractServiceURL),
	}

	edges := []EdgeSpec{
		{From: "process_kg", To: "analyze_quality", Label: "quality_analysis"},
		{From: "validate_kg", To: "query_kg", Label: "validated"},
		{From: "review_kg", To: "query_kg", Label: "reviewed"},
	}

	// Conditional routing based on quality
	conditionalEdges := []ConditionalEdgeSpec{
		{
			Source: "analyze_quality",
			PathFunc: QualityRoutingFunc,
			PathMap: map[string]string{
				"reject":  "reject_kg",
				"skip":    "skip_kg",
				"review":  "review_kg",
				"validate": "validate_kg",
				"query":   "query_kg",
			},
		},
	}

	return BuildGraphWithOptions("process_kg", "query_kg", nodes, edges, conditionalEdges, nil)
}

// convertKGResponseNodes converts KnowledgeGraphResponse nodes to unified Node format.
func convertKGResponseNodes(kgNodes []Node) []models.Node {
	nodes := make([]models.Node, 0, len(kgNodes))
	for _, kgNode := range kgNodes {
		nodes = append(nodes, models.Node{
			ID:         kgNode.ID,
			Type:       kgNode.Type,
			Label:      kgNode.Label,
			Properties: kgNode.Props,
		})
	}
	return nodes
}

// convertKGResponseEdges converts KnowledgeGraphResponse edges to unified Edge format.
func convertKGResponseEdges(kgEdges []Edge) []models.Edge {
	edges := make([]models.Edge, 0, len(kgEdges))
	for _, kgEdge := range kgEdges {
		edges = append(edges, models.Edge{
			SourceID:   kgEdge.SourceID,
			TargetID:   kgEdge.TargetID,
			Label:      kgEdge.Label,
			Properties: kgEdge.Props,
		})
	}
	return edges
}

