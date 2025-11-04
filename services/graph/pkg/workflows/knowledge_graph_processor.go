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

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
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
		newState["knowledge_graph"] = map[string]any{
			"nodes":            kgResp.Nodes,
			"edges":            kgResp.Edges,
			"metadata_entropy": kgResp.MetadataEntropy,
			"kl_divergence":    kgResp.KLDivergence,
			"quality":          kgResp.Quality,
			"root_node_id":     kgResp.RootNodeID,
		}
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

// QueryKnowledgeGraphNode returns a node that queries a knowledge graph (placeholder for future Neo4j integration).
func QueryKnowledgeGraphNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		query, ok := state["knowledge_graph_query"].(string)
		if !ok || query == "" {
			log.Println("No knowledge graph query provided; skipping")
			return state, nil
		}

		// TODO: Integrate with Neo4j or other graph database to query knowledge graphs
		// For now, this is a placeholder that demonstrates the concept
		log.Printf("Querying knowledge graph: %s", query)

		// Extract nodes/edges from knowledge_graph state
		nodesIface, _ := state["knowledge_graph_nodes"].([]any)
		edgesIface, _ := state["knowledge_graph_edges"].([]any)

		// Simple query matching (placeholder)
		results := []map[string]any{}
		if query == "root_node" || query == "root" {
			rootID, _ := state["knowledge_graph"].(map[string]any)["root_node_id"].(string)
			if rootID != "" {
				results = append(results, map[string]any{
					"type": "root_node",
					"id":   rootID,
				})
			}
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		newState["knowledge_graph_query_results"] = results

		log.Printf("Knowledge graph query returned %d results", len(results))
		return newState, nil
	})
}

// NewKnowledgeGraphProcessorWorkflow creates a workflow that processes knowledge graphs using LangGraph.
func NewKnowledgeGraphProcessorWorkflow(opts KnowledgeGraphProcessorOptions) (*stategraph.CompiledStateGraph, error) {
	extractServiceURL := opts.ExtractServiceURL
	if extractServiceURL == "" {
		extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
		if extractServiceURL == "" {
			extractServiceURL = "http://extract-service:19080"
		}
	}

	nodes := map[string]stategraph.NodeFunc{
		"process_kg":        ProcessKnowledgeGraphNode(extractServiceURL),
		"analyze_quality":    AnalyzeKnowledgeGraphQualityNode(),
		"query_kg":          QueryKnowledgeGraphNode(),
	}

	edges := []EdgeSpec{
		{From: "process_kg", To: "analyze_quality", Label: "quality_analysis"},
		{From: "analyze_quality", To: "query_kg", Label: "conditional_query"},
	}

	return BuildGraph("process_kg", "query_kg", nodes, edges)
}

