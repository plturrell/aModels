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
	"github.com/plturrell/aModels/services/graph/pkg/models"
)

// GNNProcessorOptions configures the GNN query processor.
type GNNProcessorOptions struct {
	TrainingServiceURL string // URL to training service (e.g., "http://training-service:8080")
	ExtractServiceURL  string // URL to extract service for getting graph data
}

// GNNQueryRequest represents a request to query GNN service.
type GNNQueryRequest struct {
	QueryType string                 `json:"query_type"` // "embeddings", "classify", "predict-links", "structural-insights"
	Nodes     []Node                 `json:"nodes,omitempty"`
	Edges     []Edge                 `json:"edges,omitempty"`
	Params    map[string]interface{} `json:"params,omitempty"`
}

// HybridQueryRequest represents a request that can query both KG and GNN.
type HybridQueryRequest struct {
	Query      string                 `json:"query"`
	ProjectID  string                 `json:"project_id,omitempty"`
	SystemID   string                 `json:"system_id,omitempty"`
	QueryKG    bool                    `json:"query_kg,omitempty"`    // Whether to query KG
	QueryGNN   bool                    `json:"query_gnn,omitempty"`   // Whether to query GNN
	GNNType    string                 `json:"gnn_type,omitempty"`     // "embeddings", "classify", "predict-links", "insights"
	Combine    bool                    `json:"combine,omitempty"`      // Whether to combine results
}

var gnnHTTPClient = &http.Client{
	Timeout: 60 * time.Second,
}

// isStructuralQuery determines if a query should go to GNN (structural) or KG (factual).
func isStructuralQuery(query string) bool {
	queryLower := strings.ToLower(query)
	
	// Structural keywords
	structuralKeywords := []string{
		"similar", "pattern", "anomaly", "outlier", "classify", "group",
		"embedding", "representation", "predict", "missing", "suggest",
		"relationship", "link", "connection", "structure", "structural",
		"insight", "analysis", "cluster", "grouping", "type", "domain",
		"quality", "characteristic", "feature", "vector", "distance",
	}
	
	// Factual keywords
	factualKeywords := []string{
		"find", "get", "list", "show", "return", "match", "where",
		"count", "exists", "has", "contains", "specific", "exact",
		"cypher", "query", "table", "column", "system", "project",
	}
	
	structuralScore := 0
	factualScore := 0
	
	for _, keyword := range structuralKeywords {
		if strings.Contains(queryLower, keyword) {
			structuralScore++
		}
	}
	
	for _, keyword := range factualKeywords {
		if strings.Contains(queryLower, keyword) {
			factualScore++
		}
	}
	
	// Check for Cypher syntax (factual)
	if strings.Contains(queryLower, "match") || strings.Contains(queryLower, "return") {
		return false
	}
	
	// Check for explicit structural requests
	if strings.Contains(queryLower, "structural insight") ||
		strings.Contains(queryLower, "graph pattern") ||
		strings.Contains(queryLower, "similar nodes") ||
		strings.Contains(queryLower, "predict relationship") ||
		strings.Contains(queryLower, "classify node") ||
		strings.Contains(queryLower, "anomaly detection") {
		return true
	}
	
	return structuralScore > factualScore
}

// QueryGNNNode returns a node that queries the GNN service.
func QueryGNNNode(opts GNNProcessorOptions) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		trainingServiceURL := opts.TrainingServiceURL
		if trainingServiceURL == "" {
			trainingServiceURL = os.Getenv("TRAINING_SERVICE_URL")
			if trainingServiceURL == "" {
				trainingServiceURL = "http://training-service:8080"
			}
		}
		
		// Extract GNN query request from state
		var gnnRequest GNNQueryRequest
		var graphData *models.GraphData
		
		// Try to get unified GraphData first
		if gd, ok := state["graph_data"].(*models.GraphData); ok {
			graphData = gd
		} else if gdMap, ok := state["graph_data"].(map[string]any); ok {
			// Try to reconstruct from map
			var err error
			graphData, err = models.FromNeo4j(gdMap)
			if err != nil {
				log.Printf("WARNING: Failed to parse graph_data from state: %v", err)
			}
		}
		
		if req, ok := state["gnn_query_request"].(map[string]any); ok {
			if queryType, ok := req["query_type"].(string); ok {
				gnnRequest.QueryType = queryType
			}
			
			// Use GraphData if available, otherwise fall back to direct nodes/edges
			if graphData != nil {
				gnnRequest.Nodes = convertGraphDataNodes(graphData.Nodes)
				gnnRequest.Edges = convertGraphDataEdges(graphData.Edges)
			} else {
				if nodes, ok := req["nodes"].([]any); ok {
					gnnRequest.Nodes = convertToNodes(nodes)
				}
				if edges, ok := req["edges"].([]any); ok {
					gnnRequest.Edges = convertToEdges(edges)
				}
			}
			
			if params, ok := req["params"].(map[string]any); ok {
				gnnRequest.Params = params
			}
		} else {
			return nil, fmt.Errorf("gnn_query_request not found in state")
		}
		
		// Determine endpoint based on query type
		var endpoint string
		var payload map[string]any
		
		switch gnnRequest.QueryType {
		case "embeddings":
			endpoint = fmt.Sprintf("%s/gnn/embeddings", trainingServiceURL)
			payload = map[string]any{
				"nodes":      gnnRequest.Nodes,
				"edges":      gnnRequest.Edges,
				"graph_level": gnnRequest.Params["graph_level"],
			}
		case "classify":
			endpoint = fmt.Sprintf("%s/gnn/classify", trainingServiceURL)
			payload = map[string]any{
				"nodes": gnnRequest.Nodes,
				"edges": gnnRequest.Edges,
			}
		case "predict-links":
			endpoint = fmt.Sprintf("%s/gnn/predict-links", trainingServiceURL)
			payload = map[string]any{
				"nodes": gnnRequest.Nodes,
				"edges": gnnRequest.Edges,
				"top_k": gnnRequest.Params["top_k"],
			}
		case "structural-insights", "insights":
			endpoint = fmt.Sprintf("%s/gnn/structural-insights", trainingServiceURL)
			payload = map[string]any{
				"nodes":        gnnRequest.Nodes,
				"edges":        gnnRequest.Edges,
				"insight_type": gnnRequest.Params["insight_type"],
				"threshold":    gnnRequest.Params["threshold"],
			}
		default:
			return nil, fmt.Errorf("unknown GNN query type: %s", gnnRequest.QueryType)
		}
		
		body, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal GNN request: %w", err)
		}
		
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build GNN request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		
		resp, err := gnnHTTPClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("GNN query failed: %w", err)
		}
		defer resp.Body.Close()
		
		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			return nil, fmt.Errorf("GNN query failed: HTTP %d - %s", resp.StatusCode, string(bodyBytes))
		}
		
		var result map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, fmt.Errorf("decode GNN response: %w", err)
		}
		
		// Convert GNN response to unified GraphData format if possible
		gnnGraphData, err := models.FromGNN(result)
		if err != nil {
			log.Printf("WARNING: Failed to convert GNN response to GraphData: %v", err)
		}
		
		// Add GNN results to state
		newState := make(map[string]any)
		for k, v := range state {
			newState[k] = v
		}
		newState["gnn_result"] = result
		if gnnGraphData != nil {
			newState["graph_data"] = gnnGraphData
		}
		
		return newState, nil
	})
}

// HybridQueryNode returns a node that queries both KG and GNN and combines results.
func HybridQueryNode(opts GNNProcessorOptions) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		extractServiceURL := opts.ExtractServiceURL
		if extractServiceURL == "" {
			extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
			if extractServiceURL == "" {
				extractServiceURL = "http://extract-service:19080"
			}
		}
		
		trainingServiceURL := opts.TrainingServiceURL
		if trainingServiceURL == "" {
			trainingServiceURL = os.Getenv("TRAINING_SERVICE_URL")
			if trainingServiceURL == "" {
				trainingServiceURL = "http://training-service:8080"
			}
		}
		
		// Extract hybrid query request
		var hybridReq HybridQueryRequest
		if req, ok := state["hybrid_query_request"].(map[string]any); ok {
			if query, ok := req["query"].(string); ok {
				hybridReq.Query = query
			}
			if projectID, ok := req["project_id"].(string); ok {
				hybridReq.ProjectID = projectID
			}
			if systemID, ok := req["system_id"].(string); ok {
				hybridReq.SystemID = systemID
			}
			if queryKG, ok := req["query_kg"].(bool); ok {
				hybridReq.QueryKG = queryKG
			} else {
				hybridReq.QueryKG = true // Default to querying KG
			}
			if queryGNN, ok := req["query_gnn"].(bool); ok {
				hybridReq.QueryGNN = queryGNN
			} else {
				// Auto-detect: if structural query, query GNN
				hybridReq.QueryGNN = isStructuralQuery(hybridReq.Query)
			}
			if combine, ok := req["combine"].(bool); ok {
				hybridReq.Combine = combine
			} else {
				hybridReq.Combine = true // Default to combining
			}
			if gnnType, ok := req["gnn_type"].(string); ok {
				hybridReq.GNNType = gnnType
			} else {
				hybridReq.GNNType = "structural-insights" // Default
			}
		} else {
			return nil, fmt.Errorf("hybrid_query_request not found in state")
		}
		
		newState := make(map[string]any)
		for k, v := range state {
			newState[k] = v
		}
		
		kgResult := make(map[string]any)
		gnnResult := make(map[string]any)
		
		// Query Knowledge Graph
		if hybridReq.QueryKG {
			kgEndpoint := fmt.Sprintf("%s/knowledge-graph/query", extractServiceURL)
			kgPayload := map[string]any{"query": hybridReq.Query}
			
			if hybridReq.ProjectID != "" || hybridReq.SystemID != "" {
				kgPayload["params"] = make(map[string]any)
				if hybridReq.ProjectID != "" {
					kgPayload["params"].(map[string]any)["project_id"] = hybridReq.ProjectID
				}
				if hybridReq.SystemID != "" {
					kgPayload["params"].(map[string]any)["system_id"] = hybridReq.SystemID
				}
			}
			
			kgBody, err := json.Marshal(kgPayload)
			if err == nil {
				kgReq, err := http.NewRequestWithContext(ctx, http.MethodPost, kgEndpoint, bytes.NewReader(kgBody))
				if err == nil {
					kgReq.Header.Set("Content-Type", "application/json")
					kgResp, err := knowledgeGraphHTTPClient.Do(kgReq)
					if err == nil {
						defer kgResp.Body.Close()
						if kgResp.StatusCode == http.StatusOK {
							json.NewDecoder(kgResp.Body).Decode(&kgResult)
						}
					}
				}
			}
		}
		
		// Query GNN (need graph data first)
		if hybridReq.QueryGNN {
			// Try to extract nodes/edges from KG result using unified format
			var graphData *models.GraphData
			
			if kgResult != nil {
				// Try to convert KG result to GraphData
				var err error
				graphData, err = models.FromNeo4j(kgResult)
				if err != nil {
					log.Printf("WARNING: Failed to convert KG result to GraphData: %v", err)
					// Fallback to direct extraction
					if nodesData, ok := kgResult["nodes"].([]any); ok {
						if edgesData, ok := kgResult["edges"].([]any); ok {
							graphData = &models.GraphData{
								Nodes: convertNodesFromAny(nodesData),
								Edges: convertEdgesFromAny(edgesData),
							}
						}
					}
				}
			}
			
			// If no graph data, try to get it via simple KG query
			if graphData == nil || (len(graphData.Nodes) == 0 && len(graphData.Edges) == 0) {
				simpleQuery := "MATCH (n) RETURN n LIMIT 100"
				if hybridReq.ProjectID != "" {
					simpleQuery = fmt.Sprintf("MATCH (n) WHERE n.project_id = '%s' RETURN n LIMIT 100", hybridReq.ProjectID)
				}
				
				simpleEndpoint := fmt.Sprintf("%s/knowledge-graph/query", extractServiceURL)
				simplePayload := map[string]any{"query": simpleQuery}
				simpleBody, _ := json.Marshal(simplePayload)
				simpleReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, simpleEndpoint, bytes.NewReader(simpleBody))
				simpleReq.Header.Set("Content-Type", "application/json")
				simpleResp, err := knowledgeGraphHTTPClient.Do(simpleReq)
				if err == nil {
					defer simpleResp.Body.Close()
					if simpleResp.StatusCode == http.StatusOK {
						var simpleResult map[string]any
						json.NewDecoder(simpleResp.Body).Decode(&simpleResult)
						graphData, _ = models.FromNeo4j(simpleResult)
					}
				}
			}
			
			// Convert GraphData to GNN format
			nodes := []Node{}
			edges := []Edge{}
			if graphData != nil {
				nodes = convertGraphDataNodes(graphData.Nodes)
				edges = convertGraphDataEdges(graphData.Edges)
			}
			
			// Query GNN if we have graph data
			if len(nodes) > 0 || len(edges) > 0 {
				var gnnEndpoint string
				var gnnPayload map[string]any
				
				switch hybridReq.GNNType {
				case "embeddings":
					gnnEndpoint = fmt.Sprintf("%s/gnn/embeddings", trainingServiceURL)
					gnnPayload = map[string]any{
						"nodes":      nodes,
						"edges":      edges,
						"graph_level": true,
					}
				case "classify":
					gnnEndpoint = fmt.Sprintf("%s/gnn/classify", trainingServiceURL)
					gnnPayload = map[string]any{
						"nodes": nodes,
						"edges": edges,
					}
				case "predict-links":
					gnnEndpoint = fmt.Sprintf("%s/gnn/predict-links", trainingServiceURL)
					gnnPayload = map[string]any{
						"nodes": nodes,
						"edges": edges,
						"top_k": 10,
					}
				default: // structural-insights
					gnnEndpoint = fmt.Sprintf("%s/gnn/structural-insights", trainingServiceURL)
					gnnPayload = map[string]any{
						"nodes":        nodes,
						"edges":        edges,
						"insight_type": "all",
						"threshold":    0.5,
					}
				}
				
				gnnBody, err := json.Marshal(gnnPayload)
				if err == nil {
					gnnReq, err := http.NewRequestWithContext(ctx, http.MethodPost, gnnEndpoint, bytes.NewReader(gnnBody))
					if err == nil {
						gnnReq.Header.Set("Content-Type", "application/json")
						gnnResp, err := gnnHTTPClient.Do(gnnReq)
						if err == nil {
							defer gnnResp.Body.Close()
							if gnnResp.StatusCode == http.StatusOK {
								json.NewDecoder(gnnResp.Body).Decode(&gnnResult)
							}
						}
					}
				}
			}
		}
		
		// Combine results
		if hybridReq.Combine {
			newState["hybrid_result"] = map[string]any{
				"kg_result":  kgResult,
				"gnn_result": gnnResult,
				"combined":   true,
			}
		} else {
			if hybridReq.QueryKG {
				newState["kg_result"] = kgResult
			}
			if hybridReq.QueryGNN {
				newState["gnn_result"] = gnnResult
			}
		}
		
		return newState, nil
	})
}

// convertToNodes converts []any to []Node.
func convertToNodes(nodesData []any) []Node {
	nodes := make([]Node, 0, len(nodesData))
	for _, n := range nodesData {
		if nodeMap, ok := n.(map[string]any); ok {
			node := Node{
				ID:    getString(nodeMap, "id"),
				Type:  getString(nodeMap, "type"),
				Label: getString(nodeMap, "label"),
			}
			if props, ok := nodeMap["properties"].(map[string]any); ok {
				node.Props = props
			}
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// convertToEdges converts []any to []Edge.
func convertToEdges(edgesData []any) []Edge {
	edges := make([]Edge, 0, len(edgesData))
	for _, e := range edgesData {
		if edgeMap, ok := e.(map[string]any); ok {
			edge := Edge{
				SourceID: getString(edgeMap, "source_id"),
				TargetID: getString(edgeMap, "target_id"),
				Label:    getString(edgeMap, "label"),
			}
			if props, ok := edgeMap["properties"].(map[string]any); ok {
				edge.Props = props
			}
			edges = append(edges, edge)
		}
	}
	return edges
}

// getString safely extracts string from map.
func getString(m map[string]any, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// convertGraphDataNodes converts unified models.Node to workflow Node format.
func convertGraphDataNodes(graphNodes []models.Node) []Node {
	nodes := make([]Node, 0, len(graphNodes))
	for _, gn := range graphNodes {
		nodes = append(nodes, Node{
			ID:    gn.ID,
			Type:  gn.Type,
			Label: gn.Label,
			Props: gn.Properties,
		})
	}
	return nodes
}

// convertGraphDataEdges converts unified models.Edge to workflow Edge format.
func convertGraphDataEdges(graphEdges []models.Edge) []Edge {
	edges := make([]Edge, 0, len(graphEdges))
	for _, ge := range graphEdges {
		edges = append(edges, Edge{
			SourceID: ge.SourceID,
			TargetID: ge.TargetID,
			Label:    ge.Label,
			Props:    ge.Properties,
		})
	}
	return edges
}

// convertNodesFromAny converts []any to unified models.Node format.
func convertNodesFromAny(nodesData []any) []models.Node {
	nodes := make([]models.Node, 0, len(nodesData))
	for _, nodeData := range nodesData {
		if nodeMap, ok := nodeData.(map[string]any); ok {
			node := models.Node{
				Properties: make(map[string]any),
			}
			
			if id, ok := nodeMap["id"].(string); ok {
				node.ID = id
			} else if id, ok := nodeMap["node_id"].(string); ok {
				node.ID = id
			}
			
			if nodeType, ok := nodeMap["type"].(string); ok {
				node.Type = nodeType
			} else if nodeType, ok := nodeMap["node_type"].(string); ok {
				node.Type = nodeType
			}
			
			if label, ok := nodeMap["label"].(string); ok {
				node.Label = label
			}
			
			if props, ok := nodeMap["properties"].(map[string]any); ok {
				node.Properties = props
			} else {
				// Copy all non-standard fields to properties
				for k, v := range nodeMap {
					if k != "id" && k != "node_id" && k != "type" && k != "node_type" && k != "label" {
						node.Properties[k] = v
					}
				}
			}
			
			if node.ID != "" {
				nodes = append(nodes, node)
			}
		}
	}
	return nodes
}

// convertEdgesFromAny converts []any to unified models.Edge format.
func convertEdgesFromAny(edgesData []any) []models.Edge {
	edges := make([]models.Edge, 0, len(edgesData))
	for _, edgeData := range edgesData {
		if edgeMap, ok := edgeData.(map[string]any); ok {
			edge := models.Edge{
				Properties: make(map[string]any),
			}
			
			if source, ok := edgeMap["source"].(string); ok {
				edge.SourceID = source
			} else if source, ok := edgeMap["source_id"].(string); ok {
				edge.SourceID = source
			}
			
			if target, ok := edgeMap["target"].(string); ok {
				edge.TargetID = target
			} else if target, ok := edgeMap["target_id"].(string); ok {
				edge.TargetID = target
			}
			
			if label, ok := edgeMap["label"].(string); ok {
				edge.Label = label
			} else if relType, ok := edgeMap["relation_type"].(string); ok {
				edge.Label = relType
			}
			
			if props, ok := edgeMap["properties"].(map[string]any); ok {
				edge.Properties = props
			} else {
				// Copy all non-standard fields to properties
				for k, v := range edgeMap {
					if k != "source" && k != "source_id" && k != "target" && k != "target_id" && k != "label" && k != "relation_type" {
						edge.Properties[k] = v
					}
				}
			}
			
			if edge.SourceID != "" && edge.TargetID != "" {
				edges = append(edges, edge)
			}
		}
	}
	return edges
}

