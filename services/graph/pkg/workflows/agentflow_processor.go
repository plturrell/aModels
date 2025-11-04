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

// AgentFlowProcessorOptions configures the AgentFlow processing workflow.
type AgentFlowProcessorOptions struct {
	AgentFlowServiceURL string // URL to AgentFlow service (e.g., "http://agentflow-service:9001")
	ExtractServiceURL   string // URL to extract service for knowledge graph queries
}

// AgentFlowRunRequest represents a request to run an AgentFlow flow.
type AgentFlowRunRequest struct {
	FlowID      string                 `json:"flow_id"`
	InputValue  string                 `json:"input_value,omitempty"`
	Inputs      map[string]any          `json:"inputs,omitempty"`
	Ensure      bool                   `json:"ensure,omitempty"` // Ensure flow is synced before running
}

// AgentFlowRunResponse represents the response from AgentFlow flow execution.
type AgentFlowRunResponse struct {
	LocalID  string         `json:"local_id"`
	RemoteID string         `json:"remote_id"`
	Result   map[string]any `json:"result"`
}

var agentflowHTTPClient = &http.Client{
	Timeout: 120 * time.Second,
}

// RunAgentFlowFlowNode returns a node that runs an AgentFlow flow.
func RunAgentFlowFlowNode(agentflowServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract flow request from state
		var flowRequest AgentFlowRunRequest

		// Try to get from state directly
		if req, ok := state["agentflow_request"].(map[string]any); ok {
			if flowID, ok := req["flow_id"].(string); ok {
				flowRequest.FlowID = flowID
			}
			if inputValue, ok := req["input_value"].(string); ok {
				flowRequest.InputValue = inputValue
			}
			if inputs, ok := req["inputs"].(map[string]any); ok {
				flowRequest.Inputs = inputs
			}
			if ensure, ok := req["ensure"].(bool); ok {
				flowRequest.Ensure = ensure
			}
		} else {
			// Try to construct from individual fields
			if flowID, ok := state["flow_id"].(string); ok {
				flowRequest.FlowID = flowID
			}
			if inputValue, ok := state["input_value"].(string); ok {
				flowRequest.InputValue = inputValue
			}
			if inputs, ok := state["inputs"].(map[string]any); ok {
				flowRequest.Inputs = inputs
			}
			if ensure, ok := state["ensure"].(bool); ok {
				flowRequest.Ensure = ensure
			}
		}

		if flowRequest.FlowID == "" {
			log.Println("No AgentFlow flow ID provided; skipping execution")
			return state, nil
		}

		if agentflowServiceURL == "" || agentflowServiceURL == "offline" {
			agentflowServiceURL = os.Getenv("AGENTFLOW_SERVICE_URL")
			if agentflowServiceURL == "" {
				agentflowServiceURL = "http://agentflow-service:9001"
			}
		}

		log.Printf("Running AgentFlow flow: %s (ensure=%v)", flowRequest.FlowID, flowRequest.Ensure)

		// Call AgentFlow service to run flow
		endpoint := strings.TrimRight(agentflowServiceURL, "/") + "/flows/" + flowRequest.FlowID + "/run"
		body, err := json.Marshal(map[string]any{
			"input_value": flowRequest.InputValue,
			"inputs":      flowRequest.Inputs,
			"ensure":      flowRequest.Ensure,
		})
		if err != nil {
			return nil, fmt.Errorf("marshal AgentFlow request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build AgentFlow request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := agentflowHTTPClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("request AgentFlow execution: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			return nil, fmt.Errorf("AgentFlow execution failed with status %s: %s",
				resp.Status, strings.TrimSpace(string(bodyBytes)))
		}

		var flowResp AgentFlowRunResponse
		if err := json.NewDecoder(resp.Body).Decode(&flowResp); err != nil {
			return nil, fmt.Errorf("decode AgentFlow response: %w", err)
		}

		// Store results in state
		newState := make(map[string]any, len(state)+3)
		for k, v := range state {
			newState[k] = v
		}
		newState["agentflow_result"] = flowResp
		newState["agentflow_local_id"] = flowResp.LocalID
		newState["agentflow_remote_id"] = flowResp.RemoteID

		log.Printf("AgentFlow flow executed: local_id=%s, remote_id=%s",
			flowResp.LocalID, flowResp.RemoteID)

		return newState, nil
	})
}

// QueryKnowledgeGraphForFlowNode returns a node that queries knowledge graphs to inform flow execution.
func QueryKnowledgeGraphForFlowNode(extractServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract knowledge graph query from state
		query, ok := state["knowledge_graph_query"].(string)
		if !ok || query == "" {
			log.Println("No knowledge graph query provided; skipping")
			return state, nil
		}

		// Extract project/system IDs if available
		projectID, _ := state["project_id"].(string)
		systemID, _ := state["system_id"].(string)

		if extractServiceURL == "" || extractServiceURL == "offline" {
			extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
			if extractServiceURL == "" {
				extractServiceURL = "http://extract-service:19080"
			}
		}

		log.Printf("Querying knowledge graph for flow planning: query=%s, project=%s, system=%s",
			query, projectID, systemID)

		// Query knowledge graph (placeholder - would need actual query endpoint)
		// For now, we'll use the knowledge graph from state if available
		kgIface, ok := state["knowledge_graph"]
		if !ok {
			log.Println("No knowledge graph available in state; skipping query")
			return state, nil
		}

		kgMap, ok := kgIface.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("knowledge_graph is not a map")
		}

		// Extract relevant information for flow planning
		nodesIface, _ := kgMap["nodes"].([]any)
		edgesIface, _ := kgMap["edges"].([]any)
		qualityIface, _ := kgMap["quality"].(map[string]any)

		// Store query results in state
		newState := make(map[string]any, len(state)+3)
		for k, v := range state {
			newState[k] = v
		}
		newState["knowledge_graph_query_results"] = map[string]any{
			"query":      query,
			"node_count": len(nodesIface),
			"edge_count": len(edgesIface),
			"quality":    qualityIface,
		}

		log.Printf("Knowledge graph query completed: nodes=%d, edges=%d",
			len(nodesIface), len(edgesIface))

		return newState, nil
	})
}

// AnalyzeFlowResultsNode returns a node that analyzes AgentFlow execution results.
func AnalyzeFlowResultsNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		resultIface, ok := state["agentflow_result"]
		if !ok {
			log.Println("No AgentFlow result available; skipping analysis")
			return state, nil
		}

		resultMap, ok := resultIface.(AgentFlowRunResponse)
		if !ok {
			// Try to convert from map
			if rawMap, ok := resultIface.(map[string]any); ok {
				// Extract result
				if rawResult, ok := rawMap["result"].(map[string]any); ok {
					log.Printf("Analyzing AgentFlow result: %v", rawResult)

					// Determine success/failure
					success := true
					if errorMsg, ok := rawResult["error"].(string); ok && errorMsg != "" {
						success = false
						log.Printf("AgentFlow execution failed: %s", errorMsg)
					}

					newState := make(map[string]any, len(state)+2)
					for k, v := range state {
						newState[k] = v
					}
					newState["agentflow_success"] = success
					newState["agentflow_analysis"] = map[string]any{
						"success": success,
						"result":   rawResult,
					}

					return newState, nil
				}
			}
			return nil, fmt.Errorf("agentflow_result is not a valid response")
		}

		// Analyze result
		success := true
		if resultMap.Result != nil {
			if errorMsg, ok := resultMap.Result["error"].(string); ok && errorMsg != "" {
				success = false
				log.Printf("AgentFlow execution failed: %s", errorMsg)
			}
		}

		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}
		newState["agentflow_success"] = success
		newState["agentflow_analysis"] = map[string]any{
			"success": success,
			"result":  resultMap.Result,
		}

		log.Printf("AgentFlow result analyzed: success=%v", success)
		return newState, nil
	})
}

// NewAgentFlowProcessorWorkflow creates a workflow that processes AgentFlow flows with knowledge graph integration.
func NewAgentFlowProcessorWorkflow(opts AgentFlowProcessorOptions) (*stategraph.CompiledStateGraph, error) {
	agentflowServiceURL := opts.AgentFlowServiceURL
	if agentflowServiceURL == "" {
		agentflowServiceURL = os.Getenv("AGENTFLOW_SERVICE_URL")
		if agentflowServiceURL == "" {
			agentflowServiceURL = "http://agentflow-service:9001"
		}
	}

	extractServiceURL := opts.ExtractServiceURL
	if extractServiceURL == "" {
		extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
		if extractServiceURL == "" {
			extractServiceURL = "http://extract-service:19080"
		}
	}

	nodes := map[string]stategraph.NodeFunc{
		"query_kg":      QueryKnowledgeGraphForFlowNode(extractServiceURL),
		"run_flow":      RunAgentFlowFlowNode(agentflowServiceURL),
		"analyze_result": AnalyzeFlowResultsNode(),
	}

	edges := []EdgeSpec{
		{From: "query_kg", To: "run_flow", Label: "flow_execution"},
		{From: "run_flow", To: "analyze_result", Label: "result_analysis"},
	}

	return BuildGraph("query_kg", "analyze_result", nodes, edges)
}

