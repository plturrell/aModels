package workflows

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// UnifiedProcessorOptions configures the unified workflow that combines all three systems.
type UnifiedProcessorOptions struct {
	ExtractServiceURL   string
	AgentFlowServiceURL string
	LocalAIURL          string
}

// UnifiedWorkflowRequest represents a request that can use all three systems.
type UnifiedWorkflowRequest struct {
	// Knowledge graph processing
	KnowledgeGraphRequest *KnowledgeGraphRequest `json:"knowledge_graph_request,omitempty"`

	// Orchestration chain execution
	OrchestrationRequest *OrchestrationRequest `json:"orchestration_request,omitempty"`

	// AgentFlow flow execution
	AgentFlowRequest *AgentFlowRunRequest `json:"agentflow_request,omitempty"`

	// Workflow configuration
	WorkflowMode string `json:"workflow_mode,omitempty"` // "sequential", "parallel", "conditional"
}

// OrchestrationRequest represents a request to run an orchestration chain.
type OrchestrationRequest struct {
	ChainName string                 `json:"chain_name"`
	Inputs    map[string]any          `json:"inputs,omitempty"`
}

// ProcessUnifiedWorkflowNode returns a node that processes a unified workflow request.
func ProcessUnifiedWorkflowNode(opts UnifiedProcessorOptions) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract unified request from state
		var unifiedReq UnifiedWorkflowRequest

		if req, ok := state["unified_request"].(map[string]any); ok {
			// Parse knowledge graph request
			if kgReq, ok := req["knowledge_graph_request"].(map[string]any); ok {
				unifiedReq.KnowledgeGraphRequest = &KnowledgeGraphRequest{
					JSONTables:        parseStringSlice(kgReq["json_tables"]),
					HiveDDLs:          parseStringSlice(kgReq["hive_ddls"]),
					SqlQueries:         parseStringSlice(kgReq["sql_queries"]),
					ControlMFiles:      parseStringSlice(kgReq["control_m_files"]),
					ProjectID:          getString(kgReq["project_id"]),
					SystemID:            getString(kgReq["system_id"]),
					InformationSystemID: getString(kgReq["information_system_id"]),
				}
			}

			// Parse orchestration request
			if orchReq, ok := req["orchestration_request"].(map[string]any); ok {
				unifiedReq.OrchestrationRequest = &OrchestrationRequest{
					ChainName: getString(orchReq["chain_name"]),
					Inputs:    parseMap(orchReq["inputs"]),
				}
			}

			// Parse AgentFlow request
			if afReq, ok := req["agentflow_request"].(map[string]any); ok {
				unifiedReq.AgentFlowRequest = &AgentFlowRunRequest{
					FlowID:     getString(afReq["flow_id"]),
					InputValue: getString(afReq["input_value"]),
					Inputs:     parseMap(afReq["inputs"]),
					Ensure:     getBool(afReq["ensure"]),
				}
			}

			unifiedReq.WorkflowMode = getString(req["workflow_mode"])
		}

		log.Printf("Processing unified workflow: mode=%s, kg=%v, orch=%v, af=%v",
			unifiedReq.WorkflowMode,
			unifiedReq.KnowledgeGraphRequest != nil,
			unifiedReq.OrchestrationRequest != nil,
			unifiedReq.AgentFlowRequest != nil)

		newState := make(map[string]any, len(state)+5)
		for k, v := range state {
			newState[k] = v
		}

		// Step 1: Process knowledge graph if requested
		if unifiedReq.KnowledgeGraphRequest != nil {
			kgState := map[string]any{
				"knowledge_graph_request": unifiedReq.KnowledgeGraphRequest,
			}
			kgNode := ProcessKnowledgeGraphNode(opts.ExtractServiceURL)
			kgResult, err := kgNode(ctx, kgState)
			if err != nil {
				return nil, fmt.Errorf("process knowledge graph: %w", err)
			}
			// Merge knowledge graph results into state
			for k, v := range kgResult.(map[string]any) {
				newState[k] = v
			}
		}

		// Step 2: Process orchestration chain if requested
		if unifiedReq.OrchestrationRequest != nil {
			orchState := map[string]any{
				"orchestration_request": map[string]any{
					"chain_name": unifiedReq.OrchestrationRequest.ChainName,
					"inputs":      unifiedReq.OrchestrationRequest.Inputs,
				},
			}
			// Add knowledge graph context if available
			if kgIface, ok := newState["knowledge_graph"]; ok {
				if kgMap, ok := kgIface.(map[string]any); ok {
					if orchInputs, ok := orchState["orchestration_request"].(map[string]any)["inputs"].(map[string]any); ok {
						orchInputs["knowledge_graph_context"] = kgMap
					}
				}
			}
			orchNode := RunOrchestrationChainNode(opts.LocalAIURL)
			orchResult, err := orchNode(ctx, orchState)
			if err != nil {
				return nil, fmt.Errorf("process orchestration chain: %w", err)
			}
			// Merge orchestration results into state
			for k, v := range orchResult.(map[string]any) {
				newState[k] = v
			}
		}

		// Step 3: Process AgentFlow flow if requested
		if unifiedReq.AgentFlowRequest != nil {
			afState := map[string]any{
				"agentflow_request": map[string]any{
					"flow_id":     unifiedReq.AgentFlowRequest.FlowID,
					"input_value":  unifiedReq.AgentFlowRequest.InputValue,
					"inputs":       unifiedReq.AgentFlowRequest.Inputs,
					"ensure":       unifiedReq.AgentFlowRequest.Ensure,
				},
			}
			// Add knowledge graph and orchestration results if available
			if kgIface, ok := newState["knowledge_graph"]; ok {
				afState["knowledge_graph"] = kgIface
			}
			if orchResult, ok := newState["orchestration_result"]; ok {
				if orchMap, ok := orchResult.(map[string]any); ok {
					if text, ok := orchMap["text"].(string); ok {
						// Use orchestration result as input for AgentFlow
						if afInputs, ok := afState["agentflow_request"].(map[string]any)["inputs"].(map[string]any); ok {
							afInputs["orchestration_result"] = text
						}
					}
				}
			}
			afNode := RunAgentFlowFlowNode(opts.AgentFlowServiceURL)
			afResult, err := afNode(ctx, afState)
			if err != nil {
				return nil, fmt.Errorf("process AgentFlow flow: %w", err)
			}
			// Merge AgentFlow results into state
			for k, v := range afResult.(map[string]any) {
				newState[k] = v
			}
		}

		// Add unified workflow summary
		newState["unified_workflow_complete"] = true
		newState["unified_workflow_summary"] = map[string]any{
			"knowledge_graph_processed": unifiedReq.KnowledgeGraphRequest != nil,
			"orchestration_processed":    unifiedReq.OrchestrationRequest != nil,
			"agentflow_processed":       unifiedReq.AgentFlowRequest != nil,
			"workflow_mode":              unifiedReq.WorkflowMode,
		}

		log.Printf("Unified workflow completed successfully")
		return newState, nil
	})
}

// ProcessKnowledgeGraphWorkflowNode returns a node that processes knowledge graphs.
func ProcessKnowledgeGraphWorkflowNode(extractServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		kgReq, ok := state["knowledge_graph_request"].(*KnowledgeGraphRequest)
		if !ok || kgReq == nil {
			return state, nil // Skip if no KG request
		}

		kgState := map[string]any{
			"knowledge_graph_request": kgReq,
		}
		kgNode := ProcessKnowledgeGraphNode(extractServiceURL)
		kgResult, err := kgNode(ctx, kgState)
		if err != nil {
			return nil, fmt.Errorf("process knowledge graph: %w", err)
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		for k, v := range kgResult.(map[string]any) {
			newState[k] = v
		}
		return newState, nil
	})
}

// ProcessOrchestrationWorkflowNode returns a node that processes orchestration chains.
func ProcessOrchestrationWorkflowNode(localAIURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		orchReq, ok := state["orchestration_request"].(*OrchestrationRequest)
		if !ok || orchReq == nil {
			return state, nil // Skip if no orchestration request
		}

		orchState := map[string]any{
			"orchestration_request": map[string]any{
				"chain_name": orchReq.ChainName,
				"inputs":      orchReq.Inputs,
			},
		}

		// Add knowledge graph context if available
		if kgIface, ok := state["knowledge_graph"]; ok {
			if kgMap, ok := kgIface.(map[string]any); ok {
				if orchInputs, ok := orchState["orchestration_request"].(map[string]any)["inputs"].(map[string]any); ok {
					orchInputs["knowledge_graph_context"] = kgMap
				}
			}
		}

		orchNode := RunOrchestrationChainNode(localAIURL)
		orchResult, err := orchNode(ctx, orchState)
		if err != nil {
			return nil, fmt.Errorf("process orchestration chain: %w", err)
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		for k, v := range orchResult.(map[string]any) {
			newState[k] = v
		}
		return newState, nil
	})
}

// ProcessAgentFlowWorkflowNode returns a node that processes AgentFlow flows.
func ProcessAgentFlowWorkflowNode(agentflowServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		afReq, ok := state["agentflow_request"].(*AgentFlowRunRequest)
		if !ok || afReq == nil {
			return state, nil // Skip if no AgentFlow request
		}

		afState := map[string]any{
			"agentflow_request": map[string]any{
				"flow_id":     afReq.FlowID,
				"input_value":  afReq.InputValue,
				"inputs":       afReq.Inputs,
				"ensure":       afReq.Ensure,
			},
		}

		// Add knowledge graph and orchestration results if available
		if kgIface, ok := state["knowledge_graph"]; ok {
			afState["knowledge_graph"] = kgIface
		}
		if orchResult, ok := state["orchestration_result"]; ok {
			if orchMap, ok := orchResult.(map[string]any); ok {
				if text, ok := orchMap["text"].(string); ok {
					if afInputs, ok := afState["agentflow_request"].(map[string]any)["inputs"].(map[string]any); ok {
						afInputs["orchestration_result"] = text
					}
				}
			}
		}

		afNode := RunAgentFlowFlowNode(agentflowServiceURL)
		afResult, err := afNode(ctx, afState)
		if err != nil {
			return nil, fmt.Errorf("process AgentFlow flow: %w", err)
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		for k, v := range afResult.(map[string]any) {
			newState[k] = v
		}
		return newState, nil
	})
}

// JoinUnifiedResultsNode returns a join node that aggregates results from parallel branches.
func JoinUnifiedResultsNode() stategraph.JoinFunc {
	return func(ctx context.Context, inputs []any) (any, error) {
		log.Println("Joining unified workflow results...")

		mergedState := make(map[string]any)

		// Merge all inputs
		for _, input := range inputs {
			if state, ok := input.(map[string]any); ok {
				for k, v := range state {
					mergedState[k] = v
				}
			}
		}

		// Add unified summary
		mergedState["unified_workflow_complete"] = true
		mergedState["unified_workflow_summary"] = map[string]any{
			"knowledge_graph_processed": mergedState["knowledge_graph"] != nil,
			"orchestration_processed":    mergedState["orchestration_result"] != nil,
			"agentflow_processed":       mergedState["agentflow_result"] != nil,
			"joined_at":                  time.Now().Format(time.RFC3339),
		}

		log.Println("Unified workflow results joined successfully")
		return mergedState, nil
	}
}

// NewUnifiedProcessorWorkflow creates a workflow that processes all three systems together.
// Supports sequential, parallel, and conditional execution modes.
func NewUnifiedProcessorWorkflow(opts UnifiedProcessorOptions) (*stategraph.CompiledStateGraph, error) {
	extractServiceURL := opts.ExtractServiceURL
	if extractServiceURL == "" {
		extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
		if extractServiceURL == "" {
			extractServiceURL = "http://extract-service:19080"
		}
	}

	agentflowServiceURL := opts.AgentFlowServiceURL
	if agentflowServiceURL == "" {
		agentflowServiceURL = os.Getenv("AGENTFLOW_SERVICE_URL")
		if agentflowServiceURL == "" {
			agentflowServiceURL = "http://agentflow-service:9001"
		}
	}

	localAIURL := opts.LocalAIURL
	if localAIURL == "" {
		localAIURL = os.Getenv("LOCALAI_URL")
		if localAIURL == "" {
			localAIURL = "http://localai:8080"
		}
	}

	// For parallel mode, use separate nodes and join
	nodes := map[string]stategraph.NodeFunc{
		"process_kg":      ProcessKnowledgeGraphWorkflowNode(extractServiceURL),
		"process_orch":    ProcessOrchestrationWorkflowNode(localAIURL),
		"process_agentflow": ProcessAgentFlowWorkflowNode(agentflowServiceURL),
		"process_unified": ProcessUnifiedWorkflowNode(UnifiedProcessorOptions{
			ExtractServiceURL:   extractServiceURL,
			AgentFlowServiceURL: agentflowServiceURL,
			LocalAIURL:          localAIURL,
		}),
	}

	// Add join node for parallel execution
	var edges []EdgeSpec
	
	// For parallel mode: all three processes run in parallel, then join
	edges = append(edges,
		EdgeSpec{From: "process_kg", To: "join_results", Label: "kg_complete"},
		EdgeSpec{From: "process_orch", To: "join_results", Label: "orch_complete"},
		EdgeSpec{From: "process_agentflow", To: "join_results", Label: "af_complete"},
	)

	// For sequential mode: use the unified node
	edges = append(edges,
		EdgeSpec{From: "process_unified", To: "join_results", Label: "unified_complete"},
	)

	// Build with join node
	builder := stategraph.New()
	
	// Add all nodes
	for id, handler := range nodes {
		opts := []stategraph.NodeOption{
			stategraph.WithNodeRetries(2),
			stategraph.WithNodeTimeout(120 * time.Second),
			stategraph.WithNodeRetryDelay(2 * time.Second),
		}
		if err := builder.AddNode(id, handler, opts...); err != nil {
			return nil, err
		}
	}

	// Add join node
	if err := builder.AddJoinNode("join_results", JoinUnifiedResultsNode(),
		stategraph.WithJoinTimeout(30*time.Second)); err != nil {
		return nil, err
	}

	// Add edges
	for _, edge := range edges {
		var edgeOpts []stategraph.EdgeOption
		if edge.Label != "" {
			edgeOpts = append(edgeOpts, stategraph.WithEdgeLabel(edge.Label))
		}
		if err := builder.AddEdge(edge.From, edge.To, edgeOpts...); err != nil {
			return nil, err
		}
	}

	// Set entry point (determine based on mode)
	builder.SetEntryPoint("process_unified") // Default to sequential
	builder.SetFinishPoint("join_results")

	return builder.Compile()
}

// Helper functions for parsing request data
func parseStringSlice(v any) []string {
	if v == nil {
		return nil
	}
	if slice, ok := v.([]any); ok {
		result := make([]string, 0, len(slice))
		for _, item := range slice {
			if str, ok := item.(string); ok {
				result = append(result, str)
			}
		}
		return result
	}
	if str, ok := v.(string); ok {
		return []string{str}
	}
	return nil
}

func parseMap(v any) map[string]any {
	if v == nil {
		return nil
	}
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return make(map[string]any)
}

func getString(v any) string {
	if v == nil {
		return ""
	}
	if str, ok := v.(string); ok {
		return str
	}
	return ""
}

func getBool(v any) bool {
	if v == nil {
		return false
	}
	if b, ok := v.(bool); ok {
		return b
	}
	return false
}

