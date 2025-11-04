package workflows

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// Import RunDeepAgentNode from deepagents_processor
// This is defined in deepagents_processor.go

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

// ParallelSplitNode executes all three branches in parallel by creating separate state copies.
// This is a workaround since LangGraph conditional edges can only route to one destination.
// In practice, this would be handled by the workflow runtime executing multiple branches.
func ParallelSplitNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		log.Println("Splitting workflow into parallel branches (KG, Orchestration, AgentFlow)...")
		
		// Mark that parallel execution should occur
		// The actual parallel execution happens via the join node waiting for all inputs
		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		newState["parallel_split"] = true
		newState["parallel_branches"] = []string{"process_kg", "process_orch", "process_agentflow"}

		return newState, nil
	})
}

// JoinUnifiedResultsNode returns a join node that aggregates results from parallel branches.
func JoinUnifiedResultsNode() stategraph.JoinFunc {
	return func(ctx context.Context, inputs []any) (any, error) {
		log.Printf("Joining unified workflow results from %d branches...", len(inputs))

		mergedState := make(map[string]any)

		// Merge all inputs (from parallel branches or sequential unified node)
		for i, input := range inputs {
			if state, ok := input.(map[string]any); ok {
				log.Printf("Merging input %d: keys=%v", i, getStateKeys(state))
				for k, v := range state {
					// Don't overwrite existing values unless they're nil
					if existing, exists := mergedState[k]; !exists || existing == nil {
						mergedState[k] = v
					}
				}
			}
		}

		// Determine what was processed
		kgProcessed := mergedState["knowledge_graph"] != nil
		orchProcessed := mergedState["orchestration_result"] != nil || mergedState["orchestration_text"] != nil
		afProcessed := mergedState["agentflow_result"] != nil
		deepagentsProcessed := mergedState["deepagents_result"] != nil || mergedState["deepagents_text"] != nil

		// Add unified summary
		mergedState["unified_workflow_complete"] = true
		mergedState["unified_workflow_summary"] = map[string]any{
			"knowledge_graph_processed": kgProcessed,
			"orchestration_processed":    orchProcessed,
			"agentflow_processed":       afProcessed,
			"deepagents_processed":      deepagentsProcessed,
			"branches_joined":            len(inputs),
			"joined_at":                  time.Now().Format(time.RFC3339),
		}

		log.Printf("Unified workflow results joined: KG=%v, Orch=%v, AF=%v, DeepAgents=%v, branches=%d",
			kgProcessed, orchProcessed, afProcessed, deepagentsProcessed, len(inputs))

		return mergedState, nil
	}
}

// Helper function to get state keys for logging
func getStateKeys(state map[string]any) []string {
	keys := make([]string, 0, len(state))
	for k := range state {
		keys = append(keys, k)
	}
	return keys
}

// WorkflowModeRoutingFunc determines the execution mode and routes accordingly.
func WorkflowModeRoutingFunc(ctx context.Context, value any) ([]string, error) {
	state, ok := value.(map[string]any)
	if !ok {
		return []string{"sequential"}, nil
	}

	// Check for workflow mode in request
	if unifiedReq, ok := state["unified_request"].(map[string]any); ok {
		if mode, ok := unifiedReq["workflow_mode"].(string); ok && mode != "" {
			return []string{mode}, nil
		}
	}

	// Check if multiple requests are present (parallel mode)
	hasKG := state["knowledge_graph_request"] != nil
	hasOrch := state["orchestration_request"] != nil
	hasAF := state["agentflow_request"] != nil
	hasDeepAgents := state["deepagents_request"] != nil

	count := 0
	if hasKG {
		count++
	}
	if hasOrch {
		count++
	}
	if hasAF {
		count++
	}
	if hasDeepAgents {
		count++
	}

	// If multiple independent requests, use parallel mode
	if count > 1 {
		return []string{"parallel"}, nil
	}

	return []string{"sequential"}, nil
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

	// Get DeepAgents service URL
	deepagentsServiceURL := os.Getenv("DEEPAGENTS_SERVICE_URL")
	if deepagentsServiceURL == "" {
		deepagentsServiceURL = "http://deepagents-service:9004"
	}

	// Nodes for both sequential and parallel modes
	nodes := map[string]stategraph.NodeFunc{
		"determine_mode":  DetermineWorkflowModeNode(),
		"split_parallel":  ParallelSplitNode(),
		"process_kg":      ProcessKnowledgeGraphWorkflowNode(extractServiceURL),
		"process_orch":    ProcessOrchestrationWorkflowNode(localAIURL),
		"process_agentflow": ProcessAgentFlowWorkflowNode(agentflowServiceURL),
		"process_deepagents": RunDeepAgentNode(deepagentsServiceURL),
		"process_unified": ProcessUnifiedWorkflowNode(UnifiedProcessorOptions{
			ExtractServiceURL:   extractServiceURL,
			AgentFlowServiceURL: agentflowServiceURL,
			LocalAIURL:          localAIURL,
		}),
	}

	// Edges for workflow routing
	var edges []EdgeSpec
	var conditionalEdges []ConditionalEdgeSpec

	// Determine mode routing
	conditionalEdges = append(conditionalEdges, ConditionalEdgeSpec{
		Source: "determine_mode",
		PathFunc: WorkflowModeRoutingFunc,
		PathMap: map[string]string{
			"sequential": "process_unified",
			"parallel":    "split_parallel",
		},
	})

	// For parallel mode, we need a split node
	// Since LangGraph doesn't have explicit split nodes, we use multiple entry points
	// For true parallel execution, all branches should be reachable simultaneously
	edges = append(edges,
		// Sequential mode path
		EdgeSpec{From: "process_unified", To: "join_results", Label: "unified_complete"},
		// Parallel mode paths (all branches can execute independently)
		EdgeSpec{From: "split_parallel", To: "process_kg", Label: "kg_branch"},
		EdgeSpec{From: "split_parallel", To: "process_orch", Label: "orch_branch"},
		EdgeSpec{From: "split_parallel", To: "process_agentflow", Label: "af_branch"},
		EdgeSpec{From: "split_parallel", To: "process_deepagents", Label: "deepagents_branch"},
		EdgeSpec{From: "process_kg", To: "join_results", Label: "kg_complete"},
		EdgeSpec{From: "process_orch", To: "join_results", Label: "orch_complete"},
		EdgeSpec{From: "process_agentflow", To: "join_results", Label: "af_complete"},
		EdgeSpec{From: "process_deepagents", To: "join_results", Label: "deepagents_complete"},
	)

	// Build with join node
	builder := stategraph.New()
	
	// Add all nodes with retry/timeout configuration
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

	// Add join node for aggregating results
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

	// Add conditional edges
	for _, condEdge := range conditionalEdges {
		if err := builder.AddConditionalEdges(condEdge.Source, condEdge.PathFunc, condEdge.PathMap); err != nil {
			return nil, err
		}
	}

	// For parallel mode, we need to handle multiple entry points
	// Since LangGraph supports this via conditional routing, we use a mode determination node
	// For true parallel execution, we need a split node concept
	// Workaround: Use conditional routing to start all three branches
	
	// Add a split node for parallel execution
	// In true parallel mode, all three nodes should execute simultaneously
	// For now, we'll use a workaround where the mode node routes to all three
	
	// Add edges from mode determination to parallel branches
	// Note: This requires conditional edges that can route to multiple destinations
	// For now, we'll use a simpler approach where parallel mode routes to a special node
	
	// Set entry point
	builder.SetEntryPoint("determine_mode")
	builder.SetFinishPoint("join_results")

	return builder.Compile()
}

// DetermineWorkflowModeNode returns a node that determines the workflow execution mode.
func DetermineWorkflowModeNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		workflowMode := "sequential" // default

		// Check for explicit mode in request
		if unifiedReq, ok := state["unified_request"].(map[string]any); ok {
			if mode, ok := unifiedReq["workflow_mode"].(string); ok && mode != "" {
				workflowMode = mode
			}
		}

		// Auto-detect mode based on request presence
		hasKG := state["knowledge_graph_request"] != nil
		hasOrch := state["orchestration_request"] != nil
		hasAF := state["agentflow_request"] != nil

		requestCount := 0
		if hasKG {
			requestCount++
		}
		if hasOrch {
			requestCount++
		}
		if hasAF {
			requestCount++
		}

		// If multiple independent requests and no explicit mode, use parallel
		if requestCount > 1 && workflowMode == "sequential" {
			workflowMode = "parallel"
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		newState["workflow_mode"] = workflowMode

		log.Printf("Determined workflow mode: %s (requests: KG=%v, Orch=%v, AF=%v)",
			workflowMode, hasKG, hasOrch, hasAF)

		return newState, nil
	})
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

