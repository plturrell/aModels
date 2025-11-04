package workflows

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
	orch "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
)

// OrchestrationProcessorOptions configures the orchestration chain processing workflow.
type OrchestrationProcessorOptions struct {
	LocalAIURL        string // URL to LocalAI service (for LLM chains)
	ExtractServiceURL string // URL to extract service for knowledge graph queries
}

// RunOrchestrationChainNode returns a node that runs an orchestration chain.
func RunOrchestrationChainNode(localAIURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract chain configuration from state
		var chainName string
		var chainInputs map[string]any

		// Try to get from state directly
		if req, ok := state["orchestration_request"].(map[string]any); ok {
			if name, ok := req["chain_name"].(string); ok {
				chainName = name
			}
			if inputs, ok := req["inputs"].(map[string]any); ok {
				chainInputs = inputs
			}
		} else {
			// Try to construct from individual fields
			if name, ok := state["chain_name"].(string); ok {
				chainName = name
			}
			if inputs, ok := state["chain_inputs"].(map[string]any); ok {
				chainInputs = inputs
			}
		}

		if chainName == "" {
			log.Println("No orchestration chain name provided; skipping execution")
			return state, nil
		}

		if chainInputs == nil {
			chainInputs = make(map[string]any)
		}

		if localAIURL == "" || localAIURL == "offline" {
			localAIURL = os.Getenv("LOCALAI_URL")
			if localAIURL == "" {
				localAIURL = "http://localai:8080"
			}
		}

		log.Printf("Running orchestration chain: %s", chainName)

		// Create or load orchestration chain based on chain name
		// For now, we'll create a simple LLM chain as an example
		// In production, this would load from a registry or factory
		chain, err := createOrchestrationChain(chainName, localAIURL)
		if err != nil {
			return nil, fmt.Errorf("create orchestration chain %s: %w", chainName, err)
		}

		// Execute chain
		result, err := orch.Call(ctx, chain, chainInputs)
		if err != nil {
			return nil, fmt.Errorf("execute orchestration chain %s: %w", chainName, err)
		}

		// Store results in state
		newState := make(map[string]any, len(state)+3)
		for k, v := range state {
			newState[k] = v
		}
		newState["orchestration_result"] = result
		newState["orchestration_chain_name"] = chainName

		log.Printf("Orchestration chain executed: chain=%s, output_keys=%v",
			chainName, chain.GetOutputKeys())

		return newState, nil
	})
}

// createOrchestrationChain creates an orchestration chain based on the chain name.
// This is a placeholder - in production, this would use a chain registry or factory.
func createOrchestrationChain(chainName, localAIURL string) (orch.Chain, error) {
	// For now, return an error indicating chain creation is not fully implemented
	// In production, this would:
	// 1. Load chain configuration from registry
	// 2. Create LLM instance (LocalAI, Azure, etc.)
	// 3. Create prompt template
	// 4. Combine into chain
	// 5. Return chain

	return nil, fmt.Errorf("chain creation not fully implemented: %s (requires chain registry)", chainName)
}

// QueryKnowledgeGraphForChainNode returns a node that queries knowledge graphs to inform chain execution.
func QueryKnowledgeGraphForChainNode(extractServiceURL string) stategraph.NodeFunc {
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

		log.Printf("Querying knowledge graph for chain planning: query=%s, project=%s, system=%s",
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

		// Extract relevant information for chain planning
		nodesIface, _ := kgMap["nodes"].([]any)
		edgesIface, _ := kgMap["edges"].([]any)
		qualityIface, _ := kgMap["quality"].(map[string]any)

		// Store query results in state for chain execution
		newState := make(map[string]any, len(state)+4)
		for k, v := range state {
			newState[k] = v
		}
		newState["knowledge_graph_query_results"] = map[string]any{
			"query":      query,
			"node_count": len(nodesIface),
			"edge_count": len(edgesIface),
			"quality":    qualityIface,
		}

		// Add knowledge graph context to chain inputs
		if chainInputs, ok := newState["chain_inputs"].(map[string]any); ok {
			chainInputs["knowledge_graph_context"] = map[string]any{
				"nodes":   nodesIface,
				"edges":   edgesIface,
				"quality": qualityIface,
			}
		} else {
			newState["chain_inputs"] = map[string]any{
				"knowledge_graph_context": map[string]any{
					"nodes":   nodesIface,
					"edges":   edgesIface,
					"quality": qualityIface,
				},
			}
		}

		log.Printf("Knowledge graph query completed: nodes=%d, edges=%d",
			len(nodesIface), len(edgesIface))

		return newState, nil
	})
}

// AnalyzeChainResultsNode returns a node that analyzes orchestration chain execution results.
func AnalyzeChainResultsNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		resultIface, ok := state["orchestration_result"]
		if !ok {
			log.Println("No orchestration result available; skipping analysis")
			return state, nil
		}

		resultMap, ok := resultIface.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("orchestration_result is not a map")
		}

		log.Printf("Analyzing orchestration chain result: %v", resultMap)

		// Determine success/failure
		success := true
		if errorMsg, ok := resultMap["error"].(string); ok && errorMsg != "" {
			success = false
			log.Printf("Orchestration chain execution failed: %s", errorMsg)
		}

		// Extract output keys
		chainName, _ := state["orchestration_chain_name"].(string)

		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}
		newState["orchestration_success"] = success
		newState["orchestration_analysis"] = map[string]any{
			"success":    success,
			"chain_name": chainName,
			"result":     resultMap,
		}

		log.Printf("Orchestration chain result analyzed: success=%v, chain=%s", success, chainName)
		return newState, nil
	})
}

// NewOrchestrationProcessorWorkflow creates a workflow that processes orchestration chains with knowledge graph integration.
func NewOrchestrationProcessorWorkflow(opts OrchestrationProcessorOptions) (*stategraph.CompiledStateGraph, error) {
	localAIURL := opts.LocalAIURL
	if localAIURL == "" {
		localAIURL = os.Getenv("LOCALAI_URL")
		if localAIURL == "" {
			localAIURL = "http://localai:8080"
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
		"query_kg":      QueryKnowledgeGraphForChainNode(extractServiceURL),
		"run_chain":     RunOrchestrationChainNode(localAIURL),
		"analyze_result": AnalyzeChainResultsNode(),
	}

	edges := []EdgeSpec{
		{From: "query_kg", To: "run_chain", Label: "chain_execution"},
		{From: "run_chain", To: "analyze_result", Label: "result_analysis"},
	}

	return BuildGraph("query_kg", "analyze_result", nodes, edges)
}

