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
	orch "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
	orchlocalai "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
	orchprompts "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
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
		chain, err := createOrchestrationChain(chainName, localAIURL)
		if err != nil {
			return nil, fmt.Errorf("create orchestration chain %s: %w", chainName, err)
		}

		// If knowledge graph context is available, enrich inputs
		if kgContext, ok := chainInputs["knowledge_graph_context"].(map[string]any); ok {
			if quality, ok := kgContext["quality"].(map[string]any); ok {
				chainInputs["quality_score"] = quality["score"]
				chainInputs["quality_level"] = quality["level"]
			}
			if nodes, ok := kgContext["nodes"].([]any); ok {
				chainInputs["node_count"] = len(nodes)
			}
			if edges, ok := kgContext["edges"].([]any); ok {
				chainInputs["edge_count"] = len(edges)
			}
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
// Currently supports: "llm_chain", "question_answering", "summarization"
func createOrchestrationChain(chainName, localAIURL string) (orch.Chain, error) {
	// Create LocalAI LLM instance
	llm, err := orchlocalai.New(localAIURL)
	if err != nil {
		return nil, fmt.Errorf("create LocalAI LLM: %w", err)
	}

	// Create chain based on chain name
	switch chainName {
	case "llm_chain", "default":
		// Simple LLM chain with customizable prompt
		promptTemplate := orchprompts.NewPromptTemplate(
			"Answer the following question or task:\n\n{{.input}}",
			[]string{"input"},
		)
		return orch.NewLLMChain(llm, promptTemplate), nil

	case "question_answering", "qa":
		// Question answering chain with context support
		promptTemplate := orchprompts.NewPromptTemplate(
			"Context: {{.context}}\n\nQuestion: {{.question}}\n\nAnswer:",
			[]string{"context", "question"},
		)
		return orch.NewLLMChain(llm, promptTemplate), nil

	case "summarization", "summarize":
		// Summarization chain
		promptTemplate := orchprompts.NewPromptTemplate(
			"Summarize the following text:\n\n{{.text}}\n\nSummary:",
			[]string{"text"},
		)
		return orch.NewLLMChain(llm, promptTemplate), nil

	case "knowledge_graph_analyzer":
		// Chain for analyzing knowledge graphs
		promptTemplate := orchprompts.NewPromptTemplate(
			"Analyze the following knowledge graph information:\n\n"+
				"Nodes: {{.node_count}}\n"+
				"Edges: {{.edge_count}}\n"+
				"Quality Score: {{.quality_score}}\n"+
				"Quality Level: {{.quality_level}}\n\n"+
				"Provide insights and recommendations:\n\n{{.query}}",
			[]string{"node_count", "edge_count", "quality_score", "quality_level", "query"},
		)
		return orch.NewLLMChain(llm, promptTemplate), nil

	default:
		// Default to simple LLM chain with custom input
		promptTemplate := orchprompts.NewPromptTemplate(
			"{{.input}}",
			[]string{"input"},
		)
		return orch.NewLLMChain(llm, promptTemplate), nil
	}
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

		// Call extract service Neo4j query endpoint
		endpoint := strings.TrimRight(extractServiceURL, "/") + "/knowledge-graph/query"
		
		// Build query parameters if project/system IDs are available
		queryParams := make(map[string]any)
		if projectID != "" {
			queryParams["project_id"] = projectID
		}
		if systemID != "" {
			queryParams["system_id"] = systemID
		}
		
		// Use provided params if available
		if params, ok := state["knowledge_graph_query_params"].(map[string]any); ok {
			for k, v := range params {
				queryParams[k] = v
			}
		}

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

		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			// Fall back to using knowledge graph from state if available
			log.Printf("Neo4j query failed, falling back to state: %v", err)
			kgIface, ok := state["knowledge_graph"]
			if ok {
				newState := make(map[string]any, len(state)+1)
				for k, v := range state {
					newState[k] = v
				}
				newState["knowledge_graph_context"] = kgIface
				return newState, nil
			}
			return nil, fmt.Errorf("request knowledge graph query: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			log.Printf("Knowledge graph query failed: %s", string(bodyBytes))
			// Fall back to using knowledge graph from state if available
			kgIface, ok := state["knowledge_graph"]
			if ok {
				newState := make(map[string]any, len(state)+1)
				for k, v := range state {
					newState[k] = v
				}
				newState["knowledge_graph_context"] = kgIface
				return newState, nil
			}
			return state, nil
		}

		var queryResult struct {
			Columns []string         `json:"columns"`
			Data    []map[string]any `json:"data"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&queryResult); err != nil {
			return nil, fmt.Errorf("decode query response: %w", err)
		}

		// Store knowledge graph context for the chain
		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}
		newState["knowledge_graph_context"] = map[string]any{
			"query_results": queryResult.Data,
			"columns":       queryResult.Columns,
		}
		newState["knowledge_graph_query_results"] = queryResult.Data

		log.Printf("Knowledge graph query returned %d results for chain planning", len(queryResult.Data))

		return newState, nil
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			log.Printf("Knowledge graph query failed: %s", string(bodyBytes))
			// Fall back to using knowledge graph from state if available
			kgIface, ok := state["knowledge_graph"]
			if ok {
				newState := make(map[string]any, len(state)+1)
				for k, v := range state {
					newState[k] = v
				}
				newState["knowledge_graph_context"] = kgIface
				return newState, nil
			}
			return state, nil
		}

		var queryResult struct {
			Columns []string         `json:"columns"`
			Data    []map[string]any `json:"data"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&queryResult); err != nil {
			return nil, fmt.Errorf("decode query response: %w", err)
		}

		// Store knowledge graph context for the chain
		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}
		newState["knowledge_graph_context"] = map[string]any{
			"query_results": queryResult.Data,
			"columns":       queryResult.Columns,
		}
		newState["knowledge_graph_query_results"] = queryResult.Data

		log.Printf("Knowledge graph query returned %d results for chain planning", len(queryResult.Data))

		return newState, nil
