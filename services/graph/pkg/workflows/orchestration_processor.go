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
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
	
	"github.com/langchain-ai/langgraph-go/pkg/integration"
)

// OrchestrationProcessorOptions configures the orchestration chain processing workflow.
type OrchestrationProcessorOptions struct {
	LocalAIURL        string // URL to LocalAI service (for LLM chains)
	ExtractServiceURL string // URL to extract service for knowledge graph queries
}

// RunOrchestrationChainNode returns a node that runs an orchestration chain.
//
// This node:
// - Extracts chain configuration from state
// - Enriches chain inputs with knowledge graph context (if available)
// - Creates and executes the orchestration chain using the framework
// - Uses retry logic for resilience
// - Logs operation with correlation ID and duration
//
// State Input:
//   - orchestration_request.chain_name: Chain type to execute
//   - orchestration_request.inputs: Chain input parameters
//   - knowledge_graph: Optional KG context for enrichment
//
// State Output:
//   - orchestration_result: Chain execution results
//   - orchestration_text: Extracted text output
//   - orchestration_success: Execution success flag
//   - orchestration_executed_at: Execution timestamp
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

		// Start logged operation with correlation ID
		op := integration.StartOperation(ctx, log.Default(), fmt.Sprintf("orchestration.chain.%s", chainName))
		defer op.End(nil)

		op.Log("Running orchestration chain: %s", chainName)

		// Create or load orchestration chain based on chain name
		chain, err := createOrchestrationChain(chainName, localAIURL)
		if err != nil {
			op.End(err)
			return nil, fmt.Errorf("create chain %s: %w", chainName, err)
		}

		// If knowledge graph context is available, enrich inputs
		if kgContext, ok := chainInputs["knowledge_graph_context"].(map[string]any); ok {
			// Extract quality metrics
			if quality, ok := kgContext["quality"].(map[string]any); ok {
				chainInputs["quality_score"] = quality["score"]
				chainInputs["quality_level"] = quality["level"]
				if issues, ok := quality["issues"].([]any); ok {
					chainInputs["issues"] = issues
				}
			}
			// Extract graph structure
			if nodes, ok := kgContext["nodes"].([]any); ok {
				chainInputs["node_count"] = len(nodes)
			}
			if edges, ok := kgContext["edges"].([]any); ok {
				chainInputs["edge_count"] = len(edges)
			}
			// Extract query results if available
			if queryResults, ok := kgContext["query_results"].([]any); ok {
				chainInputs["knowledge_graph_query_results"] = queryResults
			}
			// Extract metadata entropy and KL divergence if available
			if metadataEntropy, ok := kgContext["metadata_entropy"].(float64); ok {
				chainInputs["metadata_entropy"] = metadataEntropy
			}
			if klDivergence, ok := kgContext["kl_divergence"].(float64); ok {
				chainInputs["kl_divergence"] = klDivergence
			}
		}
		
		// Also check if knowledge graph is in state directly (from unified workflow)
		if kgIface, ok := state["knowledge_graph"].(map[string]any); ok {
			if quality, ok := kgIface["quality"].(map[string]any); ok {
				if chainInputs["quality_score"] == nil {
					chainInputs["quality_score"] = quality["score"]
				}
				if chainInputs["quality_level"] == nil {
					chainInputs["quality_level"] = quality["level"]
				}
			}
			if nodes, ok := kgIface["nodes"].([]any); ok {
				if chainInputs["node_count"] == nil {
					chainInputs["node_count"] = len(nodes)
				}
			}
			if edges, ok := kgIface["edges"].([]any); ok {
				if chainInputs["edge_count"] == nil {
					chainInputs["edge_count"] = len(edges)
				}
			}
		}

		// Execute chain using framework's Call function with retry logic
		var result map[string]any
		err = integration.RetryWithBackoff(
			ctx,
			integration.DefaultRetryConfig(),
			log.Default(),
			fmt.Sprintf("orchestration.chain.%s.execute", chainName),
			func() error {
				var execErr error
				result, execErr = chains.Call(ctx, chain, chainInputs)
				return execErr
			},
		)
		if err != nil {
			op.End(err)
			return nil, fmt.Errorf("execute orchestration chain %s: %w", chainName, err)
		}

		// Store results in state
		newState := make(map[string]any, len(state)+5)
		for k, v := range state {
			newState[k] = v
		}
		newState["orchestration_result"] = result
		newState["orchestration_chain_name"] = chainName
		newState["orchestration_success"] = true
		newState["orchestration_executed_at"] = time.Now().Format(time.RFC3339)

		// Extract text output if available (common for LLM chains)
		if text, ok := result["text"].(string); ok {
			newState["orchestration_text"] = text
		}
		if output, ok := result["output"].(string); ok {
			newState["orchestration_text"] = output
		}

		op.Log("Orchestration chain executed: chain=%s, output_keys=%v, success=true",
			chainName, chain.GetOutputKeys())
		op.End(nil)

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

		analysis := map[string]any{
			"success":    success,
			"chain_name": chainName,
			"result":     resultMap,
			"analyzed_at": time.Now().Format(time.RFC3339),
		}

		// Extract text output
		if text, ok := resultMap["text"].(string); ok {
			analysis["output_text"] = text
			analysis["output_length"] = len(text)
		}

		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}
		newState["orchestration_success"] = success
		newState["orchestration_analysis"] = analysis

		log.Printf("Orchestration chain result analyzed: success=%v, chain=%s", success, chainName)

		return newState, nil
	})
}

// ChainResultRoutingFunc determines the next node based on orchestration chain results.
func ChainResultRoutingFunc(ctx context.Context, value any) ([]string, error) {
	state, ok := value.(map[string]any)
	if !ok {
		return []string{"error"}, nil
	}

	success, _ := state["orchestration_success"].(bool)
	if !success {
		return []string{"error"}, nil
	}

	analysis, ok := state["orchestration_analysis"].(map[string]any)
	if !ok {
		return []string{"complete"}, nil
	}

	// Route based on chain output
	if outputText, ok := analysis["output_text"].(string); ok {
		if len(outputText) == 0 {
			return []string{"empty"}, nil
		}
		// Check if output indicates error or needs review
		lowerText := strings.ToLower(outputText)
		if strings.Contains(lowerText, "error") || strings.Contains(lowerText, "failed") {
			return []string{"review"}, nil
		}
	}

	return []string{"complete"}, nil
}

// createOrchestrationChain creates an orchestration chain based on the chain name.
//
// Supported chain types:
//   - "llm_chain", "default": Basic LLM chain with customizable prompt
//   - "question_answering", "qa": Context-aware Q&A with context and question inputs
//   - "summarization", "summarize": Text summarization with text input
//   - "knowledge_graph_analyzer", "kg_analyzer": Analyzes knowledge graphs with KG metrics
//   - "data_quality_analyzer", "quality_analyzer": Analyzes data quality metrics
//   - "pipeline_analyzer", "pipeline": Analyzes data pipelines (Control-M, SQL, tables)
//   - "sql_analyzer", "sql": Analyzes SQL queries with optimization suggestions
//   - "agentflow_analyzer", "agentflow": Analyzes AgentFlow flow execution
//
// All chains use the orchestration framework from infrastructure/third_party/orchestration/
// and are configured with LocalAI as the LLM backend.
func createOrchestrationChain(chainName, localAIURL string) (chains.Chain, error) {
	// Create LocalAI LLM instance using the orchestration framework
	llm, err := localai.New(localai.WithBaseURL(localAIURL))
	if err != nil {
		return nil, fmt.Errorf("create LocalAI LLM: %w", err)
	}

	// Create chain based on chain name
	switch chainName {
	case "llm_chain", "default":
		// Simple LLM chain with customizable prompt
		promptTemplate := prompts.NewPromptTemplate(
			"Answer the following question or task:\n\n{{.input}}",
			[]string{"input"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	case "question_answering", "qa":
		// Question answering chain with context support
		promptTemplate := prompts.NewPromptTemplate(
			"Context: {{.context}}\n\nQuestion: {{.question}}\n\nAnswer:",
			[]string{"context", "question"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	case "summarization", "summarize":
		// Summarization chain
		promptTemplate := prompts.NewPromptTemplate(
			"Summarize the following text:\n\n{{.text}}\n\nSummary:",
			[]string{"text"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	case "knowledge_graph_analyzer", "kg_analyzer":
		// Chain for analyzing knowledge graphs
		promptTemplate := prompts.NewPromptTemplate(
			"Analyze the following knowledge graph information:\n\n"+
				"Nodes: {{.node_count}}\n"+
				"Edges: {{.edge_count}}\n"+
				"Quality Score: {{.quality_score}}\n"+
				"Quality Level: {{.quality_level}}\n\n"+
				"Knowledge Graph Context: {{.knowledge_graph_context}}\n\n"+
				"Provide insights and recommendations:\n\n{{.query}}",
			[]string{"node_count", "edge_count", "quality_score", "quality_level", "knowledge_graph_context", "query"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	case "data_quality_analyzer", "quality_analyzer":
		// Chain for analyzing data quality metrics
		promptTemplate := prompts.NewPromptTemplate(
			"Analyze the following data quality metrics:\n\n"+
				"Metadata Entropy: {{.metadata_entropy}}\n"+
				"KL Divergence: {{.kl_divergence}}\n"+
				"Quality Score: {{.quality_score}}\n"+
				"Quality Level: {{.quality_level}}\n"+
				"Issues: {{.issues}}\n\n"+
				"Provide data quality assessment and recommendations:\n\n{{.query}}",
			[]string{"metadata_entropy", "kl_divergence", "quality_score", "quality_level", "issues", "query"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	case "pipeline_analyzer", "pipeline":
		// Chain for analyzing data pipelines
		promptTemplate := prompts.NewPromptTemplate(
			"Analyze the following data pipeline:\n\n"+
				"Control-M Jobs: {{.controlm_jobs}}\n"+
				"SQL Queries: {{.sql_queries}}\n"+
				"Source Tables: {{.source_tables}}\n"+
				"Target Tables: {{.target_tables}}\n"+
				"Data Flow Path: {{.data_flow_path}}\n\n"+
				"Provide pipeline insights and optimization recommendations:\n\n{{.query}}",
			[]string{"controlm_jobs", "sql_queries", "source_tables", "target_tables", "data_flow_path", "query"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	case "sql_analyzer", "sql":
		// Chain for analyzing SQL queries
		promptTemplate := prompts.NewPromptTemplate(
			"Analyze the following SQL query:\n\n{{.sql_query}}\n\n"+
				"Context: {{.context}}\n\n"+
				"Provide SQL analysis, optimization suggestions, and explain execution plan:\n\n{{.query}}",
			[]string{"sql_query", "context", "query"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	case "agentflow_analyzer", "agentflow":
		// Chain for analyzing AgentFlow flows
		promptTemplate := prompts.NewPromptTemplate(
			"Analyze the following AgentFlow flow execution:\n\n"+
				"Flow ID: {{.flow_id}}\n"+
				"Flow Result: {{.flow_result}}\n"+
				"Knowledge Graph Context: {{.knowledge_graph_context}}\n\n"+
				"Provide flow analysis and recommendations:\n\n{{.query}}",
			[]string{"flow_id", "flow_result", "knowledge_graph_context", "query"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil

	default:
		// Default to simple LLM chain with custom input
		promptTemplate := prompts.NewPromptTemplate(
			"{{.input}}",
			[]string{"input"},
		)
		return chains.NewLLMChain(llm, promptTemplate), nil
	}
}

// QueryKnowledgeGraphForChainNode returns a node that queries knowledge graphs to inform chain execution.
//
// This node:
// - Queries the knowledge graph via extract service
// - Stores query results in state for chain enrichment
// - Falls back to state knowledge graph if query fails
//
// State Input:
//   - knowledge_graph_query: Cypher query to execute
//   - knowledge_graph_query_params: Optional query parameters
//   - project_id: Optional project ID filter
//   - system_id: Optional system ID filter
//
// State Output:
//   - knowledge_graph_context: Query results for chain enrichment
//   - knowledge_graph_query_results: Raw query results
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
