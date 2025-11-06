package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph-go/internal/catalog/flightcatalog"
	extractflight "github.com/langchain-ai/langgraph-go/pkg/clients/extractflight"
	extractgrpc "github.com/langchain-ai/langgraph-go/pkg/clients/extractgrpc"
	postgresflight "github.com/langchain-ai/langgraph-go/pkg/clients/postgresflight"
	postgresgrpc "github.com/langchain-ai/langgraph-go/pkg/clients/postgresgrpc"
	"github.com/langchain-ai/langgraph-go/pkg/workflows"
	catalogprompt "github.com/langchain-ai/langgraph-go/pkg/stubs"
	postgresv1 "github.com/plturrell/aModels/services/postgres/pkg/gen/v1"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/graph"
	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
	"google.golang.org/protobuf/encoding/protojson"
	proto "google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func main() {
	searchServiceURL := strings.TrimSpace(os.Getenv("SEARCH_SERVICE_URL"))
	extractHTTPURL := strings.TrimSpace(os.Getenv("EXTRACT_SERVICE_URL"))
	if extractHTTPURL == "" {
		extractHTTPURL = "http://extract-service:8081"
	}

	extractGRPCAddr := strings.TrimSpace(os.Getenv("EXTRACT_GRPC_ADDR"))
	extractFlightAddr := strings.TrimSpace(os.Getenv("EXTRACT_FLIGHT_ADDR"))
	agentSDKFlightAddr := strings.TrimSpace(os.Getenv("AGENTSDK_FLIGHT_ADDR"))
	postgresFlightAddr := strings.TrimSpace(os.Getenv("POSTGRES_FLIGHT_ADDR"))
	postgresGRPCAddr := strings.TrimSpace(os.Getenv("POSTGRES_GRPC_ADDR"))

	var extractGRPCClient *extractgrpc.Client
	if extractGRPCAddr != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		client, err := extractgrpc.Dial(ctx, extractGRPCAddr)
		cancel()
		if err != nil {
			log.Printf("warn: failed to dial extract gRPC (%s): %v", extractGRPCAddr, err)
		} else {
			extractGRPCClient = client
			defer extractGRPCClient.Close()
		}
	}

	var postgresGRPCClient *postgresgrpc.Client
	if postgresGRPCAddr != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		client, err := postgresgrpc.Dial(ctx, postgresGRPCAddr)
		cancel()
		if err != nil {
			log.Printf("warn: failed to dial postgres gRPC (%s): %v", postgresGRPCAddr, err)
		} else {
			postgresGRPCClient = client
			defer postgresGRPCClient.Close()
		}
	}

	graph, err := workflows.NewProactiveIngestionGraph(workflows.GraphOptions{
		SearchServiceURL: searchServiceURL,
		ExtractHTTPURL:   extractHTTPURL,
		ExtractGRPC:      extractGRPCClient,
	})
	if err != nil {
		log.Fatalf("Failed to create graph: %v", err)
	}

	// Initialize Murex integration if configured
	var murexHandler *graph.MurexHandler
	neo4jURI := strings.TrimSpace(os.Getenv("NEO4J_URI"))
	neo4jUsername := strings.TrimSpace(os.Getenv("NEO4J_USERNAME"))
	neo4jPassword := strings.TrimSpace(os.Getenv("NEO4J_PASSWORD"))
	murexBaseURL := strings.TrimSpace(os.Getenv("MUREX_BASE_URL"))
	murexAPIKey := strings.TrimSpace(os.Getenv("MUREX_API_KEY"))
	murexOpenAPISpecURL := strings.TrimSpace(os.Getenv("MUREX_OPENAPI_SPEC_URL"))

	if neo4jURI != "" && murexBaseURL != "" {
		// Create Neo4j driver
		driver, err := neo4j.NewDriverWithContext(neo4jURI, neo4j.BasicAuth(neo4jUsername, neo4jPassword, ""))
		if err != nil {
			log.Printf("warn: failed to create Neo4j driver for Murex integration: %v", err)
		} else {
			defer driver.Close(context.Background())

			// Create Neo4j graph client
			graphClient := graph.NewNeo4jGraphClient(driver, log.Default())

			// Create domain model mapper
			mapper := graph.NewDefaultModelMapper()

			// Configure Murex integration
			murexConfig := map[string]interface{}{
				"base_url": murexBaseURL,
				"api_key":  murexAPIKey,
			}
			if murexOpenAPISpecURL != "" {
				murexConfig["openapi_spec_url"] = murexOpenAPISpecURL
			}

			// Create Murex integration
			murexIntegration := graph.NewMurexIntegration(murexConfig, mapper, graphClient, log.Default())

			// Create terminology extractor
			terminologyExtractor := graph.NewMurexTerminologyExtractor(connectors.NewMurexConnector(murexConfig, log.Default()), log.Default())

			// Create catalog populator (if registry available)
			var catalogPopulator *graph.MurexCatalogPopulator
			// Note: Would need to pass catalog registry here if available
			// For now, catalogPopulator can be nil

			// Create terminology learner integration
			// Use HTTP client to connect to extract service TerminologyLearner
			var terminologyLearner *graph.MurexTerminologyLearnerIntegration
			extractServiceURL := strings.TrimSpace(os.Getenv("EXTRACT_SERVICE_URL"))
			if extractServiceURL == "" {
				extractServiceURL = extractHTTPURL // Use the same URL as extract service
			}
			if extractServiceURL != "" {
				httpLearnerClient := graph.NewHTTPTerminologyLearnerClient(extractServiceURL, log.Default())
				terminologyLearner = graph.NewMurexTerminologyLearnerIntegration(
					terminologyExtractor,
					httpLearnerClient,
					log.Default(),
				)
				log.Printf("Murex terminology learner integration initialized (extract_service=%s)", extractServiceURL)
			}

			// Create handler
			murexHandler = graph.NewMurexHandler(murexIntegration, terminologyExtractor, catalogPopulator, terminologyLearner, log.Default())

			log.Printf("Murex integration initialized (base_url=%s)", murexBaseURL)
		}
	}

	http.HandleFunc("/run", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
			return
		}

		var requestBody map[string]any
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, "Failed to decode request body", http.StatusBadRequest)
			return
		}

		filePath, ok := requestBody["file_path"].(string)
		if !ok {
			http.Error(w, "file_path not found in request body", http.StatusBadRequest)
			return
		}

		initialState := map[string]any{
			"file_path": filePath,
		}
		if agentSDKFlightAddr != "" {
			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()
			if catalog, err := flightcatalog.Fetch(ctx, agentSDKFlightAddr); err != nil {
				log.Printf("warn: failed to refresh agent catalog: %v", err)
			} else {
				injectCatalogMetadata(initialState, catalog)
			}
		}

		result, err := graph.Invoke(context.Background(), initialState)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to run graph: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(result); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	})

	http.HandleFunc("/agent/catalog", newAgentCatalogHandler(agentSDKFlightAddr))

	if extractFlightAddr != "" {
		// Knowledge graph processing endpoint (uses LangGraph workflow)
		http.HandleFunc("/knowledge-graph/process", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			// Create knowledge graph processor workflow
			workflow, err := workflows.NewKnowledgeGraphProcessorWorkflow(workflows.KnowledgeGraphProcessorOptions{
				ExtractServiceURL: extractHTTPURL,
			})
			if err != nil {
				http.Error(w, fmt.Sprintf("create workflow: %v", err), http.StatusInternalServerError)
				return
			}

			// Execute workflow
			result, err := workflow.Invoke(context.Background(), req)
			if err != nil {
				http.Error(w, fmt.Sprintf("workflow execution failed: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)
		})

		// AgentFlow workflow orchestration endpoint (uses LangGraph)
		http.HandleFunc("/agentflow/process", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			// Create AgentFlow processor workflow
			workflow, err := workflows.NewAgentFlowProcessorWorkflow(workflows.AgentFlowProcessorOptions{
				AgentFlowServiceURL: os.Getenv("AGENTFLOW_SERVICE_URL"),
				ExtractServiceURL:   extractHTTPURL,
			})
			if err != nil {
				http.Error(w, fmt.Sprintf("create workflow: %v", err), http.StatusInternalServerError)
				return
			}

			// Execute workflow
			result, err := workflow.Invoke(context.Background(), req)
			if err != nil {
				http.Error(w, fmt.Sprintf("workflow execution failed: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)
		})

		// Orchestration chain workflow orchestration endpoint (uses LangGraph)
		http.HandleFunc("/orchestration/process", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			// Create orchestration processor workflow
			localAIURL := os.Getenv("LOCALAI_URL")
			if localAIURL == "" {
				localAIURL = "http://localai:8080"
			}
			workflow, err := workflows.NewOrchestrationProcessorWorkflow(workflows.OrchestrationProcessorOptions{
				LocalAIURL:        localAIURL,
				ExtractServiceURL: extractHTTPURL,
			})
			if err != nil {
				http.Error(w, fmt.Sprintf("create workflow: %v", err), http.StatusInternalServerError)
				return
			}

			// Execute workflow
			result, err := workflow.Invoke(context.Background(), req)
			if err != nil {
				http.Error(w, fmt.Sprintf("workflow execution failed: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)
		})

		// DeepAgents workflow endpoint
		http.HandleFunc("/deepagents/process", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			// Create deep agents processor workflow
			deepagentsServiceURL := os.Getenv("DEEPAGENTS_SERVICE_URL")
			if deepagentsServiceURL == "" {
				deepagentsServiceURL = "http://deepagents-service:9004"
			}
			workflow, err := workflows.NewDeepAgentsProcessorWorkflow(workflows.DeepAgentsProcessorOptions{
				DeepAgentsServiceURL: deepagentsServiceURL,
			})
			if err != nil {
				http.Error(w, fmt.Sprintf("create workflow: %v", err), http.StatusInternalServerError)
				return
			}

			// Execute workflow
			result, err := workflow.Invoke(context.Background(), req)
			if err != nil {
				http.Error(w, fmt.Sprintf("workflow execution failed: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)
		})

		// Unified workflow endpoint (combines knowledge graphs, orchestration, and AgentFlow)
		http.HandleFunc("/unified/process", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			// Create unified processor workflow
			localAIURL := os.Getenv("LOCALAI_URL")
			if localAIURL == "" {
				localAIURL = "http://localai:8080"
			}
			agentflowServiceURL := os.Getenv("AGENTFLOW_SERVICE_URL")
			if agentflowServiceURL == "" {
				agentflowServiceURL = "http://agentflow-service:9001"
			}
			workflow, err := workflows.NewUnifiedProcessorWorkflow(workflows.UnifiedProcessorOptions{
				ExtractServiceURL:   extractHTTPURL,
				AgentFlowServiceURL: agentflowServiceURL,
				LocalAIURL:          localAIURL,
			})
			if err != nil {
				http.Error(w, fmt.Sprintf("create workflow: %v", err), http.StatusInternalServerError)
				return
			}

			// Execute workflow
			result, err := workflow.Invoke(context.Background(), req)
			if err != nil {
				http.Error(w, fmt.Sprintf("workflow execution failed: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)
		})

		// Legacy extract/graph endpoint (Arrow Flight integration)
		http.HandleFunc("/extract/graph", func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()
			data, err := extractflight.Fetch(ctx, extractFlightAddr)
			if err != nil {
				http.Error(w, fmt.Sprintf("failed to fetch extract flight data: %v", err), http.StatusBadGateway)
				return
			}
			writeJSON(w, data)
		})

		// Pipeline to AgentFlow conversion endpoint
		http.HandleFunc("/pipeline/to-agentflow", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			projectID, _ := req["project_id"].(string)
			systemID, _ := req["system_id"].(string)
			flowName, _ := req["flow_name"].(string)
			flowID, _ := req["flow_id"].(string)
			if flowID == "" {
				flowID = fmt.Sprintf("pipeline_%s_%s", projectID, systemID)
			}
			if flowName == "" {
				flowName = fmt.Sprintf("Control-M Pipeline - %s", systemID)
			}
			force, _ := req["force"].(bool)
			agentFlowServiceURL := os.Getenv("AGENTFLOW_SERVICE_URL")
			if agentFlowServiceURL == "" {
				agentFlowServiceURL = "http://agentflow-service:9001"
			}

			// Create converter and query pipeline
			converter := workflows.NewControlMToAgentFlowConverter(extractHTTPURL)
			ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
			defer cancel()

			segments, err := converter.QueryPipelineFromGraph(ctx, projectID, systemID)
			if err != nil {
				http.Error(w, fmt.Sprintf("query pipeline: %v", err), http.StatusInternalServerError)
				return
			}

			if len(segments) == 0 {
				http.Error(w, "no pipeline segments found", http.StatusNotFound)
				return
			}

			// Generate LangFlow flow
			flowJSON, err := converter.GenerateLangFlowFlow(segments, flowName)
			if err != nil {
				http.Error(w, fmt.Sprintf("generate flow: %v", err), http.StatusInternalServerError)
				return
			}

			// Create flow in AgentFlow/LangFlow
			createResult, err := converter.CreateFlowInAgentFlow(ctx, agentFlowServiceURL, flowJSON, flowID, projectID, force)
			if err != nil {
				log.Printf("Failed to create flow in AgentFlow: %v", err)
				// Return flow JSON even if creation fails
				writeJSON(w, map[string]any{
					"flow_id":    flowID,
					"flow_name":  flowName,
					"segments":   segments,
					"flow_json":  flowJSON,
					"created":    false,
					"error":      err.Error(),
				})
				return
			}

			writeJSON(w, map[string]any{
				"flow_id":    flowID,
				"flow_name":  flowName,
				"segments":   segments,
				"flow_json":  flowJSON,
				"created":    true,
				"result":     createResult,
			})
		})
	}

	if extractGRPCClient != nil {
		http.HandleFunc("/extract/entities", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
				return
			}
			var payload struct {
				Document string `json:"document"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				return
			}
			ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
			defer cancel()
			resp, err := extractGRPCClient.Extract(ctx, payload.Document)
			if err != nil {
				http.Error(w, fmt.Sprintf("extract gRPC failed: %v", err), http.StatusBadGateway)
				return
			}
			marshaller := protojson.MarshalOptions{EmitUnpopulated: true}
			data, err := marshaller.Marshal(resp)
			if err != nil {
				http.Error(w, fmt.Sprintf("marshal response: %v", err), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write(data)
		})
	}

	if postgresFlightAddr != "" {
		http.HandleFunc("/postgres/operations", func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()
			rows, err := postgresflight.FetchOperations(ctx, postgresFlightAddr)
			if err != nil {
				http.Error(w, fmt.Sprintf("fetch postgres operations: %v", err), http.StatusBadGateway)
				return
			}
			writeJSON(w, map[string]any{"operations": rows})
		})
	}

	if postgresGRPCClient != nil {
		http.HandleFunc("/postgres/analytics", func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()

			req := &postgresv1.AnalyticsRequest{}
			if raw := strings.TrimSpace(r.URL.Query().Get("library_type")); raw != "" {
				req.LibraryType = raw
			}
			if raw := strings.TrimSpace(r.URL.Query().Get("start")); raw != "" {
				if parsed, err := time.Parse(time.RFC3339, raw); err == nil {
					req.StartTime = timestamppb.New(parsed)
				}
			}
			if raw := strings.TrimSpace(r.URL.Query().Get("end")); raw != "" {
				if parsed, err := time.Parse(time.RFC3339, raw); err == nil {
					req.EndTime = timestamppb.New(parsed)
				}
			}

			resp, err := postgresGRPCClient.GetAnalytics(ctx, req)
			if err != nil {
				http.Error(w, fmt.Sprintf("postgres analytics: %v", err), http.StatusBadGateway)
				return
			}
			writeProtoJSON(w, resp)
		})

		http.HandleFunc("/postgres/operations/grpc", func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()

			req := &postgresv1.ListLangOperationsRequest{}
			query := r.URL.Query()
			if raw := strings.TrimSpace(query.Get("library_type")); raw != "" {
				req.LibraryType = raw
			}
			if raw := strings.TrimSpace(query.Get("session_id")); raw != "" {
				req.SessionId = raw
			}
			if raw := strings.TrimSpace(query.Get("status")); raw != "" {
				req.Status = parseOperationStatus(raw)
			}
			if raw := strings.TrimSpace(query.Get("page_size")); raw != "" {
				if val, err := strconv.Atoi(raw); err == nil {
					req.PageSize = int32(val)
				}
			}
			if raw := strings.TrimSpace(query.Get("page_token")); raw != "" {
				req.PageToken = raw
			}

			resp, err := postgresGRPCClient.ListOperations(ctx, req)
			if err != nil {
				http.Error(w, fmt.Sprintf("postgres list operations: %v", err), http.StatusBadGateway)
				return
			}
			writeProtoJSON(w, resp)
		})
	}

	// Murex integration endpoints
	if murexHandler != nil {
		http.HandleFunc("/integrations/murex/sync", murexHandler.HandleSync)
		http.HandleFunc("/integrations/murex/trades", murexHandler.HandleIngestTrades)
		http.HandleFunc("/integrations/murex/cashflows", murexHandler.HandleIngestCashflows)
		http.HandleFunc("/integrations/murex/schema", murexHandler.HandleDiscoverSchema)
		http.HandleFunc("/integrations/murex/terminology/extract", murexHandler.HandleExtractTerminology)
		http.HandleFunc("/integrations/murex/terminology/train", murexHandler.HandleTrainTerminology)
		http.HandleFunc("/integrations/murex/terminology/export", murexHandler.HandleExportTrainingData)
		http.HandleFunc("/integrations/murex/catalog/populate", murexHandler.HandlePopulateCatalog)
		log.Println("Murex integration endpoints registered")
	}

	log.Println("Starting graph server on :8081")
	if err := http.ListenAndServe(":8081", nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func writeJSON(w http.ResponseWriter, payload any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
	}
}

func writeProtoJSON(w http.ResponseWriter, msg proto.Message) {
	marshaller := protojson.MarshalOptions{EmitUnpopulated: true}
	data, err := marshaller.Marshal(msg)
	if err != nil {
		http.Error(w, fmt.Sprintf("marshal response: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

func injectCatalogMetadata(target map[string]any, catalog flightcatalog.Catalog) {
	if target == nil {
		return
	}
	enrichment := catalogprompt.Enrich(catalogprompt.Catalog{
		Suites: catalog.Suites,
		Tools:  catalog.Tools,
	})
	target["agent_catalog"] = catalog
	target["agent_tools"] = catalog.Tools
	if enrichment.Prompt != "" {
		target["agent_catalog_context"] = enrichment.Prompt
	}
	if enrichment.Summary != "" {
		target["agent_catalog_summary"] = enrichment.Summary
	}
	if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
		target["agent_catalog_stats"] = enrichment.Stats
	}
	if len(enrichment.Implementations) > 0 {
		target["agent_catalog_matrix"] = enrichment.Implementations
	}
	if len(enrichment.UniqueTools) > 0 {
		target["agent_catalog_unique_tools"] = enrichment.UniqueTools
	}
	if len(enrichment.StandaloneTools) > 0 {
		target["agent_catalog_tool_details"] = enrichment.StandaloneTools
	}
}

func buildCatalogResponse(catalog flightcatalog.Catalog) map[string]any {
	enrichment := catalogprompt.Enrich(catalogprompt.Catalog{
		Suites: catalog.Suites,
		Tools:  catalog.Tools,
	})
	return map[string]any{
		"Suites":                     catalog.Suites,
		"Tools":                      catalog.Tools,
		"agent_catalog_summary":      enrichment.Summary,
		"agent_catalog_stats":        enrichment.Stats,
		"agent_catalog_matrix":       enrichment.Implementations,
		"agent_catalog_unique_tools": enrichment.UniqueTools,
		"agent_catalog_tool_details": enrichment.StandaloneTools,
		"agent_catalog_context":      enrichment.Prompt,
	}
}

func newAgentCatalogHandler(addr string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if strings.TrimSpace(addr) == "" {
			http.Error(w, "agent catalog source not configured", http.StatusServiceUnavailable)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		catalog, err := flightcatalog.Fetch(ctx, addr)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to fetch agent catalog: %v", err), http.StatusBadGateway)
			return
		}
		payload := buildCatalogResponse(catalog)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(payload); err != nil {
			http.Error(w, "failed to encode response", http.StatusInternalServerError)
		}
	}
}

func parseOperationStatus(raw string) postgresv1.OperationStatus {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "running":
		return postgresv1.OperationStatus_OPERATION_STATUS_RUNNING
	case "success":
		return postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS
	case "error":
		return postgresv1.OperationStatus_OPERATION_STATUS_ERROR
	default:
		return postgresv1.OperationStatus_OPERATION_STATUS_UNSPECIFIED
	}
}
