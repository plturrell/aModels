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
	adaptersconnectors "github.com/plturrell/aModels/services/graph/adapters/connectors"
	postgresv1 "github.com/plturrell/aModels/services/postgres/pkg/gen/v1"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/graph"
	graphcatalog "github.com/plturrell/aModels/services/graph/pkg/clients/catalog"
	graphneo4j "github.com/plturrell/aModels/services/graph/pkg/clients/neo4j"
	graphterminology "github.com/plturrell/aModels/services/graph/pkg/clients/terminology"
	graphmurex "github.com/plturrell/aModels/services/graph/pkg/integrations/murex"
	graphmodels "github.com/plturrell/aModels/services/graph/pkg/models"
	"github.com/langchain-ai/langgraph-go/pkg/config"
	"google.golang.org/protobuf/encoding/protojson"
	proto "google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func main() {
	// Validate required environment variables
	if err := config.ValidateGraphService(); err != nil {
		log.Fatalf("Configuration validation failed: %v", err)
	}
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

	runtimeGraph, err := workflows.NewProactiveIngestionGraph(workflows.GraphOptions{
		SearchServiceURL: searchServiceURL,
		ExtractHTTPURL:   extractHTTPURL,
		ExtractGRPC:      extractGRPCClient,
	})
	if err != nil {
		log.Fatalf("Failed to create graph: %v", err)
	}

	// Initialize Murex integration if configured
	var murexHandler *graphmurex.MurexHandler
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
			graphClient := graphneo4j.NewNeo4jGraphClient(driver, log.Default())

			// Create domain model mapper
			mapper := graphmodels.NewDefaultModelMapper()

			// Configure Murex integration
			murexConfig := map[string]interface{}{
				"base_url": murexBaseURL,
				"api_key":  murexAPIKey,
			}
			if murexOpenAPISpecURL != "" {
				murexConfig["openapi_spec_url"] = murexOpenAPISpecURL
			}

			// Create Murex integration via adapter
			murexConnector := adaptersconnectors.NewMurexAdapter(murexConfig, log.Default())
			murexIntegration := graphmurex.NewMurexIntegration(murexConnector, mapper, graphClient, log.Default())

			// Create terminology extractor
			terminologyExtractor := graphmurex.NewMurexTerminologyExtractor(murexConnector, log.Default())

			// Initialize catalog service client (HTTP-based integration)
			catalogServiceURL := strings.TrimSpace(os.Getenv("CATALOG_SERVICE_URL"))
			if catalogServiceURL == "" {
				catalogServiceURL = "http://localhost:8084" // Default catalog service URL
			}
			catalogClient := graphcatalog.NewCatalogClient(catalogServiceURL, log.Default())
			if catalogServiceURL != "" {
				log.Printf("Catalog service client initialized (url=%s)", catalogServiceURL)
			}

			// Optional catalog populator (now uses HTTP client instead of direct registry)
			var catalogPopulator *graphmurex.MurexCatalogPopulator
			// Note: registry is nil - using HTTP client only
			// For backward compatibility, we could create a registry, but HTTP is preferred
			catalogPopulator = graphmurex.NewMurexCatalogPopulator(
				terminologyExtractor,
				nil, // No direct registry - using HTTP
				catalogClient,
				log.Default(),
			)

			// Terminology learner integration
			var terminologyLearner *graphmurex.MurexTerminologyLearnerIntegration
			extractServiceURL := strings.TrimSpace(os.Getenv("EXTRACT_SERVICE_URL"))
			if extractServiceURL == "" { extractServiceURL = extractHTTPURL }
			if extractServiceURL != "" {
				httpLearnerClient := graphterminology.NewHTTPTerminologyLearnerClient(extractServiceURL, log.Default())
				terminologyLearner = graphmurex.NewMurexTerminologyLearnerIntegration(
					terminologyExtractor,
					httpLearnerClient,
					log.Default(),
				)
				log.Printf("Murex terminology learner integration initialized (extract_service=%s)", extractServiceURL)
			}

			// Create handler
			murexHandler = graphmurex.NewMurexHandler(murexIntegration, terminologyExtractor, catalogPopulator, terminologyLearner, log.Default())
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

		initialState := map[string]any{ "file_path": filePath }
		if agentSDKFlightAddr != "" {
			ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()
			if catalog, err := flightcatalog.Fetch(ctx, agentSDKFlightAddr); err != nil {
				log.Printf("warn: failed to refresh agent catalog: %v", err)
			} else {
				injectCatalogMetadata(initialState, catalog)
			}
		}

		result, err := runtimeGraph.Invoke(context.Background(), initialState)
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

		// GNN query endpoint (Priority 4: GNN processor for StateGraph workflows)
		http.HandleFunc("/gnn/query", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			trainingServiceURL := os.Getenv("TRAINING_SERVICE_URL")
			if trainingServiceURL == "" {
				trainingServiceURL = "http://training-service:8080"
			}

			// Create GNN processor workflow node
			gnnOpts := workflows.GNNProcessorOptions{
				TrainingServiceURL: trainingServiceURL,
				ExtractServiceURL:  extractHTTPURL,
			}
			gnnNode := workflows.QueryGNNNode(gnnOpts)

			// Execute GNN query
			result, err := gnnNode(context.Background(), req)
			if err != nil {
				http.Error(w, fmt.Sprintf("GNN query failed: %v", err), http.StatusInternalServerError)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)
		})

		// Hybrid query endpoint (Priority 4: KG + GNN)
		http.HandleFunc("/gnn/hybrid-query", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			trainingServiceURL := os.Getenv("TRAINING_SERVICE_URL")
			if trainingServiceURL == "" {
				trainingServiceURL = "http://training-service:8080"
			}

			// Create hybrid query processor workflow node
			gnnOpts := workflows.GNNProcessorOptions{
				TrainingServiceURL: trainingServiceURL,
				ExtractServiceURL:  extractHTTPURL,
			}
			hybridNode := workflows.HybridQueryNode(gnnOpts)

			// Execute hybrid query
			result, err := hybridNode(context.Background(), req)
			if err != nil {
				http.Error(w, fmt.Sprintf("hybrid query failed: %v", err), http.StatusInternalServerError)
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
			trainingServiceURL := os.Getenv("TRAINING_SERVICE_URL")
			if trainingServiceURL == "" {
				trainingServiceURL = "http://training-service:8080"
			}
			workflow, err := workflows.NewUnifiedProcessorWorkflow(workflows.UnifiedProcessorOptions{
				ExtractServiceURL:   extractHTTPURL,
				AgentFlowServiceURL: agentflowServiceURL,
				LocalAIURL:          localAIURL,
				TrainingServiceURL:  trainingServiceURL,
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

	// Phase 1.2: Graph API Endpoints for Visualization and Exploration
	var graphClient *graphneo4j.Neo4jGraphClient
	if neo4jURI != "" {
		driver, err := neo4j.NewDriverWithContext(neo4jURI, neo4j.BasicAuth(neo4jUsername, neo4jPassword, ""))
		if err != nil {
			log.Printf("warn: failed to create Neo4j driver for graph API: %v", err)
		} else {
			defer driver.Close(context.Background())
			graphClient = graphneo4j.NewNeo4jGraphClient(driver, log.Default())
			log.Println("Graph API Neo4j client initialized")
		}
	}

	if graphClient != nil {
		// POST /graph/visualize - Get graph data for visualization
		http.HandleFunc("/graph/visualize", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req struct {
				ProjectID  string   `json:"project_id"`
				SystemID   string   `json:"system_id,omitempty"`
				NodeTypes  []string `json:"node_types,omitempty"`
				EdgeTypes  []string `json:"edge_types,omitempty"`
				Limit      int      `json:"limit,omitempty"`
				Depth      int      `json:"depth,omitempty"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			if req.Limit <= 0 {
				req.Limit = 10000 // Default limit
			}
			if req.Depth <= 0 {
				req.Depth = 2 // Default depth
			}

			ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
			defer cancel()

			// Build Cypher query
			cypher := "MATCH (n:Node)"
			whereClauses := []string{}
			params := map[string]interface{}{}

			if req.ProjectID != "" {
				whereClauses = append(whereClauses, "n.properties_json CONTAINS $project_id")
				params["project_id"] = req.ProjectID
			}
			if req.SystemID != "" {
				whereClauses = append(whereClauses, "n.properties_json CONTAINS $system_id")
				params["system_id"] = req.SystemID
			}
			if len(req.NodeTypes) > 0 {
				whereClauses = append(whereClauses, "n.type IN $node_types")
				params["node_types"] = req.NodeTypes
			}

			if len(whereClauses) > 0 {
				cypher += " WHERE " + strings.Join(whereClauses, " AND ")
			}

			// Get nodes
			nodeQuery := cypher + " RETURN n.id as id, n.type as type, n.label as label, n.properties_json as properties LIMIT $limit"
			params["limit"] = req.Limit

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			var nodes []map[string]interface{}
			var edges []map[string]interface{}

			result, err := session.Run(ctx, nodeQuery, params)
			if err != nil {
				http.Error(w, fmt.Sprintf("query nodes failed: %v", err), http.StatusInternalServerError)
				return
			}

			nodeIds := []string{}
			for result.Next(ctx) {
				record := result.Record()
				nodeId, _ := record.Get("id")
				nodeType, _ := record.Get("type")
				label, _ := record.Get("label")
				props, _ := record.Get("properties")

				nodeData := map[string]interface{}{
					"id":   nodeId,
					"type": nodeType,
					"label": label,
				}
				if props != nil {
					nodeData["properties"] = props
				}
				nodes = append(nodes, nodeData)
				nodeIds = append(nodeIds, fmt.Sprintf("%v", nodeId))
			}

			// Get edges between selected nodes
			if len(nodeIds) > 0 {
				edgeQuery := fmt.Sprintf(
					"MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node) "+
						"WHERE source.id IN $node_ids AND target.id IN $node_ids "+
						"RETURN source.id as source_id, target.id as target_id, r.label as label, r.properties_json as properties "+
						"LIMIT %d",
					req.Limit*2,
				)
				edgeParams := map[string]interface{}{"node_ids": nodeIds}

				if len(req.EdgeTypes) > 0 {
					edgeQuery = strings.Replace(edgeQuery, "LIMIT", "AND type(r) IN $edge_types LIMIT", 1)
					edgeParams["edge_types"] = req.EdgeTypes
				}

				edgeResult, err := session.Run(ctx, edgeQuery, edgeParams)
				if err == nil {
					for edgeResult.Next(ctx) {
						record := edgeResult.Record()
						sourceId, _ := record.Get("source_id")
						targetId, _ := record.Get("target_id")
						label, _ := record.Get("label")
						props, _ := record.Get("properties")

						edgeData := map[string]interface{}{
							"source_id": sourceId,
							"target_id": targetId,
							"label":     label,
						}
						if props != nil {
							edgeData["properties"] = props
						}
						edges = append(edges, edgeData)
					}
				}
			}

			// Calculate metadata
			nodeTypeCounts := make(map[string]int)
			edgeTypeCounts := make(map[string]int)
			for _, node := range nodes {
				if nodeType, ok := node["type"].(string); ok {
					nodeTypeCounts[nodeType]++
				}
			}
			for _, edge := range edges {
				if edgeType, ok := edge["label"].(string); ok {
					edgeTypeCounts[edgeType]++
				}
			}

			response := map[string]interface{}{
				"nodes": nodes,
				"edges": edges,
				"metadata": map[string]interface{}{
					"total_nodes":  len(nodes),
					"total_edges":  len(edges),
					"node_types":   nodeTypeCounts,
					"edge_types":   edgeTypeCounts,
				},
			}

			writeJSON(w, response)
		})

		// POST /graph/explore - Explore graph from a specific node
		http.HandleFunc("/graph/explore", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req struct {
				NodeID    string `json:"node_id"`
				Depth     int    `json:"depth,omitempty"`
				Direction string `json:"direction,omitempty"` // "outgoing", "incoming", "both"
				Limit     int    `json:"limit,omitempty"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			if req.Depth <= 0 {
				req.Depth = 2
			}
			if req.Limit <= 0 {
				req.Limit = 1000
			}
			if req.Direction == "" {
				req.Direction = "both"
			}

			ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
			defer cancel()

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			// Build path query based on direction
			var pathQuery string
			switch req.Direction {
			case "outgoing":
				pathQuery = fmt.Sprintf(
					"MATCH path = (start:Node {id: $node_id})-[*1..%d]->(end:Node) "+
						"RETURN path LIMIT $limit",
					req.Depth,
				)
			case "incoming":
				pathQuery = fmt.Sprintf(
					"MATCH path = (start:Node)<-[*1..%d]-(end:Node {id: $node_id}) "+
						"RETURN path LIMIT $limit",
					req.Depth,
				)
			default: // both
				pathQuery = fmt.Sprintf(
					"MATCH path = (start:Node {id: $node_id})-[*1..%d]-(end:Node) "+
						"RETURN path LIMIT $limit",
					req.Depth,
				)
			}

			params := map[string]interface{}{
				"node_id": req.NodeID,
				"limit":   req.Limit,
			}

			result, err := session.Run(ctx, pathQuery, params)
			if err != nil {
				http.Error(w, fmt.Sprintf("explore query failed: %v", err), http.StatusInternalServerError)
				return
			}

			nodeMap := make(map[string]map[string]interface{})
			edgeMap := make(map[string]map[string]interface{})
			var paths [][]string

			for result.Next(ctx) {
				record := result.Record()
				pathValue, _ := record.Get("path")
				if path, ok := pathValue.(neo4j.Path); ok {
					pathNodes := []string{}
					pathEdges := []string{}

					for _, node := range path.Nodes {
						nodeId := fmt.Sprintf("%v", node.Props["id"])
						nodeMap[nodeId] = map[string]interface{}{
							"id":   node.Props["id"],
							"type": node.Props["type"],
							"label": node.Props["label"],
						}
						pathNodes = append(pathNodes, nodeId)
					}

					for _, rel := range path.Relationships {
						edgeId := fmt.Sprintf("%d", rel.Id)
						edgeMap[edgeId] = map[string]interface{}{
							"source_id": fmt.Sprintf("%v", rel.StartNodeId),
							"target_id": fmt.Sprintf("%v", rel.EndNodeId),
							"label":     rel.Type,
						}
						pathEdges = append(pathEdges, edgeId)
					}

					paths = append(paths, pathNodes)
				}
			}

			nodes := make([]map[string]interface{}, 0, len(nodeMap))
			for _, node := range nodeMap {
				nodes = append(nodes, node)
			}

			edges := make([]map[string]interface{}, 0, len(edgeMap))
			for _, edge := range edgeMap {
				edges = append(edges, edge)
			}

			response := map[string]interface{}{
				"nodes": nodes,
				"edges": edges,
				"paths": paths,
			}

			writeJSON(w, response)
		})

		// GET /graph/stats - Get graph statistics
		http.HandleFunc("/graph/stats", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			projectID := r.URL.Query().Get("project_id")
			systemID := r.URL.Query().Get("system_id")

			ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
			defer cancel()

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			whereClause := ""
			params := map[string]interface{}{}
			if projectID != "" {
				whereClause = "WHERE n.properties_json CONTAINS $project_id"
				params["project_id"] = projectID
				if systemID != "" {
					whereClause += " AND n.properties_json CONTAINS $system_id"
					params["system_id"] = systemID
				}
			}

			// Get total nodes
			nodeCountQuery := "MATCH (n:Node) " + whereClause + " RETURN COUNT(n) as count"
			nodeResult, _ := session.Run(ctx, nodeCountQuery, params)
			var totalNodes int64
			if nodeResult.Next(ctx) {
				if count, ok := nodeResult.Record().Get("count"); ok {
					totalNodes = count.(int64)
				}
			}

			// Get total edges
			edgeCountQuery := "MATCH ()-[r:RELATIONSHIP]->() " + whereClause + " RETURN COUNT(r) as count"
			edgeResult, _ := session.Run(ctx, edgeCountQuery, params)
			var totalEdges int64
			if edgeResult.Next(ctx) {
				if count, ok := edgeResult.Record().Get("count"); ok {
					totalEdges = count.(int64)
				}
			}

			// Get node type distribution
			nodeTypeQuery := "MATCH (n:Node) " + whereClause + " RETURN n.type as type, COUNT(n) as count"
			nodeTypeResult, _ := session.Run(ctx, nodeTypeQuery, params)
			nodeTypes := make(map[string]int64)
			for nodeTypeResult.Next(ctx) {
				record := nodeTypeResult.Record()
				if nodeType, ok := record.Get("type"); ok {
					if count, ok := record.Get("count"); ok {
						nodeTypes[fmt.Sprintf("%v", nodeType)] = count.(int64)
					}
				}
			}

			// Get edge type distribution
			edgeTypeQuery := "MATCH ()-[r:RELATIONSHIP]->() " + whereClause + " RETURN type(r) as type, COUNT(r) as count"
			edgeTypeResult, _ := session.Run(ctx, edgeTypeQuery, params)
			edgeTypes := make(map[string]int64)
			for edgeTypeResult.Next(ctx) {
				record := edgeTypeResult.Record()
				if edgeType, ok := record.Get("type"); ok {
					if count, ok := record.Get("count"); ok {
						edgeTypes[fmt.Sprintf("%v", edgeType)] = count.(int64)
					}
				}
			}

			// Calculate density and average degree
			var density float64
			var avgDegree float64
			if totalNodes > 0 {
				maxPossibleEdges := float64(totalNodes * (totalNodes - 1))
				if maxPossibleEdges > 0 {
					density = float64(totalEdges) / maxPossibleEdges
				}
				avgDegree = float64(totalEdges*2) / float64(totalNodes) // *2 because edges are undirected in calculation
			}

			response := map[string]interface{}{
				"total_nodes":           totalNodes,
				"total_edges":           totalEdges,
				"node_types":            nodeTypes,
				"edge_types":            edgeTypes,
				"density":               density,
				"average_degree":        avgDegree,
				"largest_component_size": totalNodes, // Simplified - would need actual component analysis
				"communities":           1,            // Simplified - would need community detection
			}

			writeJSON(w, response)
		})

		// POST /graph/query - Execute Cypher queries
		http.HandleFunc("/graph/query", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req struct {
				Query  string                 `json:"query"`
				Params map[string]interface{} `json:"params,omitempty"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			if req.Query == "" {
				http.Error(w, "query is required", http.StatusBadRequest)
				return
			}

			ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
			defer cancel()

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			startTime := time.Now()
			result, err := session.Run(ctx, req.Query, req.Params)
			if err != nil {
				http.Error(w, fmt.Sprintf("query execution failed: %v", err), http.StatusInternalServerError)
				return
			}

			var columns []string
			var data []map[string]interface{}

			if result.Next(ctx) {
				record := result.Record()
				columns = record.Keys
				data = append(data, recordAsMap(record))

				for result.Next(ctx) {
					data = append(data, recordAsMap(result.Record()))
				}
			}

			executionTime := time.Since(startTime).Milliseconds()

			response := map[string]interface{}{
				"columns":          columns,
				"data":             data,
				"execution_time_ms": executionTime,
			}

			writeJSON(w, response)
		})

		// POST /graph/paths - Find paths between two nodes
		http.HandleFunc("/graph/paths", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			var req struct {
				SourceID         string   `json:"source_id"`
				TargetID         string   `json:"target_id"`
				MaxDepth         int      `json:"max_depth,omitempty"`
				RelationshipTypes []string `json:"relationship_types,omitempty"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
				return
			}

			if req.MaxDepth <= 0 {
				req.MaxDepth = 5
			}

			ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
			defer cancel()

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			// Build path query
			relFilter := ""
			if len(req.RelationshipTypes) > 0 {
				relTypes := strings.Join(req.RelationshipTypes, "|")
				relFilter = fmt.Sprintf("[:%s]", relTypes)
			}

			pathQuery := fmt.Sprintf(
				"MATCH path = shortestPath((source:Node {id: $source_id})-[%s*1..%d]-(target:Node {id: $target_id})) "+
					"RETURN path",
				relFilter,
				req.MaxDepth,
			)

			params := map[string]interface{}{
				"source_id": req.SourceID,
				"target_id": req.TargetID,
			}

			result, err := session.Run(ctx, pathQuery, params)
			if err != nil {
				http.Error(w, fmt.Sprintf("path query failed: %v", err), http.StatusInternalServerError)
				return
			}

			var paths []map[string]interface{}
			var shortestPath map[string]interface{}

			for result.Next(ctx) {
				record := result.Record()
				pathValue, _ := record.Get("path")
				if path, ok := pathValue.(neo4j.Path); ok {
					pathNodes := []string{}
					pathEdges := []string{}

					for _, node := range path.Nodes {
						if nodeId, ok := node.Props["id"]; ok {
							pathNodes = append(pathNodes, fmt.Sprintf("%v", nodeId))
						}
					}

					for _, rel := range path.Relationships {
						pathEdges = append(pathEdges, fmt.Sprintf("%d", rel.Id))
					}

					pathData := map[string]interface{}{
						"nodes":  pathNodes,
						"edges":  pathEdges,
						"length": len(path.Relationships),
					}

					paths = append(paths, pathData)

					// First path is shortest
					if shortestPath == nil {
						shortestPath = pathData
					}
				}
			}

			response := map[string]interface{}{
				"paths":        paths,
				"shortest_path": shortestPath,
			}

			writeJSON(w, response)
		})

		// Phase 3.2: Graph Analytics Endpoints
		// GET /graph/analytics/communities - Community detection
		http.HandleFunc("/graph/analytics/communities", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			projectID := r.URL.Query().Get("project_id")
			systemID := r.URL.Query().Get("system_id")
			algorithm := r.URL.Query().Get("algorithm") // "louvain", "leiden", "label_propagation"
			if algorithm == "" {
				algorithm = "louvain"
			}

			ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
			defer cancel()

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			whereClause := ""
			params := map[string]interface{}{}
			if projectID != "" {
				whereClause = "WHERE n.properties_json CONTAINS $project_id"
				params["project_id"] = projectID
				if systemID != "" {
					whereClause += " AND n.properties_json CONTAINS $system_id"
					params["system_id"] = systemID
				}
			}

			// Use Neo4j GDS community detection (if available) or simple clustering
			// For now, use a simple approach based on connected components
			query := fmt.Sprintf(`
				CALL gds.graph.project.cypher(
					'community_graph',
					'MATCH (n:Node) %s RETURN id(n) as id',
					'MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node) %s RETURN id(n) as source, id(m) as target',
					{}
				)
				YIELD graphName, nodeCount, relationshipCount
				RETURN graphName, nodeCount, relationshipCount
			`, whereClause, whereClause)

			// Fallback to simple connected components if GDS not available
			simpleQuery := fmt.Sprintf(`
				MATCH (n:Node) %s
				WITH collect(n) as nodes
				CALL {
					WITH nodes
					UNWIND nodes as n
					MATCH path = (n)-[*]-(connected)
					WHERE connected IN nodes
					RETURN collect(DISTINCT id(n)) as component
				}
				RETURN size(component) as community_size, component
				LIMIT 100
			`, whereClause)

			result, err := session.Run(ctx, simpleQuery, params)
			if err != nil {
				// Try even simpler query
				simpleQuery2 := "MATCH (n:Node) RETURN count(n) as total"
				result, err = session.Run(ctx, simpleQuery2, nil)
				if err != nil {
					http.Error(w, fmt.Sprintf("community detection failed: %v", err), http.StatusInternalServerError)
					return
				}
			}

			var communities []map[string]interface{}
			var totalNodes int64
			for result.Next(ctx) {
				record := result.Record()
				if size, ok := record.Get("community_size"); ok {
					communities = append(communities, map[string]interface{}{
						"size": size,
					})
					totalNodes += size.(int64)
				} else if total, ok := record.Get("total"); ok {
					totalNodes = total.(int64)
				}
			}

			response := map[string]interface{}{
				"algorithm":    algorithm,
				"num_communities": len(communities),
				"communities":   communities,
				"total_nodes":  totalNodes,
			}

			writeJSON(w, response)
		})

		// GET /graph/analytics/centrality - Centrality metrics
		http.HandleFunc("/graph/analytics/centrality", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			projectID := r.URL.Query().Get("project_id")
			systemID := r.URL.Query().Get("system_id")
			metricType := r.URL.Query().Get("type") // "degree", "betweenness", "closeness", "pagerank"
			if metricType == "" {
				metricType = "degree"
			}
			topK := r.URL.Query().Get("top_k")
			topKInt := 20
			if topK != "" {
				if k, err := strconv.Atoi(topK); err == nil {
					topKInt = k
				}
			}

			ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
			defer cancel()

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			whereClause := ""
			params := map[string]interface{}{}
			if projectID != "" {
				whereClause = "WHERE n.properties_json CONTAINS $project_id"
				params["project_id"] = projectID
				if systemID != "" {
					whereClause += " AND n.properties_json CONTAINS $system_id"
					params["system_id"] = systemID
				}
			}

			var query string
			switch metricType {
			case "degree":
				query = fmt.Sprintf(`
					MATCH (n:Node) %s
					WITH n, size((n)-[:RELATIONSHIP]-()) as degree
					ORDER BY degree DESC
					LIMIT $top_k
					RETURN n.id as node_id, n.label as label, degree
				`, whereClause)
			case "betweenness":
				// Simplified betweenness - would need GDS for full calculation
				query = fmt.Sprintf(`
					MATCH (n:Node) %s
					WITH n, size((n)-[:RELATIONSHIP]-()) as degree
					ORDER BY degree DESC
					LIMIT $top_k
					RETURN n.id as node_id, n.label as label, degree as centrality
				`, whereClause)
			case "pagerank":
				// Simplified PageRank - would need GDS for full calculation
				query = fmt.Sprintf(`
					MATCH (n:Node) %s
					WITH n, size((n)-[:RELATIONSHIP]-()) as degree
					ORDER BY degree DESC
					LIMIT $top_k
					RETURN n.id as node_id, n.label as label, degree as centrality
				`, whereClause)
			default:
				query = fmt.Sprintf(`
					MATCH (n:Node) %s
					WITH n, size((n)-[:RELATIONSHIP]-()) as degree
					ORDER BY degree DESC
					LIMIT $top_k
					RETURN n.id as node_id, n.label as label, degree as centrality
				`, whereClause)
			}

			params["top_k"] = topKInt
			result, err := session.Run(ctx, query, params)
			if err != nil {
				http.Error(w, fmt.Sprintf("centrality calculation failed: %v", err), http.StatusInternalServerError)
				return
			}

			var nodes []map[string]interface{}
			for result.Next(ctx) {
				record := result.Record()
				nodeData := make(map[string]interface{})
				if nodeId, ok := record.Get("node_id"); ok {
					nodeData["node_id"] = nodeId
				}
				if label, ok := record.Get("label"); ok {
					nodeData["label"] = label
				}
				if centrality, ok := record.Get("centrality"); ok {
					nodeData["centrality"] = centrality
				} else if degree, ok := record.Get("degree"); ok {
					nodeData["centrality"] = degree
				}
				nodes = append(nodes, nodeData)
			}

			response := map[string]interface{}{
				"metric_type": metricType,
				"top_k":       topKInt,
				"nodes":       nodes,
			}

			writeJSON(w, response)
		})

		// GET /graph/analytics/growth - Growth trends
		http.HandleFunc("/graph/analytics/growth", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				return
			}

			projectID := r.URL.Query().Get("project_id")
			systemID := r.URL.Query().Get("system_id")
			days := r.URL.Query().Get("days")
			daysInt := 30
			if days != "" {
				if d, err := strconv.Atoi(days); err == nil {
					daysInt = d
				}
			}

			ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
			defer cancel()

			session := graphClient.Driver().NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			whereClause := ""
			params := map[string]interface{}{}
			if projectID != "" {
				whereClause = "WHERE n.properties_json CONTAINS $project_id"
				params["project_id"] = projectID
				if systemID != "" {
					whereClause += " AND n.properties_json CONTAINS $system_id"
					params["system_id"] = systemID
				}
			}

			// Get node count over time (if updated_at is available)
			query := fmt.Sprintf(`
				MATCH (n:Node) %s
				WHERE n.updated_at IS NOT NULL
				WITH date(n.updated_at) as date, count(n) as count
				ORDER BY date DESC
				LIMIT $days
				RETURN date, count
			`, whereClause)

			params["days"] = daysInt
			result, err := session.Run(ctx, query, params)
			if err != nil {
				// Fallback to total count
				query = fmt.Sprintf("MATCH (n:Node) %s RETURN count(n) as total", whereClause)
				result, err = session.Run(ctx, query, params)
				if err != nil {
					http.Error(w, fmt.Sprintf("growth analysis failed: %v", err), http.StatusInternalServerError)
					return
				}
			}

			var trends []map[string]interface{}
			var total int64
			for result.Next(ctx) {
				record := result.Record()
				if date, ok := record.Get("date"); ok {
					if count, ok := record.Get("count"); ok {
						trends = append(trends, map[string]interface{}{
							"date":  date,
							"count": count,
						})
					}
				} else if totalVal, ok := record.Get("total"); ok {
					total = totalVal.(int64)
				}
			}

			response := map[string]interface{}{
				"days":   daysInt,
				"trends": trends,
				"total":  total,
			}

			writeJSON(w, response)
		})

		log.Println("Graph API endpoints registered (Phase 1.2 & 3.2)")
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

// recordAsMap converts a Neo4j record to a map[string]interface{}
func recordAsMap(record *neo4j.Record) map[string]interface{} {
	result := make(map[string]interface{})
	for _, key := range record.Keys {
		val, _ := record.Get(key)
		result[key] = val
	}
	return result
}
