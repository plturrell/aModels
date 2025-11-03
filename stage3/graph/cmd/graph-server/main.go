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
	catalogprompt "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightcatalog/prompt"
	postgresv1 "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/gen/v1"
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
